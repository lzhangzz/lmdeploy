// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/moe_deepep.h"

#include <cuda_runtime.h>

#include "src/turbomind/core/context.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/kernels/copy/copy.h"
#include "src/turbomind/kernels/gemm/moe_utils_v2.h"

#ifdef USE_NCCL
#include "src/turbomind/comm/nccl/deepep/moe_a2a_utils.h"
#include "src/turbomind/comm/nccl/deepep/token_dispatcher.h"
#endif

#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_utils.h"

#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

// ---------------------------------------------------------------------------
// MoeDeepEPState
// ---------------------------------------------------------------------------

MoeDeepEPState::MoeDeepEPState(const EngineParam& engine, const Context& ctx, const Moe& weights):
    mlp_tp_size(engine.mlp_tp_size),
    ep_size(engine.ep_size),
    ep_rank(engine.ep_rank),
    max_token_num(engine.max_forward_token_num * engine.attn_dp_size),
    max_token_per_rank_num(engine.mlp_tp_size
                           * cdiv(engine.max_forward_token_num, engine.attn_tp_size * engine.attn_cp_size)),
    is_warm_up(*ctx.is_warm_up)
{
    TM_CHECK(engine.data_type == kBfloat16)
        << "engine.data_type only support bfloat16 for now, got " << engine.data_type;
#ifdef USE_NCCL
    InitA2A(ctx, weights);
#else
    TM_LOG_FATAL("DeepEP transport requires USE_NCCL");
#endif
}

MoeDeepEPState::~MoeDeepEPState() = default;

#ifdef USE_NCCL

void MoeDeepEPState::InitA2A(const Context& ctx, const Moe& weights)
{
    const int experts_per_token = weights.experts_per_token;

    topk_weights_ = {max_token_per_rank_num * experts_per_token, kDEVICE};
    topk_indices_ = {max_token_per_rank_num * experts_per_token, kDEVICE};

    f2n_    = {experts_per_token * max_token_num, kDEVICE};
    f2E_    = {experts_per_token * max_token_num, kDEVICE};
    en2f_   = {experts_per_token * max_token_num, kDEVICE};
    scales_ = {experts_per_token * max_token_num, kDEVICE};

    dispatcher_ = std::make_unique<comm::TokenDispatcher>(ctx.comm.h_ep_group);

    scales_copy_stream_      = Stream::create();
    scales_copy_ready_event_ = Event::create();
    scales_copy_done_event_  = Event::create();

    dispatcher_->Init(max_token_per_rank_num,
                      TM_CHECK_NOTNULL(weights.block())->hidden_dim,
                      experts_per_token,
                      weights.num_local_experts(),
                      false /* use_fp8_dispatch */);

    CompileKernels(weights);
}

void MoeDeepEPState::CompileKernels(const Moe& weights)
{
    TM_CUDA_CHECK(cudaMemsetAsync(
        topk_indices_.data(), -1, sizeof(int) * topk_indices_.size(), core::Context::stream().handle()));

    std::vector<int> tokens;
    for (int i = 1; i < max_token_per_rank_num; i *= 2) {
        tokens.push_back(i);
    }
    tokens.push_back(max_token_per_rank_num);

    for (const auto& token_num : tokens) {
        TM_LOG_INFO("Compiling a2a kernels for token_num = {}", token_num);
        Tensor input        = {{token_num, weights.block()->hidden_dim}, kBfloat16, kDEVICE};
        Tensor topk_indices = {topk_indices_, {token_num, weights.experts_per_token}};
        Tensor topk_weights = {topk_weights_, {token_num, weights.experts_per_token}};
        Tensor scales_t, output, out_moe;
        TM_SCOPE_CALL(dispatcher_->Dispatch(
            input, topk_indices, topk_weights, token_num, output, scales_t, f2n_, f2E_, en2f_, offsets_));
        TM_SCOPE_CALL(dispatcher_->Combine(output, out_moe));
    }
}

#endif  // USE_NCCL

// ---------------------------------------------------------------------------
// MoeDeepEPImpl
// ---------------------------------------------------------------------------

void MoeDeepEPImpl::bind(BindEnv& env)
{
    Moe::bind(env);
    if (auto it = env.shared.find("moe.deepep"); it != env.shared.end()) {
        state_ = std::static_pointer_cast<MoeDeepEPState>(it->second);
    }
    else {
        // First DeepEP-implementation module of the walk: create the
        // per-rank state.
        state_ = std::make_shared<MoeDeepEPState>(env.engine, env.context, *this);
        env.shared.emplace("moe.deepep", state_);
    }
}

#ifdef USE_NCCL

void MoeDeepEPImpl::Forward(const Tensor&           input,
                            Ref<Tensor>             output,
                            const bool*             token_mask,
                            const std::vector<int>& local_token_nums)
{
    TM_FUNCTION_SCOPE();

    TM_CHECK_NOTNULL(state_);

    auto& st_ = *state_;

    const MoeA2AInputPartition partition =
        GetMoeA2AInputPartition(local_token_nums, st_.ep_size, st_.ep_rank, st_.mlp_tp_size);
    TM_CHECK_LE(partition.max_tokens_per_rank, st_.max_token_per_rank_num);
    Tensor in = input.slice(partition.begin, partition.size);
    // slice token mask for current rank
    const bool* mask = TM_CHECK_NOTNULL(token_mask) + partition.begin;
    // deepep jit treat the max_tokens_per_rank parameter as template parameter, round it to the nearest power of two to
    // reduce the number of functions that need to be compiled.
    const int max_token_num_per_rank =
        std::min(CeilPowerOfTwo(partition.max_tokens_per_rank), st_.max_token_per_rank_num);

    auto&      block             = *TM_CHECK_NOTNULL(block_);
    const int  token_num         = partition.size;
    const int  hidden_dim        = block.hidden_dim;
    const int  expert_num        = num_experts();
    const int  local_expert_num  = num_local_experts();
    const int  experts_per_token = this->experts_per_token;
    const auto st                = core::Context::stream().handle();

    Tensor_<float> logits = token_num ? Gate(in, *gate.get()) : Tensor_<float>{{0, expert_num}, kDEVICE};

    if (topk_method == "noaux_tc") {
        TM_CHECK_EQ(n_group, 1);
        TM_CHECK_EQ(topk_group, 1);
        TM_CHECK(scoring_func == "sigmoid" || scoring_func == "softmax")
            << "unsupported noaux_tc scoring function: " << scoring_func;

        const float* correction_bias = nullptr;
        if (score_correction_bias && score_correction_bias.size() > 0) {
            TM_CHECK_EQ(score_correction_bias.size(), expert_num);
            correction_bias = score_correction_bias.data<float>();
        }

        TM_SCOPE_CALL(invokeMoeGate_NoAuxTC(st_.topk_weights_.data(),
                                            st_.topk_indices_.data(),
                                            logits.data_or((float*)nullptr),
                                            mask,
                                            correction_bias,
                                            token_num,
                                            expert_num,
                                            experts_per_token,
                                            norm_topk_prob,
                                            routed_scale,
                                            scoring_func == "sigmoid",
                                            st));
    }
    else {
        bool softmax = true;
        if (topk_method == "group_limited_greedy") {
            if (token_num) {
                TM_SCOPE_CALL(invokeMoeSoftmaxMaskTopKGroups(
                    logits.data(), token_num, expert_num, expert_num / n_group, topk_group, st));
            }
            softmax = false;
        }

        TM_SCOPE_CALL(invokeMoeGate_V2(st_.topk_weights_.data(),
                                       st_.topk_indices_.data(),
                                       logits.data_or((float*)nullptr),
                                       mask,
                                       token_num,
                                       expert_num,
                                       experts_per_token,
                                       softmax,
                                       norm_topk_prob,
                                       routed_scale,
                                       st));
    }

    Tensor topk_indices = {st_.topk_indices_, {token_num, experts_per_token}};
    Tensor topk_weights = {st_.topk_weights_, {token_num, experts_per_token}};
    Tensor scales_t;
    Tensor dispatched;  // dispatcher-owned view; not an allocation of ours
    TM_SCOPE_CALL(st_.dispatcher_->Dispatch(in,
                                            topk_indices,
                                            topk_weights,
                                            max_token_num_per_rank,
                                            dispatched,
                                            scales_t,
                                            st_.f2n_,
                                            st_.f2E_,
                                            st_.en2f_,
                                            st_.offsets_));

    // transpose scales to [experts_per_token, token_num] for later combine
    Tensor scales_src = scales_t.t();
    Tensor scales_dst = {st_.scales_, {experts_per_token, dispatched.shape(0)}};
    auto&  stream     = core::Context::stream();
    st_.scales_copy_ready_event_.Record(stream);
    st_.scales_copy_stream_.Wait(st_.scales_copy_ready_event_);
    core::GenericCopy(scales_src, scales_dst, st_.scales_copy_stream_.handle());
    st_.scales_copy_done_event_.Record(st_.scales_copy_stream_);

    Tensor     temp{{dispatched.shape(0) * experts_per_token, hidden_dim}, dispatched.dtype(), dispatched.device()};
    const int* num_valid_tokens = st_.offsets_.data() + local_expert_num;

    auto indices = st_.f2n_.slice(0, temp.shape(0));
    auto offsets = st_.offsets_.slice(0, local_expert_num + 1);

    block.Forward(dispatched, GroupedRoute{indices, offsets, st_.f2E_, num_valid_tokens}, temp);

    // Shared expert: dense FFN on a private buffer. Declarations live
    // outside the guard — the tail call below is unconditional.
    Tensor_<float> shared_logits;
    Tensor         shared_out;
    if (shared && token_num > 0) {
        if (shared_gate) {
            shared_logits = Gate(in, *shared_gate);
        }
        shared_out = {{token_num, hidden_dim}, in.dtype(), in.device()};
        TM_SCOPE_CALL(this->shared->Forward(in, shared_out));
    }

    stream.Wait(st_.scales_copy_done_event_);
    // The A2A path never folds via the reduce — its shared fold happens in
    // invokeMoeA2ASharedCombine below — so the shared args are null/zero.
    TM_SCOPE_CALL(invokeMoeCombine(dispatched,
                                   temp,
                                   block.w2->bias,
                                   st_.scales_.data(),
                                   st_.en2f_.data(),
                                   st_.f2E_.data(),
                                   nullptr,
                                   experts_per_token,
                                   1.f / st_.mlp_tp_size,
                                   Tensor{},
                                   0,
                                   0,
                                   stream.handle()));

    Tensor out_moe;
    TM_SCOPE_CALL(st_.dispatcher_->Combine(dispatched, out_moe));

    // Unconditional: with no shared expert this degrades to in = out_moe,
    // which is the only place the MoE output lands in the residual stream
    // (in aliases the caller's output tensor).
    TM_SCOPE_CALL(
        invokeMoeA2ASharedCombine(in, out_moe, shared_out, shared_logits.data_or((float*)nullptr), stream.handle()));
}

#else

void MoeDeepEPImpl::Forward(const Tensor&, Ref<Tensor>, const bool*, const std::vector<int>&)
{
    TM_LOG_FATAL("DeepEP transport requires USE_NCCL");
}

#endif

}  // namespace turbomind
