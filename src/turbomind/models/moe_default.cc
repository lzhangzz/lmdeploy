// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/moe_default.h"

#include <cuda_runtime.h>

#include "src/turbomind/core/context.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/kernels/gemm/moe_utils_v2.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_utils.h"

#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

// ---------------------------------------------------------------------------
// MoeDefaultState
// ---------------------------------------------------------------------------

MoeDefaultState::MoeDefaultState(const EngineParam& engine, const Context& ctx, const Moe& weights):
    mlp_tp_size(engine.mlp_tp_size),
    ep_size(engine.ep_size),
    ep_rank(engine.ep_rank),
    max_token_num(engine.max_forward_token_num * engine.attn_dp_size),
    is_warm_up(*ctx.is_warm_up)
{
    const int expert_num        = weights.num_experts();
    const int local_expert_num  = weights.num_local_experts();
    const int experts_per_token = weights.experts_per_token;

    h_offsets_ = {local_expert_num + 1, kCPU};

    const int pad_token_num = (max_token_num + kMoeGateVecSize - 1) / kMoeGateVecSize * kMoeGateVecSize;

    masks_   = {expert_num * pad_token_num, kDEVICE};
    f2n_     = {experts_per_token * max_token_num, kDEVICE};
    f2E_     = {experts_per_token * max_token_num, kDEVICE};
    en2f_    = {experts_per_token * max_token_num, kDEVICE};
    scales_  = {experts_per_token * max_token_num, kDEVICE};
    offsets_ = {local_expert_num + 1, kDEVICE};
    accum_   = {expert_num * kMoeGateMaxTiles, kDEVICE};
}

MoeDefaultState::~MoeDefaultState() = default;

// ---------------------------------------------------------------------------
// MoeDefaultImpl
// ---------------------------------------------------------------------------

void MoeDefaultImpl::bind(BindEnv& env)
{
    Moe::bind(env);
    if (auto it = env.shared.find("moe.default"); it != env.shared.end()) {
        state_ = std::static_pointer_cast<MoeDefaultState>(it->second);
    }
    else {
        // First default-implementation module of the walk: create the
        // per-rank state.
        state_ = std::make_shared<MoeDefaultState>(env.engine, env.context, *this);
        env.shared.emplace("moe.default", state_);
    }
}

void MoeDefaultImpl::Forward(const Tensor& input, Ref<Tensor> output, const bool* token_mask, const std::vector<int>&)
{
    TM_FUNCTION_SCOPE();

    TM_CHECK_NOTNULL(state_);

    const int tokens = input.shape(0);
    auto&     st_    = *state_;

    auto& block = *TM_CHECK_NOTNULL(block_);

    const int hidden_dim = block.hidden_dim;

    const size_t padded = (tokens + kMoeGateVecSize - 1) / kMoeGateVecSize * kMoeGateVecSize;

    const int expert_num        = num_experts();
    const int local_expert_num  = num_local_experts();
    const int expert_offset     = local_expert_offset();
    const int experts_per_token = this->experts_per_token;

    TM_CHECK(expert_num);

    auto logits = Gate(input, *gate.get());

    TM_DEBUG_TENSOR(logits, "logits", 2);

    const auto st = core::Context::stream().handle();

    // en2f is the skip sentinel for combine: entries not written by routing (invalid
    // tokens, or shrunk batches with stale values) must read as -1 regardless of EP.
    TM_CUDA_CHECK(cudaMemsetAsync(st_.en2f_.data(), -1, sizeof(int) * tokens * experts_per_token, st));

    if (topk_method == "noaux_tc") {
        TM_CHECK_EQ(n_group, 1);
        TM_CHECK_EQ(topk_group, 1);
        const float* correction_bias = nullptr;
        if (score_correction_bias) {
            correction_bias = score_correction_bias.size() > 0 ? score_correction_bias.data<float>() : nullptr;
        }
        TM_SCOPE_CALL(invokeMoeGate_NoAuxTC(st_.f2n_.data(),
                                            st_.f2E_.data(),
                                            st_.en2f_.data(),
                                            st_.offsets_.data(),
                                            st_.scales_.data(),
                                            st_.masks_.data(),
                                            st_.accum_.data(),
                                            logits.data(),
                                            token_mask,
                                            correction_bias,
                                            tokens,
                                            padded,
                                            expert_num,
                                            experts_per_token,
                                            expert_offset,
                                            local_expert_num,
                                            norm_topk_prob,
                                            routed_scale,
                                            scoring_func == "sigmoid",
                                            st));
    }
    else {
        // V2: accum must be cleared by caller; masks cleared internally
        TM_CUDA_CHECK(cudaMemsetAsync(st_.accum_.data(), 0, sizeof(int) * expert_num * kMoeGateMaxTiles, st));

        bool softmax = true;
        if (topk_method == "group_limited_greedy") {
            TM_SCOPE_CALL(invokeMoeSoftmaxMaskTopKGroups(
                logits.data(), tokens, expert_num, expert_num / n_group, topk_group, st));
            softmax = false;
        }

        TM_SCOPE_CALL(invokeMoeGate_V2(st_.f2n_.data(),
                                       st_.f2E_.data(),
                                       st_.en2f_.data(),
                                       st_.offsets_.data(),
                                       st_.scales_.data(),
                                       st_.masks_.data(),
                                       st_.accum_.data(),
                                       logits.data(),
                                       token_mask,
                                       tokens,
                                       padded,
                                       expert_num,
                                       experts_per_token,
                                       expert_offset,
                                       local_expert_num,
                                       softmax,
                                       norm_topk_prob,
                                       routed_scale,
                                       st));
    }

    if (st_.is_warm_up) {
        std::mt19937     g;
        const auto       expert_ids = SampleUniform(tokens, local_expert_num, experts_per_token, g);
        std::vector<int> cnt(local_expert_num);
        for (const auto& x : expert_ids) {
            ++cnt[x];
        }
        st_.h_offsets_[0] = 0;
        for (int i = 0; i < local_expert_num; ++i) {
            st_.h_offsets_[i + 1] = st_.h_offsets_[i] + cnt[i];
        }
        TM_CUDA_CHECK(cudaMemcpyAsync(
            st_.offsets_.data(), st_.h_offsets_.data(), sizeof(int) * (local_expert_num + 1), cudaMemcpyDefault, st));

        if (st_.ep_size > 1) {
            const auto entries = static_cast<size_t>(tokens) * experts_per_token;
            TM_CUDA_CHECK(cudaMemsetAsync(st_.f2n_.data(), 0, sizeof(int) * entries, st));
            TM_CUDA_CHECK(cudaMemsetAsync(st_.f2E_.data(), 0, sizeof(int) * entries, st));
            TM_CUDA_CHECK(cudaMemsetAsync(st_.en2f_.data(), -1, sizeof(int) * entries, st));
        }
    }

    Tensor temp{{tokens * experts_per_token, hidden_dim}, input.dtype(), input.device()};

    // Masked-out tokens compact the routing tables even for ep_size == 1, so the
    // routed count is device-side offsets_[local_expert_num] in all cases.
    const int* num_valid_tokens = st_.offsets_.data() + local_expert_num;

    auto indices = st_.f2n_.slice(0, temp.shape(0));
    auto offsets = st_.offsets_.slice(0, local_expert_num + 1);

    block.Forward(input, GroupedRoute{indices, offsets, st_.f2E_, num_valid_tokens}, temp);

    // Shared expert: dense FFN on a private buffer, folded into the combine.
    Tensor         shared_out;
    Tensor_<float> shared_logits;
    int            shared_begin = 0, shared_end = 0;
    if (shared) {
        if (shared_gate) {
            shared_logits = Gate(input, *shared_gate);
        }
        shared_out = {{tokens, hidden_dim}, input.dtype(), input.device()};
        // Sequence parallel under EP (geometry identical to the retired
        // GetShardFfnInput); degenerates to the full batch when st_.ep_size == 1.
        const int per = tokens / st_.ep_size, rem = tokens % st_.ep_size;
        const int num = per + (st_.ep_rank < rem);
        shared_begin  = st_.ep_rank * per + std::min(st_.ep_rank, rem);
        shared_end    = shared_begin + num;
        if (num > 0) {
            TM_SCOPE_CALL(this->shared->Forward(input.slice(shared_begin, num), shared_out.slice(shared_begin, num)));
        }
        // No Clear: the fold reads only [shared_begin, shared_end), which the
        // FFN just wrote. num == 0 ranks pass begin == end and own nothing.
    }

    TM_SCOPE_CALL(invokeMoeCombine(output,
                                   temp,
                                   block.w2->bias,
                                   st_.scales_.data(),
                                   st_.en2f_.data(),
                                   st_.f2E_.data(),
                                   shared_logits.data_or((float*)nullptr),
                                   experts_per_token,
                                   1.f / st_.mlp_tp_size,
                                   shared_out,
                                   shared_begin,
                                   shared_end,
                                   core::Context::stream().handle()));
}

}  // namespace turbomind
