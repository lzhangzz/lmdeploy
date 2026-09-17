// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/moe.h"

#include <memory>

#include <cuda_runtime.h>

#include "src/turbomind/core/context.h"
#include "src/turbomind/core/registry.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/kernels/copy/copy.h"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/moe_utils_v2.h"
#include "src/turbomind/kernels/norm/rms_norm.h"

#ifdef USE_NCCL
#include "src/turbomind/comm/nccl/deepep/moe_a2a_utils.h"
#include "src/turbomind/comm/nccl/deepep/token_dispatcher.h"
#endif

#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_utils.h"

#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

Moe::Moe(const core::MoeConfig& cfg)
{
    experts_per_token = cfg.experts_per_token;
    norm_topk_prob    = cfg.norm_topk_prob;
    routed_scale      = static_cast<float>(cfg.routed_scale);
    topk_group        = cfg.topk_group;
    topk_method       = cfg.topk_method;
    n_group           = cfg.n_group;
    scoring_func      = cfg.scoring_func;
    router_n_groups   = cfg.router_n_groups;
    data_type_        = cfg.data_type;
    act_type_         = static_cast<ActivationType>(cfg.act_type);
    fuse_silu_act_    = cfg.fuse_silu;
    expert_num        = cfg.expert_num;
    ep_size           = cfg.ep_size;
    ep_rank           = cfg.ep_rank;
}

// Shallow alias of an already-prepared linear: shares its tensors and
// copies its metadata. The alias is never prepared itself (Moe::prepare
// aliases only after its own Module::prepare() pass).
static std::unique_ptr<Linear> AliasLinear(const Linear& src)
{
    auto dst    = std::make_unique<Linear>();
    dst->weight = src.weight;
    dst->bias   = src.bias;
    dst->scales = src.scales;
    dst->zeros  = src.zeros;
    src.copy_metadata_to(*dst);
    return dst;
}

static std::unique_ptr<Ffn> AliasFfn(const Ffn& src)
{
    auto dst           = std::make_unique<Ffn>();
    dst->hidden_dim    = src.hidden_dim;
    dst->inter_size    = src.inter_size;
    dst->act_type      = src.act_type;
    dst->is_fused_silu = src.is_fused_silu;
    dst->tp_size       = src.tp_size;
    dst->tp_rank       = src.tp_rank;
    if (src.w1w3) {
        dst->add_child("w1w3", AliasLinear(*src.w1w3));
    }
    else {
        dst->add_child("w1", AliasLinear(*TM_CHECK_NOTNULL(src.w1.get())));
        dst->add_child("w3", AliasLinear(*TM_CHECK_NOTNULL(src.w3.get())));
    }
    dst->add_child("w2", AliasLinear(*TM_CHECK_NOTNULL(src.w2.get())));
    return dst;
}

void Moe::AliasRouted(const Moe& meta_pack)
{
    TM_CHECK(meta_pack.gate && meta_pack.experts);
    TM_CHECK(!gate && !experts);
    TM_CHECK_NOTNULL(meta_pack.expert(0));

    add_child("gate", AliasLinear(*meta_pack.gate));

    // Experts are EP-sharded; iterate the local slice and keep global
    // expert indices as child names (see Moe::expert()).
    TM_CHECK_EQ(ep_size, meta_pack.ep_size);
    TM_CHECK_EQ(ep_rank, meta_pack.ep_rank);
    const int local_num = meta_pack.num_local_experts();
    const int offset    = meta_pack.local_expert_offset();
    create_child("experts", core::ModuleListConfig{});
    for (int e = 0; e < local_num; ++e) {
        experts->add_child(std::to_string(offset + e), AliasFfn(*TM_CHECK_NOTNULL(meta_pack.expert(e))));
    }
    // never touch shared_gate
}

void LinkLinearExperts(const std::vector<Linear*>& experts, Linear& destination)
{
    const auto& e0 = *experts.front();

    e0.copy_metadata_to(destination);

    const int n            = experts.size();
    destination.k_desc.num = destination.q_desc.num = n;

    if (e0.bias) {
        destination.bias = Tensor{{n, e0.output_dim}, e0.bias.dtype(), kDEVICE};
    }

    std::vector<std::pair<void*, int>> weights;
    std::vector<std::pair<void*, int>> scales;
    std::vector<std::pair<void*, int>> global_scales;
    weights.reserve(n);
    scales.reserve(n);
    global_scales.reserve(n);

    for (int i = 0; i < n; ++i) {
        auto& expert = *experts[i];
        weights.emplace_back(expert.weight.raw_data(), expert.k_desc.ld);
        if (expert.scales) {
            scales.emplace_back(expert.scales.raw_data(), expert.q_desc.ld);
        }
        if (expert.global_scale) {
            global_scales.emplace_back(expert.global_scale.raw_data(), 1);
        }
        if (expert.bias) {
            Copy(expert.bias, destination.bias.slice(i, 1).squeeze(0));
        }
    }

    auto stream           = core::Context::stream();
    auto make_strided_ptr = [stream](const auto& ptrs) {
        return std::shared_ptr<void>{gemm::MakeStridedPtrs(ptrs, stream.handle()),
                                     [stream](void* p) { TM_CUDA_CHECK(cudaFreeAsync(p, stream.handle())); }};
    };
    destination.weight = Tensor{make_strided_ptr(weights), {n}, destination.weight_format.dtype, kDEVICE};
    if (e0.scales) {
        destination.scales = Tensor{make_strided_ptr(scales), {n}, e0.scales.dtype(), kDEVICE};
    }
    if (e0.global_scale) {
        destination.global_scale = Tensor{make_strided_ptr(global_scales), {n}, e0.global_scale.dtype(), kDEVICE};
    }
    destination.k_desc.ld = destination.q_desc.ld = 0;
    destination.k_desc.offsets = destination.q_desc.offsets = nullptr;
}

Ffn* Moe::expert(int i) const
{
    if (!experts) {
        return nullptr;
    }
    return static_cast<Ffn*>(experts->child(std::to_string(local_expert_offset() + i)));
}

void Moe::prepare()
{
    Module::prepare();

    if (meta_pack_) {
        // Meta-MoE layer weight: alias the routed gate/experts from the
        // shared pack. ModelWeight::prepare() prepares meta_experts before
        // the layers, so the pack is already prepared at this point.
        AliasRouted(*meta_pack_);
    }

    if (experts) {
        link_block();
    }
}

void Moe::link_block()
{
    const int local_expert_num = num_local_experts();

    // Create batched block view for fused MoE path
    auto e0 = TM_CHECK_NOTNULL(expert(0));  // exemplar expert

    core::FfnConfig block_cfg;
    block_cfg.hidden_dim = e0->hidden_dim;
    block_cfg.inter_size = e0->inter_size;
    // tp_size=1: expert weights are already TP-sharded — the block
    // just batches them and must not divide inter_size a second time.
    block_cfg.tp_size   = 1;
    block_cfg.tp_rank   = 0;
    block_cfg.data_type = data_type_;
    block_cfg.act_type  = static_cast<int>(act_type_);
    block_cfg.fuse_silu = fuse_silu_act_;
    block_              = std::make_unique<Ffn>(block_cfg);

    // Link each linear in the block to the corresponding expert linears
    auto get_expert_w1w3 = [this](int i) -> Linear* {
        auto* exp = expert(i);
        return exp ? exp->w1w3.get() : nullptr;
    };
    auto get_expert_w1 = [this](int i) -> Linear* {
        auto* exp = expert(i);
        return exp ? exp->w1.get() : nullptr;
    };
    auto get_expert_w3 = [this](int i) -> Linear* {
        auto* exp = expert(i);
        return exp ? exp->w3.get() : nullptr;
    };
    auto get_expert_w2 = [this](int i) -> Linear* {
        auto* exp = expert(i);
        return exp ? exp->w2.get() : nullptr;
    };

    auto link = [local_expert_num](const auto& get_expert, Linear& destination) {
        std::vector<Linear*> linears;
        linears.reserve(local_expert_num);
        for (int i = 0; i < local_expert_num; ++i) {
            linears.push_back(get_expert(i));
        }
        LinkLinearExperts(linears, destination);
    };

    if (get_expert_w1w3(0)) {
        // Fused w1w3 path: experts have a single fused gate+up projection
        block_->add_child("w1w3", std::make_unique<Linear>());
        link(get_expert_w1w3, *block_->w1w3);
    }
    else {
        // Separate w1/w3 path: link individually
        block_->add_child("w1", std::make_unique<Linear>());
        block_->add_child("w3", std::make_unique<Linear>());
        if (get_expert_w1(0)) {
            link(get_expert_w1, *block_->w1);
        }
        if (get_expert_w3(0)) {
            link(get_expert_w3, *block_->w3);
        }
    }

    block_->add_child("w2", std::make_unique<Linear>());
    if (get_expert_w2(0)) {
        link(get_expert_w2, *block_->w2);
    }

    // Propagate the actual fused-silu state from the first expert to
    // the block.  Each expert's prepare() has already run above, so
    // is_fused_silu() now reflects whether the GEMM epilogue applies
    // SiLU.
    block_->is_fused_silu = e0->is_fused_silu;
}

void Moe::bind(Context& ctx)
{
    Module::bind(ctx);
    if (block_) {
        block_->bind(ctx);
    }
    state_ = TM_CHECK_NOTNULL(ctx.moe_state());
}

// ---------------------------------------------------------------------------
// MoeState
// ---------------------------------------------------------------------------

MoeState::MoeState(const EngineParam& engine, const Context& ctx, const Moe& weights):
    mlp_tp_size(engine.mlp_tp_size),
    ep_size(engine.ep_size),
    ep_rank(engine.ep_rank),
    max_token_num(engine.max_forward_token_num * engine.attn_dp_size),
    max_token_per_rank_num(engine.mlp_tp_size
                           * cdiv(engine.max_forward_token_num, engine.attn_tp_size * engine.attn_cp_size)),
    is_warm_up(*ctx.is_warm_up)
{
#ifdef BUILD_MULTI_GPU
    if (ep_size > 1 && engine.moe_a2a_backend == "deepep") {
        TM_CHECK(engine.data_type == kBfloat16)
            << "engine.data_type only support bfloat16 for now, got " << engine.data_type;
        use_a2a = true;
        InitA2A(ctx, weights);
        return;
    }
#endif
    if (ep_size == 1 || engine.moe_a2a_backend == "default") {
        use_a2a = false;
        InitDefault(weights);
        return;
    }

    TM_LOG_FATAL(
        "Unsupported MoE transport config: ep_size = {}, moe_a2a_backend = {}", ep_size, engine.moe_a2a_backend);
}

MoeState::~MoeState() = default;

void MoeState::InitDefault(const Moe& weights)
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

#ifdef USE_NCCL

void MoeState::InitA2A(const Context& ctx, const Moe& weights)
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

void MoeState::CompileKernels(const Moe& weights)
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
// Forward
// ---------------------------------------------------------------------------

Tensor_<float> Moe::Gate(const Tensor& input, Linear& gate)
{
    TM_FUNCTION_SCOPE();

    TM_CHECK_EQ(input.shape(1), gate.input_dim);
    Tensor_<float> logits{{input.shape(0), gate.output_dim}, kDEVICE};
    TM_SCOPE_CALL(gate.Forward(input, logits));
    TM_SCOPE_CALL(ApplyBias(logits, gate.bias, core::Context::stream().handle()));
    return logits;
}

void Moe::Forward(const Tensor&           input,
                  Ref<Tensor>             output,
                  const bool*             token_mask,
                  const std::vector<int>& local_token_nums)
{
    TM_FUNCTION_SCOPE();

    TM_CHECK_NOTNULL(state_);

#ifdef USE_NCCL
    if (state_->use_a2a) {
        ForwardA2A(input, output, token_mask, local_token_nums);
        return;
    }
#endif
    ForwardDefault(input, output, token_mask);
}

void Moe::ForwardDefault(const Tensor& input, Ref<Tensor> output, const bool* token_mask)
{
    TM_FUNCTION_SCOPE();

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

#ifdef USE_NCCL

void Moe::ForwardA2A(const Tensor&           input,
                     Ref<Tensor>             output,
                     const bool*             token_mask,
                     const std::vector<int>& local_token_nums)
{
    TM_FUNCTION_SCOPE();

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

void Moe::ForwardA2A(const Tensor&, Ref<Tensor>, const bool*, const std::vector<int>&)
{
    TM_LOG_FATAL("DeepEP transport requires USE_NCCL");
}

#endif

TM_MODULE_REGISTER_AS("MoeWeight", Moe, core::MoeConfig);

TM_MODULE_METHODS(Moe, MOE_CHILDREN, MOE_PARAMS);

}  // namespace turbomind
