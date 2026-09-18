// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <limits>
#include <vector>

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/kernels/attention/cp_utils.h"
#include "src/turbomind/kernels/gemm/test/test_utils.h"
#include "src/turbomind/models/linear.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/llama_rope.h"
#include "src/turbomind/models/norm_weight.h"

namespace turbomind::core {

using MropeSection = std::array<int, 3>;

struct RopeConfig {
#define ROPE_FIELDS(X)                                                                                                 \
    X(int, type, 0)                                                                                                    \
    X(float, base, 10000.f)                                                                                            \
    X(int, dim, 0)                                                                                                     \
    X(float, factor, 1.f)                                                                                              \
    X(int, max_position_embeddings, 0)                                                                                 \
    X(float, yarn_attention_factor, 1.f)                                                                               \
    X(float, yarn_beta_fast, 32.f)                                                                                     \
    X(float, yarn_beta_slow, 1.f)                                                                                      \
    X(float, llama3_low_freq_factor, 1.f)                                                                              \
    X(float, llama3_high_freq_factor, 4.f)                                                                             \
    X(int, llama3_original_max_position_embeddings, 0)                                                                 \
    X(MropeSection, mrope_section, {})                                                                                 \
    X(bool, mrope_interleaved, false)

    ROPE_FIELDS(TM_MEMBER)
    TM_FOR_EACH(RopeConfig, ROPE_FIELDS)

#undef ROPE_FIELDS
};

struct AttentionConfig: ModuleConfig {
    AttentionConfig(): ModuleConfig{"AttentionWeight"} {}

#define ATTENTION_FIELDS(X)                                                                                            \
    X(int, hidden_dim)                                                                                                 \
    X(int, head_dim)                                                                                                   \
    X(int, head_num)                                                                                                   \
    X(int, kv_head_num)                                                                                                \
    X(int, kv_lora_rank)                                                                                               \
    X(int, q_lora_rank)                                                                                                \
    X(int, qk_rope_dim)                                                                                                \
    X(int, v_head_dim)                                                                                                 \
    X(int, tp_size)                                                                                                    \
    X(int, tp_rank)                                                                                                    \
    X(DataType, data_type)                                                                                             \
    X(int, window_size, 0)                                                                                             \
    X(bool, causal, true)                                                                                              \
    X(bool, output_gate, 0)                                                                                            \
    X(RopeConfig, rope, {})                                                                                            \
    X(int, qk_nope_dim)                                                                                                \
    X(float, softmax_scale, 0.f)                                                                                       \
    X(bool, use_logn_attn, false)

    ATTENTION_FIELDS(TM_MEMBER)
    TM_FOR_EACH(AttentionConfig, ATTENTION_FIELDS)

#undef ATTENTION_FIELDS
};

}  // namespace turbomind::core

namespace turbomind {

void init_rope_kernel_param(const core::RopeConfig& rope, RopeKernelParam& rope_kernel);

class Attention;

/// Batch-level per-phase attention data: ONE slot per phase per rank,
/// held by AttentionState. Layer-agnostic — `block_ptrs` are per-request
/// cache object bases shared across layers; each layer applies its own
/// `prefix_offset_` when dereferencing.
struct AttentionBatch {
    struct Stat {
        int n;
        int q_sum;
        int q_max;
        int k_sum;
        int k_max;
    } decode, prefill;

    Buffer_<void*> block_ptrs;
    Buffer_<int>   block_ptrs_offsets;

    Buffer_<float> rope_base;

    Buffer_<int> mrope_position_ids;
    Buffer_<int> mrope_position_delta;
    Buffer_<int> mrope_position_offsets;
    Buffer_<int> mrope_length;

    // borrowed from env
    Buffer_<bool> finished;
    Buffer_<int>  q_offsets;
    Buffer_<int>  k_offsets;
    Buffer_<int>  readonly_block_num;  // per-request, batch order

    // Global per-token validity mask, owned by LanguageModel and borrowed here; the
    // content is only built at Forward time. Consumed by the attention reduce.
    Buffer_<bool> token_mask;
    int           token_mask_base = 0;  // this DP rank's token offset within the global mask

    uint64_t wave{};  // setup wave that last filled this slot (order-independent guard)
};

/// Shared per-rank attention state — the retired executor's state, held
/// once by the Context (kind state) and bound to every Attention module
/// via bind(). Owns the per-phase batch slots, the fp32 split-KV
/// workspace, pinned staging, the CP post-context, and the aux
/// stream/events used to overlap prefill with decode.
struct AttentionState {
    static constexpr int kMaxKVSplits        = 128;
    static constexpr int kMaxWorkspaceTokens = 4096;

    AttentionState(const EngineParam& engine, const Context& ctx, int phases, const Attention& exemplar);
    ~AttentionState();

    AttentionBatch& batch(int phase)
    {
        return batches_.at(phase);
    }

    /// Any non-kSetup op implies engine progress: batch-level fills from
    /// the previous kSetup wave are stale. The first Attention module of
    /// the next kSetup wave refills and re-stamps its slot. Atomic: the
    /// walk runs on both the scheduler thread (kAdd/kFetch/kUpdate/kDel)
    /// and the executor thread (kSetup/kPrepare/kUnprep).
    void AdvanceWave() noexcept
    {
        ++wave_;
    }
    uint64_t wave() const noexcept
    {
        return wave_.load(std::memory_order_relaxed);
    }

    // Engine geometry
    int  quant_policy{};
    int  cache_block_seq_len{};
    int  attn_cp_size{};
    int  attn_cp_rank{};
    int  attn_dp_size{};
    int  attn_dp_rank{};
    int  max_forward_token_num{};
    int& is_warm_up;
    int  arch{};

    // Pinned staging + mrope defaults
    Buffer_<void*> block_ptrs_buf_;
    Buffer_<int>   block_ptrs_offsets_buf_;
    Buffer_<float> rope_base_buf_;
    Buffer_<int>   mrope_default_buf_;

    // fp32 split-KV workspace (symmetric-allocator backed under CP)
    Tensor_<float> partial_O_;
    Tensor_<float> partial_ML_;
    Tensor_<int>   split_cnt_;

    cudaStream_t aux_stream_{};
    cudaEvent_t  qkv_event_{};
    cudaEvent_t  aux_event_{};

    CpPostContext cp_fn_ctx_;  // context parallel

    RNG rng_;  // warm-up noise fill (warm-up only; executor-thread)

    // Base of the attention KV region within the prefix object: the
    // minimum registered prefix offset, resolved as REGISTERED modules
    // bind (vision-tree Attention modules never register and never fold
    // into it).
    size_t prefix_base{std::numeric_limits<size_t>::max()};

    RopeKernelParam rope_param_{};  // exemplar's kernel rope params (uniform across layers)

    std::vector<AttentionBatch> batches_;

    // Starts at 1: slots stamp from 0, so the first kSetup wave always fills.
    std::atomic<uint64_t> wave_{1};
};

/// Executable attention module: projections, q/k norms, core attention
/// (MLA branch internal, CP via the shared state's post-context), output
/// gate, and the output projection. Per-layer state is exactly weights,
/// the registered KV byte offset, and rope params. Batch-level data and
/// workspaces live in the shared per-rank AttentionState.
class Attention: public core::Module {
public:
    const char* type() const override
    {
        return "AttentionWeight";
    }

    Attention() = default;

    Attention(const core::AttentionConfig& cfg);

    /// Prefix category: this layer's KV bytes. Each layer registers its
    /// own layout; the accumulated prefix layout is byte-identical to the
    /// former single L-sized registration, so cache objects and
    /// checkpoints are unchanged.
    void register_cache(CacheRegistry& registry) override;

    /// Captures the shared per-rank AttentionState. Defined in
    /// attention.cc (Context is incomplete here).
    void bind(Context& ctx) override;

    /// Contract-native BatchOp subset: kAdd (dynamic-NTK init), kSetup
    /// (version-guarded batch fill), kPrepare (batch borrows).
    void Run(BatchOp op, int phase, TensorMap& env) override;

    void Forward(int phase, const Tensor& input, Ref<Tensor> output);

    // --- X-macro field lists ---
#define ATTENTION_CHILDREN(X)                                                                                          \
    X(Linear, w_qkv)                                                                                                   \
    X(Linear, wo)                                                                                                      \
    X(Linear, q_proj)                                                                                                  \
    X(Linear, q_a_proj)                                                                                                \
    X(Linear, q_b_proj)                                                                                                \
    X(Linear, kv_a_proj)                                                                                               \
    X(NormWeight, q_norm)                                                                                              \
    X(NormWeight, k_norm)                                                                                              \
    X(NormWeight, q_a_layernorm)                                                                                       \
    X(NormWeight, kv_a_layernorm)

#define ATTENTION_PARAMS(X) X(sinks)

    TM_MODULE_DECLARE(Attention, ATTENTION_CHILDREN, ATTENTION_PARAMS)

    bool is_mla() const
    {
        return kv_lora_rank > 0;
    }

    const RopeKernelParam& rope_kernel_param() const noexcept
    {
        return rope_param_;
    }

    // --- Config fields (public for runtime access) ---
    int      hidden_dim{};
    int      head_dim{};
    int      head_num{};
    int      kv_head_num{};
    int      kv_lora_rank{};
    int      q_lora_rank{};
    int      qk_rope_dim{};
    int      v_head_dim{};
    int      tp_size{};
    int      tp_rank{};
    DataType data_type{};
    int      window_size{};
    bool     causal{true};
    bool     output_gate{};
    float    softmax_scale{};
    bool     use_logn_attn{};

    core::RopeConfig rope{};

private:
    void Setup(int phase, TensorMap& env);

    Tensor forward_mla(const Tensor& hidden_state);

    /// TODO: dropping the `T` here requires deep refactor of attention dispatch
    template<class T>
    Tensor core_attention(Tensor& qkv, int phase);

    void qk_norm(Tensor& qkv);

    size_t prefix_offset_{};      // stamped by register_cache
    bool   prefix_registered_{};  // false for never-registered (vision-tree) modules

    RopeKernelParam rope_param_{};  // computed in ctor

    AttentionState* state_{};  // shared per-rank kind state, captured in bind
};

}  // namespace turbomind
