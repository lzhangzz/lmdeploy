// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <optional>
#include <utility>
#include <vector>

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/kernels/linear_attn/delta_rule.h"
#include "src/turbomind/models/linear.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/norm_weight.h"

namespace turbomind::core {

struct DeltaNetConfig: ModuleConfig {
    DeltaNetConfig(): ModuleConfig{"DeltaNetWeight"} {}

#define DELTANET_FIELDS(X)                                                                                             \
    X(int, hidden_dim)                                                                                                 \
    X(int, num_k_heads)                                                                                                \
    X(int, num_v_heads)                                                                                                \
    X(int, key_head_dim)                                                                                               \
    X(int, value_head_dim)                                                                                             \
    X(int, d_conv, 4)                                                                                                  \
    X(DataType, data_type)                                                                                             \
    X(int, tp_size)                                                                                                    \
    X(int, tp_rank)

    DELTANET_FIELDS(TM_MEMBER)
    TM_FOR_EACH(DeltaNetConfig, DELTANET_FIELDS)

#undef DELTANET_FIELDS
};

}  // namespace turbomind::core

namespace turbomind {

class DeltaNet;

/// Batch-level per-phase GDN data: ONE slot per phase per rank, held by
/// DeltaNetState. Layer-agnostic — pointer tables and plans cover all
/// layers; each layer applies its own block/offset mapping in Forward.
struct DeltaNetBatch {
    std::vector<int>                             input_lens;
    int                                          batch_size{};
    std::vector<std::pair<uint8_t*, size_t>>     reset_ptrs;
    Buffer_<int>                                 q_offsets;
    Buffer_<int>                                 k_offsets;
    Buffer_<bool>                                finished;
    Buffer_<void*>                               conv_state_ptrs;
    Buffer_<void*>                               recurrent_state_ptrs;
    int                                          decode_count{};
    int                                          prefill_count{};
    std::optional<linear_attn::delta_rule::Plan> recurrent_plan;
    std::optional<linear_attn::delta_rule::Plan> chunked_plan;
    core::Tensor                                 chunked_workspace;
    Buffer_<uint8_t>                             recurrent_state_tma_descs;

    uint64_t wave{};  // setup wave that last filled this slot (order-independent guard)
};

/// Shared per-rank GDN state — the retired executor's state, held once by
/// the Context (kind state) and bound to every DeltaNet module via bind().
/// Registers the per-rank recurrent block grid (one composite part per
/// (layer-group, head-group) block) with the checkpoint category at
/// creation — during bind, before category finalization; conv bytes are
/// registered per module in the register_cache pass and accumulate into
/// the category's part 0.
struct DeltaNetState {
    DeltaNetState(const EngineParam& engine, CacheRegistry& registry, int phases, std::vector<DeltaNet*> modules);
    ~DeltaNetState();

    DeltaNetBatch& batch(int phase)
    {
        return batches_.at(phase);
    }

    /// Same wave discipline as AttentionState: any non-kSetup op implies
    /// engine progress, so the previous kSetup fill is stale. Atomic: the
    /// walk runs on the scheduler and executor threads.
    void AdvanceWave() noexcept
    {
        ++wave_;
    }
    uint64_t wave() const noexcept
    {
        return wave_.load(std::memory_order_relaxed);
    }

    /// Assign the next sequential layer index to a binding module (bind
    /// order == layer order) and check geometry uniformity. Returns the
    /// module's GDN-local layer index.
    int RegisterLayer(const DeltaNet& module);

    /// Per-layer in-block element offset of this layer's recurrent state.
    int64_t layer_state_offset(int layer_index) const;

    // --- geometry + planning (from the engine param and module shapes) ---
    const int                                           tp_size;
    const DataType                                      recurrent_state_dtype;
    const linear_attn::delta_rule::ContextParallelLevel gdr_cp_level;

    int    layer_num_{};         // == modules.size()
    int    rec_base_{};          // composite part id of layer 0's recurrent state (== 1)
    int    layers_per_block_{};  // L_b
    int    heads_per_block_{};   // H_b
    int    num_head_groups_{};   // ceil(num_v_heads / H_b)
    int    num_layer_groups_{};  // ceil(layer_num_ / L_b)
    int    num_blocks_{};        // num_layer_groups_ * num_head_groups_
    size_t block_bytes_{};       // one recurrent block's bytes (one composite part)
    size_t conv_total_bytes_{};  // accumulated conv-state bytes (part 0)

    DataType                                input_dtype_{kNull};
    int                                     arch_{};
    int                                     num_k_heads_{};
    int                                     num_v_heads_{};
    int                                     head_dim_{};
    int                                     gate_stride_{};
    linear_attn::delta_rule::GatedDeltaRule delta_rule_;

    int          sm_count_{};
    Buffer_<int> work_counter_;

    // Pinned staging for the per-setup pointer uploads (max-batch sized)
    Buffer_<void*> conv_state_ptrs_buf_;
    Buffer_<void*> recurrent_state_ptrs_buf_;

    cudaStream_t aux_stream_{};
    cudaEvent_t  ev_before_{};
    cudaEvent_t  ev_after_{};

    std::vector<DeltaNetBatch> batches_;

    // Starts at 1: slots stamp from 0, so the first kSetup wave always fills.
    std::atomic<uint64_t> wave_{1};

private:
    // Layer order, as collected at creation; RegisterLayer checks the bind
    // walk against it and stays idempotent across update_params re-binds.
    std::vector<DeltaNet*> modules_;
    int                    next_layer_index_{};
};

/// Executable Gated DeltaNet (linear attention) module: fused input
/// projection, beta/gate computation, fused conv1d, L2-normalized QK, the
/// delta-rule recurrent/chunked kernels through the shared state's plans,
/// gated RMS norm, and the output projection. Per-layer state is weights,
/// the registered conv offset, and the layer's block-grid mapping; batch
/// data and plans live in the shared per-rank DeltaNetState.
class DeltaNet: public core::Module {
public:
    const char* type() const override
    {
        return "DeltaNetWeight";
    }

    DeltaNet() = default;

    DeltaNet(const core::DeltaNetConfig& cfg);

    void prepare() override;

    /// Checkpoint category: this layer's conv-state bytes (accumulation
    /// channel, element offset kept). The recurrent block grid is
    /// registered once per rank by DeltaNetState.
    void register_cache(CacheRegistry& registry) override;

    /// Captures the shared per-rank DeltaNetState and takes this layer's
    /// index in the block grid. Defined in delta_net.cc (Context is
    /// incomplete here).
    void bind(Context& ctx) override;

    /// Contract-native BatchOp subset: kSetup (version-guarded batch fill
    /// — plans, pointer tables, reset list), kPrepare (batch borrows,
    /// position-0 state clears, TMA descriptor preparation — batch-level,
    /// done once per pass by layer 0).
    void Run(BatchOp op, int phase, TensorMap& env) override;

    void Forward(int phase, const Tensor& input, Ref<Tensor> output);

    // --- X-macro field lists ---
#define DELTA_NET_CHILDREN(X)                                                                                          \
    X(Linear, in_proj_all)                                                                                             \
    X(Linear, out_proj)                                                                                                \
    X(NormWeight, norm)

#define DELTA_NET_PARAMS(X)                                                                                            \
    X(conv1d)                                                                                                          \
    X(A_log)                                                                                                           \
    X(dt_bias)

    TM_MODULE_DECLARE(DeltaNet, DELTA_NET_CHILDREN, DELTA_NET_PARAMS)

    // --- Config fields (public for runtime access) ---
    int      hidden_dim{};
    int      num_k_heads{};
    int      num_v_heads{};
    int      key_head_dim{};
    int      value_head_dim{};
    int      d_conv{};
    DataType data_type{};
    int      tp_size{};
    int      tp_rank{};

private:
    void Setup(int phase, TensorMap& env);

    // Conv offset in elements within checkpoint part 0 (stamped by
    // register_cache from the accumulated byte offset).
    int conv_state_offset_{};

    int     layer_index_{};          // GDN-local layer index, assigned at bind
    int64_t linear_state_offset_{};  // in-block element offset in the shared block

    DeltaNetState* state_{};  // shared per-rank kind state, captured in bind
};

}  // namespace turbomind
