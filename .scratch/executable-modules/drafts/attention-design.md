# Merged Attention (and DeltaNet) module design

Draft spec section from [Merged Attention module design](../issues/05-merged-attention-module-design.md). Builds on the validated [Linear interface](../prototype/linear-module-interface.md), `drafts/context-contract.md`, and the Ffn/Moe pattern from `drafts/ffn-moe-design.md`.

## Decisions

1. **Per-layer Attention modules.** Per-layer state is exactly: weights, the registered KV byte offset, and rope params. The per-phase `AttentionData` is *batch* state, not layer state: its `block_ptrs` are per-request cache-object bases, identical across layers; layers differ only in the offset applied when dereferencing.
2. **Registration is a dedicated pass.** `register_cache(CacheRegistry&)` runs after the Python build and before category finalization; each module registers its own per-layer bytes and keeps its offset. ExecContext is *not* extended — it stays pure execution services per the context contract.
3. **Per-phase batch data lives in the attention kind pool**: one `AttentionData` per phase per rank, filled during kSetup, read by every layer's Forward. The fill flow is mapped in BatchOp and decoder mapping.
4. **Workspaces pooled**: `partial_O_`/`partial_ML_`/`split_cnt_` and reusable pinned staging come from the attention kind pool (contract default — attention memory profile unchanged). Aux streams/events are factory-made per module.
5. **GDN is its own module**: `DeltaNet` follows the same pattern; conv/recurrent registration in `register_cache` with the checkpoint category, `contracts.gated-deltanet` semantics preserved (per-layer conv offsets, recurrent composite parts, TP×CP folding).
6. **MLA is an internal branch** of `Attention::Forward` (`is_mla()`); CP post-context comes from `xc_->comm()` at bind; warm-up noise-fill reads `env.warm_up()`.
7. **Naming**: classes `Attention` and `DeltaNet`; `type()` frozen to `"AttentionWeight"`/`"DeltaNetWeight"`.

## Attention interface

```cpp
// PROTOTYPE — merged Attention: per-layer executable module.
// Replaces models/attention_weight.h (execution side) + models/llama/unified_attention_layer.{h,cc}.
class Attention: public core::Module {
public:
    const char* type() const override
    {
        return "AttentionWeight";
    }

    Attention() = default;
    Attention(const core::AttentionConfig& cfg);

    void prepare() override;                       // weight-side only, as today
    void register_cache(CacheRegistry& registry);  // prefix category: this layer's KV bytes -> prefix_offset_
    void bind(ExecContext& xc);                    // execution services only (context-contract scope)

    void Run(BatchOp op, int phase, TensorMap& env);  // op/thread mapping: BatchOp and decoder mapping
    void Forward(const AttentionBatch& batch, Ref<Tensor> output);

    // children unchanged: w_qkv, wo, q_proj, q_a_proj, q_b_proj, kv_a_proj (Linear),
    //                      q_norm, k_norm, q_a_layernorm, kv_a_layernorm (NormWeight)
    // params unchanged: sinks; geometry + RopeConfig as today; is_mla() as today

private:
    Tensor    forward_mla(const Tensor& x);                              // internal branch
    template<class T>
    Tensor    core_attention(Tensor& qkv, const AttentionBatch& batch);
    void      qk_norm(Tensor& qkv);

    size_t           prefix_offset_{};   // stamped by register_cache (was cache_block_offset)
    RopeKernelParam  rope_param_{};      // computed in ctor; absorbs init_rope_kernel_param
    ExecContext*     xc_{};
};
```

```cpp
/// Batch-level per-phase attention data: ONE per phase per rank, attention kind pool.
/// Layer-agnostic; each layer applies its own prefix_offset_ when dereferencing.
struct AttentionBatch {
    // device, filled at kSetup
    void**       block_ptrs;          // per-request cache object bases
    int*         block_ptrs_offsets;
    float*       rope_base;           // + mrope section defaults
    // borrowed batch fields (decoder-owned)
    const int*   q_offsets;
    const int*   k_offsets;
    const bool*  finished;
    const bool*  token_mask;
    int          readonly_block_num;
};
```

Layer addressing is identical to today: pointer tables shared, `prefix_offset_` per layer — the merged shape normalizes today's single L-sized registration into L sequential per-layer registrations with the same accumulated layout, so cache-object layout and checkpoints are unchanged.

## DeltaNet interface (same pattern)

```cpp
// PROTOTYPE — merged GatedDeltaNet: per-layer executable module.
// Replaces models/delta_net_weight.h (execution side) + models/llama/GatedDeltaNetLayer.{h,cc}.
class DeltaNet: public core::Module {
public:
    const char* type() const override
    {
        return "DeltaNetWeight";
    }

    DeltaNet() = default;
    DeltaNet(const core::DeltaNetConfig& cfg);

    void prepare() override;
    void register_cache(CacheRegistry& registry);
    // checkpoint category: conv accumulation offset + recurrent composite base parts,
    // contracts.gated-deltanet semantics preserved (block grid, TP*CP folding, rank-local shards)
    void bind(ExecContext& xc);

    void Run(BatchOp op, int phase, TensorMap& env);
    void Forward(const DeltaNetBatch& batch, Ref<Tensor> output);

    // children: in_proj_all, out_proj (Linear), norm (NormWeight)
    // params: conv1d, A_log, dt_bias

private:
    int conv_state_offset_{};      // stamped by register_cache
    int linear_state_offset_{};
    linear_attn::delta_rule::GatedDeltaRule delta_rule_;  // planner/executor as today
    ExecContext*                                xc_{};
};
```

Aux streams (attention qkv/aux overlap, GDN prio−1 stream) are factory-made per module; chunked/recurrent workspaces come from the kind pools; recurrent-state TMA descriptors remain per-phase module state as today.

## Engine pass order (new, normative for the spec)

```text
Python builds the module tree by name
  -> prepare() walk                     (weight processing, as today)
  -> register_cache(CacheRegistry&) walk (every module registers its byte ranges)
  -> registry category finalization     (categories -> object ids)
  -> bind(ExecContext&) walk            (execution services injection)
  -> first Run
```

Ordering rules: `register_cache` must complete before category finalization; `bind` must complete before the first `Run`; the two injection passes are otherwise independent, with the canonical order above. This pass structure is shared by all module kinds and lands in BatchOp and decoder mapping for the op-level wiring.

## Amendment (Rework the ExecContext design)

`bind(ExecContext&)` becomes `bind(Context&)` on the surviving `turbomind::Context`: each Attention module captures a reference to `AttentionState` (the per-phase batch slots, fp32 workspace, pinned staging, CP context, aux stream — held once per rank). The "attention kind pool" language above reads as `AttentionState`. Everything else (per-layer modules, `register_cache` pass, batch-level data, offsets, naming) stands.
