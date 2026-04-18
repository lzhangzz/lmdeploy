# Decouple LlamaFfnLayer and MoeFfnLayer from ModelParam and MoeParam

## Goal

Remove both layers' dependency on `ModelParam` and `MoeParam`. LlamaFfnLayer loses its only (dead) ModelParam field. MoeFfnLayer reads dimensional fields and routing config from `MoeWeight` at runtime, with lazy buffer initialization on first Forward call. No behavioral changes.

## Current State

### LlamaFfnLayer

Constructor takes `const ModelParam& model`. Stores `hidden_units_` from `model.hidden_units` — **completely dead** (never read in `forward()`). `forward()` reads everything from `FfnWeight*` accessors and tensor shapes.

### MoeFfnLayer

Constructor takes `const ModelParam& model, const MoeParam& param, const EngineParam& engine, const Context& ctx`.

From `ModelParam`: only `model.hidden_units` → `hidden_dim_` (used for temp tensor sizing in Forward).

From `MoeParam`: full copy stored as `param_`. Used extensively in Forward for gate logic, buffer sizing in constructor, and in Combine.

From `EngineParam`: `mlp_tp_size` → `tp_size_`, `max_forward_token_num * attn_dp_size` for buffer sizing.

`MoeWeight` already stores the same data: `hidden_dim_` (private), `moe_param_` (accessible via `moe_param()` accessor), but no `inter_size_` field.

## Changes

### 1. LlamaFfnLayer: remove ModelParam

**Constructor:** `LlamaFfnLayer(const Context& ctx)`

Remove dead `hidden_units_` member. The only remaining member is `LlamaLinear& linear_` from `ctx.linear`.

No changes to `FfnWeight` needed — `forward()` already reads `inter_size()`, `act_type()`, `is_fused_silu()` from it.

### 2. MoeWeight: make fields public, add inter_size_

Move `hidden_dim_` from private to public section. Add `int inter_size_{};` to public section.

In `MoeWeight::MoeWeight(const core::MoeConfig& cfg)`, populate `inter_size_ = cfg.inter_size / cfg.tp_size;` (TP-split value, matching what MoeFfnLayer previously computed as `param.inter_size / engine.mlp_tp_size`). Note: `cfg.inter_size` is the full (pre-TP-split) value; `cfg.tp_size` is the TP degree.

Existing `moe_param()` accessor stays — used by MoeFfnLayer to read routing config at runtime.

Fields to make public: `hidden_dim_`, new `inter_size_`.

Other private fields (`layer_id_`, `moe_param_`, `mlp_bias_`, `data_type_`, `tp_size_`, `tp_rank_`, `act_type_`, `fuse_silu_act_`, `expert_num_`, `block_`) stay private — they're not needed by MoeFfnLayer at runtime.

### 3. MoeFfnLayer: shrink constructor, lazy buffer init

**New constructor:** `MoeFfnLayer(const EngineParam& engine, const Context& ctx)`

Constructor body:
- Store `tp_size_` from `engine.mlp_tp_size`
- Store `max_token_num_` from `engine.max_forward_token_num * engine.attn_dp_size`
- Store `is_warm_up_`, `linear_`, create `expert_ffn_` unconditionally
- Do NOT allocate any expert-dependent buffers (h_offsets_, masks_, f2n_, f2E_, en2f_, scales_, offsets_, accum_)
- Add `bool initialized_ = false;` flag

**Lazy Init method** (called on first Forward):
```cpp
void MoeFfnLayer::Init(ForwardParam& p) {
    const auto& moe = p.weights->moe_param();
    const int expert_num = p.weights->num_experts();
    const int experts_per_token = moe.experts_per_token;
    const int max_expert_num = expert_num;  // all layers have same expert count

    h_offsets_ = {max_expert_num + 1, kCPUpinned};

    const int pad_token_num = (max_token_num_ + kMoeGateVecSize - 1) / kMoeGateVecSize * kMoeGateVecSize;

    masks_   = {max_expert_num * pad_token_num, kDEVICE};
    f2n_     = {experts_per_token * max_token_num_, kDEVICE};
    f2E_     = {experts_per_token * max_token_num_, kDEVICE};
    en2f_    = {experts_per_token * max_token_num_, kDEVICE};
    scales_  = {experts_per_token * max_token_num_, kDEVICE};
    offsets_ = {max_expert_num + 1, kDEVICE};
    accum_   = {max_expert_num * kMoeGateMaxTiles, kDEVICE};

    initialized_ = true;
}
```

**Forward** reads from weight:
```cpp
const auto& moe = p.weights->moe_param();
const int hidden_dim = p.weights->hidden_dim_;
const int inter_size = p.weights->inter_size_;
```

All `param_.*` references become `moe.*`. `hidden_dim_` becomes local `hidden_dim`. `inter_size_` becomes local `inter_size`.

**Combine** reads from weight:
```cpp
const auto& moe = p.weights->moe_param();
// moe.experts_per_token used in invokeMoeCombine
```

`param_.experts_per_token` becomes `moe.experts_per_token`.

### 4. Update call sites in unified_decoder.cc

**LlamaFfnLayer** (line 65):
```cpp
// Before:
ffn_layer_ = std::make_unique<LlamaFfnLayer>(model, ctx);
// After:
ffn_layer_ = std::make_unique<LlamaFfnLayer>(ctx);
```

**MoeFfnLayer** (line 52):
```cpp
// Before:
moe_ffn_layer_ = std::make_unique<MoeFfnLayer>(model, moe, engine, ctx);
// After:
moe_ffn_layer_ = std::make_unique<MoeFfnLayer>(engine, ctx);
```

### 5. expert_ffn_ creation in MoeFfnLayer

Always create `expert_ffn_` in constructor (it's lightweight — just stores a reference to `linear_`). Only used when `moe.method == MoeParam::kNaive` at runtime.

## Removed private members

### LlamaFfnLayer

| Removed member | Source | Reason |
|---|---|---|
| `hidden_units_` | `model.hidden_units` | Dead — never read |

### MoeFfnLayer

| Removed member | Source | Reason |
|---|---|---|
| `hidden_dim_` | `model.hidden_units` | Read from `p.weights->hidden_dim_` at runtime |
| `inter_size_` | `param.inter_size / engine.mlp_tp_size` | Read from `p.weights->inter_size_` at runtime |
| `param_` | `MoeParam` (full copy) | Read from `p.weights->moe_param()` at runtime |

## Added members to MoeFfnLayer

| Member | Purpose |
|---|---|
| `max_token_num_` | Stores `engine.max_forward_token_num * engine.attn_dp_size` for lazy buffer init |
| `initialized_` | Flag for lazy buffer initialization |

## Files touched

- `models/moe_weight.h` (make `hidden_dim_` public, add `inter_size_`)
- `models/moe_weight.cc` (populate `inter_size_` from config)
- `models/llama/LlamaFfnLayer.h` (remove ModelParam, remove dead member)
- `models/llama/LlamaFfnLayer.cc` (update constructor)
- `models/llama/moe_ffn_layer.h` (new constructor, add Init, remove members, add lazy-init members)
- `models/llama/moe_ffn_layer.cc` (constructor body, lazy Init, Forward/Combine read from weight)
- `models/llama/unified_decoder.cc` (update both call sites)
