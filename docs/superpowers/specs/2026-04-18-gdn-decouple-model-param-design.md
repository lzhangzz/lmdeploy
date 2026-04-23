# Decouple GatedDeltaNetLayer from ModelParam and AttentionParam

## Goal

Remove GatedDeltaNetLayer's dependency on `ModelParam` and `AttentionParam`. At runtime, read dimensional fields from the `DeltaNetWeight` passed in `ForwardParam`. Pass the few non-dimensional fields (`norm_eps`, `state_dtype`, `layer_types`) directly through the constructor. `tp_size` comes from `EngineParam::attn_tp_size` (already a constructor param). No behavioral changes.

## Current State

`GatedDeltaNetLayer` constructor takes `const ModelParam& model` and `const AttentionParam& attn`. It copies 11 fields from `ModelParam` into private members and computes derived values. `AttentionParam` is completely unused.

`DeltaNetWeight` stores the same dimensional fields as private members initialized from `DeltaNetConfig`, but provides no public access.

## Changes

### 1. Make DeltaNetWeight fields public

Move the private dimensional fields in `delta_net_weight.h` to the public section. No getters — direct field access.

Fields to make public: `hidden_dim_`, `num_k_heads_`, `num_v_heads_`, `key_head_dim_`, `value_head_dim_`, `d_conv_`, `data_type_`, `tp_size_`, `tp_rank_`, `bias_`.

### 2. Rewrite GatedDeltaNetLayer constructor

**Before:**
```cpp
GatedDeltaNetLayer(const ModelParam& model, const AttentionParam& attn,
                   const EngineParam& engine, int tp_size,
                   const Context& ctx, int phases);
```

**After:**
```cpp
GatedDeltaNetLayer(float norm_eps, DataType state_dtype,
                   const std::vector<int>& layer_types,
                   const EngineParam& engine,
                   const Context& ctx, int phases);
```

Constructor body:
- Store `norm_eps_`, `state_dtype_`, `layer_types_` from params
- Store `tp_size_` from `engine.attn_tp_size`
- Compute `num_linear_layers_` from `layer_types_` (same as before)
- Buffer allocation and CUDA setup unchanged
- Drop all dimensional member initialization (hidden_units_, num_k/v_heads_, key/value_head_dim_, d_conv_, key_dim_, value_dim_, conv_dim_, dtype_)

### 3. Update Forward to read from DeltaNetWeight

In `Forward(ForwardParam p)`, read dimensional fields from `p.weights` (the `const DeltaNetWeight*` already passed via `ForwardParam`):

```cpp
const auto& w = *p.weights;
const int num_k_heads = w.num_k_heads_ / tp_size_;
const int num_v_heads = w.num_v_heads_ / tp_size_;
const int key_head_dim = w.key_head_dim_;
const int value_head_dim = w.value_head_dim_;
const int d_conv = w.d_conv_;
const int key_dim = num_k_heads * key_head_dim;
const int value_dim = num_v_heads * value_head_dim;
const int conv_dim = key_dim * 2 + value_dim;
```

These replace all uses of the removed member variables in Forward.

### 4. Remove dead `dtype_` usage in Run()

In `Run(BatchOp::kAdd)`, remove the unused `const auto dtype = dtype_` line. The handler body is a no-op (`for` loop with empty body).

### 5. Update call site in unified_decoder.cc

**Before:**
```cpp
linear_attn_layer_ = std::make_unique<GatedDeltaNetLayer>(model, attn, engine, attn_tp_size_, ctx, phases);
```

**After:**
```cpp
linear_attn_layer_ = std::make_unique<GatedDeltaNetLayer>(
    model.norm_eps, model.linear_state_dtype, model.layer_types,
    engine, ctx, phases);
```

### 6. Keep llama_params.h include (still needed for EngineParam)

`EngineParam` is defined in `llama_params.h` and is still used in the constructor. Keep the include. The dependency reduction is that `ModelParam` and `AttentionParam` are no longer used — they can be removed from any `using` declarations or forward references, but the include stays.

## Removed private members from GatedDeltaNetLayer

| Removed member | Source |
|---|---|
| `hidden_units_` | `model.hidden_units` — unused in Forward |
| `num_k_heads_` | `model.linear_num_key_heads / tp_size` — read from weight at runtime |
| `num_v_heads_` | `model.linear_num_value_heads / tp_size` — read from weight at runtime |
| `key_head_dim_` | `model.linear_key_head_dim` — read from weight at runtime |
| `value_head_dim_` | `model.linear_value_head_dim` — read from weight at runtime |
| `d_conv_` | `model.linear_conv_kernel_dim` — read from weight at runtime |
| `key_dim_` | derived — computed locally in Forward |
| `value_dim_` | derived — computed locally in Forward |
| `conv_dim_` | derived — computed locally in Forward |
| `dtype_` | `model.data_type` — dead code outside Forward; Forward uses `p.input.dtype()` |

## Scope

Pure refactoring. No behavioral changes. No kernel changes.

Files touched:
- `models/delta_net_weight.h` (make fields public)
- `models/llama/GatedDeltaNetLayer.h` (new constructor, remove members)
- `models/llama/GatedDeltaNetLayer.cc` (constructor body, Forward reads from weight)
- `models/llama/unified_decoder.cc` (call site)
