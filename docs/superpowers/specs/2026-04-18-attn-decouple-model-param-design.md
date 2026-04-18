# Decouple UnifiedAttentionLayer from ModelParam and AttentionParam

## Goal

Remove UnifiedAttentionLayer's dependency on `ModelParam` and `AttentionParam`. Dimensional fields (head counts, head dim, MLA params) and behavioral fields (softmax_scale, use_logn_attn, max_position_embeddings) are read from `AttentionWeight` at runtime. Non-dimensional fields not in the weight are passed as individual constructor params. Head-count-dependent buffers use lazy initialization on first Forward call. No behavioral changes.

## Current State

`UnifiedAttentionLayer` constructor takes `const ModelParam& model, const AttentionParam& attn, const EngineParam& engine, int tp_size, const Context& ctx, int phases, bool init`.

From `ModelParam`: copies 10+ fields. Stores `head_num_`, `kv_head_num_`, `size_per_head_`, `hidden_units_` (dead), `local_head_num_`, `local_kv_head_num_` as members. Stores full `model_param_` copy accessed extensively in Forward, core_attention, forward_mla, qk_norm.

From `AttentionParam`: stores full `param_` copy. Uses `rope` (for rope_param_), `cache_block_seq_len`, `softmax_scale`, `use_logn_attn`, `max_position_embeddings`.

From `EngineParam`: stores full `engine_param_` copy for buffer sizing and context parallel params.

`AttentionWeight` already has the same dimensional fields as private members initialized from `AttentionConfig`, plus behavioral fields that match some AttentionParam fields.

## Changes

### 1. Make AttentionWeight fields public, add softmax_scale / use_logn_attn / max_position_embeddings

Move all private config fields to the public section of `AttentionWeight` in `attention_weight.h`. Remove the `window_size()` accessor (field becomes directly accessible). Keep `is_mla()` as a convenience method.

Add three fields to `AttentionConfig` X-macro:
```cpp
X(float,    softmax_scale, 0.f)
X(bool,     use_logn_attn, false)
X(int,      max_position_embeddings, 0)
```

Add corresponding public fields to `AttentionWeight`:
```cpp
float softmax_scale_{};
bool  use_logn_attn_{};
int   max_position_embeddings_{};
```

Populate them in `AttentionWeight(const core::AttentionConfig& cfg)` constructor.

Public fields: `hidden_dim_`, `head_dim_`, `head_num_`, `kv_head_num_`, `mla_`, `bias_`, `qk_norm_`, `tp_size_`, `tp_rank_`, `data_type_`, `window_size_`, `sink_`, `attn_output_gate_`, `softmax_scale_`, `use_logn_attn_`, `max_position_embeddings_`.

### 2. Rewrite UnifiedAttentionLayer constructor

**Before:**
```cpp
UnifiedAttentionLayer(const ModelParam& model, const AttentionParam& attn,
                      const EngineParam& engine, int tp_size,
                      const Context& ctx, int phases, bool init);
```

**After:**
```cpp
UnifiedAttentionLayer(float norm_eps,
                      int quant_policy,
                      const std::vector<int>& layer_types,
                      int layer_num,
                      const RopeParam& rope,
                      int cache_block_seq_len,
                      const EngineParam& engine,
                      const Context& ctx,
                      int phases,
                      bool init);
```

Constructor body:
- Store `norm_eps_`, `quant_policy_`, `rope_`, `cache_block_seq_len_`, `init_`
- `init_rope_kernel_param(rope_, rope_param_)`
- Compute `cache_layer_ids_` from `layer_types`, `layer_num`
- CUDA stream/event creation (unchanged)
- Rope buffer allocation (unchanged, no head count dependency)
- AttentionData allocation (unchanged, no head count dependency)
- Do NOT allocate `partial_O_`, `partial_ML_`, `split_cnt_`, `tmp_attn_` (deferred to Init)

### 3. Add lazy Init method

```cpp
void UnifiedAttentionLayer::Init(const ForwardParam& p) {
    const auto& w = *p.weights;
    const int tp_size = w.tp_size_;
    const int local_head_num = w.head_num_ / tp_size;
    const int local_kv_head_num = w.kv_head_num_ / tp_size;
    const int size_per_head = w.head_dim_;

    TM_CHECK_EQ(w.head_num_ % tp_size, 0);
    TM_CHECK_EQ(w.head_num_ % w.kv_head_num_, 0);

    ssize_t workspace_tokens = kMaxWorkspaceTokens;
    Allocator alloc = core::Context::device_alloc();
    if (engine_param_.attn_cp_size > 1) {
        alloc = GetSymmAllocator(context_.comm.d_comm);
        workspace_tokens += engine_param_.max_forward_token_num;
    }

    partial_O_ = Tensor_<float>({workspace_tokens, local_head_num, size_per_head}, kDEVICE);
    partial_ML_ = Tensor_<float>({engine_param_.attn_cp_size, workspace_tokens, local_head_num, 2}, alloc);
    split_cnt_  = Tensor_<int>({workspace_tokens}, kDEVICE);

    if (init_) {
        const int dim = local_head_num * size_per_head;
        tmp_attn_ = Tensor{{engine_param_.max_forward_token_num, dim}, w.data_type_, kDEVICE};
    }
    Clear(split_cnt_.buffer());
    initialized_ = true;
}
```

### 4. Update Forward to read from weight

In `Forward(ForwardParam p)`, call `Init(p)` if not initialized. Read from weight:

| Old | New |
|---|---|
| `model_param_.qk_norm` | `weights.qk_norm_` |
| `model_param_.attn_output_gate` | `weights.attn_output_gate_` |
| `model_param_.norm_eps` | `norm_eps_` (stored member) |

### 5. Update core_attention to compute locals from weight

Compute at start of `core_attention`:
```cpp
const auto& w = *p.weights;
const int tp_size = w.tp_size_;
const int local_head_num = w.head_num_ / tp_size;
const int local_kv_head_num = w.kv_head_num_ / tp_size;
const int size_per_head = w.head_dim_;
```

Replacement table for all member accesses:

| Old | New |
|---|---|
| `local_head_num_` | local `local_head_num` |
| `local_kv_head_num_` | local `local_kv_head_num` |
| `size_per_head_` | local `size_per_head` |
| `model_param_.mla.kv_lora_rank > 0` | `w.is_mla()` |
| `model_param_.attn_output_gate` | `w.attn_output_gate_` |
| `model_param_.quant_policy` | `quant_policy_` (stored member) |
| `param_.cache_block_seq_len` | `cache_block_seq_len_` (stored member) |
| `param_.softmax_scale` | `w.softmax_scale_` |
| `param_.use_logn_attn` | `w.use_logn_attn_` |
| `param_.max_position_embeddings` | `w.max_position_embeddings_` |

### 6. Update forward_mla and qk_norm

**forward_mla:** `model_param_.norm_eps` -> `norm_eps_`

**qk_norm:** `model_param_.norm_eps` -> `norm_eps_`, `model_param_.attn_bias` -> `weights.bias_`

### 7. Update Run to use stored rope_

`param_.rope` -> `rope_` (stored RopeParam member)

### 8. Update call site in unified_decoder.cc

**Before:**
```cpp
attn_layer_ = std::make_unique<UnifiedAttentionLayer>(model, attn, engine, attn_tp_size_, ctx, phases, (bool)moe_ffn_layer_);
```

**After:**
```cpp
attn_layer_ = std::make_unique<UnifiedAttentionLayer>(
    model.norm_eps,
    model.quant_policy,
    model.layer_types,
    model.layer_num,
    attn.rope,
    attn.cache_block_seq_len,
    engine, ctx, phases, (bool)moe_ffn_layer_);
```

## Removed private members

| Removed member | Source | Reason |
|---|---|---|
| `head_num_` | `model.head_num` | Read from weight at runtime |
| `kv_head_num_` | `model.kv_head_num` | Read from weight at runtime |
| `size_per_head_` | `model.head_dim` | Read from weight at runtime |
| `hidden_units_` | `model.hidden_units` | Dead - never used |
| `local_head_num_` | derived | Computed locally |
| `local_kv_head_num_` | derived | Computed locally |
| `model_param_` | ModelParam | Fully decomposed |
| `param_` | AttentionParam | Fully decomposed |

## Added members

| Member | Source | Purpose |
|---|---|---|
| `norm_eps_` | direct param | Used in forward_mla, qk_norm |
| `quant_policy_` | direct param | Used in core_attention |
| `rope_` | direct param (RopeParam) | Used in Run for init_dynamic_ntk |
| `cache_block_seq_len_` | direct param | Used in constructor and core_attention |
| `init_` | direct param | Controls tmp_attn_ allocation in Init |
| `initialized_` | lazy init flag | Guards Init call |

## Scope

Pure refactoring. No behavioral changes. No kernel changes.

## Files touched

- `models/attention_weight.h` (make fields public, add softmax_scale/use_logn_attn/max_position_embeddings)
- `models/attention_weight.cc` (populate new fields in constructor)
- `models/llama/unified_attention_layer.h` (new constructor, remove members, add Init, add new members)
- `models/llama/unified_attention_layer.cc` (constructor body, Init, Forward/core_attention/forward_mla/qk_norm/Run/Setup read from weight or stored members)
- `models/llama/unified_decoder.cc` (call site)
