# Eliminate Dot Navigation in TextModelLoader

**Date:** 2026-04-13
**Status:** Approved

## Problem

`text_model_loader.py` has two blocks that split parameter names by `.` and
dynamically create intermediate children, all hardcoded as `NormConfig`. This
conflates two different operations — navigating to a submodule vs. committing
a direct parameter — behind a single dotted-string encoding. The pattern is
fragile and hides the semantics of what is actually being created.

## Design

Split the spec methods so that norm children and direct params are returned
separately, then replace the dot-navigation loops with straightforward
iteration.

### Spec API Changes

**Before:**
- `attn_params(layer)` → `{dotted_name: (tensor, split_side)}`
- `linear_attn_params(layer)` → `{dotted_name: (tensor, split_side)}`

**After:**
- `attn_params(layer)` → `{name: (tensor, split_side)}` — direct params only, no dots
- `attn_norm_children(layer)` → `{name: tensor}` — norm submodule weights, always broadcast
- `linear_attn_params(layer)` → `{name: (tensor, split_side)}` — direct params only, no dots
- `linear_attn_norm_children(layer)` → `{name: tensor}` — norm submodule weights, always broadcast

Base class provides empty-dict defaults for all four methods.

### Migration Per Spec

**Attention params:**

| Spec | `attn_params` (after) | `attn_norm_children` |
|------|----------------------|---------------------|
| qwen3 | `{}` | `{"q_norm": t, "k_norm": t}` |
| qwen3.5 | `{}` | `{"q_norm": t, "k_norm": t}` |
| glm4_moe_lite | `{}` | `{"q_a_layernorm": t, "kv_a_layernorm": t}` |
| gpt_oss | `{"sinks": (t, SplitSide.OUTPUT)}` | `{}` |

**Linear attention params:**

| Spec | `linear_attn_params` (after) | `linear_attn_norm_children` |
|-------|----------------------------|---------------------------|
| qwen3.5 | `{"A_log": (t, OUTPUT), "dt_bias": (t, OUTPUT), "conv1d": (t, OUTPUT)}` | `{"norm": t}` |

### Loader Changes

The two dot-navigation blocks in `text_model_loader.py` are replaced:

**`_process_attention`:**
```python
# --- Direct params (sinks, etc.) ---
for name, (tensor, split_side) in spec.attn_params(layer).items():
    attn.commit_tensor(name, tensor, split_side=split_side)

# --- Norm children (q_norm, k_norm, etc.) ---
for name, tensor in spec.attn_norm_children(layer).items():
    child = attn.create_child(name, NormConfig(
        dim=tensor.shape[-1],
        data_type=dtype))
    child.commit_tensor('weight', tensor)
```

**`_process_linear_attn`:**
```python
# --- Direct params (A_log, dt_bias, conv1d) ---
for name, (tensor, split_side) in spec.linear_attn_params(layer).items():
    linear_attn.commit_tensor(name, tensor, split_side=split_side)

# --- Norm children (norm, etc.) ---
for name, tensor in spec.linear_attn_norm_children(layer).items():
    child = linear_attn.create_child(name, NormConfig(
        dim=tensor.shape[-1],
        data_type=dtype))
    child.commit_tensor('weight', tensor)
```

No other loader methods are affected.

### Testing

No new test infrastructure needed. Verify affected models via turbomind-tester:
- qwen3 (attn_norm_children)
- qwen3.5 (both attn and linear_attn norm children)
- glm4_moe_lite (attn_norm_children)

## Files Changed

- `lmdeploy/turbomind/deploy/spec.py` — add `attn_norm_children`, `linear_attn_norm_children` defaults
- `lmdeploy/turbomind/deploy/text_model_loader.py` — replace dot-nav blocks
- `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` — split attn_params
- `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` — split both
- `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` — split attn_params
- `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` — remove dots from attn_params
