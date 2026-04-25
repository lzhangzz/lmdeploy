# model_dtype simplification

## Problem

`model_dtype` (the C++ `DataType` enum for the model's compute dtype) is redundantly threaded through the model loading pipeline. It is passed as an explicit parameter on every `_add_linear` and `_add_tensor` call, despite every builder already carrying the same value on `self.config.data_type`. This creates:

- Boilerplate at ~18 call sites
- Two parallel "sources of truth" (config field and explicit parameter) that could theoretically disagree
- An unnecessary `_infer_compute_dtype` fallback path

Investigation confirmed: every builder instance uses a single, immutable dtype throughout its lifetime. `config.data_type` is set at construction and never modified.

## Design

### Architecture

Single source of truth per layer:

```
spec._dtype (Python attr, set once in __init__)
  └── config.data_type (C++ struct field on every config template, including ModelWeightConfig)
        └── self.config.data_type (read by Builder._add_linear internally)
```

`model_dtype` parameter disappears from `_add_linear`, `_add_tensor`, and every builder method that passes it through.

### C++ change: `ModelWeightConfig.data_type`

`ModelWeightConfig` currently has no `data_type` field — only `tp_size` and `tp_rank`. This prevents `TextModelBuilder` from reading `self.config.data_type` in `_add_linear` (called from `add_lm_head`).

Add `X(DataType, data_type)` to `MODEL_WEIGHT_FIELDS` in `model_weight.h`. Copy `cfg.data_type` in `ModelWeight`'s constructor. After this change, every config type used by any builder has `data_type`, and `_add_linear` can uniformly read `self.config.data_type`.

### Spec side

`TextModelSpec.__init__` stores `self._dtype = self._cpp_dtype()` once. Every spec subclass then uses `self._dtype` to set `data_type` on config templates and `cfg.data_type` on `ModelWeightConfig`. This replaces:

- `dtype = self._cpp_dtype()` in each spec's `__init__`
- `data_type=self._cpp_dtype()` in `model()` for TextModelBuilder (now set on `cfg.data_type` before the constructor)
- `data_type=self._cpp_dtype()` in `norm()` default
- `model_dtype=self._cpp_dtype()` in `add_gate()` calls (removed entirely)

### `_add_linear` (Builder side)

Remove `model_dtype` parameter. Read `self.config.data_type` directly for both:
1. `lin_cfg.data_type` (the C++ module's compute dtype)
2. `alloc_dtype` fallback when `packed.alloc_dtype is None and kind == 'weight'`

Remove `_infer_compute_dtype` — it was only reached when `model_dtype is None`, which no longer happens since every config carries `data_type`.

### `_add_tensor` / `_commit_tensor` (Builder side)

Remove `model_dtype` parameter. Always pass `alloc_dtype=None` to `_copy_shard_to_param`, letting C++ infer the dtype from the tensor's native dtype. This is the correct behavior for all non-linear params (norms, scalars, embeddings, conv filters, etc.) — the tensor from the checkpoint already has the right dtype.

### `dequant_mixed` / `_dequant_linear` / `_ensure_compatible_formats`

These helpers still receive `data_type` as a keyword argument since they need the target compute dtype for dequantization. The callers read `self.config.data_type` to pass it. No change to these function signatures.

### TextModelBuilder

`data_type` parameter removed from constructor. `add_lm_head` no longer passes `model_dtype=` — `_add_linear` reads `self.config.data_type` (now available on `ModelWeightConfig`). `add_token_embeds` no longer passes `model_dtype=` — `_add_tensor` passes `alloc_dtype=None` and C++ infers from the embedding tensor's native dtype.

### Files changed

| File | Change |
|------|--------|
| `src/turbomind/models/model_weight.h` | Add `X(DataType, data_type)` to `MODEL_WEIGHT_FIELDS` |
| `src/turbomind/models/model_weight.cc` | Copy `cfg.data_type` in constructor |
| `builder/_base.py` | Remove `model_dtype` from `_add_linear`, `_add_tensor`, `_commit_tensor`. Read `self.config.data_type` in `_add_linear`. Pass `alloc_dtype=None` in `_add_tensor`. Remove `_infer_compute_dtype`. Remove `data_type` param from `TextModelBuilder.__init__`. |
| `builder/attention.py` | Remove `model_dtype=` from `_add_linear` calls in `add_qkv_proj`, `add_o_proj`. |
| `builder/ffn.py` | Remove `model_dtype=` from all `_add_linear` calls in `add_ffn`. |
| `builder/mla.py` | Remove `model_dtype=` from `_add_linear` calls in `add_projections`. |
| `builder/moe.py` | Remove `model_dtype` parameter from `add_gate`. |
| `builder/deltanet.py` | Remove `model_dtype=` from `_add_linear` calls in `add_input_projections`. |
| `spec.py` | Store `self._dtype = self._cpp_dtype()` in `__init__`. `norm()` defaults `data_type` to `self._dtype`. |
| `source_model/qwen3_spec.py` | Use `self._dtype`. Set `cfg.data_type` on `ModelWeightConfig`. Remove `model_dtype=` from `add_gate()`. Remove `data_type=` from `TextModelBuilder(...)`. |
| `source_model/qwen3_5_spec.py` | Same. |
| `source_model/gpt_oss_spec.py` | Same. |
| `source_model/glm4_moe_lite_spec.py` | Same. |

### Key invariants

- `_add_linear` always receives a non-None dtype from `self.config.data_type`
- `_add_tensor` always passes `alloc_dtype=None`, preserving tensor native dtype
- `config.data_type` is immutable after builder construction
- `spec._dtype` equals every `config.data_type` it sets
- Every config struct used by a builder has a `data_type` field, including `ModelWeightConfig`

## Test plan

- Build: `ninja` from the `build` directory
- Run `scripts/test_turbomind_model.py` on one dense model (Qwen3) and one MoE model (Qwen3-MoE), TP=1
- Verify each responds with meaningful text (at least 128 tokens)
- Dtype mismatch produces gibberish or crash — clean responses validate correctness
