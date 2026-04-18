# Move norm_eps from ModelParam into NormWeight

## Goal

Move `norm_eps` from `ModelParam` into `NormWeight`. Each NormWeight instance carries its own eps value (all identical). Layer constructors lose their `norm_eps` parameter and cached member. At each `invokeRMSNorm` call site, eps is read from the NormWeight being used. No behavioral changes.

## Current State

`norm_eps` is a `float` field on `ModelParam`, set once in `turbomind.cc` from YAML config. It is passed down through constructors to three layers, each caching it:

- `UnifiedDecoder::rmsnorm_eps_` — used in 5 `invokeRMSNorm` / `AllreduceResidualRMSnorm` calls
- `UnifiedAttentionLayer::norm_eps_` — used in 4 `invokeRMSNorm` calls (forward_mla, qk_norm)
- `GatedDeltaNetLayer::norm_eps_` — used in 1 `invokeRMSNormGated` call

Every usage is paired with a `NormWeight->weight` tensor — the norm weight and its eps always appear together.

`NormWeight` has no `norm_eps` field. Its X-macro declares only `weight` (Tensor). `NormConfig` has `dim` and `data_type` only.

## Changes

### 1. Add norm_eps to NormConfig X-macro and NormWeight

In `src/turbomind/models/norm_weight.h`, add to the `NormConfig` X-macro:

```cpp
#define NORM_FIELDS(X) \
    X(int,      dim) \
    X(DataType, data_type) \
    X(float,    norm_eps, 0.f)
```

Add public field to `NormWeight` (after the existing public methods, before `private:`):

```cpp
float norm_eps_{};
```

Populate it in `NormWeight(const core::NormConfig& cfg)` constructor body in `norm_weight.cc`:

```cpp
norm_eps_ = cfg.norm_eps;
```

### 2. Update Python norm config creation

Two paths create NormConfig instances from Python. Both need to set `norm_eps`.

**Path A: `make_norm_config()` in `lmdeploy/turbomind/deploy/builder/norm.py`**

Add `norm_eps` parameter:

```python
def make_norm_config(*, dim, data_type, norm_eps=0.):
    cfg = _tm.NormConfig()
    cfg.dim = dim
    cfg.data_type = data_type
    cfg.norm_eps = norm_eps
    return cfg
```

**Path B: `_add_norm_child()` in `lmdeploy/turbomind/deploy/builder/_base.py`**

Add `norm_eps` parameter and set it on the config:

```python
def _add_norm_child(self, name, tensor, data_type=None, norm_eps=0.):
    ...
    norm_cfg = make_norm_config(dim=tensor.shape[-1], data_type=data_type, norm_eps=norm_eps)
    ...
```

**Spec files** — update all spec's `norm()` / `output_norm()` methods to pass `self._norm_eps`:

```python
cfg = make_norm_config(dim=self._hidden_units, data_type=self._cpp_dtype(), norm_eps=self._norm_eps)
```

**Attention builder** — update `add_qk_norm()` and any MLA/DeltaNet norm child creation to pass `norm_eps` from the spec's attention config or directly from the spec.

### 3. Remove norm_eps from UnifiedAttentionLayer

Remove `float norm_eps` constructor parameter and `const float norm_eps_` member.

Update call sites in `forward_mla` and `qk_norm` to read from the NormWeight:

| Old | New |
|---|---|
| `norm_eps_` (with `w.q_a_layernorm->weight`) | `w.q_a_layernorm->norm_eps_` |
| `norm_eps_` (with `w.kv_a_layernorm->weight`) | `w.kv_a_layernorm->norm_eps_` |
| `norm_eps_` (with `weights.q_norm->weight`) | `weights.q_norm->norm_eps_` |
| `norm_eps_` (with `weights.k_norm->weight`) | `weights.k_norm->norm_eps_` |

### 4. Remove norm_eps from GatedDeltaNetLayer

Remove `float norm_eps` constructor parameter and `float norm_eps_` member.

Read `weights.norm->norm_eps_` at the `invokeRMSNormGated` call site.

### 5. Remove rmsnorm_eps_ from UnifiedDecoder

Remove `const float rmsnorm_eps_` member and its initialization from `model.norm_eps`.

In `Forward`, read eps from available NormWeight instances at each call site:

| Call site | Old | New |
|---|---|---|
| Initial norm (line ~197) | `rmsnorm_eps_` | `weights.at(0)->attention_norm->norm_eps_` |
| Post-attn residual (lines ~243-250) | `rmsnorm_eps_` | `weights.at(layer)->ffn_norm->norm_eps_` |
| Post-FFN residual (lines ~284-291) | `rmsnorm_eps_` | `weights.at(layer)->ffn_norm->norm_eps_` |

Note: the last-layer case uses a raw output norm tensor from args, not a NormWeight. Use `weights.at(layer)->ffn_norm->norm_eps_` (same value).

### 6. Update call sites in unified_decoder.cc

```cpp
// Before:
attn_layer_ = std::make_unique<UnifiedAttentionLayer>(
    model.norm_eps, model.quant_policy, ...);

// After:
attn_layer_ = std::make_unique<UnifiedAttentionLayer>(
    model.quant_policy, ...);
```

```cpp
// Before:
linear_attn_layer_ = std::make_unique<GatedDeltaNetLayer>(
    model.norm_eps, model.linear_state_dtype, ...);

// After:
linear_attn_layer_ = std::make_unique<GatedDeltaNetLayer>(
    model.linear_state_dtype, ...);
```

## Removed members

| Removed member | Class | Source |
|---|---|---|
| `norm_eps_` | UnifiedAttentionLayer | Constructor param → read from NormWeight |
| `norm_eps_` | GatedDeltaNetLayer | Constructor param → read from NormWeight |
| `rmsnorm_eps_` | UnifiedDecoder | ModelParam → read from NormWeight |

## Scope

Pure refactoring. No behavioral changes. No kernel changes.

## Files touched

- `src/turbomind/models/norm_weight.h` (add norm_eps to X-macro, add public field)
- `src/turbomind/models/norm_weight.cc` (populate from config)
- `lmdeploy/turbomind/deploy/builder/norm.py` (add norm_eps param to make_norm_config)
- `lmdeploy/turbomind/deploy/builder/_base.py` (add norm_eps param to _add_norm_child)
- `lmdeploy/turbomind/deploy/source_model/*.py` (pass norm_eps in norm/output_norm methods and add_qk_norm/add_mla_norm calls)
- `src/turbomind/models/llama/unified_attention_layer.h` (remove norm_eps param and member)
- `src/turbomind/models/llama/unified_attention_layer.cc` (read from NormWeight)
- `src/turbomind/models/llama/GatedDeltaNetLayer.h` (remove norm_eps param and member)
- `src/turbomind/models/llama/GatedDeltaNetLayer.cc` (read from NormWeight)
- `src/turbomind/models/llama/unified_decoder.h` (remove rmsnorm_eps_)
- `src/turbomind/models/llama/unified_decoder.cc` (read from NormWeight, update constructor calls)
