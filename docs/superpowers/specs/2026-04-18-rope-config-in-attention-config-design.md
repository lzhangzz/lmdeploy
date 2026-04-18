# Move RopeParam into AttentionConfig as RopeConfig

## Goal

Replace the C++ `RopeParam` (union-based, old-parameter-system struct) with a flat `RopeConfig` X-macro struct nested inside `AttentionConfig`. Rope params flow from Python through the X-macro/pybind pipeline instead of YAML. `UnifiedAttentionLayer` reads rope from the `AttentionWeight` config instead of a constructor parameter. No behavioral changes.

## Current State

### C++ side

`RopeParam` (in `llama_rope.h`) is a struct with a discriminated union:

```cpp
struct RopeParam {
    RopeType type;
    float base;
    int   dim;
    float factor;
    int   max_position_embeddings;
    union {
        YarnRopeParam   yarn;     // attention_factor, beta_fast, beta_slow
        Llama3RopeParam llama3;   // low_freq_factor, high_freq_factor, original_max_position_embeddings
        MropeRopeParam  mrope;    // section (int3)
    };
};
```

It lives in the old parameter system:
- `AttentionParam::rope` (in `llama_params.h:103`)
- `UnifiedAttentionLayer::rope_` (constructor param, cached member)
- Populated from YAML in `turbomind.cc:65-138` by `parse_rope_param()` and its 7 helper functions

`init_rope_kernel_param(rope_, rope_param_)` converts `RopeParam` → `RopeKernelParam` for GPU kernels. It reads the union fields conditionally by `rope.type`.

`UnifiedAttentionLayer` also reads `rope_` directly in `init_dynamic_ntk()` (`.base`, `.factor`, `.dim`, `.max_position_embeddings`).

`AttentionConfig` (X-macro) has only two rope-adjacent scalar fields:
- `int rope_dim` (line 32)
- `int max_position_embeddings` (line 37)

### Python side

Python has a flat `RopeParam` dataclass (no union) in `config.py:108-120` with all variant fields as optional attributes. `parse_rope_param()` in `utils.py:31-97` creates it from HF config.

Specs store `self._rope` (Python `RopeParam`), then scatter individual fields onto `AttentionConfig` scalars:
- `attn_cfg.rope_dim = self._rope.dim`
- `attn_cfg.max_position_embeddings = self._max_position_embeddings`

The `_build_attention_config()` method in `spec.py:219-224` passes `rope_param=self._rope` to the Python `AttentionConfig` dataclass for YAML serialization. This is the **legacy YAML path** — separate from the X-macro `_tm.AttentionConfig` used in the builder pipeline.

## Changes

### 1. Add RopeConfig X-macro struct, nest in AttentionConfig

In `src/turbomind/models/attention_weight.h`, before `AttentionConfig`:

```cpp
using MropeSection = std::array<int, 3>;

struct RopeConfig {
    #define ROPE_FIELDS(X) \
        X(int,            type, 0) \
        X(float,          base, 10000.f) \
        X(float,          factor, 1.f) \
        X(int,            max_position_embeddings, 0) \
        X(float,          yarn_attention_factor, 1.f) \
        X(float,          yarn_beta_fast, 32.f) \
        X(float,          yarn_beta_slow, 1.f) \
        X(float,          llama3_low_freq_factor, 1.f) \
        X(float,          llama3_high_freq_factor, 4.f) \
        X(int,            llama3_original_max_position_embeddings, 0) \
        X(MropeSection,   mrope_section, {})

    ROPE_FIELDS(TM_MEMBER)
    TM_FOR_EACH(RopeConfig, ROPE_FIELDS)

    #undef ROPE_FIELDS
};
```

`RopeConfig` is a plain struct — no `ModuleConfig` base. It's a data container nested in `AttentionConfig`, not an independent module config.

Add to `ATTENTION_FIELDS` in `AttentionConfig`:

```cpp
    X(RopeConfig, rope, {})
```

Remove the now-redundant scalar fields from `ATTENTION_FIELDS`:
- `X(int, rope_dim, ...)` — replaced by `rope.dim`
- `X(int, max_position_embeddings, ...)` — replaced by `rope.max_position_embeddings`

### 2. Add bind_struct template and register RopeConfig

In `src/turbomind/python/bind.cpp`, add a new template alongside the existing `bind_config`:

```cpp
template<typename T>
void bind_struct(py::module_& m, const char* name) {
    py::class_<T> cls(m, name);
    cls.def(py::init<>());
    T::for_each([&](const char* fname, auto member_ptr) {
        cls.def_readwrite(fname, member_ptr);
    });
}
```

Then register `RopeConfig`:

```cpp
bind_struct<turbomind::core::RopeConfig>(m, "RopeConfig");
```

This must come before `bind_config<AttentionConfig>` so the `RopeConfig` Python type exists when `def_readwrite("rope", &AttentionConfig::rope)` is called.

### 3. Add rope_type_to_int helper in Python

In `lmdeploy/turbomind/deploy/source_model/utils.py`, add:

```python
_ROPE_TYPE_MAP = {
    'default': 1,
    'linear': 2,
    'dynamic': 3,
    'yarn': 4,
    'llama3': 5,
    'mrope': 6,
}

def rope_type_to_int(type_str: str) -> int:
    return _ROPE_TYPE_MAP[type_str]
```

Values match the `RopeType` enum in `llama_rope.h`: `kNull=0, kDefault=1, kLinear=2, kDynamic=3, kYarn=4, kLlama3=5, kMrope=6`.

### 4. Update Python specs to populate attn_cfg.rope

Each spec's `__init__` currently sets scalar `attn_cfg.rope_dim` and `attn_cfg.max_position_embeddings`. Replace with:

```python
# Old:
self._attn_cfg.rope_dim = self._rope.dim
self._attn_cfg.max_position_embeddings = self._max_position_embeddings

# New:
self._attn_cfg.rope.type = rope_type_to_int(self._rope.type)
self._attn_cfg.rope.base = self._rope.base
self._attn_cfg.rope.dim = self._rope.dim
self._attn_cfg.rope.factor = self._rope.factor
self._attn_cfg.rope.max_position_embeddings = self._max_position_embeddings
# Variant fields (set conditionally or always — flat fields, only relevant ones used):
if self._rope.type == 'yarn':
    self._attn_cfg.rope.yarn_attention_factor = self._rope.attention_factor
    self._attn_cfg.rope.yarn_beta_fast = self._rope.beta_fast
    self._attn_cfg.rope.yarn_beta_slow = self._rope.beta_slow
elif self._rope.type == 'llama3':
    self._attn_cfg.rope.llama3_low_freq_factor = self._rope.low_freq_factor
    self._attn_cfg.rope.llama3_high_freq_factor = self._rope.high_freq_factor
    self._attn_cfg.rope.llama3_original_max_position_embeddings = self._rope.original_max_position_embeddings
elif self._rope.type == 'mrope':
    self._attn_cfg.rope.mrope_section = self._rope.mrope_section
```

Specs that also read `self._rope.dim` for `reorder_rotary_emb_linear()` continue to do so — that's a weight-layout concern, not a config concern.

**Specs affected:**
- `qwen3_spec.py` — sets `rope_dim`, `max_position_embeddings`; uses `self._rope.dim` for reorder
- `qwen3_5_spec.py` — sets `rope_dim`, `max_position_embeddings`; overrides `self._rope.dim` with `partial_rotary_factor`; uses `self._rope.dim` for reorder
- `gpt_oss_spec.py` — sets `rope_dim`, `max_position_embeddings`; uses `self._rope.dim` for reorder
- `glm4_moe_lite_spec.py` — sets `rope_dim=0` (MLA), `max_position_embeddings`; overrides `self._rope.attention_factor` from YaRN params; uses `self._rope.dim` for reorder

### 5. Adapt init_rope_kernel_param to take RopeConfig

In `src/turbomind/models/llama/llama_rope.h`, add an overload or replace the existing function:

```cpp
// New signature:
void init_rope_kernel_param(const RopeConfig& rope, RopeKernelParam& rope_kernel);
```

Implementation reads flat fields instead of union:
- `rope.type` → `RopeType(rope.type)` for the enum cast
- `rope.base`, `rope.dim` — same as before
- `rope.yarn_attention_factor` instead of `rope.yarn.attention_factor`
- `rope.llama3_low_freq_factor` instead of `rope.llama3.low_freq_factor`
- `rope.mrope_section` instead of `rope.mrope.section`

### 6. Update AttentionWeight to store RopeConfig

In `attention_weight.h`, `AttentionWeight` currently caches scalar fields from `AttentionConfig`. Add:

```cpp
core::RopeConfig rope_{};
```

Populate in constructor (`attention_weight.cc`):
```cpp
rope_ = cfg.rope;
```

### 7. Replace RopeParam with RopeConfig in UnifiedAttentionLayer

Replace `const RopeParam rope_` member with `const core::RopeConfig rope_`. The constructor receives a `const core::RopeConfig&` parameter (instead of `const RopeParam&`) and caches it.

`init_rope_kernel_param(rope_, rope_param_)` in the constructor body works unchanged after step 5's overload is added.

For `init_dynamic_ntk()`, change signature to take `const core::RopeConfig&` instead of `const RopeParam&`, reading `.base`, `.factor`, `.dim`, `.max_position_embeddings` from the flat struct. No call-site changes needed — it already receives `rope_`.

### 8. Remove RopeParam pass-through from UnifiedDecoder

In `unified_decoder.cc`, the constructor currently passes `attn.rope` to `UnifiedAttentionLayer`:

```cpp
// Old:
attn_layer_ = std::make_unique<UnifiedAttentionLayer>(
    model.quant_policy, model.layer_types, model.layer_num,
    attn.rope,                          // <-- remove
    attn.cache_block_seq_len, engine, ctx, phases, ...);
```

After the change, rope lives in `AttentionConfig` → `AttentionWeight`, so no explicit pass-through needed.

### 9. Remove AttentionParam::rope

Remove the `RopeParam rope` field from `AttentionParam` in `llama_params.h`. The YAML rope parsing functions in `turbomind.cc` (`parse_rope_param` and its 7 helpers) become dead code for the new pipeline path. Leave them in place for now (the old YAML path still uses them until fully deprecated).

### 10. Clean up scalar fields on AttentionConfig

Remove from `ATTENTION_FIELDS`:
- `X(int, rope_dim, ...)` — now `rope.dim`
- `X(int, max_position_embeddings, ...)` — now `rope.max_position_embeddings`

These are read in `attention_weight.cc` constructor and stored on `AttentionWeight`. After removal, consumers read them from `weights.rope_.dim` and `weights.rope_.max_position_embeddings` instead.

**Consumers of `weights.rope_dim_`:** None found — `AttentionWeight` doesn't expose it publicly and it's only used internally. `head_dim` is used instead for kernel param computation.

**Consumers of `weights.max_position_embeddings_`:** `unified_attention_layer.cc:525` — `params.max_position_embeddings = weights.max_position_embeddings_`. After change: `params.max_position_embeddings = weights.rope_.max_position_embeddings`.

## Removed members

| Removed member | Class | Source |
|---|---|---|
| `rope_` (RopeParam) | UnifiedAttentionLayer | Constructor param → `RopeConfig` param instead |
| `rope` (RopeParam) | AttentionParam | Old param system → RopeConfig in AttentionConfig |
| `rope_dim` (int) | AttentionConfig scalar | → `rope.dim` |
| `max_position_embeddings` (int) | AttentionConfig scalar | → `rope.max_position_embeddings` |

## Scope

Pure refactoring. No behavioral changes. No kernel changes. `RopeKernelParam` (GPU-facing struct) is unchanged.

## Files touched

- `src/turbomind/models/attention_weight.h` (add RopeConfig, nest in AttentionConfig, remove scalar rope_dim/max_position_embeddings, add rope_ to AttentionWeight)
- `src/turbomind/models/attention_weight.cc` (populate rope_ from config)
- `src/turbomind/python/bind.cpp` (register RopeConfig)
- `src/turbomind/models/llama/llama_rope.h` (add init_rope_kernel_param overload for RopeConfig)
- `src/turbomind/models/llama/unified_attention_layer.h` (remove RopeParam param and member)
- `src/turbomind/models/llama/unified_attention_layer.cc` (read rope from AttentionWeight, update init_dynamic_ntk)
- `src/turbomind/models/llama/unified_decoder.cc` (remove attn.rope from constructor call)
- `src/turbomind/models/llama/llama_params.h` (remove RopeParam from AttentionParam)
- `lmdeploy/turbomind/deploy/source_model/utils.py` (add rope_type_to_int helper)
- `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` (populate attn_cfg.rope)
- `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` (populate attn_cfg.rope)
- `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` (populate attn_cfg.rope)
- `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` (populate attn_cfg.rope)
