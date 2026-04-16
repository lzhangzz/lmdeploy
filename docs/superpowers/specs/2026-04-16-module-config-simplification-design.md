# Module Config Simplification

Eliminate the Python dataclass config layer in `module_configs.py`. Factory functions construct C++ pybind structs directly, removing the manual field-by-field `from_model_config()` and `to_cpp()` cloning.

## Problem

Config values flow through three layers with two manual field-by-field copy steps:

```
ModelConfig (config.py) --from_model_config()--> Python dataclass --to_cpp()--> C++ struct (_tm.*)
```

Adding a single field (e.g. a new attention parameter) requires touching `from_model_config()` AND `to_cpp()` on every affected config class. The Python dataclass layer provides no value -- it mirrors the C++ struct 1:1 with identical field names.

Additionally, `SpecAttnConfig` and `TextModelSpec.configure()` form a third clone layer: the loader constructs a `SpecAttnConfig` from `ModelConfig` fields, then `configure()` unpacks it back into individual `self._` attributes on the spec. These attributes are redundant -- the specs already have `self._mc` (ModelConfig) and can read the values directly.

## Solution

### 1. Delete all Python dataclass configs

Remove from `module_configs.py`:

- `LinearConfig`, `AttentionConfig`, `MLAConfig`, `FfnConfig`, `MoeConfig`, `DeltaNetConfig`
- `NormConfig`, `ModuleListConfig`, `DecoderLayerConfig`
- `SpecAttnConfig`
- `k_type_name` (dead code -- never read by Python or C++)

Replace each with a factory function that constructs the C++ pybind struct directly:

```python
def make_attention_config(mc: ModelConfig, *, tp_size, tp_rank, dtype,
                         window_size, rope_dim=0, permute_qk=True,
                         repeat_kv=0) -> _tm.AttentionConfig:
    cfg = _tm.AttentionConfig()
    cfg.hidden_dim = mc.hidden_units
    cfg.head_dim = mc.size_per_head
    cfg.head_num = mc.head_num
    cfg.kv_head_num = mc.kv_head_num
    cfg.kv_lora_rank = mc.kv_lora_rank or 0
    cfg.q_lora_rank = mc.q_lora_rank or 0
    cfg.qk_rope_dim = mc.qk_rope_dim or 0
    cfg.v_head_dim = mc.v_head_dim or 0
    cfg.has_bias = mc.attn_bias
    cfg.qk_norm = mc.qk_norm
    cfg.tp_size = tp_size
    cfg.tp_rank = tp_rank
    cfg.data_type = _tm.DataType(dtype)
    cfg.window_size = window_size
    cfg.attn_sink = mc.attn_sink
    cfg.attn_output_gate = mc.attn_output_gate
    return cfg
```

Similarly: `make_linear_config`, `make_ffn_config`, `make_moe_config`, `make_mla_config`, `make_deltanet_config`.

`NormConfig`/`ModuleListConfig`/`DecoderLayerConfig` are trivial (0-2 fields) -- construct their C++ structs inline at the call site.

### 2. Replace `for_rank()` / `to_cpp()` in Builder

The Builder currently calls:
```python
cfg = self.config.for_rank(rank).to_cpp()
```

Replace with a helper that clones a C++ struct and sets `tp_rank`:

```python
import copy

def _clone_with_rank(cfg, rank):
    """Return a copy of a pybind config struct with tp_rank changed."""
    new = copy.copy(cfg)
    new.tp_rank = rank
    return new
```

In `Builder._ensure_handles()`:
```python
cfg = _clone_with_rank(self.config, rank)
handle = _tm.create_module(cfg)
```

Configs without `tp_rank` (Norm, Linear, etc.) are unaffected -- they are already identity in `for_rank()`.

### 3. Simplify FfnBuilder config mutation

Currently:
```python
self.config = replace(self.config, fuse_silu=fused_silu)
```

With a mutable C++ struct, this becomes direct field assignment:
```python
self.config.fuse_silu = fused_silu
```

### 4. Eliminate SpecAttnConfig and configure()

Delete `SpecAttnConfig` and `TextModelSpec.configure()`.

The five `self._` attributes they set are redundant:

| Attribute | Derive from |
|---|---|
| `_head_dim` | `self._mc.size_per_head` |
| `_rope_dim` | `self._mc.size_per_head` (or rope_param from attention config) |
| `_kv_head_num` | `self._mc.kv_head_num` |
| `_attn_output_gate` | `self._mc.attn_output_gate` |
| `_permute_qk` | Always True -- remove entirely |

In `TextModelLoader.__call__()`, delete the `spec.configure(SpecAttnConfig(...))` call. Specs read these values directly from `self._mc`.

In spec files (qwen3_spec.py, qwen3_5_spec.py, gpt_oss_spec.py, glm4_moe_lite_spec.py), replace:
- `self._head_dim` -> `self._mc.size_per_head`
- `self._rope_dim` -> `self._mc.size_per_head` (or appropriate rope dim derivation)
- `self._permute_qk` -> `True` (inline, then simplify conditionals that check it)
- `self._kv_head_num` -> `self._mc.kv_head_num`

### 5. MLAConfig special case

`MLAConfig.to_cpp()` maps to `_tm.AttentionConfig` with field reinterpretation (e.g. `size_per_head -> head_dim`). The factory function handles this naturally:

```python
def make_mla_config(mc, *, tp_size, tp_rank, dtype, window_size, qk_nope_dim=0):
    qk_rope_dim = mc.qk_rope_dim or 0
    kv_lora_rank = mc.kv_lora_rank or 0
    v_head_dim = mc.v_head_dim or 0
    size_per_head = qk_nope_dim + qk_rope_dim
    if kv_lora_rank and kv_lora_rank != qk_nope_dim:
        size_per_head = kv_lora_rank + qk_rope_dim
        v_head_dim = kv_lora_rank

    cfg = _tm.AttentionConfig()
    cfg.head_dim = size_per_head  # reinterpretation
    # ... rest of fields
    return cfg
```

## Files changed

| File | Action |
|---|---|
| `module_configs.py` | Delete all dataclasses. Add factory functions + `_clone_with_rank`. |
| `builder.py` | Update `_ensure_handles()` to use `_clone_with_rank`. Update FfnBuilder mutation to direct assignment. |
| `spec.py` | Delete `configure()` method and the 5 redundant `_` attributes. |
| `text_model_loader.py` | Delete `SpecAttnConfig` import and `spec.configure(...)` call. |
| `source_model/qwen3_spec.py` | Replace `self._head_dim`/`self._rope_dim`/`self._permute_qk` with direct config reads. |
| `source_model/qwen3_5_spec.py` | Same as above. |
| `source_model/gpt_oss_spec.py` | Same as above. |
| `source_model/glm4_moe_lite_spec.py` | Same as above. |

## What does NOT change

- `ModelConfig`, `AttentionConfig` (System A), `LoraConfig`, `TurbomindModelConfig` in `config.py` -- user-facing configs stay as pydantic dataclasses.
- C++ structs in `core/module_config.h` -- unchanged.
- The factory function signatures -- same parameters as the current `from_model_config()` methods.
