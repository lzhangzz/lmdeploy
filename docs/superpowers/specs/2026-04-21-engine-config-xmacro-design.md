# Engine Config X-Macro Migration

**Date:** 2026-04-21
**Status:** Draft

## Problem

Engine config currently flows through a YAML serialization/deserialization bridge:

```
Python dataclass -> dict -> YAML string -> pybind string -> C++ YAML::Load -> field-by-field extraction
```

This is ~50 lines of boilerplate YAML parsing in `turbomind.cc` (lines 183-240) that is error-prone, type-unsafe, and requires the `yaml-cpp` dependency solely for this purpose. The rest of the config system (module configs) already uses x-macros for type-safe, auto-reflected structs.

## Solution

Replace the YAML transport with an x-macro `EngineConfig` struct, passed directly through pybind. Same pattern as `RopeConfig` — a standalone struct with `bind_struct` auto-binding.

## Design

### 1. EngineConfig struct

New header `src/turbomind/engine/engine_config.h`:

```cpp
struct EngineConfig {
    #define ENGINE_FIELDS(X)                          \
        X(DataType,       data_type)                  \
        X(int,            cache_block_seq_len, 0)     \
        X(int,            quant_policy, 0)            \
        X(int,            tune_layer_num, 1)          \
        X(int,            max_batch_size, 0)          \
        X(int,            max_prefill_token_num, 0)   \
        X(int,            max_context_token_num, 0)   \
        X(int,            session_len, 0)             \
        X(float,          cache_max_block_count, 0)   \
        X(int,            cache_chunk_size, 0)        \
        X(bool,           enable_prefix_caching, false)\
        X(bool,           enable_metrics, false)      \
        X(int,            num_tokens_per_iter, 0)     \
        X(int,            max_prefill_iters, 1)       \
        X(int,            async_, 0)                  \
        X(int,            outer_dp_size)              \
        X(int,            attn_dp_size)               \
        X(int,            attn_tp_size)               \
        X(int,            attn_cp_size)               \
        X(int,            mlp_tp_size)                \
        X(std::vector<int>, devices)                  \
        X(int,            nnodes)                     \
        X(int,            node_rank)                  \
        X(std::string,    communicator)

    ENGINE_FIELDS(TM_MEMBER)
    TM_FOR_EACH(EngineConfig, ENGINE_FIELDS)
    #undef ENGINE_FIELDS
};
```

Key decisions:
- `data_type` is `DataType` enum (not string) — eliminates `data_type_from_string()` call
- `devices` is `std::vector<int>` — pybind natively converts Python list <-> std::vector, no x-macro changes needed
- Field names follow C++ conventions (e.g., `cache_max_block_count`, not Python's `cache_max_entry_count`)
- Python-only fields (`model_format`, `tp`, `dp`, `cp`, `hf_overrides`, etc.) stay in the Python `TurbomindEngineConfig` dataclass

### 2. Python binding

Add to `src/turbomind/python/bind.cpp`:

```cpp
bind_struct<turbomind::EngineConfig>(m, "EngineConfig");
```

This auto-exposes all 24 fields as Python read/write attributes.

### 3. TurboMind.create interface change

**pybind binding** (`bind.cpp`) changes from:
```cpp
// Before: (string model_dir, string config)
.def_static("create", [](string model_dir, string config) -> shared_ptr<TurboMind> { ... })
```
to:
```cpp
// After: (string model_dir, EngineConfig config)
.def_static("create", [](string model_dir, EngineConfig config) -> shared_ptr<TurboMind> { ... })
```

**C++ Impl constructor** (`turbomind.cc`) changes from:
```cpp
TurboMind::Impl::Impl(string model_dir, string config, FFICtxFactory ffi_ctx_factory)
```
to:
```cpp
TurboMind::Impl::Impl(string model_dir, EngineConfig config, FFICtxFactory ffi_ctx_factory)
```

The YAML parsing block (lines 183-240) collapses to direct field access:

```cpp
data_type_ = config.data_type;
TM_CHECK(data_type_ == kBfloat16 || data_type_ == kHalf);

engine_param_.cache_block_seq_len = config.cache_block_seq_len;
engine_param_.quant_policy        = config.quant_policy;
// ... direct assignments ...
engine_param_.devices             = std::move(config.devices);
communicator_type_                = std::move(config.communicator);
```

Derivation logic stays in C++:
- `max_forward_token_num = max_prefill_token_num + max_batch_size`
- `phases_ = async_ ? 2 : 1`
- `HandleMissingParams()` stays as-is

### 4. Python-side mapping

In `lmdeploy/turbomind/turbomind.py`, `_from_hf` changes from:

```python
config_dict = {'engine_config': asdict(engine_config)}
_tm.TurboMind.create(model_dir='', config=yaml.safe_dump(config_dict))
```

to:

```python
# Map Python dataclass fields to EngineConfig (C++ naming conventions)
dtype_map = {'float16': _tm.DataType.TYPE_FP16, 'bfloat16': _tm.DataType.TYPE_BF16}

ec = _tm.EngineConfig()
ec.data_type = dtype_map[engine_config.dtype]  # string -> DataType enum
ec.cache_block_seq_len = engine_config.cache_block_seq_len
ec.quant_policy = engine_config.quant_policy
ec.tune_layer_num = 1  # always default; Python spec sets this separately
ec.max_batch_size = engine_config.max_batch_size
ec.max_prefill_token_num = engine_config.max_prefill_token_num
ec.max_context_token_num = 0  # always 0; C++ HandleMissingParams() resolves it
ec.session_len = engine_config.session_len
ec.cache_max_block_count = engine_config.cache_max_entry_count  # name mapping
ec.cache_chunk_size = engine_config.cache_chunk_size
ec.enable_prefix_caching = engine_config.enable_prefix_caching
ec.enable_metrics = engine_config.enable_metrics
ec.num_tokens_per_iter = engine_config.num_tokens_per_iter
ec.max_prefill_iters = engine_config.max_prefill_iters
ec.async_ = engine_config.async_
ec.outer_dp_size = engine_config.outer_dp_size
ec.attn_dp_size = engine_config.attn_dp_size
ec.attn_tp_size = engine_config.attn_tp_size
ec.attn_cp_size = engine_config.attn_cp_size
ec.mlp_tp_size = engine_config.mlp_tp_size
ec.devices = engine_config.devices
ec.nnodes = engine_config.nnodes
ec.node_rank = engine_config.node_rank
ec.communicator = engine_config.communicator
_tm.TurboMind.create(model_dir='', engine_config=ec)
```

Notes on the mapping:
- `dtype` is resolved from string to `DataType` enum: `_tm.DataType.TYPE_BF16` or `_tm.DataType.TYPE_FP16`
- `tune_layer_num` is always `1` from the Python side (the spec object sets it separately, it never flows through engine_config)
- `max_context_token_num` is always `0` from Python — C++ `HandleMissingParams()` defaults it to `session_len`
- `cache_max_entry_count` (Python) maps to `cache_max_block_count` (C++) — the only name mismatch

A small helper function (e.g., `_engine_config_to_tm()`) in `turbomind.py` keeps `_from_hf` clean.

### 5. What gets deleted

- `yaml.safe_dump()` call in Python
- `YAML::Load()` + 50 lines of `.as<T>()` extraction in C++
- `#include <yaml-cpp/yaml.h>` from `turbomind.cc`
- `yaml-cpp::yaml-cpp` from `src/turbomind/CMakeLists.txt` (if no other consumer)
- `data_type_from_string()` call (dtype is now a DataType enum)

### 6. No x-macro changes needed

`TM_MEMBER` already handles arbitrary types — `X(std::vector<int>, devices)` expands to `std::vector<int> devices{};` which is valid C++. Pybind natively converts Python `list[int]` to `std::vector<int>`. The x-macro infrastructure requires zero modifications.

## Scope

- **In scope:** Replace YAML transport layer for engine config
- **Out of scope:** Merging `EngineParam` into `EngineConfig`, removing `TurbomindEngineConfig` Python dataclass, changing field names

## Files changed

| File | Change |
|------|--------|
| `src/turbomind/engine/engine_config.h` | New file — EngineConfig x-macro struct |
| `src/turbomind/turbomind.cc` | Replace YAML parsing with direct EngineConfig access |
| `src/turbomind/turbomind.h` | Update constructor signature |
| `src/turbomind/python/bind.cpp` | Add `bind_struct<EngineConfig>`, update `TurboMind.create` signature |
| `src/turbomind/CMakeLists.txt` | Add new header, potentially remove yaml-cpp dependency |
| `lmdeploy/turbomind/turbomind.py` | Replace YAML serialization with EngineConfig construction |

## Verification Notes

### yaml-cpp is engine-config-only

Confirmed: `yaml-cpp` is used in exactly one file (`turbomind.cc`) for exactly one purpose (parsing the engine config YAML string). The `#include <yaml-cpp/yaml.h>` and `target_link_libraries(... yaml-cpp::yaml-cpp)` can be fully removed after migration.

### DataType Python binding names

The pybind binding in `bind.cpp` (lines 395-415) exposes DataType as `_tm.DataType.TYPE_BF16`, `_tm.DataType.TYPE_FP16`, etc. — not the C++ `kBfloat16`/`kHalf` names. The Python-side mapping must use these names.

### Dead fields in EngineParam (future cleanup)

These `EngineParam` fields are set but never read by any consumer:
- `step_length` — never set or read anywhere (completely dead)
- `num_tokens_per_iter` — set from YAML, never read
- `max_prefill_iters` — set from YAML, never read
- `outer_dp_size` — set from YAML, never read
- `outer_dp_rank` — computed in `CreateContext`, never read

These are out of scope for this migration but worth noting for a follow-up cleanup.

### Fields with constant values from Python

Two EngineConfig fields always receive the same value from the Python side:
- `tune_layer_num` — always `1` (Python spec sets it on the spec object, not in engine_config)
- `max_context_token_num` — always `0` (C++ `HandleMissingParams()` resolves it to `session_len`)

They are included in EngineConfig for completeness, since the C++ side does read them from the config.
