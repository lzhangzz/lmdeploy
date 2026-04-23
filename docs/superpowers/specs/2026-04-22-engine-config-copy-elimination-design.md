# Engine Config Copy Elimination — EngineParam Inherits EngineConfig

**Date:** 2026-04-22
**Status:** Draft

## Problem

`TurboMind::Impl`'s constructor manually copies ~23 fields from `EngineConfig` to `EngineParam` one-by-one (turbomind.cc:155-198). This boilerplate exists because `EngineConfig` (the pybind-facing struct) and `EngineParam` (the internal runtime struct) were defined independently, despite sharing most fields.

The duplication is an architectural debt — two structs with overlapping fields require manual synchronization and copy logic.

## Solution

Make `EngineParam` inherit from `EngineConfig`. This establishes a single source of truth for config fields while preserving `EngineParam`'s runtime-derived rank fields and derived values.

## Design

### 1. `EngineParam` inherits `EngineConfig`

In `src/turbomind/models/llama/llama_params.h`:

```cpp
struct EngineParam : EngineConfig {
    // Runtime-derived fields (set in CreateContext)
    int outer_dp_rank = 0;
    int attn_dp_rank = 0;
    int attn_tp_rank = 0;
    int attn_cp_rank = 0;
    int mlp_tp_rank = 0;

    // Derived field (set in Impl ctor)
    int max_forward_token_num = 0;

    // step_length REMOVED — never read anywhere
};
```

All 24 fields from `EngineConfig` become accessible on `EngineParam` via inheritance — no consumer changes needed.

### 2. Copy elimination in `turbomind.cc`

The 44-line block of field-by-field assignments in `Impl::Impl` is replaced by:

```cpp
static_cast<EngineConfig&>(engine_param_) = config;
```

Everything else stays:
- `max_forward_token_num` derivation (TP/CP size based)
- `HandleMissingParams()` unchanged
- `phases_ = engine_param_.async_ ? 2 : 1` (reads from inherited field)

### 3. Consumers — zero changes

9 files that use `EngineParam` are untouched:
- `src/turbomind/engine/engine.cc`
- `src/turbomind/models/language_model.cc`
- `src/turbomind/models/model_weight.cc`
- `src/turbomind/models/llama/unified_decoder.cc`
- `src/turbomind/models/llama/unified_attention_layer.cc`
- `src/turbomind/models/llama/GatedDeltaNetLayer.cc`
- `src/turbomind/models/llama/moe_ffn_layer.cc`
- `src/turbomind/models/input_processor.cc`

All existing access patterns (`param.devices`, `param.attn_tp_size`, etc.) work identically via inheritance.

### 4. pybind — no change

Only `EngineConfig` is exposed to Python (via `bind_struct`). `EngineParam` stays C++-internal.

### 5. What gets deleted

- 44 lines of field-to-field assignments in `turbomind.cc`
- `step_length` field from `EngineParam` (never read anywhere)

## Files changed

| File | Change |
|------|--------|
| `src/turbomind/models/llama/llama_params.h` | `EngineParam` inherits `EngineConfig`, remove duplicate fields |
| `src/turbomind/turbomind.cc` | Replace field-by-field copy with single cast-assignment |

## Scope

- **In scope:** Eliminate field-to-field copy by making `EngineParam` inherit `EngineConfig`; remove dead `step_length` field
- **Out of scope:** Migrating constant-default fields, changing pybind exposure
