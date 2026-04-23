# Deduplicate dtype-cast code across weight classes

**Date:** 2026-04-11
**Status:** Draft

## Problem

The "cast tensor to target dtype if both are dense float" pattern is copy-pasted across 3 weight classes:

| File | Cast targets | Local helper |
|---|---|---|
| `linear_weight.cc` | `weight` (dense path), `scales` (FP8 path) | `IsDenseFloatType` |
| `norm_weight.cc` | `weight` | `IsDenseFloatType` |
| `delta_net_weight.cc` | `A_log`, `dt_bias`, `conv1d` | `IsDenseFloatType` + `CastIfNeeded` |

Each copy defines its own `IsDenseFloatType` in an anonymous namespace and manually allocates + casts + moves.

## Design

### 1. Add `EnsureFloatDtype()` to `memory_utils.h/.cu`

```cpp
// memory_utils.h
void EnsureFloatDtype(Tensor& tensor, DataType target_dtype);
```

No stream parameter — the function uses `core::Context::stream()` internally since it allocates a temporary tensor and needs the context's allocator. Implementation lives in `memory_utils.cu` alongside the existing `invokeDtypeCast` kernel. Internally checks `IsDenseFloatType` (file-local) on both source and target, and if they differ, allocates a new tensor, calls `invokeDtypeCast`, and move-assigns back.

### 2. Simplify call sites

Each `prepare()` method replaces its inline cast block with a single `EnsureFloatDtype(tensor, target_dtype)` call:

- **`linear_weight.cc`**: 2 calls — `EnsureFloatDtype(weight, data_type)` for the dense path, `EnsureFloatDtype(scales, kFloat)` for the FP8 scale path
- **`norm_weight.cc`**: 1 call — `EnsureFloatDtype(weight, dtype_)`
- **`delta_net_weight.cc`**: 3 calls — `EnsureFloatDtype(A_log, data_type_)`, `EnsureFloatDtype(dt_bias, data_type_)`, `EnsureFloatDtype(conv1d, data_type_)`

### 3. Remove local helpers

Remove the anonymous-namespace `IsDenseFloatType` from `linear_weight.cc` and `norm_weight.cc`, and remove both `IsDenseFloatType` and `CastIfNeeded` from `delta_net_weight.cc`.

## File changes

| File | Change |
|---|---|
| `src/turbomind/utils/memory_utils.h` | Add `EnsureFloatDtype` declaration |
| `src/turbomind/utils/memory_utils.cu` | Add `EnsureFloatDtype` implementation |
| `src/turbomind/models/linear_weight.cc` | Replace inline casts with `EnsureFloatDtype`, remove local `IsDenseFloatType` |
| `src/turbomind/models/norm_weight.cc` | Replace inline cast with `EnsureFloatDtype`, remove local `IsDenseFloatType` |
| `src/turbomind/models/delta_net_weight.cc` | Replace inline casts with `EnsureFloatDtype`, remove local `IsDenseFloatType` + `CastIfNeeded` |

## What stays the same

- `invokeDtypeCast` kernel in `memory_utils.cu` — unchanged (called internally by `EnsureFloatDtype`)
- `IsDenseFloatType` stays as a file-local helper inside `memory_utils.cu` only, not exposed in `data_type.h`
- All Python-side changes in `load_context.py` — unchanged
- `bind.cpp` shared_ptr holder change — unchanged

## Dependencies

- `EnsureFloatDtype` implementation includes `src/turbomind/core/context.h` for `core::Context::stream()`
