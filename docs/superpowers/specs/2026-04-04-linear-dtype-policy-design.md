# Linear Dtype Policy: Design Spec

Date: 2026-04-04

## Problem

TurboMind's `LinearWeight` manages three dtype fields (`data_type`, `weight_type`, `input_type`) that are set at different lifecycle stages and have implicit constraints based on weight format and GPU SM version. The derivation rules are scattered across `do_allocate()`, `GetConverters()`, and `LlamaLinear.cu`, making it hard to reason about which dtype combinations are valid and how SM version affects them.

## Goal

1. **Document** the complete dtype policy matrix — all valid `(data_type, weight_format, SM)` combinations and their derived dtypes.
2. **Refactor** the code to separate immutable inputs from derived state, centralizing dtype derivation in one place.

## Design

### Concepts

- **`data_type`** — model-scope default compute dtype (FP16/BF16/FP32). Set at `configure()` time, never changes. This is the baseline dtype used wherever there's no strong reason to pick something else.
- **`weight_format`** — the checkpoint's weight storage format. Set at `do_allocate()` time. Can be a quantized format (FP8, FP4, UINT4, UINT8) or a dense float type.
- **Derived dtypes** — `input_dtype`, `output_dtype`, `scale_dtype`, and quantization descriptors — all computed from `(data_type, weight_format, group_size, SM)` by a single function.

### Dtype Policy Matrix

#### Dense (non-quantized) weights

| data_type | weight_format | SM | input_dtype | output_dtype | scale_dtype | Notes |
|---|---|---|---|---|---|---|
| FP16 | FP16 | any | FP16 | FP16 | — | Trivial, no conversion |
| BF16 | BF16 | any | BF16 | BF16 | — | Trivial, no conversion |
| FP16 | BF16 | any | FP16 | FP16 | — | Weights cast to FP16 at allocation |
| BF16 | FP16 | any | BF16 | BF16 | — | Weights cast to BF16 at allocation |
| FP16/BF16 | FP32 | any | data_type | data_type | — | Weights cast down at allocation |

The FP16↔BF16 cross-cast is handled in `alloc()` by the `IsDenseFloatType` check, not by the dtype derivation function.

#### Quantized — group-quantized (INT4/INT8)

| data_type | weight_format | SM | input_dtype | output_dtype | scale_dtype | weight_quant | input_quant |
|---|---|---|---|---|---|---|---|
| FP16/BF16 | UINT4 | any | data_type | data_type | data_type | QuantB(group) | — |
| FP16/BF16 | UINT8 | any | data_type | data_type | data_type | QuantB(group) | — |

No SM-dependent dtype behavior. `GetConverters()` picks different kernel layouts per SM version, but the dtypes are uniform.

#### Quantized — FP8

| data_type | weight_format | SM | input_dtype | output_dtype | scale_dtype | weight_quant | input_quant |
|---|---|---|---|---|---|---|---|
| FP16/BF16 | FP8_e4m3 | == 90 | FP8_e4m3 | data_type | FP32 | QuantB(128) | QuantK(128) |
| FP16/BF16 | FP8_e4m3 | != 90 | data_type | data_type | data_type | QuantB(128) | — |

SM == 90 enables native GMMA FP8: input is quantized to FP8, scales are FP32 block-scales. Non-SM90 falls back to dequantization path — input stays in `data_type`, `GetConverters()` handles layout conversion.

**This is the only SM-dependent dtype override.** All other formats have uniform dtypes across SM versions.

#### Quantized — FP4

| data_type | weight_format | SM | input_dtype | output_dtype | scale_dtype | weight_quant | input_quant |
|---|---|---|---|---|---|---|---|
| FP16/BF16 | FP4_e2m1 | any | data_type | data_type | UE8M0 (uint8) | QuantK(group) | — |

Scales are `uint8` (UE8M0 format). Special case: when `data_type == FP16`, scales are adjusted via `AdjustUe8m0ScaleForHalf()`. No SM-dependent dtype behavior.

### Code Changes

#### 1. New `LinearDtypes` struct and `ResolveDtypes()` function

Location: `src/turbomind/models/linear_weight.cc`

```cpp
struct LinearDtypes {
    DataType input_dtype;
    DataType output_dtype;
    DataType scale_dtype;
    QuantDesc input_quant;
    QuantDesc weight_quant;
};

LinearDtypes ResolveDtypes(DataType data_type, DataType weight_format, int group_size, int sm);
```

This function centralizes all dtype derivation logic. It replaces the scattered if-else currently in `do_allocate()` (lines 49-84 of `linear_weight.cc`).

`ResolveDtypes()` also performs validation:
- FP8 requires `group_size == 128`
- UINT4/UINT8 require compatible `group_size`
- Unsupported `(weight_format, SM)` combinations fail with a clear error message

#### 2. `LinearWeight` field restructuring

```cpp
// Before (3 mutable fields):
DataType data_type{};    // set in configure(), used as output too
DataType weight_type{};  // set in do_allocate()
DataType input_type{};   // set in do_allocate(), overridden for SM90

// After (inputs + derived):
// Inputs (immutable after their respective setters)
DataType data_type{};       // set in configure(), never changes
DataType weight_format{};   // set in do_allocate(), never changes after

// Derived (computed once in do_allocate via ResolveDtypes)
LinearDtypes resolved_;
// Accessors:
DataType input_dtype() const;   // was input_type
DataType output_dtype() const;  // shorthand for data_type in output contexts
```

Key changes:
- `data_type` name stays — it's the model-scope default.
- `weight_type` renamed to `weight_format` — clarifies it refers to the checkpoint's quantization format, not a runtime compute dtype.
- `input_type` replaced by `resolved_.input_dtype` via accessor — no longer a mutable field, it's derived once.
- Public quant fields (`weight_quant`, `input_quant`) move into `resolved_`.
- `output_dtype()` accessor makes the output role of `data_type` explicit at call sites. Always returns `data_type` — it's purely for readability.

#### 3. `do_allocate()` simplification

```cpp
void LinearWeight::do_allocate(DataType actual_weight_type, int actual_group_size) {
    weight_format = actual_weight_type;
    group_size    = actual_group_size;
    resolved_     = ResolveDtypes(data_type, actual_weight_type, actual_group_size, getSMVersion());

    // Tensor allocation uses resolved_ fields
    weight = Tensor({input_dim, output_dim}, actual_weight_type, kDEVICE);
    // scales/zeros allocated per resolved_.scale_dtype
    // ...
}
```

No more scattered SM-dependent if-else in this function.

#### 4. `LlamaLinear.cu` changes

Minimal accessor renames:
- `dense.input_type` → `dense.input_dtype()`
- `dense.data_type` in output allocation context → `dense.output_dtype()`

No structural changes.

#### 5. `alloc()` — no change

The FP16↔BF16 cross-cast logic in `alloc()` stays as-is. It's a tensor allocation concern, not a dtype derivation concern.

#### 6. `GetConverters()` in `convert_v3.cu` — no structural change

Callers pass dtype parameters from `resolved_` instead of from separate fields. The function signature stays the same.

### Files Touched

| File | Change |
|---|---|
| `src/turbomind/models/linear_weight.h` | Field restructuring, add `LinearDtypes`, accessors |
| `src/turbomind/models/linear_weight.cc` | `ResolveDtypes()` + simplified `do_allocate()` |
| `src/turbomind/models/llama/LlamaLinear.cu` | Accessor name updates |
| Other consumers of `LinearWeight` | Update `weight_type` → `weight_format`, `input_type` → `input_dtype()` |

### Summary of SM-Dependent Behavior

| Aspect | SM-dependent? | Where |
|---|---|---|
| Dtype derivation (input/output/scale) | Only for FP8 (SM == 90) | `ResolveDtypes()` |
| Kernel layout converters | Yes, all formats | `GetConverters()` |
| Weight format conversion (prepare) | Indirectly (via converters) | `prepare()` |
