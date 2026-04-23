# Universal Data Format Descriptor for TurboMind

## Problem

TurboMind's linear dtype policy (added recently via `LinearDtypes` + `ResolveDtypes`) covers dtype derivation for linear modules only. In general, quantized tensors carry quantization parameters (scales, zero-points) with their own shapes, dtypes, and block sizes. This information is needed across TurboMind — not just in linear layers, but also in KV cache, weight processing, and future components like vision encoders.

Currently, block sizes, scale dtypes, and layout information are scattered: `group_size` fields on `LinearWeight`, hardcoded `128` in `blocked_fp8.py`, shape logic inline in `do_allocate()`. There is no universal way to describe "what format is this tensor in."

Additionally, the model loading pipeline (`lmdeploy/turbomind/deploy/`) needs to check whether operations like `permute`, `split`, or `interleave` are valid on quantized tensors — but has no structured way to reason about block boundaries.

## Scope

This design covers:

1. A universal `DataFormat` descriptor (C++ struct)
2. Application to the model loading pipeline (Python, `lmdeploy/turbomind/deploy/`)
3. Operation validity checks using block size info during weight processing
4. Application to the linear module (C++ `LinearWeight` and Python loading pipeline)

Out of scope: `lmdeploy/pytorch/` (PyTorch engine), `lmdeploy/lite/` (quantization toolkit).

## Design

### 1. Core `DataFormat` struct (C++)

**File:** `src/turbomind/core/data_format.h`

A flat, pure-data struct describing the storage format of a quantized tensor. No behavioral methods — validity checkers live in Python.

```cpp
struct QuantParamDesc {
    DataType dtype{};       // kNull means "not present"
    bool     transposed{};  // stored transposed w.r.t. data tensor
    bool     present() const { return dtype != kNull; }
};

struct DataFormat {
    DataType              dtype{};        // element type of the data tensor
    std::vector<int>      block_sizes;    // per-dimension block sizes (1 = no quantization)
    QuantParamDesc        scales{};
    QuantParamDesc        zeros{};

    // Minimal convenience
    bool is_quantized() const;
    int  rank() const;
};
```

**`block_sizes`** has one entry per tensor dimension:
- AWQ/GPTQ weight `(out, in)` with group_size=128: `{1, 128}`
- Blocked FP8 weight `(out, in)`: `{128, 128}`
- KV cache key `(seq, head_dim)`: `{k_group_dim, 1}`
- Dense weight: `{1, 1}` (or empty)

**`QuantParamDesc`**:
- `dtype == kNull` means the parameter is absent
- `transposed` indicates whether the parameter is stored transposed relative to the data tensor (e.g. data `(batch, feat)`, scales `(feat/group, batch)`)

### 2. Factory functions (C++)

**File:** `src/turbomind/core/data_format.cc`

Factory functions create `DataFormat` instances for specific scenarios:

```cpp
DataFormat MakeLinearWeightFormat(DataType data_type,
                                   DataType weight_format,
                                   int group_size);
```

`MakeLinearWeightFormat` sets:
- `dtype` from the weight storage format
- `block_sizes` based on quantization scheme
- `scales` / `zeros` with appropriate dtypes and transposed flags

Scale/zeros dtype derivation by format:

| weight_format | block_sizes | scales.dtype | zeros.dtype |
|---|---|---|---|
| Dense (FP16/BF16/FP32) | `{1, 1}` | kNull (absent) | kNull (absent) |
| kFloat8_e4m3 | `{128, 128}` | kFloat | kNull |
| kFloat4_e2m1 | `{1, group_size}` | kUint8 | kNull |
| kUint4 / kUint8 | `{1, group_size}` | data_type | data_type |

### 3. `LinearPolicy` — compute dtype derivation (C++)

**File:** `src/turbomind/models/linear_weight.h` (or new file)

`DataFormat` describes storage. Compute dtypes and GEMM dispatch parameters depend on both the format AND hardware capabilities (SM version). This is a separate concern.

```cpp
struct LinearPolicy {
    DataType           input_dtype{};
    DataType           output_dtype{};
    gemm::QuantDesc    input_quant{};
    gemm::QuantDesc    weight_quant{};
};

LinearPolicy ResolveLinearPolicy(const DataFormat& format, DataType data_type, int sm);
```

This replaces the current `ResolveDtypes()` function. It takes a `DataFormat` (from `MakeLinearWeightFormat`) plus the model's compute dtype and SM version, and derives:
- `input_dtype` / `output_dtype` — compute dtypes for execution layers
- `input_quant` / `weight_quant` — GEMM quantization descriptors for kernel dispatch

Hardware-dependent logic (e.g. "on SM90, use native FP8 inputs") lives here, not in `DataFormat`.

### 4. C++ `LinearWeight` migration

**File:** `src/turbomind/models/linear_weight.h`, `linear_weight.cc`

Changes:
- `LinearDtypes resolved_` → `DataFormat format_` + `LinearPolicy policy_`
- `ResolveDtypes()` call → `MakeLinearWeightFormat()` + `ResolveLinearPolicy()`
- `input_dtype()` / `output_dtype()` → read from `policy_`
- Scale/zeros tensor shapes → computed from `format_` fields
- `group_size` stays as a convenience accessor: `format_.block_sizes[1]`
- `LinearDtypes` struct removed entirely

### 5. pybind11 exposure

**File:** `src/turbomind/python/bind.cpp`

Expose `DataFormat`, `QuantParamDesc`, and all their fields to Python:

```cpp
py::class_<QuantParamDesc>(m, "QuantParamDesc")
    .def_readonly("dtype", &QuantParamDesc::dtype)
    .def_readonly("transposed", &QuantParamDesc::transposed)
    .def("present", &QuantParamDesc::present);

py::class_<DataFormat>(m, "DataFormat")
    .def_readonly("dtype", &DataFormat::dtype)
    .def_readonly("block_sizes", &DataFormat::block_sizes)
    .def_readonly("scales", &DataFormat::scales)
    .def_readonly("zeros", &DataFormat::zeros)
    .def("is_quantized", &DataFormat::is_quantized)
    .def("rank", &DataFormat::rank);
```

### 6. Python validity checkers

**File:** `lmdeploy/turbomind/deploy/format_ops.py` (new)

Pure Python functions that read `DataFormat` fields (exposed via pybind11) and check operation validity:

```python
from _turbomind import DataFormat

def can_permute(fmt: DataFormat, dim0: int, dim1: int) -> bool:
    """Can we permute dim0 <-> dim1 without breaking block structure?
    False if either dimension has block_size > 1."""
    if fmt.block_sizes[dim0] > 1 or fmt.block_sizes[dim1] > 1:
        return False
    return True

def can_split(fmt: DataFormat, dim: int) -> bool:
    """Can we split along dim without breaking block structure?
    Requires block_size == 1 along that dimension."""
    return fmt.block_sizes[dim] == 1

def param_shape(fmt: DataFormat, data_shape: list[int], param: QuantParamDesc) -> list[int]:
    """Compute quant param shape given data shape, block sizes, and transposed flag."""
    shape = []
    for i, s in enumerate(data_shape):
        bs = fmt.block_sizes[i] if i < len(fmt.block_sizes) else 1
        shape.append((s + bs - 1) // bs if bs > 1 else s)
    if param.transposed:
        shape.reverse()
    return shape
```

### 7. Python loading pipeline migration

**File:** `lmdeploy/turbomind/deploy/linear.py`, `kind_map.py`

1. `Linear` dataclass gains a `data_format: DataFormat | None` field
2. `WeightFormat` in `kind_map.py` produces a `DataFormat` from its `block_in`/`block_out` and weight dtype
3. `preprocess_linear()` uses `param_shape()` instead of hardcoding `[K/gs, N/gs]`
4. Weight transformation operations (`permute_out_dim`, `interleave_linears`, etc.) use validity checkers as predicates to select the correct code path:
   ```python
   if can_permute(linear.data_format, dim0, dim1):
       # Fast path: permute all component tensors uniformly
       ...
   else:
       # Careful path: permute each component individually,
       # mapping dims through reduced scale dimensions
       ...
   ```
   The checkers are decision predicates, not exception guards. When an operation is not directly safe on the quantized tensor bundle, the caller adapts — e.g. applies the operation to each component tensor with correctly mapped dimensions, or dequantizes first.
5. `WeightFormat` retains its current role for checkpoint-specific concerns (suffix maps, normalizers, packers). `DataFormat` handles tensor format description.

## Migration summary

| Current | New |
|---------|-----|
| `LinearDtypes` struct | `DataFormat` (storage) + `LinearPolicy` (compute) |
| `ResolveDtypes(data_type, weight_format, group_size, sm)` | `MakeLinearWeightFormat(...)` + `ResolveLinearPolicy(format, data_type, sm)` |
| Hardcoded scale/zeros shapes in `do_allocate()` | Derived from `DataFormat` fields |
| `WeightFormat.block_in` / `block_out` | `DataFormat.block_sizes` |
| No operation validity checks | `can_permute()`, `can_split()`, `param_shape()` in Python |
