# Universal Data Format Descriptor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `LinearDtypes` with a universal `DataFormat` descriptor for quantized tensors, plus a separate `LinearPolicy` resolver for compute dtypes.

**Architecture:** Flat C++ `DataFormat` struct in `core/` for storage format description, `LinearPolicy` struct for compute dtype derivation, pybind11 exposure, and Python validity checkers for the loading pipeline. `LinearDtypes` is removed entirely.

**Tech Stack:** C++17, pybind11, Python 3, Catch2 for C++ tests.

**Design spec:** `docs/superpowers/specs/2026-04-06-universal-data-format-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `src/turbomind/core/data_format.h` | `QuantParamDesc`, `DataFormat` structs |
| Create | `src/turbomind/core/data_format.cc` | `MakeLinearWeightFormat()`, `DataFormat` helpers |
| Create | `src/turbomind/core/test_data_format.cc` | Catch2 tests for `DataFormat` and factory |
| Modify | `src/turbomind/models/linear_weight.h` | Replace `LinearDtypes` with `DataFormat` + `LinearPolicy` |
| Modify | `src/turbomind/models/linear_weight.cc` | Replace `ResolveDtypes()` with new functions, update `do_allocate()` |
| Modify | `src/turbomind/models/llama/LlamaLinear.cu` | Update `resolved_` field access |
| Modify | `src/turbomind/models/moe_weight.cc` | Update `input_dtype()` call (accessor stays same) |
| Modify | `src/turbomind/python/bind.cpp` | Expose `DataFormat` and `QuantParamDesc` |
| Create | `lmdeploy/turbomind/deploy/format_ops.py` | Python validity checkers |
| Modify | `lmdeploy/turbomind/deploy/linear.py` | Add `data_format` field to `Linear`, use `param_shape` |
| Modify | `lmdeploy/turbomind/deploy/kind_map.py` | Add `DataFormat` construction to `WeightFormat` |

---

### Task 1: Create `DataFormat` and `QuantParamDesc` structs (C++ header)

**Files:**
- Create: `src/turbomind/core/data_format.h`

- [ ] **Step 1: Write the header file**

```cpp
// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/data_type.h"
#include <vector>

namespace turbomind {

/// Descriptor for a single quantization parameter (scales or zeros).
struct QuantParamDesc {
    DataType dtype{};       // kNull means "not present"
    bool     transposed{};  // stored transposed w.r.t. data tensor

    bool present() const noexcept { return dtype != kNull; }
};

/// Universal descriptor for the storage format of a (possibly quantized) tensor.
struct DataFormat {
    DataType         dtype{};      // element type of the data tensor
    std::vector<int> block_sizes;  // per-dimension block sizes (1 = no quantization)
    QuantParamDesc   scales{};
    QuantParamDesc   zeros{};

    /// True if any quantization parameter is present or any block_size > 1.
    bool is_quantized() const noexcept;

    /// Number of dimensions described by this format.
    int rank() const noexcept { return static_cast<int>(block_sizes.size()); }
};

}  // namespace turbomind
```

- [ ] **Step 2: Commit**

```bash
git add src/turbomind/core/data_format.h
git commit -m "feat(core): add DataFormat and QuantParamDesc structs"
```

---

### Task 2: Implement `DataFormat` helpers and `MakeLinearWeightFormat` (C++)

**Files:**
- Create: `src/turbomind/core/data_format.cc`

- [ ] **Step 1: Write the implementation**

```cpp
// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/core/data_format.h"
#include "src/turbomind/core/check.h"

namespace turbomind {

bool DataFormat::is_quantized() const noexcept
{
    if (scales.present() || zeros.present()) {
        return true;
    }
    for (int bs : block_sizes) {
        if (bs > 1) {
            return true;
        }
    }
    return false;
}

static bool IsDenseFloatType(DataType t)
{
    return t == kFloat || t == kHalf || t == kBfloat16;
}

DataFormat MakeLinearWeightFormat(DataType data_type, DataType weight_format, int group_size)
{
    DataFormat fmt;
    fmt.dtype = weight_format;

    if (IsDenseFloatType(weight_format)) {
        // Dense FP16/BF16/FP32 — no quantization
        fmt.block_sizes = {1, 1};
        return fmt;
    }

    if (weight_format == kFloat8_e4m3) {
        TM_CHECK_EQ(group_size, 128)
            << "FP8 weight format requires group_size=128, got " << group_size;
        fmt.block_sizes  = {128, 128};
        fmt.scales.dtype = kFloat;
        return fmt;
    }

    if (weight_format == kFloat4_e2m1) {
        TM_CHECK(group_size > 0)
            << "FP4 weight format requires group_size > 0, got " << group_size;
        fmt.block_sizes  = {1, group_size};
        fmt.scales.dtype = kUint8;
        return fmt;
    }

    const bool is_qweight = weight_format == kUint4 || weight_format == kUint8;
    if (is_qweight) {
        TM_CHECK(group_size > 0 && group_size <= 256)
            << "Invalid group_size for quantized weight: " << group_size;
        fmt.block_sizes  = {1, group_size};
        fmt.scales.dtype = data_type;
        fmt.zeros.dtype  = data_type;
        return fmt;
    }

    TM_CHECK(0) << "Unsupported weight format: " << to_string(weight_format);
    return fmt;
}

}  // namespace turbomind
```

- [ ] **Step 2: Build to verify compilation**

Run: `cd build && ninja` (or build just the target that includes this file)
Expected: Compiles without errors.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/core/data_format.cc
git commit -m "feat(core): implement DataFormat helpers and MakeLinearWeightFormat"
```

---

### Task 3: Write C++ unit tests for `DataFormat`

**Files:**
- Create: `src/turbomind/core/test_data_format.cc`

- [ ] **Step 1: Write the tests**

```cpp
#include "src/turbomind/core/data_format.h"
#include "catch2/catch_test_macros.hpp"

using namespace turbomind;

TEST_CASE("DataFormat default is not quantized", "[data_format]")
{
    DataFormat fmt;
    REQUIRE(!fmt.is_quantized());
    REQUIRE(fmt.rank() == 0);
    REQUIRE(!fmt.scales.present());
    REQUIRE(!fmt.zeros.present());
}

TEST_CASE("DataFormat dense is not quantized", "[data_format]")
{
    DataFormat fmt = MakeLinearWeightFormat(kHalf, kHalf, 0);
    REQUIRE(!fmt.is_quantized());
    REQUIRE(fmt.rank() == 2);
    REQUIRE(fmt.block_sizes == std::vector<int>{1, 1});
    REQUIRE(!fmt.scales.present());
    REQUIRE(!fmt.zeros.present());
}

TEST_CASE("DataFormat FP8 blocked", "[data_format]")
{
    DataFormat fmt = MakeLinearWeightFormat(kHalf, kFloat8_e4m3, 128);
    REQUIRE(fmt.is_quantized());
    REQUIRE(fmt.dtype == kFloat8_e4m3);
    REQUIRE(fmt.block_sizes == std::vector<int>{128, 128});
    REQUIRE(fmt.scales.present());
    REQUIRE(fmt.scales.dtype == kFloat);
    REQUIRE(!fmt.zeros.present());
}

TEST_CASE("DataFormat FP4", "[data_format]")
{
    DataFormat fmt = MakeLinearWeightFormat(kHalf, kFloat4_e2m1, 128);
    REQUIRE(fmt.is_quantized());
    REQUIRE(fmt.dtype == kFloat4_e2m1);
    REQUIRE(fmt.block_sizes == std::vector<int>{1, 128});
    REQUIRE(fmt.scales.present());
    REQUIRE(fmt.scales.dtype == kUint8);
    REQUIRE(!fmt.zeros.present());
}

TEST_CASE("DataFormat AWQ uint4", "[data_format]")
{
    DataFormat fmt = MakeLinearWeightFormat(kHalf, kUint4, 128);
    REQUIRE(fmt.is_quantized());
    REQUIRE(fmt.dtype == kUint4);
    REQUIRE(fmt.block_sizes == std::vector<int>{1, 128});
    REQUIRE(fmt.scales.present());
    REQUIRE(fmt.scales.dtype == kHalf);
    REQUIRE(fmt.zeros.present());
    REQUIRE(fmt.zeros.dtype == kHalf);
}

TEST_CASE("DataFormat uint8 quantized", "[data_format]")
{
    DataFormat fmt = MakeLinearWeightFormat(kBfloat16, kUint8, 64);
    REQUIRE(fmt.is_quantized());
    REQUIRE(fmt.block_sizes == std::vector<int>{1, 64});
    REQUIRE(fmt.scales.dtype == kBfloat16);
    REQUIRE(fmt.zeros.dtype == kBfloat16);
}

TEST_CASE("DataFormat dense BF16", "[data_format]")
{
    DataFormat fmt = MakeLinearWeightFormat(kBfloat16, kBfloat16, 0);
    REQUIRE(!fmt.is_quantized());
    REQUIRE(fmt.dtype == kBfloat16);
}
```

- [ ] **Step 2: Add test target to CMake and build**

Add to the appropriate `CMakeLists.txt` (likely `src/turbomind/core/CMakeLists.txt` or the catch2 test list):
```cmake
target_sources(test_core PRIVATE
    src/turbomind/core/test_data_format.cc
)
```
Run: `cd build && ninja test_core && ./test_core "[data_format]"`
Expected: All 7 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/core/test_data_format.cc
git commit -m "test(core): add DataFormat unit tests"
```

---

### Task 4: Add `LinearPolicy` and replace `LinearDtypes` in the header

**Files:**
- Modify: `src/turbomind/models/linear_weight.h` (lines 14-21: `LinearDtypes` struct, line 23: `ResolveDtypes` decl, line 63: `resolved_` field, lines 65-66: accessors)

- [ ] **Step 1: Replace `LinearDtypes` with `LinearPolicy` and update `LinearWeight`**

In `linear_weight.h`, replace lines 14-21:

```cpp
// REMOVE this:
struct LinearDtypes {
    DataType input_dtype{};
    DataType output_dtype{};
    DataType scale_dtype{};
    gemm::QuantDesc input_quant{};
    gemm::QuantDesc weight_quant{};
};

LinearDtypes ResolveDtypes(DataType data_type, DataType weight_format, int group_size, int sm);
```

With:

```cpp
/// Compute-time dtype policy derived from DataFormat + hardware.
struct LinearPolicy {
    DataType        input_dtype{};
    DataType        output_dtype{};
    gemm::QuantDesc input_quant{};
    gemm::QuantDesc weight_quant{};
};

/// Derive compute dtypes and GEMM quant descriptors from storage format + hardware.
LinearPolicy ResolveLinearPolicy(const DataFormat& format, DataType data_type, int sm);
```

In `LinearWeight` class, replace line 63:

```cpp
// REMOVE:
    LinearDtypes resolved_{};
// ADD:
    DataFormat    format_{};
    LinearPolicy  policy_{};
```

Replace accessors at lines 65-66:

```cpp
// REMOVE:
    DataType input_dtype() const  { return resolved_.input_dtype; }
    DataType output_dtype() const { return resolved_.output_dtype; }
// ADD:
    DataType input_dtype() const  { return policy_.input_dtype; }
    DataType output_dtype() const { return policy_.output_dtype; }
```

Add the include for `data_format.h` at the top:

```cpp
#include "src/turbomind/core/data_format.h"
```

- [ ] **Step 2: Commit**

```bash
git add src/turbomind/models/linear_weight.h
git commit -m "refactor(linear): replace LinearDtypes with DataFormat + LinearPolicy"
```

---

### Task 5: Implement `ResolveLinearPolicy` and update `linear_weight.cc`

**Files:**
- Modify: `src/turbomind/models/linear_weight.cc` (lines 17-63: `IsDenseFloatType` + `ResolveDtypes`, lines 81-123: `do_allocate`, line 220: `input_dtype()` call)

- [ ] **Step 1: Replace `ResolveDtypes` with `ResolveLinearPolicy`**

Replace lines 17-63 in `linear_weight.cc`:

```cpp
// REMOVE IsDenseFloatType (moved to data_format.cc) and ResolveDtypes

// ADD:
LinearPolicy ResolveLinearPolicy(const DataFormat& format, DataType data_type, int sm)
{
    LinearPolicy p;
    p.output_dtype = data_type;
    p.input_dtype  = data_type;

    // Dense float — no quantization
    if (!format.is_quantized()) {
        return p;
    }

    // FP8 blocked: on SM90 use native FP8 input
    if (format.dtype == kFloat8_e4m3) {
        int gs = format.block_sizes[1];  // group_size (128 for FP8)
        p.weight_quant = gemm::QuantDesc{gemm::QuantType::kB, gs};
        if (sm == 90) {
            p.input_dtype  = kFloat8_e4m3;
            p.input_quant  = gemm::QuantDesc{gemm::QuantType::kK, gs};
        }
        return p;
    }

    // FP4
    if (format.dtype == kFloat4_e2m1) {
        int gs = format.block_sizes[1];
        p.weight_quant = gemm::QuantDesc{gemm::QuantType::kK, gs};
        return p;
    }

    // UINT4 / UINT8 (AWQ, GPTQ, compressed-tensors)
    if (format.dtype == kUint4 || format.dtype == kUint8) {
        int gs = format.block_sizes[1];
        p.weight_quant = gemm::QuantDesc{gemm::QuantType::kK, gs};
        return p;
    }

    TM_CHECK(0) << "Unsupported weight format for policy: " << to_string(format.dtype);
    return p;
}
```

- [ ] **Step 2: Update `do_allocate` to use `DataFormat`**

Replace the body of `do_allocate` (lines 81-123):

```cpp
void LinearWeight::do_allocate(DataType actual_weight_type, int actual_group_size)
{
    weight_format = actual_weight_type;
    group_size    = actual_group_size;
    format_       = MakeLinearWeightFormat(data_type, actual_weight_type, actual_group_size);
    policy_       = ResolveLinearPolicy(format_, data_type, getSMVersion());

    weight = Tensor({input_dim, output_dim}, actual_weight_type, kDEVICE);
    add_param("weight", weight);

    if (has_bias_) {
        bias = Tensor{{output_dim}, data_type, kDEVICE};
        add_param("bias", bias);
    }

    scales = {};
    zeros  = {};

    if (format_.scales.present()) {
        if (actual_weight_type == kFloat8_e4m3) {
            scales = Tensor{{cdiv(input_dim, actual_group_size), cdiv(output_dim, actual_group_size)},
                            format_.scales.dtype, kDEVICE};
        }
        else if (actual_weight_type == kFloat4_e2m1) {
            scales = Tensor{{cdiv(input_dim, actual_group_size), output_dim},
                            format_.scales.dtype, kDEVICE};
        }
        else {
            // uint4 / uint8
            TM_CHECK(input_dim % actual_group_size == 0) << input_dim << " " << actual_group_size;
            scales = Tensor{{input_dim / actual_group_size, output_dim},
                            format_.scales.dtype, kDEVICE};
        }
        add_param("scales", scales);
    }

    if (format_.zeros.present()) {
        TM_CHECK(input_dim % actual_group_size == 0) << input_dim << " " << actual_group_size;
        zeros = Tensor{{input_dim / actual_group_size, output_dim},
                        format_.zeros.dtype, kDEVICE};
        add_param("zeros", zeros);
    }

    k_desc = {};
    q_desc = {};

    k_desc.type  = weight.dtype();
    k_desc.order = gemm::kRowMajor;
    k_desc.rows  = input_dim;
    k_desc.cols  = output_dim;
    k_desc.ld    = output_dim;
}
```

- [ ] **Step 3: Build to verify compilation**

Run: `cd build && ninja`
Expected: Compiles without errors.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/linear_weight.cc
git commit -m "refactor(linear): implement ResolveLinearPolicy, update do_allocate"
```

---

### Task 6: Update LlamaLinear and MoeWeight consumers

**Files:**
- Modify: `src/turbomind/models/llama/LlamaLinear.cu` (lines 125-126: `resolved_.input_quant` → `policy_.input_quant`)
- Modify: `src/turbomind/models/moe_weight.cc` (line 84: `d.input_dtype()` — accessor unchanged, no edit needed)

- [ ] **Step 1: Update LlamaLinear.cu**

At line 125-126, replace:

```cpp
        op.quant_a   = dense.resolved_.input_quant;
        op.quant_b   = dense.resolved_.weight_quant;
```

With:

```cpp
        op.quant_a   = dense.policy_.input_quant;
        op.quant_b   = dense.policy_.weight_quant;
```

- [ ] **Step 2: Build to verify**

Run: `cd build && ninja`
Expected: Compiles without errors.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/models/llama/LlamaLinear.cu
git commit -m "refactor(llama): update LlamaLinear to use policy_ instead of resolved_"
```

---

### Task 7: Expose `DataFormat` and `QuantParamDesc` via pybind11

**Files:**
- Modify: `src/turbomind/python/bind.cpp` (add after the DataType enum block, around line 394)

- [ ] **Step 1: Add the include and bindings**

Add near the top of `bind.cpp`:

```cpp
#include "src/turbomind/core/data_format.h"
```

After the DataType/MemoryType block (after line 394), add:

```cpp
    // DataFormat descriptors
    py::class_<turbomind::QuantParamDesc>(m, "QuantParamDesc")
        .def_readonly("dtype", &turbomind::QuantParamDesc::dtype)
        .def_readonly("transposed", &turbomind::QuantParamDesc::transposed)
        .def("present", &turbomind::QuantParamDesc::present);

    py::class_<turbomind::DataFormat>(m, "DataFormat")
        .def_readonly("dtype", &turbomind::DataFormat::dtype)
        .def_readonly("block_sizes", &turbomind::DataFormat::block_sizes)
        .def_readonly("scales", &turbomind::DataFormat::scales)
        .def_readonly("zeros", &turbomind::DataFormat::zeros)
        .def("is_quantized", &turbomind::DataFormat::is_quantized)
        .def("rank", &turbomind::DataFormat::rank);
```

- [ ] **Step 2: Build and verify the Python extension compiles**

Run: `cd build && ninja`
Expected: Compiles without errors.

- [ ] **Step 3: Verify Python import works**

Run: `PYTHONPATH=lmdeploy:build/lib python -c "from _turbomind import DataFormat, QuantParamDesc; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/python/bind.cpp
git commit -m "feat(bind): expose DataFormat and QuantParamDesc to Python"
```

---

### Task 8: Create Python validity checkers

**Files:**
- Create: `lmdeploy/turbomind/deploy/format_ops.py`

- [ ] **Step 1: Write the module**

```python
# Copyright (c) OpenMMLab. All rights reserved.
"""Operation validity checkers for quantized tensor bundles.

These functions read `DataFormat` fields (exposed via pybind11 from C++)
and determine whether dimension-level operations are safe on quantized
tensor bundles.  They are decision predicates used for branching — not
exception guards.
"""

from __future__ import annotations

from _turbomind import DataFormat
from _turbomind import QuantParamDesc


def can_permute(fmt: DataFormat, dim0: int, dim1: int) -> bool:
    """Can we permute dim0 <-> dim1 without breaking block structure?

    Returns False if either dimension has block_size > 1, because permuting
    would cross block boundaries and invalidate quantization parameter
    alignment.
    """
    if dim0 >= len(fmt.block_sizes) or dim1 >= len(fmt.block_sizes):
        return True
    return fmt.block_sizes[dim0] == 1 and fmt.block_sizes[dim1] == 1


def can_split(fmt: DataFormat, dim: int) -> bool:
    """Can we split along *dim* without breaking block structure?

    Requires block_size == 1 along that dimension.
    """
    if dim >= len(fmt.block_sizes):
        return True
    return fmt.block_sizes[dim] == 1


def param_shape(data_shape: list[int], block_sizes: list[int],
                param: QuantParamDesc) -> list[int]:
    """Compute quant param shape given data shape and block sizes.

    Accounts for reduced dimensions (ceil-div by block_size) and
    transposed layout.
    """
    shape: list[int] = []
    for i, s in enumerate(data_shape):
        bs = block_sizes[i] if i < len(block_sizes) else 1
        if bs > 1:
            shape.append(-(-s // bs))  # ceil div
        else:
            shape.append(s)
    if param.transposed:
        shape.reverse()
    return shape
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/format_ops.py
git commit -m "feat(deploy): add Python validity checkers for DataFormat"
```

---

### Task 9: Add `data_format` field to `Linear` and use `param_shape` in loading

**Files:**
- Modify: `lmdeploy/turbomind/deploy/linear.py` (line 120-136: `Linear` dataclass, lines 190-206: `preprocess_linear`)

- [ ] **Step 1: Add `data_format` field to `Linear` dataclass**

Add import at the top of `linear.py`:

```python
from _turbomind import DataFormat
```

In the `Linear` dataclass (around line 120), add a field after `weight_format`:

```python
@dataclass
class Linear:
    """Bundle of tensors for a single linear layer."""

    tensors: dict[str, Tensor]
    weight_format: WeightFormat | None = field(default=None, compare=False, repr=False)
    data_format: DataFormat | None = field(default=None, compare=False, repr=False)
```

Update all `Linear(...)` construction sites to pass `data_format=self.data_format` (in `split_out_dim`, `split_in_dim`, `concat_out_dim`, `concat_in_dim`, `interleave_linears`, `chunk_linears`). For example in `split_out_dim`:

```python
return [Linear(tensors=b, weight_format=self.weight_format, data_format=self.data_format)
        for b in buckets]
```

Same pattern for all other factory methods.

- [ ] **Step 2: Update `preprocess_linear` to use `param_shape`**

Add import:

```python
from .format_ops import param_shape
```

Replace the body of `preprocess_linear` (lines 190-206):

```python
def preprocess_linear(linear: Linear) -> Linear:
    """Expand FP8 block scales to group scales (blockscale → groupscale)."""
    fmt = linear.data_format
    if fmt is not None and fmt.is_quantized() and fmt.scales.present():
        scales = linear.tensors.get("scales")
        if scales is not None and scales.dim() == 2:
            block_size = fmt.block_sizes[-1] if len(fmt.block_sizes) > 1 else 1
            if block_size > 1:
                new_scales = scales.repeat_interleave(block_size, dim=-1)
                new_tensors = dict(linear.tensors)
                new_tensors["scales"] = new_scales
                return Linear(tensors=new_tensors,
                              weight_format=linear.weight_format,
                              data_format=linear.data_format)
    return linear
```

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/linear.py
git commit -m "feat(deploy): add data_format to Linear, use param_shape in preprocess"
```

---

### Task 10: Wire `DataFormat` construction into `WeightFormat` and loading pipeline

**Files:**
- Modify: `lmdeploy/turbomind/deploy/kind_map.py`
- Modify: `lmdeploy/turbomind/deploy/module.py` (where `Linear` objects are created with `WeightFormat`)

- [ ] **Step 1: Add `to_data_format()` method to `WeightFormat`**

In `kind_map.py`, add import at top:

```python
from _turbomind import DataFormat as CppDataFormat
from _turbomind import DataType as CppDataType
```

Add a method to `WeightFormat`:

```python
def to_data_format(self, cpp_dtype: int, group_size: int = 0) -> CppDataFormat | None:
    """Construct a C++ DataFormat from this WeightFormat."""
    if self.block_in is None:
        return None
    gs = group_size if self.block_in == 0 else self.block_in
    from _turbomind import MakeLinearWeightFormat  # if exposed, or construct inline
    # Construct DataFormat directly since it's a pybind11 class with readonly fields
    fmt = CppDataFormat()
    fmt.dtype = CppDataType(cpp_dtype)
    if self.block_out is not None and self.block_out > 1:
        fmt.block_sizes = [self.block_out, self.block_out] if self.block_in == self.block_out else [1, gs]
    else:
        fmt.block_sizes = [1, gs]
    return fmt
```

Note: If pybind11 `def_readonly` prevents field assignment, change the binding to `def_readwrite` in Task 7, or expose a constructor. See the note below.

- [ ] **Step 2: Wire DataFormat creation where Linear objects are built**

Find where `Linear(tensors=..., weight_format=...)` is constructed in `module.py` and add `data_format=wfmt.to_data_format(cpp_dtype, group_size)`.

- [ ] **Step 3: Build and run a model test**

Run: `PYTHONPATH=lmdeploy:build/lib python scripts/test_turbomind_model.py --help` to verify imports work.
Then test with a model (use turbomind-tester agent).

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/kind_map.py lmdeploy/turbomind/deploy/module.py
git commit -m "feat(deploy): wire DataFormat construction into WeightFormat and loading"
```

---

### Task 11: End-to-end model test

**Files:** None (testing only)

- [ ] **Step 1: Test a dense FP16 model**

Use the turbomind-tester agent to verify a standard model (e.g. a small Llama or Qwen model available in the model cache) still produces correct output.

- [ ] **Step 2: Test a quantized model (AWQ or FP8)**

Use the turbomind-tester agent to verify a quantized model still produces correct output. Check that `input_dtype()`, `output_dtype()`, scale shapes, and weight processing all work correctly.

- [ ] **Step 3: Verify the Python validity checkers**

Run: `PYTHONPATH=lmdeploy:build/lib python -c "
from lmdeploy.turbomind.deploy.format_ops import can_permute, can_split
from _turbomind import DataFormat
fmt = DataFormat()
# Dense: should allow all ops
assert can_split(fmt, 0)
assert can_permute(fmt, 0, 1)
print('All Python checker tests passed')
"`
Expected: `All Python checker tests passed`

---

## Self-Review

**1. Spec coverage:**
- Section 1 (DataFormat struct) → Task 1 + 2
- Section 2 (MakeLinearWeightFormat) → Task 2
- Section 3 (LinearPolicy + ResolveLinearPolicy) → Task 4 + 5
- Section 4 (C++ LinearWeight migration) → Task 4 + 5 + 6
- Section 5 (pybind11) → Task 7
- Section 6 (Python validity checkers) → Task 8
- Section 7 (Python loading pipeline) → Task 9 + 10

**2. Placeholder scan:** No TBD/TODO found. All steps contain concrete code.

**3. Type consistency:**
- `DataFormat::is_quantized()` → defined in Task 2, used in Task 5, Task 9
- `MakeLinearWeightFormat(DataType, DataType, int)` → declared in Task 2, called in Task 5
- `ResolveLinearPolicy(const DataFormat&, DataType, int)` → declared in Task 4, implemented in Task 5
- `policy_.input_quant` / `policy_.weight_quant` → defined in Task 4, used in Task 6
- `format_.scales.dtype` / `format_.zeros.dtype` → defined in Task 1, used in Task 5

**Note on Task 10:** The pybind11 `def_readonly` bindings in Task 7 will prevent field assignment from Python. Either:
- Change to `def_readwrite` in Task 7 for `block_sizes`, `dtype` fields
- Or expose `MakeLinearWeightFormat` to Python via pybind11 and call it from `WeightFormat.to_data_format()`

The plan assumes `def_readwrite` is the simpler path. Update Task 7 accordingly if needed.
