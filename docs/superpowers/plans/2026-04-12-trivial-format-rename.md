# Trivial Format Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename every occurrence of "dense" that refers to the weight-format concept to "trivial" across Python and C++.

**Architecture:** Mechanical symbol rename — no behavioral changes. Python `WeightFormat` name `"dense"` → `"trivial"`, C++ `IsDenseFloatType` → `IsTrivialFloatType` with deduplication into `data_format.h`, plus legacy naming cleanup.

**Tech Stack:** Python, C++, CMake

**Spec:** `docs/superpowers/specs/2026-04-12-trivial-format-rename-design.md`

---

### Task 1: Add `IsTrivialFloatType` to `data_format.h` and update `data_format.cc`

**Files:**
- Modify: `src/turbomind/core/data_format.h`
- Modify: `src/turbomind/core/data_format.cc`

- [ ] **Step 1: Add `IsTrivialFloatType` to `data_format.h`**

Add after the `#include` block, inside `namespace turbomind {`, before the `QuantParamDesc` struct:

```cpp
/// True for trivial (non-quantized) float dtypes: FP32, FP16, BF16.
inline bool IsTrivialFloatType(DataType t) noexcept
{
    return t == kFloat || t == kHalf || t == kBfloat16;
}
```

- [ ] **Step 2: Update `data_format.cc`**

Remove the static `IsDenseFloatType` function (lines 21-24) and update the call site:

```cpp
// DELETE lines 21-24:
// static bool IsDenseFloatType(DataType t)
// {
//     return t == kFloat || t == kHalf || t == kBfloat16;
// }

// UPDATE line 31 — change IsDenseFloatType to IsTrivialFloatType:
    if (IsTrivialFloatType(weight_format)) {
```

- [ ] **Step 3: Build to verify**

Run: `cd /data/lmdeploy-modeling/build && ninja`
Expected: Clean build, no errors.

- [ ] **Step 4: Run data_format tests**

Run: `cd /data/lmdeploy-modeling/build && ctest -R data_format --output-on-failure`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/core/data_format.h src/turbomind/core/data_format.cc
git commit -m "refactor(core): add IsTrivialFloatType to data_format.h, remove local IsDenseFloatType"
```

---

### Task 2: Update `memory_utils.h` and `memory_utils.cu`

**Files:**
- Modify: `src/turbomind/utils/memory_utils.h`
- Modify: `src/turbomind/utils/memory_utils.cu`

- [ ] **Step 1: Update comment in `memory_utils.h:32`**

Change:
```
/// If *tensor* is a dense float type that differs from *target_dtype*, cast
```
To:
```
/// If *tensor* is a trivial float type that differs from *target_dtype*, cast
```

- [ ] **Step 2: Update `memory_utils.cu`**

Remove the anonymous-namespace `IsDenseFloatType` (lines 114-118) and update call sites:

```cpp
// DELETE lines 114-118:
// namespace {
// bool IsDenseFloatType(DataType t)
// {
//     return t == kFloat || t == kHalf || t == kBfloat16;
// }
// }  // namespace

// UPDATE comment on line 111:
// EnsureFloatDtype — cast tensor to target dtype if both are trivial float

// UPDATE line 126:
    if (!IsTrivialFloatType(tensor.dtype()) || !IsTrivialFloatType(target_dtype)) {
```

Add include at top of file:
```cpp
#include "src/turbomind/core/data_format.h"
```

- [ ] **Step 3: Build to verify**

Run: `cd /data/lmdeploy-modeling/build && ninja`
Expected: Clean build.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/utils/memory_utils.h src/turbomind/utils/memory_utils.cu
git commit -m "refactor(utils): replace IsDenseFloatType with IsTrivialFloatType from data_format.h"
```

---

### Task 3: Update `linear_weight.h` and `linear_weight.cc`

**Files:**
- Modify: `src/turbomind/models/linear_weight.h`
- Modify: `src/turbomind/models/linear_weight.cc`

- [ ] **Step 1: Update comment in `linear_weight.h:38`**

Change:
```
    /// For dense float weights, coerces to model compute dtype to avoid
```
To:
```
    /// For trivial float weights, coerces to model compute dtype to avoid
```

- [ ] **Step 2: Update `linear_weight.cc`**

Update comment on line 69:
```cpp
    // Default policy for trivial (non-quantized) weights.
```

Update comment on line 97:
```cpp
    // For trivial float weights, coerce to model compute dtype
```

Replace the inline lambda on line 98 with `IsTrivialFloatType`:
```cpp
    // DELETE:
    // auto is_dense_float = [](DataType t) { return t == kFloat || t == kHalf || t == kBfloat16; };
    // if (weight_dtype != data_type && is_dense_float(weight_dtype) && is_dense_float(data_type)) {

    // REPLACE WITH:
    if (weight_dtype != data_type && IsTrivialFloatType(weight_dtype) && IsTrivialFloatType(data_type)) {
```

Update comment on line 134:
```cpp
    // No format conversion needed if weight_spec was never set (trivial weights
```

- [ ] **Step 3: Build to verify**

Run: `cd /data/lmdeploy-modeling/build && ninja`
Expected: Clean build.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/linear_weight.h src/turbomind/models/linear_weight.cc
git commit -m "refactor(models): replace is_dense_float lambda with IsTrivialFloatType"
```

---

### Task 4: Update C++ test names and test infrastructure

**Files:**
- Modify: `src/turbomind/core/test_data_format.cc`
- Modify: `src/turbomind/kernels/gemm/test/testbed_v3.h`

- [ ] **Step 1: Rename test cases in `test_data_format.cc`**

Line 18: change `"DataFormat dense is not quantized"` to `"DataFormat trivial is not quantized"`.
Line 71: change `"DataFormat dense BF16"` to `"DataFormat trivial BF16"`.

- [ ] **Step 2: Remove `DenseWeight` alias in `testbed_v3.h`**

Line 24: delete `using DenseWeight = LinearWeight;`.

Replace all `DenseWeight` with `LinearWeight` throughout the file (~30 occurrences at lines 81, 160-167, 297, 365, 369, 374, 377, 442, 487-489, 499-501).

- [ ] **Step 3: Build to verify**

Run: `cd /data/lmdeploy-modeling/build && ninja`
Expected: Clean build.

- [ ] **Step 4: Run data_format tests**

Run: `cd /data/lmdeploy-modeling/build && ctest -R data_format --output-on-failure`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/core/test_data_format.cc src/turbomind/kernels/gemm/test/testbed_v3.h
git commit -m "refactor(test): rename dense test cases to trivial, remove DenseWeight alias"
```

---

### Task 5: Rename `dense` parameter in `LlamaLinear.cu`

**Files:**
- Modify: `src/turbomind/models/llama/LlamaLinear.cu`

- [ ] **Step 1: Rename parameter `dense` → `weight` in all three functions**

`GetOperandB` (line 56): `const LinearWeight& dense` → `const LinearWeight& weight`, then update `dense.weight` → `weight.weight`, `dense.scales` → `weight.scales`, `dense.k_desc` → `weight.k_desc`, `dense.q_desc` → `weight.q_desc`.

`GetOperandA` (line 66): same rename, update `dense.input_dtype()` → `weight.input_dtype()`, `dense.k_desc.num` → `weight.k_desc.num`.

`Forward` (line 116): same rename, update all `dense.` references to `weight.` (epilogue, policy_, output_dim, output_dtype, k_desc).

- [ ] **Step 2: Build to verify**

Run: `cd /data/lmdeploy-modeling/build && ninja`
Expected: Clean build.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/models/llama/LlamaLinear.cu
git commit -m "refactor(llama): rename 'dense' parameter to 'weight' in LlamaLinear"
```

---

### Task 6: Update remaining C++ comments and clean up CMake dead references

**Files:**
- Modify: `src/turbomind/kernels/gemm/convert_v3.cu`
- Modify: `src/turbomind/models/moe_weight.cc`
- Modify: `src/turbomind/models/llama/CMakeLists.txt`
- Modify: `src/turbomind/kernels/gemm/CMakeLists.txt`

- [ ] **Step 1: Update `convert_v3.cu:115`**

Change: `return {};  //  trivial case: dense floating point`
To: `return {};  //  trivial case: no quantization`

- [ ] **Step 2: Update `moe_weight.cc:39`**

Change: `// Adapted from LinkExperts in LlamaDenseWeight.cc for LinearWeight`
To: `// Adapted from LinkExperts for LinearWeight`

- [ ] **Step 3: Update `moe_weight.cc:157`**

Change: `// SiLU (true for quantized formats, false for dense bf16/fp16).`
To: `// SiLU (true for quantized formats, false for trivial bf16/fp16).`

- [ ] **Step 4: Remove dead CMake references**

In `src/turbomind/models/llama/CMakeLists.txt:16`, remove the line `LlamaDenseWeight.cc`.

In `src/turbomind/kernels/gemm/CMakeLists.txt:57`, remove the line `../../models/llama/LlamaDenseWeight.cc`.

- [ ] **Step 5: Build to verify**

Run: `cd /data/lmdeploy-modeling/build && ninja`
Expected: Clean build.

- [ ] **Step 6: Commit**

```bash
git add src/turbomind/kernels/gemm/convert_v3.cu src/turbomind/models/moe_weight.cc \
        src/turbomind/models/llama/CMakeLists.txt src/turbomind/kernels/gemm/CMakeLists.txt
git commit -m "refactor(cpp): update dense->trivial comments, remove dead CMake references"
```

---

### Task 7: Rename Python `DENSE_FORMAT` → `TRIVIAL_FORMAT` in `kind_map.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/kind_map.py`

- [ ] **Step 1: Rename symbols and strings**

Apply these renames:

| Location | Current | New |
|----------|---------|-----|
| Line 123 | `DENSE_SUFFIXES` | `TRIVIAL_SUFFIXES` |
| Line 203 | `_normalize_dense` | `_normalize_trivial` |
| Line 275-276 | `None: _normalize_dense` / `"hf": _normalize_dense` | `None: _normalize_trivial` / `"hf": _normalize_trivial` |
| Line 328 | `_accepts_dense` | `_accepts_trivial` |
| Line 418-428 | `DENSE_FORMAT`, `name="dense"`, `DENSE_SUFFIXES`, `_normalize_dense`, `_accepts_dense` | `TRIVIAL_FORMAT`, `name="trivial"`, `TRIVIAL_SUFFIXES`, `_normalize_trivial`, `_accepts_trivial` |
| Line 497-498 | `None: DENSE_FORMAT`, `"hf": DENSE_FORMAT` | `None: TRIVIAL_FORMAT`, `"hf": TRIVIAL_FORMAT` |
| Line 524 | `DENSE_FORMAT,` | `TRIVIAL_FORMAT,` |

- [ ] **Step 2: Update docstring comments referencing "dense"**

Update these docstrings/comments that use "dense" in the weight-format sense:

Line 4: `Each model format (dense, AWQ, GPTQ, compressed-tensors, FP8, mxfp4)` → `Each model format (trivial, AWQ, GPTQ, compressed-tensors, FP8, mxfp4)`

Line 37: `Canonical format name (``None`` for dense/HF).` → `Canonical format name (``None`` for trivial/HF).`

Line 47: `or ``None`` for dense formats whose dtype is inferred from the tensor.` → `or ``None`` for trivial formats whose dtype is inferred from the tensor.`

Line 50: ```None`` → no per-element scale (dense).` → ```None`` → no per-element scale (trivial).`

Line 67: `stored without ``weight_scale_inv`` is correctly classified as dense` → `stored without ``weight_scale_inv`` is correctly classified as trivial`

Line 70: `maps TM ``tensors`` to a dense {weight, bias?} dict.` → `maps TM ``tensors`` to a trivial {weight, bias?} dict.`

Line 93: `or when the format is dense (block_in is None).` → `or when the format is trivial (block_in is None).`

Line 329: `"""Dense: only .weight and/or .bias present...` → `"""Trivial: only .weight and/or .bias present...`

Line 381: `# Fusion-time dequantizers: TM tensors -> dense {weight, bias?}` → `# Fusion-time dequantizers: TM tensors -> trivial {weight, bias?}`

Line 517: `#: Quantized formats are listed first so they win over dense when tensors match.` → `#: Quantized formats are listed first so they win over trivial when tensors match.`

- [ ] **Step 3: Verify Python import still works**

Run: `cd /data/lmdeploy-modeling && PYTHONPATH=lmdeploy:build/lib python -c "from lmdeploy.turbomind.deploy.kind_map import TRIVIAL_FORMAT; print(TRIVIAL_FORMAT.name)"`
Expected: `trivial`

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/kind_map.py
git commit -m "refactor(deploy): rename DENSE_FORMAT to TRIVIAL_FORMAT in kind_map.py"
```

---

### Task 8: Update Python consumers — `spec.py`, `commit.py`, `gpt_oss_spec.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/spec.py`
- Modify: `lmdeploy/turbomind/deploy/commit.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py`

- [ ] **Step 1: Update `spec.py`**

Line 6: `from .kind_map import DENSE_FORMAT` → `from .kind_map import TRIVIAL_FORMAT`

Line 334: `weight_format=DENSE_FORMAT` → `weight_format=TRIVIAL_FORMAT`

Comments to update:
- Line 329: `Dequantize a quantized Linear to dense` → `Dequantize a quantized Linear to trivial`
- Line 338: `Dequant linears to a common dense format` → `Dequant linears to a common trivial format`
- Line 367: `the caller should dequantise to dense first.` → `the caller should dequantise to trivial first.`
- Line 409: `the entire group is dequantised to dense` → `the entire group is dequantised to trivial`
- Line 425: `f"dequantising to dense."` → `f"dequantising to trivial."`

Note: Line 208 `dense FFN` is MoE-architecture sense — DO NOT change.

- [ ] **Step 2: Update `commit.py`**

Line 121: `fmt.name != 'dense'` → `fmt.name != 'trivial'`

Comments to update:
- Line 73: `# Dense (or missing format):` → `# Trivial (or missing format):`
- Line 83: `For dense formats the weight itself carries the compute dtype.` → `For trivial formats the weight itself carries the compute dtype.`
- Line 156: `# Dense weight: use model compute dtype for dtype coercion.` → `# Trivial weight: use model compute dtype for dtype coercion.`
- Line 208: `dense (non-quantized) weights use this dtype` → `trivial (non-quantized) weights use this dtype`

- [ ] **Step 3: Update `gpt_oss_spec.py`**

Line 55: `The dense normalizer assumes HF` → `The trivial normalizer assumes HF`

Line 62: `if lin.weight_format.name == "dense":` → `if lin.weight_format.name == "trivial":`

Note: Line 5 `no dense FFN` is MoE-architecture sense — DO NOT change.

- [ ] **Step 4: Verify Python imports still work**

Run: `cd /data/lmdeploy-modeling && PYTHONPATH=lmdeploy:build/lib python -c "from lmdeploy.turbomind.deploy.spec import LlamaModelSpec; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add lmdeploy/turbomind/deploy/spec.py lmdeploy/turbomind/deploy/commit.py \
        lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py
git commit -m "refactor(deploy): update dense->trivial in spec.py, commit.py, gpt_oss_spec.py"
```

---

### Task 9: Run model test to verify end-to-end correctness

**Files:** None (verification only)

- [ ] **Step 1: Check GPU availability**

Run: Use the `get_gpu_usage` MCP tool to verify a GPU is free.

- [ ] **Step 2: Test a trivial-format model (e.g. Qwen2.5-7B BF16)**

Run the turbomind-tester agent with a BF16 model to verify that trivial-format weight loading still works correctly. Request at least 128 tokens and verify the response contains meaningful human words.

- [ ] **Step 3: Test a quantized model (e.g. an AWQ or GPTQ model)**

Run the turbomind-tester agent with a quantized model to verify that quantized weight loading is unaffected.

- [ ] **Step 4: Final commit (if any fixes were needed)**

Only if bugs were found and fixed.
