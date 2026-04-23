# Dtype-Cast Dedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract a shared `EnsureFloatDtype()` helper to eliminate duplicated dtype-casting code across 3 weight classes.

**Architecture:** Add `EnsureFloatDtype(Tensor&, DataType)` to `memory_utils.h/.cu`. It uses `core::Context::stream()` internally (no stream parameter). Each weight class's `prepare()` replaces its inline cast logic with a call to this helper. File-local `IsDenseFloatType` copies are removed from weight classes and consolidated inside `memory_utils.cu`.

**Tech Stack:** C++17, CUDA

**Spec:** `docs/superpowers/specs/2026-04-11-dtype-cast-dedup-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/turbomind/utils/memory_utils.h` | **Modify** | Add `EnsureFloatDtype` declaration |
| `src/turbomind/utils/memory_utils.cu` | **Modify** | Add `EnsureFloatDtype` implementation + file-local `IsDenseFloatType` |
| `src/turbomind/models/linear_weight.cc` | **Modify** | Replace 2 inline casts with `EnsureFloatDtype`, remove local `IsDenseFloatType` |
| `src/turbomind/models/norm_weight.cc` | **Modify** | Replace 1 inline cast with `EnsureFloatDtype`, remove local `IsDenseFloatType` |
| `src/turbomind/models/delta_net_weight.cc` | **Modify** | Replace 3 inline casts with `EnsureFloatDtype`, remove local `IsDenseFloatType` + `CastIfNeeded` |

---

### Task 1: Add `EnsureFloatDtype` to `memory_utils.h/.cu`

**Files:**
- Modify: `src/turbomind/utils/memory_utils.h`
- Modify: `src/turbomind/utils/memory_utils.cu`

- [ ] **Step 1: Add declaration to `memory_utils.h`**

After the existing `invokeDtypeCast` declaration, add:

```cpp
/// If *tensor* is a dense float type that differs from *target_dtype*, cast
/// it in-place (allocates a temporary, casts, move-assigns).  Uses
/// Context::stream() internally — no stream parameter needed.
void EnsureFloatDtype(Tensor& tensor, DataType target_dtype);
```

The final section of `memory_utils.h` should be:

```cpp
/// Element-wise dtype cast kernel.  Supports fp32 <-> fp16 <-> bf16.
void invokeDtypeCast(void* dst, const void* src, size_t count, DataType dst_dtype, DataType src_dtype, cudaStream_t stream = 0);

/// If *tensor* is a dense float type that differs from *target_dtype*, cast
/// it in-place (allocates a temporary, casts, move-assigns).  Uses
/// Context::stream() internally — no stream parameter needed.
void EnsureFloatDtype(Tensor& tensor, DataType target_dtype);

}  // namespace turbomind
```

Note: `Tensor` is already available in scope — `memory_utils.h` includes `core/data_type.h` (line 20) which brings in `DataType`. The `Tensor` type itself is from `core/tensor.h` which is brought in transitively through the models that include this header. Since `memory_utils.h` only declares the function (no inline body), forward declaration is not needed — the definition in `.cu` provides the full type.

- [ ] **Step 2: Add implementation to `memory_utils.cu`**

Add these includes at the top (after the existing includes):

```cpp
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/tensor.h"
```

Add `EnsureFloatDtype` implementation after the existing `invokeDtypeCast` function, before the closing `}  // namespace turbomind`:

```cpp
// -----------------------------------------------------------------------
// EnsureFloatDtype — cast tensor to target dtype if both are dense float
// -----------------------------------------------------------------------

namespace {
bool IsDenseFloatType(DataType t)
{
    return t == kFloat || t == kHalf || t == kBfloat16;
}
}  // namespace

void EnsureFloatDtype(Tensor& tensor, DataType target_dtype)
{
    if (!tensor || tensor.dtype() == target_dtype) {
        return;
    }
    if (!IsDenseFloatType(tensor.dtype()) || !IsDenseFloatType(target_dtype)) {
        return;
    }
    auto stream = core::Context::stream().handle();
    Tensor casted{tensor.shape(), target_dtype, tensor.device()};
    invokeDtypeCast(casted.raw_data(), tensor.raw_data(), tensor.size(),
                    target_dtype, tensor.dtype(), stream);
    tensor = std::move(casted);
}
```

- [ ] **Step 3: Build to verify compilation**

Run: `cd /data/lmdeploy-modeling/build && ninja 2>&1 | tail -20`
Expected: Clean build (no errors). The new code compiles but is not yet called by anyone.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/utils/memory_utils.h src/turbomind/utils/memory_utils.cu
git commit -m "refactor: add EnsureFloatDtype helper to memory_utils"
```

---

### Task 2: Simplify `linear_weight.cc`

**Files:**
- Modify: `src/turbomind/models/linear_weight.cc`

- [ ] **Step 1: Remove local `IsDenseFloatType` and replace inline casts**

The current file has `IsDenseFloatType` in an anonymous namespace at lines 22-25. Remove it entirely:

```cpp
// DELETE these lines (22-25):
static bool IsDenseFloatType(DataType t)
{
    return t == kFloat || t == kHalf || t == kBfloat16;
}
```

The anonymous namespace block for `LinearWeightRegistrar` (starting around line 311) is separate and must be kept.

In `prepare()`, replace the dense-weight cast block (the `if (weight_format == DataType{})` branch). The current code at lines 139-151:

```cpp
    if (weight_format == DataType{}) {
        // Cast dense weight to match configured data_type if they differ
        if (weight.dtype() != data_type && IsDenseFloatType(weight.dtype()) && IsDenseFloatType(data_type)) {
            auto stream = core::Context::stream().handle();
            Tensor casted{weight.shape(), data_type, weight.device()};
            invokeDtypeCast(casted.raw_data(), weight.raw_data(), weight.size(),
                            data_type, weight.dtype(), stream);
            weight   = std::move(casted);
            k_desc.type = data_type;
        }
        return;
    }
```

Replace with:

```cpp
    if (weight_format == DataType{}) {
        EnsureFloatDtype(weight, data_type);
        if (weight.dtype() == data_type) {
            k_desc.type = data_type;
        }
        return;
    }
```

Note: `EnsureFloatDtype` modifies `weight` in-place. After the call, if the cast happened, `weight.dtype()` will now equal `data_type` and we update `k_desc.type`. If no cast was needed, the dtype was already matching.

Also in `prepare()`, replace the FP8 scale cast block (lines 169-174). The current code:

```cpp
        // FP8 native path requires f32 scales; cast if loaded as bf16/fp16.
        if (scales && scales.dtype() != kFloat && IsDenseFloatType(scales.dtype())) {
            Tensor casted{scales.shape(), kFloat, scales.device()};
            invokeDtypeCast(casted.raw_data(), scales.raw_data(), scales.size(),
                            kFloat, scales.dtype(), stream);
            scales = std::move(casted);
        }
```

Replace with:

```cpp
        // FP8 native path requires f32 scales; cast if loaded as bf16/fp16.
        EnsureFloatDtype(scales, kFloat);
```

The `#include "src/turbomind/utils/memory_utils.h"` is already present (line 13), so no include changes needed.

The `#include "src/turbomind/kernels/gemm/cast.h"` include was used only for the old cast — but it's also used for `invokeTransposeAxis01` via other code, so leave it.

- [ ] **Step 2: Build to verify compilation**

Run: `cd /data/lmdeploy-modeling/build && ninja 2>&1 | tail -20`
Expected: Clean build.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/models/linear_weight.cc
git commit -m "refactor(linear_weight): use EnsureFloatDtype for dtype casting"
```

---

### Task 3: Simplify `norm_weight.cc`

**Files:**
- Modify: `src/turbomind/models/norm_weight.cc`

- [ ] **Step 1: Remove local `IsDenseFloatType` and replace inline cast**

Remove the anonymous namespace block at lines 12-17:

```cpp
// DELETE these lines (12-17):
namespace {
bool IsDenseFloatType(DataType t)
{
    return t == kFloat || t == kHalf || t == kBfloat16;
}
}  // namespace
```

Replace the `prepare()` method body (lines 52-61):

```cpp
void NormWeight::prepare()
{
    if (weight && weight.dtype() != dtype_ && IsDenseFloatType(weight.dtype()) && IsDenseFloatType(dtype_)) {
        auto stream = core::Context::stream().handle();
        Tensor casted{weight.shape(), dtype_, weight.device()};
        invokeDtypeCast(casted.raw_data(), weight.raw_data(), weight.size(),
                        dtype_, weight.dtype(), stream);
        weight = std::move(casted);
    }
}
```

Replace with:

```cpp
void NormWeight::prepare()
{
    EnsureFloatDtype(weight, dtype_);
}
```

The `#include "src/turbomind/utils/memory_utils.h"` is already present (line 8). The `#include "src/turbomind/utils/cuda_utils.h"` was used for the old `core::Context::stream()` — it may still be needed by other code in this file. Check: `cuda_utils.h` is not otherwise used. Remove it.

The final include block should be:

```cpp
#include "src/turbomind/models/norm_weight.h"

#include "src/turbomind/core/module_config.h"
#include "src/turbomind/core/registry.h"
#include "src/turbomind/utils/memory_utils.h"
```

- [ ] **Step 2: Build to verify compilation**

Run: `cd /data/lmdeploy-modeling/build && ninja 2>&1 | tail -20`
Expected: Clean build.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/models/norm_weight.cc
git commit -m "refactor(norm_weight): use EnsureFloatDtype for dtype casting"
```

---

### Task 4: Simplify `delta_net_weight.cc`

**Files:**
- Modify: `src/turbomind/models/delta_net_weight.cc`

- [ ] **Step 1: Remove local helpers and replace inline casts**

Remove the entire anonymous namespace block at lines 11-26:

```cpp
// DELETE these lines (11-26):
namespace {
bool IsDenseFloatType(DataType t)
{
    return t == kFloat || t == kHalf || t == kBfloat16;
}

void CastIfNeeded(Tensor& tensor, DataType target_dtype, cudaStream_t stream)
{
    if (tensor && tensor.dtype() != target_dtype && IsDenseFloatType(tensor.dtype()) && IsDenseFloatType(target_dtype)) {
        Tensor casted{tensor.shape(), target_dtype, tensor.device()};
        invokeDtypeCast(casted.raw_data(), tensor.raw_data(), tensor.size(),
                        target_dtype, tensor.dtype(), stream);
        tensor = std::move(casted);
    }
}
}  // namespace
```

Replace the `prepare()` method body (lines 42-50):

```cpp
void DeltaNetWeight::prepare()
{
    Module::prepare();

    auto stream = core::Context::stream().handle();
    CastIfNeeded(A_log, data_type_, stream);
    CastIfNeeded(dt_bias, data_type_, stream);
    CastIfNeeded(conv1d, data_type_, stream);
}
```

Replace with:

```cpp
void DeltaNetWeight::prepare()
{
    Module::prepare();

    EnsureFloatDtype(A_log, data_type_);
    EnsureFloatDtype(dt_bias, data_type_);
    EnsureFloatDtype(conv1d, data_type_);
}
```

The `#include "src/turbomind/utils/memory_utils.h"` is already present (line 7). The `#include "src/turbomind/utils/cuda_utils.h"` was used for `core::Context::stream().handle()` — no longer needed. Remove it.

The final include block should be:

```cpp
#include "src/turbomind/models/delta_net_weight.h"

#include "src/turbomind/core/registry.h"
#include "src/turbomind/utils/memory_utils.h"
```

- [ ] **Step 2: Build to verify compilation**

Run: `cd /data/lmdeploy-modeling/build && ninja 2>&1 | tail -20`
Expected: Clean build.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/models/delta_net_weight.cc
git commit -m "refactor(delta_net_weight): use EnsureFloatDtype for dtype casting"
```

---

### Task 5: Verify with model test

**Files:** None (verification only)

- [ ] **Step 1: Run a model test to verify no regression**

Use the turbomind-tester agent to verify at least one model loads and produces correct output with both TP=1 and TP=2. The dtype-casting code is exercised by models whose checkpoint dtype differs from the configured compute dtype (e.g., BF16 weights in FP16 models).

---

## Self-Review

**1. Spec coverage:**
- Section "Add EnsureFloatDtype to memory_utils.h/.cu" → Task 1 ✓
- Section "Simplify call sites" (linear_weight.cc) → Task 2 ✓
- Section "Simplify call sites" (norm_weight.cc) → Task 3 ✓
- Section "Simplify call sites" (delta_net_weight.cc) → Task 4 ✓
- Section "Remove local helpers" → Tasks 2, 3, 4 each remove their own ✓
- Dependencies section (context.h include) → Task 1 Step 2 ✓

**2. Placeholder scan:** No TBDs, TODOs, or "implement later" patterns. All code shown.

**3. Type consistency:**
- `EnsureFloatDtype(Tensor&, DataType)` — consistent across declaration (Task 1 Step 1), definition (Task 1 Step 2), and all 6 call sites (Tasks 2-4)
- `Tensor` type comes from `core/tensor.h` which is included via `core.h` in weight classes
- `DataType` comes from `core/data_type.h` which is already included in `memory_utils.h`
- Function name `EnsureFloatDtype` is identical everywhere
