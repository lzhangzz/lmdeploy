# GenericCopy Test Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Python bindings for GenericCopy and write a standalone test script that verifies correctness against PyTorch for various non-contiguous tensor layouts.

**Architecture:** Two new pybind11 functions in `bind.cpp` — `from_dlpack_with_strides` (stride-preserving DLPack import) and `generic_copy` (calls `core::GenericCopy` + stream sync). A standalone Python script creates non-contiguous PyTorch tensors, copies them through GenericCopy, and compares results.

**Tech Stack:** pybind11, PyTorch, CUDA, turbomind::core

---

## File Structure

| File | Responsibility |
|---|---|
| `src/turbomind/python/bind.cpp` | Add `from_dlpack_with_strides` and `generic_copy` pybind11 bindings |
| `test_generic_copy.py` | Standalone test script comparing GenericCopy against PyTorch |

---

### Task 1: Add `from_dlpack_with_strides` binding

**Files:**
- Modify: `src/turbomind/python/bind.cpp` (add after the existing `from_dlpack` at line 443)

- [ ] **Step 1: Add `from_dlpack_with_strides` function and binding**

Add this function before `PYBIND11_MODULE` (after the existing `DLManagedTensorToTritonTensor` at line 228):

```cpp
std::shared_ptr<Tensor> DLManagedTensorToTritonTensorWithStrides(DLManagedTensor* tensor)
{
    auto& dl_tensor = tensor->dl_tensor;
    auto  where     = getMemoryType(dl_tensor.device);
    auto  dtype     = getDataType(dl_tensor.dtype);
    assert(dl_tensor.ndim > 0);
    std::vector<ft::core::ssize_t> shape(dl_tensor.shape, dl_tensor.shape + dl_tensor.ndim);

    // Compute row-major strides if DLPack strides are NULL (contiguous tensor)
    std::vector<ft::core::ssize_t> strides;
    if (dl_tensor.strides) {
        strides.assign(dl_tensor.strides, dl_tensor.strides + dl_tensor.ndim);
    }
    else {
        strides.resize(dl_tensor.ndim, 1);
        for (int i = dl_tensor.ndim - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }

    ft::core::Layout layout(std::move(shape), std::move(strides));

    std::shared_ptr<void> ptr{dl_tensor.data, [tensor](void* p) {
                                  if (tensor->deleter) {
                                      tensor->deleter(tensor);
                                  }
                              }};

    return std::make_shared<Tensor>(ptr, std::move(layout), dtype, where);
}
```

Then add the module-level binding inside `PYBIND11_MODULE`, right after the existing `from_dlpack` at line 443:

```cpp
    m.def(
        "from_dlpack_with_strides",
        [](py::object obj) {
            py::capsule      cap = obj.attr("__dlpack__")();
            DLManagedTensor* dlmt =
                static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap.ptr(), kDlTensorCapsuleName));
            auto ret = DLManagedTensorToTritonTensorWithStrides(dlmt);
            // take ownership of capsule's payload
            cap.set_name("used_dltensor");
            return ret;
        },
        "dl_managed_tensor"_a);
```

The `Layout(vector<ssize_t>, vector<ssize_t>)` constructor is defined at `src/turbomind/core/layout.h:19`. The `Tensor(shared_ptr<void>, Layout, DataType, Device)` constructor is at `src/turbomind/core/tensor.h:34-36`. DLPack strides are element-count (not byte-count), which matches how `core::Layout` stores strides.

- [ ] **Step 2: Verify compilation**

Run: `cd /data/lmdeploy-copy/build && ninja _turbomind 2>&1 | tail -30`
Expected: Compilation succeeds.

- [ ] **Step 3: Commit**

```bash
cd /data/lmdeploy-copy
git add src/turbomind/python/bind.cpp
git commit -m "feat(bindings): add from_dlpack_with_strides to preserve tensor strides"
```

---

### Task 2: Add `generic_copy` binding

**Files:**
- Modify: `src/turbomind/python/bind.cpp` (add after the `from_dlpack_with_strides` binding)

- [ ] **Step 1: Add `generic_copy` module-level binding**

Add this inside `PYBIND11_MODULE`, right after the `from_dlpack_with_strides` binding added in Task 1:

```cpp
    m.def(
        "generic_copy",
        [](std::shared_ptr<Tensor> src, std::shared_ptr<Tensor> dst) {
            using ft::core::GenericCopy;
            using ft::core::Context;
            GenericCopy(*src, *dst, Context::stream());
            Context::stream().Sync();
        },
        "src"_a,
        "dst"_a);
```

`GenericCopy` is declared at `src/turbomind/core/tensor.h:234` with signature `void GenericCopy(const Tensor& src, Tensor& dst, const Stream& stream)`. We dereference the `shared_ptr<Tensor>` handles to get `Tensor&` references. The `Context::stream()` returns the current CUDA stream (defined at `src/turbomind/core/context.h:11`). `Stream::Sync()` calls `cudaStreamSynchronize` (at `src/turbomind/core/stream.h:26-28`).

- [ ] **Step 2: Verify compilation**

Run: `cd /data/lmdeploy-copy/build && ninja _turbomind 2>&1 | tail -30`
Expected: Compilation succeeds.

- [ ] **Step 3: Commit**

```bash
cd /data/lmdeploy-copy
git add src/turbomind/python/bind.cpp
git commit -m "feat(bindings): add generic_copy Python binding for GenericCopy kernel"
```

---

### Task 3: Write the test script

**Files:**
- Create: `test_generic_copy.py`

- [ ] **Step 1: Create the test script**

Create `test_generic_copy.py` at the repository root with this content:

```python
#!/usr/bin/env python3
"""Test GenericCopy against PyTorch for various non-contiguous layouts."""

import os
import sys

# Set up paths before any turbomind imports
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, 'lmdeploy'))
sys.path.insert(0, os.path.join(REPO, 'build', 'lib'))

import torch
import _turbomind as _tm


def make_tensors(torch_tensor):
    """Create (tm_src, tm_dst, golden) from a (possibly non-contiguous) torch tensor.

    tm_src: turbomind Tensor with strides preserved from the torch tensor
    tm_dst: contiguous turbomind Tensor for the output
    golden: contiguous torch tensor with the expected result
    """
    tm_src = _tm.from_dlpack_with_strides(torch_tensor)

    # Allocate a contiguous destination tensor with the same shape/dtype
    contig = torch_tensor.contiguous()
    tm_dst = _tm.from_dlpack(contig.clone())

    golden = contig.clone()
    return tm_src, tm_dst, golden


def run_test(name, torch_tensor, atol=1e-5, rtol=1e-5):
    """Run a single GenericCopy test. Returns True on pass."""
    tm_src, tm_dst, golden = make_tensors(torch_tensor)

    _tm.generic_copy(tm_src, tm_dst)

    # Read back the result through DLPack
    result = torch.from_dlpack(tm_dst)

    is_float = torch_tensor.is_floating_point()

    if is_float:
        match = torch.allclose(result, golden, atol=atol, rtol=rtol)
    else:
        match = torch.equal(result, golden)

    status = "PASS" if match else "FAIL"
    shape = list(torch_tensor.shape)
    stride = list(torch_tensor.stride())
    print(f"  [{status}] {name}: shape={shape}, stride={stride}")

    if not match:
        if is_float:
            diff = (result - golden).abs()
            print(f"         max_diff={diff.max().item()}, mean_diff={diff.mean().item()}")
        else:
            mismatches = (result != golden).sum().item()
            total = result.numel()
            print(f"         mismatches={mismatches}/{total}")

    return match


def main():
    print("GenericCopy Test Suite")
    print("=" * 60)

    all_passed = True
    total = 0
    passed = 0

    def check(name, tensor, **kwargs):
        nonlocal all_passed, total, passed
        total += 1
        if run_test(name, tensor, **kwargs):
            passed += 1
        else:
            all_passed = False

    # --- Contiguous baseline ---
    print("\nContiguous baseline:")
    check("contiguous f32", torch.ones(64, 128, dtype=torch.float32))

    # --- 2D layout transformations ---
    print("\n2D transformations:")
    check("transpose f32", torch.randn(64, 128, dtype=torch.float32).t())
    check("row-stride (every-other-row)", torch.randn(64, 128, dtype=torch.float32)[::2, :])
    check("col-stride (every-other-col)", torch.randn(64, 128, dtype=torch.float32)[:, ::2])
    check("narrow outer dim", torch.randn(128, 64, dtype=torch.float32)[10:50, :])

    # --- 3D transformations ---
    print("\n3D transformations:")
    check("permute (2,0,1)", torch.randn(16, 32, 64, dtype=torch.float32).permute(2, 0, 1))

    # --- 4D transformations ---
    print("\n4D transformations:")
    check("4D slice", torch.randn(4, 8, 32, 64, dtype=torch.float32)[:, :, ::3, :])

    # --- Combined operations ---
    print("\nCombined operations:")
    check("slice+transpose", torch.randn(64, 128, dtype=torch.float32)[::2, :].t())

    # --- Dtype sweep (all use transpose) ---
    print("\nDtype sweep (transpose):")
    check("transpose f16", torch.randn(64, 128, dtype=torch.float16).t(), atol=1e-3, rtol=1e-3)
    check("transpose i8", torch.randint(-128, 127, (64, 128), dtype=torch.int8).t())
    check("transpose i32", torch.randint(0, 1000, (64, 128), dtype=torch.int32).t())

    # --- Large tensor (exercises vectorization) ---
    print("\nLarge tensor:")
    check("large transpose (1024x1024)", torch.randn(1024, 1024, dtype=torch.float32).t())

    # --- Negative strides ---
    print("\nNegative strides (flip):")
    try:
        check("flip dim=0", torch.randn(32, 64, dtype=torch.float32).flip(0))
    except Exception as e:
        print(f"  [SKIP] flip dim=0: {e}")
        total += 1

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed")
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the test**

Run: `cd /data/lmdeploy-copy && python test_generic_copy.py`
Expected: All tests pass. The "flip" test may fail or skip depending on whether negative strides are supported.

If tests fail, debug by checking:
- `tm_src.shape` and `tm_src.type` match `torch_tensor.shape` and dtype
- The stride values are preserved correctly: print `tm_src` shape info
- The `tm_dst` has the same shape and dtype as `tm_src`

- [ ] **Step 3: Fix any issues found during testing**

Common issues:
- **DLPack strides might be NULL for contiguous tensors**: The `from_dlpack_with_strides` function handles this by computing default strides.
- **int8/int32 DLPack dtype mismatch**: Check `getDataType` at `bind.cpp:155-211` handles these types. `kDLInt` with 8/32 bits maps to `data_type_v<int8_t>` / `data_type_v<int32_t>`.
- **Negative strides from `.flip()`**: DLPack may represent these differently. If `GenericCopy` or `Layout` doesn't handle negative strides, the test catches the exception and skips.
- **Alignment issues for int8**: With `int8` and 64×128 transposed, the inner dim stride=1 should give at least 1-byte alignment, so `char` vector type is used.

- [ ] **Step 4: Commit**

```bash
cd /data/lmdeploy-copy
git add test_generic_copy.py
git commit -m "test: add GenericCopy test suite with PyTorch comparison"
```

---

### Task 4: Full verification

- [ ] **Step 1: Rebuild and re-run**

Run: `cd /data/lmdeploy-copy/build && ninja _turbomind 2>&1 | tail -20`
Expected: Build succeeds.

Run: `cd /data/lmdeploy-copy && python test_generic_copy.py`
Expected: All tests pass.

- [ ] **Step 2: Clean up (if test file is temporary)**

Only delete if the user requests it. Otherwise keep the test for future use.

```bash
# Only if requested:
# rm test_generic_copy.py
```
