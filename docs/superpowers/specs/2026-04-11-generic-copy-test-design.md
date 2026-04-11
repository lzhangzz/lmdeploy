# GenericCopy Test Suite Design

## Problem

The GenericCopy kernel was implemented in `tensor.cu` using CuTE but has no tests. We need a way to exercise it from Python and verify correctness against PyTorch ground truth for various non-contiguous layout patterns.

## Solution

Add stride-preserving DLPack conversion and a `generic_copy` Python binding, then write a standalone test script that creates non-contiguous PyTorch tensors, copies them through GenericCopy, and compares results.

## Scope

- **Python bindings:** Two new functions in `bind.cpp` — `from_dlpack_with_strides` and `generic_copy`
- **Test script:** Standalone `test_generic_copy.py` at repo root
- **No changes** to existing `from_dlpack`, `copy_from`, `Copy` dispatcher, or production code paths

## Architecture

### Files

| File | Change |
|---|---|
| `src/turbomind/python/bind.cpp` | Add `from_dlpack_with_strides` and `generic_copy` functions |
| `test_generic_copy.py` | New standalone test script |

### Binding: `from_dlpack_with_strides`

Like the existing `from_dlpack` but preserves DLPack strides when constructing the `core::Layout`:

1. Extract `shape` and `strides` from the `DLManagedTensor`
2. Convert DLPack strides from element-count to the format expected by `Layout` (if needed)
3. Construct `Layout(shape, strides)` instead of `Layout(shape)`
4. Construct `Tensor(shared_ptr<void>, Layout, dtype, device)`
5. Return as `shared_ptr<Tensor>`

If DLPack strides are NULL (contiguous tensor), compute default row-major strides.

### Binding: `generic_copy`

A module-level function:

1. Takes two `_turbomind.Tensor` objects (src and dst)
2. Validates `dtype` and `shape` match
3. Calls `core::GenericCopy(src, dst, core::Context::stream())`
4. Calls `core::Context::stream().Sync()` to ensure completion before returning to Python
5. Returns `None`

### Test Script: `test_generic_copy.py`

**Setup:**
- Set `PYTHONPATH` to find `lmdeploy` and `build/lib`
- Import `_turbomind` as `_tm`
- Import `torch`

**Helper: `make_tensors(torch_tensor) -> (tm_src, tm_dst, golden)`**
- `tm_src = _tm.from_dlpack_with_strides(torch_tensor)` — preserves non-contiguous strides
- Allocate `tm_dst` as contiguous tensor with same shape/dtype
- `golden = torch_tensor.contiguous().clone()` — PyTorch ground truth

**Helper: `run_test(name, torch_tensor) -> bool`**
- Call `make_tensors`
- Call `_tm.generic_copy(tm_src, tm_dst)`
- Convert `tm_dst` back to torch via `torch.from_dlpack(tm_dst)`
- Compare: `torch.allclose` for float types, `torch.equal` for integer types
- Print PASS/FAIL with shape and stride info

**Test cases:**

| # | Name | Construction | What it tests |
|---|---|---|---|
| 1 | Contiguous baseline | `torch.ones(64, 128)` | Simplest path, should match cudaMemcpy |
| 2 | Transpose 2D | `torch.randn(64, 128).t()` | Classic non-contiguous: row<->col |
| 3 | Row stride-slice | `torch.randn(64, 128)[::2, :]` | Outer dim stride > 1 |
| 4 | Col stride-slice | `torch.randn(64, 128)[:, ::2]` | Inner dim stride > 1 |
| 5 | Permute 3D | `torch.randn(16, 32, 64).permute(2, 0, 1)` | 3D layout shuffle |
| 6 | Slice 4D | `torch.randn(4, 8, 32, 64)[:, :, ::3, :]` | 4D, rank=4 path |
| 7 | Narrow dim | `torch.randn(128, 64)[10:50, :]` | Narrowed outer dim |
| 8 | Combined slice+transpose | `torch.randn(64, 128)[::2, :].t()` | Non-trivial strides from both ops |

**Dtype sweep:** Repeat test #2 (transpose) with `float16`, `int8`, `int32` to verify alignment handling.

**Large tensor:** `torch.randn(1024, 1024).t()` to exercise vectorization path (16-byte aligned).

**Negative strides / flip:** `torch.randn(32, 64).flip(0)` — if GenericCopy handles negative strides, test them; if not, skip with a note.

### How GenericCopy gets the right strides

The key flow:
1. PyTorch creates a non-contiguous tensor (e.g., `.t()` sets stride to `[1, 64]`)
2. `_tm.from_dlpack_with_strides(torch_tensor)` reads the DLPack strides and constructs `Layout(shape, strides)`
3. `_tm.generic_copy(src, dst)` calls `core::GenericCopy(src, dst, stream)`
4. GenericCopy reads `src.layout()` and `dst.layout()`, extracts strides, launches kernel
5. The kernel uses CuTE dynamic layouts with the exact strides for addressing

## Limitations

- GPU-only (GenericCopy is a CUDA kernel)
- No dtype conversion (src and dst must match)
- Negative strides from PyTorch `.flip()` may not work (depends on whether the Layout/DLPack path supports them)
- Test requires a GPU with sufficient memory
