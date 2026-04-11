# GenericCopy with CuTE Design

## Problem

The `core::Copy(Tensor, Tensor)` function only handles contiguous tensors (via `cudaMemcpyAsync`). A disabled (`#if 0`) implementation in `tensor.cu` and `tensor.cc` attempted to handle non-contiguous copies but was abandoned with bugs (uninitialized variable in occupancy tuning, incomplete rank dispatch). We need a working generic copy that handles arbitrary stride patterns.

## Solution

Replace the disabled GenericCopy with a CuTE-based implementation. Construct CuTE tensors with dynamic (runtime) layouts from `core::Layout` shape/stride, and use CuTE's `copy()` which auto-vectorizes along stride-1 dimensions.

## Scope

- **Purpose:** General Tensor Copy for arbitrary non-contiguous stride patterns
- **No type conversion** — src and dst must have the same dtype
- **SM target:** SM70+ (Volta and later)
- **Max rank:** 6 dimensions
- **CuTE usage:** Copy primitives only (no layout system replacement)

## Architecture

### Files Modified

| File | Change |
|---|---|
| `src/turbomind/core/tensor.cu` | Rewrite `#if 0` block with CuTE-based GenericCopy |
| `src/turbomind/core/tensor.cc` | Re-enable Copy dispatcher outside `#if 0` |
| `src/turbomind/core/tensor.h` | Uncomment `GenericCopy` declaration |
| `src/turbomind/core/CMakeLists.txt` | Add `nvidia::cutlass::cutlass` to core link deps |

### Kernel: `GenericCopyKernel<VecT, IndexT, kRank>`

A rank-templated CUDA kernel that:

1. Receives raw shape/stride arrays and pointers
2. Constructs CuTE tensors with dynamic layouts:
   ```
   auto gA = make_tensor((VecT*)src, make_layout(shape_tuple, stride_tuple));
   auto gB = make_tensor((VecT*)dst, make_layout(shape_tuple, stride_tuple));
   ```
3. Each thread maps to one element of the flattened iteration space (threadIdx + blockIdx * blockDim)
4. Converts linear thread index to a multi-dim coordinate using the shape
5. Uses CuTE tensor's `operator()` with the coordinate to load/store with stride-aware addressing
6. CuTE's `make_tensor` with dynamic layout handles the address arithmetic internally

**Vectorization:** The innermost dimension with stride=1 is vectorized by choosing `VecT`:
- `uint4` (16 bytes) when 16-byte aligned
- `uint2` (8 bytes) when 8-byte aligned
- `uint` (4 bytes) when 4-byte aligned
- `ushort` (2 bytes) when 2-byte aligned
- `char` (1 byte) otherwise

When vectorizing, shape[innermost] and strides are divided by `vec_size`.

**Index type:** `int32_t` when total elements fit, `int64_t` otherwise.

**Rank specialization:** Template on `kRank ∈ {2, 4, 6}`. Host pads shape/stride with 1s to fit the nearest rank bucket.

### Host Dispatcher: `GenericCopy(src, dst, stream)`

1. **Normalize layouts:**
   - Sort dimensions by stride ascending
   - Coalesce adjacent contiguous dimensions
   - Pad to target rank bucket (2/4/6)

2. **Determine alignment:**
   - Start at 16 bytes
   - Reduce by GCD with: pointer alignment of src and dst, element alignment of innermost dim shape, stride alignment of outer dims
   - If no stride-1 dimension exists, reduce to element size

3. **Select VecT:** Based on alignment (16→uint4, 8→uint2, 4→uint, 2→ushort, 1→char)

4. **Select IndexT:** int32 if `size / vec_size < INT_MAX`, else int64

5. **Select kRank:** smallest of {2, 4, 6} ≥ normalized rank

6. **Launch kernel:** Grid size = `ceil(size / vec_size / block_size)`, block size chosen for occupancy

### Copy Dispatcher: `Copy(Tensor src, Tensor dst, Stream)`

Fast-path hierarchy (in `tensor.cc`):

1. Both contiguous → `cudaMemcpyAsync`
2. After stride-sort, both contiguous → `cudaMemcpyAsync`
3. Innermost dim not contiguous → GenericCopy
4. 2D with one stride-1 dim → `cudaMemcpy2DAsync`
5. 3D with regular cube structure → `cudaMemcpy3DAsync`
6. Fallback → GenericCopy

### CMake Integration

```cmake
target_link_libraries(core PUBLIC ... nvidia::cutlass::cutlass)
```

CUTLASS is headers-only, so this only propagates include paths. No additional compiled objects.

## Limitations

- No type conversion (src.dtype must equal dst.dtype)
- src.shape must equal dst.shape (broadcasting not supported)
- Maximum 6 dimensions
- Sub-byte types (fp4, uint2, uint4, uint6) not vectorized — always scalar access
- No async copy (cp.async) for SM80+ — could be a future enhancement
