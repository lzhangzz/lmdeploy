# GenericCopy with CuTe TiledCopy Design

## Problem

The GenericCopy kernel in `tensor.cu` uses plain CUDA with manual stride arithmetic and a 1-element-per-thread mapping. This performs well for contiguous copies but underperforms on pathological stride patterns: non-contiguous innermost dimensions (no stride-1 dim after sorting) and high-rank tensors (5-6D). We need to adopt CuTe's TiledCopy for structured thread-to-element mapping and auto-vectorization.

## Solution

Replace the plain-CUDA `GenericCopyKernel` with a CuTe TiledCopy-based kernel. Each CTA (thread block) handles one "row" of the outer dimensions. Within the CTA, TiledCopy distributes the innermost dimension across 256 threads, and `cute::copy()` auto-vectorizes each thread's contiguous chunk.

## Scope

- **Purpose:** Improve GenericCopy performance for pathological stride patterns
- **SM target:** SM70+ (Volta and later)
- **Max rank:** 6 dimensions (unchanged)
- **No type conversion** — src and dst must have the same dtype
- **Fixed block size:** 256 threads (compile-time constant for TiledCopy)
- **CuTe usage:** `make_tensor`, `make_layout`, `TiledCopy`, `Copy_Atom`, `copy()`

## Architecture

### Files Modified

| File | Change |
|---|---|
| `src/turbomind/core/tensor.cu` | Rewrite GenericCopyKernel to use CuTe TiledCopy |
| `src/turbomind/core/CMakeLists.txt` | Add `nvidia::cutlass::cutlass` to core link deps |

### Kernel: `GenericCopyKernel<VecT, kRank>`

A rank-templated CUDA kernel with TiledCopy-structured thread mapping.

**CTA-level mapping:**
- `blockIdx.x` maps to a linear index over the outer dimensions (dims 1..kRank-1)
- The linear index is decomposed to multi-dim coordinates at runtime
- Each CTA handles one "row" (one set of outer-dim coordinates)

**Thread-level mapping via TiledCopy:**
- The innermost dimension (dim 0, stride-1 after sorting) is distributed across threads
- `TiledCopy` with `Copy_Atom<UniversalCopy<VecT>, VecT>` defines the copy atom
- Thread layout: 256 threads along innermost dim (compile-time)
- `cute::copy()` auto-vectorizes each thread's contiguous elements

**Kernel structure:**

```
1. Build CuTe tensors with dynamic multi-dim layouts
   auto gSrc = make_tensor(src_ptr, make_layout(shape_tuple, src_stride_tuple));
   auto gDst = make_tensor(dst_ptr, make_layout(shape_tuple, dst_stride_tuple));

2. Map CTA to outer-dim coordinate
   outer_idx = blockIdx.x;
   if (outer_idx >= outer_total) return;
   outer_coord = decompose(outer_idx, shape[1..kRank-1]);

3. Extract CTA's inner-dim slice (1D sub-tensor via CuTe tensor indexing)
   auto cta_src = gSrc(outer_coord, _);  // 1D tensor along inner dim
   auto cta_dst = gDst(outer_coord, _);

4. TiledCopy partitions inner dim across threads
   auto tiled_copy = make_tiled_copy(
       Copy_Atom<UniversalCopy<VecT>, VecT>{},
       Layout<Shape<Int<256>>, Stride<_1>>{},
       Layout<Shape<_1>, Stride<_1>>{});
   auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
   auto thr_src = thr_copy.partition_S(cta_src);
   auto thr_dst = thr_copy.partition_D(cta_dst);

5. Auto-vectorized copy
   copy(tiled_copy, thr_src, thr_dst);
```

**Vectorization:** VecT is selected on the host (same as current: uint4/uint2/uint/ushort/char based on alignment). The TiledCopy's Copy_Atom is parameterized on VecT via the kernel template.

**Rank specialization:** Template on kRank in {1, 2, 3, 4, 5, 6}. Host dispatches the exact rank — no padding needed. Each rank generates a specialized kernel with optimal coordinate decomposition for that rank.

**Bounds handling:** CuTe's `copy()` handles boundary elements correctly when inner_size is not a multiple of 256. The CTA-level check `blockIdx.x < outer_total` skips out-of-bounds CTAs.

### Host Dispatcher: `GenericCopy(src, dst, stream)`

The host dispatcher logic is mostly unchanged:

1. **Normalize layouts** (unchanged): Sort dimensions by stride ascending, coalesce
2. **Determine alignment** (unchanged): GCD-reduce pointer alignment, shape alignment, stride alignment
3. **Select VecT** (unchanged): Based on alignment
4. **Compute inner/outer split** (new):
   - `inner_size = shape[0]` (after vec division)
   - `outer_total = product(shape[1:])`
5. **Launch kernel** (changed):
   - Grid size = `outer_total`
   - Block size = 256 (fixed)
   - Pass `inner_size` and `outer_total` to kernel

### CMake Integration

```cmake
target_link_libraries(core PUBLIC ... nvidia::cutlass::cutlass)
```

CUTLASS is headers-only, so this only propagates include paths. No additional compiled objects.

## Performance Expectations

**Common case (contiguous innermost dim):** Similar to current kernel. TiledCopy auto-vectorizes along stride-1 dim with the same VecT width. Possible slight improvement from better instruction scheduling by CuTe's copy atoms.

**Pathological case — non-contiguous innermost dim:** TiledCopy falls back to scalar Copy_Atom (1-byte access). Similar to current kernel's scalar fallback. The benefit is from CuTe's structured thread mapping ensuring coalesced memory access within warps.

**Pathological case — high-rank 5-6D:** Improvement expected. The outer-dim coordinate decomposition is done once per CTA (amortized across 256 threads), not per thread as in the current kernel. This reduces per-element address computation overhead.

## Limitations

- No type conversion (src.dtype must equal dst.dtype)
- src.shape must equal dst.shape (broadcasting not supported)
- Maximum 6 dimensions
- Fixed block size 256 (not occupancy-tuned)
- Sub-byte types (fp4, uint2, uint4, uint6) not vectorized — always scalar access
- No async copy (cp.async) for SM80+ — future enhancement
- No TMA for SM90+ — future enhancement

## Future Upgrade Path

- **SM80+:** Swap `Copy_Atom<UniversalCopy<VecT>, VecT>` for `Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<VecT>, VecT>` — same TiledCopy structure, different atom
- **SM90:** Use TMA-based TiledCopy for 2D+ bulk async copies — different atom, same kernel structure
- **Variable block size:** Template on kBlockThreads, let host select based on occupancy
