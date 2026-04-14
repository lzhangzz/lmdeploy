# GenericCopy: CuTe-First Layout Algebra

**Date**: 2026-04-14
**Status**: Design approved

## Problem

`GenericCopy` does layout normalization (stride-sort, coalesce, rank-match) on
`core::Layout` before converting to `cute::Tensor`. This duplicates layout algebra
logic and prevents composable CuTe operations from being applied early.

## Goal

Convert `core::Tensor` to `cute::Tensor` at the entry point, then perform all
layout algebra using CuTe operations. Eliminate `core::Layout` algebra from
`GenericCopy` entirely.

## Approach

Pad all tensors to rank 6, build `cute::Tensor` immediately, then simplify using
CuTe layout algebra. Dispatch on the simplified rank only for kernel launch.

## Design

### 1. Entry Conversion: core::Tensor → cute::Tensor

A helper `to_cute_tensor<T>(core::Tensor)` that:

1. Extracts `raw_data()`, `shape()`, `stride()` from `core::Tensor`
2. **Reverses** mode order (turbomind convention: mode 0 = outermost; CuTe
   convention: mode 0 = innermost)
3. Pads to rank 6: appends `(size=1, stride=0)` trailing modes
4. Returns `cute::Tensor` with compile-time rank 6

Both `src` and `dst` are converted this way. The stride-sort (step 2 below)
applies the same permutation (from src strides) to both.

**Why rank 6**: Covers NCHW+batch (5D) with one spare. Coalesce removes padding
dims naturally. Template bloat is minimal at rank 6.

### 2. CuTe Layout Algebra Pipeline

After entry conversion, both tensors are `cute::Tensor` at rank 6.

**Step 1 — Stride-sort (`sort_modes_by_stride`)**:
- Extract 6 stride values into a runtime array
- Sort indices ascending by stride value (derived from src)
- Apply the **same permutation** to both src and dst tensors
- Rebuild CuTe layouts from sorted shape/stride via
  `detail::make_cute_layout<6>()`

This is the only custom helper needed — CuTe has no runtime permutation.
Since rank is fixed at 6, the helper is trivial.

**Step 2 — Coalesce (CuTe built-in)**:
- `cute::coalesce(src_tensor)` and `cute::coalesce(dst_tensor)` merge
  adjacent contiguous modes
- Size-1 padding modes (stride 0) are absorbed automatically
- Result is a `cute::Tensor` with effective rank ≤ 6

**Step 3 — Determine effective rank**:
- Count modes where `size > 1` (non-trivial)
- Expected effective rank: 1–4 after coalescing

**Step 4 — Rank dispatch for kernel launch**:
- `switch (effective_rank)` dispatches over 1–4
- Inside each case: group modes, create TiledCopy, launch kernel
- This is the narrow bridge where runtime rank becomes compile-time rank

### 3. Transpose Path Integration

Transpose detection operates on the coalesced `cute::Tensor`:

- Effective rank == 2
- `stride<0>(src) == 1` (src contiguous on innermost dim)
- `stride<1>(dst) == 1` (dst contiguous on outermost — transposed)
- Shapes divisible by `kTileDim`

When detected, dispatch directly to `TransposeCopyKernel` using the already-
constructed cute::Tensors. No separate layout construction needed.

### 4. Alignment and Vectorization

Alignment detection stays conceptually the same, reading from cute::Tensor:

- Inner stride check: `stride<0>(src) == 1`
- Pointer alignment: `reinterpret_cast<uintptr_t>(tensor.data())`
- Outer stride alignment: `stride<I>(tensor)` for outer modes
- Shape divisibility: `size<0>(src) % (vec_size * kBlockThreads)`

### 5. Code Structure

```
// New helper: to_cute_tensor<T>(core::Tensor) -> cute::Tensor (rank 6)
// New helper: sort_modes_by_stride(src, dst) -> (sorted_src, sorted_dst)

void GenericCopy(const Tensor& src, Tensor& dst, cudaStream_t stream) {
    // 1. Dispatch over element type T
    //    Inside each T:
    //      a. to_cute_tensor<T>(src), to_cute_tensor<T>(dst)  -- rank 6
    //      b. sort_modes_by_stride(src_cute, dst_cute)        -- same perm
    //      c. coalesce(src_cute), coalesce(dst_cute)          -- CuTe built-in
    //      d. Count effective rank
    //      e. Transpose check (on coalesced tensors)
    //      f. Alignment + vec_size
    //      g. Dispatch(rank, vec_size) → group, tile, launch
}
```

### What Gets Eliminated

- `core::Layout::permute` usage in GenericCopy
- `core::Layout::coalesce` usage in GenericCopy
- `core::Layout::view` usage in GenericCopy
- Manual stride-sort logic on `core::Layout`
- Manual rank-matching logic
- Separate layout construction for transpose path

### What Stays

- `TransposeCopyKernel` — unchanged, already takes cute::Tensor
- `CopyKernelND` — unchanged, already takes cute::Tensor
- `detail::make_cute_layout` — reused for stride-sort rebuild
- Byte-size dispatch (uint8/16/32/64)
- Alignment detection logic (reads from cute::Tensor instead of core::Layout)
