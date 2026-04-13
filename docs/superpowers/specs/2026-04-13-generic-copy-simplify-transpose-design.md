# Design: Simplify TransposeCopyKernel

**Date:** 2026-04-13
**Status:** Approved

## Context

The current TransposeCopyKernel (commit 66e7eb5f) uses padded smem and a single smem view with auto-vectorized Phase 1 and scalar Phase 2. The padding and alignment dispatch add complexity without clear benefit since Phase 2 is already scalar. This simplification removes auto-vectorization, padding, and the alignment dispatch.

## Kernel Changes

### Smem: no padding, two naive views

Replace the padded `smem_tile` with a flat `kTileDim × kTileDim` buffer and two views:

- `smem_w`: shape `(kTileDim, kTileDim)`, stride `(1, kTileDim)` — row-major write view
- `smem_r`: shape `(kTileDim, kTileDim)`, stride `(kTileDim, 1)` — column-major read view

No padding. Both views share the same `__shared__ T smem[kTileDim * kTileDim]` buffer.

### Both phases scalar via kMaxVecBits

`kMaxVecBits` remains a template parameter on the kernel (always `8*sizeof(T)` from host). Both cooperative_copy calls use `kMaxVecBits`:

- Phase 1: `cooperative_copy<256, kMaxVecBits>(tid, src_tile, smem_w)` — gmem→smem, row→row
- Phase 2: `cooperative_copy<256, kMaxVecBits>(tid, smem_r, dst_tile)` — smem→gmem, col→col (transpose via view)

### What stays the same

- `zipped_divide` tiling and bounds checking
- 256 threads, `__launch_bounds__(256)`
- Grid: `(N/kTileDim, M/kTileDim)`
- Template signature: `kTileDim`, `kMaxVecBits`, engine/layout types

## Host Dispatch Changes

Remove the alignment computation and vec-bits switch:

- Remove `tr_alignment`, `max_vec_bits` computation
- Remove `tr_dispatch_vec` lambda and its 5-way switch
- `tr_dispatch_elem_size` directly launches the kernel with `kMaxVecBits = 8 * sizeof(T)`

## Files Changed

- `src/turbomind/core/tensor.cu` — kernel body + host dispatch
