# Iteration 06: Threadblock Swizzling

## Goal

Close the 4096^3 performance gap (86.2% of cuBLAS vs 97.6% at 8192^3) by swizzling the persistent kernel's tile mapping to improve L2 cache locality for operand B.

## Background

Iter 05 achieves 679 TFLOP/s at 4096^3 (86.2% of cuBLAS) but 676 TFLOP/s at 8192^3 (97.6%). The gap at 4096^3 is consistent with L2 cache thrashing — without swizzling, consecutive CTAs in the grid-stride loop map to adjacent M-tiles that share L2 cache lines for operand B. CUTLASS's persistent kernels solve this via the tile scheduler's swizzle logic.

## Scope

Only the WGMMA kernel's tile mapping changes. The pack kernel, consumer mainloop, pipeline, smem layout, packed format, and epilogue are all unchanged from iter 05.

### Files

1. **`06_split_a_pack.h`** — thin wrapper including `05_split_a_pack.h`. Pack kernel unchanged.
2. **`06_bf16_gemm_sm90_split_a_pack.cu`** — identical to iter 05 pack test, references 06 header.
3. **`06_bf16_gemm_sm90_split_a_wgmma.cu`** — tile mapping restructured with swizzle (only file with substantive changes).
4. **`CMakeLists.txt`** — add `06_bf16_gemm_sm90_split_a_pack` and `06_bf16_gemm_sm90_split_a_wgmma` targets.

## Swizzle Mapping

Replace the row-major mapping (`m = linear / n_tiles`, `n = linear % n_tiles`) in both producer and consumer with a CUTLASS-style bit-decomposition swizzle, simplified for single-CTA clusters (cluster_shape = 1):

```cpp
auto swizzle_tile = [&](uint64_t linear_idx) -> cute::tuple<int, int> {
  int offset = linear_idx & (swizzle_size - 1);
  int extra  = linear_idx >> log_swizzle;
  int n_tile = extra % n_tiles;
  int m_tile = (extra / n_tiles) * swizzle_size + offset;
  return {m_tile, n_tile};
};
```

This interleaves consecutive CTAs across the M dimension while keeping N shared within swizzle groups. For swizzle_size=8, CTAs 0-7 all process different M tiles at the same N position, maximizing B reuse in L2.

Bounds check after mapping — skip tiles where `m_tile >= m_tiles || n_tile >= n_tiles`.

## Swizzle Heuristic

Adapted from CUTLASS `PersistentTileSchedulerSm90Params::get_log_swizzle_size()`:

| Condition | log_swizzle | swizzle_size |
|-----------|-------------|--------------|
| min(m_tiles, n_tiles) >= 6 | 3 | 8 |
| min(m_tiles, n_tiles) >= 3 | 2 | 4 |
| min(m_tiles, n_tiles) >= 2 | 1 | 2 |
| otherwise | 0 | 1 (no swizzle) |

Computed on the host side, passed to the kernel as arguments.

## Grid Launch

Pad `total_tiles` up to a multiple of `swizzle_size` (cluster_shape = 1). Extra tiles from padding are skipped at runtime by the bounds check. Grid size remains `min(num_SMs, total_tiles_padded)`.

## What Doesn't Change

- Producer warp group internals (bulk copy A, TMA B) — only the (m_idx, n_idx) inputs change
- Consumer mainloop (k_block interleaving, delayed release, try_wait prefetch) — identical to iter 05
- Pipeline depth (3 stages), smem layout, packed format (64x16 units)
- Pack kernel — unchanged
- Epilogue (scaling, BF16 convert, r2s, TMA store) — unchanged
- Test/benchmark harness — unchanged

## Validation

Same correctness tests as iter 05 (single tile through 2048x1024x512). Performance benchmark at 256..8192 with cuBLAS reference. Target: close the gap at 4096^3 from 86.2% toward ~95%+.

## Reference

CUTLASS `sm90_tile_scheduler.hpp` `get_work_idx_m_and_n()` and `tile_scheduler_params.h` `get_log_swizzle_size()`.
