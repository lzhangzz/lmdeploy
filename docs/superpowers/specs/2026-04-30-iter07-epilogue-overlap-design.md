# Iteration 07: Epilogue-Compute Overlap

## Goal

Hide the TMA store latency in the epilogue by deferring `tma_store_wait<0>()` to the next tile's epilogue, allowing the TMA store for tile N to overlap with tile N+1's compute (consumer_wait + S2R + WGMMA).

## Background

Iter 06 achieves ~679 TFLOP/s at 4096^3 (85.5% of cuBLAS) and ~672 TFLOP/s at 8192^3 (100.4% of cuBLAS). The consumer's epilogue stalls all 256 consumer threads at `tma_store_wait<0>()` while the TMA store completes. This stall is serial — no compute happens during it. By deferring the wait, the TMA store latency overlaps with the next tile's compute.

## Scope

Only the consumer's epilogue wait placement changes. One ~5 line change in the WGMMA kernel.

### Files

1. **`07_split_a_pack.h`** — thin wrapper including `06_split_a_pack.h`. Pack kernel unchanged.
2. **`07_bf16_gemm_sm90_split_a_pack.cu`** — pack test, updated includes/labels.
3. **`07_bf16_gemm_sm90_split_a_wgmma.cu`** — WGMMA kernel with deferred TMA store wait (only file with substantive changes).
4. **`CMakeLists.txt`** — add 07 targets.

## Change Description

### Current consumer epilogue (per tile)

```cpp
// Scale, convert, r2s, sync
cutlass::arch::NamedBarrier consumer_sync(256, 6);
consumer_sync.sync();

if (threadIdx.x == 0) {
  tma_store_fence();
  copy(tma_store_c, tSsC(_, 0), tSgC(_, rest_idx));
  tma_store_arrive();
}
tma_store_wait<0>();  // STALL: all 256 consumer threads block until TMA store completes
```

### New consumer epilogue (per tile)

```cpp
tma_store_wait<0>();  // Wait for PREVIOUS tile's TMA store to complete (no-op for first tile)

// Scale, convert, r2s, sync
cutlass::arch::NamedBarrier consumer_sync(256, 6);
consumer_sync.sync();

if (threadIdx.x == 0) {
  tma_store_fence();
  copy(tma_store_c, tSsC(_, 0), tSgC(_, rest_idx));
  tma_store_arrive();
}
// NO tma_store_wait here — TMA store overlaps with next tile's compute
```

### Timeline

```
Current (iter 06):                     New (iter 07):
Tile N:                                Tile N:
  compute (S2R + WGMMA)                  compute (S2R + WGMMA)
  epilogue: r2s, TMA store issue         epilogue: tma_store_wait<0>() ← no-op (first)
  tma_store_wait<0>() ← STALL           r2s, TMA store issue
                                        (no wait)
Tile N+1:                              Tile N+1:
  consumer_wait                          consumer_wait ← TMA store N in flight
  compute (S2R + WGMMA)                  compute (S2R + WGMMA) ← overlapped with store N
  epilogue: r2s, TMA store issue         epilogue: tma_store_wait<0>() ← store N done
  tma_store_wait<0>() ← STALL           r2s, TMA store issue
                                        (no wait)
```

### Correctness guarantees

1. **sC not corrupted**: The `tma_store_wait<0>()` at the start of each tile's epilogue ensures the previous tile's TMA store completed before r2s overwrites sC. C smem (`sC`) is separate from A/B smem, so the next tile's compute (S2R + WGMMA) doesn't conflict.

2. **First tile**: `tma_store_wait<0>()` is a no-op (no pending stores).

3. **Last tile**: The producer's existing `tma_store_wait<0>()` at the end of its loop (line 160) ensures the last tile's TMA store completes before kernel exit.

4. **Skipped tiles** (from swizzle padding): The `tma_store_wait<0>()` is inside the epilogue, which is only reached for valid tiles. Skipped tiles `continue` before reaching it, so the wait carries forward to the next valid tile's epilogue.

## What Doesn't Change

- Pack kernel — unchanged
- Producer mainloop (bulk copy A, TMA B, pipeline stages) — unchanged
- Consumer mainloop (k_block interleaving, delayed release, try_wait prefetch) — unchanged
- Pipeline depth (3 stages), smem layout, packed format (64x16 units)
- Epilogue arithmetic (scale, convert, r2s, sync, TMA store issue) — unchanged
- Host function — unchanged (no new kernel parameters)
- Swizzle (default log_swizzle=0, effectively row-major)

## Validation

Same correctness tests as iter 06 (single tile through 2048x1024x512). Performance benchmark at 256..8192 with cuBLAS reference. Target: reduce or eliminate the per-tile epilogue stall.
