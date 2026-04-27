# Sample 12: Optimized RS GEMM with k_block Double-Buffering

## Problem

Sample 11 (RS variant) has a 4-8% performance gap vs sample 09 (SS variant). Root cause: the S2R copy of operand A is on the critical path with no overlap with WGMMA compute.

## Current vs Target Mainloop

### Sample 11 (current, serial):
```
for each k_tile:
    wait(pipeline)
    copy(A[entire K=64])       <- blocks until complete
    gemm(A, B)                 <- 4 WGMMA calls, fully serial
    warpgroup_wait<0>()        <- blocks until all done
    release(pipeline)
```

### Sample 12 (target, double-buffered):
```
for each k_tile:
    for each k_block (0..3):
        if k_block == 0:  try_wait(next stage)
        if k_block == last:
            wait(next stage)
            prefetch(A[k0] of next stage)
        else:
            copy(A[k_block+1])   <- overlaps with current WGMMA
        gemm(A[k_block], B[k_block])
        warpgroup_wait<2>()      <- allows 2 in-flight GMMAs
        if k_block == 1:
            release(pipeline)    <- early release
```

## What Changes

Only the **consumer mainloop**. Everything else (producer, epilogue, fragments, tile sizes) is identical to sample 11.

### Fragment setup additions

CUTLASS creates a separate tiled copy for the S2R path:
```cpp
auto smem_tiled_copy_A = make_tiled_copy_A(
    Copy_Atom<AutoVectorizingCopy, bf16_t>{}, mma);
auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(threadIdx.x);
Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);   // (CPY, CPY_M, CPY_K)
Tensor tCsA_copy_view = smem_thr_copy_A.partition_S(sA);   // (CPY, CPY_M, CPY_K, PIPE)
```

This enables k_block-by-k_block copies with proper vectorization:
```cpp
copy(smem_tiled_copy_A,
     tCsA_copy_view(_,_,k_block+1, read_stage),
     tCrA_copy_view(_,_,k_block+1));
```

### Key optimizations

1. **Copy/compute overlap**: S2R copy for k_block+1 overlaps with WGMMA for k_block
2. **warpgroup_wait<2>**: Allows 2 in-flight WGMMA instructions (vs <0> which blocks)
3. **Early pipeline release**: Producer gets stage back at k_block==1
4. **Pipeline prefetch**: First k_block of next stage copied while last k_block computes
5. **Early try_wait**: Pipeline barrier probe starts at k_block==0

## Register Budget

Identical to sample 11. The A fragment `tCrA` is `(MMA, MMA_M, MMA_K)` without PIPE. The k_block loop overwrites the same registers each iteration. The `tCrA_copy_view` is a retiled view into the same register space.

## Expected Outcome

Close the gap from 4-8% to ~1-2% (residual being inherent S2R overhead that SS avoids).

## CUTLASS Reference

`build/_deps/repo-cutlass-src/include/cutlass/gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized.hpp`
- Lines 564-568: tiled copy setup
- Lines 617-628: prologue k_block loop
- Lines 649-691: main k_tile loop with k_block inner loop and prefetch

## File Structure

- Source: `cute-reference/samples/12_bf16_gemm_sm90_pipe_tma_ws_persistent_rs_opt.cu`
- Based on: sample 11 (copy as starting point)
- Build: add to `cute-reference/samples/CMakeLists.txt`
