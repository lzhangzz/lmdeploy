# BF16 GEMM SM80 Pipe — STSM Epilogue with BF16 Output

## Overview

Add a BF16 epilogue to the existing pipelined BF16 GEMM that uses SM90 STSM (store-to-shared-memory
matrix) instructions to rearrange accumulator data before writing to global memory with vectorized
128-bit stores. This demonstrates the register → smem → gmem epilogue pipeline pattern used in
production CUTLASS kernels.

Based on `bf16_gemm_sm80_pipe_256x128.cu`. New file: `bf16_gemm_sm80_pipe_epilogue.cu`.

## What Changes

The MMA loop (Steps 1–5) is completely unchanged. Only the epilogue (Step 6) and host function
configuration change.

### C Output Type Change

C is now **BF16** instead of F32:
- `float* C` → `bf16_t* C` in both kernel and host function
- `ldC` still in elements (M for column-major)
- Verification must account for BF16 precision (use relative tolerance)

### Epilogue Pipeline (replaces direct axpby)

The current epilogue writes F32 directly to global memory:
```
axpby(alpha, tCrC, beta, tCgC)   // F32 register → F32 global (read-modify-write)
```

The new epilogue is a 3-stage pipeline:

**Stage 1: Load and scale (F32 registers, element-wise)**
- Partition gC (BF16) using MMA thread slice: `tCgC = thr_mma.partition_C(gC)`
- Element-wise load, convert, and scale (avoids allocating a full F32 register tensor for C):
```cpp
CUTE_UNROLL
for (int i = 0; i < size(tCrC); ++i) {
    tCrC(i) = alpha * tCrC(i) + beta * static_cast<float>(tCgC(i));
}
```
- This keeps register pressure at 128 (accum) + ~20 (other) = ~148 regs

**Stage 2: Convert and STSM (F32 registers → BF16 smem)**
- Convert F32 accumulators to BF16: `tCrC_bf16 = convert<bf16_t>(tCrC)`
- Create sC smem tensor reusing sA's buffer (64 KB fits in 96 KB)
- R2S TiledCopy: `make_tiled_copy_C(Copy_Atom<SM90_U16x8_STSM_T, bf16_t>{}, mma)`
- Retile BF16 registers for STSM layout: `thr_r2s.retile_S(tCrC_bf16)`
- Partition sC as destination: `thr_r2s.partition_D(sC)`
- Execute: `copy(r2s_tiled, tCrC_bf16_retiled, tCsC)`
- `__syncthreads()` — ensure all STSM writes are visible

**Stage 3: Vectorized S2G (BF16 smem → BF16 global)**
- S2G TiledCopy: `make_tiled_copy(AutoVec128, thread_layout, value_layout)`
- Thread layout: `Layout<Shape<_32, _8>, Stride<_8, _1>>` — 32 threads M, 8 threads N
- Value layout: `Layout<Shape<_8, _1>>` — 8 bf16 per store (128 bits), contiguous in M
- Coverage per copy-tile: (256, 8). Loops 16 times in N.
- Partition sC as source and gC as destination
- Execute: `copy(s2g_tiled, tSsC, tSgC)`

### Host Function Changes

1. **C pointer type**: `float* C` → `bf16_t* C`
2. **sC layout**: now actually used (was unused placeholder)
3. **R2S TiledCopy**: `make_tiled_copy_C(Copy_Atom<SM90_U16x8_STSM_T, bf16_t>{}, mma)`
4. **S2G TiledCopy**: `make_tiled_copy(AutoVec128, Layout<Shape<_32, _8>, Stride<_8, _1>>{}, Layout<Shape<_8, _1>>{})`
5. **sC smem layout**: `make_layout(make_shape(bM, bN))` — (256, 128) column-major, 64 KB
6. **Template parameters**: kernel gains R2S and S2G TiledCopy parameters, loses CSmemLayout unused placeholder

### Kernel Signature Changes

The kernel template gains two new parameters:
- `R2SCopy` — TiledCopy for register → smem (STSM)
- `S2GCopy` — TiledCopy for smem → global (vectorized stores)

The kernel's `TC` template parameter changes from `float` to `bf16_t`.

## STSM Atom Selection

Our C is column-major: `dC = make_stride(Int<1>{}, ldC)` with stride-1 in M.

Following CUTLASS's `sm90_get_smem_store_op_for_accumulator` selection logic:
- `sizeof(bf16_t) == 2` and `size<0>(Stride<Int<1>, int>{}) == 1` → `SM90_U16x8_STSM_T`

`SM90_U16x8_STSM_T` issues `stmatrix.sync.aligned.x4.trans.m8n8.shared.b16`:
- 32 threads (1 warp) cooperate to write an 8×8 transposed matrix
- Each thread provides 4 × uint32 (8 bf16 values)
- Transposed store rearranges from MMA accumulator layout to column-major smem layout

## Smem Reuse Strategy

After MMA completes, sA (96 KB) and sB (48 KB) are no longer needed.

sC needs 256 × 128 × 2 bytes = 64 KB, which fits within sA's 96 KB space.

Implementation:
```cpp
// After MMA loop completes:
Tensor sC = make_tensor(
    make_smem_ptr(reinterpret_cast<bf16_t*>(smem.A.begin())),
    make_layout(make_shape(bM, bN)));  // (256, 128) column-major
```

No additional smem allocation. Occupancy unchanged (1 block/SM × 256 threads).

## S2G Vectorized Store Details

Thread layout `Layout<Shape<_32, _8>, Stride<_8, _1>>`:
- 32 threads in M dimension, 8 threads in N dimension = 256 threads total
- Coordinate (m, n) → thread index = m * 8 + n
- Thread (m, n) covers logical position m in M and position n in N
- Value layout `Layout<Shape<_8, _1>>`: each thread stores 8 contiguous bf16 along M (128 bits)

Coverage per copy-tile: M = 32 × 8 = 256, N = 8 × 1 = 8. Loops 128/8 = 16 times in N.

Each thread stores 8 consecutive bf16 values along M (stride-1 in column-major smem/gmem),
producing coalesced 128-bit global memory stores.

## Register Pressure Analysis

The epilogue runs after the MMA loop, so A/B register fragments are freed (~16 regs).

| Component | Registers |
|-----------|-----------|
| C accumulators (F32) | 128 regs |
| C loaded from gmem | 0 (element-wise inline, no register tensor) |
| C converted to BF16 | 64 regs (after F32 accumulators freed) |
| Other | ~20 regs |
| **Peak** | **~148 regs** (128 accum + 20 other during element-wise scale) |

Peak is well within the L20Y max of 255 regs/thread. The element-wise inline load+scale
avoids allocating a full 128-register F32 tensor for the loaded C values.

## What Stays the Same

- MMA atom: `SM80_16x8x16_F32BF16BF16F32_TN`
- Atom layout: `Layout<Shape<_4, _2>>` (8 warps, 256 threads)
- Tile override: `Tile<Underscore, _64, Underscore>`
- CTA tile: (256, 128, 64)
- Pipeline depth: 3 stages
- G2S TiledCopy: `Layout<Shape<_32, _8>, Stride<_8, _1>>` with `SM80_CP_ASYNC_CACHEALWAYS<uint128_t>`
- S2R atoms: `SM75_U32x4_LDSM_N`
- Pipeline logic, main loop, prefetch, drain — all unchanged
- Swizzle pattern: `Swizzle<3, 3, 3>` for sA and sB

## File Layout

```
cute-reference/samples/bf16_gemm_sm80_pipe_epilogue.cu   # new file
cute-reference/samples/build_and_run.sh                   # updated: add new sample
```

## Testing

- Verify correctness at 1024³ with BF16 C (tolerance adjusted for BF16 precision)
- Benchmark at 512³, 1024³, 2048³, 4096³, 8192³
- Compare performance with baseline (F32 output, direct axpby)
- Expect: slightly lower TFLOP/s due to smem staging overhead, but demonstrating the STSM pattern
