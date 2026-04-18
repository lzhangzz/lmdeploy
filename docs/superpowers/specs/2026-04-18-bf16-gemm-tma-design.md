# BF16 GEMM SM80 Pipe — TMA Load/Store

## Overview

Add TMA (Tensor Memory Accelerator) load and store to the existing pipelined BF16 GEMM,
replacing cp.async G2S with TMA load and the STSM + vectorized S2G epilogue with TMA store.
The kernel keeps SM80 HMMA tensor cores and LDSM S2R unchanged, demonstrating incremental
TMA adoption on top of the existing pipeline structure.

Based on `bf16_gemm_sm80_pipe_epilogue.cu`. New file: `bf16_gemm_sm80_pipe_tma.cu`.

## What Changes

### G2S: cp.async → TMA Load

The current G2S uses `SM80_CP_ASYNC_CACHEALWAYS<uint128_t>` where all 256 threads each
issue cp.async instructions. TMA replaces this with a single bulk tensor copy issued by
1 thread (thread 0) via a TMA descriptor.

Host changes:
- Create TMA descriptors via `make_tma_atom(SM90_TMA_LOAD{}, gtensor, slayout_2d, cta_tile)`
- Pass TMA atoms to kernel with `CUTLASS_GRID_CONSTANT` annotation
- Launch via `cutlass::launch_kernel_on_cluster` (required for TMA, even for single-CTA cluster)

Kernel changes:
- Replace `ThrCopy::partition_S/D` with `tma_partition(tma_a, Int<0>{}, Layout<_1>{}, ...)`
- Replace `copy(g2s_copy, ...)` with `copy(tma_a.with(barrier[pipe]), ...)`
- Replace `cp_async_fence/wait` with `ClusterTransactionBarrier` arrive/wait
- Add barrier arrays to SharedStorage

### S2G: STSM + AutoVec → STSM + TMA Store

The current epilogue has 3 stages: element-wise scaling → STSM to smem → vectorized S2G.
The new epilogue keeps the first two stages and replaces only the final S2G with TMA store.

The TMA store replaces the per-thread vectorized 128-bit global stores with a single bulk
tensor copy from smem to gmem, issued by 1 thread.

Host changes:
- Create TMA store descriptor via `make_tma_atom(SM90_TMA_STORE{}, gC, sC_layout, cta_tile_mn)`
- Keep R2S TiledCopy (`make_tiled_copy_C(SM90_U16x8_STSM_T)`) unchanged — still needed for
  register→smem because the MMA accumulator layout doesn't map trivially to column-major smem,
  and STSM's transposed store handles this rearrangement.
- Remove S2G TiledCopy (`make_tiled_copy(AutoVectorizingCopy...)`) — replaced by TmaC

Kernel epilogue:
1. Element-wise alpha/beta scaling (unchanged)
2. F32→BF16 conversion, STSM to smem (unchanged from pipe_epilogue.cu)
3. `__syncthreads()`, then thread 0 issues TMA store with fence/arrive/wait:
```cpp
if (threadIdx.x == 0) {
    tma_store_fence();
    copy(tma_c, tCsC, tCgC);
    tma_store_arrive();
}
tma_store_wait<0>();
```

### SharedStorage

Added barrier array:
```cpp
uint64_t tma_barrier[K_PIPE_MAX];  // 3 × 8 bytes = 24 bytes
```

Initialized in kernel:
```cpp
if (threadIdx.x == 0) {
    for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
        ProducerBarType::init(&tma_barrier[pipe], 1);  // 1 = single TMA thread arrives
    }
}
__syncthreads();
```

### Launch Mechanism

TMA requires cluster launch. Use `cutlass::launch_kernel_on_cluster` with `dimCluster(1,1,1)`
for single-CTA clusters. This replaces the triple-chevron launch.

Includes added:
```cpp
#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/barrier.h"
```

## Kernel Signature Changes

Old template parameters (from pipe_epilogue.cu):
```
ProblemShape, CtaTiler,
TA, AStride, ASmemLayout, AG2SCopy, S2RCopyAtomA,
TB, BStride, BSmemLayout, BG2SCopy, S2RCopyAtomB,
TC, CStride, TiledMma,
R2SCopy, S2GCopy,
Alpha, Beta
```

New template parameters:
```
ProblemShape, CtaTiler,
TA, SmemLayoutA, TmaA, S2RCopyAtomA,
TB, SmemLayoutB, TmaB, S2RCopyAtomB,
TC, SmemLayoutC, TmaC, R2SCopy, CStride, TiledMma,
Alpha, Beta
```

Key differences:
- `AStride`/`BStride` removed (TMA encodes strides in the descriptor)
- `AG2SCopy`/`BG2SCopy` removed (replaced by `TmaA`/`TmaB`)
- `S2GCopy` removed (replaced by `TmaC`)
- `SmemLayoutA`/`SmemLayoutB` renamed (were `ASmemLayout`/`BSmemLayout`)
- `SmemLayoutC` added (for sC tensor in kernel)
- `R2SCopy` kept (STSM register→smem for epilogue)
- `TmaA`/`TmaB`/`TmaC` use `CUTLASS_GRID_CONSTANT` annotation

## Host Function Changes

### Stride Handling

TMA descriptors encode strides internally. The host creates gmem tensors for TMA inspection:
```cpp
Tensor mA = make_tensor(A, make_shape(M,K), dA);  // (M,K) row-major
Tensor mB = make_tensor(B, make_shape(N,K), dB);  // (N,K) K-contiguous
Tensor mC = make_tensor(C, make_shape(M,N), dC);  // (M,N) column-major
```

The kernel no longer receives stride arguments for A and B — TMA handles addressing.
C still needs `dC` for the epilogue's element-wise load (reading existing C for beta blending).

For the beta term (reading existing C), we keep the element-wise inline load via
`thr_mma.partition_C(gC)` — the same approach as the STSM epilogue. This avoids adding
a TMA load descriptor for C. The gmem tensor gC is constructed from a regular gmem
pointer + stride (not a TMA tensor), and threads read C values one at a time inline
during the alpha/beta scaling loop. The `CStride` (dC) parameter remains in the kernel
signature for this purpose.

### TMA Atom Creation

```cpp
// TMA load for A
Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM, bK));
// TMA load for B
Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN, bK));
// TMA store for C
Copy_Atom tmaC = make_tma_atom(SM90_TMA_STORE{}, mC, sC_layout, make_shape(bM, bN));
```

### Kernel Launch

```cpp
dim3 dimCluster(1, 1, 1);
dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

cutlass::launch_kernel_on_cluster(params, kernel_ptr,
    prob_shape, cta_tiler,
    A, tmaA, s2r_atom_a,
    B, tmaB, s2r_atom_b,
    C, sC_layout, tmaC, dC, mma,
    alpha, beta);
```

## Smem Layout Compatibility

### A and B (TMA Load)

Current smem uses `Swizzle<3,3,3>` XOR pattern. TMA hardware supports specific swizzle
modes (SW128, SW64, SW32, SW-none). `Swizzle<3,3,3>` maps to SW64, which TMA supports.

`make_tma_atom` inspects the smem layout and automatically encodes the swizzle into
the TMA descriptor. No smem layout changes needed for A and B.

The LDSM S2R stage (`SM75_U32x4_LDSM_N`) continues to work with `Swizzle<3,3,3>` smem
since it reads from the swizzled smem addresses — the swizzle is transparent to LDSM
as long as the layout is consistent.

### C (TMA Store)

The sC smem layout is simple column-major `make_layout(make_shape(bM, bN))` — no swizzle.
TMA store doesn't need bank-conflict avoidance since the data is written once and read
once by TMA hardware. The sC buffer reuses sA's space (64 KB fits in 96 KB).

## Smem Usage

| Buffer | Size | Notes |
|--------|------|-------|
| sA (3 stages) | ~96 KB | Swizzle<3,3,3>, reused by sC after MMA |
| sB (3 stages) | ~48 KB | Swizzle<3,3,3> |
| tma_barrier | 24 bytes | 3 × 8-byte mbarriers |
| **Total** | ~144 KB | Same as baseline |

Occupancy unchanged: 1 block/SM × 256 threads.

## Register Pressure

Unchanged from the STSM epilogue. Peak ~148 regs (128 F32 accum + 20 other during
element-wise scaling). TMA store reads from smem, not registers, so it adds no
register pressure.

## What Stays the Same

- MMA atom: `SM80_16x8x16_F32BF16BF16F32_TN`
- MMA layout: `Layout<Shape<_4, _2>>` (8 warps, 256 threads)
- Tile override: `Tile<Underscore, _64, Underscore>`
- CTA tile: (256, 128, 64)
- Pipeline depth: 3 stages
- S2R atoms: `SM75_U32x4_LDSM_N`
- S2R tiled copy: `make_tiled_copy_A/B` (bridges LDSM with MMA)
- Swizzle: `Swizzle<3,3,3>` for sA, sB
- F32 accumulators, alpha/beta scaling, F32→BF16 conversion
- Element-wise inline load of existing C for beta blending
- Grid: `dim3(ceil_div(M,bM), ceil_div(N,bN))`
- Block: `dim3(256)`

## File Layout

```
cute-reference/samples/bf16_gemm_sm80_pipe_tma.cu   # new file, based on pipe_epilogue.cu
cute-reference/samples/build_and_run.sh               # updated: add TMA sample
```

## Testing

- Verify correctness at 1024³ with BF16 C (tolerance 0.5f, same as STSM epilogue)
- Benchmark at 512³, 1024³, 2048³, 4096³, 8192³
- Compare performance with baseline (cp.async + STSM epilogue)
- Expect: comparable or better TFLOP/s due to TMA's hardware-managed addressing
  and reduced instruction overhead (1 TMA instruction vs 256 cp.async per tile)
