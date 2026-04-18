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
// Smem layouts — GMMA atoms (TMA-compatible Swizzle<3,4,3>)
auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16_t>{}, make_shape(bM, bK, bP));
auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16_t>{}, make_shape(bN, bK, bP));
auto sC_layout = tile_to_shape(GMMA::Layout_MN_SW128_Atom<bf16_t>{}, make_shape(bM, bN));

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

### Swizzle<3,3,3> is NOT TMA-compatible — must switch to GMMA layouts

The current smem uses `Swizzle<3,3,3>` which has M=3. TMA hardware only supports swizzles
with M=4 (SW128/SW64/SW32/DISABLE), M=5, or M=6. `Swizzle<3,3,3>` triggers:

```
static_assert(M < 0, "Unsupported layout swizzle.")
```

in `detail::get_tma_swizzle_bits`. This is a compile-time failure.

**Solution:** Switch to GMMA layout atoms, which use `Swizzle<B,4,3>` (M=4, TMA-compatible).

### A and B (TMA Load — K-major GMMA layout)

Both A and B are K-contiguous (stride-1 in K). The TMA-compatible K-major layout for bf16:

```
GMMA::Layout_K_SW128_Atom<bf16_t>
  = Swizzle<3,4,3> o Layout<Shape<_8,_64>, Stride<_64,_1>>    (after upcast for bf16)
```

Host setup:
```cpp
auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16_t>{}, make_shape(bM, bK, bP));
auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16_t>{}, make_shape(bN, bK, bP));
```

Layout shape math:
- Atom covers (8, 64) = 512 bf16 per swizzle tile
- sA: (256, 64, 3) → 256/8=32 atoms in M, 64/64=1 in K → 32 atoms per stage, 3 stages
- sB: (128, 64, 3) → 128/8=16 atoms in N, 64/64=1 in K → 16 atoms per stage, 3 stages

LDSM compatibility: `make_tiled_copy_A(SM75_U32x4_LDSM_N, mma)` partitions the swizzled
smem layout automatically. The swizzle is transparent to LDSM — CuTe's partitioning handles
the address mapping. The atom shape (8, 64) is compatible with LDSM's 8-row reads.

### C (TMA Store — MN-major GMMA layout)

C is column-major (stride-1 in M). The TMA-compatible MN-major layout for bf16:

```
GMMA::Layout_MN_SW128_Atom<bf16_t>
  = Swizzle<3,4,3> o Layout<Shape<_64,_8>, Stride<_1,_64>>    (after upcast for bf16)
```

Host setup:
```cpp
auto sC_layout = tile_to_shape(GMMA::Layout_MN_SW128_Atom<bf16_t>{}, make_shape(bM, bN));
```

Layout shape math:
- Atom covers (64, 8) = 512 bf16 per swizzle tile
- sC: (256, 128) → 256/64=4 atoms in M, 128/8=16 in N → 64 atoms total = 32768 bf16 = 64 KB

The sC buffer still reuses sA's smem (64 KB fits in 96 KB). The swizzle pattern differs
from A/B (MN-major vs K-major), but this is fine — sC is only used after MMA completes.

Note: The STSM R2S copy (`make_tiled_copy_C(SM90_U16x8_STSM_T, mma)`) writes to sC, and
TMA store reads from sC. Both operations see the same swizzled smem addresses. The STSM
writes column-major data into the swizzled layout, and TMA store reads it back using the
same swizzle descriptor. Consistency is maintained because both use CuTe's layout system.

## Smem Usage

| Buffer | Size | Notes |
|--------|------|-------|
| sA (3 stages) | ~96 KB | GMMA::Layout_K_SW128 (Swizzle<3,4,3>), reused by sC after MMA |
| sB (3 stages) | ~48 KB | GMMA::Layout_K_SW128 (Swizzle<3,4,3>) |
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
- Smem swizzle: `Swizzle<3,4,3>` (GMMA layout atoms) for sA, sB, sC — TMA-compatible
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
