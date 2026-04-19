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

### S2G: AutoVec → TMA Store

The current epilogue has 3 stages: element-wise scaling → STSM to smem → vectorized S2G.
The new epilogue replaces all three with: element-wise scaling → element-wise register→smem
write → TMA store.

The TMA store replaces the per-thread vectorized 128-bit global stores with a single bulk
tensor copy from smem to gmem, issued by 1 thread.

**Why not STSM + TMA store:** STSM (`stmatrix`) applies hardware swizzle to smem addresses.
TMA store reads from smem using a logical layout interpretation. In a single-role (non-warp-
specialized) kernel, these two address mappings conflict, producing garbage output. Two
independent implementation attempts confirmed this. CUTLASS's production kernels that combine
STSM + TMA store use warp-specialized epilogue pipelines with separate producer/consumer
threads and coordinated barrier sync, which resolves the conflict. Our simple single-role
kernel uses element-wise register→smem copy with a plain column-major sC layout, which is
TMA-compatible without swizzle conflicts.

Host changes:
- Create TMA store TiledCopy via `make_tma_copy(SM90_TMA_STORE{}, gC, sC_layout, cta_tile_mn, Int<1>{})`
- sC layout: plain column-major `make_layout(make_shape(bM, bN), make_stride(Int<1>{}, bM))`
- Remove R2S TiledCopy (replaced by element-wise copy)
- Remove S2G TiledCopy (replaced by TMA store)

Kernel epilogue:
1. Element-wise alpha/beta scaling (unchanged)
2. F32→BF16 conversion, element-wise register→smem write via `thr_mma.partition_C(sC)` + `copy()`
3. `__syncthreads()`, then thread 0 issues TMA store with fence/arrive/wait:
```cpp
if (threadIdx.x == 0) {
    tma_store_fence();
    copy(tma_store_c, tSsC(_, 0), tSgC(_, rest_idx));
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
TC, SmemLayoutC, TmaStoreC, CStride, TiledMma,
Alpha, Beta
```

Key differences:
- `AStride`/`BStride` removed (TMA encodes strides in the descriptor)
- `AG2SCopy`/`BG2SCopy` removed (replaced by `TmaA`/`TmaB`)
- `S2GCopy` removed (replaced by `TmaStoreC`)
- `R2SCopy` removed (element-wise copy replaces STSM for register→smem)
- `SmemLayoutA`/`SmemLayoutB` renamed (were `ASmemLayout`/`BSmemLayout`)
- `SmemLayoutC` added (for sC tensor in kernel)
- `TmaA`/`TmaB` use `CUTLASS_GRID_CONSTANT` annotation
- `TmaStoreC` is a TiledCopy from `make_tma_copy`, annotated `CUTLASS_GRID_CONSTANT`

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

### TMA Atom/TiledCopy Creation

```cpp
// Smem layouts — GMMA atoms (TMA-compatible Swizzle<3,4,3>)
auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16_t>{}, make_shape(bM, bK, bP));
auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16_t>{}, make_shape(bN, bK, bP));
auto sC_layout = make_layout(make_shape(bM, bN), make_stride(Int<1>{}, bM));  // plain column-major

// TMA load for A
Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM, bK));
// TMA load for B
Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN, bK));
// TMA store for C (TiledCopy, not Copy_Atom)
auto tma_store_c = make_tma_copy(SM90_TMA_STORE{}, mC_for_tma, sC_layout, make_shape(bM, bN), Int<1>{});
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
    C, sC_layout, tma_store_c, dC, mma,
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

### C (TMA Store — plain column-major layout)

C is column-major (stride-1 in M). The sC layout uses plain column-major (no swizzle):

```
sC_layout = Layout<Shape<_256,_128>, Stride<_1,_256>>
```

Host setup:
```cpp
auto sC_layout = make_layout(make_shape(bM, bN), make_stride(Int<1>{}, bM));  // column-major, plain
```

Layout shape math:
- sC: (256, 128) = 32768 bf16 = 64 KB

The sC buffer reuses sA's smem (64 KB fits in 96 KB). Since the epilogue uses element-wise
register→smem copy (not STSM), no hardware swizzle is applied. TMA store reads from plain
smem using the TiledCopy descriptor, which encodes the column-major layout. This avoids the
STSM/TMA store swizzle conflict described in the S2G section above.

## Smem Usage

| Buffer | Size | Notes |
|--------|------|-------|
| sA (3 stages) | ~96 KB | GMMA::Layout_K_SW128 (Swizzle<3,4,3>), reused by sC after MMA |
| sB (3 stages) | ~48 KB | GMMA::Layout_K_SW128 (Swizzle<3,4,3>) |
| tma_barrier | 24 bytes | 3 × 8-byte mbarriers |
| sC (epilogue) | 64 KB | Plain column-major, reuses sA's smem space |
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
- Smem swizzle: `Swizzle<3,4,3>` (GMMA layout atoms) for sA, sB — TMA-compatible
- sC: plain column-major layout (no swizzle)
- F32 accumulators, alpha/beta scaling, F32→BF16 conversion
- Element-wise register→smem copy for epilogue (replaces STSM)
- Element-wise inline load of existing C for beta blending
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
