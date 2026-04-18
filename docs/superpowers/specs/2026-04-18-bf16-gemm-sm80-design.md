# BF16 GEMM Learning Sample — SM80 Tensor Cores with CuTe

## Overview

A single-file, heavily-commented CUDA kernel demonstrating BF16 matrix
multiplication using SM80 tensor cores via the CuTe library. The kernel uses
the simplest possible architecture — no pipelining, no swizzling — to teach
the core CuTe concepts: tensors, layouts, TiledMMA, register fragments, and
the tensor core programming model.

## Parameters

| Parameter | Value |
|-----------|-------|
| A, B element type | `bf16` (NVBfloat16) |
| C element type (output + accumulator) | `f32` |
| Matrix layout | TN (A row-major, B col-major in traditional terms) |
| MMA atom | `SM80_16x8x16_F32BF16BF16F32_TN` |
| Atom tiling | `Layout<Shape<_2,_2>>{}` (128 threads / 4 warps) |
| CTA tile | (128, 128, 32) in (M, N, K) |
| Shared memory | Plain layouts, no swizzle, no padding |
| gmem → smem | Plain thread layouts + `local_partition` |
| smem → regs | Default `copy()` (auto-vectorizing) |
| Pipeline | None — `__syncthreads()` barriers only |

## Data Flow

```
For each K-tile (k=0, 32, 64, ...):
  1. Each thread copies its portion of A, B from global → shared memory
  2. __syncthreads()
  3. Each thread copies its MMA-partitioned view from shared → registers
  4. Each thread issues tensor core MMA (gemm) on registers
  5. __syncthreads()
After all K-tiles:
  6. Each thread writes accumulators → global memory (axpby)
```

## Kernel Architecture

### Device kernel (`bf16_gemm_device`)

Section-by-section:

**1. Preconditions** — `static_assert` checks on ranks, sizes, congruence.

**2. Full and tiled tensors**
- `make_tensor(make_gmem_ptr(...), shape, stride)` for mA, mB, mC
- `local_tile()` to extract per-CTA views: gA (BLK_M, BLK_K, k), gB (BLK_N, BLK_K, k), gC (BLK_M, BLK_N)

**3. Shared memory**
- Static `__shared__` arrays sized via `cosize_v<Layout>`
- `make_tensor(make_smem_ptr(...), layout)` for sA (128, 32), sB (128, 32)

**4. gmem → smem partitioning**
- Plain thread layouts: `tA = make_layout(make_shape(Int<32>{}, Int<4>{}))` (32×4, M-major for A)
- `local_partition(gA, tA, threadIdx.x)` gives per-thread gmem and smem views

**5. TiledMMA setup**
- `TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{}, Layout<Shape<_2,_2>>{})`
- `ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x)`
- Partition smem for MMA: `thr_mma.partition_A(sA)` → tCsA (MMA, MMA_M, MMA_K)
- Partition gmem output: `thr_mma.partition_C(gC)` → tCgC (MMA, MMA_M, MMA_N)
- Allocate register fragments: `thr_mma.make_fragment_A/B/C(...)`

**6. Main loop**
```
clear(tCrC)
for k_tile in 0..K_TILE_MAX:
    copy(tAgA(_,_,k_tile), tAsA)    // gmem → smem
    copy(tBgB(_,_,k_tile), tBsB)
    __syncthreads()

    copy(tCsA, tCrA)                 // smem → regs
    copy(tCsB, tCrB)
    gemm(mma, tCrA, tCrB, tCrC)     // tensor core MMA

    __syncthreads()
```

**7. Epilogue**
- `axpby(alpha, tCrC, beta, tCgC)` — write results to global memory

### Host function (`bf16_gemm_tn`)

- Define problem shape (M, N, K) and strides (dynamic)
- Define CTA tile sizes (static: 128, 128, 32)
- Define smem layouts (static, plain)
- Define thread layouts for copy (static)
- Create TiledMMA
- Compute grid dims, launch kernel

### Main

- Parse M, N, K from command line (defaults: 1024×1024×1024)
- Allocate host/device tensors via thrust
- Fill A, B with random bf16 in [-1, 1], C with -1
- Run kernel once, verify against CPU reference
- Benchmark 100 iterations, report GFLOP/s
- Print max error

## File Layout

```
cute-reference/samples/bf16_gemm_sm80.cu     # kernel + host + main
cute-reference/samples/build_and_run.sh       # compile & run script
```

## Build

```bash
CUTLASS_INC=/path/to/build/_deps/repo-cutlass-src/include
nvcc -std=c++17 -arch=sm_90a \
     -I${CUTLASS_INC} \
     cute-reference/samples/bf16_gemm_sm80.cu \
     -o bf16_gemm_sm80
./bf16_gemm_sm80 [M] [N] [K]
```

## Thread Layout Details

With 128 threads and CTA tile (128, 128, 32):

**gmem → smem copy (per thread)**:
- tA: `make_layout(make_shape(Int<32>{}, Int<4>{}))` — covers (128, 32) smem
  - 32 threads in M, 4 threads in K → each thread copies 4×8 = 32 bf16 values
- tB: `make_layout(make_shape(Int<32>{}, Int<4>{}))` — covers (128, 32) smem
  - Same structure

**MMA partitioning**:
- TiledMMA with 2×2 tiling of 16×8×16 atom → 128 threads, effective MMA (32, 16, 16)
- K-blocks per K-tile: 32/16 = 2
- Each thread accumulates 128×128/128 = 128 f32 values across the CTA tile

## Smem Usage

- sA: 128 × 32 × 2 bytes = 8 KB
- sB: 128 × 32 × 2 bytes = 8 KB
- Total: 16 KB (well within 48 KB limit)
