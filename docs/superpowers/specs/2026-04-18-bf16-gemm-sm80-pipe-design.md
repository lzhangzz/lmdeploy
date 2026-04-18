# BF16 GEMM SM80 Pipelined — cp.async 3-Stage Pipeline

## Overview

A single-file, heavily-commented CUDA kernel demonstrating BF16 matrix
multiplication using SM80 tensor cores via CuTe, with a 3-stage software
pipeline using cp.async for overlapping global-to-shared memory transfers
with tensor core computation. This is the natural successor to
`bf16_gemm_sm80_opt.cu` (swizzle + LDSM, synchronous gmem→smem).

## Parameters

| Parameter | Value |
|-----------|-------|
| A, B element type | `bf16` (NVBfloat16) |
| C element type (output + accumulator) | `f32` |
| Matrix layout | TN (A row-major, B K-contiguous) |
| MMA atom | `SM80_16x8x16_F32BF16BF16F32_TN` |
| Atom tiling | `Layout<Shape<_2,_2>>{}` + `Tile<_32,_32,_16>` (128 threads) |
| CTA tile | (128, 128, 64) in (M, N, K) |
| Pipeline stages | 3 (bP = Int<3>) |
| Smem layout | Swizzled: `Swizzle<3,3,3>` composed with 8x(8x8) base, per-stage |
| gmem → smem | `SM80_CP_ASYNC_CACHEALWAYS<uint128_t>` via `make_tiled_copy` |
| smem → regs | `SM75_U32x4_LDSM_N` via `make_tiled_copy_A/B` |
| Synchronization | `cp_async_fence()` + `cp_async_wait<N>()` + `__syncthreads()` |

## Data Flow

```
Prefetch phase (fill pipeline):
  for stage = 0 .. bP-2:
    copy(g2s_copy, gA[k_tile], sA[stage])    // async gmem→smem
    copy(g2s_copy, gB[k_tile], sB[stage])    // async gmem→smem
    cp_async_fence()                           // commit group

  cp_async_wait<bP-2>()                       // wait for stage 0
  __syncthreads()
  copy(s2r_atom_a, sA[read_pipe], tCrA)       // LDSM smem→regs
  copy(s2r_atom_b, sB[read_pipe], tCrB)
  advance read_pipe

Main loop (overlap G2S with S2R+MMA):
  while k_tile_count > -(bP-1):
    gemm(mma, tCrA, tCrB, tCrC)              // tensor core MMA

    cp_async_wait<bP-2>()                     // wait for next stage
    __syncthreads()

    copy(s2r_atom_a, sA[read_pipe], tCrA)     // LDSM smem→regs
    copy(s2r_atom_b, sB[read_pipe], tCrB)
    advance read_pipe

    if k_tile_count > 0:
      copy(g2s_copy, gA[k_tile], sA[write_pipe])  // async gmem→smem
      copy(g2s_copy, gB[k_tile], sB[write_pipe])
      cp_async_fence()
      advance write_pipe

    --k_tile_count

Epilogue:
  axpby(alpha, tCrC, beta, tCgC)
```

## gmem→smem Async Copy (G2S)

### Copy atom

```cpp
Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, bf16_t>
```

- Wraps `cp.async.ca.shared.global.L2::128B` PTX instruction
- Moves 128 bits = 8 bf16 elements per instruction
- `CACHEALWAYS`: data flows through L2 → L1 → shared memory
- Single-element copy atom: each call issues one cp.async instruction

### TiledCopy construction

```cpp
TiledCopy g2s_copy_a = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, bf16_t>{},
    Layout<Shape<_16, _8>, Stride<_8, _1>>{},   // 128 threads, K-major
    Layout<Shape<_1, _8>>{});                    // 1x8 vals = 128 bits
```

- Thread layout (16, 8) stride (8, 1): 16 threads in M, 8 in K = 128 threads
- Value layout (1, 8): each thread issues 1 cp.async per K-chunk (8 contiguous bf16 = 128 bits)
- Coverage per copy-tile: (16, 64) in (M, K)
- For full (128, 64) smem, `copy()` loops 8x in M automatically

### Async synchronization

- `cp_async_fence()` — emit `cp.async.commit_group` (non-blocking)
- `cp_async_wait<N>()` — block until only N groups remain in flight
  - `wait<0>()` = wait_all (all groups complete)
  - `wait<1>()` = all but 1 group complete
- `__syncthreads()` — after wait, to ensure all threads see the smem data

## Pipeline Structure

### Smem buffering

3 copies of A and B smem buffers. Each buffer uses the same swizzled layout
as the opt sample. The pipe stage is an extra dimension managed by index
arithmetic (`smem_pipe_read % bP`, `smem_pipe_write % bP`).

Smem allocation:
```cpp
__shared__ TA smem_a[cosize_v<ASmemLayout> * 3];  // 3 stages
__shared__ TB smem_b[cosize_v<BSmemLayout> * 3];  // 3 stages
```

### Pipe counters

```cpp
int smem_pipe_read  = 0;   // which stage to read from (S2R)
int smem_pipe_write = 0;   // which stage to write to (G2S)
int k_tile_count    = K_TILE_MAX;  // remaining K-tiles
```

### Overlap achieved

While MMA computes on registers loaded from stage N, cp.async transfers
stage N+1 from global memory. The `cp_async_wait` only stalls if the
transfer hasn't completed — for large CTA tiles, the transfer typically
finishes before MMA, making the wait nearly free.

## Kernel Architecture

### What changes from opt sample

| Component | Opt sample (sync) | Pipe sample (async) |
|-----------|-------------------|---------------------|
| gmem→smem | `local_partition` + `copy()` (sync) | `make_tiled_copy` + cp.async |
| Thread layout for G2S | `Layout<Shape<32,4>>` via `local_partition` | Built into `TiledCopy` |
| Smem buffers | 1 per A/B | 3 per A/B |
| Main loop | Simple for loop, 2× syncthreads | Prefetch + while, fence/wait/sync |
| Smem size | ~16KB per matrix | ~48KB per matrix |

### What stays the same

- TiledMMA, CTA tile sizes, problem shape, strides
- Swizzled smem layout per buffer (same as opt sample)
- S2R (smem→regs): LDSM via `make_tiled_copy_A/B` + `retile_D`
- Epilogue: `axpby`
- Host main: verify once at 1024^3, benchmark 4 sizes

### Device kernel template

New template parameters: `class G2SCopyA, class G2SCopyB` (the TiledCopy
objects for async gmem→smem).

Removed template parameters: `class AThreadLayout, class BThreadLayout`
(thread layouts for sync copy are no longer needed).

### Device kernel setup changes

Replace Step 3 (gmem→smem partitioning):
```cpp
// Old: local_partition with plain thread layouts
// Tensor tAgA = local_partition(gA, tA, threadIdx.x);
// Tensor tAsA = local_partition(sA, tA, threadIdx.x);

// New: TiledCopy partitioning
ThrCopy thr_g2s_a = g2s_copy_a.get_slice(threadIdx.x);
Tensor tAgA = thr_g2s_a.partition_S(gA);    // (CPY, THR_M, THR_K, k)
Tensor tAsA = thr_g2s_a.partition_D(sA);    // (CPY, THR_M, THR_K, bP)
```

Smem tensors now include pipe dimension:
```cpp
Tensor sA = make_tensor(make_smem_ptr(smem_a), sA_layout);  // (BLK_M, BLK_K, bP)
Tensor sB = make_tensor(make_smem_ptr(smem_b), sB_layout);  // (BLK_N, BLK_K, bP)
```

### Host function changes

Replace thread layouts and local_partition setup:
```cpp
// Old:
// auto tA = make_layout(make_shape(Int<32>{}, Int<4>{}));
// auto tB = make_layout(make_shape(Int<32>{}, Int<4>{}));

// New: TiledCopy for async gmem→smem
TiledCopy g2s_copy_a = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, bf16_t>{},
    Layout<Shape<_16, _8>, Stride<_8, _1>>{},
    Layout<Shape<_1, _8>>{});

TiledCopy g2s_copy_b = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, bf16_t>{},
    Layout<Shape<_16, _8>, Stride<_8, _1>>{},
    Layout<Shape<_1, _8>>{});
```

Smem layout gains pipe dimension (or array is 3× larger with per-stage tensors):
```cpp
auto sA_stage = tile_to_shape(swizzle_atom, make_shape(bM, bK));  // per-stage layout
// Allocate 3 stages, create tensor with pipe dimension
```

Pass TiledCopy objects instead of thread layouts to kernel launch.

## Smem Usage

- Per-stage sA: swizzled (128, 64) bf16 ≈ 16 KB
- Per-stage sB: swizzled (128, 64) bf16 ≈ 16 KB
- 3 stages: ~48 KB per matrix, ~96 KB total
- L20Y has 192 KB shared memory per SM — fits with room for occupancy

## File Layout

```
cute-reference/samples/bf16_gemm_sm80_pipe.cu   # pipelined kernel + host + main
cute-reference/samples/build_and_run.sh          # updated: compile all three
```

## Build

```bash
CUTLASS_INC=/path/to/build/_deps/repo-cutlass-src/include
nvcc -std=c++17 -arch=sm_90a \
     -I${CUTLASS_INC} \
     cute-reference/samples/bf16_gemm_sm80_pipe.cu \
     -o bf16_gemm_sm80_pipe
./bf16_gemm_sm80_pipe [M] [N] [K]
```

## Testing

- Same verification as opt sample: CPU reference GEMM, max error < 0.01
- Benchmark 100 iterations at 512^3, 1024^3, 2048^3, 4096^3
- Compare performance with opt sample to quantify cp.async pipeline benefit
- Expected: significant speedup at large sizes where gmem latency dominates
