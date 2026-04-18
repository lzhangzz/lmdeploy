# BF16 GEMM SM80 Optimized — Swizzle + LDSM

## Overview

A single-file, heavily-commented CUDA kernel demonstrating BF16 matrix
multiplication using SM80 tensor cores via CuTe, with smem optimizations:
XOR swizzle to eliminate bank conflicts and LDSM for vectorized smem→register
loads. This is the natural successor to `bf16_gemm_sm80.cu` (plain smem,
scalar copy).

## Parameters

| Parameter | Value |
|-----------|-------|
| A, B element type | `bf16` (NVBfloat16) |
| C element type (output + accumulator) | `f32` |
| Matrix layout | TN (A row-major, B K-contiguous) |
| MMA atom | `SM80_16x8x16_F32BF16BF16F32_TN` |
| Atom tiling | `Layout<Shape<_2,_2>>{}` (128 threads / 4 warps) |
| CTA tile | (128, 128, 32) in (M, N, K) |
| Smem layout | Swizzled: `Swizzle<3,3,3>` composed with 8x(8x8) base, tiled via `tile_to_shape` |
| gmem → smem | Plain thread layouts + `local_partition` (same as sample 1) |
| smem → regs | `SM75_U32x4_LDSM_N` via `make_tiled_copy_A/B` + `partition_S` + `retile_D` |
| Pipeline | None — `__syncthreads()` barriers only |

## Data Flow

```
For each K-tile (k=0, 32, 64, ...):
  1. Each thread copies its portion of A, B from global → shared memory
     (writing into swizzled smem — swizzle is transparent to the copy)
  2. __syncthreads()
  3. Each thread loads from swizzled smem → registers via LDSM
     (make_tiled_copy_A/B derives thread mapping from TiledMMA)
  4. Each thread issues tensor core MMA (gemm) on registers
  5. __syncthreads()
After all K-tiles:
  6. Each thread writes accumulators → global memory (axpby)
```

## Swizzled Shared Memory Layout

### Swizzle atom

```cpp
auto swizzle_atom = composition(
    Swizzle<3, 3, 3>{},
    Layout<Shape <_8, Shape<_8, _8>>,
           Stride<_8, Stride<_1, _64>>>{});
```

- `Swizzle<3,3,3>`: 3-bit XOR between bit groups at byte-address positions
  [5:3] and [8:6]. Prevents bank conflicts when a warp reads across rows.
- The 8x(8x8) base layout with strides 8x(1,64) defines one 128-byte swizzle
  tile (8 rows of 8 x 16-bit elements, grouped in 8 banks of 8).
- `tile_to_shape(swizzle_atom, make_shape(bM, bK))` replicates this tile to
  cover the full (128, 32) smem shape.

### How it works

`composition(Swizzle, Layout)` produces a `ComposedLayout`. Address computation
applies the layout first (logical coord → linear offset), then the XOR swizzle
(linear offset → physical address). All CuTe copy operations handle this
transparently — no special handling needed for swizzled tensors.

## LDSM smem → regs Copy

### Copy atom

```cpp
Copy_Atom<SM75_U32x4_LDSM_N, bf16_t> s2r_atom;
```

- `ldmatrix.sync.aligned.x4.m8n8.shared.b16` PTX instruction.
- One warp (32 threads) cooperates to load a 8x8 matrix of 16-bit values
  (128 bytes) from smem. Each thread receives 4 x 32-bit = 128 bits.
- Works for BF16 because the instruction operates on `.b16` (16-bit) without
  interpreting the data — it just moves bits.
- `_N` suffix: normal (non-transposing) layout, matching our K-contiguous smem.

### TiledCopy construction

```cpp
TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
ThrCopy  thr_s2r_a   = s2r_copy_a.get_slice(threadIdx.x);
Tensor tXsA = thr_s2r_a.partition_S(sA);   // smem source: (CPY, MMA_M, MMA_K)
Tensor tXrA = thr_s2r_a.retile_D(tCrA);    // register dst: (CPY, MMA_M, MMA_K)
```

- `make_tiled_copy_A(s2r_atom, mma)` derives the entire s2r TiledCopy from
  the TiledMMA's thread-value layout and tile shape. No manual layout needed.
- `partition_S(sA)` gives each thread's smem view for the copy.
- `retile_D(tCrA)` reshapes the MMA register fragment to match the LDSM
  destination layout. The MMA and LDSM may organize the same registers
  differently; retile handles this at the layout level (no data movement).

### Execution

```cpp
copy(s2r_atom_a, tXsA, tXrA);
copy(s2r_atom_b, tXsB, tXrB);
```

The raw `s2r_atom` is passed (not the `TiledCopy`) because tiling/retiling
has already been done by `partition_S` and `retile_D`.

## Kernel Architecture

### What changes from sample 1

| Component | Sample 1 (plain) | Sample 2 (optimized) |
|-----------|-------------------|----------------------|
| Smem layout | `make_layout(make_shape(bM, bK))` | `tile_to_shape(swizzle_atom, make_shape(bM, bK))` |
| smem→regs copy | `copy(tCsA, tCrA)` (scalar) | `copy(s2r_atom, tXsA, tXrA)` (LDSM) |
| s2r setup | None | `make_tiled_copy_A/B` + `partition_S` + `retile_D` |

### What stays the same

- TiledMMA, CTA tile, problem shape, strides
- gmem→smem: `local_partition` with plain thread layouts `(32, 4)`
- Main loop structure: load → sync → compute → sync
- Epilogue: `axpby`
- Host main: allocate, verify, benchmark

### Device kernel changes

New template parameter: `class S2RCopyAtomA, class S2RCopyAtomB` for the LDSM atoms.

New device-side setup (between TiledMMA setup and main loop):

```cpp
// Build s2r TiledCopy from atom + MMA
TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
ThrCopy  thr_s2r_a   = s2r_copy_a.get_slice(threadIdx.x);
Tensor tXsA = thr_s2r_a.partition_S(sA);
Tensor tXrA = thr_s2r_a.retile_D(tCrA);

TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
ThrCopy  thr_s2r_b   = s2r_copy_b.get_slice(threadIdx.x);
Tensor tXsB = thr_s2r_b.partition_S(sB);
Tensor tXrB = thr_s2r_b.retile_D(tCrB);
```

Main loop body changes:

```cpp
// Before: copy(tCsA, tCrA);  copy(tCsB, tCrB);
// After:
copy(s2r_atom_a, tXsA, tXrA);
copy(s2r_atom_b, tXsB, tXrB);
```

The `tCsA/tCsB` (MMA smem partitions) and `tCrA/tCrB` (register fragments
allocated by `thr_mma.make_fragment_A/B`) still exist — `tCrA/tCrB` are the
register buffers that get retiled for LDSM. The MMA smem partitions (`tCsA`,
`tCsB`) are no longer used for copying; they're superseded by the LDSM
partitions (`tXsA`, `tXsB`).

### Host function changes

New parameters:

```cpp
Copy_Atom<SM75_U32x4_LDSM_N, bf16_t> s2r_atom_a;
Copy_Atom<SM75_U32x4_LDSM_N, bf16_t> s2r_atom_b;
```

Changed smem layout construction:

```cpp
auto swizzle_atom = composition(
    Swizzle<3, 3, 3>{},
    Layout<Shape <_8, Shape<_8, _8>>,
           Stride<_8, Stride<_1, _64>>>{});
auto sA = tile_to_shape(swizzle_atom, make_shape(bM, bK));
auto sB = tile_to_shape(swizzle_atom, make_shape(bN, bK));
```

## File Layout

```
cute-reference/samples/bf16_gemm_sm80_opt.cu   # optimized kernel + host + main
cute-reference/samples/build_and_run.sh         # updated: compile both files
```

## Build

```bash
CUTLASS_INC=/path/to/build/_deps/repo-cutlass-src/include
nvcc -std=c++17 -arch=sm_90a \
     -I${CUTLASS_INC} \
     cute-reference/samples/bf16_gemm_sm80_opt.cu \
     -o bf16_gemm_sm80_opt
./bf16_gemm_sm80_opt [M] [N] [K]
```

## Smem Usage

- sA: swizzled layout for (128, 32) bf16 — size determined by `cosize_v<ASmemLayout>`
- sB: swizzled layout for (128, 32) bf16 — size determined by `cosize_v<BSmemLayout>`
- Swizzle may add padding; exact size is compile-time computable via `cosize_v`
- Expected to be close to 8 KB each (16 KB total), similar to sample 1

## Testing

- Same verification as sample 1: CPU reference GEMM, max error < 0.01
- Benchmark 100 iterations, report GFLOP/s
- Compare performance with sample 1 to quantify swizzle+LDSM benefit
- Test sizes: 1024^3 (default), 256^3, 2048x2048x1024
