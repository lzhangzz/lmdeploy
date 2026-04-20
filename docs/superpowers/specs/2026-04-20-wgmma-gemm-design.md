# SM90 WGMMA Persistent Warp-Specialized GEMM

## Overview

Create sample 09: replace SM80 HMMA tensor cores with SM90 WGMMA in the persistent warp-specialized GEMM kernel. This is a copy of 08 with the MMA plumbing swapped — all other architecture (warp specialization, PipelineTmaAsync, separate C smem, STSM+TMA store epilogue, persistence) is preserved.

Creates: `cute-reference/samples/09_bf16_gemm_sm90_pipe_tma_ws_persistent.cu`
Modifies: `cute-reference/samples/CMakeLists.txt`, `cute-reference/samples/run.sh`

## Problem

08 uses SM80 HMMA (`SM80_16x8x16_F32BF16BF16F32_TN`) — warp-level MMA with register-based operands. On SM90 hardware, WGMMA (warpgroup MMA) provides significantly higher throughput: 64x128 per instruction vs 16x8, and reads operands directly from shared memory without LDSM copies.

## Solution

Replace the HMMA atom with a WGMMA atom and remove the S2R (smem-to-register) copy infrastructure. WGMMA reads A/B operands directly from smem via 64-bit GMMA descriptors, eliminating the LDSM instructions and the k_block inner loop.

## Key Changes

### 1. Atom and Tile Size

```
Old: SM80_16x8x16_F32BF16BF16F32_TN, Layout<Shape<_4,_2>>, bM=256, bN=128
New: SM90_64x128x16_F32BF16BF16F32_SS<K,K>, Layout<Shape<_2,_1>>, bM=128, bN=128
```

The WGMMA atom is 64x128x16 (128 threads = 1 warpgroup). Tiling 2x in M covers the 128x128 CTA tile with 2 warpgroups = 256 consumer threads. bM halves from 256 to 128.

**Smem layouts unchanged:** `GMMA::Layout_K_SW128_Atom` is already WGMMA-compatible. The atom's `Major::K` parameter matches this layout.

### 2. Kernel Template Parameters

Remove `S2RCopyAtomA` and `S2RCopyAtomB` from the kernel template. No LDSM atoms needed.

### 3. Consumer Setup (Replace S2R + Register Fragments)

Old (HMMA):
```cpp
Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));   // register fragment
Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));   // register fragment
TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
// ... S2R copy setup, tXsA, tXrA, tXsB, tXrB ...
auto K_BLOCK_MAX = size<2>(tCrA);
```

New (WGMMA):
```cpp
Tensor tCsA = thr_mma.partition_A(sA);            // (MMA, MMA_M, MMA_K, PIPE) smem view
Tensor tCsB = thr_mma.partition_B(sB);            // (MMA, MMA_N, MMA_K, PIPE) smem view
Tensor tCrA = thr_mma.make_fragment_A(tCsA);      // GMMA descriptors
Tensor tCrB = thr_mma.make_fragment_B(tCsB);      // GMMA descriptors
```

`make_fragment_A/B` creates tensors of 64-bit GMMA descriptors from the smem views. These descriptors point directly at smem addresses — no data movement into registers.

### 4. Consumer Main Loop

Old (HMMA — inner k_block loop with s2r prefetch):
```cpp
for (int k_tile_iter = 0; k_tile_iter < k_tile_count; ++k_tile_iter) {
  pipeline.consumer_wait(smem_pipe_read);
  // s2r prefetch k_block 0
  // inner loop: s2r prefetch next k_block, gemm current k_block
  pipeline.consumer_release(smem_pipe_release);
}
```

New (WGMMA — single gemm call per pipeline stage):
```cpp
for (int k_tile_iter = 0; k_tile_iter < k_tile_count; ++k_tile_iter) {
  pipeline.consumer_wait(smem_pipe_read);

  warpgroup_fence_operand(tCrC);           // compiler barrier on accum registers
  warpgroup_arrive();                      // hw fence: wgmma.fence.sync.aligned
  gemm(mma, tCrA(_,_,_,smem_pipe_read.index()),
            tCrB(_,_,_,smem_pipe_read.index()), tCrC);
  warpgroup_commit_batch();                // wgmma.commit_group.sync.aligned
  warpgroup_wait<0>();                     // wgmma.wait_group.sync.aligned 0
  warpgroup_fence_operand(tCrC);           // compiler barrier before reading accum

  pipeline.consumer_release(smem_pipe_release);
  ++smem_pipe_read;
  ++smem_pipe_release;
}
```

No k_block loop. A single `gemm()` call per pipeline stage internally loops over all M/N/K atom tiles, issuing multiple `wgmma.mma_async` PTX instructions. The `warpgroup_fence_operand` is a compiler-only barrier (empty asm with memory clobber) that prevents reordering accumulator register accesses across the async WGMMA boundary — required because WGMMA is asynchronous and the compiler could otherwise read stale accumulator values. The `warpgroup_arrive/commit_batch/wait` triplet provides hardware-level warpgroup synchronization.

The `gemm()` function accepts `TiledMMA` (which inherits from `MMA_Atom`) and the partitioned descriptor/accumulator tensors. It dispatches through Dispatch [5] (loops K) → Dispatch [4] (loops M/N with serpentine traversal) → Dispatch [1] (calls `mma.call()` → `mma_unpack` → `MMA_Op::fma()`). For WGMMA SS, `fma()` takes `uint64_t` smem descriptors and 64 F32 accumulator references per atom invocation.

### 5. Host Function

```cpp
auto bM = Int<128>{};   // was Int<256>{}
auto bN = Int<128>{};
auto bK = Int<64>{};

TiledMMA mma = make_tiled_mma(
    SM90_64x128x16_F32BF16BF16F32_SS<GMMA::Major::K, GMMA::Major::K>{},
    Layout<Shape<_2, _1>>{});

// Remove: s2r_atom_a, s2r_atom_b
```

Smem for C: `make_layout(make_shape(Int<128>{}, bN), ...)` instead of `make_shape(bM, bN)`.

### 6. Smem Budget

| Component | Elements | Bytes |
|-----------|----------|-------|
| A (128x64x3, GMMA swizzled) | 24,576 bf16 | 49,152 |
| B (128x64x3, GMMA swizzled) | 24,576 bf16 | 49,152 |
| C (128x128, plain col-major) | 16,384 bf16 | 32,768 |
| Pipeline barriers | — | 48 |
| **Total** | | **~131 KB** |
| SM90 max per SM | | **228 KB** |
| Headroom | | **~97 KB** |

Half the smem of 08 due to halving bM.

### 7. Build and Run Configuration

- CMakeLists.txt: Add `09_bf16_gemm_sm90_pipe_tma_ws_persistent` to `CUTE_GEMM_SAMPLES`
- run.sh: Add invocation for 09
- Alignment assertion: `m % 128 == 0 && n % 128 == 0 && k % 64 == 0` (was `m % 256 == 0`)
- Print banner: "SM90 WGMMA" instead of "SM80 HMMA"

## What Stays the Same

- **Producer warp group:** TMA loads, pipeline acquire/release, persistent while loop — identical code
- **Epilogue:** F32→BF16 conversion, STSM write to C smem, NamedBarrier(256,6), TMA store — identical structure. `make_tiled_copy_C` adapts STSM tiling to WGMMA's CLayout automatically. Note: WGMMA accumulators are 4x larger per thread (128 F32 vs 32 F32) — the `make_tensor<bf16_t>(tCrC.layout())` conversion adapts automatically.
- **Pipeline:** PipelineTmaAsync with 3 stages, register dealloc/alloc (40/232)
- **SharedStorage:** Same struct with separate C buffer
- **Kernel launch:** 384 threads, `__launch_bounds__(384, 1)`, cluster launch API
- **Correctness test:** 1024³, max error < 0.5f
- **Benchmark sweep:** 256³, 512³, 1024³, 2048³, 4096³, 8192³

## WGMMA vs HMMA Summary

| Property | HMMA (08) | WGMMA (09) |
|----------|-----------|------------|
| Atom shape | 16x8x16 | 64x128x16 |
| Threads per atom | 32 (1 warp) | 128 (1 warpgroup) |
| Operands | Registers (via LDSM) | Smem descriptors |
| S2R copies | Required (LDSM) | Not needed |
| Main loop | k_block inner loop | Single gemm() per stage |
| Warpgroup sync | N/A | arrive/commit_batch/wait |
| CTA tile | 256x128x64 | 128x128x64 |
| Consumer threads | 256 (8 warps) | 256 (2 warpgroups) |
| Total threads | 384 | 384 |
| Smem usage | ~208 KB | ~131 KB |
| Accum/thread | 32 F32 | 128 F32 |
| Fence operand | Not needed | Required (compiler barrier) |

## Testing

- Correctness at 1024³ (max error < 0.5f)
- Full benchmark sweep: 256³ through 8192³
- Compare 09 vs 08 TFLOP/s — WGMMA expected to be faster
