# Iteration 03: Vectorized Pack/Unpack

## Context

Iteration 02 achieved parity with sample 13 (668 TFLOP/s at 4096^3) by replacing scalar gmem
loads with `cp.async.bulk` smem pipeline for packed A. However, both the pack kernel's gmem
store and the consumer's smem-to-register load use individual bf16 accesses (32 per thread).

## Goal

Vectorize both the pack store and consumer S2R using 128-bit (8 bf16) operations by changing
the packed data layout so each thread has 8 contiguous bf16 values per k_block.

## Packed Format Change

**Current format (iter 01/02):** `offset(i, t) = i * 256 + t`
- Registers outermost, threads inner
- Stride between same-thread registers: 256 bf16 (512 bytes)
- No vectorization possible

**New format (iter 03):** `offset(k, t, j) = k * 2048 + t * 8 + j`
- k_blocks outermost (4), threads middle (256), registers inner (8)
- CuTe tensor uses mode order `(REG, THREAD, K_BLOCK)` = `Shape<_8, _256, _4>` with default
  column-major stride `(1, 8, 2048)` — matches our packed format without explicit stride specification
- Each thread's 8 registers per k_block are contiguous
- Total size unchanged: 8192 bf16 (16 KB) per tile

## Files

- `cute-reference/mixed-gemm/03_split_a_pack.h` — pack device kernel and host function (based on `01_split_a_pack.h`, vectorized stores)
- `cute-reference/mixed-gemm/03_bf16_gemm_sm90_split_a_pack.cu` — pack test harness (based on iter 01)
- `cute-reference/mixed-gemm/03_bf16_gemm_sm90_split_a_wgmma.cu` — consumer kernel (based on iter 02, vectorized S2R)
- `cute-reference/mixed-gemm/CMakeLists.txt` — add 03_* targets

## Architecture

Unchanged from iter 02. 384 threads (128 producer + 256 consumer), persistent scheduling.
The only changes are the packed format and the load/store access patterns.

```
Pack kernel:
  TMA g2s → swizzled smem → S2R (AutoVectorizingCopy) → tCrA registers
  tCrA reshape (4, 8) → copy(AutoVectorizingCopy, rA, gP) → vectorized gmem store

Consumer kernel:
  cp.async.bulk gmem → smem_A[stage] (format-agnostic)
  smem tensor (4, 256, 8) → per-thread slice (4, 8) → copy(AutoVectorizingCopy, sP, rA) → tCrA
  WGMMA with warpgroup_wait<2>() (unchanged)
```

## Pack Kernel

Based on iter 01's pack kernel. Changes:

1. **Store pattern:** Replace 32 individual bf16 stores with CuTe tensor copy:

```cpp
// Packed gmem tensor: (REG, THREAD, K_BLOCK) stride (1, 8, 2048)
// Mode order chosen so default column-major strides match our format
Tensor gPacked = make_tensor(make_gmem_ptr(packed_A + tile_base),
                              make_shape(Int<8>{}, Int<256>{}, Int<4>{}));
Tensor gP = gPacked(_, threadIdx.x, _);  // (8, 4) stride (1, 2048)

// Reshape tCrA (8, 1, 4) → (REG, K_BLOCK) = (8, 4) stride (1, 8)
// Default stride matches tCrA flat layout: rA(j, k) = j + k * 8
Tensor rA = make_tensor(tCrA.data(), make_shape(Int<8>{}, Int<4>{}));

copy(AutoVectorizingCopy{}, rA, gP);
```

CuTe's `make_layout(shape)` uses column-major strides (leftmost mode gets stride 1). Mode order
`(REG, THREAD, K_BLOCK)` gives default stride `(1, 8, 2048)` matching `offset = k*2048 + t*8 + j`.
`AutoVectorizingCopy` sees stride-1 innermost mode of size 8 in both src and dst, recasts to
`uint128_t`, and emits 4 × 128-bit store instructions.

2. **TMA load and S2R copy:** Unchanged from iter 01.

3. **Register tensor view:** `tCrA.data()` returns a `bf16_t*` into the register array. Creating
   a new tensor view with `make_tensor(tCrA.data(), layout)` aliases the same register storage.
   This is safe for reads (pack) and writes (consumer S2R) because the underlying storage is
   an `ArrayEngine<bf16_t, 32>` with guaranteed contiguous layout.

4. **Host function:** Same as iter 01 (`split_a_pack`) — grid dimensions, smem size, TMA setup
   all unchanged. Only the device kernel's store pattern changes.

## Consumer Kernel

Based on iter 02's WGMMA kernel. Changes:

1. **S2R load pattern:** Replace 32 individual bf16 loads with CuTe tensor copy:

```cpp
// Smem tensor for this stage: (REG, THREAD, K_BLOCK) stride (1, 8, 2048)
Tensor sA = make_tensor(
    make_smem_ptr(smem.A.begin() + read_stage * a_stage_elements),
    make_shape(Int<8>{}, Int<256>{}, Int<4>{}));
Tensor sP = sA(_, threadIdx.x, _);  // (8, 4) stride (1, 2048)

// Register view: (REG, K_BLOCK) = (8, 4) stride (1, 8)
Tensor rA = make_tensor(tCrA.data(), make_shape(Int<8>{}, Int<4>{}));

copy(AutoVectorizingCopy{}, sP, rA);
```

2. **Bulk copy:** Unchanged — copies raw bytes from gmem to smem.

3. **Transaction bytes, pipeline, WGMMA:** All unchanged.

4. **Host function:** Same as iter 02 — kernel launch parameters, smem size, benchmark harness
   all unchanged. Only the device kernel's S2R load pattern changes.

## Smem Budget

Unchanged from iter 02: 144 KB (3 stages × 48 KB). Same headroom (84 KB on L20Y).

## Bank Conflict Analysis

Consumer S2R with 128-bit loads: **zero bank conflicts**.

- Thread t's k_block k starts at byte offset `k * 4096 + t * 16`
- 128-bit access spans 4 consecutive banks (16 bytes)
- SM90 hardware splits the 32-thread warp into 4 groups of 8 threads for wide smem accesses
- Within each 8-thread group, threads access banks `0-3, 4-7, ..., 28-31` — all 32 banks
  covered exactly once per group
- Groups execute serialized, so no inter-group conflicts either

This is strictly better than the current iter 02 S2R (2-way bank conflicts with 32-bit loads).

## Validation

Bit-exact output matching against iter 02 results for the same test sizes
(128x256x64 through 2048x1024x512). The packed A data contains the same values in a different
layout; only the load/store paths change.

## Performance Expectation

- Pack kernel: faster gmem stores (4 × 128-bit vs 32 × 16-bit per thread)
- Consumer: faster S2R (4 × 128-bit vs 32 × 16-bit per thread, ~4x fewer effective cycles)
- WGMMA throughput should remain at iter 02 / sample 13 parity
