# Split A loading WGMMA RS design

Goal: Split the rearrangement of operand A in RS WGMMA to a separate pipeline

## Motivation

This is a step in a new mixed precision GEMM pipeline. Which consists of 2 major steps

### Offline packing

1. bit-extend the quantized weights to 16-bit (matching FP16/BF16)
2. Load the bit-extended weights pipeline using the data pipeline (g2s and s2r, as if they were BF16 data) of operand A in RS WGMMA
3. bit-truncate the weights, store weights of the whole warpgroup contiguously


### Runtime

1. Load the packed weights naively (g2s and s2r, without layout transformation)
2. dequantize the weights
3. feed to RS WGMMA

## Status

We are iterating on POCs. Each iteration validates a piece of the design.

### Iteration 01: BF16 split — validate packed register layout

**Files:** `cute-reference/mixed-gemm/01_*`

Splits sample 13 (BF16 GEMM, RS WGMMA, persistent, 384t WS) into two kernels:

1. **Pack kernel** (`01_bf16_gemm_sm90_split_a_pack.cu`): Transforms operand A from gmem
   tensor layout to per-tile GMMA register layout via TMA g2s + S2R copy, then dumps
   registers to a packed gmem buffer. 256 threads (2 warpgroups), grid-stride loop
   over all (M, K) tiles.

2. **WGMMA kernel** (`01_bf16_gemm_sm90_split_a_wgmma.cu`): Reads packed A from gmem
   directly into registers (no smem pipeline for A), runs RS WGMMA with B-only TMA
   pipeline. 384 threads = 128 producer + 256 consumer, persistent scheduling.

**Validated:**
- Correctness of the packed register layout (pack → WGMMA produces bit-exact results)
- Deterministic output across all test sizes (128x256x64 through 2048x1024x512)
- Pack output is 100% deterministic across repeated runs

**Key finding — `warpgroup_wait<0>()` requirement:**
In sample 13, operand A flows through the smem pipeline with separate register sets
per pipeline stage (`tCrA(_,_,k_block,read_stage)`). The split approach reuses the
same `tCrA` registers across k_tiles. With `warpgroup_wait<2>()`, up to 2 WGMMA
operations remain pending after the k_block loop — they read from `tCrA` while the
next k_tile overwrites it. Fix: `warpgroup_wait<0>()` after each k_tile's WGMMA loop
to drain all pending operations before overwriting `tCrA`.

**Performance:** ~60-65% of sample 13 throughput at 4096^3, due to:
- A loaded from gmem via scalar loads (no TMA, no vectorization)
- `warpgroup_wait<0>()` prevents overlapping WGMMA across k_tiles
- Not a concern for the production pipeline, which has a different A loading path
