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


### Iteration 02: Optimize WGMMA kernel

Optimize the WGMMA kernel by loading packed A with `cp.async.bulk` via smem pipeline
and remove the extra `warpgroup_wait<0>()`

**Files:** `cute-reference/mixed-gemm/02_*`

Replaces iter 01's scalar gmem→register loads with `SM90_BULK_COPY_G2S` gmem→smem + per-thread
smem→register loads through a 3-stage pipeline. A single mbarrier per stage coordinates both
A bulk copy and B TMA arrivals.

**Validated:**
- Correctness matches iter 01 (same test sizes, same max errors)
- Performance at 4096^3: **668 TFLOP/s** vs iter 01's 369 TFLOP/s (1.81x speedup)
- Achieves **parity with sample 13** at large sizes (668 vs 669 TFLOP/s)
- The `warpgroup_wait<0>()` elimination fully restores WGMMA overlap across k_tiles


### Iteration 03: Vectorized pack/unpack

New packed format `offset(k, t, j) = k*2048 + t*8 + j` enables vectorized 128-bit stores (pack)
and loads (consumer S2R). Uses CuTe `copy(AutoVectorizingCopy{}, ...)` with mode order
`(REG, THREAD, K_BLOCK)` for correct default column-major strides.

**Files:** `cute-reference/mixed-gemm/03_*`

**Validated:**
- Correctness matches iter 02 (same max errors across all test sizes)
- Pack kernel: vectorized gmem stores (4 × 128-bit vs 32 × 16-bit per thread)
- Consumer S2R: vectorized smem loads with zero bank conflicts
- Performance at 4096^3: **665 TFLOP/s** — parity with iter 02 (666 TFLOP/s)


### Iteration 04: Reduce packing unit to 64×16

Reduces the packing unit from (128, 64) to (64, 16) — one WGMMA instruction's A operand —
for maximum composability in the production pipeline.

**Files:** `cute-reference/mixed-gemm/04_*`

Keeps the (128, 64) processing tile in the pack kernel (same TMA+S2R pipeline), but restructures
the output into 8 independent (64, 16) units organized by (warpgroup, k_block). Each warpgroup's
4 units are contiguous (4096 bf16). Consumer bulk copy is unchanged; only the smem-to-register
mapping is updated to load per-warpgroup regions.

Packed format per unit: `(REG=8, THREAD=128)` strides `(1, 8)`, 1024 bf16.
Full packed tensor: `Shape<_8, _128, (4, K_tiles), (2, M_tiles)>` with composite K and M modes
that flatten to `K_units = ceil_div(K, 16)` and `M_units = ceil_div(M, 64)`.

**Validated:**
- Correctness matches iter 03 (same max errors across all test sizes)
- Pack output: per-warpgroup contiguous regions, vectorized 128-bit stores
- Consumer S2R: per-warpgroup loads from contiguous 4096 bf16 smem regions
- Performance at 4096^3: **667 TFLOP/s** — parity with iter 03 (665 TFLOP/s)
- Performance at 8192^3: **662 TFLOP/s** — parity with iter 03 (654 TFLOP/s)


### Iteration 05: k_block-level interleaving and delayed stage release

Pipeline optimization to close the gap with cuBLAS. The iter 04 pipeline loads all 4 k_blocks
of A to registers, then issues 4 WGMMA back-to-back, stalling the WGMMA pipeline during S2R.
Iter 05 interleaves S2R loads with WGMMA at k_block granularity and delays stage release.

**Files:** `cute-reference/mixed-gemm/05_*`

**Implemented:**

1. **k_block-level interleaving**: While WGMMA k_block N is in flight, load k_block N+1 from
   smem to registers. S2R load latency is hidden behind WGMMA execution.
   ```
   iter 04:                     iter 05:
   load all 4 k_blocks          load k_block 0
   wgmma 0, wait<2>             wgmma 0, wait<2>
   wgmma 1, wait<2>             load k_block 1   ← overlapped
   wgmma 2, wait<2>             wgmma 1, wait<2>
   wgmma 3, wait<2>             load k_block 2   ← overlapped
                                wgmma 2, wait<2>
                                load k_block 3   ← overlapped
                                wgmma 3, wait<2>
   ```

2. **Delayed stage release**: Release pipeline stages at `k_block == 1` of the next k_tile
   instead of after the full WGMMA loop. Gives the producer more time to refill stages.

3. **No wait<0>() drain**: Eliminated the wait<0>() drain between prologue and main loop.
   The 2 pending WGMMA from the prologue carry over naturally into the main loop's wait<2>
   chain, recovering ~120 TFLOP/s at 4096^3.

**Not implemented (attempted but failed):**

- `consumer_try_wait` prefetch in prologue: Adding try_wait after `++smem_pipe_read` in the
  prologue causes Multi-K failures (error 8.09) even when the token is unused. The PTX docs
  confirm try_wait is non-consuming, and the asm volatile should be a compiler barrier.
  The exact failure mechanism is unknown.
- CUTLASS no-wait<2> prologue: Accumulating WGMMA without draining in the prologue's k_block
  loop produces wrong results (error 7.97 for single tile). Adding wait<2> doesn't help.
- CUTLASS k_block==last consumer_wait pattern: Using consumer_wait inside the k_block loop
  (at k_block==last) instead of at the start of each k_tile iteration produces errors.

**Validated:**
- Correctness matches iter 04 (same max errors across all test sizes)
- Performance at 4096^3: **679 TFLOP/s** (86.2% of cuBLAS 788 TFLOP/s)
- Performance at 8192^3: **676 TFLOP/s** (97.6% of cuBLAS 693 TFLOP/s)
- Up from iter 04's 667 TFLOP/s (85%) at 4096^3 and 654 TFLOP/s at 8192^3
- The consumer_try_wait prefetch in the main loop closes the gap at 8192^3


### Iteration 06: Threadblock swizzling (negative result)

Investigated whether CUTLASS-style tile swizzling could close the 4096^3 gap (86.2% vs 97.6% at
8192^3). Tested both M-axis swizzle (CUTLASS pattern: consecutive CTAs share N tile) and N-axis
swizzle (consecutive CTAs within M group get spread-out N tiles). Both directions hurt performance.

**Files:** `cute-reference/mixed-gemm/06_*`

**Tested:**

1. **M-axis swizzle** (CUTLASS bit-decomposition): Groups `swizzle_size` consecutive CTAs to
   share the same N tile, maximizing B reuse in L2. At the cost of breaking A reuse (consecutive
   CTAs no longer share the same M tile).

2. **N-axis swizzle**: Keeps M grouping (preserving A reuse) but interleaves N within each M
   group: `n_idx = (n_base >> log_sw) + (n_base & (sw-1)) * (n_tiles >> log_sw)`. Spreads B
   accesses across L2 sets.

**M-axis swizzle results** (4096^3):

| log_swizzle | sw | GFLOP/s | vs cuBLAS |
|---|---|---|---|
| 0 (none) | 1 | **677,740** | 85.5% |
| 1 | 2 | 673,133 | 84.9% |
| 2 | 4 | 657,763 | 83.0% |
| 3 | 8 | 610,332 | 77.0% |

**N-axis swizzle results** (4096^3):

| log_swizzle | sw | GFLOP/s | vs cuBLAS |
|---|---|---|---|
| 0 (none) | 1 | **677,040** | 85.4% |
| 1 | 2 | 677,168 | 85.4% |
| 2 | 4 | 675,951 | 85.2% |
| 3 | 8 | 635,164 | 80.1% |

**Conclusion:** No swizzle (log_swizzle=0, row-major) is optimal. The row-major tile ordering
provides maximum L2 reuse for packed A — consecutive CTAs share the same m_tile and load the same
A data via bulk copy. Any swizzle breaks this A reuse, and the B reuse gained does not compensate.
The CUTLASS swizzle is designed for kernels where both operands use TMA; our kernel loads A via
bulk copy from a packed buffer, which has different L2 behavior.

The 4096^3 gap is not a tile ordering issue. It may be a compute-to-memory ratio issue at that
problem size, or cuBLAS using a different kernel configuration.

**Reference:** CUTLASS `sm90_tile_scheduler.hpp` `get_work_idx_m_and_n()` and
`tile_scheduler_params.h` `get_log_swizzle_size()`.


### Iteration 07: Epilogue-compute overlap

Defers `tma_store_wait<0>()` from after the TMA store issue to the start of the next
tile's epilogue, allowing the TMA store for tile N to overlap with tile N+1's compute
(consumer_wait + S2R + WGMMA).

**Files:** `cute-reference/mixed-gemm/07_*`

**Implemented:**

1. **Deferred TMA store wait**: Moved `tma_store_wait<0>()` from the end of the epilogue
   (after TMA store issue) to the beginning (before r2s copy). The first tile's wait is
   a no-op; subsequent tiles' waits ensure the previous tile's store completed before
   sC is overwritten. The producer's existing `tma_store_wait<0>()` ensures the last
   tile's store completes before kernel exit.

**Validated:**
- Correctness matches iter 06 (same max errors across all test sizes)
- Performance at 4096^3: **683,441 TFLOP/s** (86.1% of cuBLAS 793,717 TFLOP/s)
- Performance at 8192^3: **667,109 TFLOP/s** (96.1% of cuBLAS 694,459 TFLOP/s)
- vs iter 06: 679 TFLOP/s (85.5%) at 4096^3, 672 TFLOP/s (97.6%) at 8192^3
- Modest improvement at 4096^3 (+4.5 TFLOP/s), parity at 8192^3
