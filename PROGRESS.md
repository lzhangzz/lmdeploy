# Iter 05 Progress: k_block-level Interleaving

## Goal
Close performance gap from ~85% to ~95%+ of cuBLAS at 4096^3 by implementing CUTLASS-style pipeline: k_block interleaving, delayed release, consumer_try_wait prefetch.

## Baseline (iter 04)
- Simple prologue (all loads + wait<2> per gemm) + simple main loop + wait<0>() drain
- Performance: ~558 TFLOP/s at 4096^3 (limited by wait<0>() drain)

## What was tried (chronological)

### Attempt 1: Full CUTLASS 3-part structure (prologue/main/tail)
- CUTLASS-style prologue with no wait<2> in k_block loop
- Multi-K: FAIL (error 7.97 even for single tile)
- Root cause: the CUTLASS no-wait<2> prologue pattern doesn't work in our configuration

### Attempt 2: CUTLASS prologue + wait<2> in k_block loop
- Added wait<2> inside the prologue's k_block loop
- Single tile: STILL FAILS (error 7.97)
- Root cause: NOT pending WGMMA count — something else in the CUTLASS prologue

### Attempt 3: iter 04 prologue + token-based transition (plain consumer_wait)
- Kept iter 04 prologue, used plain consumer_wait in transition
- Single tile: PASS, Multi-K: FAIL (error 8.09)
- Same error with or without consumer_try_wait in prologue

### Attempt 4: iter 04 prologue + CUTLASS main loop structure (consumer_wait inside loop)
- Moved consumer_wait inside the main loop (iter 04 style)
- Added consumer_try_wait BEFORE each iteration, consumer_wait with token at start
- ALL TESTS PASS
- **KEY FINDING**: The loop structure matters. consumer_wait must be at the START of each
  iteration, not at the END (CUTLASS's k_block==last pattern). The try_wait goes BEFORE
  the iteration (prefetching), and the token-based consumer_wait consumes the token.

### Attempt 5: consumer_try_wait in prologue (after ++smem_pipe_read)
- Added try_wait after ++smem_pipe_read in the iter 04 prologue
- Single tile: PASS, Multi-K: FAIL (error 8.09)
- **Root cause still unknown**: the try_wait for stage 1 in the prologue causes Multi-K
  failures even when the token is unused. The exact mechanism is unclear.

## Final working configuration

1. **Prologue**: iter 04 style (plain consumer_wait, all loads, wait<2> per gemm)
   - NO consumer_try_wait in prologue (causes Multi-K failures for unknown reason)
   - 2 pending WGMMA carry over to main loop

2. **Main loop**: consumer_try_wait prefetch before each iteration
   - `barrier_token = consumer_try_wait(smem_pipe_read)` before entering iteration
   - `consumer_wait(smem_pipe_read, barrier_token)` at start of iteration
   - k_block interleaving with wait<2>
   - Delayed release at k_block==1

3. **Tail**: wait<0>() drain, release last stage

## Performance Results

| Size     | Custom (GFLOP/s) | cuBLAS (GFLOP/s) | % cuBLAS |
|----------|-------------------|-------------------|----------|
| 4096^3   | 678,831           | 787,527           | 86.2%    |
| 8192^3   | 675,989           | 692,947           | 97.6%    |

vs iter 04 baseline: 667 TFLOP/s (85%) at 4096^3, 654 TFLOP/s at 8192^3

## Key Findings

1. **consumer_try_wait DOES work** — when placed correctly:
   - try_wait before the iteration, consumer_wait with token at the start
   - The token-based approach is NOT fundamentally broken

2. **consumer_try_wait in prologue causes failures** — unknown why
   - Adding try_wait for stage 1 after ++smem_pipe_read in the prologue
   - Even when the token is unused, Multi-K tests fail (error 8.09)
   - According to PTX docs, try_wait is non-consuming (pure observation with acquire)
   - The PTX instruction is `asm volatile` so it's a compiler barrier
   - The mechanism of failure is still unknown

3. **CUTLASS's loop structure doesn't directly translate** — CUTLASS uses consumer_wait
   at k_block==last inside the k_block loop, with smem_pipe_read not incremented.
   Our simpler approach (consumer_wait at start of k_tile iteration, with ++smem_pipe_read)
   works correctly with the try_wait prefetch.

4. **The CUTLASS no-wait<2> prologue doesn't work for us** — accumulating WGMMA without
   draining in the prologue's k_block loop produces wrong results (error 7.97 for single
   tile). Adding wait<2> inside the loop doesn't help. The root cause is unknown.

## Open questions

1. Why does consumer_try_wait in the prologue cause Multi-K failures?
   - It's a non-consuming, non-blocking observation with acquire semantics
   - The asm volatile should be a compiler barrier
   - But empirically, removing it fixes the issue

2. Why does the CUTLASS no-wait<2> prologue fail?
   - Even with wait<2> added, the single tile test fails
   - The rest of the prologue code is identical to the working iter 04 prologue
   - Must be something about the CUTLASS prologue structure (separate k_block loop + last gemm)
