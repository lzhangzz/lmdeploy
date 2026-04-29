# Iter 05 Progress: k_block-level Interleaving

## Goal
Close performance gap from ~85% to ~95%+ of cuBLAS at 4096^3 by implementing CUTLASS-style pipeline: k_block interleaving, delayed release, consumer_try_wait prefetch.

## Baseline (iter 04)
- iter 04 prologue (all loads + wait<2> per gemm) + wait<0>() drain + simple main loop
- ALL TESTS PASS
- Performance: ~558 TFLOP/s at 4096^3 (limited by wait<0>() drain)

## What was tried (chronological)

### Attempt 1: Full CUTLASS 3-part structure (no drain)
- CUTLASS prologue (no wait<2> in k_block loop) + transition + main loop + tail
- Single-K: PASS, Multi-K: FAIL (error 8.09)
- Hypothesis: register safety issue during transition

### Attempt 2: Full CUTLASS + wait<0>() drain between prologue and transition
- Added wait<0>() drain after CUTLASS prologue to eliminate register safety concern
- Single-K: PASS, Multi-K: FAIL (error 8.09)
- **KEY FINDING**: drain doesn't help! Issue is NOT register safety

### Attempt 3: CUTLASS prologue + drain + CUTLASS transition + wait<2> fix in tail
- Found that CUTLASS tail was missing warpgroup_wait<2>() after last gemm
- CUTLASS uses separate mma()/mma_tail() function calls — compiler handles dependency tracking
- When inlined (our case), need explicit wait<2> before tail's wait<0>
- Still fails without drain — issue is the token-based consumer_wait pattern

### Attempt 4: CUTLASS prologue + drain + plain consumer_wait transition → HANGS
- Replacing token-based consumer_wait with plain consumer_wait causes hang
- **ROOT CAUSE**: `mbarrier.try_wait.parity` in consumer_try_wait CONSUMES the barrier signal
  - After try_wait succeeds, subsequent plain consumer_wait tries to wait on already-consumed barrier → hang
  - Must use token-based consumer_wait(state, barrier_token) which skips wait when token==WaitDone
  - `test_wait.parity` is the non-consuming alternative, but CUTLASS pipeline doesn't expose it

### Attempt 5: Full CUTLASS without drain (with wait<2> fix) — still fails
- Error 8.42 with token-based consumer_wait + non-incrementing smem_pipe_read
- Even partial drain (wait<2> to reduce 3→2 pending) fails
- Exact mechanism unclear — register safety chain says it should work

### Attempt 6 (FINAL WORKING): iter 04 prologue + NO drain + iter 04 main loop
- Uses iter 04 prologue: all loads + wait<2> per gemm (leaves 2 pending WGMMA)
- NO drain between prologue and main loop
- iter 04 main loop with k_block interleaving + delayed release at k_block==1
- Plain consumer_wait (no try_wait, no token)
- ALL TESTS PASS

## Key Findings

1. **wait<0>() drain kills performance** (~558 vs ~680 TFLOP/s at 4096^3)
   - The 2 pending WGMMA from prologue can safely carry over into the main loop
   - The main loop's wait<2> chain naturally drains the pipeline

2. **k_block interleaving works with iter 04 style pipeline**
   - Loading k_block N+1 before WGMMA N hides S2R latency behind WGMMA
   - No need for CUTLASS's token-based approach

3. **Delayed release at k_block==1 works**
   - Releases stage from 2 k_tiles ago while current k_tile still processing
   - Gives producer more time to refill

4. **consumer_try_wait is NOT safe with plain consumer_wait**
   - try_wait CONSUMES the barrier signal (uses mbarrier.try_wait.parity)
   - Must pair with token-based consumer_wait(state, token)
   - Or avoid try_wait entirely (our approach)

5. **Full CUTLASS token-based pipeline still fails without drain**
   - The non-incrementing smem_pipe_read + token-based consumer_wait pattern produces wrong results
   - Likely a compiler optimization issue when the no-op consumer_wait allows code reordering

## Performance Results

| Size     | Custom (GFLOP/s) | cuBLAS (GFLOP/s) | % cuBLAS |
|----------|-------------------|-------------------|----------|
| 4096^3   | 682,047           | 793,714           | 85.9%    |
| 8192^3   | 659,906           | 718,280           | 91.9%    |

vs iter 04 baseline with drain: 558 TFLOP/s at 4096^3 (70.3% of cuBLAS)

## What's implemented

- k_block-level interleaving (load k_block N+1 before WGMMA N)
- Delayed stage release at k_block==1
- No consumer_try_wait (removed — token-based approach not needed)

## What's NOT implemented

- consumer_try_wait prefetch (token-based pipeline not working without drain)
- CUTLASS 3-part structure (prologue/transition/tail with non-incrementing smem_pipe_read)
