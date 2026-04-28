# Sample 13: Fully Optimized RS Mainloop (CUTLASS-Style)

## Problem

Sample 12 (RS optimized with k_block double-buffering) has a small but consistent GFLOP/s gap compared to sample 09 (SS variant). Root cause: the consumer mainloop releases pipeline buffers too late and doesn't overlap k_tile boundaries with compute, starving the TMA producer.

## Root Causes

### 1. Late Pipeline Release

Sample 12 releases the pipeline buffer after all 4 k_blocks complete. CUTLASS releases at k_block==1 (after only 2 WGMMA instructions). With a 3-stage pipeline, the producer sits idle for ~2 WGMMA latencies waiting for a buffer that could have been released sooner.

**Impact:** Producer starvation → TMA loads don't overlap well with compute → pipeline bubbles.

### 2. No k_tile Prefetch Overlap

Sample 12 does a blocking `consumer_wait` at the start of each k_tile. CUTLASS uses `consumer_try_wait` (non-blocking) at k_block==0, then does the actual `consumer_wait` at the last k_block while simultaneously copying the first k_block of the next stage. This eliminates the bubble between pipeline stages.

**Impact:** Every k_tile transition has a full pipeline wait on the critical path.

## Design: Full CUTLASS RS Mainloop Pattern

Restructure the consumer mainloop into three sections: prologue, mainloop, epilogue.

### Prologue (first k_tile)

```
barrier_token = consumer_try_wait(stage 0)         // non-blocking probe
consumer_wait(stage 0, barrier_token)               // block until stage 0 ready
read_stage = stage 0 index
++smem_pipe_read
barrier_token = consumer_try_wait(stage 1)          // probe next stage early

copy(smem->reg, k_block 0 of stage 0)               // first S2R load

for k_block 0..2 (skip last):                       // only 3 of 4 k_blocks
    copy(smem->reg, k_block+1 of stage 0)
    warpgroup_arrive()
    gemm(k_block, stage 0)
    warpgroup_commit_batch()
warpgroup_wait<2>()

// Last k_block of stage 0 + prefetch of stage 1
warpgroup_arrive()
gemm(k_block 3, stage 0)
warpgroup_commit_batch()

consumer_wait(stage 1, barrier_token)               // now block for stage 1
copy(smem->reg, k_block 0 of stage 1)               // prefetch next stage
barrier_token = consumer_try_wait(stage 2)          // probe stage 2
warpgroup_wait<2>()
```

### Mainloop (middle k_tiles, k_tile_count > 1)

```
for each remaining k_tile (except last):
    read_stage = current stage index
    ++smem_pipe_read

    warpgroup_fence_operand(accum)
    for k_block 0..3:
        if k_block == 0:
            barrier_token = consumer_try_wait(next stage)     // early probe

        if k_block == 3 (last):
            consumer_wait(next stage, barrier_token)           // now block
            copy(smem->reg, k_block 0 of next stage)           // prefetch
        else:
            copy(smem->reg, k_block+1 of current stage)        // normal prefetch

        warpgroup_arrive()
        gemm(k_block, current stage)
        warpgroup_commit_batch()
        warpgroup_wait<2>()

        if k_block == 1:
            pipeline.consumer_release(smem_pipe_release)        // EARLY RELEASE
            ++smem_pipe_release

    warpgroup_fence_operand(accum)
```

### Epilogue (last k_tile)

```
read_stage = current stage index
warpgroup_fence_operand(accum)

for k_block 0..2 (skip last):
    copy(smem->reg, k_block+1 of current stage)
    warpgroup_arrive()
    gemm(k_block, current stage)
    warpgroup_commit_batch()
    warpgroup_wait<2>()

    if k_block == 1:
        pipeline.consumer_release(smem_pipe_release)
        ++smem_pipe_release

// Last k_block (no next stage to prefetch)
warpgroup_arrive()
gemm(k_block 3, current stage)
warpgroup_commit_batch()

warpgroup_fence_operand(accum)
```

### Pipeline State Tracking

- `smem_pipe_read`: advanced at the start of each k_tile (prologue, mainloop) or not at all (epilogue)
- `smem_pipe_release`: advanced at k_block==1 within each k_tile
- `barrier_token`: carried across k_tile boundaries, consumed by `consumer_wait`

### What Changes vs Sample 12

| Aspect | Sample 12 | New |
|---|---|---|
| Pipeline release | After all k_blocks | At k_block==1 (2 WGMMA sooner) |
| k_tile wait | Blocking `consumer_wait` at top | Non-blocking `consumer_try_wait` at k_block==0, deferred wait at last k_block |
| k_tile prefetch | None | First k_block of next stage copied at last k_block of current stage |
| Mainloop structure | Single loop | Prologue / mainloop / epilogue (3 sections) |
| Unchanged | Everything else (tile sizes, smem, producer, epilogue, TMA) | Same |

### What Does NOT Change

- Tile sizes (128x256x64), pipeline depth (3 stages)
- Producer mainloop (TMA loads)
- STSM epilogue
- Register allocation (warpgroup_reg_alloc<232>)
- Shared memory layouts
- Persistent scheduling
- Benchmark harness

## Files

- **New:** `cute-reference/samples/13_bf16_gemm_sm90_pipe_tma_ws_persistent_rs_cutlass.cu` — copy of sample 12 with restructured consumer mainloop
- **Modify:** `cute-reference/samples/CMakeLists.txt` — add sample 13 to build

## Success Criteria

- Correctness: max error < 0.5 for 1024^3 BF16 GEMM
- Performance: GFLOP/s gap vs sample 09 reduced to < 1% at 2048x2048x2048 and larger sizes
- Sample 12 left untouched
