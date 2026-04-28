# Iteration 02: Optimize WGMMA Kernel — TMA Async Bulk Copy for Packed A

## Context

Iteration 01 validated the packed register layout: the pack kernel dumps GMMA register
contents to gmem, and the WGMMA kernel reads them back via scalar loads. This produces
bit-exact results but runs at ~60-65% of sample 13 throughput due to:

- Scalar gmem-to-register loads (no TMA, no vectorization)
- `warpgroup_wait<0>()` after each k_tile prevents overlapping WGMMA across k_tiles

Iteration 02 addresses both bottlenecks.

## Goal

1. Load packed A from gmem using `cp.async.bulk` (TMA async bulk copy)
2. Add smem pipeline stages for A to overlap loading with WGMMA execution
3. Remove `warpgroup_wait<0>()`, restore `warpgroup_wait<2>()` behavior

## Approach

Keep the packed A format unchanged (register layout, thread-interleaved). Add smem
pipeline stages for A alongside the existing B TMA pipeline. Both share a single
mbarrier per stage.

## Files

- `cute-reference/mixed-gemm/02_bf16_gemm_sm90_split_a_wgmma.cu` — optimized WGMMA kernel
- Pack kernel (`01_bf16_gemm_sm90_split_a_pack.cu`) unchanged

## Architecture

```
Producer (thread 256):
  packed_A gmem ──cp.async.bulk──> smem_A[stage] ──┐
  B tensor gmem   ──TMA───────────> smem_B[stage] ──┤ mbarrier
                                                     └──> consumer wait
Consumer (256 threads):
  smem_A[stage] ──per-thread load──> tCrA
  smem_B[stage] ──AutoVect S2R───> tCrB
  WGMMA(tCrA, tCrB, tCrC) with warpgroup_wait<2>()
```

Thread count: 384 (128 producer + 256 consumer), persistent scheduling. Unchanged from iter 01.

## Pack Format

Unchanged from iteration 01. Each tile is stored as `regs_per_thread * 256` bf16 values
in gmem. Thread `t`'s register `i` is at gmem offset `tile_base + i * 256 + t`.

## Smem Layout for A

Three pipeline stages (`bP = 3`). Each stage holds one tile of packed A in register layout:

```
smem_A: [stage_0: regs_per_tile bf16s][stage_1: ...][stage_2: ...]
```

`regs_per_tile = regs_per_thread * 256`. The layout within each stage matches the gmem
packed format exactly — bulk copy transfers the tile verbatim.

S2R bank conflict analysis: 2-way (adjacent thread pairs share smem banks). Acceptable.

## Producer

Single thread (thread 256) drives both transfers per pipeline stage:

1. `pipeline.producer_acquire(smem_pipe_write)` — wait for stage release
2. Compute packed A gmem address: `packed_A + tile_idx * regs_per_tile`
3. Issue `cp.async.bulk` from gmem → `smem_A[stage]`, arriving at stage's mbarrier
4. Issue TMA for B → `smem_B[stage]`, arriving at the same mbarrier
5. `++smem_pipe_write`

## Consumer

Per k_tile iteration:

1. `pipeline.consumer_wait(smem_pipe_read)` — wait for A+B stage ready
2. S2R for A — per-thread loads from register-layout smem:
   ```
   auto* smem_ptr = smem_A_base + stage * regs_per_tile;
   for (int i = 0; i < regs_per_thread; ++i)
       tCrA(i) = smem_ptr[i * 256 + threadIdx.x];
   ```
3. S2R for B — unchanged from sample 13 (AutoVectorizingCopy)
4. WGMMA k_block loop with `warpgroup_wait<2>()`
5. `pipeline.consumer_release(smem_pipe_release)`

`warpgroup_wait<0>()` is removed. `warpgroup_wait<2>()` suffices because
`consumer_wait` on the next stage provides the synchronization barrier before
`tCrA` is overwritten by the next k_tile's S2R.

## Pipeline Synchronization

Both `cp.async.bulk` and TMA arrive at the same mbarrier per stage. On SM90, a single
mbarrier supports multiple arrival mechanisms. The consumer waits on this combined
mbarrier, which signals only after both transfers complete.

Implementation options for the POC:
- Manual barrier management with `cp.async.bulk` and TMA mbarrier arrivals
- Or extend cute's pipeline abstraction to support both

## Smem Budget

| Component | Per stage | 3 stages |
|-----------|-----------|----------|
| A (register layout) | 128 x 64 x 2B = 16 KB | 48 KB |
| B (K-swizzled) | 256 x 64 x 2B = 32 KB | 96 KB |
| **Total** | | **144 KB** |

L20Y has 228 KB shared memory per SM. 84 KB headroom.

## Validation

Bit-exact output matching against iter 01 results for the same test sizes
(128x256x64 through 2048x1024x512). The packed A data is identical; only the
loading path changes.

## Performance Expectation

The `warpgroup_wait<2>()` restoration should close most of the gap with sample 13.
Remaining gap (if any) comes from 2-way smem bank conflicts on A's S2R, which is
minor compared to the iter 01 bottlenecks.
