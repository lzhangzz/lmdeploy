# Iteration 05: k_block-level Interleaving with Delayed Release + Prefetch

## Goal

Restructure the consumer mainloop to interleave A S2R loads with WGMMA at k_block granularity, add delayed pipeline stage release, and prefetch next stage via `consumer_try_wait`. Target: close the performance gap from ~85% to ~95%+ of cuBLAS at 4096^3.

## Background

Iter 04 loads all 4 k_blocks of A in one bulk copy, then fires 4 WGMMA back-to-back. The WGMMA pipeline stalls during the S2R phase. The CUTLASS production RS WGMMA collective (`sm90_mma_tma_gmma_rs_warpspecialized.hpp` lines 597-727) hides this latency by interleaving at k_block granularity.

Current performance: 667 TFLOP/s at 4096^3 (85% of cuBLAS 786 TFLOP/s).

## Scope

Only the consumer WGMMA kernel changes. The pack kernel and header are reused from iter 04 unchanged.

### Files

1. **`05_split_a_pack.h`** — thin wrapper that includes `04_split_a_pack.h`. Pack kernel is unchanged.
2. **`05_bf16_gemm_sm90_split_a_pack.cu`** — identical to iter 04 pack test, references 05 header.
3. **`05_bf16_gemm_sm90_split_a_wgmma.cu`** — consumer mainloop restructured (only this file has substantive changes).
4. **`CMakeLists.txt`** — add `05_bf16_gemm_sm90_split_a_pack` and `05_bf16_gemm_sm90_split_a_wgmma` targets.

## Consumer Mainloop Design

### Single-loop with preloaded k_block 0

The prologue/main/tail structure from CUTLASS is simplified into a single loop. Before the loop, we acquire the first stage and load k_block 0. Inside the loop, k_block N+1 is loaded before WGMMA k_block N, overlapping the S2R load latency with WGMMA execution.

### k_block load helper

```cpp
// Load a single k_block from per-warpgroup smem region into registers
auto load_k_block = [&](int kb, int stage) {
    Tensor sA_packed = make_tensor(
        make_smem_ptr(smem.A.begin() + stage * a_stage_elements + wg_id * 4096),
        make_shape(Int<8>{}, Int<128>{}, Int<4>{}));
    Tensor sP_k = sA_packed(_, local_tid, kb);  // (8,) stride (1,) — 128-bit vector
    Tensor rA_k = make_tensor(tCrA.data() + kb * size<0>(tCrA), make_shape(Int<8>{}));
    copy(AutoVectorizingCopy{}, sP_k, rA_k);
};
```

Each k_block loads 8 contiguous bf16 (128 bits) from the per-warpgroup smem region. Registers for different k_blocks don't overlap, so loading k_block k+1 while WGMMA reads k_block k is safe.

### Mainloop structure

```cpp
// Before loop: acquire first stage, load k_block 0, prefetch next
pipeline.consumer_wait(smem_pipe_read);
int read_stage = smem_pipe_read.index();
++smem_pipe_read;
auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
load_k_block(0, read_stage);

warpgroup_fence_operand(tCrC);

CUTE_NO_UNROLL
for (int k_tile_iter = 0; k_tile_iter < k_tile_count; ++k_tile_iter)
{
    // k_block 0 already loaded from read_stage

    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < k_block_count; ++k_block)
    {
        // Load next k_block BEFORE WGMMA (overlaps with WGMMA pipeline)
        if (k_block < k_block_count - 1) {
            load_k_block(k_block + 1, read_stage);
        }

        // WGMMA
        warpgroup_arrive();
        gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block,read_stage), tCrC);
        warpgroup_commit_batch();
        warpgroup_wait<2>();

        // Delayed release at k_block 1 (releases stage from 2 k_tiles ago)
        if (k_block == 1 && k_tile_iter >= 1) {
            pipeline.consumer_release(smem_pipe_release);
            ++smem_pipe_release;
        }
    }

    warpgroup_fence_operand(tCrC);

    // Prepare next k_tile: finalize prefetch, load k_block 0 of next stage
    if (k_tile_iter < k_tile_count - 1) {
        pipeline.consumer_wait(smem_pipe_read, barrier_token);
        read_stage = smem_pipe_read.index();
        ++smem_pipe_read;
        barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
        load_k_block(0, read_stage);
    }
}

warpgroup_wait<0>();  // drain all WGMMA batches
warpgroup_fence_operand(tCrC);
pipeline.consumer_release(smem_pipe_release);  // release last stage
++smem_pipe_release;
```

### Pipeline state invariants

- **smem_pipe_read**: always 1 ahead of the current read_stage at the start of each k_tile (advanced after capturing read_stage). At the end of each k_tile, `consumer_wait` finalizes the prefetch for the next stage, read_stage is updated, and `smem_pipe_read` is advanced again (now 2 ahead).
- **smem_pipe_release**: starts at the initial position. First release happens at k_block==1 of k_tile_iter==1 (releases the prologue's stage). Subsequent releases happen at k_block==1 of each following k_tile. One final release after the loop handles the last unreleased stage.
- **barrier_token**: returned by `consumer_try_wait` and passed to `consumer_wait` for efficient non-blocking→blocking wait handoff.

### Three optimizations working together

1. **k_block-level interleaving**: `load_k_block(k+1)` is issued before WGMMA k. The S2R load goes through the memory pipeline while WGMMA goes through the tensor core pipeline. Load latency is hidden behind WGMMA execution.

2. **Delayed stage release**: `consumer_release` at k_block==1 instead of after the full WGMMA loop. The stage from 2 k_tiles ago is released while the current k_tile is still processing, giving the producer more time to refill.

3. **`consumer_try_wait` prefetch**: Non-blocking barrier probe at the end of each k_tile starts polling for the next-next stage early. The blocking `consumer_wait` at the end of the next k_tile finalizes it, reducing stall time.

## What doesn't change

- Producer warp group (bulk copy for A + TMA for B) — unchanged
- Pack kernel — unchanged (reuses `04_split_a_pack.h`)
- Host function layout/template parameters — unchanged
- Packed format — unchanged (64x16 units, per-warpgroup contiguous)
- Epilogue (scaling, STSM, TMA store) — unchanged
- Test/benchmark harness — unchanged

## Validation

Same correctness tests as iter 04 (single tile through 2048x1024x512). Performance benchmark at 256..8192 with cuBLAS reference. Expect ~95%+ of cuBLAS at 4096^3 (up from ~85%).

## Reference

CUTLASS collective `sm90_mma_tma_gmma_rs_warpspecialized.hpp` lines 597-727.
