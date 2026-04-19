# BF16 GEMM SM80 — Warp-Specialized TMA Load/Store

## Overview

Add warp-group specialization to the existing TMA GEMM kernel, splitting the 384-thread block into a producer warp group (TMA loads) and two consumer warp groups (SM80 HMMA + LDSM + epilogue). The kernel keeps SM80 HMMA tensor cores and uses CUTLASS's `PipelineTmaAsync` for producer-consumer synchronization.

Based on `bf16_gemm_sm80_pipe_tma.cu`. New file: `bf16_gemm_sm80_pipe_tma_ws.cu`.

## Thread Layout

384 threads (12 warps), organized as 3 warp groups:

| Warp Group | threadIdx range | Warps | Role |
|---|---|---|---|
| WG 0 (consumer) | 0–127 | 0–3 | SM80 HMMA + LDSM S2R + epilogue |
| WG 1 (consumer) | 128–255 | 4–7 | SM80 HMMA + LDSM S2R + epilogue |
| WG 2 (producer) | 256–383 | 8–11 | TMA load (1 warp via elect_one_sync) |

The 256 consumer threads use `threadIdx.x % 256` as the MMA thread index, mapping 0–255 directly to the existing 256-thread HMMA layout. No MMA partitioning changes needed.

Register allocation:
- Producer WG: `warpgroup_reg_dealloc<40>` — only TMA descriptor handling
- Consumer WGs: `warpgroup_reg_alloc<232>` — accumulators + LDSM buffers

## What Changes

### Kernel Structure

The current single-path kernel splits into three code paths gated by `warp_group_idx`:

```
if (warp_group_idx == 2) {
    // Producer: TMA loads
} else {
    // Consumer 0,1: LDSM + MMA + epilogue
}
```

### Producer Warp Group (wg_idx == 2)

One warp (warp 8, threads 256–287) issues TMA loads via `elect_one_sync()`. The other 3 warps in the producer WG are idle during TMA operations but participate in barrier init and `tma_store_wait<0>()` at kernel exit.

Producer flow:
1. Barrier init (all 384 threads participate)
2. `__syncthreads()` after init
3. `warpgroup_reg_dealloc<40>()`
4. Prologue: issue `min(Stages, k_tile_count)` TMA loads
5. Main loop: `producer_acquire` → TMA load A + B → advance pipe → `producer_commit`
6. Tail: `producer_tail` to drain remaining stages
7. Wait at `tma_store_wait<0>()` until consumer finishes epilogue

### Consumer Warp Groups (wg_idx 0, 1)

Both consumer warp groups together form the 256-thread MMA. Each thread computes its MMA partition via `threadIdx.x % 256` (WG0 maps to MMA threads 0–127, WG1 maps to MMA threads 128–255). The MMA layout `Layout<Shape<_4, _2>>` assigns warps 0–3 to threads 0–127 and warps 4–7 to threads 128–255, so the consumer threads map directly without remapping.

Consumer flow:
1. `warpgroup_reg_alloc<232>()`
2. Setup ThrMMA, LDSM copies, STSM copy — using `threadIdx.x % 256` as the MMA thread index
3. Wait for first stage via `consumer_wait`
4. Main loop: `consumer_wait` → LDSM S2R → MMA → `consumer_release` → advance pipe
5. MMA tail: drain remaining pipeline stages
6. Epilogue (identical to current kernel):
   a. Alpha/beta scaling
   b. F32→BF16 conversion
   c. STSM to plain sC
   d. Consumer-only sync (NamedBarrier)
   e. TMA store (thread 0 of consumers)
   f. `tma_store_wait<0>()` (all 384 threads)

### PipelineTmaAsync

Replaces the current manual `ClusterTransactionBarrier` + phase tracking with CUTLASS's `PipelineTmaAsync<Stages>`.

Params setup:
```cpp
typename MainloopPipeline::Params pipeline_params;
if (warp_group_idx == 2) {
    pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
} else {
    pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
}
pipeline_params.is_leader = (warp_group_thread_idx == 0);
pipeline_params.num_consumers = 256;  // total consumer threads
pipeline_params.num_producers = 1;    // single TMA thread
pipeline_params.transaction_bytes = tma_transaction_bytes;
```

Barrier init (called by the constructor with `cute::true_type{}`):
- `full_barrier_` (ClusterTransactionBarrier): initialized with `num_producers = 1`
- `empty_barrier_` (ClusterBarrier): initialized with `num_consumers = 256`
- Warp 0 performs the init; `fence_barrier_init()` ensures visibility

Producer APIs:
- `producer_acquire(state)`: waits on empty_barrier (consumers released), then leader does `full_barrier.arrive_and_expect_tx(tx_bytes)`
- TMA copy instruction completes the transaction (hardware signals full_barrier)
- `producer_commit(state)`: NOP for TMA (hardware commit)
- `producer_tail(state)`: drain remaining stages

Consumer APIs:
- `consumer_wait(state)`: waits on full_barrier (TMA load complete)
- `consumer_release(state)`: arrives at empty_barrier (signals producer stage is free)

### Consumer-Only Synchronization

The epilogue needs a consumer-only sync after STSM writes (before TMA store). `__syncthreads()` requires all 384 threads, but the producer may still be in its tail drain. Solution: use a NamedBarrier scoped to the 256 consumer threads.

```cpp
cutlass::arch::NamedBarrier consumer_barrier(256, /*id=*/6);
// After STSM writes:
consumer_barrier.sync();
```

NamedBarrier ID assignment:
- IDs 2–3: reserved for intra-WG sync (if needed later)
- ID 6: consumer-only barrier (256 threads)

### SharedStorage

```cpp
template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB, int Stages>
struct SharedStorage {
    alignas(128) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
    alignas(128) cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
    typename cutlass::PipelineTmaAsync<Stages>::SharedStorage pipeline;
};
```

The `PipelineTmaAsync::SharedStorage` contains:
- `full_barrier_[Stages]`: ClusterTransactionBarrier array (producer barrier)
- `empty_barrier_[Stages]`: ClusterBarrier array (consumer barrier)

Total smem: ~144 KB (same as baseline — pipeline barriers add only ~192 bytes).

### Host Function Changes

```cpp
dim3 dimBlock(384);  // was 256
// __launch_bounds__ changes from 256 to 384
// Everything else unchanged: TMA descriptors, smem layouts, MMA layout, STSM epilogue
```

The host function creates the same TMA descriptors, smem layouts, TiledMMA, r2s_copy, and s2r atoms. The kernel launch increases from 256 to 384 threads.

### Kernel Signature

Unchanged from `bf16_gemm_sm80_pipe_tma.cu`. The template parameters and function arguments remain the same. The kernel internally determines the warp group role.

## What Stays the Same

- MMA atom: `SM80_16x8x16_F32BF16BF16F32_TN`
- MMA layout: `Layout<Shape<_4, _2>>` (8 warps = 256 threads)
- Tile override: `Tile<Underscore, _64, Underscore>`
- CTA tile: (256, 128, 64)
- Pipeline depth: 3 stages
- S2R atoms: `SM75_U32x4_LDSM_N`
- S2R tiled copy: `make_tiled_copy_A/B`
- Smem layouts: `GMMA::Layout_K_SW128_Atom` for A/B, plain column-major for C
- STSM register→smem copy: `make_tiled_copy_C(SM90_U16x8_STSM_T, mma)`
- TMA store: `make_tma_copy(SM90_TMA_STORE{}, mC, sC_layout, ...)`
- Epilogue: alpha/beta scaling, F32→BF16, STSM, TMA store
- Grid: `dim3(ceil_div(M,bM), ceil_div(N,bN))`

## Kernel Pseudocode

```cpp
__global__ void bf16_gemm_ws_device(...) {
    // All threads
    extern __shared__ char shared_memory[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);

    int warp_group_idx = canonical_warp_group_idx();  // 0, 1, or 2
    int warp_group_thread_idx = threadIdx.x % 128;

    // Setup pipeline (role-based params)
    PipelineTmaAsync pipeline(smem.pipeline, params, Layout<_1>{});  // single-CTA cluster
    __syncthreads();

    if (warp_group_idx == 2) {
        // ---- Producer ----
        warpgroup_reg_dealloc<40>();
        int lane_predicate = elect_one_sync();

        auto smem_pipe_write = make_producer_start_state<MainloopPipeline>();

        // Prologue + Main loop
        CUTE_NO_UNROLL
        for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
            pipeline.producer_acquire(smem_pipe_write);
            if (lane_predicate) {
                copy(tma_a.with(pipeline.producer_get_barrier(smem_pipe_write)),
                     tAgA(_,k_tile_next), tAsA(_,smem_pipe_write.index()));
                copy(tma_b.with(pipeline.producer_get_barrier(smem_pipe_write)),
                     tBgB(_,k_tile_next), tBsB(_,smem_pipe_write.index()));
            }
            ++smem_pipe_write;
            ++k_tile_next;
        }

        // Tail drain
        pipeline.producer_tail(smem_pipe_write);

        // Wait for epilogue TMA store
        tma_store_wait<0>();

    } else {
        // ---- Consumer (wg 0 or 1) ----
        warpgroup_reg_alloc<232>();

        int consumer_tid = threadIdx.x % 256;  // 0-255 for both WGs
        ThrMMA thr_mma = mma.get_thread_slice(consumer_tid);
        // Setup LDSM, STSM, accumulators using consumer_tid (same as current kernel)

        PipelineState smem_pipe_read;
        PipelineState smem_pipe_release;

        // Prologue: wait for first stage, LDSM first k_block
        pipeline.consumer_wait(smem_pipe_read);
        if (K_BLOCK_MAX > 1) {
            // LDSM k_block 0 from first stage
            copy(s2r_atom_a, tXsA(_,_,0,smem_pipe_read.index()), tXrA(_,_,0));
            copy(s2r_atom_b, tXsB(_,_,0,smem_pipe_read.index()), tXrB(_,_,0));
        }

        // Main loop — adapted from current kernel's inner loop
        // Key changes from non-specialized version:
        //   - Replace ProducerBarType::wait with pipeline.consumer_wait
        //   - Replace TMA load (threadIdx.x==0) with pipeline.consumer_release
        //   - Remove __syncthreads (pipeline barriers replace it)
        CUTE_NO_UNROLL
        for (int k_tile_count = ...; k_tile_count > -(K_PIPE_MAX - 1); --k_tile_count)
        {
            CUTE_UNROLL
            for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
            {
                if (k_block == K_BLOCK_MAX - 1)
                {
                    // Wait for next pipe stage (TMA load complete)
                    pipeline.consumer_wait(smem_pipe_read);
                }

                // Prefetch next k_block via LDSM
                auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;
                copy(s2r_atom_a, tXsA_p(_,_,k_block_next), tXrA(_,_,k_block_next));
                copy(s2r_atom_b, tXsB_p(_,_,k_block_next), tXrB(_,_,k_block_next));

                if (k_block == 0)
                {
                    // Release previous stage so producer can reuse it
                    pipeline.consumer_release(smem_pipe_release);
                    ++smem_pipe_release;
                }

                // MMA on current k_block
                gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
            }
            ++smem_pipe_read;
        }

        // Epilogue (consumer-only, same as current kernel)
        // Alpha/beta scaling + F32→BF16 + STSM to plain sC
        cutlass::arch::NamedBarrier consumer_sync(256, 6);
        consumer_sync.sync();  // consumer-only: all 256 consumer threads

        // TMA store (thread 0 of consumers)
        if (consumer_tid == 0) {
            tma_store_fence();
            copy(tma_store_c, tSsC, tSgC);
            tma_store_arrive();
        }
        tma_store_wait<0>();  // all 384 threads
    }
}
```

## Smem Usage

| Buffer | Size | Notes |
|--------|------|-------|
| sA (3 stages) | ~96 KB | GMMA::Layout_K_SW128, reused by sC after MMA |
| sB (3 stages) | ~48 KB | GMMA::Layout_K_SW128 |
| PipelineTmaAsync storage | ~48 B | 3 × (full_barrier + empty_barrier), each 8 bytes |
| sC (epilogue) | 64 KB | Plain column-major, reuses sA's smem |
| **Total** | ~144 KB | Same as baseline |

Occupancy: 1 block/SM × 384 threads.

## Testing

- Verify correctness at 1024³ with BF16 C (tolerance 0.5f, same as current)
- Benchmark at 512³, 1024³, 2048³, 4096³, 8192³
- Compare performance with non-specialized TMA kernel (385 TFLOP/s at 8192³)
- Expect: improved throughput from overlapped TMA load and MMA (producer issues next TMA load while consumer does MMA on current stage)
