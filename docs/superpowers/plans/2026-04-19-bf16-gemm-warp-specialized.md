# BF16 GEMM Warp-Specialized Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add warp-group specialization to the TMA GEMM kernel, splitting 384 threads into 1 producer WG (TMA loads) and 2 consumer WGs (SM80 HMMA + epilogue).

**Architecture:** Copy `bf16_gemm_sm80_pipe_tma.cu` to `bf16_gemm_sm80_pipe_tma_ws.cu`. Replace manual `ClusterTransactionBarrier` + phase tracking with CUTLASS's `PipelineTmaAsync<Stages>`. The kernel dispatches by `canonical_warp_group_idx()`: WG2=producer, WG0/WG1=consumer. Consumer-only epilogue sync uses `NamedBarrier(256, 6)`.

**Tech Stack:** CUDA C++17, CuTe layout algebra, CUTLASS PipelineTmaAsync, SM90 setmaxnreg, SM90 NamedBarrier

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `cute-reference/samples/bf16_gemm_sm80_pipe_tma_ws.cu` | Create | New warp-specialized kernel (based on pipe_tma.cu) |
| `cute-reference/samples/build_and_run.sh` | Modify | Add WS kernel build + run entries |

## Reference Files

- **Source:** `cute-reference/samples/bf16_gemm_sm80_pipe_tma.cu` — base file to copy from
- **Spec:** `docs/superpowers/specs/2026-04-19-bf16-gemm-warp-specialized-design.md` — design decisions

---

### Task 1: Copy base file and update includes + SharedStorage

**Files:**
- Create: `cute-reference/samples/bf16_gemm_sm80_pipe_tma_ws.cu` (copy of `bf16_gemm_sm80_pipe_tma.cu`)

- [ ] **Step 1: Copy the base file**

```bash
cd /data/lmdeploy-cute/cute-reference/samples
cp bf16_gemm_sm80_pipe_tma.cu bf16_gemm_sm80_pipe_tma_ws.cu
```

- [ ] **Step 2: Add includes for warp-group specialization**

After line 37 (`#include "cutlass/arch/barrier.h"`), add:

```cpp
#include "cutlass/arch/reg_reconfig.hpp"
#include "cutlass/pipeline/sm90_pipeline.hpp"
```

- [ ] **Step 3: Update file header comment (lines 2-8)**

Change from:
```
 * BF16 GEMM using SM80 tensor cores with CuTe — TMA Load + TMA Store
 *
 * A variant of bf16_gemm_sm80_pipe_epilogue.cu that replaces:
 *   - cp.async G2S with SM90 TMA load (1 thread issues bulk tensor copy via descriptor)
 *   - Vectorized S2G epilogue with SM90 TMA store
```

To:
```
 * BF16 GEMM using SM80 tensor cores with CuTe — Warp-Specialized TMA Load + TMA Store
 *
 * A variant of bf16_gemm_sm80_pipe_tma.cu that adds warp-group specialization:
 *   - 384 threads: 1 producer warp group (TMA loads) + 2 consumer warp groups (HMMA + epilogue)
 *   - CUTLASS PipelineTmaAsync for producer-consumer synchronization
 *   - Consumer-only NamedBarrier for epilogue sync
 *   - setmaxnreg register reallocation between producer/consumer warp groups
```

- [ ] **Step 4: Update SharedStorage struct (lines 42-60)**

Change from:
```cpp
// Encapsulates shared memory allocation for both A and B matrices plus TMA barriers.
// Uses CuTe's ArrayEngine which provides properly aligned storage for the layout's elements.
// The tma_barrier array holds one ClusterTransactionBarrier per pipeline stage — TMA thread
// arrives, hardware signals completion via transaction bytes.

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage
{
  alignas(128) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;

  // TMA load barriers — one per pipeline stage
  // ClusterTransactionBarrier: TMA thread arrives, hardware signals completion via Tx bytes
  uint64_t tma_barrier[cute::size<2>(SmemLayoutA{})];
};
```

To:
```cpp
// Encapsulates shared memory allocation for both A and B matrices plus pipeline barriers.
// Uses CuTe's ArrayEngine which provides properly aligned storage for the layout's elements.
// PipelineTmaAsync::SharedStorage contains full_barrier_[Stages] + empty_barrier_[Stages],
// replacing the manual tma_barrier array from the non-specialized kernel.

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB, int Stages>
struct SharedStorage
{
  alignas(128) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
  typename cutlass::PipelineTmaAsync<Stages>::SharedStorage pipeline;
};
```

- [ ] **Step 5: Update epilogue comment block (lines 77-79)**

Change from:
```
 *   1. Element-wise alpha/beta scaling
 *   2. F32->BF16 conversion, STSM write to smem
 *   3. TMA store: thread 0 issues bulk smem->gmem copy via TMA descriptor
```

To:
```
 *   1. Element-wise alpha/beta scaling (consumer threads only)
 *   2. F32->BF16 conversion, STSM write to smem (consumer threads only)
 *   3. Consumer-only NamedBarrier sync (256 threads)
 *   4. TMA store: thread 0 of consumers issues bulk smem->gmem copy
```

- [ ] **Step 6: Update __launch_bounds__ (line 88)**

Change from:
```cpp
__launch_bounds__(decltype(size(TiledMma{}))::value)
```

To:
```cpp
__launch_bounds__(decltype(size(TiledMma{}))::value * 3 / 2, 1)  // 384 threads, 1 block/SM
```

Note: `size(TiledMma{}) = 256`, so `256 * 3 / 2 = 384`. This keeps the MMA thread count as the source of truth.

- [ ] **Step 7: Commit**

```bash
cd /data/lmdeploy-cute
git add cute-reference/samples/bf16_gemm_sm80_pipe_tma_ws.cu
git commit -m "Copy TMA kernel as base for warp-specialized version with updated includes and SharedStorage"
```

---

### Task 2: Rewrite kernel function — warp group dispatch and pipeline setup

**Files:**
- Modify: `cute-reference/samples/bf16_gemm_sm80_pipe_tma_ws.cu`

This task replaces the entire kernel body from Step 3b (barrier init) through Step 5 (main loop) with the warp-specialized version. Lines 98-113 (preconditions) and lines 115-153 (global tensors, smem tensors, TMA partitioning) stay the same.

- [ ] **Step 1: Replace barrier init + TMA prefetch + MMA setup + main loop (lines 154-281)**

Replace everything from line 154 (`// ---- Step 3b: Barrier init and TMA prefetch ----`) through line 281 (closing brace of the while loop) with:

```cpp
  // ---- Step 3b: Pipeline setup and warp group dispatch ----
  //
  // PipelineTmaAsync replaces the manual ClusterTransactionBarrier + phase tracking.
  // All 384 threads construct the pipeline (barrier init happens in constructor for warp 0).
  // The role (Producer/Consumer) is set based on warp_group_idx.

  auto K_PIPE_MAX = size<1>(tAsA);   // = bP = 3
  int k_tile_count = size<1>(tAgA);  // total K-tiles
  int k_tile_next  = 0;

  constexpr int tma_transaction_bytes = sizeof(make_tensor_like(tensor<0>(tAsA)))
                                      + sizeof(make_tensor_like(tensor<0>(tBsB)));

  int warp_group_idx = cutlass::canonical_warp_group_idx();        // 0, 1, or 2
  int warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

  // Pipeline params — role depends on warp group
  using MainloopPipeline = cutlass::PipelineTmaAsync<cute::size<2>(SmemLayoutA{})>;
  typename MainloopPipeline::Params pipeline_params;
  if (warp_group_idx == 2) {
    pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
  } else {
    pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
  }
  pipeline_params.is_leader = (warp_group_thread_idx == 0);
  pipeline_params.num_consumers = 256;   // both consumer WGs
  pipeline_params.num_producers = 1;     // single TMA thread
  pipeline_params.transaction_bytes = tma_transaction_bytes;

  // Constructor initializes barriers (warp 0) + fence_barrier_init
  MainloopPipeline pipeline(smem.pipeline, pipeline_params, cute::Layout<cute::_1>{});
  __syncthreads();

  if (warp_group_idx == 2) {
    // ==================================================================
    // Producer warp group — TMA loads
    // ==================================================================
    cutlass::arch::warpgroup_reg_dealloc<40>();

    // Only the leader thread (warp_group_thread_idx == 0) runs the producer loop.
    if (warp_group_thread_idx == 0) {
      auto smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
      using BarrierType = typename MainloopPipeline::ProducerBarrierType;

      CUTE_NO_UNROLL
      for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
        pipeline.producer_acquire(smem_pipe_write);

        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
        copy(tma_a.with(*tma_barrier), tAgA(_,k_tile_next), tAsA(_,smem_pipe_write.index()));
        copy(tma_b.with(*tma_barrier), tBgB(_,k_tile_next), tBsB(_,smem_pipe_write.index()));

        ++smem_pipe_write;
        ++k_tile_next;
      }

      pipeline.producer_tail(smem_pipe_write);
    }

    // All 128 producer threads wait for epilogue TMA store to complete
    cute::tma_store_wait<0>();

  } else {
    // ==================================================================
    // Consumer warp groups (wg 0 and 1) — LDSM + MMA + epilogue
    // ==================================================================
    cutlass::arch::warpgroup_reg_alloc<232>();

    // ---- Step 4: TiledMMA setup and register allocation ----

    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);

    Tensor tCgC = thr_mma.partition_C(gC);                                 // (MMA, MMA_M, MMA_N)
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));                  // (MMA, MMA_M, MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));                  // (MMA, MMA_N, MMA_K)
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);                           // (MMA, MMA_M, MMA_N)
    clear(tCrC);

    // ---- Step 4b: S2R (smem->register) copy setup ----

    TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
    ThrCopy  thr_s2r_a   = s2r_copy_a.get_slice(threadIdx.x);
    Tensor tXsA = thr_s2r_a.partition_S(sA);                               // (CPY, MMA_M, MMA_K, PIPE)
    Tensor tXrA = thr_s2r_a.retile_D(tCrA);                               // (CPY, MMA_M, MMA_K)

    TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
    ThrCopy  thr_s2r_b   = s2r_copy_b.get_slice(threadIdx.x);
    Tensor tXsB = thr_s2r_b.partition_S(sB);                               // (CPY, MMA_N, MMA_K, PIPE)
    Tensor tXrB = thr_s2r_b.retile_D(tCrB);                               // (CPY, MMA_N, MMA_K)

    // ---- Step 4c: R2S (register->smem) STSM copy setup ----

    ThrCopy thr_r2s = r2s_copy.get_slice(threadIdx.x);

    // ---- Step 5: Pipelined main loop ----
    //
    // Consumer pipeline: wait for TMA load → LDSM prefetch → MMA → release stage.
    // smem_pipe_read and smem_pipe_release track pipeline state with phase.
    // smem_pipe_release lags behind smem_pipe_read — release happens at k_block==0
    // of the NEXT k_tile, after all k_blocks of the current stage are in registers.

    typename MainloopPipeline::PipelineState smem_pipe_read;
    typename MainloopPipeline::PipelineState smem_pipe_release;

    Tensor tXsA_p = tXsA(_,_,_,smem_pipe_read.index());
    Tensor tXsB_p = tXsB(_,_,_,smem_pipe_read.index());

    auto K_BLOCK_MAX = size<2>(tCrA);

    // Prologue: wait for first stage, prefetch k_block 0
    pipeline.consumer_wait(smem_pipe_read);
    if (K_BLOCK_MAX > 1) {
      copy(s2r_atom_a, tXsA_p(_,_,Int<0>{}), tXrA(_,_,Int<0>{}));
      copy(s2r_atom_b, tXsB_p(_,_,Int<0>{}), tXrB(_,_,Int<0>{}));
    }

    // Adjust k_tile_count for pipeline depth. In the non-WS kernel, the prologue
    // loaded K_PIPE_MAX-1 stages and decremented k_tile_count by K_PIPE_MAX-1.
    // Here the producer handles all loads, but the consumer still needs the same
    // accounting: total k_tiles - (K_PIPE_MAX-1) real iterations + K_PIPE_MAX-1
    // tail drain = total k_tiles main loop iterations.
    k_tile_count -= (K_PIPE_MAX - 1);

    // Main loop — same structure as non-WS kernel but with pipeline barriers
    // replacing manual ClusterTransactionBarrier + __syncthreads
    CUTE_NO_UNROLL
    while (k_tile_count > -(K_PIPE_MAX - 1))
    {
      CUTE_UNROLL
      for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
      {
        if (k_block == K_BLOCK_MAX - 1)
        {
          // Advance to next stage, then wait for its TMA load to complete
          ++smem_pipe_read;
          pipeline.consumer_wait(smem_pipe_read);
          tXsA_p = tXsA(_,_,_,smem_pipe_read.index());
          tXsB_p = tXsB(_,_,_,smem_pipe_read.index());
        }

        auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;
        copy(s2r_atom_a, tXsA_p(_,_,k_block_next), tXrA(_,_,k_block_next));
        copy(s2r_atom_b, tXsB_p(_,_,k_block_next), tXrB(_,_,k_block_next));

        if (k_block == 0)
        {
          // Release previous stage — safe because all k_blocks are in registers
          pipeline.consumer_release(smem_pipe_release);
          ++smem_pipe_release;
        }

        gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
      }
      --k_tile_count;
    }
```

Note: This block does NOT close the `} else {` — the epilogue (Task 3) will close it.

- [ ] **Step 2: Commit**

```bash
cd /data/lmdeploy-cute
git add cute-reference/samples/bf16_gemm_sm80_pipe_tma_ws.cu
git commit -m "Add warp group dispatch, PipelineTmaAsync, producer/consumer main loop"
```

---

### Task 3: Rewrite epilogue with NamedBarrier + close kernel function

**Files:**
- Modify: `cute-reference/samples/bf16_gemm_sm80_pipe_tma_ws.cu`

This task replaces the epilogue section (originally lines 283-342) with the consumer-only version using NamedBarrier. It also closes the `} else {` consumer branch opened in Task 2.

- [ ] **Step 1: Replace the epilogue section**

The old epilogue starts at the line with `// ---- Step 6: Epilogue ----` and ends at the closing `}` of the kernel function. Replace that entire section with:

```cpp
    // ---- Step 6: Epilogue ----
    //
    // Consumer-only epilogue. Producer threads are waiting at tma_store_wait<0>().

    // Stage 1: Element-wise alpha/beta scaling
    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
      tCrC(i) = alpha * tCrC(i) + beta * static_cast<float>(tCgC(i));
    }

    // Stage 2: Convert F32 -> BF16, write to smem via STSM
    Tensor sC = make_tensor(
        make_smem_ptr(reinterpret_cast<bf16_t*>(smem.A.begin())),
        SmemLayoutC{});

    Tensor tCrC_bf16 = make_tensor<bf16_t>(tCrC.layout());
    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
      tCrC_bf16(i) = static_cast<bf16_t>(tCrC(i));
    }

    Tensor tRS_rAcc = thr_r2s.retile_S(tCrC_bf16);
    Tensor tRS_sC   = thr_r2s.partition_D(sC);
    copy(r2s_copy, tRS_rAcc, tRS_sC);

    // Consumer-only sync after STSM writes — NamedBarrier(256 threads, id=6)
    // Replaces __syncthreads() which requires all 384 threads.
    cutlass::arch::NamedBarrier consumer_sync(256, 6);
    consumer_sync.sync();

    // Stage 3: TMA store (smem -> gmem)
    auto cta_tile_mn = product_each(shape(SmemLayoutC{}));
    Tensor mC_tma = tma_store_c.get_tma_tensor(make_shape(M, N));
    Tensor gC_tma_full = flat_divide(mC_tma, cta_tile_mn);

    auto cta_tma_store = tma_store_c.get_slice(Int<0>{});
    Tensor tSsC_x = cta_tma_store.partition_S(sC);
    Tensor tSgC_x = cta_tma_store.partition_D(gC_tma_full);

    Tensor tSgC = group_modes<1, rank(tSgC_x)>(tSgC_x);
    Tensor tSsC = group_modes<1, rank(tSsC_x)>(tSsC_x);

    int rest_idx = blockIdx.x + blockIdx.y * gridDim.x;

    if (threadIdx.x == 0) {
      tma_store_fence();
      copy(tma_store_c, tSsC(_, 0), tSgC(_, rest_idx));
      tma_store_arrive();
    }
    tma_store_wait<0>();  // all 384 threads participate (producer is also waiting)
  }  // end consumer else-branch
}
```

- [ ] **Step 2: Commit**

```bash
cd /data/lmdeploy-cute
git add cute-reference/samples/bf16_gemm_sm80_pipe_tma_ws.cu
git commit -m "Add consumer-only epilogue with NamedBarrier sync and TMA store"
```

---

### Task 4: Update host function for 384 threads

**Files:**
- Modify: `cute-reference/samples/bf16_gemm_sm80_pipe_tma_ws.cu`

The host function needs three changes: `dimBlock(384)`, updated `SharedStorage` template with `Stages` parameter, and the `__launch_bounds__` is already handled in Task 1.

- [ ] **Step 1: Update dimBlock**

Find the line:
```cpp
  dim3 dimBlock(size(mma));
```

Change to:
```cpp
  dim3 dimBlock(size(mma) * 3 / 2);  // 384 threads: 256 MMA + 128 producer
```

- [ ] **Step 2: Update SharedStorage template instantiation**

Find the line:
```cpp
  int smem_size = int(sizeof(SharedStorage<bf16_t, bf16_t, decltype(sA), decltype(sB)>));
```

Change to:
```cpp
  int smem_size = int(sizeof(SharedStorage<bf16_t, bf16_t, decltype(sA), decltype(sB), cute::size<2>(decltype(sA){})>));
```

Note: `size<2>(decltype(sA){})` extracts the pipeline stages from the smem layout (same as `bP = 3`).

- [ ] **Step 3: Commit**

```bash
cd /data/lmdeploy-cute
git add cute-reference/samples/bf16_gemm_sm80_pipe_tma_ws.cu
git commit -m "Update host function: 384 threads, SharedStorage with Stages param"
```

---

### Task 5: Update main() banner and build_and_run.sh

**Files:**
- Modify: `cute-reference/samples/bf16_gemm_sm80_pipe_tma_ws.cu`
- Modify: `cute-reference/samples/build_and_run.sh`

- [ ] **Step 1: Update main() banner**

Find the line:
```cpp
  printf("BF16 GEMM (SM80 HMMA + SM90 TMA load/store, tile 256x128x64, 256 threads, STSM+TMAStore epilogue)\n\n");
```

Change to:
```cpp
  printf("BF16 GEMM (SM80 HMMA + SM90 TMA load/store, tile 256x128x64, 384 threads WS, PipelineTmaAsync)\n\n");
```

- [ ] **Step 2: Add WS kernel entries to build_and_run.sh**

At the end of `build_and_run.sh`, append:

```bash

echo ""
echo "Compiling bf16_gemm_sm80_pipe_tma_ws.cu (warp-specialized TMA load/store) ..."
nvcc -std=c++17 -arch=sm_90a \
     -I${CUTLASS_INC} \
     bf16_gemm_sm80_pipe_tma_ws.cu \
     -o bf16_gemm_sm80_pipe_tma_ws

echo ""
echo "=== Running warp-specialized TMA load/store sample ==="
echo "--- 1024x1024x1024 ---"
./bf16_gemm_sm80_pipe_tma_ws 1024 1024 1024
```

- [ ] **Step 3: Commit**

```bash
cd /data/lmdeploy-cute
git add cute-reference/samples/bf16_gemm_sm80_pipe_tma_ws.cu cute-reference/samples/build_and_run.sh
git commit -m "Update banner and build script for warp-specialized kernel"
```

---

### Task 6: Compile, test correctness, and benchmark

**Files:**
- Modify: `cute-reference/samples/bf16_gemm_sm80_pipe_tma_ws.cu` (only if fixes needed)

- [ ] **Step 1: Compile**

Run:
```bash
cd /data/lmdeploy-cute/cute-reference/samples && nvcc -std=c++17 -arch=sm_90a -I/data/lmdeploy-cute/build/_deps/repo-cutlass-src/include bf16_gemm_sm80_pipe_tma_ws.cu -o bf16_gemm_sm80_pipe_tma_ws 2>&1 | grep -E "error|Error" | head -20
```

Expected: No errors (warnings about constexpr are OK).

- [ ] **Step 2: Run correctness test**

Run:
```bash
cd /data/lmdeploy-cute/cute-reference/samples && ./bf16_gemm_sm80_pipe_tma_ws 1024 1024 1024 2>&1 | head -5
```

Expected: `max error ... — PASS` with max error < 0.5

- [ ] **Step 3: Run benchmark**

Run:
```bash
cd /data/lmdeploy-cute/cute-reference/samples && ./bf16_gemm_sm80_pipe_tma_ws 2>&1
```

Expected: Similar or better TFLOP/s compared to non-specialized TMA kernel (389 TFLOP/s at 8192^3). The warp-specialized kernel should show improved throughput from overlapped TMA load and MMA.

- [ ] **Step 4: Commit if any fixes were needed**

```bash
cd /data/lmdeploy-cute
git add cute-reference/samples/bf16_gemm_sm80_pipe_tma_ws.cu
git commit -m "Fix warp-specialized kernel compilation/test issues"
```
