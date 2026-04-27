# Optimized RS GEMM with k_block Double-Buffering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create sample 12 that closes the 4-8% performance gap between the RS variant (sample 11) and the SS variant (sample 09) by implementing k_block double-buffering in the consumer mainloop.

**Architecture:** Copy sample 11 as the base. Add a separate tiled copy for the S2R path via `make_tiled_copy_A(AutoVectorizingCopy)`. Restructure the consumer mainloop to split each K=64 tile into 4 k_blocks of K=16, overlapping the smem-to-register copy of k_block+1 with the WGMMA of k_block using `warpgroup_wait<2>()`. Everything else (producer, epilogue, fragments, tile sizes) stays identical.

**Tech Stack:** CUDA (SM90), CuTe, CUTLASS pipeline, WGMMA RS atom

---

### Task 1: Create sample 12 source file

**Files:**
- Create: `cute-reference/samples/12_bf16_gemm_sm90_pipe_tma_ws_persistent_rs_opt.cu`

- [ ] **Step 1: Copy sample 11 as the starting point**

```bash
cp cute-reference/samples/11_bf16_gemm_sm90_pipe_tma_ws_persistent_rs.cu \
   cute-reference/samples/12_bf16_gemm_sm90_pipe_tma_ws_persistent_rs_opt.cu
```

- [ ] **Step 2: Update the file header comment (lines 1-29)**

Replace lines 1-29 with:

```cpp
/***************************************************************************************************
 * BF16 GEMM using SM90 WGMMA tensor cores — RS Optimized Variant
 *
 * An optimized variant of 11_bf16_gemm_sm90_pipe_tma_ws_persistent_rs.cu that adds k_block
 * double-buffering to overlap the S2R (smem-to-register) copy of operand A with WGMMA compute.
 *
 * Key optimization over sample 11 (naive RS):
 *   - K=64 tile split into 4 k_blocks of K=16 (matching WGMMA atom K dimension)
 *   - S2R copy for k_block+1 overlaps with WGMMA for k_block (double-buffering)
 *   - warpgroup_wait<2>() allows 2 in-flight WGMMA instructions for overlap
 *
 * Unchanged from sample 11:
 *   - Tile size 128x256x64, PipelineTmaAsync 3-stage
 *   - Persistent scheduling, warp-specialized (384 threads)
 *   - STSM BF16 epilogue
 *   - Register budget (same A fragment size, no PIPE dimension)
 *
 * C = alpha * A * B^T + beta * C
 *   A: bf16, M x K, row-major (TN layout)
 *   B: bf16, K x N, stored as (N, K) in CuTe with K-contiguous stride
 *   C: bf16, M x N, column-major
 *
 * Target: SM90 (uses SM90 WGMMA + SM90 TMA for gmem<->smem transfers)
 **************************************************************************************************/
```

- [ ] **Step 3: Update the banner in main() (line ~503)**

Change:
```cpp
printf("BF16 GEMM (SM90 WGMMA + TMA load/store, tile 128x256x64, 384 threads WS, PipelineTmaAsync, PERSISTENT, RS variant)\n\n");
```
To:
```cpp
printf("BF16 GEMM (SM90 WGMMA + TMA load/store, tile 128x256x64, 384 threads WS, PipelineTmaAsync, PERSISTENT, RS OPT kblock)\n\n");
```

- [ ] **Step 4: Commit**

```bash
git add cute-reference/samples/12_bf16_gemm_sm90_pipe_tma_ws_persistent_rs_opt.cu
git commit -m "Copy sample 11 as starting point for sample 12 (optimized RS)"
```

---

### Task 2: Add tiled copy A setup

This replaces sample 11's naive `copy(tCsA, tCrA)` with a proper tiled copy that enables k_block-by-k_block transfers.

**Files:**
- Modify: `cute-reference/samples/12_bf16_gemm_sm90_pipe_tma_ws_persistent_rs_opt.cu`

- [ ] **Step 1: Replace the consumer fragment setup (lines 210-225)**

Replace lines 210-225 with:

```cpp
    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);

    // RS variant: A is register-source (ALayout_64x16), B is smem-source (GMMA descriptor).
    // Fragment created WITHOUT the PIPE dimension to keep register pressure low.
    Tensor tCsA = thr_mma.partition_A(sA);                                  // (MMA, MMA_M, MMA_K, PIPE)
    Tensor tCsB = thr_mma.partition_B(sB);                                  // (MMA, MMA_N, MMA_K, PIPE)
    Tensor tCrA = thr_mma.make_fragment_A(tCsA(_,_,_,Int<0>{}));            // (MMA, MMA_M, MMA_K) — NO PIPE
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);                            // GMMA descriptors (has PIPE)

    // Tiled copy for k_block-level S2R transfers.
    // make_tiled_copy_A creates a copy-specific thread partition of A that enables
    // vectorized smem-to-register loads for individual k_blocks.
    auto smem_tiled_copy_A = make_tiled_copy_A(
        Copy_Atom<AutoVectorizingCopy, bf16_t>{}, mma);
    auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(threadIdx.x);
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);                 // (CPY, CPY_M, CPY_K)
    Tensor tCsA_copy_view = smem_thr_copy_A.partition_S(sA);                // (CPY, CPY_M, CPY_K, PIPE)

    constexpr int k_block_count = size<2>(tCrA);  // 4 for BF16 with K=64 (MMA_K_atom=16)
```

This adds 5 new lines of setup compared to sample 11, creating the tiled copy infrastructure needed for k_block-level S2R copies.

- [ ] **Step 2: Commit**

```bash
git add cute-reference/samples/12_bf16_gemm_sm90_pipe_tma_ws_persistent_rs_opt.cu
git commit -m "Add tiled copy A setup for k_block S2R transfers"
```

---

### Task 3: Rewrite the consumer mainloop with k_block double-buffering

This is the core optimization. The mainloop transforms from "copy all of A, then gemm all of A" to "copy k_block+1 overlapped with gemm k_block".

**Files:**
- Modify: `cute-reference/samples/12_bf16_gemm_sm90_pipe_tma_ws_persistent_rs_opt.cu`

- [ ] **Step 1: Replace the mainloop (lines ~267-290)**

Replace the entire per-tile main loop (from `// ---- Step 5:` through the closing `}` of the `for` loop at line 290) with:

```cpp
      // ---- Step 5: Pipelined main loop with k_block double-buffering ----

      CUTE_NO_UNROLL
      for (int k_tile_iter = 0; k_tile_iter < k_tile_count; ++k_tile_iter)
      {
        pipeline.consumer_wait(smem_pipe_read);
        int read_stage = smem_pipe_read.index();
        ++smem_pipe_read;

        // Load first k_block of this pipeline stage
        copy(smem_tiled_copy_A,
             tCsA_copy_view(_,_,0,read_stage),
             tCrA_copy_view(_,_,0));

        warpgroup_fence_operand(tCrC);

        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < k_block_count; ++k_block)
        {
          // Prefetch next k_block (overlaps with current WGMMA via warpgroup_wait<2>)
          if (k_block < k_block_count - 1) {
            copy(smem_tiled_copy_A,
                 tCsA_copy_view(_,_,k_block+1,read_stage),
                 tCrA_copy_view(_,_,k_block+1));
          }

          warpgroup_arrive();
          gemm(mma, tCrA(_,_,k_block),
                    tCrB(_,_,k_block,read_stage), tCrC);
          warpgroup_commit_batch();
          warpgroup_wait<2>();
        }

        warpgroup_fence_operand(tCrC);

        pipeline.consumer_release(smem_pipe_release);
        ++smem_pipe_release;
      }
```

Key differences from sample 11's mainloop:
- `copy(tCsA, tCrA)` for entire K replaced by `copy(tCsA_copy_view(_,_,k_block,stage), tCrA_copy_view(_,_,k_block))` per k_block
- `gemm(mma, tCrA, tCrB, tCrC)` for entire K replaced by `gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block,stage), tCrC)` per k_block
- `warpgroup_wait<0>()` replaced by `warpgroup_wait<2>()` to allow overlap
- New inner `CUTLASS_PRAGMA_UNROLL` loop over k_blocks

- [ ] **Step 2: Commit**

```bash
git add cute-reference/samples/12_bf16_gemm_sm90_pipe_tma_ws_persistent_rs_opt.cu
git commit -m "Rewrite consumer mainloop with k_block double-buffering"
```

---

### Task 4: Add to build system and compile

**Files:**
- Modify: `cute-reference/samples/CMakeLists.txt`

- [ ] **Step 1: Add sample 12 to CMakeLists.txt**

Add `12_bf16_gemm_sm90_pipe_tma_ws_persistent_rs_opt` to the `CUTE_GEMM_SAMPLES` list in `cute-reference/samples/CMakeLists.txt` after the sample 11 entry. The result should be:

```cmake
set(CUTE_GEMM_SAMPLES
    01_bf16_gemm_sm80
    02_bf16_gemm_sm80_opt
    03_bf16_gemm_sm80_pipe
    04_bf16_gemm_sm80_pipe_256x128
    05_bf16_gemm_sm80_pipe_epilogue
    06_bf16_gemm_sm80_pipe_tma
    07_bf16_gemm_sm80_pipe_tma_ws
    08_bf16_gemm_sm80_pipe_tma_ws_persistent
    09_bf16_gemm_sm90_pipe_tma_ws_persistent
    10_bf16_gemm_sm90_pipe_tma_ws_persistent_mcast
    11_bf16_gemm_sm90_pipe_tma_ws_persistent_rs
    12_bf16_gemm_sm90_pipe_tma_ws_persistent_rs_opt
    test_mcast_shapes
)
```

- [ ] **Step 2: Build**

```bash
cd build && ninja bin/12_bf16_gemm_sm90_pipe_tma_ws_persistent_rs_opt
```

Expected: compilation succeeds with **0 bytes spill stores, 0 bytes spill loads**. Register count should be similar to sample 11 (~168 registers).

- [ ] **Step 3: Commit**

```bash
git add cute-reference/samples/CMakeLists.txt
git commit -m "Add sample 12 (optimized RS) to build"
```

---

### Task 5: Test correctness and benchmark

- [ ] **Step 1: Run correctness test**

```bash
CUDA_VISIBLE_DEVICES=0 ./build/bin/12_bf16_gemm_sm90_pipe_tma_ws_persistent_rs_opt
```

Expected: `Correctness (1024^3): max error 1.250305e-01 — PASS` (same tolerance as sample 11).

The benchmark numbers will also print. Record them for comparison.

- [ ] **Step 2: Run sample 11 for comparison**

```bash
CUDA_VISIBLE_DEVICES=0 ./build/bin/11_bf16_gemm_sm90_pipe_tma_ws_persistent_rs
```

- [ ] **Step 3: Run sample 09 for SS baseline**

```bash
CUDA_VISIBLE_DEVICES=0 ./build/bin/09_bf16_gemm_sm90_pipe_tma_ws_persistent
```

- [ ] **Step 4: Compare results**

Verify sample 12's GFLOP/s is closer to sample 09 than sample 11 was. Expected improvement: closing most of the 4-8% gap.

If correctness fails, debug by comparing the k_block loop logic against CUTLASS reference at `build/_deps/repo-cutlass-src/include/cutlass/gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized.hpp` lines 617-628.

If the binary hangs (no output), the pipeline state management is wrong — check that `smem_pipe_read` is advanced correctly and `consumer_release` matches the producer's expectations.

- [ ] **Step 5: Commit benchmark results to git notes (optional)**

No code changes in this step — just verification.
