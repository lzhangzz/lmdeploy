# Sample 13: Fully Optimized RS Mainloop (CUTLASS-Style)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create sample 13 that closes the performance gap with sample 09 (SS) by restructuring the RS consumer mainloop with CUTLASS's prologue/mainloop/epilogue pattern: early pipeline release at k_block==1 and k_tile-level prefetch overlap.

**Architecture:** Copy sample 12 as the base. Replace the single-loop consumer mainloop with three sections: prologue (first k_tile), mainloop (middle k_tiles), and epilogue (last k_tile). The prologue prefetches the next stage before the mainloop starts. The mainloop releases pipeline buffers at k_block==1 (2 WGMMA instructions sooner than sample 12) and overlaps the wait for the next stage with compute. The epilogue handles the last k_tile without spurious waits.

**Tech Stack:** CUDA SM90 WGMMA, CuTe, CUTLASS PipelineTmaAsync, BF16 arithmetic.

---

### Task 1: Copy sample 12 to sample 13

**Files:**
- Create: `cute-reference/samples/13_bf16_gemm_sm90_pipe_tma_ws_persistent_rs_cutlass.cu` (copy of sample 12)

- [ ] **Step 1: Copy the file**

```bash
cp cute-reference/samples/12_bf16_gemm_sm90_pipe_tma_ws_persistent_rs_opt.cu \
   cute-reference/samples/13_bf16_gemm_sm90_pipe_tma_ws_persistent_rs_cutlass.cu
```

- [ ] **Step 2: Commit**

```bash
git add cute-reference/samples/13_bf16_gemm_sm90_pipe_tma_ws_persistent_rs_cutlass.cu
git commit -m "Copy sample 12 as base for sample 13 (CUTLASS-style RS mainloop)"
```

---

### Task 2: Update header comment and banner

**Files:**
- Modify: `cute-reference/samples/13_bf16_gemm_sm90_pipe_tma_ws_persistent_rs_cutlass.cu` (lines 1-24, line 513)

- [ ] **Step 1: Replace the file header comment (lines 1-24)**

Replace the entire comment block at the top with:

```cpp
/***************************************************************************************************
 * BF16 GEMM using SM90 WGMMA tensor cores — RS CUTLASS-Style Variant
 *
 * An optimized variant of sample 12 that restructures the consumer mainloop to match
 * CUTLASS's canonical RS pattern (sm90_mma_tma_gmma_rs_warpspecialized.hpp):
 *
 *   - Prologue/mainloop/epilogue structure instead of a single k_tile loop
 *   - Early pipeline release at k_block==1 (producer can refill 2 WGMMA sooner)
 *   - k_tile-level prefetch overlap via consumer_try_wait + deferred consumer_wait
 *
 * These optimizations close the performance gap with sample 09 (SS variant) by eliminating
 * producer starvation and pipeline bubbles at k_tile boundaries.
 *
 * Unchanged from sample 12:
 *   - Tile size 128x256x64, PipelineTmaAsync 3-stage
 *   - Persistent scheduling, warp-specialized (384 threads)
 *   - STSM BF16 epilogue, register budget, shared memory layouts
 *
 * C = alpha * A * B^T + beta * C
 *   A: bf16, M x K, row-major (TN layout)
 *   B: bf16, K x N, stored as (N, K) in CuTe with K-contiguous stride
 *   C: bf16, M x N, column-major
 *
 * Target: SM90 (uses SM90 WGMMA + SM90 TMA for gmem<->smem transfers)
 **************************************************************************************************/
```

- [ ] **Step 2: Update the banner print in main() (line 513)**

Replace:
```cpp
  printf("BF16 GEMM (SM90 WGMMA + TMA load/store, tile 128x256x64, 384 threads WS, PipelineTmaAsync, PERSISTENT, RS OPT kblock)\n\n");
```

With:
```cpp
  printf("BF16 GEMM (SM90 WGMMA + TMA load/store, tile 128x256x64, 384 threads WS, PipelineTmaAsync, PERSISTENT, RS CUTLASS)\n\n");
```

- [ ] **Step 3: Commit**

```bash
git add cute-reference/samples/13_bf16_gemm_sm90_pipe_tma_ws_persistent_rs_cutlass.cu
git commit -m "Update header comment and banner for sample 13"
```

---

### Task 3: Restructure the consumer mainloop

**Files:**
- Modify: `cute-reference/samples/13_bf16_gemm_sm90_pipe_tma_ws_persistent_rs_cutlass.cu` (lines 265-300)

This is the core change. Replace the single k_tile loop with CUTLASS's prologue/mainloop/epilogue pattern.

- [ ] **Step 1: Replace the consumer mainloop (lines 265-300)**

Find the section starting with:
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

Replace with:

```cpp
      // ---- Step 5: Pipelined main loop (CUTLASS-style prologue/mainloop/epilogue) ----
      //
      // Three-phase structure matching CUTLASS's sm90_mma_tma_gmma_rs_warpspecialized.hpp:
      //   Prologue: first k_tile — computes on stage 0, prefetches stage 1
      //   Mainloop: middle k_tiles — early release at k_block==1, k_tile prefetch overlap
      //   Epilogue: last k_tile — no next-stage wait, early release at k_block==1

      cutlass::ConsumerToken barrier_token = {cutlass::BarrierStatus::WaitAgain};

      // ---- Prologue: first k_tile ----
      {
        barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
        pipeline.consumer_wait(smem_pipe_read, barrier_token);

        int read_stage = smem_pipe_read.index();
        ++smem_pipe_read;
        barrier_token = pipeline.consumer_try_wait(smem_pipe_read);

        // Load first k_block of stage 0
        copy(smem_tiled_copy_A,
             tCsA_copy_view(_,_,0,read_stage),
             tCrA_copy_view(_,_,0));

        warpgroup_fence_operand(tCrC);

        // k_blocks 0..2 (skip last — it's handled separately below)
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < k_block_count - 1; ++k_block) {
          copy(smem_tiled_copy_A,
               tCsA_copy_view(_,_,k_block+1,read_stage),
               tCrA_copy_view(_,_,k_block+1));

          warpgroup_arrive();
          gemm(mma, tCrA(_,_,k_block),
                    tCrB(_,_,k_block,read_stage), tCrC);
          warpgroup_commit_batch();
        }

        warpgroup_wait<2>();

        // Last k_block of stage 0
        warpgroup_arrive();
        gemm(mma, tCrA(_,_,k_block_count-1),
                  tCrB(_,_,k_block_count-1,read_stage), tCrC);
        warpgroup_commit_batch();

        --k_tile_count;

        if (k_tile_count > 0) {
          // Wait for next stage and prefetch its first k_block.
          // This overlaps the pipeline wait with the in-flight last-k_block WGMMA.
          // NOTE: smem_pipe_read is NOT incremented here. It stays pointing at the
          // prefetched stage so the mainloop can pick it up as read_stage.
          pipeline.consumer_wait(smem_pipe_read, barrier_token);
          copy(smem_tiled_copy_A,
               tCsA_copy_view(_,_,0,smem_pipe_read.index()),
               tCrA_copy_view(_,_,0));
          warpgroup_wait<2>();
        }
      }

      warpgroup_fence_operand(tCrC);

      // ---- Mainloop: middle k_tiles ----
      CUTE_NO_UNROLL
      for (; k_tile_count > 1; --k_tile_count)
      {
        int read_stage = smem_pipe_read.index();
        ++smem_pipe_read;

        warpgroup_fence_operand(tCrC);

        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < k_block_count; ++k_block) {
          if (k_block == 0) {
            // Non-blocking probe for the next stage (smem_pipe_read was
            // already incremented at the top of this k_tile iteration).
            // The token is consumed at k_block == k_block_count-1.
            barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
          }

          if (k_block == k_block_count - 1) {
            // Last k_block: block for the next stage and prefetch its first k_block.
            // This overlaps the pipeline wait with the current WGMMA.
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
            copy(smem_tiled_copy_A,
                 tCsA_copy_view(_,_,0,smem_pipe_read.index()),
                 tCrA_copy_view(_,_,0));
          } else {
            // Normal: prefetch next k_block within current stage.
            copy(smem_tiled_copy_A,
                 tCsA_copy_view(_,_,k_block+1,read_stage),
                 tCrA_copy_view(_,_,k_block+1));
          }

          warpgroup_arrive();
          gemm(mma, tCrA(_,_,k_block),
                    tCrB(_,_,k_block,read_stage), tCrC);
          warpgroup_commit_batch();
          warpgroup_wait<2>();

          if (k_block == 1) {
            // Early release: producer can start refilling this buffer
            // 2 WGMMA instructions sooner than sample 12.
            pipeline.consumer_release(smem_pipe_release);
            ++smem_pipe_release;
          }
        }

        warpgroup_fence_operand(tCrC);
      }

      // ---- Epilogue: last k_tile ----
      if (k_tile_count == 1)
      {
        int read_stage = smem_pipe_read.index();

        warpgroup_fence_operand(tCrC);

        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < k_block_count - 1; ++k_block) {
          copy(smem_tiled_copy_A,
               tCsA_copy_view(_,_,k_block+1,read_stage),
               tCrA_copy_view(_,_,k_block+1));

          warpgroup_arrive();
          gemm(mma, tCrA(_,_,k_block),
                    tCrB(_,_,k_block,read_stage), tCrC);
          warpgroup_commit_batch();
          warpgroup_wait<2>();

          if (k_block == 1) {
            pipeline.consumer_release(smem_pipe_release);
            ++smem_pipe_release;
          }
        }

        // Last k_block (no next stage to prefetch)
        warpgroup_arrive();
        gemm(mma, tCrA(_,_,k_block_count-1),
                  tCrB(_,_,k_block_count-1,read_stage), tCrC);
        warpgroup_commit_batch();

        warpgroup_fence_operand(tCrC);
      }

      // Wait for all in-flight WGMMA to complete, then release the
      // prologue's buffer (which was never released during the mainloop).
      warpgroup_wait<0>();
      pipeline.consumer_release(smem_pipe_release);
      ++smem_pipe_release;
```

- [ ] **Step 2: Commit**

```bash
git add cute-reference/samples/13_bf16_gemm_sm90_pipe_tma_ws_persistent_rs_cutlass.cu
git commit -m "Restructure consumer mainloop to CUTLASS prologue/mainloop/epilogue pattern"
```

---

### Task 4: Add sample 13 to the build system

**Files:**
- Modify: `cute-reference/samples/CMakeLists.txt` (line 16)

- [ ] **Step 1: Add sample 13 to CUTE_GEMM_SAMPLES**

In `cute-reference/samples/CMakeLists.txt`, after line 16 (`12_bf16_gemm_sm90_pipe_tma_ws_persistent_rs_opt`), add:

```cmake
    13_bf16_gemm_sm90_pipe_tma_ws_persistent_rs_cutlass
```

- [ ] **Step 2: Commit**

```bash
git add cute-reference/samples/CMakeLists.txt
git commit -m "Add sample 13 (CUTLASS-style RS) to build"
```

---

### Task 5: Build and verify correctness

- [ ] **Step 1: Reconfigure cmake**

```bash
cd /data/lmdeploy-cute/build && sh ../my_generate.sh
```

Expected: cmake completes without errors, sample 13 target appears.

- [ ] **Step 2: Build sample 13**

```bash
cd /data/lmdeploy-cute/build && ninja 13_bf16_gemm_sm90_pipe_tma_ws_persistent_rs_cutlass
```

Expected: Build succeeds. Check PTXAS output for register count (should be similar to sample 12).

- [ ] **Step 3: Run correctness test**

```bash
cd /data/lmdeploy-cute/build && ./cute-reference/samples/13_bf16_gemm_sm90_pipe_tma_ws_persistent_rs_cutlass
```

Expected: "Correctness (1024^3): max error X — PASS" and benchmark results printed.

- [ ] **Step 4: Fix if correctness fails**

If the correctness check fails (max error >= 0.5), debug the pipeline state tracking. Common issues:
- Missing `smem_pipe_release` increment causing stale buffer indices
- `barrier_token` not being consumed correctly between k_tiles
- `warpgroup_wait<0>()` in tail section missing or placed incorrectly

---

### Task 6: Benchmark and compare with sample 09

- [ ] **Step 1: Run sample 09 benchmark**

```bash
cd /data/lmdeploy-cute/build && ./cute-reference/samples/09_bf16_gemm_sm90_pipe_tma_ws_persistent
```

Record GFLOP/s for each size.

- [ ] **Step 2: Run sample 12 benchmark**

```bash
cd /data/lmdeploy-cute/build && ./cute-reference/samples/12_bf16_gemm_sm90_pipe_tma_ws_persistent_rs_opt
```

Record GFLOP/s for each size.

- [ ] **Step 3: Run sample 13 benchmark**

```bash
cd /data/lmdeploy-cute/build && ./cute-reference/samples/13_bf16_gemm_sm90_pipe_tma_ws_persistent_rs_cutlass
```

Record GFLOP/s for each size.

- [ ] **Step 4: Compare results**

Expected: Sample 13 GFLOP/s should be within 1% of sample 09 at 2048^3 and above. Sample 13 should be measurably faster than sample 12 at all sizes.

If the gap persists, investigate:
- Register spilling (check PTXAS verbose output for spill counts)
- Occupancy differences (compare register counts between samples 09, 12, 13)
- Whether the `consumer_try_wait` is actually non-blocking on the hardware

---

### Task 7: Commit all changes

- [ ] **Step 1: Verify no uncommitted changes**

```bash
cd /data/lmdeploy-cute && git status
```

All changes should already be committed incrementally. If not, commit any remaining files.

- [ ] **Step 2: Verify git log**

```bash
git log --oneline -5
```

Expected: commits for spec, copy, header update, mainloop restructure, CMakeLists update.
