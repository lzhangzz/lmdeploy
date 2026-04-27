# SM90 WGMMA RS Variant (Sample 11) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create sample 11, an educational variant of sample 09 that swaps WGMMA atom from SS to RS (A in registers, B in shared memory).

**Architecture:** Copy sample 09, swap the MMA atom, add an S2R copy for operand A before each gemm call. Everything else (tile sizes, TMA, pipeline, persistent scheduling, STSM epilogue) stays the same.

**Tech Stack:** CUDA 12.8, CuTe (CUTLASS), SM90 WGMMA, TMA, PipelineTmaAsync

---

### Task 1: Copy sample 09 to sample 11

**Files:**
- Create: `cute-reference/samples/11_bf16_gemm_sm90_pipe_tma_ws_persistent_rs.cu`

- [ ] **Step 1: Copy the file**

```bash
cp cute-reference/samples/09_bf16_gemm_sm90_pipe_tma_ws_persistent.cu cute-reference/samples/11_bf16_gemm_sm90_pipe_tma_ws_persistent_rs.cu
```

- [ ] **Step 2: Commit**

```bash
git add cute-reference/samples/11_bf16_gemm_sm90_pipe_tma_ws_persistent_rs.cu
git commit -m "Copy sample 09 as starting point for sample 11 (RS variant)"
```

---

### Task 2: Update file header

**Files:**
- Modify: `cute-reference/samples/11_bf16_gemm_sm90_pipe_tma_ws_persistent_rs.cu:1-22`

- [ ] **Step 1: Replace the header comment block**

Replace lines 1-22:

```cpp
/***************************************************************************************************
 * BF16 GEMM using SM90 WGMMA tensor cores with CuTe — Persistent Warp-Specialized TMA GEMM
 *
 * A persistent variant of 08_bf16_gemm_sm80_pipe_tma_ws_persistent.cu that replaces SM80 HMMA
 * with SM90 WGMMA (warpgroup matrix multiply-accumulate). WGMMA reads operands directly from
 * smem via 64-bit GMMA descriptors, eliminating LDSM copies and the k_block inner loop.
 *
 * Key changes from 08:
 *   - SM90 WGMMA atom (64x256x16_SS) replaces SM80 HMMA atom (16x8x16)
 *   - Tile size 128x256 (was 256x128) — 2 warpgroups of 64x256 each, high arithmetic intensity
 *   - No S2R (smem-to-register) copies — WGMMA reads smem via descriptors
 *   - No k_block inner loop — single gemm() call per pipeline stage
 *   - warpgroup_arrive/commit_batch/wait replaces manual mma.sync scheduling
 *   - Smem ~213 KB (128x256x64 tile, 3 pipeline stages)
 *
 * C = alpha * A * B^T + beta * C
 *   A: bf16, M x K, row-major (TN layout)
 *   B: bf16, K x N, stored as (N, K) in CuTe with K-contiguous stride
 *   C: bf16, M x N, column-major
 *
 * Target: SM90 (uses SM90 WGMMA + SM90 TMA for gmem<->smem transfers)
 **************************************************************************************************/
```

with:

```cpp
/***************************************************************************************************
 * BF16 GEMM using SM90 WGMMA tensor cores — RS Variant (A in Registers, B in Shared Memory)
 *
 * An educational variant of 09_bf16_gemm_sm90_pipe_tma_ws_persistent.cu that swaps the WGMMA
 * atom from SS (both operands in shared memory via GMMA descriptors) to RS (A in registers,
 * B in shared memory via descriptor). This demonstrates the register-vs-descriptor trade-off in
 * SM90 WGMMA.
 *
 * Key changes from 09 (SS variant):
 *   - WGMMA atom: RS (64x256x16_F32BF16BF16_RS) replaces SS (64x256x16_F32BF16BF16_SS)
 *   - S2R copy for A operand — A is loaded from smem into registers before each gemm call
 *     (RS requires register-source A with GMMA::Major::K layout)
 *   - B operand unchanged — still uses GMMA smem descriptor
 *   - Higher register pressure — A uses ~64B of register storage per thread (vs ~32B for SS descriptors)
 *
 * Unchanged from 09:
 *   - Tile size 128x256x64 (2 warpgroups of 64x256 each)
 *   - TMA load (A, B) + TMA store (C), PipelineTmaAsync 3-stage
 *   - Persistent scheduling, warp-specialized (384 threads: 128 producer + 256 consumer)
 *   - STSM BF16 epilogue
 *   - Smem ~213 KB (128x256x64 tile, 3 pipeline stages)
 *
 * C = alpha * A * B^T + beta * C
 *   A: bf16, M x K, row-major (TN layout)
 *   B: bf16, K x N, stored as (N, K) in CuTe with K-contiguous stride
 *   C: bf16, M x N, column-major
 *
 * Target: SM90 (uses SM90 WGMMA + SM90 TMA for gmem<->smem transfers)
 **************************************************************************************************/
```

- [ ] **Step 2: Commit**

```bash
git add cute-reference/samples/11_bf16_gemm_sm90_pipe_tma_ws_persistent_rs.cu
git commit -m "Update header for RS variant (sample 11)"
```

---

### Task 3: Update consumer setup comments (device kernel)

**Files:**
- Modify: `cute-reference/samples/11_bf16_gemm_sm90_pipe_tma_ws_persistent_rs.cu` (lines ~201-210)

- [ ] **Step 1: Replace consumer TiledMMA setup comment**

Replace:
```cpp
    // ---- Step 4: TiledMMA setup and register allocation (done once) ----

    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);

    // WGMMA: partition smem into GMMA descriptors (no register-based A/B fragments)
    Tensor tCsA = thr_mma.partition_A(sA);                                  // (MMA, MMA_M, MMA_K, PIPE)
    Tensor tCsB = thr_mma.partition_B(sB);                                  // (MMA, MMA_N, MMA_K, PIPE)
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);                            // GMMA descriptors
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);                            // GMMA descriptors
```

with:
```cpp
    // ---- Step 4: TiledMMA setup and register allocation (done once) ----

    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);

    // RS variant: A is register-source (ALayout_64x16), B is smem-source (GMMA descriptor).
    // partition_A yields smem views for each thread's S2R copy (no descriptor).
    // make_fragment_A allocates register storage — 32 bf16 values/thread (4 K-tiles x 8 val).
    // B is unchanged from SS: 4 GMMA descriptors (one per K-tile of 16).
    Tensor tCsA = thr_mma.partition_A(sA);                                  // (MMA, MMA_M, MMA_K, PIPE)
    Tensor tCsB = thr_mma.partition_B(sB);                                  // (MMA, MMA_N, MMA_K, PIPE)
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);                            // register bf16 fragment
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);                            // GMMA descriptors
```

- [ ] **Step 2: Commit**

```bash
git add cute-reference/samples/11_bf16_gemm_sm90_pipe_tma_ws_persistent_rs.cu
git commit -m "Update consumer setup comments for RS variant"
```

---

### Task 4: Add S2R copy for A in consumer main loop

**Files:**
- Modify: `cute-reference/samples/11_bf16_gemm_sm90_pipe_tma_ws_persistent_rs.cu` (the pipelined main loop, around the gemm call)

- [ ] **Step 1: Add S2R copy before gemm**

Replace the gemm section in the pipelined main loop:

Change from:
```cpp
        pipeline.consumer_wait(smem_pipe_read);

        warpgroup_fence_operand(tCrC);
        warpgroup_arrive();
        gemm(mma, tCrA(_,_,_,smem_pipe_read.index()),
                  tCrB(_,_,_,smem_pipe_read.index()), tCrC);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tCrC);
```

to:
```cpp
        pipeline.consumer_wait(smem_pipe_read);

        // S2R copy: load A from swizzled smem into registers.
        // tCsA is partitioned via ALayout_64x16 — each thread reads its 32 bf16
        // values (4 K-tiles x 8 val) from the correct swizzled smem addresses.
        copy(tCsA(_,_,_,smem_pipe_read.index()), tCrA);

        warpgroup_fence_operand(tCrC);
        warpgroup_arrive();
        // tCrA: register values (no pipe index — already loaded above)
        // tCrB: GMMA descriptors (pipe-indexed per K-tile)
        gemm(mma, tCrA,
                  tCrB(_,_,_,smem_pipe_read.index()), tCrC);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tCrC);
```

- [ ] **Step 2: Commit**

```bash
git add cute-reference/samples/11_bf16_gemm_sm90_pipe_tma_ws_persistent_rs.cu
git commit -m "Add S2R copy for A operand in RS variant"
```

---

### Task 5: Swap MMA atom in host function

**Files:**
- Modify: `cute-reference/samples/11_bf16_gemm_sm90_pipe_tma_ws_persistent_rs.cu` (host function, TiledMMA construction)

- [ ] **Step 1: Change MMA atom from SS to RS**

Replace:
```cpp
  // TiledMMA — SM90 WGMMA (warpgroup-level, smem descriptors, no S2R copies)
  TiledMMA mma = make_tiled_mma(
      SM90_64x256x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{},
      Layout<Shape<_2, _1>>{});
```

with:
```cpp
  // TiledMMA — SM90 WGMMA RS variant (A in registers, B via smem descriptor)
  // RS requires A to be K-major (static_assert enforced in the arch atom).
  TiledMMA mma = make_tiled_mma(
      SM90_64x256x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::K>{},
      Layout<Shape<_2, _1>>{});
```

- [ ] **Step 2: Commit**

```bash
git add cute-reference/samples/11_bf16_gemm_sm90_pipe_tma_ws_persistent_rs.cu
git commit -m "Swap MMA atom from SS to RS in sample 11"
```

---

### Task 6: Add sample 11 to CMakeLists.txt

**Files:**
- Modify: `cute-reference/samples/CMakeLists.txt`

- [ ] **Step 1: Add sample 11 to the CUTE_GEMM_SAMPLES list**

Add after line 14 (`10_bf16_gemm_sm90_pipe_tma_ws_persistent_mcast`):

```cmake
    11_bf16_gemm_sm90_pipe_tma_ws_persistent_rs
```

The list should read:
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
    test_mcast_shapes
)
```

- [ ] **Step 2: Commit**

```bash
git add cute-reference/samples/CMakeLists.txt
git commit -m "Add sample 11 (RS variant) to build"
```

---

### Task 7: Build and check register usage

**Files:**
- Read: build output (ptxas info)

- [ ] **Step 1: Configure and build**

```bash
cd build && cmake .. 2>&1 | tail -3 && ninja 11_bf16_gemm_sm90_pipe_tma_ws_persistent_rs 2>&1
```

Expected: Clean compilation (no errors, CuTe warnings OK).

- [ ] **Step 2: Check ptxas register output**

Find the ptxas output line with "registers" for the device kernel. Note the register count per thread.

Expected: somewhat higher than sample 09's 168 registers (see the RS comment). If spilling occurs (ptxas reports spill stores > 0), increase `warpgroup_reg_alloc` in the device kernel and rebuild.

- [ ] **Step 3: If register count OK, commit (no code change needed)**

If register count is acceptable and no spilling, no code change. If spilling, adjust `warpgroup_reg_alloc` and record the new value.

---

### Task 8: Run benchmark and compare against sample 09

**Files:** None (testing only)

- [ ] **Step 1: Check GPU availability**

Use `get_gpu_usage` MCP tool to confirm a GPU is free.

- [ ] **Step 2: Run sample 09 (SS baseline)**

```bash
CUDA_VISIBLE_DEVICES=0 build/bin/09_bf16_gemm_sm90_pipe_tma_ws_persistent 2>&1
```

Expected: correctness PASS, record GFLOP/s numbers for comparison.

- [ ] **Step 3: Run sample 11 (RS variant)**

```bash
CUDA_VISIBLE_DEVICES=0 build/bin/11_bf16_gemm_sm90_pipe_tma_ws_persistent_rs 2>&1
```

Expected: correctness PASS, compare GFLOP/s against sample 09.

- [ ] **Step 4: Compare results**

Note the performance delta between SS and RS. RS is expected to be slower due to smem-to-register copy overhead and higher register pressure.

---

### Task 9: Final bench and record

- [ ] **Step 1: Run both samples side by side**

```bash
echo "=== Sample 09 (SS) ===" && CUDA_VISIBLE_DEVICES=0 build/bin/09_bf16_gemm_sm90_pipe_tma_ws_persistent 2>&1 && echo "" && echo "=== Sample 11 (RS) ===" && CUDA_VISIBLE_DEVICES=0 build/bin/11_bf16_gemm_sm90_pipe_tma_ws_persistent_rs 2>&1
```

- [ ] **Step 2: Verify correctness of both samples**

Both must PASS. If sample 11 fails, debug before proceeding.

---

## Self-Review

**1. Spec coverage:**
- MMA atom swap: Task 5
- Consumer setup comment update: Task 3
- S2R copy in main loop: Task 4
- Register allocation check: Task 7
- Comment updates (header): Task 2
- File creation: Task 1
- CMakeLists: Task 6
- Build & benchmark: Tasks 7-9

**2. Placeholder scan:** No TBDs, TODOs, or incomplete sections. All steps have exact code or commands. Register alloc value gated behind compile check (Task 7) — if spilling, adjust and rebuild.

**3. Type consistency:** MMA type, thread layout, and tile sizes consistent across all tasks. Variable names (`tCsA`, `tCrA`, `tCrB`, `tCrC`) match sample 09 exactly.
