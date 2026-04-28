# Iteration 03: Vectorized Pack/Unpack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change the packed A data layout to enable vectorized 128-bit stores (pack kernel) and loads (consumer S2R), using CuTe tensor copy with `AutoVectorizingCopy`.

**Architecture:** New packed format `offset(k, t, j) = k * 2048 + t * 8 + j` makes each thread's 8 registers per k_block contiguous. Pack kernel uses `copy(AutoVectorizingCopy{}, rA, gP)` for vectorized gmem stores. Consumer uses `copy(AutoVectorizingCopy{}, sP, rA)` for vectorized smem-to-register loads. Everything else (bulk copy, TMA pipeline, WGMMA) is unchanged from iter 02.

**Tech Stack:** CUDA 12.8, SM90 (Hopper), CuTe/CUTLASS, `AutoVectorizingCopy`, `PipelineTmaAsync`

---

### Task 1: Create pack kernel header with vectorized stores

**Files:**
- Create: `cute-reference/mixed-gemm/03_split_a_pack.h`
- Reference: `cute-reference/mixed-gemm/01_split_a_pack.h` (copy and modify)

- [ ] **Step 1: Copy iter 01 pack header**

```bash
cp cute-reference/mixed-gemm/01_split_a_pack.h \
   cute-reference/mixed-gemm/03_split_a_pack.h
```

- [ ] **Step 2: Update header comment**

Replace the header comment (lines 1-4):

```cpp
/***************************************************************************************************
 * Shared pack kernel and host function for split A loading (iteration 03).
 *
 * Vectorized version: stores packed A using CuTe copy with AutoVectorizingCopy,
 * producing a (K_BLOCK, THREAD, REG) layout with contiguous 8-bf16 per thread per k_block.
 * Included by 03_bf16_gemm_sm90_split_a_pack.cu and 03_bf16_gemm_sm90_split_a_wgmma.cu.
 **************************************************************************************************/
```

- [ ] **Step 3: Replace the scalar store loop with CuTe vectorized copy**

Replace lines 108-113 (the store loop):

```cpp
    // ---- Write registers to packed gmem buffer (coalesced) ----
    bf16_t* packed_ptr = packed_A + linear_idx * 256 * regs_per_thread;
    CUTE_UNROLL
    for (int i = 0; i < regs_per_thread; ++i) {
      packed_ptr[i * 256 + threadIdx.x] = tCrA(i);
    }
```

With:

```cpp
    // ---- Write registers to packed gmem buffer (vectorized 128-bit stores) ----
    // Packed layout: (K_BLOCK, THREAD, REG) stride (2048, 8, 1)
    // Each thread stores 4 x 128-bit (one per k_block of 8 contiguous bf16)
    Tensor gPacked = make_tensor(make_gmem_ptr(packed_A + linear_idx * 256 * regs_per_thread),
                                  make_shape(Int<4>{}, Int<256>{}, Int<8>{}));
    Tensor gP = gPacked(_, threadIdx.x, _);
    Tensor rA = make_tensor(tCrA.data(), make_layout(make_shape(Int<4>{}, Int<8>{})));
    copy(AutoVectorizingCopy{}, rA, gP);
```

The reshape from `tCrA (8, 1, 4)` to `(4, 8)` works because tCrA's flat layout is `k*8 + j`,
so `make_layout(Shape<_4, _8>{})` with default stride `(8, 1)` maps `rA(k, j)` = `tCrA(j, 0, k)`.

- [ ] **Step 4: Commit**

```bash
git add cute-reference/mixed-gemm/03_split_a_pack.h
git commit -m "Add iter 03 pack header: vectorized 128-bit stores for packed A"
```

---

### Task 2: Create pack test harness

**Files:**
- Create: `cute-reference/mixed-gemm/03_bf16_gemm_sm90_split_a_pack.cu`
- Reference: `cute-reference/mixed-gemm/01_bf16_gemm_sm90_split_a_pack.cu` (copy and modify)

- [ ] **Step 1: Copy iter 01 pack test**

```bash
cp cute-reference/mixed-gemm/01_bf16_gemm_sm90_split_a_pack.cu \
   cute-reference/mixed-gemm/03_bf16_gemm_sm90_split_a_pack.cu
```

- [ ] **Step 2: Update include and banner**

Replace the include (line 11):

```cpp
#include "01_split_a_pack.h"
```

With:

```cpp
#include "03_split_a_pack.h"
```

Replace the banner in main (line 17):

```cpp
  printf("BF16 Split A Pack (SM90 TMA + RS WGMMA register layout)\n\n");
```

With:

```cpp
  printf("BF16 Split A Pack iter 03 (SM90 TMA + RS WGMMA, vectorized 128-bit stores)\n\n");
```

- [ ] **Step 3: Commit**

```bash
git add cute-reference/mixed-gemm/03_bf16_gemm_sm90_split_a_pack.cu
git commit -m "Add iter 03 pack test harness"
```

---

### Task 3: Create consumer kernel with vectorized S2R

**Files:**
- Create: `cute-reference/mixed-gemm/03_bf16_gemm_sm90_split_a_wgmma.cu`
- Reference: `cute-reference/mixed-gemm/02_bf16_gemm_sm90_split_a_wgmma.cu` (copy and modify)

- [ ] **Step 1: Copy iter 02 consumer kernel**

```bash
cp cute-reference/mixed-gemm/02_bf16_gemm_sm90_split_a_wgmma.cu \
   cute-reference/mixed-gemm/03_bf16_gemm_sm90_split_a_wgmma.cu
```

- [ ] **Step 2: Update header comment**

Replace lines 1-14:

```cpp
/***************************************************************************************************
 * BF16 GEMM using SM90 WGMMA with pre-packed operand A (iteration 03).
 *
 * Vectorized version of iteration 02: uses CuTe copy with AutoVectorizingCopy for
 * smem-to-register loads of packed A. New packed format groups each thread's 8 registers
 * per k_block contiguously, enabling zero-bank-conflict 128-bit smem loads.
 *
 * Changes from iteration 02:
 *   - New packed format: (K_BLOCK, THREAD, REG) stride (2048, 8, 1)
 *   - Consumer S2R: CuTe copy(AutoVectorizingCopy, sP, rA) instead of scalar loop
 *
 * Target: SM90
 **************************************************************************************************/
```

- [ ] **Step 3: Update include**

Replace line 16:

```cpp
#include "01_split_a_pack.h"
```

With:

```cpp
#include "03_split_a_pack.h"
```

- [ ] **Step 4: Replace scalar S2R load with CuTe vectorized copy**

Replace the S2R load section (lines 201-206):

```cpp
        // Load packed A from smem pipeline stage into registers
        const bf16_t* smem_a_ptr = smem.A.begin() + read_stage * a_stage_elements;
        CUTE_UNROLL
        for (int i = 0; i < regs_per_thread; ++i) {
          tCrA(i) = smem_a_ptr[i * 256 + threadIdx.x];
        }
```

With:

```cpp
        // Load packed A from smem pipeline stage into registers (vectorized 128-bit loads)
        // Packed layout: (K_BLOCK, THREAD, REG) stride (2048, 8, 1) in smem
        Tensor sA_packed = make_tensor(
            make_smem_ptr(smem.A.begin() + read_stage * a_stage_elements),
            make_shape(Int<4>{}, Int<256>{}, Int<8>{}));
        Tensor sP = sA_packed(_, threadIdx.x, _);
        Tensor rA = make_tensor(tCrA.data(), make_layout(make_shape(Int<4>{}, Int<8>{})));
        copy(AutoVectorizingCopy{}, sP, rA);
```

- [ ] **Step 5: Update banner in main()**

Replace line 365:

```cpp
  printf("BF16 Split A WGMMA iter 02 (SM90, bulk-copy A pipeline, tile 128x256x64, 384t WS, persistent)\n\n");
```

With:

```cpp
  printf("BF16 Split A WGMMA iter 03 (SM90, vectorized pack/unpack, tile 128x256x64, 384t WS, persistent)\n\n");
```

- [ ] **Step 6: Commit**

```bash
git add cute-reference/mixed-gemm/03_bf16_gemm_sm90_split_a_wgmma.cu
git commit -m "Add iter 03 consumer kernel: vectorized 128-bit S2R for packed A"
```

---

### Task 4: Update build system

**Files:**
- Modify: `cute-reference/mixed-gemm/CMakeLists.txt`

- [ ] **Step 1: Add iter 03 targets**

Replace the `MIXED_GEMM_TARGETS` list:

```cmake
set(MIXED_GEMM_TARGETS
    01_bf16_gemm_sm90_split_a_pack
    01_bf16_gemm_sm90_split_a_wgmma
    02_bf16_gemm_sm90_split_a_wgmma
    03_bf16_gemm_sm90_split_a_pack
    03_bf16_gemm_sm90_split_a_wgmma
)
```

- [ ] **Step 2: Commit**

```bash
git add cute-reference/mixed-gemm/CMakeLists.txt
git commit -m "Add iter 03 targets to build"
```

---

### Task 5: Build and verify correctness

- [ ] **Step 1: Reconfigure build**

```bash
cd /data/lmdeploy-cute/build && sh ../my_generate.sh 2>&1 | tail -5
```

Expected: cmake configuration succeeds.

- [ ] **Step 2: Build iter 03 targets**

```bash
cd /data/lmdeploy-cute/build && ninja 03_bf16_gemm_sm90_split_a_pack 03_bf16_gemm_sm90_split_a_wgmma 2>&1
```

Expected: compiles without errors.

- [ ] **Step 3: Check GPU availability**

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
```

Expected: at least one GPU with minimal memory usage (< 1000 MiB used).

- [ ] **Step 4: Run pack test**

```bash
cd /data/lmdeploy-cute/build && ./03_bf16_gemm_sm90_split_a_pack
```

Expected: "PASS (all valid)" and benchmark numbers.

- [ ] **Step 5: Run consumer correctness tests**

```bash
cd /data/lmdeploy-cute/build && ./03_bf16_gemm_sm90_split_a_wgmma
```

Expected: all test sizes (128x256x64 through 2048x1024x512) report PASS with max error < 0.5.

If any test fails, debug and fix before proceeding.

- [ ] **Step 6: Commit (if any fixes were needed)**

```bash
git add cute-reference/mixed-gemm/03_*.h cute-reference/mixed-gemm/03_*.cu
git commit -m "Fix iter 03 correctness: <description>"
```

---

### Task 6: Benchmark and compare

- [ ] **Step 1: Run iter 02 benchmark for comparison**

```bash
cd /data/lmdeploy-cute/build && ninja 02_bf16_gemm_sm90_split_a_wgmma && ./02_bf16_gemm_sm90_split_a_wgmma
```

Record TFLOP/s for each size.

- [ ] **Step 2: Run iter 03 benchmark**

```bash
cd /data/lmdeploy-cute/build && ./03_bf16_gemm_sm90_split_a_wgmma
```

Record TFLOP/s for each size.

- [ ] **Step 3: Update design doc with results**

Update `docs/superpowers/specs/2026-04-28-iter03-vectorized-pack-design.md` with observed performance numbers and comparison with iter 02.

- [ ] **Step 4: Update main design doc**

Update `split-A-loading-rs-degisn.md` iter 03 section with validation results and performance numbers.

- [ ] **Step 5: Commit results**

```bash
git add docs/superpowers/specs/2026-04-28-iter03-vectorized-pack-design.md split-A-loading-rs-degisn.md
git commit -m "Add iter 03 performance results"
```
