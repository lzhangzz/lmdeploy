# Iteration 07: Epilogue-Compute Overlap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hide the per-tile TMA store latency by deferring `tma_store_wait<0>()` to the next tile's epilogue, overlapping the store with the next tile's compute.

**Architecture:** Copy the iter 06 WGMMA kernel and move the `tma_store_wait<0>()` call from after the TMA store issue to before the r2s copy (start of epilogue). This allows tile N's TMA store to complete in the background while tile N+1's consumer_wait + S2R + WGMMA executes. Pack kernel is unchanged.

**Tech Stack:** CUDA 12+, CuTe, CUTLASS pipeline primitives, SM90 WGMMA (Hopper/L20Y)

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `cute-reference/mixed-gemm/07_split_a_pack.h` | Create | Thin wrapper including `06_split_a_pack.h` |
| `cute-reference/mixed-gemm/07_bf16_gemm_sm90_split_a_pack.cu` | Create | Pack test, identical to iter 06 with updated includes/labels |
| `cute-reference/mixed-gemm/07_bf16_gemm_sm90_split_a_wgmma.cu` | Create | WGMMA kernel with deferred TMA store wait (copy of 06 + epilogue change) |
| `cute-reference/mixed-gemm/CMakeLists.txt` | Modify | Add 07 targets |

---

### Task 1: Create boilerplate files and build targets

**Files:**
- Create: `cute-reference/mixed-gemm/07_split_a_pack.h`
- Create: `cute-reference/mixed-gemm/07_bf16_gemm_sm90_split_a_pack.cu`
- Modify: `cute-reference/mixed-gemm/CMakeLists.txt`

- [ ] **Step 1: Create `07_split_a_pack.h`**

Create the thin wrapper header:

```cpp
/***************************************************************************************************
 * Shared pack kernel and host function for split A loading (iteration 07).
 *
 * Pack kernel is unchanged from iteration 06. This header is a thin wrapper.
 * The consumer mainloop (07_bf16_gemm_sm90_split_a_wgmma.cu) defers TMA store
 * wait for epilogue-compute overlap.
 **************************************************************************************************/
#pragma once

#include "06_split_a_pack.h"
```

- [ ] **Step 2: Create `07_bf16_gemm_sm90_split_a_pack.cu`**

Copy `06_bf16_gemm_sm90_split_a_pack.cu` and update include path and label:

```cpp
/***************************************************************************************************
 * Standalone test for the A packing kernel (iteration 07).
 *
 * Pack kernel is unchanged from iteration 06. This test verifies the pack output
 * is still valid when used by the iter 07 consumer kernel.
 *
 * Target: SM90 (uses TMA for gmem->smem, SM90 WGMMA register layout)
 **************************************************************************************************/

#include "07_split_a_pack.h"

int main(int argc, char** argv)
{
  using namespace cute;

  printf("BF16 Split A Pack iter 07 (SM90 TMA + RS WGMMA, 64x16 unit packing)\n\n");

  // ---- Test at 1024x1024 ----
  {
    int m = 1024, k = 1024;
    int ldA = k;

    thrust::host_vector<bf16_t> h_A(m * k);
    for (int i = 0; i < m * k; ++i) h_A[i] = static_cast<bf16_t>(2.0 * (rand() / double(RAND_MAX)) - 1.0);

    thrust::device_vector<bf16_t> d_A = h_A;

    // Packed buffer: same total size as A (each element maps to exactly one register slot)
    thrust::device_vector<bf16_t> d_packed(m * k);

    // Run pack kernel
    split_a_pack(m, k, d_A.data().get(), ldA, d_packed.data().get());
    CUTE_CHECK_LAST();

    // Verify: copy packed data back and check all values are valid BF16 (no NaN/Inf from uninit)
    thrust::host_vector<bf16_t> h_packed = d_packed;
    bool has_bad = false;
    for (int i = 0; i < m * k; ++i) {
      float v = float(h_packed[i]);
      if (std::isnan(v) || std::isinf(v)) { has_bad = true; break; }
    }

    printf("Pack test (1024x1024): %s (%d elements)\n\n",
           has_bad ? "FAIL (bad values)" : "PASS (all valid)", m * k);
    if (has_bad) return 1;
  }

  // ---- Benchmark ----
  printf("Pack benchmark:\n");
  for (int size : {256, 512, 1024, 2048, 4096, 8192}) {
    int m = size, k = size;
    int ldA = k;

    thrust::device_vector<bf16_t> d_A(m * k), d_packed(m * k);
    thrust::host_vector<bf16_t> h_A(m * k);
    for (int i = 0; i < m * k; ++i) h_A[i] = static_cast<bf16_t>(2.0 * (rand() / double(RAND_MAX)) - 1.0);
    d_A = h_A;

    const int timing_iterations = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    split_a_pack(m, k, d_A.data().get(), ldA, d_packed.data().get());
    CUTE_CHECK_LAST();

    cudaEventRecord(start);
    for (int i = 0; i < timing_iterations; ++i) {
      split_a_pack(m, k, d_A.data().get(), ldA, d_packed.data().get());
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    double avg_ms = total_ms / timing_iterations;
    double gb = 2.0 * m * k * sizeof(bf16_t) * 1e-9;  // read + write
    printf("  %dx%d: %.1f GB/s (%.4f ms)\n", m, k, gb / (avg_ms * 1e-3), avg_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  return 0;
}
```

- [ ] **Step 3: Update `CMakeLists.txt`**

Add `07_bf16_gemm_sm90_split_a_pack` and `07_bf16_gemm_sm90_split_a_wgmma` to the target list:

```cmake
# Mixed-precision GEMM kernels (split A loading POC)
set(MIXED_GEMM_TARGETS
    01_bf16_gemm_sm90_split_a_pack
    01_bf16_gemm_sm90_split_a_wgmma
    02_bf16_gemm_sm90_split_a_wgmma
    03_bf16_gemm_sm90_split_a_pack
    03_bf16_gemm_sm90_split_a_wgmma
    04_bf16_gemm_sm90_split_a_pack
    04_bf16_gemm_sm90_split_a_wgmma
    05_bf16_gemm_sm90_split_a_pack
    05_bf16_gemm_sm90_split_a_wgmma
    06_bf16_gemm_sm90_split_a_pack
    06_bf16_gemm_sm90_split_a_wgmma
    07_bf16_gemm_sm90_split_a_pack
    07_bf16_gemm_sm90_split_a_wgmma
)

foreach(target ${MIXED_GEMM_TARGETS})
    add_executable(${target} ${target}.cu)
    target_link_libraries(${target} PRIVATE nvidia::cutlass::cutlass cublas)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
    )
    set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES "90a-real")
endforeach()
```

- [ ] **Step 4: Build the pack target to verify compilation**

Run: `cd /data/lmdeploy-cute/build && ninja 07_bf16_gemm_sm90_split_a_pack`
Expected: Build succeeds with no errors.

- [ ] **Step 5: Commit boilerplate files**

```bash
git add cute-reference/mixed-gemm/07_split_a_pack.h \
        cute-reference/mixed-gemm/07_bf16_gemm_sm90_split_a_pack.cu \
        cute-reference/mixed-gemm/CMakeLists.txt
git commit -m "Add iter 07 boilerplate: pack wrapper and build targets"
```

---

### Task 2: WGMMA kernel with deferred TMA store wait

**Files:**
- Create: `cute-reference/mixed-gemm/07_bf16_gemm_sm90_split_a_wgmma.cu`

This file is a copy of `06_bf16_gemm_sm90_split_a_wgmma.cu` with two changes:
1. Updated header comment and include
2. Epilogue: `tma_store_wait<0>()` moved from after TMA store issue to before r2s copy

- [ ] **Step 1: Copy iter 06 WGMMA kernel to iter 07**

```bash
cp cute-reference/mixed-gemm/06_bf16_gemm_sm90_split_a_wgmma.cu \
   cute-reference/mixed-gemm/07_bf16_gemm_sm90_split_a_wgmma.cu
```

- [ ] **Step 2: Update header comment and include**

In `07_bf16_gemm_sm90_split_a_wgmma.cu`, replace lines 1-18 (the header comment and include):

```cpp
/***************************************************************************************************
 * BF16 GEMM using SM90 WGMMA with pre-packed operand A (iteration 07).
 *
 * Pipeline optimization: epilogue-compute overlap via deferred TMA store wait.
 *
 * Defers tma_store_wait<0>() from after the TMA store issue to the start of the
 * next tile's epilogue. This allows tile N's TMA store to complete in the background
 * while tile N+1's compute (consumer_wait + S2R + WGMMA) executes.
 *
 * Changes from iteration 06:
 *   - Epilogue: tma_store_wait<0>() moved to before r2s copy (deferred from previous tile)
 *   - Removed tma_store_wait<0>() after TMA store issue
 *
 * Target: SM90
 **************************************************************************************************/

#include "07_split_a_pack.h"
```

- [ ] **Step 3: Update epilogue — move tma_store_wait to before r2s**

Replace lines 312-340 (the entire EPILOGUE section):

```cpp
      // ================================================================
      // EPILOGUE — deferred TMA store wait for epilogue-compute overlap
      // ================================================================

      // Wait for PREVIOUS tile's TMA store to complete before overwriting sC.
      // For the first tile this is a no-op (no pending stores).
      // For subsequent tiles, the previous tile's TMA store has been overlapping
      // with this tile's compute (consumer_wait + S2R + WGMMA).
      tma_store_wait<0>();

      CUTE_UNROLL
      for (int i = 0; i < size(tCrC); ++i) {
        tCrC(i) = alpha * tCrC(i) + beta * static_cast<float>(tCgC(i));
      }

      Tensor tCrC_bf16 = make_tensor<bf16_t>(tCrC.layout());
      CUTE_UNROLL
      for (int i = 0; i < size(tCrC); ++i) {
        tCrC_bf16(i) = static_cast<bf16_t>(tCrC(i));
      }

      Tensor tRS_rAcc = thr_r2s.retile_S(tCrC_bf16);
      Tensor tRS_sC   = thr_r2s.partition_D(sC);
      copy(r2s_copy, tRS_rAcc, tRS_sC);

      cutlass::arch::NamedBarrier consumer_sync(256, 6);
      consumer_sync.sync();

      int rest_idx = m_idx + n_idx * m_tiles;
      if (threadIdx.x == 0) {
        tma_store_fence();
        copy(tma_store_c, tSsC(_, 0), tSgC(_, rest_idx));
        tma_store_arrive();
      }
      // NO tma_store_wait here — TMA store overlaps with next tile's compute.
      // The producer's tma_store_wait<0>() (line ~160) ensures the last tile's
      // store completes before kernel exit.
```

- [ ] **Step 4: Update printf label in main**

Replace the printf line:

```cpp
  printf("BF16 Split A WGMMA iter 06 (SM90, threadblock swizzling)\n\n");
```

with:

```cpp
  printf("BF16 Split A WGMMA iter 07 (SM90, epilogue-compute overlap)\n\n");
```

- [ ] **Step 5: Build**

Run: `cd /data/lmdeploy-cute/build && ninja 07_bf16_gemm_sm90_split_a_wgmma`
Expected: Build succeeds with no errors.

- [ ] **Step 6: Run correctness tests**

Run: `cd /data/lmdeploy-cute/build && ./07_bf16_gemm_sm90_split_a_wgmma 2>&1 | head -20`
Expected: All tests PASS (single tile through 2048x1024x512). The max errors should match iter 06 results exactly — the deferred wait only changes timing, not computation.

- [ ] **Step 7: Run benchmark and compare with iter 06**

Run: `cd /data/lmdeploy-cute/build && ./07_bf16_gemm_sm90_split_a_wgmma 2>&1 | tail -20`
Expected: Performance at 4096^3 and 8192^3 should be equal to or better than iter 06's ~679 TFLOP/s and ~672 TFLOP/s respectively. The epilogue stall is hidden, so the improvement depends on how much time was spent in tma_store_wait<0>() per tile.

- [ ] **Step 8: Commit**

```bash
git add cute-reference/mixed-gemm/07_bf16_gemm_sm90_split_a_wgmma.cu
git commit -m "Add iter 07 consumer kernel: deferred TMA store wait for epilogue-compute overlap"
```

---

### Task 3: Update design document with iter 07 results

**Files:**
- Modify: `split-A-loading-rs-degisn.md`

- [ ] **Step 1: Add iter 07 results to the design document**

In `split-A-loading-rs-degisn.md`, replace the iter 07 header line:

```markdown
### Iteration 07: Persistent kernel
```

with the completed section using actual benchmark numbers:

```markdown
### Iteration 07: Epilogue-compute overlap

Defers `tma_store_wait<0>()` from after the TMA store issue to the start of the next
tile's epilogue, allowing the TMA store for tile N to overlap with tile N+1's compute
(consumer_wait + S2R + WGMMA).

**Files:** `cute-reference/mixed-gemm/07_*`

**Implemented:**

1. **Deferred TMA store wait**: Moved `tma_store_wait<0>()` from the end of the epilogue
   (after TMA store issue) to the beginning (before r2s copy). The first tile's wait is
   a no-op; subsequent tiles' waits ensure the previous tile's store completed before
   sC is overwritten. The producer's existing `tma_store_wait<0>()` ensures the last
   tile's store completes before kernel exit.

**Validated:**
- Correctness matches iter 06 (same max errors across all test sizes)
- Performance at 4096^3: **XXX TFLOP/s** (XX.X% of cuBLAS XXX TFLOP/s)
- Performance at 8192^3: **XXX TFLOP/s** (XX.X% of cuBLAS XXX TFLOP/s)
- [compare with iter 06: 679 TFLOP/s (85.5%) at 4096^3, 672 TFLOP/s (100.4%) at 8192^3]
```

Fill in the XXX placeholders with actual numbers from step 7.

- [ ] **Step 2: Commit**

```bash
git add split-A-loading-rs-degisn.md
git commit -m "Add iter 07 performance results"
```
