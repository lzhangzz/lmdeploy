# Iteration 06: Threadblock Swizzling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CUTLASS-style threadblock swizzling to the persistent WGMMA kernel to improve L2 cache locality for operand B, closing the performance gap at 4096^3 from 86.2% toward ~95%+ of cuBLAS.

**Architecture:** Replace the row-major tile mapping (`m = linear / n_tiles`, `n = linear % n_tiles`) in both the producer and consumer with a bit-decomposition swizzle that interleaves consecutive CTAs across the M dimension while keeping N shared within swizzle groups. The pack kernel is unchanged.

**Tech Stack:** CUDA 12+, CuTe, CUTLASS pipeline primitives, SM90 WGMMA (Hopper/L20Y)

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `cute-reference/mixed-gemm/06_split_a_pack.h` | Create | Thin wrapper including `05_split_a_pack.h` |
| `cute-reference/mixed-gemm/06_bf16_gemm_sm90_split_a_pack.cu` | Create | Pack test, identical to iter 05 with updated includes/labels |
| `cute-reference/mixed-gemm/06_bf16_gemm_sm90_split_a_wgmma.cu` | Create | WGMMA kernel with swizzled tile mapping (copy of 05 + changes) |
| `cute-reference/mixed-gemm/CMakeLists.txt` | Modify | Add 06 targets |

---

### Task 1: Create boilerplate files and build targets

**Files:**
- Create: `cute-reference/mixed-gemm/06_split_a_pack.h`
- Create: `cute-reference/mixed-gemm/06_bf16_gemm_sm90_split_a_pack.cu`
- Modify: `cute-reference/mixed-gemm/CMakeLists.txt`

- [ ] **Step 1: Create `06_split_a_pack.h`**

Create the thin wrapper header:

```cpp
/***************************************************************************************************
 * Shared pack kernel and host function for split A loading (iteration 06).
 *
 * Pack kernel is unchanged from iteration 05. This header is a thin wrapper.
 * The consumer mainloop (06_bf16_gemm_sm90_split_a_wgmma.cu) adds threadblock swizzling.
 **************************************************************************************************/
#pragma once

#include "05_split_a_pack.h"
```

- [ ] **Step 2: Create `06_bf16_gemm_sm90_split_a_pack.cu`**

Copy `05_bf16_gemm_sm90_split_a_pack.cu` and update:
- Include path: `"05_split_a_pack.h"` → `"06_split_a_pack.h"`
- Printf label: `"iter 05"` → `"iter 06"`

Full file:

```cpp
/***************************************************************************************************
 * Standalone test for the A packing kernel (iteration 06).
 *
 * Pack kernel is unchanged from iteration 05. This test verifies the pack output
 * is still valid when used by the iter 06 consumer kernel.
 *
 * Target: SM90 (uses TMA for gmem->smem, SM90 WGMMA register layout)
 **************************************************************************************************/

#include "06_split_a_pack.h"

int main(int argc, char** argv)
{
  using namespace cute;

  printf("BF16 Split A Pack iter 06 (SM90 TMA + RS WGMMA, 64x16 unit packing)\n\n");

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

Add `06_bf16_gemm_sm90_split_a_pack` and `06_bf16_gemm_sm90_split_a_wgmma` to the target list. The full file becomes:

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

Run: `cd /data/lmdeploy-cute/build && ninja 06_bf16_gemm_sm90_split_a_pack`
Expected: Build succeeds with no errors.

- [ ] **Step 5: Commit boilerplate files**

```bash
git add cute-reference/mixed-gemm/06_split_a_pack.h \
        cute-reference/mixed-gemm/06_bf16_gemm_sm90_split_a_pack.cu \
        cute-reference/mixed-gemm/CMakeLists.txt
git commit -m "Add iter 06 boilerplate: pack wrapper and build targets"
```

---

### Task 2: WGMMA kernel with threadblock swizzling

**Files:**
- Create: `cute-reference/mixed-gemm/06_bf16_gemm_sm90_split_a_wgmma.cu`

This file is a copy of `05_bf16_gemm_sm90_split_a_wgmma.cu` with the following changes:

- [ ] **Step 1: Copy iter 05 WGMMA kernel to iter 06**

```bash
cp cute-reference/mixed-gemm/05_bf16_gemm_sm90_split_a_wgmma.cu \
   cute-reference/mixed-gemm/06_bf16_gemm_sm90_split_a_wgmma.cu
```

- [ ] **Step 2: Update header comment and include**

In `06_bf16_gemm_sm90_split_a_wgmma.cu`, replace the header comment (lines 1-17):

```cpp
/***************************************************************************************************
 * BF16 GEMM using SM90 WGMMA with pre-packed operand A (iteration 06).
 *
 * Pipeline optimization: threadblock swizzling for L2 cache locality.
 *
 * CUTLASS-style bit-decomposition swizzle interleaves consecutive CTAs across
 * the M dimension while keeping N shared within swizzle groups, improving L2
 * reuse of operand B. Consumer mainloop unchanged from iter 05.
 *
 * Changes from iteration 05:
 *   - Swizzled tile mapping in both producer and consumer
 *   - Grid padding: m_tiles rounded up to multiple of swizzle_size
 *   - Bounds check for padded tiles
 *
 * Target: SM90
 **************************************************************************************************/
```

Replace the include on line 19:

```cpp
#include "06_split_a_pack.h"
```

- [ ] **Step 3: Add swizzle parameters to kernel signature**

Replace the kernel signature's last two parameters (line 51):

```cpp
                     int total_tiles, int k_tile_count)
```

with:

```cpp
                     int total_tiles, int k_tile_count,
                     int log_swizzle, int swizzle_size)
```

- [ ] **Step 4: Replace producer tile mapping with swizzle**

Replace lines 117-119 (the producer's tile mapping):

```cpp
      while (linear_idx < total_tiles) {
        int m_idx = linear_idx / n_tiles;
        int n_idx = linear_idx % n_tiles;
```

with:

```cpp
      while (linear_idx < total_tiles) {
        int offset = linear_idx & (swizzle_size - 1);
        int extra  = linear_idx >> log_swizzle;
        int n_idx  = extra % n_tiles;
        int m_idx  = (extra / n_tiles) * swizzle_size + offset;

        if (m_idx >= m_tiles || n_idx >= n_tiles) {
          linear_idx += grid_size;
          continue;
        }
```

- [ ] **Step 5: Replace consumer tile mapping with swizzle**

Replace lines 203-205 (the consumer's tile mapping):

```cpp
    while (linear_idx < total_tiles) {
      int m_idx = linear_idx / n_tiles;
      int n_idx = linear_idx % n_tiles;
```

with:

```cpp
    while (linear_idx < total_tiles) {
      int offset = linear_idx & (swizzle_size - 1);
      int extra  = linear_idx >> log_swizzle;
      int n_idx  = extra % n_tiles;
      int m_idx  = (extra / n_tiles) * swizzle_size + offset;

      if (m_idx >= m_tiles || n_idx >= n_tiles) {
        linear_idx += grid_size;
        continue;
      }
```

- [ ] **Step 6: Update host function — add swizzle heuristic and grid padding**

In the host function `split_a_wgmma`, after computing `total_tiles` and `k_tile_count` (line 386), replace:

```cpp
  int total_tiles = size(ceil_div(M, bM)) * size(ceil_div(N, bN));
  int k_tile_count = size(ceil_div(K, bK));

  dim3 dimBlock(size(mma) * 3 / 2);
  dim3 dimCluster(1, 1, 1);
  dim3 dimGrid(std::min(num_SMs, total_tiles));
```

with:

```cpp
  int m_tiles = size(ceil_div(M, bM));
  int n_tiles = size(ceil_div(N, bN));
  int total_tiles = m_tiles * n_tiles;
  int k_tile_count = size(ceil_div(K, bK));

  // Swizzle heuristic (from CUTLASS get_log_swizzle_size)
  int min_cta_dim = std::min(m_tiles, n_tiles);
  int log_swizzle = 0;
  if (min_cta_dim >= 6)      log_swizzle = 3;
  else if (min_cta_dim >= 3) log_swizzle = 2;
  else if (min_cta_dim >= 2) log_swizzle = 1;
  int swizzle_size = 1 << log_swizzle;

  // Pad m_tiles to multiple of swizzle_size for bijective swizzle mapping.
  // The swizzle groups swizzle_size consecutive M-tiles. If m_tiles is not a
  // multiple of swizzle_size, some tiles in the last group would be missed.
  // Padding m_tiles ensures every valid tile is reachable by the mapping.
  int m_tiles_padded = ((m_tiles + swizzle_size - 1) / swizzle_size) * swizzle_size;
  int total_tiles_padded = m_tiles_padded * n_tiles;

  dim3 dimBlock(size(mma) * 3 / 2);
  dim3 dimCluster(1, 1, 1);
  dim3 dimGrid(std::min(num_SMs, total_tiles_padded));
```

- [ ] **Step 7: Update kernel launch to pass swizzle parameters**

Replace the kernel launch call (line 414):

```cpp
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*)kernel_ptr,
      prob_shape, cta_tiler,
      packed_A, sA,
      B, tmaB,
      C, sC_layout, tma_store_c, r2s_copy, dC, mma,
      alpha, beta,
      total_tiles, k_tile_count);
```

with:

```cpp
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*)kernel_ptr,
      prob_shape, cta_tiler,
      packed_A, sA,
      B, tmaB,
      C, sC_layout, tma_store_c, r2s_copy, dC, mma,
      alpha, beta,
      total_tiles_padded, k_tile_count,
      log_swizzle, swizzle_size);
```

- [ ] **Step 8: Update printf label in main**

Replace line 436:

```cpp
  printf("BF16 Split A WGMMA iter 05 (SM90, k_block interleaving, delayed release)\n\n");
```

with:

```cpp
  printf("BF16 Split A WGMMA iter 06 (SM90, threadblock swizzling)\n\n");
```

- [ ] **Step 9: Build**

Run: `cd /data/lmdeploy-cute/build && ninja 06_bf16_gemm_sm90_split_a_wgmma`
Expected: Build succeeds with no errors.

- [ ] **Step 10: Run correctness tests**

Run: `cd /data/lmdeploy-cute/build && ./06_bf16_gemm_sm90_split_a_wgmma 2>&1 | head -20`
Expected: All tests PASS (single tile through 2048x1024x512). The max errors should match iter 05 results exactly — the swizzle only changes tile ordering, not computation.

- [ ] **Step 11: Run benchmark and compare with iter 05**

Run: `cd /data/lmdeploy-cute/build && ./06_bf16_gemm_sm90_split_a_wgmma 2>&1 | tail -20`
Expected: At 4096^3, performance should be closer to cuBLAS (target ~95%+) vs iter 05's 86.2%. At 8192^3, performance should maintain parity (~97%+).

- [ ] **Step 12: Commit**

```bash
git add cute-reference/mixed-gemm/06_bf16_gemm_sm90_split_a_wgmma.cu
git commit -m "Add iter 06 consumer kernel: threadblock swizzling for L2 locality"
```

---

### Task 3: Update design document with iter 06 results

**Files:**
- Modify: `split-A-loading-rs-degisn.md`

- [ ] **Step 1: Add iter 06 results to the design document**

In `split-A-loading-rs-degisn.md`, after the iter 06 "Target changes" section (after the **Reference:** line at the end), replace the section starting at "### Iteration 06: Threadblock swizzling" with a completed version including results. Use the actual benchmark numbers from step 11.

The template for the results (fill in actual numbers):

```markdown
### Iteration 06: Threadblock swizzling

CUTLASS-style bit-decomposition swizzle for L2 cache locality. Interleaves consecutive
CTAs across the M dimension while keeping N shared within swizzle groups, maximizing B
reuse in L2.

**Files:** `cute-reference/mixed-gemm/06_*`

**Implemented:**

1. **Swizzled tile mapping**: Replaced row-major `m = linear / n_tiles, n = linear % n_tiles`
   with bit-decomposition: `offset = linear & (swizzle_size-1)`, `extra = linear >> log_swizzle`,
   `n = extra % n_tiles`, `m = (extra / n_tiles) * swizzle_size + offset`. Groups of
   `swizzle_size` consecutive CTAs share the same N tile (B reuse in L2).

2. **Swizzle heuristic**: From CUTLASS `get_log_swizzle_size()`. log_swizzle=3 when
   `min(m_tiles, n_tiles) >= 6`, log_swizzle=2 when >= 3, log_swizzle=1 when >= 2.

3. **Grid padding**: Pad `m_tiles` to multiple of `swizzle_size`, then
   `total_tiles_padded = m_tiles_padded * n_tiles`. Bounds check in kernel
   skips invalid tiles from padding.

**Validated:**
- Correctness matches iter 05 (same max errors across all test sizes)
- Performance at 4096^3: **XXX TFLOP/s** (XX.X% of cuBLAS XXX TFLOP/s)
- Performance at 8192^3: **XXX TFLOP/s** (XX.X% of cuBLAS XXX TFLOP/s)
- [compare with iter 05: 679 TFLOP/s (86.2%) at 4096^3, 676 TFLOP/s (97.6%) at 8192^3]
```

- [ ] **Step 2: Commit**

```bash
git add split-A-loading-rs-degisn.md
git commit -m "Add iter 06 performance results"
```
