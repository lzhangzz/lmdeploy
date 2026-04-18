# BF16 GEMM STSM Epilogue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a new CuTe BF16 GEMM sample that outputs BF16 C using SM90 STSM epilogue with vectorized 128-bit global stores.

**Architecture:** Copy `bf16_gemm_sm80_pipe_256x128.cu` to a new file. Replace the direct `axpby` F32 register→gmem epilogue with a 3-stage pipeline: element-wise F32 alpha/beta scaling → BF16 conversion + STSM to smem → vectorized 128-bit smem→gmem stores. The MMA loop is completely unchanged.

**Tech Stack:** CuTe, CUDA SM90 (stmatrix), SM80 tensor cores, cp.async pipeline

---

### Task 1: Copy base file and update header + C type to BF16

**Files:**
- Create: `cute-reference/samples/bf16_gemm_sm80_pipe_epilogue.cu` (copy from `bf16_gemm_sm80_pipe_256x128.cu`)
- Modify: `cute-reference/samples/bf16_gemm_sm80_pipe_epilogue.cu`

- [ ] **Step 1: Copy the base file**

```bash
cp cute-reference/samples/bf16_gemm_sm80_pipe_256x128.cu cute-reference/samples/bf16_gemm_sm80_pipe_epilogue.cu
```

- [ ] **Step 2: Update the file header comment**

Replace lines 1–62 (the header comment block) with the new header that describes the STSM epilogue variant. Keep all `#include` lines unchanged. The new header:

```
/***************************************************************************************************
 * BF16 GEMM using SM80 tensor cores with CuTe — STSM Epilogue, BF16 Output
 *
 * A variant of bf16_gemm_sm80_pipe_256x128.cu that adds an SM90 STSM epilogue:
 *   - C output is BF16 instead of F32
 *   - Accumulators (F32) are scaled, converted to BF16, then staged through shared memory
 *     using SM90 stmatrix instructions before vectorized 128-bit global stores
 *
 * Epilogue data flow:
 *   1. Element-wise: F32 accum = alpha * accum + beta * BF16_C_from_gmem (F32 math)
 *   2. Convert F32 → BF16 in registers
 *   3. STSM: stmatrix.sync writes BF16 to smem (reuses sA buffer, 64 KB fits in 96 KB)
 *   4. Vectorized 128-bit stores: smem → gmem (coalesced along M for column-major C)
 *
 * All MMA pipeline features are identical to bf16_gemm_sm80_pipe_256x128.cu:
 *   - (256, 128, 64) CTA tile, 256 threads (8 warps)
 *   - cp.async 3-stage pipeline
 *   - Swizzled smem layouts for bank conflict avoidance
 *
 * C = alpha * A * B^T + beta * C
 *   A: bf16, M x K, row-major (TN layout)
 *   B: bf16, K x N, stored as (N, K) in CuTe with K-contiguous stride
 *   C: bf16, M x N, column-major
 *
 * Target: SM90 (uses SM80 tensor cores + SM90 stmatrix for epilogue)
 **************************************************************************************************/
```

- [ ] **Step 3: Change kernel signature — TC template parameter and C pointer**

In the kernel template (around line 124), change `TC` to `bf16_t` usage. The kernel signature changes:

Replace:
```cpp
template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AG2SCopy, class S2RCopyAtomA,
          class TB, class BStride, class BSmemLayout, class BG2SCopy, class S2RCopyAtomB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class Alpha, class Beta>
```

With:
```cpp
template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AG2SCopy, class S2RCopyAtomA,
          class TB, class BStride, class BSmemLayout, class BG2SCopy, class S2RCopyAtomB,
          class TC, class CStride, class TiledMma,
          class R2SCopy, class S2GCopy,
          class Alpha, class Beta>
```

Note: `CSmemLayout` is removed (sC layout is now internal to the kernel). `R2SCopy` and `S2GCopy` are added.

- [ ] **Step 4: Change kernel function parameter list**

Replace:
```cpp
bf16_gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                 TA const* A, AStride dA, ASmemLayout sA_layout, AG2SCopy g2s_copy_a, S2RCopyAtomA s2r_atom_a,
                 TB const* B, BStride dB, BSmemLayout sB_layout, BG2SCopy g2s_copy_b, S2RCopyAtomB s2r_atom_b,
                 TC      * C, CStride dC, CSmemLayout          , TiledMma mma,
                 Alpha alpha, Beta beta)
```

With:
```cpp
bf16_gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                 TA const* A, AStride dA, ASmemLayout sA_layout, AG2SCopy g2s_copy_a, S2RCopyAtomA s2r_atom_a,
                 TB const* B, BStride dB, BSmemLayout sB_layout, BG2SCopy g2s_copy_b, S2RCopyAtomB s2r_atom_b,
                 TC      * C, CStride dC,                        TiledMma mma,
                 R2SCopy r2s_copy, S2GCopy s2g_copy,
                 Alpha alpha, Beta beta)
```

- [ ] **Step 5: Commit**

```bash
git add cute-reference/samples/bf16_gemm_sm80_pipe_epilogue.cu
git commit -m "Copy 256x128 base file for STSM epilogue variant"
```

---

### Task 2: Rewrite kernel epilogue with STSM + S2G pipeline

**Files:**
- Modify: `cute-reference/samples/bf16_gemm_sm80_pipe_epilogue.cu` (lines ~293–455)

This task replaces Step 4 (TiledMMA setup) through Step 6 (Epilogue) in the kernel device function. Steps 1–3 (global tensors, smem tensors, G2S partitioning, prefetch) remain unchanged.

- [ ] **Step 1: Add static_asserts for new TiledCopy parameters**

After the existing `CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));` (around line 166), add:

```cpp
  // Verify the R2S and S2G TiledCopy involve the same number of threads as the TiledMMA.
  CUTE_STATIC_ASSERT_V(size(r2s_copy) == size(mma));                    // NumThreads
  CUTE_STATIC_ASSERT_V(size(s2g_copy) == size(mma));                    // NumThreads
```

- [ ] **Step 2: Update the Step 4 section — add epilogue tensors alongside MMA setup**

After the existing `clear(tCrC);` (around line 308), add the epilogue-related tensor setup:

```cpp
  // ---- Step 4c: Epilogue tensor setup (BF16 C, register-to-smem, smem-to-global) ----
  //
  // R2S (register-to-smem) copy setup using STSM.
  // make_tiled_copy_C bridges the MMA accumulator layout with the stmatrix register layout.
  // The STSM atom writes BF16 values from registers to smem in a transposed 8x8 pattern.
  ThrCopy thr_r2s = r2s_copy.get_slice(threadIdx.x);

  // S2G (smem-to-global) copy setup using 128-bit vectorized stores.
  ThrCopy thr_s2g = s2g_copy.get_slice(threadIdx.x);
```

This section just stores the thread slices — the actual smem and gmem tensors for the epilogue are created after the MMA loop.

- [ ] **Step 3: Replace Step 6 (Epilogue)**

Replace the existing Step 6 (the `axpby` epilogue, around lines 449–454):

```cpp
  // ---- Step 6: Epilogue ----
  //
  // Write accumulators to global memory: C = alpha * accum + beta * C
  // axpby does: dst = alpha * src + beta * dst (element-wise)

  axpby(alpha, tCrC, beta, tCgC);
```

With the new 3-stage epilogue:

```cpp
  // ---- Step 6: Epilogue — STSM pipeline (F32 accum → BF16 smem → BF16 gmem) ----
  //
  // Stage 1: Element-wise load existing C (BF16) from gmem and apply alpha/beta scaling.
  // We read BF16 directly from tCgC (partitioned for MMA layout), convert to F32,
  // and blend into the F32 accumulators. This avoids allocating a full F32 register
  // tensor for the loaded C values, keeping peak register pressure at ~148 regs.
  CUTE_UNROLL
  for (int i = 0; i < size(tCrC); ++i) {
    tCrC(i) = alpha * tCrC(i) + beta * static_cast<float>(tCgC(i));
  }

  // Stage 2: Convert F32 accumulators to BF16 and write to smem via STSM.
  // After MMA, sA is no longer needed. Reuse its buffer for sC (64 KB fits in 96 KB).
  // sC has a simple column-major layout (stride-1 in M) suitable for vectorized S2G stores.
  Tensor sC = make_tensor(
      make_smem_ptr(reinterpret_cast<bf16_t*>(smem.A.begin())),
      make_layout(make_shape(bM, bN)));                                    // (256, 128) col-major

  // Create BF16 register tensor matching the MMA accumulator shape and convert.
  Tensor tCrC_bf16 = make_tensor<bf16_t>(tCrC.layout());
  CUTE_UNROLL
  for (int i = 0; i < size(tCrC); ++i) {
    tCrC_bf16(i) = static_cast<bf16_t>(tCrC(i));
  }

  // R2S: Retile BF16 accumulators for STSM register layout and partition smem.
  Tensor tRS_rAcc = thr_r2s.retile_S(tCrC_bf16);                         // BF16 regs in STSM layout
  Tensor tRS_sC   = thr_r2s.partition_D(sC);                             // smem destination

  // Execute the STSM copy: registers -> smem.
  copy(r2s_copy, tRS_rAcc, tRS_sC);
  __syncthreads();

  // Stage 3: Vectorized S2G copy from smem to gmem.
  // The S2G TiledCopy uses AutoVectorizingCopyWithAssumedAlignment<128> for 128-bit stores.
  // Thread layout Layout<Shape<_32,_8>, Stride<_1,_32>> gives coalesced writes:
  //   warp 0 writes 256 contiguous bf16 along M at N=0 (512 bytes, perfectly coalesced).
  Tensor tSG_sC = thr_s2g.partition_S(sC);                                // smem source
  Tensor tSG_gC = thr_s2g.partition_D(gC);                                // gmem destination

  copy(s2g_copy, tSG_sC, tSG_gC);
```

- [ ] **Step 4: Commit**

```bash
git add cute-reference/samples/bf16_gemm_sm80_pipe_epilogue.cu
git commit -m "Add STSM epilogue pipeline with BF16 output to kernel"
```

---

### Task 3: Update host function for BF16 C, STSM, and S2G parameters

**Files:**
- Modify: `cute-reference/samples/bf16_gemm_sm80_pipe_epilogue.cu` (host function ~lines 457–657)

- [ ] **Step 1: Change host function C type to BF16**

Replace the host function signature (around line 477–485):

```cpp
template <class Alpha, class Beta>
void
bf16_gemm_tn(int m, int n, int k,
             Alpha alpha,
             bf16_t const* A, int ldA,
             bf16_t const* B, int ldB,
             Beta beta,
             float* C, int ldC,
             cudaStream_t stream = 0)
```

With:

```cpp
template <class Alpha, class Beta>
void
bf16_gemm_tn(int m, int n, int k,
             Alpha alpha,
             bf16_t const* A, int ldA,
             bf16_t const* B, int ldB,
             Beta beta,
             bf16_t* C, int ldC,
             cudaStream_t stream = 0)
```

- [ ] **Step 2: Remove unused sC layout, add R2S and S2G TiledCopy**

In the host function, replace the sC layout line and add the new TiledCopy atoms. Find (around line 538):

```cpp
  auto sC = make_layout(make_shape(bM, bN));                             // (256, 128) — unused in kernel
```

Replace with:

```cpp
  // R2S (register-to-smem) TiledCopy for STSM epilogue (static)
  //
  // SM90_U16x8_STSM_T: transposed stmatrix for column-major C output.
  //   - 32 threads (1 warp) cooperatively write a transposed 8x8 BF16 matrix to smem
  //   - Each thread provides 4 x uint32 (8 bf16 values)
  //   - Transposed store rearranges from MMA accumulator layout to column-major smem layout
  //   - Selected per CUTLASS's sm90_get_smem_store_op_for_accumulator logic:
  //     sizeof(bf16_t)==2 and stride-1 in M → SM90_U16x8_STSM_T
  //
  // make_tiled_copy_C bridges the MMA's get_layoutC_TV() with the STSM atom's register layout.
  // The tiler matches the full MMA tile: (tile_M, tile_N) = (64, 64).

  auto r2s_copy = make_tiled_copy_C(
      Copy_Atom<SM90_U16x8_STSM_T, bf16_t>{},
      mma);

  // S2G (smem-to-global) TiledCopy for vectorized BF16 stores (static)
  //
  // AutoVectorizingCopyWithAssumedAlignment<128>: compiler-vectorized 128-bit stores.
  // The copy atom has ThrID=Layout<_1> (no warp cooperation), so the TiledCopy's
  // thread layout fully determines the access pattern.
  //
  // Thread layout Layout<Shape<_32, _8>, Stride<_1, _32>>:
  //   - 32 threads in M, 8 in N = 256 threads total
  //   - tid = m + n*32: warp 0 (tid 0-31) all have n=0, m=0..31
  //   - Thread (m, n) covers M=[m*8 .. m*8+7], N=n
  //   - Warp 0 writes 256 contiguous bf16 at N=0 → perfectly coalesced
  //
  // Value layout Layout<Shape<_8, _1>>:
  //   - 8 bf16 contiguous in M (128 bits), 1 in N
  //   - Matches the 128-bit vectorized store width
  //
  // Coverage per copy-tile: (256, 8). Loops 128/8 = 16 times in N.

  auto s2g_copy = make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, bf16_t>{},
      Layout<Shape<_32, _8>, Stride<_1, _32>>{},
      Layout<Shape<_8, _1>>{});
```

- [ ] **Step 3: Update kernel_fptr template arguments**

Replace the kernel_fptr declaration (around lines 627–632):

```cpp
  auto kernel_fptr = bf16_gemm_device<
      decltype(prob_shape), decltype(cta_tiler),
      bf16_t, decltype(dA), decltype(sA), decltype(g2s_copy_a), decltype(s2r_atom_a),
      bf16_t, decltype(dB), decltype(sB), decltype(g2s_copy_b), decltype(s2r_atom_b),
      float, decltype(dC), decltype(sC), decltype(mma),
      Alpha, Beta>;
```

With:

```cpp
  auto kernel_fptr = bf16_gemm_device<
      decltype(prob_shape), decltype(cta_tiler),
      bf16_t, decltype(dA), decltype(sA), decltype(g2s_copy_a), decltype(s2r_atom_a),
      bf16_t, decltype(dB), decltype(sB), decltype(g2s_copy_b), decltype(s2r_atom_b),
      bf16_t, decltype(dC),                        decltype(mma),
      decltype(r2s_copy), decltype(s2g_copy),
      Alpha, Beta>;
```

Note the changes: `float` → `bf16_t` for TC, `decltype(sC)` removed, `decltype(r2s_copy)` and `decltype(s2g_copy)` added.

- [ ] **Step 4: Update kernel launch arguments**

Replace the kernel launch (around lines 651–656):

```cpp
  kernel_fptr<<<dimGrid, dimBlock, smem_size, stream>>>(
      prob_shape, cta_tiler,
      A, dA, sA, g2s_copy_a, s2r_atom_a,
      B, dB, sB, g2s_copy_b, s2r_atom_b,
      C, dC, sC, mma,
      alpha, beta);
```

With:

```cpp
  kernel_fptr<<<dimGrid, dimBlock, smem_size, stream>>>(
      prob_shape, cta_tiler,
      A, dA, sA, g2s_copy_a, s2r_atom_a,
      B, dB, sB, g2s_copy_b, s2r_atom_b,
      C, dC,            mma,
      r2s_copy, s2g_copy,
      alpha, beta);
```

- [ ] **Step 5: Commit**

```bash
git add cute-reference/samples/bf16_gemm_sm80_pipe_epilogue.cu
git commit -m "Update host function: BF16 C, STSM R2S, vectorized S2G TiledCopy"
```

---

### Task 4: Update main() and benchmark for BF16 C output

**Files:**
- Modify: `cute-reference/samples/bf16_gemm_sm80_pipe_epilogue.cu` (main + benchmark ~lines 659–778)

- [ ] **Step 1: Update benchmark_size for BF16 C**

Replace the `benchmark_size` function entirely with:

```cpp
void benchmark_size(int m, int n, int k,
                    float alpha, float beta,
                    cudaStream_t stream)
{
  using namespace cute;

  assert(m % 256 == 0 && n % 128 == 0 && k % 64 == 0);

  int ldA = k, ldB = k, ldC = m;

  thrust::device_vector<bf16_t> d_A(m * k), d_B(n * k), d_C(m * n);
  thrust::host_vector<bf16_t> h_A(m * k), h_B(n * k);
  for (int i = 0; i < m * k; ++i) h_A[i] = static_cast<bf16_t>(2.0 * (rand() / double(RAND_MAX)) - 1.0);
  for (int i = 0; i < n * k; ++i) h_B[i] = static_cast<bf16_t>(2.0 * (rand() / double(RAND_MAX)) - 1.0);
  d_A = h_A; d_B = h_B;

  const int timing_iterations = 100;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warmup
  bf16_gemm_tn(m, n, k, alpha,
               d_A.data().get(), ldA,
               d_B.data().get(), ldB,
               beta,
               d_C.data().get(), ldC, stream);
  CUTE_CHECK_LAST();

  cudaEventRecord(start);
  for (int i = 0; i < timing_iterations; ++i) {
    bf16_gemm_tn(m, n, k, alpha,
                 d_A.data().get(), ldA,
                 d_B.data().get(), ldB,
                 beta,
                 d_C.data().get(), ldC, stream);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float total_ms = 0.0f;
  cudaEventElapsedTime(&total_ms, start, stop);
  double avg_ms = total_ms / timing_iterations;
  double gflops = (2.0 * m * n * k) * 1e-9;
  printf("  %dx%dx%d: %.1f GFLOP/s (%.4f ms)\n", m, n, k, gflops / (avg_ms * 1e-3), avg_ms);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
```

- [ ] **Step 2: Update main() correctness check for BF16 C**

Replace the main function's correctness block and printf. Replace `float` C vectors with `bf16_t`:

```cpp
int main(int argc, char** argv)
{
  using namespace cute;

  printf("BF16 GEMM (SM80, cp.async 3-stage, tile 256x128x64, 256 threads, STSM BF16 epilogue)\n\n");

  float alpha = 1.0f;
  float beta  = 0.0f;

  // ---- Verify correctness once at 1024^3 ----
  {
    int m = 1024, n = 1024, k = 1024;
    int ldA = k, ldB = k, ldC = m;

    thrust::host_vector<bf16_t> h_A(m * k), h_B(n * k);
    thrust::host_vector<bf16_t> h_C(m * n);
    for (int i = 0; i < m * k; ++i) h_A[i] = static_cast<bf16_t>(2.0 * (rand() / double(RAND_MAX)) - 1.0);
    for (int i = 0; i < n * k; ++i) h_B[i] = static_cast<bf16_t>(2.0 * (rand() / double(RAND_MAX)) - 1.0);
    for (int i = 0; i < m * n; ++i) h_C[i] = static_cast<bf16_t>(-1.0f);

    thrust::device_vector<bf16_t> d_A = h_A, d_B = h_B;
    thrust::device_vector<bf16_t> d_C = h_C;

    bf16_gemm_tn(m, n, k, alpha,
                 d_A.data().get(), ldA,
                 d_B.data().get(), ldB,
                 beta,
                 d_C.data().get(), ldC);
    CUTE_CHECK_LAST();

    thrust::host_vector<bf16_t> h_result = d_C;

    // CPU reference: C[m,n] = alpha * sum_k A[m,k] * B[n,k] + beta * C[m,n]
    thrust::host_vector<float> h_ref(m * n, 0.0f);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) {
        float sum = 0.0f;
        for (int l = 0; l < k; ++l)
          sum += float(h_A[i * k + l]) * float(h_B[j * k + l]);
        h_ref[i + j * ldC] = alpha * sum + beta * float(h_C[i + j * ldC]);
      }

    float max_err = 0.0f;
    for (int i = 0; i < m * n; ++i)
      max_err = std::max(max_err, std::abs(float(h_result[i]) - h_ref[i]));

    // BF16 has ~3 decimal digits of precision, so tolerance is larger than F32.
    // With beta=0, the error comes from BF16 input quantization + tensor core rounding.
    printf("Correctness (1024^3): max error %e — %s\n\n", max_err, max_err < 0.1f ? "PASS" : "FAIL");
    if (max_err >= 0.1f) return 1;
  }

  // ---- Benchmark ----
  printf("Benchmark (100 iterations each):\n");
  benchmark_size(512,  512,  512,  alpha, beta, 0);
  benchmark_size(1024, 1024, 1024, alpha, beta, 0);
  benchmark_size(2048, 2048, 2048, alpha, beta, 0);
  benchmark_size(4096, 4096, 4096, alpha, beta, 0);
  benchmark_size(8192, 8192, 8192, alpha, beta, 0);

  return 0;
}
```

Key changes from the 256x128 version:
- `float` C vectors → `bf16_t` C vectors
- Tolerance changed from 0.01f to 0.1f (BF16 precision)
- printf banner updated

- [ ] **Step 3: Commit**

```bash
git add cute-reference/samples/bf16_gemm_sm80_pipe_epilogue.cu
git commit -m "Update main/benchmark for BF16 C output with relaxed tolerance"
```

---

### Task 5: Update build_and_run.sh and compile/test

**Files:**
- Modify: `cute-reference/samples/build_and_run.sh`

- [ ] **Step 1: Add compile and run section for the new sample**

Append to `build_and_run.sh` (after the existing 256x128 section):

```bash
echo ""
echo "Compiling bf16_gemm_sm80_pipe_epilogue.cu (STSM BF16 epilogue) ..."
nvcc -std=c++17 -arch=sm_90a \
     -I${CUTLASS_INC} \
     bf16_gemm_sm80_pipe_epilogue.cu \
     -o bf16_gemm_sm80_pipe_epilogue

echo ""
echo "=== Running STSM epilogue sample (BF16 output) ==="
echo "--- 1024x1024x1024 ---"
./bf16_gemm_sm80_pipe_epilogue 1024 1024 1024
```

- [ ] **Step 2: Compile and verify**

Run:
```bash
cd cute-reference/samples && bash build_and_run.sh 2>&1 | tail -40
```

Expected: compilation succeeds, correctness check passes (max error < 0.1), benchmark runs at 5 sizes.

- [ ] **Step 3: If compilation fails, fix and re-compile**

Common issues:
- Missing `#include <cute/atom/copy_sm90.hpp>` for STSM — add after existing includes if needed
- Template argument mismatch in kernel_fptr — verify types match kernel signature
- `SM90_U16x8_STSM_T` not found — ensure CuTe headers include SM90 copy atoms

- [ ] **Step 4: Commit**

```bash
git add cute-reference/samples/build_and_run.sh
git commit -m "Add STSM epilogue sample to build_and_run.sh"
```

---

### Task 6: Comprehensive comments pass

**Files:**
- Modify: `cute-reference/samples/bf16_gemm_sm80_pipe_epilogue.cu`

- [ ] **Step 1: Review and add comments for the epilogue code**

Read through the new epilogue code and add explanatory comments matching the style of the existing kernel comments (which use `// ----` section headers and detailed block comments). Ensure:

1. The Step 4c section header has a block comment explaining the R2S/S2G setup
2. The Step 6 section header has a block comment explaining the 3-stage epilogue pipeline
3. The host function R2S/S2G sections have block comments explaining the atom selection and layout choices
4. No redundant comments — the MMA loop comments don't need changes

- [ ] **Step 2: Final commit**

```bash
git add cute-reference/samples/bf16_gemm_sm80_pipe_epilogue.cu
git commit -m "Add comprehensive comments for STSM epilogue sample"
```
