# Restore STSM Epilogue for TMA Store Kernel

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace element-wise register→smem copy with STSM (stmatrix) in the TMA GEMM kernel, matching the spec's epilogue design.

**Architecture:** The current `bf16_gemm_sm80_pipe_tma.cu` uses element-wise `copy(tCrC_bf16, tCsC)` for register→smem. We restore STSM via `make_tiled_copy_C(SM90_U16x8_STSM_T, mma)` — the same pattern proven in `bf16_gemm_sm80_pipe_epilogue.cu`. The plain column-major sC layout and `make_tma_copy` TMA store are unchanged.

**Tech Stack:** CUDA C++17, CuTe layout algebra, SM90 STSM, SM90 TMA store

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `cute-reference/samples/bf16_gemm_sm80_pipe_tma.cu` | Modify | Restore STSM epilogue |

## Reference Files

- **Current file:** `cute-reference/samples/bf16_gemm_sm80_pipe_tma.cu` — the file being modified
- **Reference:** `cute-reference/samples/bf16_gemm_sm80_pipe_epilogue.cu` — proven STSM pattern to copy from
- **Spec:** `docs/superpowers/specs/2026-04-18-bf16-gemm-tma-design.md` — design decisions

---

### Task 1: Add R2SCopy template parameter to kernel signature and setup

**Files:**
- Modify: `cute-reference/samples/bf16_gemm_sm80_pipe_tma.cu`

- [ ] **Step 1: Update kernel template parameters (line 81-86)**

Change from:
```cpp
template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA, class S2RCopyAtomA,
          class TB, class SmemLayoutB, class TmaB, class S2RCopyAtomB,
          class TC, class SmemLayoutC,
          class TmaStoreC, class CStride, class TiledMma,
          class Alpha, class Beta>
```

To:
```cpp
template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA, class S2RCopyAtomA,
          class TB, class SmemLayoutB, class TmaB, class S2RCopyAtomB,
          class TC, class SmemLayoutC,
          class TmaStoreC, class R2SCopy, class CStride, class TiledMma,
          class Alpha, class Beta>
```

- [ ] **Step 2: Update kernel function signature (line 90-96)**

Change from:
```cpp
bf16_gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                 TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a, S2RCopyAtomA s2r_atom_a,
                 TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b, S2RCopyAtomB s2r_atom_b,
                 TC      * C, SmemLayoutC,
                 CUTLASS_GRID_CONSTANT TmaStoreC const tma_store_c,
                 CStride dC, TiledMma mma,
                 Alpha alpha, Beta beta)
```

To:
```cpp
bf16_gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                 TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a, S2RCopyAtomA s2r_atom_a,
                 TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b, S2RCopyAtomB s2r_atom_b,
                 TC      * C, SmemLayoutC,
                 CUTLASS_GRID_CONSTANT TmaStoreC const tma_store_c,
                 R2SCopy r2s_copy, CStride dC, TiledMma mma,
                 Alpha alpha, Beta beta)
```

- [ ] **Step 3: Add STSM thread copy setup after S2R copy setup (after line 205)**

After the `tXrB` line (line 205), add:
```cpp

  // ---- Step 4c: R2S (register->smem) STSM copy setup ----

  ThrCopy thr_r2s = r2s_copy.get_slice(threadIdx.x);
```

- [ ] **Step 4: Commit**

```bash
cd /data/lmdeploy-cute
git add cute-reference/samples/bf16_gemm_sm80_pipe_tma.cu
git commit -m "Add R2SCopy template parameter and thr_r2s setup to TMA kernel"
```

---

### Task 2: Replace element-wise epilogue with STSM

**Files:**
- Modify: `cute-reference/samples/bf16_gemm_sm80_pipe_tma.cu`

- [ ] **Step 1: Update header comment (line 16)**

Change from:
```
 *   - Epilogue: element-wise F32->BF16 to plain smem + TMA store (replaces vectorized S2G)
```

To:
```
 *   - Epilogue: STSM F32->BF16 to plain smem + TMA store (replaces vectorized S2G)
```

- [ ] **Step 2: Update kernel epilogue comment block (lines 77-79)**

Change from:
```
 *   2. F32->BF16 conversion, element-wise write to smem
 *   3. TMA store: thread 0 issues bulk smem->gmem copy via TMA descriptor
```

To:
```
 *   2. F32->BF16 conversion, STSM write to smem
 *   3. TMA store: thread 0 issues bulk smem->gmem copy via TMA descriptor
```

- [ ] **Step 3: Replace Stage 2 epilogue code (lines 287-300)**

Change from:
```cpp
  // Stage 2: Convert F32 -> BF16, write to smem via element-wise copy
  auto sC_base_ptr = make_smem_ptr(reinterpret_cast<bf16_t*>(smem.A.begin()));
  Tensor sC = make_tensor(sC_base_ptr, SmemLayoutC{});

  Tensor tCrC_bf16 = make_tensor<bf16_t>(tCrC.layout());
  CUTE_UNROLL
  for (int i = 0; i < size(tCrC); ++i) {
    tCrC_bf16(i) = static_cast<bf16_t>(tCrC(i));
  }

  // Direct element-wise copy from register to smem
  Tensor tCsC = thr_mma.partition_C(sC);
  copy(tCrC_bf16, tCsC);
  __syncthreads();
```

To:
```cpp
  // Stage 2: Convert F32 -> BF16, write to smem via STSM
  Tensor sC = make_tensor(
      make_smem_ptr(reinterpret_cast<bf16_t*>(smem.A.begin())),
      SmemLayoutC{});

  Tensor tCrC_bf16 = make_tensor<bf16_t>(tCrC.layout());
  CUTE_UNROLL
  for (int i = 0; i < size(tCrC); ++i) {
    tCrC_bf16(i) = static_cast<bf16_t>(tCrC(i));
  }

  // STSM: retile BF16 registers for stmatrix layout, partition smem, copy
  Tensor tRS_rAcc = thr_r2s.retile_S(tCrC_bf16);
  Tensor tRS_sC   = thr_r2s.partition_D(sC);
  copy(r2s_copy, tRS_rAcc, tRS_sC);
  __syncthreads();
```

- [ ] **Step 4: Commit**

```bash
cd /data/lmdeploy-cute
git add cute-reference/samples/bf16_gemm_sm80_pipe_tma.cu
git commit -m "Replace element-wise epilogue with STSM register->smem copy"
```

---

### Task 3: Update host function for STSM

**Files:**
- Modify: `cute-reference/samples/bf16_gemm_sm80_pipe_tma.cu`

- [ ] **Step 1: Update TMA store comment (lines 398-402)**

Change from:
```cpp
  // TMA store TiledCopy for C
  // Uses a plain column-major smem layout. Data is written to smem via element-wise copy
  // (not STSM), so no hardware swizzle is applied. TMA store reads from plain smem.
  Tensor mC_for_tma = make_tensor(C, make_shape(M, N), dC);                 // (M,N) for TMA inspection
  auto tma_store_c = make_tma_copy(SM90_TMA_STORE{}, mC_for_tma, sC_layout, make_shape(bM, bN), Int<1>{});
```

To:
```cpp
  // TMA store TiledCopy for C
  // Uses a plain column-major smem layout. STSM writes to plain smem (no hardware swizzle).
  // TMA store reads from the same plain smem with no swizzle conflicts.
  Tensor mC = make_tensor(C, make_shape(M, N), dC);                         // (M,N) for TMA inspection
  auto tma_store_c = make_tma_copy(SM90_TMA_STORE{}, mC, sC_layout, make_shape(bM, bN), Int<1>{});

  // R2S TiledCopy for STSM register->smem (same as pipe_epilogue.cu)
  auto r2s_copy = make_tiled_copy_C(
      Copy_Atom<SM90_U16x8_STSM_T, bf16_t>{},
      mma);
```

- [ ] **Step 2: Update kernel function pointer template args (lines 425-431)**

Change from:
```cpp
  auto* kernel_ptr = &bf16_gemm_device<
      decltype(prob_shape), decltype(cta_tiler),
      bf16_t, decltype(sA), decltype(tmaA), decltype(s2r_atom_a),
      bf16_t, decltype(sB), decltype(tmaB), decltype(s2r_atom_b),
      bf16_t, decltype(sC_layout),
      decltype(tma_store_c), decltype(dC), decltype(mma),
      Alpha, Beta>;
```

To:
```cpp
  auto* kernel_ptr = &bf16_gemm_device<
      decltype(prob_shape), decltype(cta_tiler),
      bf16_t, decltype(sA), decltype(tmaA), decltype(s2r_atom_a),
      bf16_t, decltype(sB), decltype(tmaB), decltype(s2r_atom_b),
      bf16_t, decltype(sC_layout),
      decltype(tma_store_c), decltype(r2s_copy), decltype(dC), decltype(mma),
      Alpha, Beta>;
```

- [ ] **Step 3: Update kernel launch args (lines 447-452)**

Change from:
```cpp
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*)kernel_ptr,
      prob_shape, cta_tiler,
      A, tmaA, s2r_atom_a,
      B, tmaB, s2r_atom_b,
      C, sC_layout, tma_store_c, dC, mma,
      alpha, beta);
```

To:
```cpp
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*)kernel_ptr,
      prob_shape, cta_tiler,
      A, tmaA, s2r_atom_a,
      B, tmaB, s2r_atom_b,
      C, sC_layout, tma_store_c, r2s_copy, dC, mma,
      alpha, beta);
```

- [ ] **Step 4: Commit**

```bash
cd /data/lmdeploy-cute
git add cute-reference/samples/bf16_gemm_sm80_pipe_tma.cu
git commit -m "Add r2s_copy to host function and kernel launch args"
```

---

### Task 4: Compile, test correctness, and benchmark

**Files:**
- Modify: `cute-reference/samples/bf16_gemm_sm80_pipe_tma.cu` (only if fixes needed)

- [ ] **Step 1: Compile**

Run:
```bash
cd /data/lmdeploy-cute/cute-reference/samples && nvcc -std=c++17 -arch=sm_90a -I/data/lmdeploy-cute/build/_deps/repo-cutlass-src/include bf16_gemm_sm80_pipe_tma.cu -o bf16_gemm_sm80_pipe_tma 2>&1 | grep -E "error|Error" | head -20
```

Expected: No errors (warnings about constexpr are OK).

- [ ] **Step 2: Run correctness test**

Run:
```bash
cd /data/lmdeploy-cute/cute-reference/samples && ./bf16_gemm_sm80_pipe_tma 1024 1024 1024 2>&1 | head -5
```

Expected: `max error ... — PASS` with max error < 0.5

- [ ] **Step 3: Run benchmark**

Run:
```bash
cd /data/lmdeploy-cute/cute-reference/samples && ./bf16_gemm_sm80_pipe_tma 2>&1
```

Expected: Similar or better TFLOP/s compared to the previous element-wise epilogue (389 TFLOP/s at 8192^3).

- [ ] **Step 4: Commit if any fixes were needed**

```bash
cd /data/lmdeploy-cute
git add cute-reference/samples/bf16_gemm_sm80_pipe_tma.cu
git commit -m "Fix STSM+TMA store compilation/test issues"
```

---

### Task 5: Update main banner and build_and_run.sh description

**Files:**
- Modify: `cute-reference/samples/bf16_gemm_sm80_pipe_tma.cu` (line 518)
- Modify: `cute-reference/samples/build_and_run.sh`

- [ ] **Step 1: Update main() banner (line 518)**

Change from:
```cpp
  printf("BF16 GEMM (SM80 HMMA + SM90 TMA load/store, tile 256x128x64, 256 threads, TMAStore epilogue)\n\n");
```

To:
```cpp
  printf("BF16 GEMM (SM80 HMMA + SM90 TMA load/store, tile 256x128x64, 256 threads, STSM+TMAStore epilogue)\n\n");
```

- [ ] **Step 2: Update build_and_run.sh description**

In `cute-reference/samples/build_and_run.sh`, change the TMA section comment from:
```bash
echo "Compiling bf16_gemm_sm80_pipe_tma.cu (TMA load/store) ..."
```

To:
```bash
echo "Compiling bf16_gemm_sm80_pipe_tma.cu (TMA load + STSM/TMA store) ..."
```

And change the run section from:
```bash
echo "=== Running TMA load/store sample ==="
```

To:
```bash
echo "=== Running TMA load + STSM/TMA store sample ==="
```

- [ ] **Step 3: Commit**

```bash
cd /data/lmdeploy-cute
git add cute-reference/samples/bf16_gemm_sm80_pipe_tma.cu cute-reference/samples/build_and_run.sh
git commit -m "Update banner and build script for STSM+TMA store epilogue"
```
