# SM90 WGMMA Persistent Warp-Specialized GEMM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create sample 09 that replaces SM80 HMMA with SM90 WGMMA in the persistent warp-specialized GEMM kernel.

**Architecture:** Copy 08 to 09, surgically replace HMMA atom + S2R copy infrastructure with WGMMA atom + GMMA descriptors. The main loop simplifies from a k_block inner loop with LDSM copies to a single `gemm()` call with warpgroup sync primitives. All other architecture (PipelineTmaAsync, warp specialization, separate C smem, STSM+TMA store epilogue, persistence) is preserved.

**Tech Stack:** CUDA, CuTe, SM90 WGMMA (wgmma.mma_async), SM90 TMA, CUTLASS PipelineTmaAsync

---

### Task 1: Create 09 with all code changes

**Files:**
- Create: `cute-reference/samples/09_bf16_gemm_sm90_pipe_tma_ws_persistent.cu`

- [ ] **Step 1: Copy 08 to 09**

```bash
cp cute-reference/samples/08_bf16_gemm_sm80_pipe_tma_ws_persistent.cu \
   cute-reference/samples/09_bf16_gemm_sm90_pipe_tma_ws_persistent.cu
```

- [ ] **Step 2: Update the file header comment**

File: `cute-reference/samples/09_bf16_gemm_sm90_pipe_tma_ws_persistent.cu`, lines 1-22

Change:

```cpp
/***************************************************************************************************
 * BF16 GEMM using SM80 tensor cores with CuTe — Persistent Warp-Specialized TMA GEMM
 *
 * A persistent variant of 07_bf16_gemm_sm80_pipe_tma_ws.cu that launches exactly num_SMs
 * blocks in a 1D grid. Each block processes multiple output tiles in a while loop, striding
 * by gridDim.x between tiles. This eliminates kernel launch overhead for small GEMMs.
 *
 * Key changes from 07:
 *   - 1D grid of num_SMs blocks instead of 2D grid of (M_tiles x N_tiles) blocks
 *   - Each block loops over tiles: linear_idx starts at blockIdx.x, advances by gridDim.x
 *   - Producer/consumer pipeline states persist across tiles
 *   - gmem tensors (gA, gB, gC) computed per-tile inside the loop
 *   - producer_tail called once after the while loop exits
 *   - Separate epilogue smem buffer (C output tile uses dedicated smem, not overlay on smem.A)
 *
 * C = alpha * A * B^T + beta * C
 *   A: bf16, M x K, row-major (TN layout)
 *   B: bf16, K x N, stored as (N, K) in CuTe with K-contiguous stride
 *   C: bf16, M x N, column-major
 *
 * Target: SM90 (uses SM80 tensor cores + SM90 TMA for gmem<->smem transfers)
 **************************************************************************************************/
```

To:

```cpp
/***************************************************************************************************
 * BF16 GEMM using SM90 WGMMA tensor cores with CuTe — Persistent Warp-Specialized TMA GEMM
 *
 * A persistent variant of 08_bf16_gemm_sm80_pipe_tma_ws_persistent.cu that replaces SM80 HMMA
 * with SM90 WGMMA (warpgroup matrix multiply-accumulate). WGMMA reads operands directly from
 * smem via 64-bit GMMA descriptors, eliminating LDSM copies and the k_block inner loop.
 *
 * Key changes from 08:
 *   - SM90 WGMMA atom (64x128x16_SS) replaces SM80 HMMA atom (16x8x16)
 *   - Tile size 128x128 (was 256x128) — matches 2 warpgroups of 64x128 each
 *   - No S2R (smem-to-register) copies — WGMMA reads smem via descriptors
 *   - No k_block inner loop — single gemm() call per pipeline stage
 *   - warpgroup_arrive/commit_batch/wait replaces manual mma.sync scheduling
 *   - Smem reduced to ~131 KB (from ~208 KB) due to smaller A tile
 *
 * C = alpha * A * B^T + beta * C
 *   A: bf16, M x K, row-major (TN layout)
 *   B: bf16, K x N, stored as (N, K) in CuTe with K-contiguous stride
 *   C: bf16, M x N, column-major
 *
 * Target: SM90 (uses SM90 WGMMA + SM90 TMA for gmem<->smem transfers)
 **************************************************************************************************/
```

- [ ] **Step 3: Update kernel template parameters**

File: `cute-reference/samples/09_bf16_gemm_sm90_pipe_tma_ws_persistent.cu`, lines 60-65

Change:

```cpp
template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA, class S2RCopyAtomA,
          class TB, class SmemLayoutB, class TmaB, class S2RCopyAtomB,
          class TC, class SmemLayoutC,
          class TmaStoreC, class R2SCopy, class CStride, class TiledMma,
          class Alpha, class Beta>
```

To:

```cpp
template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class SmemLayoutC,
          class TmaStoreC, class R2SCopy, class CStride, class TiledMma,
          class Alpha, class Beta>
```

- [ ] **Step 4: Update kernel function signature**

File: `cute-reference/samples/09_bf16_gemm_sm90_pipe_tma_ws_persistent.cu`, lines 69-76

Change:

```cpp
bf16_gemm_persistent_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                             TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a, S2RCopyAtomA s2r_atom_a,
                             TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b, S2RCopyAtomB s2r_atom_b,
                             TC      * C, SmemLayoutC,
                             CUTLASS_GRID_CONSTANT TmaStoreC const tma_store_c,
                             R2SCopy r2s_copy, CStride dC, TiledMma mma,
                             Alpha alpha, Beta beta,
                             int total_tiles)
```

To:

```cpp
bf16_gemm_persistent_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                             TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
                             TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
                             TC      * C, SmemLayoutC,
                             CUTLASS_GRID_CONSTANT TmaStoreC const tma_store_c,
                             R2SCopy r2s_copy, CStride dC, TiledMma mma,
                             Alpha alpha, Beta beta,
                             int total_tiles)
```

- [ ] **Step 5: Update consumer section header**

File: `cute-reference/samples/09_bf16_gemm_sm90_pipe_tma_ws_persistent.cu`, line 197

Change:

```cpp
    // Consumer warp groups (wg 0 and 1) — LDSM + MMA + epilogue (persistent)
```

To:

```cpp
    // Consumer warp groups (wg 0 and 1) — WGMMA + epilogue (persistent)
```

- [ ] **Step 6: Replace consumer MMA setup (remove S2R, add WGMMA descriptors)**

File: `cute-reference/samples/09_bf16_gemm_sm90_pipe_tma_ws_persistent.cu`, lines 201-244

Change:

```cpp
    // ---- Step 4: TiledMMA setup and register allocation (done once) ----

    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);

    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));                  // (MMA, MMA_M, MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));                  // (MMA, MMA_N, MMA_K)
    // tCrC from a static-shape dummy tile (layout only matters, not the data)
    Tensor gC_dummy = make_tensor(make_gmem_ptr(C),
                                  make_shape(size<0>(cta_tiler), size<1>(cta_tiler)),
                                  dC);
    Tensor tCrC = thr_mma.make_fragment_C(thr_mma.partition_C(gC_dummy));  // (MMA, MMA_M, MMA_N)

    // ---- Step 4b: S2R (smem->register) copy setup (done once) ----

    TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
    ThrCopy  thr_s2r_a   = s2r_copy_a.get_slice(threadIdx.x);
    Tensor tXsA = thr_s2r_a.partition_S(sA);                               // (CPY, MMA_M, MMA_K, PIPE)
    Tensor tXrA = thr_s2r_a.retile_D(tCrA);                               // (CPY, MMA_M, MMA_K)

    TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
    ThrCopy  thr_s2r_b   = s2r_copy_b.get_slice(threadIdx.x);
    Tensor tXsB = thr_s2r_b.partition_S(sB);                               // (CPY, MMA_N, MMA_K, PIPE)
    Tensor tXrB = thr_s2r_b.retile_D(tCrB);                               // (CPY, MMA_N, MMA_K)

    // ---- Step 4c: R2S (register->smem) STSM copy setup (done once) ----

    ThrCopy thr_r2s = r2s_copy.get_slice(threadIdx.x);

    // ---- TMA store setup (done once) ----

    Tensor sC = make_tensor(make_smem_ptr(smem.C.begin()), SmemLayoutC{});

    auto cta_tile_mn = product_each(shape(SmemLayoutC{}));
    Tensor mC_tma = tma_store_c.get_tma_tensor(make_shape(M, N));
    Tensor gC_tma_full = flat_divide(mC_tma, cta_tile_mn);

    auto cta_tma_store = tma_store_c.get_slice(Int<0>{});
    Tensor tSsC_x = cta_tma_store.partition_S(sC);
    Tensor tSgC_x = cta_tma_store.partition_D(gC_tma_full);

    Tensor tSgC = group_modes<1, rank(tSgC_x)>(tSgC_x);
    Tensor tSsC = group_modes<1, rank(tSsC_x)>(tSsC_x);

    auto K_BLOCK_MAX = size<2>(tCrA);
```

To:

```cpp
    // ---- Step 4: TiledMMA setup and register allocation (done once) ----

    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);

    // WGMMA: partition smem into GMMA descriptors (no register-based A/B fragments)
    Tensor tCsA = thr_mma.partition_A(sA);                                  // (MMA, MMA_M, MMA_K, PIPE)
    Tensor tCsB = thr_mma.partition_B(sB);                                  // (MMA, MMA_N, MMA_K, PIPE)
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);                            // GMMA descriptors
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);                            // GMMA descriptors

    // tCrC from a static-shape dummy tile (layout only matters, not the data)
    Tensor gC_dummy = make_tensor(make_gmem_ptr(C),
                                  make_shape(size<0>(cta_tiler), size<1>(cta_tiler)),
                                  dC);
    Tensor tCrC = thr_mma.make_fragment_C(thr_mma.partition_C(gC_dummy));  // (MMA, MMA_M, MMA_N)

    // ---- Step 4b: R2S (register->smem) STSM copy setup (done once) ----

    ThrCopy thr_r2s = r2s_copy.get_slice(threadIdx.x);

    // ---- TMA store setup (done once) ----

    Tensor sC = make_tensor(make_smem_ptr(smem.C.begin()), SmemLayoutC{});

    auto cta_tile_mn = product_each(shape(SmemLayoutC{}));
    Tensor mC_tma = tma_store_c.get_tma_tensor(make_shape(M, N));
    Tensor gC_tma_full = flat_divide(mC_tma, cta_tile_mn);

    auto cta_tma_store = tma_store_c.get_slice(Int<0>{});
    Tensor tSsC_x = cta_tma_store.partition_S(sC);
    Tensor tSgC_x = cta_tma_store.partition_D(gC_tma_full);

    Tensor tSgC = group_modes<1, rank(tSgC_x)>(tSgC_x);
    Tensor tSsC = group_modes<1, rank(tSsC_x)>(tSsC_x);
```

- [ ] **Step 7: Replace consumer main loop**

File: `cute-reference/samples/09_bf16_gemm_sm90_pipe_tma_ws_persistent.cu`, lines 259-288

Change:

```cpp
      // ---- Step 5: Pipelined main loop (per tile) ----

      CUTE_NO_UNROLL
      for (int k_tile_iter = 0; k_tile_iter < k_tile_count; ++k_tile_iter)
      {
        pipeline.consumer_wait(smem_pipe_read);

        Tensor tXsA_p = tXsA(_,_,_,smem_pipe_read.index());
        Tensor tXsB_p = tXsB(_,_,_,smem_pipe_read.index());

        // Prefetch k_block 0 before the MMA loop
        copy(s2r_atom_a, tXsA_p(_,_,Int<0>{}), tXrA(_,_,Int<0>{}));
        copy(s2r_atom_b, tXsB_p(_,_,Int<0>{}), tXrB(_,_,Int<0>{}));

        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
        {
          auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;
          if (k_block < K_BLOCK_MAX - 1) {
            copy(s2r_atom_a, tXsA_p(_,_,k_block_next), tXrA(_,_,k_block_next));
            copy(s2r_atom_b, tXsB_p(_,_,k_block_next), tXrB(_,_,k_block_next));
          }

          gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
        }

        pipeline.consumer_release(smem_pipe_release);
        ++smem_pipe_read;
        ++smem_pipe_release;
      }
```

To:

```cpp
      // ---- Step 5: Pipelined main loop (per tile) ----

      CUTE_NO_UNROLL
      for (int k_tile_iter = 0; k_tile_iter < k_tile_count; ++k_tile_iter)
      {
        pipeline.consumer_wait(smem_pipe_read);

        warpgroup_fence_operand(tCrC);
        warpgroup_arrive();
        gemm(mma, tCrA(_,_,_,smem_pipe_read.index()),
                  tCrB(_,_,_,smem_pipe_read.index()), tCrC);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tCrC);

        pipeline.consumer_release(smem_pipe_release);
        ++smem_pipe_read;
        ++smem_pipe_release;
      }
```

- [ ] **Step 8: Update host function — bM, TiledMMA, remove s2r atoms**

File: `cute-reference/samples/09_bf16_gemm_sm90_pipe_tma_ws_persistent.cu`, lines 356-395

Change:

```cpp
  auto bM = Int<256>{};
  auto bN = Int<128>{};
  auto bK = Int<64>{};
  auto cta_tiler = make_shape(bM, bN, bK);

  // Pipeline depth
  auto bP = Int<3>{};

  // Smem layouts — GMMA atoms with Swizzle<3,4,3> (TMA-compatible)
  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16_t>{}, make_shape(bM, bK, bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16_t>{}, make_shape(bN, bK, bP));
  auto sC_layout = make_layout(make_shape(bM, bN), make_stride(Int<1>{}, bM));  // column-major, plain

  // TMA load atoms for A and B
  Tensor tA = make_tensor(A, make_shape(M, K), dA);                       // (M,K) for TMA inspection
  Tensor tB = make_tensor(B, make_shape(N, K), dB);                       // (N,K) for TMA inspection

  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, tA, sA(_,_,0), make_shape(bM, bK));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, tB, sB(_,_,0), make_shape(bN, bK));

  // TMA store TiledCopy for C
  Tensor tC = make_tensor(C, make_shape(M, N), dC);                         // (M,N) for TMA inspection
  auto tma_store_c = make_tma_copy(SM90_TMA_STORE{}, tC, sC_layout, make_shape(bM, bN), Int<1>{});

  // TiledMMA (unchanged)
  TiledMMA mma = make_tiled_mma(
      SM80_16x8x16_F32BF16BF16F32_TN{},
      Layout<Shape<_4, _2>>{},
      Tile<Underscore, _64, Underscore>{});

  static_assert(decltype(size(mma))::value == 256, "Expected 256 threads");

  // R2S TiledCopy for STSM register->smem
  auto r2s_copy = make_tiled_copy_C(
      Copy_Atom<SM90_U16x8_STSM_T, bf16_t>{},
      mma);

  // S2R (smem->register) copy atoms (unchanged)
  Copy_Atom<SM75_U32x4_LDSM_N, bf16_t> s2r_atom_a;
  Copy_Atom<SM75_U32x4_LDSM_N, bf16_t> s2r_atom_b;
```

To:

```cpp
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<64>{};
  auto cta_tiler = make_shape(bM, bN, bK);

  // Pipeline depth
  auto bP = Int<3>{};

  // Smem layouts — GMMA atoms with Swizzle<3,4,3> (TMA-compatible + WGMMA-compatible)
  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16_t>{}, make_shape(bM, bK, bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16_t>{}, make_shape(bN, bK, bP));
  auto sC_layout = make_layout(make_shape(bM, bN), make_stride(Int<1>{}, bM));  // column-major, plain

  // TMA load atoms for A and B
  Tensor tA = make_tensor(A, make_shape(M, K), dA);                       // (M,K) for TMA inspection
  Tensor tB = make_tensor(B, make_shape(N, K), dB);                       // (N,K) for TMA inspection

  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, tA, sA(_,_,0), make_shape(bM, bK));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, tB, sB(_,_,0), make_shape(bN, bK));

  // TMA store TiledCopy for C
  Tensor tC = make_tensor(C, make_shape(M, N), dC);                         // (M,N) for TMA inspection
  auto tma_store_c = make_tma_copy(SM90_TMA_STORE{}, tC, sC_layout, make_shape(bM, bN), Int<1>{});

  // TiledMMA — SM90 WGMMA (warpgroup-level, smem descriptors, no S2R copies)
  TiledMMA mma = make_tiled_mma(
      SM90_64x128x16_F32BF16BF16F32_SS<GMMA::Major::K, GMMA::Major::K>{},
      Layout<Shape<_2, _1>>{});

  static_assert(decltype(size(mma))::value == 256, "Expected 256 threads");

  // R2S TiledCopy for STSM register->smem
  auto r2s_copy = make_tiled_copy_C(
      Copy_Atom<SM90_U16x8_STSM_T, bf16_t>{},
      mma);
```

- [ ] **Step 9: Update host function — kernel_ptr template args**

File: `cute-reference/samples/09_bf16_gemm_sm90_pipe_tma_ws_persistent.cu`, lines 411-417

Change:

```cpp
  auto* kernel_ptr = &bf16_gemm_persistent_device<
      decltype(prob_shape), decltype(cta_tiler),
      bf16_t, decltype(sA), decltype(tmaA), decltype(s2r_atom_a),
      bf16_t, decltype(sB), decltype(tmaB), decltype(s2r_atom_b),
      bf16_t, decltype(sC_layout),
      decltype(tma_store_c), decltype(r2s_copy), decltype(dC), decltype(mma),
      Alpha, Beta>;
```

To:

```cpp
  auto* kernel_ptr = &bf16_gemm_persistent_device<
      decltype(prob_shape), decltype(cta_tiler),
      bf16_t, decltype(sA), decltype(tmaA),
      bf16_t, decltype(sB), decltype(tmaB),
      bf16_t, decltype(sC_layout),
      decltype(tma_store_c), decltype(r2s_copy), decltype(dC), decltype(mma),
      Alpha, Beta>;
```

- [ ] **Step 10: Update host function — launch_kernel_on_cluster args**

File: `cute-reference/samples/09_bf16_gemm_sm90_pipe_tma_ws_persistent.cu`, lines 433-439

Change:

```cpp
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*)kernel_ptr,
      prob_shape, cta_tiler,
      A, tmaA, s2r_atom_a,
      B, tmaB, s2r_atom_b,
      C, sC_layout, tma_store_c, r2s_copy, dC, mma,
      alpha, beta,
      total_tiles);
```

To:

```cpp
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*)kernel_ptr,
      prob_shape, cta_tiler,
      A, tmaA,
      B, tmaB,
      C, sC_layout, tma_store_c, r2s_copy, dC, mma,
      alpha, beta,
      total_tiles);
```

- [ ] **Step 11: Update benchmark assertion**

File: `cute-reference/samples/09_bf16_gemm_sm90_pipe_tma_ws_persistent.cu`, line 457

Change:

```cpp
  assert(m % 256 == 0 && n % 128 == 0 && k % 64 == 0);
```

To:

```cpp
  assert(m % 128 == 0 && n % 128 == 0 && k % 64 == 0);
```

- [ ] **Step 12: Update print banner**

File: `cute-reference/samples/09_bf16_gemm_sm90_pipe_tma_ws_persistent.cu`, line 505

Change:

```cpp
  printf("BF16 GEMM (SM80 HMMA + SM90 TMA load/store, tile 256x128x64, 384 threads WS, PipelineTmaAsync, PERSISTENT)\n\n");
```

To:

```cpp
  printf("BF16 GEMM (SM90 WGMMA + TMA load/store, tile 128x128x64, 384 threads WS, PipelineTmaAsync, PERSISTENT)\n\n");
```

### Task 2: Add to build system and run script

**Files:**
- Modify: `cute-reference/samples/CMakeLists.txt`
- Modify: `cute-reference/samples/run.sh`

- [ ] **Step 1: Add 09 to CMakeLists.txt**

File: `cute-reference/samples/CMakeLists.txt`

Add `09_bf16_gemm_sm90_pipe_tma_ws_persistent` to the `CUTE_GEMM_SAMPLES` list after `08_bf16_gemm_sm80_pipe_tma_ws_persistent`:

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
)
```

- [ ] **Step 2: Add 09 to run.sh**

File: `cute-reference/samples/run.sh`

Add after the 08 block at the end of the file:

```bash

echo ""
echo "=== Running persistent warp-specialized WGMMA sample ==="
echo "--- 1024x1024x1024 ---"
$BIN/09_bf16_gemm_sm90_pipe_tma_ws_persistent 1024 1024 1024
```

### Task 3: Build the binary

- [ ] **Step 1: Build**

Run:

```bash
cd /data/lmdeploy-cute/build && ninja 09_bf16_gemm_sm90_pipe_tma_ws_persistent
```

Expected: Clean build with no errors. PTXAS should report ~131 KB smem usage (down from ~208 KB in 08 due to halved bM). Register usage will be higher per thread (128 F32 accumulators vs 32).

### Task 4: Run correctness test

- [ ] **Step 1: Correctness at 1024³**

Run:

```bash
cd /data/lmdeploy-cute/build && ./bin/09_bf16_gemm_sm90_pipe_tma_ws_persistent 1024 1024 1024
```

Expected output includes:

```
Correctness (1024^3): max error <some_value> — PASS
```

The max error must be < 0.5f. If it fails, check that GMMA descriptors are constructed correctly — the smem layout `GMMA::Layout_K_SW128_Atom` must be compatible with `SM90_64x128x16_F32BF16BF16F32_SS<GMMA::Major::K, GMMA::Major::K>`.

### Task 5: Run full benchmark sweep

- [ ] **Step 1: Benchmark all sizes**

Run:

```bash
cd /data/lmdeploy-cute/build && ./bin/09_bf16_gemm_sm90_pipe_tma_ws_persistent
```

Expected: All 6 sizes (256³ through 8192³) run without errors. Compare TFLOP/s with 08 results — WGMMA should show higher throughput per instruction.

### Task 6: Commit

- [ ] **Step 1: Commit all changes**

```bash
cd /data/lmdeploy-cute && git add \
  cute-reference/samples/09_bf16_gemm_sm90_pipe_tma_ws_persistent.cu \
  cute-reference/samples/CMakeLists.txt \
  cute-reference/samples/run.sh && \
git commit -m "Add SM90 WGMMA persistent warp-specialized GEMM (09)

Replaces SM80 HMMA with SM90 WGMMA (64x128x16_SS atom). Eliminates
S2R copy infrastructure — WGMMA reads smem directly via GMMA descriptors.
Simplifies main loop to single gemm() per pipeline stage with warpgroup
sync. Tile shrinks to 128x128, smem drops to ~131 KB."
```
