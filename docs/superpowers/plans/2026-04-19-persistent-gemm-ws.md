# Persistent WS TMA GEMM Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a persistent variant of the warp-specialized TMA GEMM kernel (07) that uses stride-based work distribution to process multiple output tiles per CTA, eliminating kernel launch overhead for small GEMMs.

**Architecture:** Copy `07_bf16_gemm_sm80_pipe_tma_ws.cu` to `08_bf16_gemm_sm80_pipe_tma_ws_persistent.cu`. Wrap producer and consumer in a persistent while-loop with CUTLASS-style stride scheduling (linear tile index advances by grid size each iteration). Pipeline states are created once and advanced naturally by `++` inside k_tile loops. Host launches a 1D grid of `num_SMs` blocks instead of `(M_tiles, N_tiles)`.

**Tech Stack:** CUDA (nvcc), CUTLASS 3.9.2 (via FetchContent), CuTe, SM90 TMA, SM80 HMMA tensor cores

---

### Task 1: Create `08_bf16_gemm_sm80_pipe_tma_ws_persistent.cu` — Device Kernel

**Files:**
- Create: `cute-reference/samples/08_bf16_gemm_sm80_pipe_tma_ws_persistent.cu`

- [ ] **Step 1: Copy 07 as the starting point**

```bash
cd /data/lmdeploy-cute/cute-reference/samples
cp 07_bf16_gemm_sm80_pipe_tma_ws.cu 08_bf16_gemm_sm80_pipe_tma_ws_persistent.cu
```

- [ ] **Step 2: Update the file header comment**

Replace the header comment (lines 1-26) with:

```cpp
/***************************************************************************************************
 * BF16 GEMM using SM80 tensor cores with CuTe — Persistent Warp-Specialized TMA
 *
 * Persistent variant of 07_bf16_gemm_sm80_pipe_tma_ws.cu using CUTLASS stride-based
 * work distribution. Each CTA processes multiple output tiles by striding linear_idx
 * by grid_size between tiles.
 *
 * Key differences from 07:
 *   - 1D grid (dim3(num_SMs)) instead of 2D grid (dim3(M_tiles, N_tiles))
 *   - Outer while loop over tiles: linear_idx += gridDim.x per iteration
 *   - linear_idx -> (m_idx, n_idx) via divmod for per-tile gmem partitioning
 *   - PipelineState objects created once, advanced by ++ inside k_tile loops
 *   - producer_tail called once after loop, not per tile
 *   - k_tile_count computed once (ceil_div(K, bK)) instead of from tma_partition
 *   - total_tiles passed as additional kernel parameter
 *
 * C = alpha * A * B^T + beta * C
 *   A: bf16, M x K, row-major (TN layout)
 *   B: bf16, K x N, stored as (N, K) in CuTe with K-contiguous stride
 *   C: bf16, M x N, column-major
 *
 * Target: SM90 (uses SM80 tensor cores + SM90 TMA for gmem↔smem transfers)
 **************************************************************************************************/
```

- [ ] **Step 3: Add `total_tiles` parameter to kernel signature**

Change the kernel function signature (line 92) from:

```cpp
void
bf16_gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                 TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a, S2RCopyAtomA s2r_atom_a,
                 TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b, S2RCopyAtomB s2r_atom_b,
                 TC      * C, SmemLayoutC,
                 CUTLASS_GRID_CONSTANT TmaStoreC const tma_store_c,
                 R2SCopy r2s_copy, CStride dC, TiledMma mma,
                 Alpha alpha, Beta beta)
```

to:

```cpp
void
bf16_gemm_persistent_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                 TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a, S2RCopyAtomA s2r_atom_a,
                 TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b, S2RCopyAtomB s2r_atom_b,
                 TC      * C, SmemLayoutC,
                 CUTLASS_GRID_CONSTANT TmaStoreC const tma_store_c,
                 R2SCopy r2s_copy, CStride dC, TiledMma mma,
                 Alpha alpha, Beta beta,
                 int total_tiles)
```

- [ ] **Step 4: Replace the gmem tensor + tile partitioning section (Step 1 in 07)**

Replace lines 117-127 (the "Step 1: Global memory tensors" and `cta_coord` / `local_tile` calls) with per-tile-gmem setup that computes `m_tiles`, `n_tiles`, and `k_tile_count` once, but defers `local_tile` to inside the loop:

```cpp
  // ---- Step 1: Global memory tensors (full tensors, computed once) ----

  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

  static_assert(is_static<SmemLayoutA>::value);
  static_assert(is_static<SmemLayoutB>::value);

  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutA{}) == size<0>(cta_tiler));    // BLK_M
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutA{}) == size<2>(cta_tiler));    // BLK_K
  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutB{}) == size<1>(cta_tiler));    // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutB{}) == size<2>(cta_tiler));    // BLK_K

  CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));           // dC for MN

  auto [M, N, K] = shape_MNK;
  Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));                    // (M,K) TMA Tensor
  Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));                    // (N,K) TMA Tensor
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);       // (M,N) regular gmem

  // Tile counts (once — same for all tiles)
  int m_tiles = size(ceil_div(M, size<0>(cta_tiler)));
  int n_tiles = size(ceil_div(N, size<1>(cta_tiler)));
  int k_tile_count = size(ceil_div(K, size<2>(cta_tiler)));
```

Note: `gA`, `gB`, `gC` (the `local_tile` results) are removed from here — they move inside the while loop.

- [ ] **Step 5: Update smem tensor section (Step 2 — unchanged except remove local_tile references)**

Replace lines 129-135 with:

```cpp
  // ---- Step 2: Shared memory tensors (computed once) ----

  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB, cute::size<2>(SmemLayoutA{})>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});  // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});  // (BLK_N,BLK_K,PIPE)
```

Note: The TMA partitioning section (old Step 3, lines 137-151) is removed from here — it moves inside the while loop. The `K_PIPE_MAX`, `k_tile_next` variables (old lines 158-160) are also removed.

- [ ] **Step 6: Update pipeline setup section (Step 3b)**

Replace lines 152-183 with:

```cpp
  // ---- Step 3: Pipeline setup and warp group dispatch ----

  constexpr int tma_transaction_bytes = sizeof(make_tensor_like(tensor<0>(
      tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                    group_modes<0,2>(sA),
                    group_modes<0,2>(local_tile(mA, cta_tiler, make_coord(0, 0, _), Step<_1, X, _1>{}))))))
                                      + sizeof(make_tensor_like(tensor<0>(
      tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                    group_modes<0,2>(sB),
                    group_modes<0,2>(local_tile(mB, cta_tiler, make_coord(0, 0, _), Step<X, _1, _1>{}))))));
```

Actually, that's overly complex. A simpler approach: use the same constexpr computation as 07 but with a dummy tile. Since `tma_transaction_bytes` only depends on the tile size (not the tile position), compute it the same way as 07 but move the tma_partition into the pipeline setup, using a dummy coord. But this is awkward.

**Simpler approach:** Just compute it from the smem layout sizes directly, since `tma_transaction_bytes` = size of one A stage + size of one B stage:

```cpp
  // ---- Step 3: Pipeline setup ----
  //
  // PipelineTmaAsync replaces the manual ClusterTransactionBarrier + phase tracking.
  // All 384 threads construct the pipeline (barrier init happens in constructor for warp 0).

  constexpr int tma_transaction_bytes =
      sizeof(TA) * cute::cosize_v<decltype(SmemLayoutA{})> / cute::size<2>(SmemLayoutA{})
      + sizeof(TB) * cute::cosize_v<decltype(SmemLayoutB{})> / cute::size<2>(SmemLayoutB{});

  int warp_group_idx = cutlass::canonical_warp_group_idx();        // 0, 1, or 2
  int warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

  using MainloopPipeline = cutlass::PipelineTmaAsync<cute::size<2>(SmemLayoutA{})>;
  typename MainloopPipeline::Params pipeline_params;
  if (warp_group_idx == 2) {
    pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
  } else {
    pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
  }
  pipeline_params.is_leader = (warp_group_thread_idx == 0);
  pipeline_params.num_consumers = 256;
  pipeline_params.num_producers = 1;
  pipeline_params.transaction_bytes = tma_transaction_bytes;

  MainloopPipeline pipeline(smem.pipeline, pipeline_params, cute::make_layout(cute::make_shape(cute::_1{}, cute::_1{})));
  __syncthreads();

  // Pipeline states — created once, advanced by ++ inside k_tile loops (never reset)
  auto smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
  typename MainloopPipeline::PipelineState smem_pipe_read;
  typename MainloopPipeline::PipelineState smem_pipe_release;

  uint64_t linear_idx = blockIdx.x;
  uint64_t grid_size  = gridDim.x;
```

- [ ] **Step 7: Replace the producer branch with persistent loop**

Replace the producer section (lines 185-212) with:

```cpp
  if (warp_group_idx == 2) {
    // ==================================================================
    // Producer warp group — TMA loads (persistent loop)
    // ==================================================================
    cutlass::arch::warpgroup_reg_dealloc<40>();

    if (warp_group_thread_idx == 0) {
      using BarrierType = typename MainloopPipeline::ProducerBarrierType;

      while (linear_idx < total_tiles) {
        int m_idx = linear_idx / n_tiles;
        int n_idx = linear_idx % n_tiles;
        if (m_idx >= m_tiles) break;

        // Per-tile gmem partitioning
        auto cta_coord = make_coord(m_idx, n_idx, _);
        Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
        Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
        auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                           group_modes<0,2>(sA), group_modes<0,2>(gA));
        auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                           group_modes<0,2>(sB), group_modes<0,2>(gB));

        CUTE_NO_UNROLL
        for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
          pipeline.producer_acquire(smem_pipe_write);

          BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
          copy(tma_a.with(*tma_barrier), tAgA(_,k_tile), tAsA(_,smem_pipe_write.index()));
          copy(tma_b.with(*tma_barrier), tBgB(_,k_tile), tBsB(_,smem_pipe_write.index()));

          ++smem_pipe_write;
        }

        linear_idx += grid_size;
      }

      // Drain remaining barrier waits once after loop exits
      pipeline.producer_tail(smem_pipe_write);
    }

    // All 128 producer threads wait for final epilogue TMA store to complete
    cute::tma_store_wait<0>();
```

- [ ] **Step 8: Replace the consumer branch with persistent loop**

Replace the consumer section (lines 214-341) with:

```cpp
  } else {
    // ==================================================================
    // Consumer warp groups (wg 0 and 1) — LDSM + MMA + epilogue (persistent loop)
    // ==================================================================
    cutlass::arch::warpgroup_reg_alloc<232>();

    // ---- Step 4: TiledMMA setup and register allocation (once) ----

    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);

    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));                  // (MMA, MMA_M, MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));                  // (MMA, MMA_N, MMA_K)
    // tCrC layout computed from full mC shape (same layout regardless of tile)
    Tensor tCrC = thr_mma.make_fragment_C(thr_mma.partition_C(
        make_tensor(make_gmem_ptr(C), make_shape(M, N), dC)));              // (MMA, MMA_M, MMA_N)

    // ---- Step 4b: S2R (smem->register) copy setup (once) ----

    TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
    ThrCopy  thr_s2r_a   = s2r_copy_a.get_slice(threadIdx.x);
    Tensor tXsA = thr_s2r_a.partition_S(sA);                               // (CPY, MMA_M, MMA_K, PIPE)
    Tensor tXrA = thr_s2r_a.retile_D(tCrA);                               // (CPY, MMA_M, MMA_K)

    TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
    ThrCopy  thr_s2r_b   = s2r_copy_b.get_slice(threadIdx.x);
    Tensor tXsB = thr_s2r_b.partition_S(sB);                               // (CPY, MMA_N, MMA_K, PIPE)
    Tensor tXrB = thr_s2r_b.retile_D(tCrB);                               // (CPY, MMA_N, MMA_K)

    // ---- Step 4c: R2S (register->smem) STSM copy setup (once) ----

    ThrCopy thr_r2s = r2s_copy.get_slice(threadIdx.x);

    Tensor sC = make_tensor(
        make_smem_ptr(reinterpret_cast<bf16_t*>(smem.A.begin())),
        SmemLayoutC{});
    Tensor tRS_sC = thr_r2s.partition_D(sC);

    // ---- TMA store setup (once — covers full (M,N), indexed by linear_idx) ----

    auto cta_tile_mn = product_each(shape(SmemLayoutC{}));
    Tensor mC_tma = tma_store_c.get_tma_tensor(make_shape(M, N));
    Tensor gC_tma_full = flat_divide(mC_tma, cta_tile_mn);

    auto cta_tma_store = tma_store_c.get_slice(Int<0>{});
    Tensor tSsC_x = cta_tma_store.partition_S(sC);
    Tensor tSgC_x = cta_tma_store.partition_D(gC_tma_full);

    Tensor tSgC = group_modes<1, rank(tSgC_x)>(tSgC_x);
    Tensor tSsC = group_modes<1, rank(tSsC_x)>(tSsC_x);

    auto K_BLOCK_MAX = size<2>(tCrA);

    // ---- Persistent loop ----

    while (linear_idx < total_tiles) {
      int m_idx = linear_idx / n_tiles;
      int n_idx = linear_idx % n_tiles;
      if (m_idx >= m_tiles) break;

      // Per-tile: gmem view for beta-load
      auto cta_coord = make_coord(m_idx, n_idx, _);
      Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});
      Tensor tCgC = thr_mma.partition_C(gC);                                // (MMA, MMA_M, MMA_N)

      // Clear accumulators for this tile
      clear(tCrC);

      // ---- Step 5: Pipelined main loop (same structure as 07) ----

      CUTE_NO_UNROLL
      for (int k_tile_iter = 0; k_tile_iter < k_tile_count; ++k_tile_iter)
      {
        pipeline.consumer_wait(smem_pipe_read);

        Tensor tXsA_p = tXsA(_,_,_,smem_pipe_read.index());
        Tensor tXsB_p = tXsB(_,_,_,smem_pipe_read.index());

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

      // ---- Step 6: Epilogue ----

      // Stage 1: Element-wise alpha/beta scaling
      CUTE_UNROLL
      for (int i = 0; i < size(tCrC); ++i) {
        tCrC(i) = alpha * tCrC(i) + beta * static_cast<float>(tCgC(i));
      }

      // Stage 2: Convert F32 -> BF16, write to smem via STSM
      Tensor tCrC_bf16 = make_tensor<bf16_t>(tCrC.layout());
      CUTE_UNROLL
      for (int i = 0; i < size(tCrC); ++i) {
        tCrC_bf16(i) = static_cast<bf16_t>(tCrC(i));
      }

      Tensor tRS_rAcc = thr_r2s.retile_S(tCrC_bf16);
      copy(r2s_copy, tRS_rAcc, tRS_sC);

      // Consumer-only sync after STSM writes
      cutlass::arch::NamedBarrier consumer_sync(256, 6);
      consumer_sync.sync();

      // Stage 3: TMA store (smem -> gmem)
      if (threadIdx.x == 0) {
        tma_store_fence();
        copy(tma_store_c, tSsC(_, 0), tSgC(_, linear_idx));
        tma_store_arrive();
      }
      tma_store_wait<0>();  // inter-tile sync: ensures TMA store completes before next tile

      linear_idx += grid_size;
    }
  }  // end consumer else-branch
}
```

- [ ] **Step 9: Verify the file compiles**

```bash
cd /data/lmdeploy-cute/build && ninja 08_bf16_gemm_sm80_pipe_tma_ws_persistent
```

Expected: Compilation succeeds (the host function hasn't been updated yet, so there may be linker errors — that's OK, we just want the device kernel to compile).

Note: Skip this step if the host function update is needed for compilation. Move to Task 2 first and compile after.

- [ ] **Step 10: Commit**

```bash
git add cute-reference/samples/08_bf16_gemm_sm80_pipe_tma_ws_persistent.cu
git commit -m "Add persistent WS TMA GEMM kernel device code (08)"
```

---

### Task 2: Host Function — `bf16_gemm_persistent`

**Files:**
- Modify: `cute-reference/samples/08_bf16_gemm_sm80_pipe_tma_ws_persistent.cu` (host function section)

- [ ] **Step 1: Replace the host function**

Replace the entire `bf16_gemm_tn` function (lines 347-467 in the original 07, or whatever lines it now occupies in 08) with:

```cpp
template <class Alpha, class Beta>
void
bf16_gemm_persistent(int m, int n, int k,
                     Alpha alpha,
                     bf16_t const* A, int ldA,
                     bf16_t const* B, int ldB,
                     Beta beta,
                     bf16_t* C, int ldC,
                     cudaStream_t stream = 0)
{
  using namespace cute;

  // Problem shape
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);

  // TN strides (for TMA descriptor creation)
  auto dA = make_stride(ldA, Int<1>{});                                   // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{});                                   // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                                   // (dM, dN)

  // CTA tile sizes (static)
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
  Tensor mA = make_tensor(A, make_shape(M, K), dA);
  Tensor mB = make_tensor(B, make_shape(N, K), dB);

  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM, bK));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN, bK));

  // TMA store TiledCopy for C
  Tensor mC = make_tensor(C, make_shape(M, N), dC);
  auto tma_store_c = make_tma_copy(SM90_TMA_STORE{}, mC, sC_layout, make_shape(bM, bN), Int<1>{});

  // TiledMMA
  TiledMMA mma = make_tiled_mma(
      SM80_16x8x16_F32BF16BF16F32_TN{},
      Layout<Shape<_4, _2>>{},
      Tile<Underscore, _64, Underscore>{});

  static_assert(decltype(size(mma))::value == 256, "Expected 256 threads");

  // R2S TiledCopy for STSM register->smem
  auto r2s_copy = make_tiled_copy_C(
      Copy_Atom<SM90_U16x8_STSM_T, bf16_t>{},
      mma);

  // S2R (smem->register) copy atoms
  Copy_Atom<SM75_U32x4_LDSM_N, bf16_t> s2r_atom_a;
  Copy_Atom<SM75_U32x4_LDSM_N, bf16_t> s2r_atom_b;

  // Grid and block dimensions — persistent: 1D grid of num_SMs blocks
  int num_SMs;
  CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0));

  int total_tiles = size(ceil_div(M, bM)) * size(ceil_div(N, bN));

  dim3 dimBlock(size(mma) * 3 / 2);  // 384 threads
  dim3 dimCluster(1, 1, 1);
  dim3 dimGrid(std::min(num_SMs, total_tiles));  // 1D grid, capped at tile count

  // Shared memory
  int smem_size = int(sizeof(SharedStorage<bf16_t, bf16_t, decltype(sA), decltype(sB), cute::size<2>(decltype(sA){})>));

  // Kernel function pointer
  auto* kernel_ptr = &bf16_gemm_persistent_device<
      decltype(prob_shape), decltype(cta_tiler),
      bf16_t, decltype(sA), decltype(tmaA), decltype(s2r_atom_a),
      bf16_t, decltype(sB), decltype(tmaB), decltype(s2r_atom_b),
      bf16_t, decltype(sC_layout),
      decltype(tma_store_c), decltype(r2s_copy), decltype(dC), decltype(mma),
      Alpha, Beta>;

  // Set shared memory attributes
  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      (void const*)kernel_ptr,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size));

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      (void const*)kernel_ptr,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      100));

  // Launch via cluster launch API
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*)kernel_ptr,
      prob_shape, cta_tiler,
      A, tmaA, s2r_atom_a,
      B, tmaB, s2r_atom_b,
      C, sC_layout, tma_store_c, r2s_copy, dC, mma,
      alpha, beta, total_tiles);

  CUTE_CHECK_LAST();
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed at kernel Launch" << std::endl;
  }
}
```

- [ ] **Step 2: Compile**

```bash
cd /data/lmdeploy-cute/build && ninja 08_bf16_gemm_sm80_pipe_tma_ws_persistent
```

Expected: Compiles without errors, produces `build/bin/08_bf16_gemm_sm80_pipe_tma_ws_persistent`.

- [ ] **Step 3: Commit**

```bash
git add cute-reference/samples/08_bf16_gemm_sm80_pipe_tma_ws_persistent.cu
git commit -m "Add persistent WS TMA GEMM host function (08)"
```

---

### Task 3: Main Function — Correctness Test + Benchmark

**Files:**
- Modify: `cute-reference/samples/08_bf16_gemm_sm80_pipe_tma_ws_persistent.cu` (main function section)

- [ ] **Step 1: Replace the main function**

Replace the entire `benchmark_size` and `main` functions with:

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
  bf16_gemm_persistent(m, n, k, alpha,
                       d_A.data().get(), ldA,
                       d_B.data().get(), ldB,
                       beta,
                       d_C.data().get(), ldC, stream);
  CUTE_CHECK_LAST();

  cudaEventRecord(start);
  for (int i = 0; i < timing_iterations; ++i) {
    bf16_gemm_persistent(m, n, k, alpha,
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

int main(int argc, char** argv)
{
  using namespace cute;

  printf("BF16 GEMM (SM80 HMMA + SM90 TMA load/store, tile 256x128x64, 384 threads WS, PERSISTENT)\n\n");

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

    bf16_gemm_persistent(m, n, k, alpha,
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

    printf("Correctness (1024^3): max error %e — %s\n\n", max_err, max_err < 0.5f ? "PASS" : "FAIL");
    if (max_err >= 0.5f) return 1;
  }

  // ---- Benchmark ----
  printf("Benchmark (100 iterations each):\n");
  benchmark_size(256,  256,  256,  alpha, beta, 0);
  benchmark_size(512,  512,  512,  alpha, beta, 0);
  benchmark_size(1024, 1024, 1024, alpha, beta, 0);
  benchmark_size(2048, 2048, 2048, alpha, beta, 0);
  benchmark_size(4096, 4096, 4096, alpha, beta, 0);
  benchmark_size(8192, 8192, 8192, alpha, beta, 0);

  return 0;
}
```

- [ ] **Step 2: Compile and run correctness test**

```bash
cd /data/lmdeploy-cute/build && ninja 08_bf16_gemm_sm80_pipe_tma_ws_persistent
```

Expected: Compilation succeeds.

```bash
./build/bin/08_bf16_gemm_sm80_pipe_tma_ws_persistent 1024 1024 1024
```

Expected: `Correctness (1024^3): max error < 0.5 — PASS` followed by benchmark results.

- [ ] **Step 3: Run full benchmark**

```bash
./build/bin/08_bf16_gemm_sm80_pipe_tma_ws_persistent
```

Expected: All 6 sizes (256³ through 8192³) benchmark successfully. TFLOP/s reported for each.

- [ ] **Step 4: Commit**

```bash
git add cute-reference/samples/08_bf16_gemm_sm80_pipe_tma_ws_persistent.cu
git commit -m "Add persistent WS TMA GEMM main function with correctness test and benchmarks"
```

---

### Task 4: Add to CMake Build + run.sh

**Files:**
- Modify: `cute-reference/samples/CMakeLists.txt`
- Modify: `cute-reference/samples/run.sh`

- [ ] **Step 1: Add target to CMakeLists.txt**

Add `08_bf16_gemm_sm80_pipe_tma_ws_persistent` to the `CUTE_GEMM_SAMPLES` list in `cute-reference/samples/CMakeLists.txt`:

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
)
```

- [ ] **Step 2: Add to run.sh**

Append to `cute-reference/samples/run.sh`:

```bash

echo ""
echo "=== Running persistent warp-specialized TMA sample ==="
echo "--- 1024x1024x1024 ---"
$BIN/08_bf16_gemm_sm80_pipe_tma_ws_persistent 1024 1024 1024
```

- [ ] **Step 3: Reconfigure and build**

```bash
cd /data/lmdeploy-cute/build && cmake .. -DBUILD_CUTE_SAMPLES=ON && ninja 08_bf16_gemm_sm80_pipe_tma_ws_persistent
```

Expected: Builds successfully.

- [ ] **Step 4: Verify via run.sh**

```bash
cd /data/lmdeploy-cute/cute-reference/samples && bash run.sh
```

Expected: All 8 samples run. 08 prints PASS and benchmark results.

- [ ] **Step 5: Commit**

```bash
git add cute-reference/samples/CMakeLists.txt cute-reference/samples/run.sh
git commit -m "Add persistent kernel (08) to CMake targets and run.sh"
```
