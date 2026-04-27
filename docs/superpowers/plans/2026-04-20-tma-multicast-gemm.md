# TMA Multicast GEMM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create sample 10 that adds TMA multicast support to the persistent warp-specialized WGMMA GEMM, parameterized over cluster shape (default 2x1).

**Architecture:** Copy 09 to 10, add ClusterShape template parameter throughout. Host selects `SM90_TMA_LOAD_MULTICAST` vs `SM90_TMA_LOAD` via `std::conditional`, creates TMA atoms with multicast cluster size, launches with multi-CTA cluster dimensions. Device computes cluster layout via `make_layout(ClusterShape{})`, derives per-CTA coordinates and multicast masks, passes cluster shape to `PipelineTmaAsync`, and uses cluster-aware `tma_partition`. All CTAs issue cooperative multicast TMA loads. Persistent loop iterates at cluster granularity. Consumer (WGMMA) and epilogue are unchanged.

**Tech Stack:** CUDA, CuTe, SM90 WGMMA, SM90 TMA multicast, SM90 thread block clusters, CUTLASS PipelineTmaAsync

**Spec:** `docs/superpowers/specs/2026-04-20-tma-multicast-gemm-design.md`

---

### Task 1: Create sample 10 file with all code changes

**Files:**
- Create: `cute-reference/samples/10_bf16_gemm_sm90_pipe_tma_ws_persistent_mcast.cu`

- [ ] **Step 1: Copy 09 to 10**

```bash
cp cute-reference/samples/09_bf16_gemm_sm90_pipe_tma_ws_persistent.cu \
   cute-reference/samples/10_bf16_gemm_sm90_pipe_tma_ws_persistent_mcast.cu
```

- [ ] **Step 2: Update the file header comment**

Replace the entire header block (lines 1–22) with:

```cpp
/***************************************************************************************************
 * BF16 GEMM using SM90 WGMMA + TMA Multicast with CuTe — Persistent Warp-Specialized GEMM
 *
 * Extends 09_bf16_gemm_sm90_pipe_tma_ws_persistent.cu with TMA multicast and thread block
 * clusters. When CTAs in a cluster share the same operand tile (e.g., same B when stacked along
 * M), TMA multicast allows each CTA to load a different portion and broadcast it to all CTAs in
 * the cluster. The combination of cooperative loads fills each CTA's smem while reducing total
 * TMA bandwidth for the shared operand.
 *
 * Key changes from 09:
 *   - Thread block clusters (ClusterShape template parameter, default Shape<_2,_1,_1>)
 *   - SM90_TMA_LOAD_MULTICAST for shared operands (selected via std::conditional)
 *   - Cluster layout via make_layout(ClusterShape{}) — 3-mode (M,N,K), NOT tiled_divide
 *     (SM90 WGMMA AtomThrID=Layout<_128> is too large for tiled_divide; that pattern is SM100 only)
 *   - Multicast masks via create_tma_multicast_mask (mode 0=M, mode 1=N in the 3-mode layout)
 *   - Cluster-aware tma_partition with per-CTA coordinates along multicast dimension
 *   - PipelineTmaAsync receives cluster_shape for cross-CTA barrier management
 *   - cluster_arrive/cluster_wait replaces __syncthreads for pipeline init
 *   - Persistent loop iterates at cluster granularity (linear_idx per cluster, not per CTA)
 *   - All CTAs issue cooperative multicast TMA loads (not just the leader)
 *
 * How cooperative multicast works:
 *   In a 2x1 cluster (CTA 0 and CTA 1 along M), both CTAs need the same B tile. The TMA
 *   descriptor's SMEM box is truncated by the multicast factor (2). Each CTA loads its 1/2
 *   portion via tma_partition offset, then multicasts it to both CTAs via the mask (0x0003).
 *   CTA 0 loads the first half, CTA 1 loads the second half. Both CTAs receive both halves.
 *   Total TMA traffic for B is unchanged vs unicast, but the data is delivered to 2 CTAs for
 *   the price of 1 — effectively halving per-CTA TMA bandwidth for B.
 *
 * C = alpha * A * B^T + beta * C
 *   A: bf16, M x K, row-major (TN layout)
 *   B: bf16, K x N, stored as (N, K) in CuTe with K-contiguous stride
 *   C: bf16, M x N, column-major
 *
 * Target: SM90 (uses SM90 WGMMA + SM90 TMA multicast + thread block clusters)
 **************************************************************************************************/
```

- [ ] **Step 3: Add ClusterShape template parameter to device kernel**

Replace the kernel template (currently lines 60–65):

```cpp
template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class SmemLayoutC,
          class TmaStoreC, class R2SCopy, class CStride, class TiledMma,
          class Alpha, class Beta>
```

With:

```cpp
template <class ClusterShape, class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class SmemLayoutC,
          class TmaStoreC, class R2SCopy, class CStride, class TiledMma,
          class Alpha, class Beta>
```

- [ ] **Step 4: Add ClusterShape parameter to kernel function signature**

Replace the kernel function signature (currently lines 66–76):

```cpp
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value * 3 / 2, 1)  // 384 threads, 1 block/SM
void
bf16_gemm_persistent_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                             TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
                             TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
                             TC      * C, SmemLayoutC,
                             CUTLASS_GRID_CONSTANT TmaStoreC const tma_store_c,
                             R2SCopy r2s_copy, CStride dC, TiledMma mma,
                             Alpha alpha, Beta beta,
                             int total_tiles)
```

With:

```cpp
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value * 3 / 2, 1)  // 384 threads, 1 block/SM
void
bf16_gemm_persistent_device(ClusterShape cluster_shape,
                             ProblemShape shape_MNK, CtaTiler cta_tiler,
                             TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
                             TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
                             TC      * C, SmemLayoutC,
                             CUTLASS_GRID_CONSTANT TmaStoreC const tma_store_c,
                             R2SCopy r2s_copy, CStride dC, TiledMma mma,
                             Alpha alpha, Beta beta,
                             int total_cluster_tiles)
```

Note: `total_tiles` renamed to `total_cluster_tiles` for clarity.

- [ ] **Step 5: Add cluster layout, coordinates, and multicast masks**

After the existing precondition asserts (ending at line 93), and before "Step 1: Global memory tensors", insert a new section. Also replace the tile counts section. Find and replace the tile counts block (currently lines 110–114):

```cpp
  // ---- Tile counts (computed once) ----

  int m_tiles = size(ceil_div(M, size<0>(cta_tiler)));
  int n_tiles = size(ceil_div(N, size<1>(cta_tiler)));
  int k_tile_count = size(ceil_div(K, size<2>(cta_tiler)));
```

With:

```cpp
  // ---- Cluster layout, coordinates, and multicast masks ----
  // SM90 uses make_layout(ClusterShape{}) directly, producing a 3-mode (M,N,K) layout.
  // Do NOT use tiled_divide(make_layout(cluster_shape), make_tile(AtomThrID{})) — that
  // pattern is SM100-only where AtomThrID=Layout<_1>. For SM90 WGMMA, AtomThrID=Layout<_128>
  // which is too large to divide any realistic cluster shape.
  //
  // Mode numbering: mode 0 = M, mode 1 = N, mode 2 = K (always 1)
  // For a 2x1 cluster (Shape<_2,_1,_1>):
  //   cluster_layout maps (m,n,k) -> linear CTA rank: (0,0,0)->0, (1,0,0)->1
  constexpr int cluster_size = cute::size(cluster_shape);
  auto cluster_layout = make_layout(cluster_shape);
  auto cta_coord = cluster_layout.get_flat_coord(int(cute::block_rank_in_cluster()));
  int my_rank_m = get<0>(cta_coord);   // CTA's M-rank within cluster
  int my_rank_n = get<1>(cta_coord);   // CTA's N-rank within cluster

  // Multicast masks: which CTAs in the cluster participate in multicast for each operand.
  // A is shared when CTAs are along N (mode 1) — multicast mask includes all CTAs with same M.
  // B is shared when CTAs are along M (mode 0) — multicast mask includes all CTAs with same N.
  // For a 2x1 cluster: mcast_mask_a = 0x0001 (self only), mcast_mask_b = 0x0003 (both CTAs).
  constexpr bool multicast_A = (cute::size<1>(ClusterShape{}) > 1);
  constexpr bool multicast_B = (cute::size<0>(ClusterShape{}) > 1);
  uint16_t tma_mcast_mask_a = multicast_A
      ? create_tma_multicast_mask<1>(cluster_layout, cta_coord) : uint16_t(0);
  uint16_t tma_mcast_mask_b = multicast_B
      ? create_tma_multicast_mask<0>(cluster_layout, cta_coord) : uint16_t(0);

  // ---- Tile counts (computed once, cluster-level) ----

  int m_tiles = size(ceil_div(M, size<0>(cta_tiler)));
  int n_tiles = size(ceil_div(N, size<1>(cta_tiler)));
  int k_tile_count = size(ceil_div(K, size<2>(cta_tiler)));

  // Cluster-level tile counts: how many cluster-tiles in each dimension
  int cluster_m_tiles = m_tiles / size<0>(cluster_shape);
  int cluster_n_tiles = n_tiles / size<1>(cluster_shape);
```

- [ ] **Step 6: Update pipeline setup — cluster_shape and cluster_sync**

Replace the pipeline setup block (currently lines 116–138):

```cpp
  // ---- Pipeline setup ----

  constexpr int tma_transaction_bytes =
      sizeof(TA) * cute::cosize_v<SmemLayoutA> / cute::size<2>(SmemLayoutA{})
    + sizeof(TB) * cute::cosize_v<SmemLayoutB> / cute::size<2>(SmemLayoutB{});

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
  pipeline_params.num_consumers = 256;   // both consumer WGs
  pipeline_params.num_producers = 1;     // single TMA thread
  pipeline_params.transaction_bytes = tma_transaction_bytes;

  MainloopPipeline pipeline(smem.pipeline, pipeline_params, cute::make_layout(cute::make_shape(cute::_1{}, cute::_1{})));
  __syncthreads();
```

With:

```cpp
  // ---- Pipeline setup ----

  // Transaction bytes per TMA load (one A tile + one B tile per stage)
  constexpr int tma_transaction_bytes =
      sizeof(TA) * cute::cosize_v<SmemLayoutA> / cute::size<2>(SmemLayoutA{})
    + sizeof(TB) * cute::cosize_v<SmemLayoutB> / cute::size<2>(SmemLayoutB{});

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
  pipeline_params.num_consumers = 256;   // both consumer WGs
  pipeline_params.num_producers = 1;     // single TMA thread
  pipeline_params.transaction_bytes = tma_transaction_bytes;

  // Pass cluster_shape to PipelineTmaAsync — it configures cross-CTA barrier arrival counts
  // for multicast. The barrier init knows that CTAs in the same row/column share data via
  // multicast and adjusts the consumer arrival count accordingly: (M+N-1)*warpgroups_per_cluster
  // instead of M*N*warpgroups_per_cluster.
  MainloopPipeline pipeline(smem.pipeline, pipeline_params, cluster_shape);

  // Cluster-wide sync: all CTAs must see the barrier initialization before any CTA proceeds.
  // Single-CTA cluster uses __syncthreads (intra-CTA); multi-CTA uses cluster_arrive/wait.
  if constexpr (cluster_size > 1) {
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
  } else {
    __syncthreads();
  }
```

- [ ] **Step 7: Update pipeline states and linear_idx init**

Replace the pipeline states block (currently lines 140–147):

```cpp
  // ---- Pipeline states (created once, before warp group branch) ----

  auto smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
  typename MainloopPipeline::PipelineState smem_pipe_read;
  typename MainloopPipeline::PipelineState smem_pipe_release;

  uint64_t linear_idx = blockIdx.x;
  uint64_t grid_size = gridDim.x;
```

With:

```cpp
  // ---- Pipeline states (created once, before warp group branch) ----

  auto smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
  typename MainloopPipeline::PipelineState smem_pipe_read;
  typename MainloopPipeline::PipelineState smem_pipe_release;

  // Persistent scheduling: each CLUSTER processes a sequence of cluster-tiles.
  // linear_idx is per-cluster (each CTA in the cluster has the same starting linear_idx).
  // With 2D grid, derive cluster index from blockIdx (x,y) and cluster dimensions.
  int clusters_per_row = gridDim.x / size<0>(cluster_shape);
  uint64_t linear_idx = (blockIdx.x / size<0>(cluster_shape))
                      + (blockIdx.y / size<1>(cluster_shape)) * clusters_per_row;
  uint64_t num_clusters = clusters_per_row * (gridDim.y / size<1>(cluster_shape));
```

- [ ] **Step 8: Update producer branch — tma_partition and multicast copy**

Replace the producer branch (currently lines 149–194):

```cpp
  if (warp_group_idx == 2) {
    // ==================================================================
    // Producer warp group — TMA loads (persistent)
    // ==================================================================
    cutlass::arch::warpgroup_reg_dealloc<40>();

    if (warp_group_thread_idx == 0) {
      using BarrierType = typename MainloopPipeline::ProducerBarrierType;

      while (linear_idx < total_tiles) {
        int m_idx = linear_idx / n_tiles;
        int n_idx = linear_idx % n_tiles;

        // Compute gmem tensors for this tile
        Tensor gA = local_tile(mA, cta_tiler, make_coord(m_idx, n_idx, _), Step<_1, X, _1>{});
        Tensor gB = local_tile(mB, cta_tiler, make_coord(m_idx, n_idx, _), Step< X,_1, _1>{});

        // TMA partition for this tile (smem side is pure layout, cheap to recompute)
        auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                           group_modes<0,2>(sA), group_modes<0,2>(gA));
        auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                           group_modes<0,2>(sB), group_modes<0,2>(gB));

        int k_tile_next = 0;

        CUTE_NO_UNROLL
        for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
          pipeline.producer_acquire(smem_pipe_write);

          BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
          copy(tma_a.with(*tma_barrier), tAgA(_,k_tile_next), tAsA(_,smem_pipe_write.index()));
          copy(tma_b.with(*tma_barrier), tBgB(_,k_tile_next), tBsB(_,smem_pipe_write.index()));

          ++smem_pipe_write;
          ++k_tile_next;
        }

        linear_idx += grid_size;
      }

      pipeline.producer_tail(smem_pipe_write);
    }

    // All 128 producer threads wait for epilogue TMA store to complete
    cute::tma_store_wait<0>();
```

With:

```cpp
  if (warp_group_idx == 2) {
    // ==================================================================
    // Producer warp group — TMA loads (persistent, multicast-aware)
    // ==================================================================
    cutlass::arch::warpgroup_reg_dealloc<40>();

    if (warp_group_thread_idx == 0) {
      using BarrierType = typename MainloopPipeline::ProducerBarrierType;

      while (linear_idx < total_cluster_tiles) {
        // Derive per-CTA tile coordinates from cluster-level tile index
        int cluster_m = linear_idx / cluster_n_tiles;
        int cluster_n = linear_idx % cluster_n_tiles;
        int m_idx = cluster_m * size<0>(cluster_shape) + my_rank_m;
        int n_idx = cluster_n * size<1>(cluster_shape) + my_rank_n;

        // Compute gmem tensors for this CTA's tile
        Tensor gA = local_tile(mA, cta_tiler, make_coord(m_idx, n_idx, _), Step<_1, X, _1>{});
        Tensor gB = local_tile(mB, cta_tiler, make_coord(m_idx, n_idx, _), Step< X,_1, _1>{});

        // TMA partition with cluster-aware coordinates.
        // For multicast operands, tma_partition offsets each CTA to load a different portion
        // of the shared tile. The multicast mask then broadcasts that portion to all CTAs.
        // For non-multicast operands (coord=0, layout=_1), this is identical to sample 09.
        auto [tAgA, tAsA] = tma_partition(tma_a, get<1>(cta_coord),
                                           make_layout(size<1>(cluster_layout)),
                                           group_modes<0,2>(sA), group_modes<0,2>(gA));
        auto [tBgB, tBsB] = tma_partition(tma_b, get<0>(cta_coord),
                                           make_layout(size<0>(cluster_layout)),
                                           group_modes<0,2>(sB), group_modes<0,2>(gB));

        int k_tile_next = 0;

        CUTE_NO_UNROLL
        for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
          pipeline.producer_acquire(smem_pipe_write);

          BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
          // Cooperative multicast: every CTA issues its own TMA loads. For multicast operands,
          // the multicast mask (e.g., 0x0003 for 2x1 B) broadcasts to all CTAs in the group.
          // For non-multicast operands, the mask is 0 (SM90_TMA_LOAD ignores it).
          copy(tma_a.with(*tma_barrier, tma_mcast_mask_a), tAgA(_,k_tile_next), tAsA(_,smem_pipe_write.index()));
          copy(tma_b.with(*tma_barrier, tma_mcast_mask_b), tBgB(_,k_tile_next), tBsB(_,smem_pipe_write.index()));

          ++smem_pipe_write;
          ++k_tile_next;
        }

        linear_idx += num_clusters;
      }

      pipeline.producer_tail(smem_pipe_write);
    }

    // All 128 producer threads wait for epilogue TMA store to complete
    cute::tma_store_wait<0>();
```

- [ ] **Step 9: Update consumer branch — persistent loop with cluster iteration and cluster sync**

Replace the consumer branch (currently lines 195–304):

```cpp
  } else {
    // ==================================================================
    // Consumer warp groups (wg 0 and 1) — WGMMA + epilogue (persistent)
    // ==================================================================
    cutlass::arch::warpgroup_reg_alloc<232>();

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

    // ---- Persistent while loop over tiles ----

    while (linear_idx < total_tiles) {
      int m_idx = linear_idx / n_tiles;
      int n_idx = linear_idx % n_tiles;

      // Compute gC for this tile
      Tensor gC = local_tile(mC, cta_tiler, make_coord(m_idx, n_idx, _), Step<_1, _1, X>{});
      Tensor tCgC = thr_mma.partition_C(gC);                                // (MMA, MMA_M, MMA_N)

      // Clear accumulators for this tile
      clear(tCrC);

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

      // ---- Step 6: Epilogue (per tile) ----

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
      Tensor tRS_sC   = thr_r2s.partition_D(sC);
      copy(r2s_copy, tRS_rAcc, tRS_sC);

      // Consumer-only sync after STSM writes
      cutlass::arch::NamedBarrier consumer_sync(256, 6);
      consumer_sync.sync();

      // Stage 3: TMA store (smem -> gmem)
      // rest_idx matches flat_divide's tile ordering: m_tile + n_tile * m_tiles
      int rest_idx = m_idx + n_idx * m_tiles;
      if (threadIdx.x == 0) {
        tma_store_fence();
        copy(tma_store_c, tSsC(_, 0), tSgC(_, rest_idx));
        tma_store_arrive();
      }
      tma_store_wait<0>();

      linear_idx += grid_size;
    }  // end while loop over tiles
  }  // end consumer else-branch
}
```

With:

```cpp
  } else {
    // ==================================================================
    // Consumer warp groups (wg 0 and 1) — WGMMA + epilogue (persistent)
    // ==================================================================
    cutlass::arch::warpgroup_reg_alloc<232>();

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

    // ---- Persistent while loop over cluster-tiles ----

    while (linear_idx < total_cluster_tiles) {
      // Derive per-CTA tile coordinates from cluster-level tile index
      int cluster_m = linear_idx / cluster_n_tiles;
      int cluster_n = linear_idx % cluster_n_tiles;
      int m_idx = cluster_m * size<0>(cluster_shape) + my_rank_m;
      int n_idx = cluster_n * size<1>(cluster_shape) + my_rank_n;

      // Compute gC for this tile
      Tensor gC = local_tile(mC, cta_tiler, make_coord(m_idx, n_idx, _), Step<_1, _1, X>{});
      Tensor tCgC = thr_mma.partition_C(gC);                                // (MMA, MMA_M, MMA_N)

      // Clear accumulators for this tile
      clear(tCrC);

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

      // ---- Step 6: Epilogue (per tile) ----

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
      Tensor tRS_sC   = thr_r2s.partition_D(sC);
      copy(r2s_copy, tRS_rAcc, tRS_sC);

      // Consumer-only sync after STSM writes
      cutlass::arch::NamedBarrier consumer_sync(256, 6);
      consumer_sync.sync();

      // Stage 3: TMA store (smem -> gmem)
      // rest_idx matches flat_divide's tile ordering: m_tile + n_tile * m_tiles
      int rest_idx = m_idx + n_idx * m_tiles;
      if (threadIdx.x == 0) {
        tma_store_fence();
        copy(tma_store_c, tSsC(_, 0), tSgC(_, rest_idx));
        tma_store_arrive();
      }
      tma_store_wait<0>();

      linear_idx += num_clusters;
    }  // end while loop over cluster-tiles

    // Cluster sync after mainloop: ensure all CTAs finish before any CTA exits
    if constexpr (cluster_size > 1) {
      cute::cluster_arrive();
      cute::cluster_wait();
    }
  }  // end consumer else-branch
}
```

- [ ] **Step 10: Rewrite host function with ClusterShape and multicast TMA atoms**

Replace the entire host function (currently lines 311–419) with:

```cpp
// ================================================================================================
// Host Function — configure and launch the persistent multicast kernel
// ================================================================================================

template <class ClusterShape_ = cute::Shape<cute::_2, cute::_1, cute::_1>,
          class Alpha, class Beta>
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

  // Cluster shape — determines which operands are multicast
  // e.g., Shape<_2,_1,_1> = 2 CTAs along M → B is multicast, A is not
  //       Shape<_1,_2,_1> = 2 CTAs along N → A is multicast, B is not
  //       Shape<_2,_2,_1> = 2x2 cluster  → both A and B are multicast
  using ClusterShape = ClusterShape_;

  constexpr int cluster_m = size<0>(ClusterShape{});
  constexpr int cluster_n = size<1>(ClusterShape{});
  constexpr int cluster_size = cluster_m * cluster_n;

  // Multicast selection: A shared along N, B shared along M
  using GmemTiledCopyA = std::conditional_t<(cluster_n > 1), SM90_TMA_LOAD_MULTICAST, SM90_TMA_LOAD>;
  using GmemTiledCopyB = std::conditional_t<(cluster_m > 1), SM90_TMA_LOAD_MULTICAST, SM90_TMA_LOAD>;

  // Problem shape
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);

  // TN strides (for TMA descriptor creation)
  auto dA = make_stride(ldA, Int<1>{});                                   // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{});                                   // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                                   // (dM, dN)

  // CTA tile sizes (static) — same per-CTA tile as sample 09
  auto bM = Int<128>{};
  auto bN = Int<256>{};
  auto bK = Int<64>{};
  auto cta_tiler = make_shape(bM, bN, bK);

  // Pipeline depth
  auto bP = Int<3>{};

  // Smem layouts — GMMA atoms with Swizzle<3,4,3> (TMA-compatible)
  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16_t>{}, make_shape(bM, bK, bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16_t>{}, make_shape(bN, bK, bP));
  auto sC_layout = make_layout(make_shape(bM, bN), make_stride(Int<1>{}, bM));  // column-major, plain

  // TMA load atoms — multicast operands get SM90_TMA_LOAD_MULTICAST + cluster size
  // make_tma_atom's 5th parameter (cluster_size) truncates the TMA SMEM box by the multicast
  // factor. Each CTA then loads 1/N of the tile via tma_partition offset, and multicasts it.
  Tensor tA = make_tensor(A, make_shape(M, K), dA);                       // (M,K) for TMA inspection
  Tensor tB = make_tensor(B, make_shape(N, K), dB);                       // (N,K) for TMA inspection

  Copy_Atom tmaA = make_tma_atom(GmemTiledCopyA{}, tA, sA(_,_,0),
                                 make_shape(bM, bK), Int<cluster_n>{});   // cluster_n CTAs share A
  Copy_Atom tmaB = make_tma_atom(GmemTiledCopyB{}, tB, sB(_,_,0),
                                 make_shape(bN, bK), Int<cluster_m>{});   // cluster_m CTAs share B

  // TMA store TiledCopy for C (unchanged — TMA store has no multicast)
  Tensor tC = make_tensor(C, make_shape(M, N), dC);                         // (M,N) for TMA inspection
  auto tma_store_c = make_tma_copy(SM90_TMA_STORE{}, tC, sC_layout, make_shape(bM, bN), Int<1>{});

  // TiledMMA — SM90 WGMMA (warpgroup-level, smem descriptors, no S2R copies)
  TiledMMA mma = make_tiled_mma(
      SM90_64x256x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{},
      Layout<Shape<_2, _1>>{});

  static_assert(decltype(size(mma))::value == 256, "Expected 256 threads");

  // R2S TiledCopy for STSM register->smem
  auto r2s_copy = make_tiled_copy_C(
      Copy_Atom<SM90_U16x8_STSM_T, bf16_t>{},
      mma);

  // Grid and block dimensions — persistent: 1D grid of clusters
  int num_SMs = 0;
  CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0));

  int m_tiles = size(ceil_div(M, bM));
  int n_tiles = size(ceil_div(N, bN));
  int cluster_m_tiles = m_tiles / cluster_m;
  int cluster_n_tiles = n_tiles / cluster_n;
  int total_cluster_tiles = cluster_m_tiles * cluster_n_tiles;

  dim3 dimBlock(size(mma) * 3 / 2);  // 384 threads: 256 MMA + 128 producer
  dim3 dimCluster(cluster_m, cluster_n, 1);

  // 2D grid: both dimensions must be multiples of cluster dimensions.
  // This matches CUTLASS's approach and ensures PipelineTmaAsync's
  // is_same_row_or_col() works correctly with block_id_in_cluster().
  int target_clusters = std::min(num_SMs / cluster_size, total_cluster_tiles);
  int grid_clusters_m = std::min(cluster_m_tiles, target_clusters);
  int grid_clusters_n = std::min(cluster_n_tiles,
      (target_clusters + grid_clusters_m - 1) / grid_clusters_m);
  dim3 dimGrid(grid_clusters_m * cluster_m, grid_clusters_n * cluster_n, 1);

  // Shared memory
  int smem_size = int(sizeof(SharedStorage<bf16_t, bf16_t, bf16_t, decltype(sA), decltype(sB), decltype(sC_layout), cute::size<2>(decltype(sA){})>));

  // Kernel function pointer
  auto* kernel_ptr = &bf16_gemm_persistent_device<
      ClusterShape,
      decltype(prob_shape), decltype(cta_tiler),
      bf16_t, decltype(sA), decltype(tmaA),
      bf16_t, decltype(sB), decltype(tmaB),
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

  // Launch via cluster launch API (required for TMA and clusters)
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*)kernel_ptr,
      ClusterShape{},
      prob_shape, cta_tiler,
      A, tmaA,
      B, tmaB,
      C, sC_layout, tma_store_c, r2s_copy, dC, mma,
      alpha, beta,
      total_cluster_tiles);

  CUTE_CHECK_LAST();
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed at kernel Launch" << std::endl;
  }
}
```

- [ ] **Step 11: Update benchmark and main functions**

Replace the `benchmark_size` function assertion (currently around line 431):

```cpp
  assert(m % 128 == 0 && n % 256 == 0 && k % 64 == 0);
```

With:

```cpp
  assert(m % 128 == 0 && n % 256 == 0 && k % 64 == 0);
  assert(m % (128 * 2) == 0);  // cluster_m=2, each cluster covers 2*128 M rows
```

Replace the main function banner (currently around line 479):

```cpp
  printf("BF16 GEMM (SM90 WGMMA + TMA load/store, tile 128x256x64, 384 threads WS, PipelineTmaAsync, PERSISTENT)\n\n");
```

With:

```cpp
  printf("BF16 GEMM (SM90 WGMMA + TMA multicast, tile 128x256x64, cluster 2x1, 384 threads WS, PipelineTmaAsync, PERSISTENT)\n\n");
```

### Task 2: Add to build system

**Files:**
- Modify: `cute-reference/samples/CMakeLists.txt`

- [ ] **Step 1: Add 10 to CMakeLists.txt**

Add `10_bf16_gemm_sm90_pipe_tma_ws_persistent_mcast` to the `CUTE_GEMM_SAMPLES` list after `09_bf16_gemm_sm90_pipe_tma_ws_persistent`:

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
)
```

### Task 3: Build the binary

- [ ] **Step 1: Build**

Run:

```bash
cd /data/lmdeploy-cute/build && ninja 10_bf16_gemm_sm90_pipe_tma_ws_persistent_mcast
```

Expected: Clean build with no errors. PTXAS should report similar smem usage to sample 09 (~213 KB) since per-CTA smem is unchanged.

If the build fails with errors about `create_tma_multicast_mask`, `block_rank_in_cluster`, `cluster_arrive_relaxed`, or `cluster_wait` not being found, ensure the CuTe/CUTLASS headers are properly included (they should be via `cute/tensor.hpp` and the existing includes).

### Task 4: Run correctness test

- [ ] **Step 1: Correctness at 1024^3**

Run:

```bash
cd /data/lmdeploy-cute/build && ./bin/10_bf16_gemm_sm90_pipe_tma_ws_persistent_mcast 1024 1024 1024
```

Expected output includes:

```
Correctness (1024^3): max error <some_value> — PASS
```

The max error must be < 0.5f. If it fails:
- Check that multicast masks are correct for the 2x1 cluster (mcast_mask_a=0x0001, mcast_mask_b=0x0003)
- Check that `tma_partition` offsets each CTA to the correct portion of the B tile
- Check that `PipelineTmaAsync` is initialized with the correct `cluster_shape`
- Check that `cluster_arrive_relaxed/cluster_wait` is used instead of `__syncthreads`

### Task 5: Run full benchmark sweep

- [ ] **Step 1: Benchmark all sizes**

Run:

```bash
cd /data/lmdeploy-cute/build && ./bin/10_bf16_gemm_sm90_pipe_tma_ws_persistent_mcast
```

Expected: All 6 sizes (256^3 through 8192^3) run without errors. For the 256^3 size, note that with a 2x1 cluster and 128x256 tile, M must be >= 256 (2 CTAs * 128). The assert at the start of `benchmark_size` should catch any incompatible sizes.

### Task 6: Commit

- [ ] **Step 1: Commit all changes**

```bash
cd /data/lmdeploy-cute && git add \
  cute-reference/samples/10_bf16_gemm_sm90_pipe_tma_ws_persistent_mcast.cu \
  cute-reference/samples/CMakeLists.txt && \
git commit -m "Add SM90 TMA multicast persistent GEMM (10)

Extends sample 09 with thread block clusters and TMA multicast.
Parameterized ClusterShape (default 2x1) selects multicast via
std::conditional. Uses SM90 3-mode cluster layout, cooperative
multicast TMA loads, cluster-aware PipelineTmaAsync barriers."
```
