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
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cmath>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/device_kernel.h"
#include <iostream>

using bf16_t = cute::bfloat16_t;

// ================================================================================================
// SharedStorage struct
// ================================================================================================

template <class ElementA, class ElementB, class ElementC,
          class SmemLayoutA, class SmemLayoutB, class SmemLayoutC, int Stages>
struct SharedStorage
{
  alignas(128) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
  alignas(128) cute::ArrayEngine<ElementC, cute::cosize_v<SmemLayoutC>> C;
  typename cutlass::PipelineTmaAsync<Stages>::SharedStorage pipeline;
};

// ================================================================================================
// Device Kernel (Persistent)
// ================================================================================================

template <class ClusterShape, class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class SmemLayoutC,
          class TmaStoreC, class R2SCopy, class CStride, class TiledMma,
          class Alpha, class Beta>
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
{
  using namespace cute;

  // ---- Preconditions ----

  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

  static_assert(is_static<SmemLayoutA>::value);
  static_assert(is_static<SmemLayoutB>::value);

  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutA{}) == size<0>(cta_tiler));    // BLK_M
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutA{}) == size<2>(cta_tiler));    // BLK_K
  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutB{}) == size<1>(cta_tiler));    // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutB{}) == size<2>(cta_tiler));    // BLK_K

  CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));           // dC for MN

  // ---- Step 1: Global memory tensors (full, untiled) ----

  auto [M, N, K] = shape_MNK;
  Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));                    // (M,K) TMA Tensor
  Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));                    // (N,K) TMA Tensor
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);       // (M,N) regular gmem

  // ---- Step 2: Shared memory tensors ----

  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, TC, SmemLayoutA, SmemLayoutB, SmemLayoutC, cute::size<2>(SmemLayoutA{})>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});  // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});  // (BLK_N,BLK_K,PIPE)

  // ---- Cluster layout, coordinates, and multicast masks ----
  //
  // SM90 uses make_layout(ClusterShape{}) directly, producing a 3-mode (M,N,K) layout.
  // Do NOT use tiled_divide(make_layout(cluster_shape), make_tile(AtomThrID{})) — that
  // pattern is for SM100 only where AtomThrID=Layout<_1> (1 CTA per MMA atom).
  // For SM90 WGMMA, AtomThrID=Layout<_128> (128 threads per warpgroup atom), which is
  // far too large to divide any realistic cluster shape (1-8 CTAs per dimension).
  //
  // Mode numbering in the 3-mode layout: mode 0 = M, mode 1 = N, mode 2 = K (always 1)
  // For a 2x1 cluster (Shape<_2,_1,_1>):
  //   cluster_layout maps (m,n,k) -> linear CTA rank: (0,0,0)->0, (1,0,0)->1
  constexpr int cluster_size = cute::size(cluster_shape);
  auto cluster_layout = make_layout(cluster_shape);
  auto cta_coord = cluster_layout.get_flat_coord(int(cute::block_rank_in_cluster()));
  int my_rank_m = get<0>(cta_coord);   // CTA's M-rank within cluster
  int my_rank_n = get<1>(cta_coord);   // CTA's N-rank within cluster

  // Multicast masks: which CTAs in the cluster participate in multicast for each operand.
  // A is shared when CTAs are along N (mode 1) — create_tma_multicast_mask<1> includes
  //   all CTAs with the same M-rank but different N-ranks.
  // B is shared when CTAs are along M (mode 0) — create_tma_multicast_mask<0> includes
  //   all CTAs with the same N-rank but different M-ranks.
  // For a 2x1 cluster: mcast_mask_a = 0x0001 (self only, no A multicast since cluster_N=1),
  //                    mcast_mask_b = 0x0003 (both CTAs participate in B multicast).
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

  // Cluster-level tile counts: how many cluster-tiles in each dimension.
  // A cluster of (CM, CN) CTAs processes CM*bM rows and CN*bN columns per step.
  int cluster_m_tiles = m_tiles / size<0>(cluster_shape);
  int cluster_n_tiles = n_tiles / size<1>(cluster_shape);

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

  // Pass cluster_shape to PipelineTmaAsync — it configures cross-CTA barrier arrival counts
  // for multicast. When cluster_size > 1, the barrier init adjusts the consumer arrival count:
  //   (cluster_M + cluster_N - 1) * num_consumer_warpgroups
  // instead of the simple num_consumers. This accounts for the cross-shaped dependency pattern
  // where each CTA only needs to signal CTAs in its same row or column (not diagonals).
  MainloopPipeline pipeline(smem.pipeline, pipeline_params, cluster_shape);

  // Cluster-wide sync: all CTAs must see the barrier initialization before any CTA proceeds.
  // Single-CTA cluster uses __syncthreads (intra-CTA only); multi-CTA uses cluster_arrive/wait
  // which synchronizes across all CTAs in the cluster via the hardware cluster barrier.
  if constexpr (cluster_size > 1) {
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
  } else {
    __syncthreads();
  }

  // ---- Pipeline states (created once, before warp group branch) ----

  auto smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
  typename MainloopPipeline::PipelineState smem_pipe_read;
  typename MainloopPipeline::PipelineState smem_pipe_release;

  // Persistent scheduling at cluster granularity:
  // linear_idx is per-cluster (all CTAs in a cluster have the same starting linear_idx).
  // num_clusters = total_grid_CTAs / cluster_size.
  uint64_t linear_idx = blockIdx.x / cluster_size;
  uint64_t num_clusters = gridDim.x / cluster_size;

  if (warp_group_idx == 2) {
    // ==================================================================
    // Producer warp group — TMA loads (persistent, multicast-aware)
    // ==================================================================
    cutlass::arch::warpgroup_reg_dealloc<40>();

    if (warp_group_thread_idx == 0) {
      using BarrierType = typename MainloopPipeline::ProducerBarrierType;

      while (linear_idx < total_cluster_tiles) {
        // Derive per-CTA tile coordinates from cluster-level tile index.
        // Each cluster processes a "cluster-tile" of (cluster_M*bM) x (cluster_N*bN).
        // CTA (my_rank_m, my_rank_n) within the cluster handles tile at
        // (cluster_m * cluster_M + my_rank_m, cluster_n * cluster_N + my_rank_n).
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
        //
        // A is multicast along N (mode 1): get<1>(cta_coord) is the CTA's N-rank in cluster.
        //   For 2x1 cluster, this is always 0 (no N variation), so no offset — same as 09.
        // B is multicast along M (mode 0): get<0>(cta_coord) is the CTA's M-rank in cluster.
        //   For 2x1 cluster, CTA 0 gets offset 0, CTA 1 gets offset for the second half.
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
          // Cooperative multicast: EVERY CTA issues its own TMA loads.
          // For multicast operands, the multicast mask (e.g., 0x0003 for 2x1 B) tells the
          // TMA hardware to broadcast this CTA's portion to all CTAs in the mask.
          // For non-multicast operands (SM90_TMA_LOAD), the mask is 0 and silently ignored.
          //
          // The result: each CTA's smem receives data from ALL CTAs' multicast contributions,
          // filling the complete tile cooperatively. The PipelineTmaAsync's ClusterTransactionBarrier
          // tracks the total bytes and signals consumer_wait on each CTA when all data arrives.
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
    }  // end while loop over tiles

    // Cluster sync after mainloop: ensure all CTAs finish before any CTA exits.
    // Without this, a fast CTA could exit and its smem/barriers could be reclaimed
    // by the driver while another CTA in the cluster is still using them.
    if constexpr (cluster_size > 1) {
      cute::cluster_arrive();
      cute::cluster_wait();
    }
  }  // end consumer else-branch
}

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

  // ---- Cluster shape configuration ----
  // Determines which operands are multicast:
  //   Shape<_2,_1,_1> = 2 CTAs along M -> B is multicast (shared), A is not
  //   Shape<_1,_2,_1> = 2 CTAs along N -> A is multicast (shared), B is not
  //   Shape<_2,_2,_1> = 2x2 cluster  -> both A and B are multicast
  using ClusterShape = ClusterShape_;

  constexpr int cluster_m = size<0>(ClusterShape{});
  constexpr int cluster_n = size<1>(ClusterShape{});
  constexpr int cluster_size = cluster_m * cluster_n;

  // Select TMA copy atom type based on cluster dimensions.
  // SM90_TMA_LOAD_MULTICAST requires a multicast_mask at .with() time.
  // SM90_TMA_LOAD accepts but ignores the mask (harmless to pass 0).
  // This std::conditional pattern matches CUTLASS example 50.
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

  // TMA load atoms — multicast operands get SM90_TMA_LOAD_MULTICAST + cluster size.
  //
  // make_tma_atom's 5th parameter (cluster_size) truncates the TMA SMEM box by the multicast
  // factor. For example, with B multicast factor 2 and bN=256:
  //   - The TMA descriptor describes a box of (bN/2, bK) = (128, 64)
  //   - Each CTA loads its portion via tma_partition offset (CTA 0: first 128, CTA 1: second 128)
  //   - The multicast mask broadcasts each portion to all CTAs
  //   - Combined: each CTA's smem receives the full 256x64 B tile cooperatively
  Tensor tA = make_tensor(A, make_shape(M, K), dA);                       // (M,K) for TMA inspection
  Tensor tB = make_tensor(B, make_shape(N, K), dB);                       // (N,K) for TMA inspection

  Copy_Atom tmaA = make_tma_atom(GmemTiledCopyA{}, tA, sA(_,_,0),
                                 make_shape(bM, bK), Int<cluster_n>{});   // A shared along N
  Copy_Atom tmaB = make_tma_atom(GmemTiledCopyB{}, tB, sB(_,_,0),
                                 make_shape(bN, bK), Int<cluster_m>{});   // B shared along M

  // TMA store TiledCopy for C (unchanged — TMA store has no multicast on SM90)
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

  // ---- Grid and block dimensions ----
  // Persistent grid: launch enough CTAs to fill all SMs, grouped into clusters.
  // dimGrid is in units of CTAs (not clusters) — launch_kernel_on_cluster handles grouping.
  int num_SMs = 0;
  CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0));

  int m_tiles = size(ceil_div(M, bM));
  int n_tiles = size(ceil_div(N, bN));
  int cluster_m_tiles = m_tiles / cluster_m;
  int cluster_n_tiles = n_tiles / cluster_n;
  int total_cluster_tiles = cluster_m_tiles * cluster_n_tiles;

  dim3 dimBlock(size(mma) * 3 / 2);  // 384 threads: 256 MMA + 128 producer
  dim3 dimCluster(cluster_m, cluster_n, 1);
  dim3 dimGrid(std::min(num_SMs / cluster_size, total_cluster_tiles) * cluster_size);

  // Shared memory (unchanged per-CTA smem)
  int smem_size = int(sizeof(SharedStorage<bf16_t, bf16_t, bf16_t, decltype(sA), decltype(sB), decltype(sC_layout), cute::size<2>(decltype(sA){})>));

  // Kernel function pointer (ClusterShape added as first template arg)
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

  // Launch via cluster launch API (required for TMA and for multi-CTA clusters)
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

// ================================================================================================
// Main — allocate, run, verify, benchmark
// ================================================================================================

void benchmark_size(int m, int n, int k,
                    float alpha, float beta,
                    cudaStream_t stream)
{
  using namespace cute;

  assert(m % 128 == 0 && n % 256 == 0 && k % 64 == 0);
  assert(m % (128 * 2) == 0);  // cluster_m=2: each cluster covers 2*128=256 M rows

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

  printf("BF16 GEMM (SM90 WGMMA + TMA multicast, tile 128x256x64, cluster 2x1, 384 threads WS, PipelineTmaAsync, PERSISTENT)\n\n");

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
