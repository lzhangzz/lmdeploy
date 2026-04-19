/***************************************************************************************************
 * BF16 GEMM using SM80 tensor cores with CuTe — Warp-Specialized TMA Load + TMA Store
 *
 * A variant of bf16_gemm_sm80_pipe_tma.cu that adds warp-group specialization:
 *   - 384 threads: 1 producer warp group (TMA loads) + 2 consumer warp groups (HMMA + epilogue)
 *   - CUTLASS PipelineTmaAsync for producer-consumer synchronization
 *   - Consumer-only NamedBarrier for epilogue sync
 *   - setmaxnreg register reallocation between producer/consumer warp groups
 *
 * The kernel keeps SM80 HMMA tensor cores + LDSM S2R unchanged.
 *
 * Key differences from pipe_epilogue:
 *   - Smem layouts use GMMA atoms (Swizzle<3,4,3>) instead of Swizzle<3,3,3>
 *     (Swizzle<3,3,3> is not TMA-compatible — TMA requires M=4 swizzle)
 *   - TMA descriptors encode gmem strides, so A/B strides are not passed to kernel
 *   - ClusterTransactionBarrier replaces cp_async_fence/wait for pipeline sync
 *   - Launched via cutlass::launch_kernel_on_cluster (required for TMA)
 *   - Epilogue: STSM F32->BF16 to plain smem + TMA store (replaces vectorized S2G)
 *
 * C = alpha * A * B^T + beta * C
 *   A: bf16, M x K, row-major (TN layout)
 *   B: bf16, K x N, stored as (N, K) in CuTe with K-contiguous stride
 *   C: bf16, M x N, column-major
 *
 * Target: SM90 (uses SM80 tensor cores + SM90 TMA for gmem↔smem transfers)
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
//
// Encapsulates shared memory allocation for both A and B matrices plus pipeline barriers.
// Uses CuTe's ArrayEngine which provides properly aligned storage for the layout's elements.
// PipelineTmaAsync::SharedStorage contains full_barrier and empty_barrier arrays for
// producer-consumer synchronization across pipeline stages.

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB, int Stages>
struct SharedStorage
{
  alignas(128) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
  typename cutlass::PipelineTmaAsync<Stages>::SharedStorage pipeline;
};

// ================================================================================================
// Device Kernel
// ================================================================================================
//
// Each thread block computes a (BLK_M x BLK_N) tile of output C.
// The K dimension is processed in chunks of BLK_K using a 3-stage pipeline.
//
// Data flow (pipelined, per K-tile):
//   1. TMA thread (threadIdx.x == 0) issues bulk tensor load via mbarrier
//   2. ClusterTransactionBarrier::wait ensures TMA load completes
//   3. LDSM vectorized smem -> register copy for the CURRENT pipe stage
//   4. Tensor core MMA on registers
//   5. Pipe indices rotate: read advances, write takes the slot just freed by MMA
//
// Epilogue:
//   1. Element-wise alpha/beta scaling (consumer threads only)
//   2. F32->BF16 conversion, STSM write to smem (consumer threads only)
//   3. Consumer-only NamedBarrier sync (256 threads)
//   4. TMA store: thread 0 of consumers issues bulk smem->gmem copy

template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA, class S2RCopyAtomA,
          class TB, class SmemLayoutB, class TmaB, class S2RCopyAtomB,
          class TC, class SmemLayoutC,
          class TmaStoreC, class R2SCopy, class CStride, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value * 3 / 2, 1)  // 384 threads, 1 block/SM
void
bf16_gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                 TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a, S2RCopyAtomA s2r_atom_a,
                 TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b, S2RCopyAtomB s2r_atom_b,
                 TC      * C, SmemLayoutC,
                 CUTLASS_GRID_CONSTANT TmaStoreC const tma_store_c,
                 R2SCopy r2s_copy, CStride dC, TiledMma mma,
                 Alpha alpha, Beta beta)
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

  // ---- Step 1: Global memory tensors ----

  auto [M, N, K] = shape_MNK;
  Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));                    // (M,K) TMA Tensor
  Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));                    // (N,K) TMA Tensor
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);       // (M,N) regular gmem for beta load + store

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});   // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1, _1>{});   // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});   // (BLK_M,BLK_N)

  // ---- Step 2: Shared memory tensors ----

  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});  // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});  // (BLK_N,BLK_K,PIPE)

  // ---- Step 3: TMA partitioning for gmem -> smem ----
  //
  // tma_partition returns (gmem_view, smem_view) pair:
  //   tAgA: (TMA, k) — TMA coord tensor + K-tile index
  //   tAsA: (TMA, PIPE) — SMEM view + pipeline stage
  //
  // group_modes<0,2> transforms (X,Y,Z) -> ((X,Y),Z) so TMA handles the 2D tile
  // as mode-0 and the rest (k or PIPE) as mode-1.

  auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                     group_modes<0,2>(sA), group_modes<0,2>(gA));

  auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                     group_modes<0,2>(sB), group_modes<0,2>(gB));

  // ---- Step 3b: Pipeline setup and warp group dispatch ----
  //
  // PipelineTmaAsync replaces the manual ClusterTransactionBarrier + phase tracking.
  // All 384 threads construct the pipeline (barrier init happens in constructor for warp 0).
  // The role (Producer/Consumer) is set based on warp_group_idx.

  auto K_PIPE_MAX = size<1>(tAsA);   // = bP = 3
  int k_tile_count = size<1>(tAgA);  // total K-tiles
  int k_tile_next  = 0;

  constexpr int tma_transaction_bytes = sizeof(make_tensor_like(tensor<0>(tAsA)))
                                      + sizeof(make_tensor_like(tensor<0>(tBsB)));

  int warp_group_idx = cutlass::canonical_warp_group_idx();        // 0, 1, or 2
  int warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

  // Pipeline params — role depends on warp group
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

  // Constructor initializes barriers (warp 0) + fence_barrier_init
  MainloopPipeline pipeline(smem.pipeline, pipeline_params, cute::Layout<cute::_1>{});
  __syncthreads();

  if (warp_group_idx == 2) {
    // ==================================================================
    // Producer warp group — TMA loads
    // ==================================================================
    cutlass::arch::warpgroup_reg_dealloc<40>();

    // Only the leader thread (warp_group_thread_idx == 0) runs the producer loop.
    if (warp_group_thread_idx == 0) {
      auto smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
      using BarrierType = typename MainloopPipeline::ProducerBarrierType;

      CUTE_NO_UNROLL
      for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
        pipeline.producer_acquire(smem_pipe_write);

        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
        copy(tma_a.with(*tma_barrier), tAgA(_,k_tile_next), tAsA(_,smem_pipe_write.index()));
        copy(tma_b.with(*tma_barrier), tBgB(_,k_tile_next), tBsB(_,smem_pipe_write.index()));

        ++smem_pipe_write;
        ++k_tile_next;
      }

      pipeline.producer_tail(smem_pipe_write);
    }

    // All 128 producer threads wait for epilogue TMA store to complete
    cute::tma_store_wait<0>();

  } else {
    // ==================================================================
    // Consumer warp groups (wg 0 and 1) — LDSM + MMA + epilogue
    // ==================================================================
    cutlass::arch::warpgroup_reg_alloc<232>();

    // ---- Step 4: TiledMMA setup and register allocation ----

    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);

    Tensor tCgC = thr_mma.partition_C(gC);                                 // (MMA, MMA_M, MMA_N)
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));                  // (MMA, MMA_M, MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));                  // (MMA, MMA_N, MMA_K)
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);                           // (MMA, MMA_M, MMA_N)
    clear(tCrC);

    // ---- Step 4b: S2R (smem->register) copy setup ----

    TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
    ThrCopy  thr_s2r_a   = s2r_copy_a.get_slice(threadIdx.x);
    Tensor tXsA = thr_s2r_a.partition_S(sA);                               // (CPY, MMA_M, MMA_K, PIPE)
    Tensor tXrA = thr_s2r_a.retile_D(tCrA);                               // (CPY, MMA_M, MMA_K)

    TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
    ThrCopy  thr_s2r_b   = s2r_copy_b.get_slice(threadIdx.x);
    Tensor tXsB = thr_s2r_b.partition_S(sB);                               // (CPY, MMA_N, MMA_K, PIPE)
    Tensor tXrB = thr_s2r_b.retile_D(tCrB);                               // (CPY, MMA_N, MMA_K)

    // ---- Step 4c: R2S (register->smem) STSM copy setup ----

    ThrCopy thr_r2s = r2s_copy.get_slice(threadIdx.x);

    // ---- Step 5: Pipelined main loop ----
    //
    // Consumer pipeline: wait for TMA load -> LDSM prefetch -> MMA -> release stage.
    // smem_pipe_read and smem_pipe_release track pipeline state with Phase.
    // smem_pipe_release lags behind smem_pipe_read — release happens at k_block==0
    // of the NEXT k_tile, after all k_blocks of the current stage are in registers.

    typename MainloopPipeline::PipelineState smem_pipe_read;
    typename MainloopPipeline::PipelineState smem_pipe_release;

    Tensor tXsA_p = tXsA(_,_,_,smem_pipe_read.index());
    Tensor tXsB_p = tXsB(_,_,_,smem_pipe_read.index());

    auto K_BLOCK_MAX = size<2>(tCrA);

    // Prologue: wait for first stage, prefetch k_block 0
    pipeline.consumer_wait(smem_pipe_read);
    if (K_BLOCK_MAX > 1) {
      copy(s2r_atom_a, tXsA_p(_,_,Int<0>{}), tXrA(_,_,Int<0>{}));
      copy(s2r_atom_b, tXsB_p(_,_,Int<0>{}), tXrB(_,_,Int<0>{}));
    }

    // Adjust k_tile_count for pipeline depth. In the non-WS kernel, the prologue
    // loaded K_PIPE_MAX-1 stages and decremented k_tile_count by K_PIPE_MAX-1.
    // Here the producer handles all loads, but the consumer still needs the same
    // accounting: total k_tiles - (K_PIPE_MAX-1) real iterations + K_PIPE_MAX-1
    // tail drain = total k_tiles main loop iterations.
    k_tile_count -= (K_PIPE_MAX - 1);

    // Main loop — same structure as non-WS kernel but with pipeline barriers
    // replacing manual ClusterTransactionBarrier + __syncthreads
    CUTE_NO_UNROLL
    while (k_tile_count > -(K_PIPE_MAX - 1))
    {
      CUTE_UNROLL
      for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
      {
        if (k_block == K_BLOCK_MAX - 1)
        {
          // Advance to next stage, then wait for its TMA load to complete
          ++smem_pipe_read;
          pipeline.consumer_wait(smem_pipe_read);
          tXsA_p = tXsA(_,_,_,smem_pipe_read.index());
          tXsB_p = tXsB(_,_,_,smem_pipe_read.index());
        }

        auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;
        copy(s2r_atom_a, tXsA_p(_,_,k_block_next), tXrA(_,_,k_block_next));
        copy(s2r_atom_b, tXsB_p(_,_,k_block_next), tXrB(_,_,k_block_next));

        if (k_block == 0)
        {
          // Release previous stage — safe because all k_blocks are in registers
          pipeline.consumer_release(smem_pipe_release);
          ++smem_pipe_release;
        }

        gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
      }
      --k_tile_count;
    }

  // ---- Step 6: Epilogue ----
  //
  // Stage 1: Element-wise alpha/beta scaling
  CUTE_UNROLL
  for (int i = 0; i < size(tCrC); ++i) {
    tCrC(i) = alpha * tCrC(i) + beta * static_cast<float>(tCgC(i));
  }

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

  // Stage 3: TMA store (smem -> gmem)
  //
  // The TMA store uses the testbed pattern:
  //   1. get_tma_tensor creates a TMA coord tensor for the FULL gmem
  //   2. flat_divide tiles it by the CTA tile size
  //   3. get_slice(Int<0>{}) partitions for single-CTA TMA
  //   4. partition_S/partition_D create the src/dst views
  //   5. copy issues the TMA store
  //
  // IMPORTANT: Unlike the testbed which iterates over REST modes, we select the
  // specific tile for this CTA using the CTA coord from blockIdx.
  auto cta_tile_mn = product_each(shape(SmemLayoutC{}));
  Tensor mC_tma = tma_store_c.get_tma_tensor(make_shape(M, N));
  Tensor gC_tma_full = flat_divide(mC_tma, cta_tile_mn);

  auto cta_tma_store = tma_store_c.get_slice(Int<0>{});
  Tensor tSsC_x = cta_tma_store.partition_S(sC);
  Tensor tSgC_x = cta_tma_store.partition_D(gC_tma_full);

  // Group the REST modes (tiles beyond the first)
  Tensor tSgC = group_modes<1, rank(tSgC_x)>(tSgC_x);
  Tensor tSsC = group_modes<1, rank(tSsC_x)>(tSsC_x);

  // Select the tile for this CTA
  // flat_divide creates (TILE_M, TILE_N, REST_M, REST_N) from the full gmem tensor.
  // After partition_D + group_modes, tSgC has shape (TMA, REST) where REST = REST_M * REST_N.
  // The linear REST index for this CTA is blockIdx.x + blockIdx.y * gridDim.x.
  int rest_idx = blockIdx.x + blockIdx.y * gridDim.x;

  if (threadIdx.x == 0) {
    tma_store_fence();
    copy(tma_store_c, tSsC(_, 0), tSgC(_, rest_idx));
    tma_store_arrive();
  }
  tma_store_wait<0>();
}

// ================================================================================================
// Host Function — configure and launch the kernel
// ================================================================================================

template <class Alpha, class Beta>
void
bf16_gemm_tn(int m, int n, int k,
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
  //
  // GMMA::Layout_K_SW128_Atom<bf16_t>: K-contiguous, SW128 swizzle
  //   Atom shape: (8, 64) in bf16 elements
  //   Swizzle<3,4,3> maps to TMA B128 mode
  //
  // GMMA::Layout_MN_SW128_Atom<bf16_t>: M-contiguous (column-major), SW128 swizzle
  //   Atom shape: (64, 8) in bf16 elements

  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16_t>{}, make_shape(bM, bK, bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16_t>{}, make_shape(bN, bK, bP));
  auto sC_layout = make_layout(make_shape(bM, bN), make_stride(Int<1>{}, bM));  // column-major, plain

  // TMA load atoms for A and B
  //
  // make_tma_atom inspects the gmem tensor and smem layout to create a TMA descriptor
  // that encodes strides, swizzle, and tile dimensions.
  Tensor mA = make_tensor(A, make_shape(M, K), dA);                       // (M,K) for TMA inspection
  Tensor mB = make_tensor(B, make_shape(N, K), dB);                       // (N,K) for TMA inspection

  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM, bK));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN, bK));

  // TMA store TiledCopy for C
  // Uses a plain column-major smem layout. STSM writes to plain smem (no hardware swizzle).
  // TMA store reads from the same plain smem with no swizzle conflicts.
  Tensor mC = make_tensor(C, make_shape(M, N), dC);                         // (M,N) for TMA inspection
  auto tma_store_c = make_tma_copy(SM90_TMA_STORE{}, mC, sC_layout, make_shape(bM, bN), Int<1>{});

  // TiledMMA (unchanged)
  TiledMMA mma = make_tiled_mma(
      SM80_16x8x16_F32BF16BF16F32_TN{},
      Layout<Shape<_4, _2>>{},
      Tile<Underscore, _64, Underscore>{});

  static_assert(decltype(size(mma))::value == 256, "Expected 256 threads");

  // R2S TiledCopy for STSM register->smem (same as pipe_epilogue.cu)
  auto r2s_copy = make_tiled_copy_C(
      Copy_Atom<SM90_U16x8_STSM_T, bf16_t>{},
      mma);

  // S2R (smem->register) copy atoms (unchanged)
  Copy_Atom<SM75_U32x4_LDSM_N, bf16_t> s2r_atom_a;
  Copy_Atom<SM75_U32x4_LDSM_N, bf16_t> s2r_atom_b;

  // Grid and block dimensions
  dim3 dimBlock(size(mma));
  dim3 dimCluster(1, 1, 1);
  dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));

  // Shared memory
  int smem_size = int(sizeof(SharedStorage<bf16_t, bf16_t, decltype(sA), decltype(sB)>));

  // Kernel function pointer
  auto* kernel_ptr = &bf16_gemm_device<
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

  // Launch via cluster launch API (required for TMA, even with single-CTA cluster)
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*)kernel_ptr,
      prob_shape, cta_tiler,
      A, tmaA, s2r_atom_a,
      B, tmaB, s2r_atom_b,
      C, sC_layout, tma_store_c, r2s_copy, dC, mma,
      alpha, beta);

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

int main(int argc, char** argv)
{
  using namespace cute;

  printf("BF16 GEMM (SM80 HMMA + SM90 TMA load/store, tile 256x128x64, 256 threads, STSM+TMAStore epilogue)\n\n");

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

    printf("Correctness (1024^3): max error %e — %s\n\n", max_err, max_err < 0.5f ? "PASS" : "FAIL");
    if (max_err >= 0.5f) return 1;
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
