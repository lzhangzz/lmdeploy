/***************************************************************************************************
 * BF16 GEMM using SM90 WGMMA tensor cores — RS Variant (A in Registers, B in Shared Memory)
 *
 * An educational variant of 09_bf16_gemm_sm90_pipe_tma_ws_persistent.cu that swaps the WGMMA
 * atom from SS (both operands in shared memory via GMMA descriptors) to RS (A in registers,
 * B in shared memory via descriptor). This demonstrates the register-vs-descriptor trade-off in
 * SM90 WGMMA.
 *
 * Key changes from 09 (SS variant):
 *   - WGMMA atom: RS (64x256x16_F32BF16BF16_RS) replaces SS (64x256x16_F32BF16BF16_SS)
 *   - S2R copy for A operand — A is loaded from smem into registers before each gemm call
 *     (RS requires register-source A with GMMA::Major::K layout)
 *   - B operand unchanged — still uses GMMA smem descriptor
 *   - Higher register pressure — A uses ~64B of register storage per thread (vs ~32B for SS descriptors)
 *
 * Unchanged from 09:
 *   - Tile size 128x256x64 (2 warpgroups of 64x256 each)
 *   - TMA load (A, B) + TMA store (C), PipelineTmaAsync 3-stage
 *   - Persistent scheduling, warp-specialized (384 threads: 128 producer + 256 consumer)
 *   - STSM BF16 epilogue
 *   - Smem ~213 KB (128x256x64 tile, 3 pipeline stages)
 *
 * C = alpha * A * B^T + beta * C
 *   A: bf16, M x K, row-major (TN layout)
 *   B: bf16, K x N, stored as (N, K) in CuTe with K-contiguous stride
 *   C: bf16, M x N, column-major
 *
 * Target: SM90 (uses SM90 WGMMA + SM90 TMA for gmem<->smem transfers)
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

template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class SmemLayoutC,
          class TmaStoreC, class R2SCopy, class CStride, class TiledMma,
          class Alpha, class Beta>
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

  // ---- Tile counts (computed once) ----

  int m_tiles = size(ceil_div(M, size<0>(cta_tiler)));
  int n_tiles = size(ceil_div(N, size<1>(cta_tiler)));
  int k_tile_count = size(ceil_div(K, size<2>(cta_tiler)));

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

  // ---- Pipeline states (created once, before warp group branch) ----

  auto smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
  typename MainloopPipeline::PipelineState smem_pipe_read;
  typename MainloopPipeline::PipelineState smem_pipe_release;

  uint64_t linear_idx = blockIdx.x;
  uint64_t grid_size = gridDim.x;

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

  } else {
    // ==================================================================
    // Consumer warp groups (wg 0 and 1) — WGMMA + epilogue (persistent)
    // ==================================================================
    cutlass::arch::warpgroup_reg_alloc<232>();

    // ---- Step 4: TiledMMA setup and register allocation (done once) ----

    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);

    // RS variant: A is register-source (ALayout_64x16), B is smem-source (GMMA descriptor).
    // partition_A yields smem views for each thread's S2R copy (no descriptor).
    // make_fragment_A allocates register storage — 32 bf16 values/thread (4 K-tiles x 8 val).
    // B is unchanged from SS: 4 GMMA descriptors (one per K-tile of 16).
    Tensor tCsA = thr_mma.partition_A(sA);                                  // (MMA, MMA_M, MMA_K, PIPE)
    Tensor tCsB = thr_mma.partition_B(sB);                                  // (MMA, MMA_N, MMA_K, PIPE)
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);                            // register bf16 fragment
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

        // S2R copy: load A from swizzled smem into registers.
        // tCsA is partitioned via ALayout_64x16 — each thread reads its 32 bf16
        // values (4 K-tiles x 8 val) from the correct swizzled smem addresses.
        copy(tCsA(_,_,_,smem_pipe_read.index()), tCrA);

        warpgroup_fence_operand(tCrC);
        warpgroup_arrive();
        // tCrA: register values (no pipe index — already loaded above)
        // tCrB: GMMA descriptors (pipe-indexed per K-tile)
        gemm(mma, tCrA,
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

// ================================================================================================
// Host Function — configure and launch the persistent kernel
// ================================================================================================

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

  // TMA load atoms for A and B
  Tensor tA = make_tensor(A, make_shape(M, K), dA);                       // (M,K) for TMA inspection
  Tensor tB = make_tensor(B, make_shape(N, K), dB);                       // (N,K) for TMA inspection

  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, tA, sA(_,_,0), make_shape(bM, bK));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, tB, sB(_,_,0), make_shape(bN, bK));

  // TMA store TiledCopy for C
  Tensor tC = make_tensor(C, make_shape(M, N), dC);                         // (M,N) for TMA inspection
  auto tma_store_c = make_tma_copy(SM90_TMA_STORE{}, tC, sC_layout, make_shape(bM, bN), Int<1>{});

  // TiledMMA — SM90 WGMMA RS variant (A in registers, B via smem descriptor)
  // RS requires A to be K-major (static_assert enforced in the arch atom).
  TiledMMA mma = make_tiled_mma(
      SM90_64x256x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::K>{},
      Layout<Shape<_2, _1>>{});

  static_assert(decltype(size(mma))::value == 256, "Expected 256 threads");

  // R2S TiledCopy for STSM register->smem
  auto r2s_copy = make_tiled_copy_C(
      Copy_Atom<SM90_U16x8_STSM_T, bf16_t>{},
      mma);

  // Grid and block dimensions — persistent: 1D grid of num_SMs blocks
  int num_SMs = 0;
  CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0));

  int total_tiles = size(ceil_div(M, bM)) * size(ceil_div(N, bN));

  dim3 dimBlock(size(mma) * 3 / 2);  // 384 threads: 256 MMA + 128 producer
  dim3 dimCluster(1, 1, 1);
  dim3 dimGrid(std::min(num_SMs, total_tiles));

  // Shared memory
  int smem_size = int(sizeof(SharedStorage<bf16_t, bf16_t, bf16_t, decltype(sA), decltype(sB), decltype(sC_layout), cute::size<2>(decltype(sA){})>));

  // Kernel function pointer
  auto* kernel_ptr = &bf16_gemm_persistent_device<
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

  // Launch via cluster launch API (required for TMA, even with single-CTA cluster)
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*)kernel_ptr,
      prob_shape, cta_tiler,
      A, tmaA,
      B, tmaB,
      C, sC_layout, tma_store_c, r2s_copy, dC, mma,
      alpha, beta,
      total_tiles);

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

  printf("BF16 GEMM (SM90 WGMMA + TMA load/store, tile 128x256x64, 384 threads WS, PipelineTmaAsync, PERSISTENT, RS variant)\n\n");

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
