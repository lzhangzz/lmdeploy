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

#include "06_split_a_pack.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include <cublas_v2.h>
#include <cmath>

template <class ElementA, int AStageElements, class ElementB, class ElementC,
          class SmemLayoutB, class SmemLayoutC, int Stages>
struct WgmmaSharedStorage
{
  alignas(128) cute::ArrayEngine<ElementA, AStageElements * Stages> A;
  alignas(128) cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
  alignas(128) cute::ArrayEngine<ElementC, cute::cosize_v<SmemLayoutC>> C;
  typename cutlass::PipelineTmaAsync<Stages>::SharedStorage pipeline;
};

template <class ProblemShape, class CtaTiler,
          class SmemLayoutA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class SmemLayoutC,
          class TmaStoreC, class R2SCopy, class CStride, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value * 3 / 2, 1)
void
split_a_wgmma_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                     bf16_t const* packed_A, SmemLayoutA,
                     TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
                     TC      * C, SmemLayoutC,
                     CUTLASS_GRID_CONSTANT TmaStoreC const tma_store_c,
                     R2SCopy r2s_copy, CStride dC, TiledMma mma,
                     Alpha alpha, Beta beta,
                     int total_tiles, int k_tile_count,
                     int log_swizzle, int swizzle_size)
{
  using namespace cute;

  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

  auto [M, N, K] = shape_MNK;

  Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);

  extern __shared__ char shared_memory[];
  static_assert(decltype(size<0>(cta_tiler))::value == 128);
  static_assert(decltype(size<2>(cta_tiler))::value == 64);
  constexpr int a_stage_elements = 128 * 64;
  using SharedStorage = WgmmaSharedStorage<bf16_t, a_stage_elements, TB, TC,
      SmemLayoutB, SmemLayoutC, cute::size<2>(SmemLayoutB{})>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});
  Tensor sC = make_tensor(make_smem_ptr(smem.C.begin()), SmemLayoutC{});

  int m_tiles = size(ceil_div(M, size<0>(cta_tiler)));
  int n_tiles = size(ceil_div(N, size<1>(cta_tiler)));

  constexpr int tma_transaction_bytes =
      sizeof(bf16_t) * a_stage_elements
    + sizeof(TB) * cute::cosize_v<SmemLayoutB> / cute::size<2>(SmemLayoutB{});

  int warp_group_idx = cutlass::canonical_warp_group_idx();
  int warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

  using MainloopPipeline = cutlass::PipelineTmaAsync<cute::size<2>(SmemLayoutB{})>;
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

  auto smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
  typename MainloopPipeline::PipelineState smem_pipe_read;
  typename MainloopPipeline::PipelineState smem_pipe_release;

  uint64_t linear_idx = blockIdx.x;
  uint64_t grid_size = gridDim.x;

  if (warp_group_idx == 2) {
    // ==================================================================
    // Producer warp group — bulk copy for A + TMA for B (persistent)
    //   UNCHANGED from iter 04
    // ==================================================================
    cutlass::arch::warpgroup_reg_dealloc<40>();

    if (warp_group_thread_idx == 0) {
      using BarrierType = typename MainloopPipeline::ProducerBarrierType;

      if (linear_idx == 0) { } // avoid unused variable warning

      while (linear_idx < total_tiles) {
        int offset = linear_idx & (swizzle_size - 1);
        int extra  = linear_idx >> log_swizzle;
        int n_idx  = extra % n_tiles;
        int m_idx  = (extra / n_tiles) * swizzle_size + offset;

        if (m_idx >= m_tiles || n_idx >= n_tiles) {
          linear_idx += grid_size;
          continue;
        }

        Tensor gB = local_tile(mB, cta_tiler, make_coord(m_idx, n_idx, _), Step< X,_1, _1>{});
        auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                           group_modes<0,2>(sB), group_modes<0,2>(gB));

        CUTE_NO_UNROLL
        for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
          pipeline.producer_acquire(smem_pipe_write);
          BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
          int write_stage = smem_pipe_write.index();

          // Issue bulk copy for packed A: gmem -> smem_A[write_stage]
          {
            int a_tile_idx = m_idx * k_tile_count + k_tile;
            SM90_BULK_COPY_G2S::copy(
                packed_A + a_tile_idx * a_stage_elements,
                tma_barrier,
                smem.A.begin() + write_stage * a_stage_elements,
                a_stage_elements * sizeof(bf16_t));
          }

          // Issue TMA for B
          copy(tma_b.with(*tma_barrier), tBgB(_,k_tile), tBsB(_,write_stage));

          ++smem_pipe_write;
        }

        linear_idx += grid_size;
      }
      pipeline.producer_tail(smem_pipe_write);
    }

    cute::tma_store_wait<0>();

  } else {
    // ==================================================================
    // Consumer warp groups (wg 0 and 1)
    //   Prologue + k_block-interleaved main loop + tail
    // ==================================================================
    cutlass::arch::warpgroup_reg_alloc<232>();

    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);

    // A smem tensor for fragment creation only. Points to stage 0 of A smem.
    Tensor dummy_sA = make_tensor(make_smem_ptr(reinterpret_cast<bf16_t*>(shared_memory)), SmemLayoutA{});
    Tensor dummy_tCsA = thr_mma.partition_A(dummy_sA);
    Tensor tCrA = thr_mma.make_fragment_A(dummy_tCsA(_,_,_,Int<0>{}));

    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);

    constexpr int k_block_count = size<2>(tCrA);

    Tensor gC_dummy = make_tensor(make_gmem_ptr(C),
                                  make_shape(size<0>(cta_tiler), size<1>(cta_tiler)), dC);
    Tensor tCrC = thr_mma.make_fragment_C(thr_mma.partition_C(gC_dummy));

    ThrCopy thr_r2s = r2s_copy.get_slice(threadIdx.x);

    auto cta_tile_mn = product_each(shape(SmemLayoutC{}));
    Tensor mC_tma = tma_store_c.get_tma_tensor(make_shape(M, N));
    Tensor gC_tma_full = flat_divide(mC_tma, cta_tile_mn);

    auto cta_tma_store = tma_store_c.get_slice(Int<0>{});
    Tensor tSsC_x = cta_tma_store.partition_S(sC);
    Tensor tSgC_x = cta_tma_store.partition_D(gC_tma_full);
    Tensor tSgC = group_modes<1, rank(tSgC_x)>(tSgC_x);
    Tensor tSsC = group_modes<1, rank(tSsC_x)>(tSsC_x);

    // Per-warpgroup constants for S2R loads
    int wg_id = warp_group_idx;
    int local_tid = threadIdx.x % 128;

    // Helper: load a single k_block from per-warpgroup smem region into registers
    auto load_k_block = [&](int kb, int stage) {
      Tensor sA_packed = make_tensor(
          make_smem_ptr(smem.A.begin() + stage * a_stage_elements + wg_id * 4096),
          make_shape(Int<8>{}, Int<128>{}, Int<4>{}));
      Tensor sP_k = sA_packed(_, local_tid, kb);  // (8,) stride (1,)
      Tensor rA_k = make_tensor(tCrA.data() + kb * size<0>(tCrA), make_shape(Int<8>{}));
      copy(AutoVectorizingCopy{}, sP_k, rA_k);
    };

    while (linear_idx < total_tiles) {
      int offset = linear_idx & (swizzle_size - 1);
      int extra  = linear_idx >> log_swizzle;
      int n_idx  = extra % n_tiles;
      int m_idx  = (extra / n_tiles) * swizzle_size + offset;

      if (m_idx >= m_tiles || n_idx >= n_tiles) {
        linear_idx += grid_size;
        continue;
      }

      Tensor gC = local_tile(mC, cta_tiler, make_coord(m_idx, n_idx, _), Step<_1, _1, X>{});
      Tensor tCgC = thr_mma.partition_C(gC);

      clear(tCrC);

      int read_stage;
      int k_tiles = k_tile_count;

      warpgroup_fence_operand(tCrC);

      // ================================================================
      // PROLOGUE: iter 04 style (all loads + wait<2> per gemm)
      // ================================================================
      {
        pipeline.consumer_wait(smem_pipe_read);
        read_stage = smem_pipe_read.index();
        ++smem_pipe_read;

        for (int kb = 0; kb < k_block_count; ++kb) {
          load_k_block(kb, read_stage);
        }

        warpgroup_fence_operand(tCrC);

        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < k_block_count; ++k_block) {
          warpgroup_arrive();
          gemm(mma, tCrA(_,_,k_block),
                    tCrB(_,_,k_block,read_stage), tCrC);
          warpgroup_commit_batch();
          warpgroup_wait<2>();
        }

        warpgroup_fence_operand(tCrC);
        --k_tiles;
      }

      // NO drain — 2 pending WGMMA from prologue

      if (k_tiles > 0)
      {
        // Main loop with try_wait prefetch (first attempt)
        auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);

        CUTE_NO_UNROLL
        for (int k_tile_iter = 0; k_tile_iter < k_tiles; ++k_tile_iter)
        {
          pipeline.consumer_wait(smem_pipe_read, barrier_token);
          read_stage = smem_pipe_read.index();
          ++smem_pipe_read;

          load_k_block(0, read_stage);

          warpgroup_fence_operand(tCrC);

          CUTLASS_PRAGMA_UNROLL
          for (int k_block = 0; k_block < k_block_count; ++k_block) {
            if (k_block < k_block_count - 1) {
              load_k_block(k_block + 1, read_stage);
            }

            warpgroup_arrive();
            gemm(mma, tCrA(_,_,k_block),
                      tCrB(_,_,k_block,read_stage), tCrC);
            warpgroup_commit_batch();
            warpgroup_wait<2>();

            if (k_block == 1) {
              pipeline.consumer_release(smem_pipe_release);
              ++smem_pipe_release;
            }
          }

          warpgroup_fence_operand(tCrC);

          if (k_tile_iter < k_tiles - 1) {
            barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
          }
        }
      }

      // ================================================================
      // TAIL: drain all WGMMA, release last stage
      // ================================================================
      warpgroup_wait<0>();
      warpgroup_fence_operand(tCrC);
      pipeline.consumer_release(smem_pipe_release);
      ++smem_pipe_release;

      // ================================================================
      // EPILOGUE — UNCHANGED from iter 04
      // ================================================================

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
      tma_store_wait<0>();

      linear_idx += grid_size;
    }
  }
}

// ================================================================================================
// Host function
// ================================================================================================

template <class Alpha, class Beta>
void
split_a_wgmma(int m, int n, int k,
              Alpha alpha,
              bf16_t const* packed_A,
              bf16_t const* B, int ldB,
              Beta beta,
              bf16_t* C, int ldC,
              cudaStream_t stream = 0)
{
  using namespace cute;

  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);

  auto dB = make_stride(ldB, Int<1>{});
  auto dC = make_stride(Int<1>{}, ldC);

  auto bM = Int<128>{};
  auto bN = Int<256>{};
  auto bK = Int<64>{};
  auto cta_tiler = make_shape(bM, bN, bK);
  auto bP = Int<3>{};

  // Dummy A smem layout for fragment creation (1 stage — no A pipeline)
  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16_t>{}, make_shape(bM, bK, Int<1>{}));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16_t>{}, make_shape(bN, bK, bP));
  auto sC_layout = make_layout(make_shape(bM, bN), make_stride(Int<1>{}, bM));

  Tensor tB = make_tensor(B, make_shape(N, K), dB);
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, tB, sB(_,_,0), make_shape(bN, bK));

  Tensor tC = make_tensor(C, make_shape(M, N), dC);
  auto tma_store_c = make_tma_copy(SM90_TMA_STORE{}, tC, sC_layout, make_shape(bM, bN), Int<1>{});

  TiledMMA mma = make_tiled_mma(
      SM90_64x256x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::K>{},
      Layout<Shape<_2, _1>>{});

  static_assert(decltype(size(mma))::value == 256, "Expected 256 threads");

  auto r2s_copy = make_tiled_copy_C(
      Copy_Atom<SM90_U16x8_STSM_T, bf16_t>{},
      mma);

  int num_SMs = 0;
  CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0));

  int m_tiles = size(ceil_div(M, bM));
  int n_tiles = size(ceil_div(N, bN));
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

  constexpr int a_stage_elements_host = 128 * 64;
  int smem_size = int(sizeof(WgmmaSharedStorage<bf16_t, a_stage_elements_host, bf16_t, bf16_t,
      decltype(sB), decltype(sC_layout), cute::size<2>(decltype(sB){})>));

  auto* kernel_ptr = &split_a_wgmma_device<
      decltype(prob_shape), decltype(cta_tiler),
      decltype(sA),
      bf16_t, decltype(sB), decltype(tmaB),
      bf16_t, decltype(sC_layout),
      decltype(tma_store_c), decltype(r2s_copy), decltype(dC), decltype(mma),
      Alpha, Beta>;

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      (void const*)kernel_ptr,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size));
  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      (void const*)kernel_ptr,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      100));

  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*)kernel_ptr,
      prob_shape, cta_tiler,
      packed_A, sA,
      B, tmaB,
      C, sC_layout, tma_store_c, r2s_copy, dC, mma,
      alpha, beta,
      total_tiles_padded, k_tile_count,
      log_swizzle, swizzle_size);

  CUTE_CHECK_LAST();
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed at WGMMA kernel launch" << std::endl;
  }
}

// ================================================================================================
// Main — end-to-end test: pack A, run WGMMA, verify against CPU reference
// ================================================================================================

int main(int argc, char** argv)
{
  using namespace cute;

  printf("BF16 Split A WGMMA iter 06 (SM90, threadblock swizzling)\n\n");

  float alpha = 1.0f;
  float beta  = 0.0f;

  auto run_test = [&](int m, int n, int k, const char* label) -> bool {
    int ldA = k, ldB = k, ldC = m;

    thrust::host_vector<bf16_t> h_A(m * k), h_B(n * k), h_C(m * n, static_cast<bf16_t>(0.0f));
    for (int i = 0; i < m * k; ++i) h_A[i] = static_cast<bf16_t>(2.0 * (rand() / double(RAND_MAX)) - 1.0);
    for (int i = 0; i < n * k; ++i) h_B[i] = static_cast<bf16_t>(2.0 * (rand() / double(RAND_MAX)) - 1.0);

    thrust::device_vector<bf16_t> d_A = h_A, d_B = h_B, d_C = h_C;
    thrust::device_vector<bf16_t> d_packed(m * k);

    split_a_pack(m, k, d_A.data().get(), ldA, d_packed.data().get());
    CUTE_CHECK_LAST();

    split_a_wgmma(m, n, k, alpha, d_packed.data().get(),
                  d_B.data().get(), ldB, beta, d_C.data().get(), ldC);
    CUTE_CHECK_LAST();

    thrust::host_vector<bf16_t> h_result = d_C;

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

    printf("%s (%dx%dx%d): max error %e — %s\n", label, m, n, k, max_err, max_err < 0.5f ? "PASS" : "FAIL");
    return max_err < 0.5f;
  };

  // Correctness tests
  if (!run_test(128,  256,  64,   "Single tile"))  return 1;
  if (!run_test(256,  256,  64,   "Multi-M"))      return 1;
  if (!run_test(128,  256,  128,  "Multi-K"))      return 1;
  if (!run_test(256,  256,  128,  "Multi-MK"))     return 1;
  if (!run_test(128,  512,  64,   "Multi-N"))      return 1;
  if (!run_test(128,  256,  256,  "K=4"))          return 1;
  if (!run_test(512,  512,  512,  "Large"))        return 1;
  if (!run_test(1024, 1024, 1024, "1024^3"))       return 1;
  if (!run_test(2048, 1024, 512,  "2048x1024x512"))return 1;

  printf("\n");

  // Benchmark
  printf("Benchmark (100 iterations each):\n");
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);

  for (int size : {256, 512, 1024, 2048, 4096, 8192}) {
    int m = size, n = size, k = size;
    int ldA = k, ldB = k, ldC = m;

    thrust::device_vector<bf16_t> d_A(m * k), d_B(n * k), d_C(m * n), d_packed(m * k);
    thrust::host_vector<bf16_t> h_A(m * k), h_B(n * k);
    for (int i = 0; i < m * k; ++i) h_A[i] = static_cast<bf16_t>(2.0 * (rand() / double(RAND_MAX)) - 1.0);
    for (int i = 0; i < n * k; ++i) h_B[i] = static_cast<bf16_t>(2.0 * (rand() / double(RAND_MAX)) - 1.0);
    d_A = h_A; d_B = h_B;

    split_a_pack(m, k, d_A.data().get(), ldA, d_packed.data().get());
    CUTE_CHECK_LAST();

    const int timing_iterations = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Custom kernel benchmark
    split_a_wgmma(m, n, k, alpha, d_packed.data().get(),
                  d_B.data().get(), ldB, beta, d_C.data().get(), ldC);
    CUTE_CHECK_LAST();

    cudaEventRecord(start);
    for (int i = 0; i < timing_iterations; ++i) {
      split_a_wgmma(m, n, k, alpha, d_packed.data().get(),
                    d_B.data().get(), ldB, beta, d_C.data().get(), ldC);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    double avg_ms = total_ms / timing_iterations;
    double gflops = (2.0 * m * n * k) * 1e-9;
    printf("  %dx%dx%d custom: %.1f GFLOP/s (%.4f ms)\n", m, n, k, gflops / (avg_ms * 1e-3), avg_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // cuBLAS benchmark
    thrust::device_vector<bf16_t> d_C_ref(m * n);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float alpha_f = 1.0f, beta_f = 0.0f;
    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 m, n, k, &alpha_f,
                 d_A.data().get(), CUDA_R_16BF, ldA,
                 d_B.data().get(), CUDA_R_16BF, ldB,
                 &beta_f,
                 d_C_ref.data().get(), CUDA_R_16BF, ldC,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cudaEventRecord(start);
    for (int i = 0; i < timing_iterations; ++i) {
      cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                   m, n, k, &alpha_f,
                   d_A.data().get(), CUDA_R_16BF, ldA,
                   d_B.data().get(), CUDA_R_16BF, ldB,
                   &beta_f,
                   d_C_ref.data().get(), CUDA_R_16BF, ldC,
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&total_ms, start, stop);
    avg_ms = total_ms / timing_iterations;
    printf("  %dx%dx%d cublas: %.1f GFLOP/s (%.4f ms)\n", m, n, k, gflops / (avg_ms * 1e-3), avg_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  cublasDestroy(cublas_handle);

  return 0;
}
