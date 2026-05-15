/***************************************************************************************************
 * BF16 GEMM using SM90 WGMMA with pre-packed operand A (iteration 09).
 *
 * Real asymmetric W4A16: per-(M, K_group) bf16 scales and zero points (group size
 * 128 along K), applied on the fly via unpack_dequant_to_bf16 in the consumer.
 *
 * Changes from iteration 08:
 *   - Second smem pipeline (PipelineTmaAsync<2>, period = 2 k_tiles, 512 B/stage)
 *     delivers per-group (scale, eff_zero) pairs into smem
 *   - Consumer derives m_0 = 8*(tid%4) + tid/32, refreshes 4 cached bf16x2 pairs
 *     (lo_pair_s/z, hi_pair_s/z) at every group boundary via 4 LDS.32 + 4 PRMT
 *   - load_k_block calls unpack_dequant_to_bf16 (4 lop3 + 4 HSUB2 + 4 HMUL2)
 *     instead of unpack_u4_to_bf16 (the −128 step is folded into eff_zero by host)
 *
 * Mock data: random uint4 weights + random bf16 scales/zeros. CPU reference
 * dequantizes with the same per-group params and matmul-accumulates in fp32.
 *
 * Target: SM90
 **************************************************************************************************/

#include "09_split_a_pack.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include <cublas_v2.h>
#include <cmath>

template <class ElementA, int AStageElements, class ElementB, class ElementC,
          class SmemLayoutB, class SmemLayoutC, int Stages, int SzStages>
struct WgmmaSharedStorage
{
  alignas(128) cute::ArrayEngine<ElementA, AStageElements * Stages> A;
  alignas(128) cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
  alignas(128) cute::ArrayEngine<ElementC, cute::cosize_v<SmemLayoutC>> C;
  alignas(128) cute::ArrayEngine<uint32_t, 128 * SzStages> SZ;  // 128 uint32 per stage
  typename cutlass::PipelineTmaAsync<Stages>::SharedStorage   pipeline;
  typename cutlass::PipelineTmaAsync<SzStages>::SharedStorage pipeline_sz;
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
                     uint32_t const* packed_A, SmemLayoutA,
                     uint32_t const* packed_sz,
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
  constexpr int a_stage_elements = 1024;  // 2 warpgroups x 128 threads x 4 k_blocks x 1 uint32
  constexpr int sz_stages   = 2;           // SZ pipeline stages
  constexpr int group_size  = 128;         // K elements per quant group
  // Iter 09 precondition: each group is exactly 2 k_tiles (group_size / bK = 128 / 64 = 2),
  // and every k_tile must lie wholly within one group → K must be a multiple of group_size.
  // The runtime check on K is done in the host function.
  static_assert(group_size % static_cast<int>(decltype(size<2>(cta_tiler))::value) == 0,
                "iter 09 requires group_size (128) to be a multiple of bK");

  using SharedStorage = WgmmaSharedStorage<uint32_t, a_stage_elements, TB, TC,
      SmemLayoutB, SmemLayoutC, cute::size<2>(SmemLayoutB{}), sz_stages>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});
  Tensor sC = make_tensor(make_smem_ptr(smem.C.begin()), SmemLayoutC{});

  int m_tiles = size(ceil_div(M, size<0>(cta_tiler)));
  int n_tiles = size(ceil_div(N, size<1>(cta_tiler)));

  constexpr int tma_transaction_bytes =
      sizeof(uint32_t) * a_stage_elements
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

  // SZ pipeline (per-group): 1 bulk copy of 128 uint32 = 512 B per group
  using SzPipeline = cutlass::PipelineTmaAsync<sz_stages>;
  typename SzPipeline::Params sz_pipeline_params;
  sz_pipeline_params.role = (warp_group_idx == 2)
      ? SzPipeline::ThreadCategory::Producer
      : SzPipeline::ThreadCategory::Consumer;
  sz_pipeline_params.is_leader = (warp_group_thread_idx == 0);
  sz_pipeline_params.num_consumers = 256;
  sz_pipeline_params.num_producers = 1;
  sz_pipeline_params.transaction_bytes = 128 * sizeof(uint32_t);

  SzPipeline sz_pipeline(smem.pipeline_sz, sz_pipeline_params,
                         cute::make_layout(cute::make_shape(cute::_1{}, cute::_1{})));
  __syncthreads();

  auto smem_pipe_write    = cutlass::make_producer_start_state<MainloopPipeline>();
  typename MainloopPipeline::PipelineState smem_pipe_read;
  typename MainloopPipeline::PipelineState smem_pipe_release;

  auto smem_sz_pipe_write = cutlass::make_producer_start_state<SzPipeline>();
  typename SzPipeline::PipelineState smem_sz_pipe_read;

  uint64_t linear_idx = blockIdx.x;
  uint64_t grid_size = gridDim.x;

  if (warp_group_idx == 2) {
    // ==================================================================
    // Producer warp group -- bulk copy for A + TMA for B (persistent)
    // Group-aware: 1 SZ bulk copy + 2 AB k_tile copies per group.
    // ==================================================================
    cutlass::arch::warpgroup_reg_dealloc<40>();

    if (warp_group_thread_idx == 0) {
      using BarrierType = typename MainloopPipeline::ProducerBarrierType;

      if (linear_idx == 0) { } // avoid unused variable warning

      int k_group_count = k_tile_count / 2;

      while (linear_idx < total_tiles) {
        // M-axis swizzle
        int m_base = linear_idx / n_tiles;
        int n_idx  = linear_idx % n_tiles;
        int m_low  = m_base & (swizzle_size - 1);
        int m_high = m_base >> log_swizzle;
        int m_idx  = m_high + m_low * (m_tiles >> log_swizzle);

        if (m_idx >= m_tiles || n_idx >= n_tiles) {
          linear_idx += grid_size;
          continue;
        }

        Tensor gB = local_tile(mB, cta_tiler, make_coord(m_idx, n_idx, _), Step< X,_1, _1>{});
        auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                           group_modes<0,2>(sB), group_modes<0,2>(gB));

        CUTE_NO_UNROLL
        for (int k_group_idx = 0; k_group_idx < k_group_count; ++k_group_idx) {
          // SZ acquire + bulk copy: 1 group's (scale, eff_zero) pairs into smem.SZ
          sz_pipeline.producer_acquire(smem_sz_pipe_write);
          typename SzPipeline::ProducerBarrierType* sz_barrier =
              sz_pipeline.producer_get_barrier(smem_sz_pipe_write);
          int sz_write_stage = smem_sz_pipe_write.index();

          SM90_BULK_COPY_G2S::copy(
              packed_sz + k_group_idx * M + m_idx * 128,
              sz_barrier,
              smem.SZ.begin() + sz_write_stage * 128,
              128 * sizeof(uint32_t));

          ++smem_sz_pipe_write;

          // 2 k_tiles per group: AB acquire + A bulk copy + B TMA (unchanged from iter 08)
          CUTLASS_PRAGMA_UNROLL
          for (int k_tile_in_group = 0; k_tile_in_group < 2; ++k_tile_in_group) {
            int k_tile = k_group_idx * 2 + k_tile_in_group;

            pipeline.producer_acquire(smem_pipe_write);
            BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
            int write_stage = smem_pipe_write.index();

            // A bulk copy
            {
              int a_tile_idx = m_idx * k_tile_count + k_tile;
              SM90_BULK_COPY_G2S::copy(
                  packed_A + a_tile_idx * a_stage_elements,
                  tma_barrier,
                  smem.A.begin() + write_stage * a_stage_elements,
                  a_stage_elements * sizeof(uint32_t));
            }

            // B TMA
            copy(tma_b.with(*tma_barrier), tBgB(_,k_tile), tBsB(_,write_stage));

            ++smem_pipe_write;
          }
        }

        linear_idx += grid_size;
      }
      pipeline.producer_tail(smem_pipe_write);
      sz_pipeline.producer_tail(smem_sz_pipe_write);
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

    // Per-thread M row indices for SZ access (global within the CTA M-tile of 128 rows).
    // CuTe ALayout_64x16 has K-major codomain: each thread holds 8 BF16 values covering
    // exactly 2 distinct M rows × 4 distinct K cols. Empirically:
    //   m_lo (per-WG, ∈ [0, 63)) = (local_tid / 4) % 8 + 16 * (local_tid / 32)
    //   m_hi = m_lo + 8
    // Each warpgroup covers 64 of the 128 M rows (WG 0: 0..63, WG 1: 64..127), so
    // we offset by wg_id * 64 to index smem.SZ[0..127].
    int m_lo = wg_id * 64 + (local_tid / 4) % 8 + 16 * (local_tid / 32);
    int m_hi = m_lo + 8;

    // Cached scale/zero broadcast pairs, refreshed at every group boundary.
    // Each is bf16x2 with the SAME bf16 in both halves (broadcast).
    uint32_t scale_lo_pair = 0, zero_lo_pair = 0;
    uint32_t scale_hi_pair = 0, zero_hi_pair = 0;

    // Helper: load a single k_block from per-warpgroup smem region and dequantize
    // using the cached (scale, eff_zero) pairs.
    // NOTE: smem.A is ArrayEngine<uint32_t, ...>, so begin() returns uint32_t*.
    // Pointer arithmetic is in uint32 units -- do NOT multiply by sizeof(uint32_t).
    auto load_k_block = [&](int kb, int stage) {
      uint32_t* smem_base = smem.A.begin() + stage * a_stage_elements + wg_id * 512;
      uint32_t packed = smem_base[local_tid + kb * 128];
      unpack_dequant_to_bf16(packed,
                             scale_lo_pair, zero_lo_pair,
                             scale_hi_pair, zero_hi_pair,
                             reinterpret_cast<nv_bfloat16*>(tCrA.data() + kb * size<0>(tCrA)));
    };

    // Helper: refresh cached (scale, eff_zero) bf16x2 broadcast pairs from current
    // SZ stage. Called once at the start of each new group (= every 2 k_tiles).
    // SZ smem entry layout: low halfword = scale_bf16, high halfword = eff_zero_bf16.
    auto refresh_sz = [&]() {
      sz_pipeline.consumer_wait(smem_sz_pipe_read);
      int sz_stage = smem_sz_pipe_read.index();
      uint32_t* sz_smem_stage = smem.SZ.begin() + sz_stage * 128;

      uint32_t r_lo = sz_smem_stage[m_lo];
      uint32_t r_hi = sz_smem_stage[m_hi];

      // Broadcast each bf16 into both halves of a bf16x2.
      uint32_t s_lo = r_lo & 0xFFFFu;
      uint32_t z_lo = (r_lo >> 16) & 0xFFFFu;
      uint32_t s_hi = r_hi & 0xFFFFu;
      uint32_t z_hi = (r_hi >> 16) & 0xFFFFu;

      scale_lo_pair = (s_lo << 16) | s_lo;
      zero_lo_pair  = (z_lo << 16) | z_lo;
      scale_hi_pair = (s_hi << 16) | s_hi;
      zero_hi_pair  = (z_hi << 16) | z_hi;

      sz_pipeline.consumer_release(smem_sz_pipe_read);
      ++smem_sz_pipe_read;
    };

    while (linear_idx < total_tiles) {
      // M-axis swizzle (matches producer)
      int m_base = linear_idx / n_tiles;
      int n_idx  = linear_idx % n_tiles;
      int m_low  = m_base & (swizzle_size - 1);
      int m_high = m_base >> log_swizzle;
      int m_idx  = m_high + m_low * (m_tiles >> log_swizzle);

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

      // SZ refresh for group 0 (covers prologue + 1 main-loop iter)
      refresh_sz();

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

      // NO drain -- 2 pending WGMMA from prologue

      if (k_tiles > 0)
      {
        // Main loop with try_wait prefetch (first attempt)
        auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);

        CUTE_NO_UNROLL
        for (int k_tile_iter = 0; k_tile_iter < k_tiles; ++k_tile_iter)
        {
          // Group boundary: k_tile_iter is odd ↔ next k_tile starts a new group.
          // (k_tile_iter==0 corresponds to global k_tile_iter==1, still in group 0.)
          if ((k_tile_iter & 1) == 1) {
            refresh_sz();
          }

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
      // EPILOGUE -- deferred TMA store wait for epilogue-compute overlap
      // ================================================================

      // Wait for PREVIOUS tile's TMA store to complete before overwriting sC.
      // For the first tile this is a no-op (no pending stores).
      tma_store_wait<0>();

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
      // NO tma_store_wait here -- TMA store overlaps with next tile's compute.
      // The producer's tma_store_wait<0>() (line ~160) ensures the last tile's
      // store completes before kernel exit.

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
              uint32_t const* packed_A,
              uint32_t const* packed_sz,
              bf16_t const* B, int ldB,
              Beta beta,
              bf16_t* C, int ldC,
              int log_swizzle_override = -1,
              cudaStream_t stream = 0)
{
  using namespace cute;

  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);

  // Iter 09 precondition: K must be a multiple of group_size (128) so every
  // k_tile lies wholly within one (scale, zero) group.
  if (K % 128 != 0) {
    std::cerr << "iter 09: K (" << K << ") must be a multiple of group_size (128)\n";
    return;
  }

  auto dB = make_stride(ldB, Int<1>{});
  auto dC = make_stride(Int<1>{}, ldC);

  auto bM = Int<128>{};
  auto bN = Int<256>{};
  auto bK = Int<64>{};
  auto cta_tiler = make_shape(bM, bN, bK);
  auto bP = Int<3>{};

  // Dummy A smem layout for fragment creation (1 stage -- no A pipeline)
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
  int total_tiles = m_tiles * n_tiles;
  int k_tile_count = size(ceil_div(K, bK));

  // Swizzle: no swizzle is optimal for this kernel (empirically verified).
  // Override via CLI for experimentation. log_swizzle=0 means row-major (no swizzle).
  int log_swizzle = (log_swizzle_override >= 0) ? log_swizzle_override : 0;
  int swizzle_size = 1 << log_swizzle;

  // N-swizzle: m stays row-major, n is interleaved. No m_tiles padding needed.
  // n_tiles must be a multiple of swizzle_size for bijective mapping (true for
  // power-of-2 problem sizes where n_tiles = N/256 is a power of 2).
  int total_tiles_padded = total_tiles;

  dim3 dimBlock(size(mma) * 3 / 2);
  dim3 dimCluster(1, 1, 1);
  dim3 dimGrid(std::min(128, total_tiles_padded));

  constexpr int a_stage_elements_host = 1024;  // uint32 count per stage (2 warpgroups)
  constexpr int sz_stages_host = 2;
  int smem_size = int(sizeof(WgmmaSharedStorage<uint32_t, a_stage_elements_host, bf16_t, bf16_t,
      decltype(sB), decltype(sC_layout), cute::size<2>(decltype(sB){}), sz_stages_host>));

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
      packed_sz,
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
// Main -- end-to-end test: pack A, run WGMMA, verify against CPU reference
// ================================================================================================

// Generate per-group bf16 scales and uint4 zero points, pack them into a single
// device buffer of bf16x2 (scale_lo, eff_zero_hi) per (group, M).  Returns the
// integer zero points h_z_int (uint8) for the CPU reference.
inline thrust::host_vector<uint8_t>
generate_and_pack_sz(int M, int K,
                     thrust::host_vector<bf16_t>& h_scales,
                     thrust::device_vector<uint32_t>& d_sz_packed)
{
  constexpr int group_size = 128;
  int K_g = K / group_size;

  thrust::host_vector<uint8_t>  h_z_int(K_g * M);
  h_scales.resize(K_g * M);
  thrust::host_vector<uint32_t> h_sz_packed(K_g * M);

  for (int g = 0; g < K_g; ++g) {
    for (int m = 0; m < M; ++m) {
      float s = 0.25f + 3.75f * (rand() / float(RAND_MAX));
      int   z = rand() % 16;
      h_scales[g * M + m] = static_cast<bf16_t>(s);
      h_z_int [g * M + m] = static_cast<uint8_t>(z);

      bf16_t   eff_zero = static_cast<bf16_t>(float(z + 128));
      uint32_t s_bits   = uint32_t(reinterpret_cast<uint16_t const&>(h_scales[g * M + m]));
      uint32_t z_bits   = uint32_t(reinterpret_cast<uint16_t const&>(eff_zero));
      h_sz_packed[g * M + m] = (z_bits << 16) | s_bits;
    }
  }

  d_sz_packed.resize(K_g * M);
  d_sz_packed = h_sz_packed;
  return h_z_int;
}

int main(int argc, char** argv)
{
  using namespace cute;

  printf("BF16 Split A WGMMA iter 09 (SM90, uint4 dequant + per-group bf16 scales/zeros)\n\n");

  float alpha = 1.0f;
  float beta  = 0.0f;

  auto run_test = [&](int m, int n, int k, const char* label) -> bool {
    int ldA = k, ldB = k, ldC = m;
    constexpr int group_size = 128;

    // A: random uint4 in [0,15] encoded into bf16's low nibble (iter 08 convention)
    thrust::host_vector<bf16_t> h_A(m * k), h_B(n * k), h_C(m * n, static_cast<bf16_t>(0.0f));
    thrust::host_vector<float>  h_A_val(m * k);
    for (int i = 0; i < m * k; ++i) {
      float val = float(rand() % 16);
      h_A_val[i] = val;
      uint16_t raw = 0x3F80u | (uint16_t(val) & 0xFu);
      h_A[i] = reinterpret_cast<bf16_t const&>(raw);
    }
    for (int i = 0; i < n * k; ++i) h_B[i] = static_cast<bf16_t>(2.0 * (rand() / double(RAND_MAX)) - 1.0);

    // Scales / zero points (per-(K_group, M))
    thrust::host_vector<bf16_t>     h_scales;
    thrust::device_vector<uint32_t> d_sz_packed;
    thrust::host_vector<uint8_t>    h_z_int =
        generate_and_pack_sz(m, k, h_scales, d_sz_packed);

    thrust::device_vector<bf16_t> d_A = h_A, d_B = h_B, d_C = h_C;
    int total_tiles = ((m + 127) / 128) * ((k + 63) / 64);
    thrust::device_vector<uint32_t> d_packed(total_tiles * 1024);

    split_a_pack(m, k, d_A.data().get(), ldA, d_packed.data().get());
    CUTE_CHECK_LAST();

    split_a_wgmma(m, n, k, alpha,
                  d_packed.data().get(),
                  d_sz_packed.data().get(),
                  d_B.data().get(), ldB, beta, d_C.data().get(), ldC);
    CUTE_CHECK_LAST();

    thrust::host_vector<bf16_t> h_result = d_C;

    // CPU reference: dequantize per group, accumulate in fp32
    thrust::host_vector<float> h_ref(m * n, 0.0f);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) {
        float sum = 0.0f;
        for (int l = 0; l < k; ++l) {
          int   g = l / group_size;
          float q = h_A_val[i * k + l];
          float z = float(h_z_int[g * m + i]);
          float s = float(h_scales[g * m + i]);
          sum += (q - z) * s * float(h_B[j * k + l]);
        }
        h_ref[i + j * ldC] = alpha * sum + beta * float(h_C[i + j * ldC]);
      }

    float max_err = 0.0f;
    for (int i = 0; i < m * n; ++i)
      max_err = std::max(max_err, std::abs(float(h_result[i]) - h_ref[i]));

    float tol = float(k) * 0.05f;
    printf("%s (%dx%dx%d): max error %e (tol %.3f) -- %s\n",
           label, m, n, k, max_err, tol, max_err < tol ? "PASS" : "FAIL");
    return max_err < tol;
  };

  // Correctness tests (K must be a multiple of 128 = group_size)
  if (!run_test(128,  256,  128,  "Single tile"))  return 1;
  if (!run_test(256,  256,  128,  "Multi-M"))      return 1;
  if (!run_test(128,  256,  256,  "Multi-K"))      return 1;
  if (!run_test(256,  256,  256,  "Multi-MK"))     return 1;
  if (!run_test(128,  512,  128,  "Multi-N"))      return 1;
  if (!run_test(128,  256,  512,  "K=4"))          return 1;
  if (!run_test(512,  512,  512,  "Large"))        return 1;
  if (!run_test(1024, 1024, 1024, "1024^3"))       return 1;
  if (!run_test(2048, 1024, 512,  "2048x1024x512"))return 1;

  printf("\n");

  // Benchmark: sweep over log_swizzle values at key sizes
  printf("Benchmark (100 iterations each):\n");
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);

  for (int size : {4096, 8192}) {
    int m = size, n = size, k = size;
    int ldA = k, ldB = k, ldC = m;

    int total_tiles_bench = ((m + 127) / 128) * ((k + 63) / 64);
    thrust::device_vector<bf16_t>   d_A(m * k), d_B(n * k), d_C(m * n);
    thrust::device_vector<uint32_t> d_packed(total_tiles_bench * 1024);
    thrust::host_vector<bf16_t>     h_A(m * k), h_B(n * k);
    for (int i = 0; i < m * k; ++i) {
      float val = float(rand() % 16);
      uint16_t raw = 0x3F80u | (uint16_t(val) & 0xFu);
      h_A[i] = reinterpret_cast<bf16_t const&>(raw);
    }
    for (int i = 0; i < n * k; ++i) h_B[i] = static_cast<bf16_t>(2.0 * (rand() / double(RAND_MAX)) - 1.0);
    d_A = h_A; d_B = h_B;

    // Per-group scales / zeros for benchmark
    thrust::host_vector<bf16_t>     h_scales_bench;
    thrust::device_vector<uint32_t> d_sz_bench;
    (void)generate_and_pack_sz(m, k, h_scales_bench, d_sz_bench);

    split_a_pack(m, k, d_A.data().get(), ldA, d_packed.data().get());
    CUTE_CHECK_LAST();

    const int timing_iterations = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int log_sw : {0, 1, 2, 3}) {
      split_a_wgmma(m, n, k, alpha,
                    d_packed.data().get(), d_sz_bench.data().get(),
                    d_B.data().get(), ldB, beta, d_C.data().get(), ldC, log_sw);
      CUTE_CHECK_LAST();

      cudaEventRecord(start);
      for (int i = 0; i < timing_iterations; ++i) {
        split_a_wgmma(m, n, k, alpha,
                      d_packed.data().get(), d_sz_bench.data().get(),
                      d_B.data().get(), ldB, beta, d_C.data().get(), ldC, log_sw);
      }
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);

      float total_ms = 0.0f;
      cudaEventElapsedTime(&total_ms, start, stop);
      double avg_ms = total_ms / timing_iterations;
      double gflops = (2.0 * m * n * k) * 1e-9;
      printf("  %dx%dx%d log_swizzle=%d (sw=%d): %.1f GFLOP/s (%.4f ms)\n",
             m, n, k, log_sw, 1 << log_sw, gflops / (avg_ms * 1e-3), avg_ms);
    }

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

    float cublas_ms = 0.0f;
    cudaEventElapsedTime(&cublas_ms, start, stop);
    double cublas_avg = cublas_ms / timing_iterations;
    double gflops_ref = (2.0 * m * n * k) * 1e-9;
    printf("  %dx%dx%d cublas: %.1f GFLOP/s (%.4f ms)\n", m, n, k, gflops_ref / (cublas_avg * 1e-3), cublas_avg);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  cublasDestroy(cublas_handle);

  return 0;
}
