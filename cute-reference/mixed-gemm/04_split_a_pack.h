/***************************************************************************************************
 * Shared pack kernel and host function for split A loading (iteration 04).
 *
 * Reduces packing unit to (64, 16) = one WGMMA instruction's A operand.
 * Output is organized as (m_wg, k_block) units within each (128, 64) tile.
 * Per unit: (REG=8, THREAD=128) strides (1, 8), 1024 bf16.
 * Each warpgroup's 4 units are contiguous (4096 bf16).
 * Included by 04_bf16_gemm_sm90_split_a_pack.cu and 04_bf16_gemm_sm90_split_a_wgmma.cu.
 **************************************************************************************************/
#pragma once

#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>
#include "cutlass/cluster_launch.hpp"
#include "cutlass/device_kernel.h"

using bf16_t = cute::bfloat16_t;

// SharedStorage for pack kernel: A smem (1 stage) + TMA mbarrier
template <class ElementA, class SmemLayoutA>
struct PackSharedStorage
{
  alignas(128) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  alignas(16)  uint64_t mbarrier;
};

// ================================================================================================
// Pack kernel: transforms A from gmem tensor layout to per-tile GMMA register layout
// ================================================================================================
template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TiledMma>
__global__ static
__launch_bounds__(256, 1)
void
split_a_pack_device(ProblemShape shape_MK, CtaTiler cta_tiler,
                    TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
                    TA* packed_A, TiledMma mma,
                    int total_tiles)
{
  using namespace cute;

  CUTE_STATIC_ASSERT_V(rank(shape_MK) == Int<2>{});
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<2>{});
  static_assert(is_static<SmemLayoutA>::value);

  auto [M, K] = shape_MK;

  // ---- Global memory tensor ----
  Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));                    // (M, K)

  // ---- Shared memory ----
  extern __shared__ char shared_memory[];
  using SharedStorage = PackSharedStorage<TA, SmemLayoutA>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});  // (BLK_M, BLK_K, 1)

  // ---- Tile counts ----
  constexpr int bK = size<1>(CtaTiler{});
  int k_tile_count = size(ceil_div(K, bK));

  // ---- TiledMMA and S2R copy setup (same as sample 13 consumer) ----
  ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);                                  // (MMA, MMA_M, MMA_K, PIPE)
  Tensor tCrA = thr_mma.make_fragment_A(tCsA(_,_,_,Int<0>{}));            // (MMA, MMA_M, MMA_K)

  auto smem_tiled_copy_A = make_tiled_copy_A(
      Copy_Atom<AutoVectorizingCopy, bf16_t>{}, mma);
  auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(threadIdx.x);
  Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);                 // (CPY, CPY_M, CPY_K)
  Tensor tCsA_copy_view = smem_thr_copy_A.partition_S(sA);                // (CPY, CPY_M, CPY_K, PIPE)

  constexpr int regs_per_thread = size(tCrA);

  // TMA transaction bytes (one stage of A smem)
  constexpr int tma_transaction_bytes = sizeof(TA) * cute::cosize_v<SmemLayoutA>;

  // ---- Grid-stride loop over all (M, K) tiles ----
  uint64_t linear_idx = blockIdx.x;
  uint64_t grid_size = gridDim.x;

  while (linear_idx < total_tiles) {
    int tile_m = linear_idx / k_tile_count;
    int tile_k = linear_idx % k_tile_count;

    // Compute gmem tensor for this tile
    Tensor gA = local_tile(mA, cta_tiler, make_coord(tile_m, tile_k));

    // ---- TMA load using raw mbarrier (pattern from tma_load_testbed.hpp) ----
    if (threadIdx.x == 0) {
      smem.mbarrier = 0;
      cute::initialize_barrier(smem.mbarrier, 1);
      cute::set_barrier_transaction_bytes(smem.mbarrier, tma_transaction_bytes);

      // group_modes<0,2> flattens 2D (bM, bK) to 1D ((bM, bK)) so the full tile
      // is treated as one TMA transfer (same pattern as sample 13's tma_partition calls)
      auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                         group_modes<0,2>(sA(_,_,0)),
                                         group_modes<0,2>(gA));
      copy(tma_a.with(smem.mbarrier), tAgA, tAsA);
    }
    __syncthreads();
    cute::wait_barrier(smem.mbarrier, 0);

    // ---- S2R copy: swizzled smem -> GMMA register layout ----
    copy(smem_tiled_copy_A, tCsA_copy_view(_,_,_,0), tCrA_copy_view(_,_,_));

    // ---- Write registers to packed gmem buffer (per-warpgroup 64x16 units) ----
    // Each warpgroup writes its own contiguous 4096 bf16 region (4 units of 1024 bf16).
    // Per unit: (REG=8, THREAD=128) strides (1, 8).
    // Per warpgroup: (REG=8, THREAD=128, K_BLOCK=4) strides (1, 8, 1024).
    int wg_id = threadIdx.x / 128;
    int local_tid = threadIdx.x % 128;

    Tensor gPacked_wg = make_tensor(
        make_gmem_ptr(packed_A + linear_idx * 8192 + wg_id * 4096),
        make_shape(Int<8>{}, Int<128>{}, Int<4>{}));   // strides (1, 8, 1024)
    Tensor gP = gPacked_wg(_, local_tid, _);            // (8, 4) stride (1, 1024)
    // tCrA flat layout: j + k*8, so (REG, K_BLOCK) = (8, 4) stride (1, 8) matches
    Tensor rA = make_tensor(tCrA.data(), make_shape(Int<8>{}, Int<4>{}));
    copy(AutoVectorizingCopy{}, rA, gP);

    __syncthreads();
    linear_idx += grid_size;
  }
}

// ================================================================================================
// Host function for pack kernel
// ================================================================================================
inline void
split_a_pack(int m, int k,
             bf16_t const* A, int ldA,
             bf16_t* packed_A,
             cudaStream_t stream = 0)
{
  using namespace cute;

  auto M = int(m);
  auto K = int(k);
  auto shape_MK = make_shape(M, K);

  // CTA tile size (matches sample 13's A tile dimensions)
  auto bM = Int<128>{};
  auto bK = Int<64>{};
  auto cta_tiler = make_shape(bM, bK);

  // Smem layout for A (1 stage, no pipelining)
  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16_t>{}, make_shape(bM, bK, Int<1>{}));

  // TMA load atom
  Tensor tA = make_tensor(A, make_shape(M, K), make_stride(ldA, Int<1>{}));
  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, tA, sA(_,_,0), make_shape(bM, bK));

  // TiledMMA (same as sample 13 -- required for correct register layout)
  TiledMMA mma = make_tiled_mma(
      SM90_64x256x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::K>{},
      Layout<Shape<_2, _1>>{});

  static_assert(decltype(size(mma))::value == 256, "Expected 256 threads");

  // Grid
  int total_tiles = size(ceil_div(M, bM)) * size(ceil_div(K, bK));
  int num_SMs = 0;
  CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0));

  dim3 dimBlock(256);
  dim3 dimCluster(1, 1, 1);
  dim3 dimGrid(std::min(num_SMs, total_tiles));

  int smem_size = int(sizeof(PackSharedStorage<bf16_t, decltype(sA)>));

  auto* kernel_ptr = &split_a_pack_device<
      decltype(shape_MK), decltype(cta_tiler),
      bf16_t, decltype(sA), decltype(tmaA),
      decltype(mma)>;

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
      shape_MK, cta_tiler,
      A, tmaA,
      packed_A, mma,
      total_tiles);

  CUTE_CHECK_LAST();
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed at pack kernel launch" << std::endl;
  }
}
