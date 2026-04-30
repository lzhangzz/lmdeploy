/***************************************************************************************************
 * Shared pack kernel and host function for split A loading (iteration 08).
 *
 * Packs operand A from GMMA register layout to uint4 format (4x smaller than BF16).
 * Each thread packs 8 BF16 values into 1 uint32 via pack_bf16_to_u4().
 *
 * Packed format per warpgroup: (THREAD=128, K_BLOCK=4) strides (1, 128) in uint32.
 * Total per tile: 2 warpgroups x 512 uint32 = 1024 uint32 = 4096 bytes (down from 8192).
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

// ================================================================================================
// Standalone pack/unpack functions (extracted from turbomind, no Array<T> dependency)
// ================================================================================================

// Pack 8 BF16 values into 1 uint32 containing 8 uint4 nibbles.
// Each input value must be in [0, 15].
__device__ uint32_t
pack_bf16_to_u4(const uint16_t v[8])
{
    uint32_t w0 = uint32_t(v[0] & 0xF)
                | (uint32_t(v[1] & 0xF) << 8)
                | (uint32_t(v[2] & 0xF) << 16)
                | (uint32_t(v[3] & 0xF) << 24);
    uint32_t w1 = uint32_t(v[4] & 0xF)
                | (uint32_t(v[5] & 0xF) << 8)
                | (uint32_t(v[6] & 0xF) << 16)
                | (uint32_t(v[7] & 0xF) << 24);
    w0 |= (w0 >> 12);
    w1 |= (w1 >> 12);
    return __byte_perm(w0, w1, 0x5140);
}

// Unpack 1 uint32 (8 uint4 nibbles) to 8 BF16 values via fast I2F.
// Subtracts implicit zero point 128 from each output value.
__device__ void
unpack_u4_to_bf16(uint32_t packed, nv_bfloat16 out[8])
{
    static constexpr uint32_t TEMPLATE = 0x43004300;  // bf162(128, 128)
    static constexpr uint32_t MASK     = 0x000f000f;
    static constexpr uint32_t immLut   = (0xf0 & 0xcc) | 0xaa;

    uint32_t* h = reinterpret_cast<uint32_t*>(out);
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;" : "=r"(h[0]) : "r"(packed),       "n"(MASK), "n"(TEMPLATE), "n"(immLut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;" : "=r"(h[1]) : "r"(packed >> 4),  "n"(MASK), "n"(TEMPLATE), "n"(immLut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;" : "=r"(h[2]) : "r"(packed >> 8),  "n"(MASK), "n"(TEMPLATE), "n"(immLut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;" : "=r"(h[3]) : "r"(packed >> 12), "n"(MASK), "n"(TEMPLATE), "n"(immLut));

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[i] -= nv_bfloat16(128.f);
    }
}

// SharedStorage for pack kernel: A smem (1 stage) + TMA mbarrier
template <class ElementA, class SmemLayoutA>
struct PackSharedStorage
{
  alignas(128) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  alignas(16)  uint64_t mbarrier;
};

// ================================================================================================
// Pack kernel: transforms A from gmem tensor layout to uint4-packed GMMA register layout
// ================================================================================================
template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TiledMma>
__global__ static
__launch_bounds__(256, 1)
void
split_a_pack_device(ProblemShape shape_MK, CtaTiler cta_tiler,
                    TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
                    uint32_t* packed_A, TiledMma mma,
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

  // ---- TiledMMA and S2R copy setup ----
  ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCrA = thr_mma.make_fragment_A(tCsA(_,_,_,Int<0>{}));

  auto smem_tiled_copy_A = make_tiled_copy_A(
      Copy_Atom<AutoVectorizingCopy, bf16_t>{}, mma);
  auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(threadIdx.x);
  Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
  Tensor tCsA_copy_view = smem_thr_copy_A.partition_S(sA);

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

    // ---- TMA load using raw mbarrier ----
    if (threadIdx.x == 0) {
      smem.mbarrier = 0;
      cute::initialize_barrier(smem.mbarrier, 1);
      cute::set_barrier_transaction_bytes(smem.mbarrier, tma_transaction_bytes);

      auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                         group_modes<0,2>(sA(_,_,0)),
                                         group_modes<0,2>(gA));
      copy(tma_a.with(smem.mbarrier), tAgA, tAsA);
    }
    __syncthreads();
    cute::wait_barrier(smem.mbarrier, 0);

    // ---- S2R copy: swizzled smem -> GMMA register layout ----
    copy(smem_tiled_copy_A, tCsA_copy_view(_,_,_,0), tCrA_copy_view(_,_,_));

    // ---- Write registers to uint4-packed gmem buffer ----
    // Each thread packs 8 BF16 values into 1 uint32 per k_block.
    // Packed gmem per warpgroup: (THREAD=128, K_BLOCK=4) strides (1, 128) in uint32.
    int wg_id = threadIdx.x / 128;
    int local_tid = threadIdx.x % 128;

    // tCrA has shape (MMA_M=8, MMA_K=1, K_BLOCK=4), stride in K_BLOCK = size<0>(tCrA) = 8
    constexpr int regs_per_kb = 8;

    CUTE_UNROLL
    for (int kb = 0; kb < 4; ++kb) {
      uint16_t vals[8];
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        vals[j] = reinterpret_cast<uint16_t const&>(tCrA(j, 0, kb));
      }
      uint32_t packed = pack_bf16_to_u4(vals);

      uint32_t* dst = packed_A + linear_idx * 1024 + wg_id * 512
                    + local_tid + kb * 128;
      *dst = packed;
    }

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
             uint32_t* packed_A,
             cudaStream_t stream = 0)
{
  using namespace cute;

  auto M = int(m);
  auto K = int(k);
  auto shape_MK = make_shape(M, K);

  auto bM = Int<128>{};
  auto bK = Int<64>{};
  auto cta_tiler = make_shape(bM, bK);

  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16_t>{}, make_shape(bM, bK, Int<1>{}));

  Tensor tA = make_tensor(A, make_shape(M, K), make_stride(ldA, Int<1>{}));
  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, tA, sA(_,_,0), make_shape(bM, bK));

  TiledMMA mma = make_tiled_mma(
      SM90_64x256x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::K>{},
      Layout<Shape<_2, _1>>{});

  static_assert(decltype(size(mma))::value == 256, "Expected 256 threads");

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
