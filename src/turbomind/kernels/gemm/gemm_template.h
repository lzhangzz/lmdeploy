// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/gemm_universal.h"
#include "src/turbomind/kernels/gemm/impl_81616.h"
#include "src/turbomind/kernels/gemm/mainloop_sm80.h"
#include "src/turbomind/kernels/gemm/tile_iterator.h"

namespace turbomind::gemm {

template<int CTA_M, int CTA_N, int CTA_K>
struct CtaMap {

    __host__ static dim3 get_grid_shape(int m, int n, int k)
    {
        return dim3((m + CTA_M - 1) / CTA_M, ((n + CTA_N - 1) / CTA_N), 1);
    }

    __device__ int m_idx() const
    {
        return blockIdx.x;
    }
    __device__ int n_idx() const
    {
        return blockIdx.y;
    }
    __device__ int k_idx() const
    {
        return 0;
        // return threadIdx.z;
    }
};

template<class T>
void invoke(T* C, const T* A, const T* B, int m, int n, int k, cudaStream_t st)
{
    constexpr int CTA_M  = 128;
    constexpr int CTA_N  = 128;
    constexpr int CTA_K  = 32;
    constexpr int WARP_M = 64;
    constexpr int WARP_N = 64;
    constexpr int WARP_K = 32;
    using Impl           = Impl<MMA_81616, T, T, T, CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, WARP_K, 4>;
    using Kernel = GemmUniversal<void, Mainloop_sm80<Impl>, TileIterator<T, CTA_K>, CtaMap<CTA_M, CTA_N, CTA_K>>;

    auto grid = CtaMap<CTA_M, CTA_N, CTA_K>::get_grid_shape(m, n, k);
    // printf("grid = [%d %d %d]\n", (int)grid.x, (int)grid.y, (int)grid.z);
    auto block = Impl::WARP_CNT * WARP_SIZE;

    [[maybe_unused]] static const int _ = [] {
        Print(typename Impl::ThreadMapA{});
        Print(typename Impl::ThreadMapB{});
        printf("warp count: %d\n", Impl::WARP_CNT);
        return 0;
    }();

    static constexpr int kSmemSize = sizeof(typename Kernel::SharedStorage);
    if constexpr (kSmemSize > (48 << 10)) {
        cudaFuncSetAttribute(gemm_kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);
    }

    gemm_kernel<Kernel>
        <<<grid, block, kSmemSize, st>>>(typename Kernel::Param{A, B, C, m, n, k}, typename Kernel::CtaMap{});
}

}  // namespace turbomind::gemm