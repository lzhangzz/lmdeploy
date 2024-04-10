// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/impl_81616.h"
#include "src/turbomind/kernels/gemm/tile_iterator.h"
#include "src/turbomind/kernels/gemm/transcript.h"
#include "src/turbomind/kernels/gemm/transcript_impl.h"

namespace turbomind::gemm {

template<class T>
void transcript(T* dst, const T* src, int n, int k, cudaStream_t st)
{
    constexpr int CTA_M  = 128;
    constexpr int CTA_N  = 128;
    constexpr int CTA_K  = 32;
    constexpr int WARP_M = 64;
    constexpr int WARP_N = 64;
    constexpr int WARP_K = 32;

    using Gemm   = Impl<MMA_81616, T, T, T, CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, WARP_K, 4, 0>;
    using Kernel = Transcript<void, Gemm, TileIterator<T, CTA_M, CTA_N, CTA_K, 0>, CtaSwizzleMap<0>>;

    static constexpr int kSmemSize = sizeof(typename Kernel::SharedStorage);
    if constexpr (kSmemSize > (48 << 10)) {
        cudaFuncSetAttribute(transcript_kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);
    }

    using Map = typename Kernel::CtaMap;

    auto tiles = Map::get_tiled_shape(CTA_M, n, k, CTA_M, CTA_N, 1);
    auto grid  = Map::get_grid_shape(tiles);
    auto block = Gemm::WARP_CNT * WARP_SIZE;

    transcript_kernel<Kernel><<<grid, block, kSmemSize, st>>>({nullptr, src, dst, CTA_M, n, k});
}

template void transcript(half* dst, const half* src, int n, int k, cudaStream_t st);

}  // namespace turbomind::gemm