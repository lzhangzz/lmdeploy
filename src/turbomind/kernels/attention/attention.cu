// Copyright (c) OpenMMLab. All rights reserved.

#include "attention_template.h"
#include "src/turbomind/kernels/attention/impl_sm70.h"
#include "src/turbomind/kernels/attention/impl_sm80.h"
#include "src/turbomind/kernels/attention/mainloop_sm70.h"
#include "src/turbomind/kernels/attention/mainloop_sm80.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <iostream>
#include <type_traits>

namespace turbomind {

namespace {

template<class TMap>
void Print(TMap)
{
    std::cout << "     warps: " << TMap::kWarpCount << "\n";
    std::cout << "     shape: (" << TMap::kDimC << ", " << TMap::kDimS << ")\n";
    std::cout << "    access: (" << TMap::kAccessC << ", " << 1 << ")\n";
    std::cout << "warpThread: (" << TMap::kWarpThreadC << ", " << TMap::kWarpThreadS << ")\n";
    std::cout << "warpAccess: (" << TMap::kWarpAccessC << ", " << TMap::kWarpAccessS << ")\n";
    std::cout << "  warpIter: (" << TMap::kWarpIterC << ", " << TMap::kWarpIterS << ")\n";
    std::cout << "      warp: (" << TMap::kWarpC << ", " << TMap::kWarpS << ")\n";
    std::cout << "      iter: (" << TMap::kIterC << ", " << TMap::kIterS << ")\n";
    std::cout << " footprint: (" << TMap::kFootprintC << ", " << TMap::kFootprintS << ")\n";
    std::cout << "     delta: (" << TMap::kDeltaC << ", " << TMap::kDeltaS << ")\n";
}

}  // namespace

template<typename T, typename Tkv, int HeadDim>
void invokeAttention(const AttentionParams<T>& params)
{
    auto invoke = [&](auto* type) {
        using Kernel = std::remove_reference_t<decltype(*type)>;

        static const size_t kDynSmemSize = sizeof(typename Kernel::SharedStorage);

        [[maybe_unused]] static const int _ = [&] {
            std::cout << "GmemMap:\n";
            Print(typename Kernel::Impl::ThreadMapKV{});
            // std::cout << "\nSmemMap:\n";
            // Print(typename Attn::SmemMap{});
            std::cout << "\nDynamic smem size: " << kDynSmemSize << "\n";
            return 0;
        }();

        constexpr int CTA_Q = Kernel::CTA_Q;

        // const int slice_count = (params.max_seq_len + Attn::kSliceLen - 1) / Attn::kSliceLen;
        // const int max_split_k = std::min(params.max_split_k, std::max(1, slice_count));

        const int max_q_tile = (params.max_input_len + CTA_Q - 1) / CTA_Q;

        dim3 block(Kernel::kWarpCount * WARP_SIZE);
        // dim3 grid(max_q_tile, params.num_heads, params.batch_size);
        dim3 grid(max_q_tile, params.batch_size, params.num_heads);

        std::cout << "(" << grid.x << " " << grid.y << " " << grid.z << ") " << block.x << "\n";

        auto err =
            cudaFuncSetAttribute(attention_kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, kDynSmemSize);
        if (err) {
            std::cout << cudaGetErrorString(err) << "\n";
            std::abort();
        }

        attention_kernel<Kernel><<<grid, block, kDynSmemSize, params.stream>>>(params);

        if (auto err = cudaGetLastError(); err != cudaSuccess) {
            std::cout << cudaGetErrorString(err) << "\n";
            std::abort();
        }
    };

    if (false && params.arch >= 80) {
        using Impl     = attention::Impl<attention::Sm80_16816, half, half, 64, 64, 16, 64, 128>;
        using Mainloop = attention::Mainloop<attention::Sm80_CpAsync, Impl>;
        using Kernel   = Attention<Mainloop, std::integral_constant<int, 64>>;
        invoke((Kernel*)0);
    }
    else if (true || params.arch == 70) {
        using Impl     = attention::Impl<attention::Sm70_884, half, half, 128, 64, 16, 64, 128>;
        using Mainloop = attention::Mainloop<attention::Sm70_Ldg, Impl>;
        using Kernel   = Attention<Mainloop, std::integral_constant<int, 64>>;
        invoke((Kernel*)0);
    }
}

template<typename T>
void dispatchAttention(const AttentionParams<T>& params)
{
    static constexpr int HeadDim = 128;

    FT_CHECK(params.size_per_head == HeadDim);

    if constexpr (std::is_same_v<T, half>) {
        invokeAttention<T, T, HeadDim>(params);
    }
}

template void dispatchAttention(const AttentionParams<half>& params);
template void dispatchAttention(const AttentionParams<float>& params);

template<class T>
void invokeProcessKV(const AttentionParams<T>& params)
{
    constexpr int WARPS = 4;
    constexpr int DIMS  = 128;
    constexpr int CTA_Q = 64;
    using Tkv           = T;

    int  block = WARPS * WARP_SIZE;
    dim3 grid((params.max_input_len + CTA_Q - 1) / CTA_Q, params.num_kv_heads, params.batch_size);

    ProcessKV<T, Tkv, CTA_Q, DIMS, WARPS><<<grid, block, 0, params.stream>>>(params);
}

template void invokeProcessKV(const AttentionParams<half>& params);

}  // namespace turbomind
