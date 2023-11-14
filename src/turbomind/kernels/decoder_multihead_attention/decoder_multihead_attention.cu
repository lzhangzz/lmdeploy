// Copyright (c) OpenMMLab. All rights reserved.

#include "decoder_multihead_attention_template.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <iostream>

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

template<typename T, typename Tkv, int HeadDim, int HeadPerCta>
void invokeDecoderMultiheadAttention(const DecoderMultiHeadAttentionParams<T>& params)
{
    auto invoke = [&](auto* type) {
        using Attn = std::remove_reference_t<decltype(*type)>;

        static const size_t kDynSmemSize = Attn::GetDynamicSmemSize();

        [[maybe_unused]] static const int _ = [&] {
            std::cout << "GmemMap:\n";
            Print(typename Attn::GmemMap{});
            std::cout << "\nSmemMap:\n";
            Print(typename Attn::SmemMap{});
            std::cout << "\nDynamic smem size: " << kDynSmemSize << "\n";
            return 0;
        }();

        const int slice_count = (params.max_seq_len + Attn::kSliceLen - 1) / Attn::kSliceLen;
        const int max_split_k = std::min(params.max_split_k, std::max(1, slice_count));

        dim3 block(Attn::kWarpCount * WARP_SIZE);
        dim3 grid(params.num_heads / HeadPerCta, params.batch_size, max_split_k);

        // if (params.layer_offset == 0) {
        //     std::cout << "max_split_k' = " << max_split_k << ", arch = " << params.arch << "\n";
        // }

        cudaFuncSetAttribute(
            decoder_multihead_attention<Attn>, cudaFuncAttributeMaxDynamicSharedMemorySize, kDynSmemSize);

        decoder_multihead_attention<Attn><<<grid, block, kDynSmemSize, params.stream>>>(params);

        // cudaStreamSynchronize(params.stream);

        // if (auto err = cudaGetLastError(); err != cudaSuccess) {
        //     std::cout << cudaGetErrorString(err) << "\n";
        //     std::abort();
        // }

        // if (max_split_k > 1) {
        //     dim3 grid(params.num_heads, params.batch_size);
        //     decoder_multihead_attention_reduce<Attn><<<grid, block, 0, params.stream>>>(params);
        // }
    };

    if (params.arch >= 80) {
        using Type = DecoderMultiHeadAttentionKernel<T, Tkv, HeadPerCta, HeadDim, 32, HeadDim, 512, 5, true>;
        invoke((Type*)0);
    }
    else {
        // DecoderMultiHeadAttentionKernel<T, Tkv, HeadPerCta, HeadDim, 32, HeadDim, 2048, 3>; // 34k
        // DecoderMultiHeadAttentionKernel<T, Tkv, HeadPerCta, HeadDim, 64, HeadDim, 2048, 3>;  // 34k

        // using Type = DecoderMultiHeadAttentionKernel<T, Tkv, HeadPerCta, HeadDim, 64, HeadDim, 1024, 3, true>;
        // invoke((Type*)0);
    }
}

template<typename T>
void DispatchDecoderMultiheadAttention(const DecoderMultiHeadAttentionParams<T>& params)
{
    static constexpr int HeadDim = 128;

    FT_CHECK(params.size_per_head == HeadDim);

    if constexpr (std::is_same_v<T, half>) {
        if (params.quant_policy & QuantPolicy::kCacheKVInt8) {
            // invokeDecoderMultiheadAttention<T, int8_t, HeadDim, 1>(params);
            return;
        }

        int group_size = params.num_heads / params.num_kv_heads;

        // invokeDecoderMultiheadAttention<T, T, HeadDim, 8>(params);
        if (0) {}
        // else if (group_size % 8 == 0) {
        //     invokeDecoderMultiheadAttention<T, T, HeadDim, 8>(params);
        // }
        // else if (group_size % 4 == 0) {
        // invokeDecoderMultiheadAttention<T, T, HeadDim, 4>(params);
        // }
        // else if (group_size % 2 == 0) {
        //     invokeDecoderMultiheadAttention<T, T, HeadDim, 2>(params);
        // }
        else {
            invokeDecoderMultiheadAttention<T, T, HeadDim, 1>(params);
        }
    }
}

template void DispatchDecoderMultiheadAttention(const DecoderMultiHeadAttentionParams<half>& params);
template void DispatchDecoderMultiheadAttention(const DecoderMultiHeadAttentionParams<float>& params);

}  // namespace turbomind
