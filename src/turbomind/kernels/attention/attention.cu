// Copyright (c) OpenMMLab. All rights reserved.

#include "attention_template.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <iostream>

namespace turbomind {

namespace {

template<typename MHAType>
bool Print(size_t dynamic_smem_size)
{
    using MapKv = typename MHAType::MapKv;

    std::cout << "     warps: " << MapKv::kWarpCount << "\n";
    std::cout << "     shape: (" << MapKv::kC << ", " << MapKv::kS << ")\n";
    std::cout << "    access: (" << MapKv::kAccessC << ", " << 1 << ")\n";
    std::cout << "warpThread: (" << MapKv::kWarpThreadC << ", " << MapKv::kWarpThreadS << ")\n";
    std::cout << "warpAccess: (" << MapKv::kWarpAccessC << ", " << MapKv::kWarpAccessS << ")\n";
    std::cout << "  warpIter: (" << MapKv::kWarpIterC << ", " << MapKv::kWarpIterS << ")\n";
    std::cout << "      warp: (" << MapKv::kWarpC << ", " << MapKv::kWarpS << ")\n";
    std::cout << "      iter: (" << MapKv::kIterC << ", " << MapKv::kIterS << ")\n";
    std::cout << " footprint: (" << MapKv::kFootprintC << ", " << MapKv::kFootprintS << ")\n";
    std::cout << "     delta: (" << MapKv::kDeltaC << ", " << MapKv::kDeltaS << ")\n";
    std::cout << "dynamic smem size: " << dynamic_smem_size << "\n";

    return true;
}

}  // namespace

template<typename T, typename Tkv, int HeadDim, int HeadPerCta>
void invokedAttention(const AttentionParams<T>& params)
{
    auto invoke = [&](auto* type) {
        using Attn = std::remove_reference_t<decltype(*type)>;

        static const size_t kDynSmemSize = Attn::GetDynamicSmemSize();

        [[maybe_unused]] static const bool _ = Print<Attn>(kDynSmemSize);

        const int slice_count = (params.max_seq_len + Attn::kSliceLen - 1) / Attn::kSliceLen;
        const int max_split_k = std::min(params.max_split_k, std::max(1, slice_count));

        dim3 block(Attn::kWarpCount * WARP_SIZE);
        dim3 grid(params.num_heads / HeadPerCta, params.batch_size, max_split_k);

        // if (params.layer_offset == 0) {
        //     std::cout << "max_split_k' = " << max_split_k << ", arch = " << params.arch << "\n";
        // }

        cudaFuncSetAttribute(attention<Attn>, cudaFuncAttributeMaxDynamicSharedMemorySize, kDynSmemSize);

        attention<Attn><<<grid, block, kDynSmemSize, params.stream>>>(params);

        if (max_split_k > 1) {
            dim3 grid(params.num_heads, params.batch_size);
            attention_reduce<Attn><<<grid, block, 0, params.stream>>>(params);
        }
    };

    if (params.arch >= 80) {
        using Type = Attention<T, Tkv, HeadPerCta, HeadDim, 32, HeadDim, 1024, 5, true>;
        invoke((Type*)0);
    }
}

template<typename T>
void DispatchAttention(const AttentionParams<T>& params)
{
    static constexpr int HeadDim = 128;

    FT_CHECK(params.size_per_head == HeadDim);

    if constexpr (std::is_same_v<T, half>) {
        invokedAttention<T, T, HeadDim, 1>(params);
    }
}

template void DispatchAttention(const AttentionParams<half>& params);
template void DispatchAttention(const AttentionParams<float>& params);

template<typename T>
void invokeApplyRotaryEmbedding(
    T* k_cache, int seq_len, int head_num, float rotary_embedding_base, int batch_size, cudaStream_t st)
{
    using Type = Attention<T, T, 1, 128, 32, 128, 1024, 5, true>;
    dim3 block(Type::kWarpCount * WARP_SIZE);
    dim3 grid(head_num, batch_size);

    apply_rotary_embedding<Type><<<grid, block, 0, st>>>(k_cache, seq_len, head_num, rotary_embedding_base);
}

template void invokeApplyRotaryEmbedding(half*, int, int, float, int, cudaStream_t st);

}  // namespace turbomind
