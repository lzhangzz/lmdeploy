// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include <cub/block/block_scan.cuh>

#include "src/turbomind/kernels/batch_kernels.h"
#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

template<typename T, int vec_size>
__global__ void castFloat2D(const T* input, float* output, int channels)
{
    const int vi = blockIdx.x * blockDim.x + threadIdx.x;
    const int bi = blockIdx.y;
    input += (size_t)bi * channels;
    output += (size_t)bi * channels;

    const int step = gridDim.x * blockDim.x * vec_size;

    for (int i = vi * vec_size; i < channels; i += step) {
        Array<T, vec_size> src;

        if constexpr (sizeof(src) >= sizeof(uint)) {
            Load(src, input + i);
        }
        else {
            PRAGMA_UNROLL
            for (int j = 0; j < vec_size; ++j) {
                src[j] = input[i + j];
            }
        }

        auto dst = cast<float>(src);

        // store
        Store(output + i, dst);
    }
}

void invokeCastFloat2D(const core::Tensor& src, core::Tensor& dst, cudaStream_t stream)
{
    TM_CHECK(src.is_contiguous());
    TM_CHECK(dst.is_contiguous());
    TM_CHECK(src.shape() == dst.shape());

    auto batch_size = src.shape(0);
    auto channels   = src.shape(1);

    auto invoke = [&](auto t, auto vec_size) {
        using T                      = decltype(t);
        constexpr int threads        = 256;
        const int     blocks_per_tok = (channels + threads * vec_size - 1) / (threads * vec_size);
        const dim3    blocks(blocks_per_tok, batch_size);
        castFloat2D<T, vec_size.value><<<blocks, threads, 0, stream>>>(  //
            src.data<T>(),
            dst.data<float>(),
            channels);
    };

    auto dispatch_t = [&](auto vec_size) {
        switch (src.dtype()) {
            case kFloat32:
                invoke(float{}, vec_size);
                break;
            case kFloat16:
                invoke(half{}, vec_size);
                break;
#ifdef ENABLE_BF16
            case kBfloat16:
                invoke(__nv_bfloat16{}, vec_size);
                break;
#endif
            default:
                TM_UNREACHABLE;
        }
    };

    if (channels % 4 == 0) {
        dispatch_t(std::integral_constant<int, 4>{});
    }
    else if (channels % 2 == 0) {
        dispatch_t(std::integral_constant<int, 2>{});
    }
    else {
        dispatch_t(std::integral_constant<int, 1>{});
    }
    TM_CUDA_CHECK(cudaGetLastError());
}

template<class T>
__global__ void CollectHiddenStates_Kernel(const T* src, const int* idxs, T* dst, int dim)
{
    const int bi = blockIdx.x;
    const int ti = idxs[bi];

    if (ti < 0) {
        return;
    }

    src += ti * dim;
    dst += bi * dim;

    for (int di = threadIdx.x; di < dim; di += blockDim.x) {
        dst[di] = src[di];
    }
}

void CollectHiddenStates(const Tensor& src, const Buffer_<int>& idxs, Ref<Tensor> dst, cudaStream_t st)
{
    if (idxs.size() == 0) {
        return;
    }

    const auto stride = byte_size(src.dtype(), src.stride(0));

    auto invoke = [&](auto t) {
        using T           = decltype(t);
        const int dim     = stride / sizeof(T);
        const int threads = round_up(min(dim, 1024), WARP_SIZE);
        const int blocks  = idxs.size();
        CollectHiddenStates_Kernel<<<blocks, threads, 0, st>>>(
            (const T*)src.raw_data(), idxs.data(), (T*)dst.get().raw_data(), dim);
    };

    if (stride % sizeof(uint4) == 0) {
        invoke(uint4{});
    }
    else if (stride % sizeof(uint2) == 0) {
        invoke(uint2{});
    }
    else if (stride % sizeof(uint1) == 0) {
        invoke(uint1{});
    }
    else if (stride % sizeof(ushort) == 0) {
        invoke(ushort{});
    }
    else {
        __builtin_unreachable();
    }
}

template<int BLOCK_DIM, int MAX_COUNT>
__global__ void
BatchPrefixSumKernel(Array<const int*, MAX_COUNT> srcs, Array<int, MAX_COUNT> ns, Array<int*, MAX_COUNT> dsts)
{
    const int  bi  = blockIdx.x;
    const int* src = srcs[bi];
    int*       dst = dsts[bi];
    const int  n   = ns[bi];

    using BlockScan = cub::BlockScan<int, BLOCK_DIM>;

    __shared__ typename BlockScan::TempStorage temp_storage;

    int prefix{};
    for (int i = threadIdx.x; i < round_up(n, BLOCK_DIM); i += BLOCK_DIM) {
        if (i >= BLOCK_DIM) {
            __syncthreads();
        }
        int data = i < n ? src[i] : 0;
        int sum{};
        BlockScan{temp_storage}.ExclusiveSum(data, data, sum);
        if (i < n) {
            dst[i] = prefix + data;
        }
        prefix += sum;
    }

    if (threadIdx.x == 0) {
        dst[n] = prefix;
    }
}

void BatchPrefixSum(const int** srcs, const int* ns, int** dsts, int count, cudaStream_t st)
{
    constexpr int max_count = 1;

    Array<const int*, max_count> p_srcs{};
    Array<int*, max_count>       p_dsts{};
    Array<int, max_count>        p_ns{};

    for (int i = 0; i < count; ++i) {
        p_srcs[i] = srcs[i];
        p_dsts[i] = dsts[i];
        p_ns[i]   = ns[i];
    }

    TM_CHECK_LE(count, max_count);

    constexpr int block = 256;
    const int     grid  = count;

    BatchPrefixSumKernel<block><<<grid, block, 0, st>>>(p_srcs, p_ns, p_dsts);
    TM_CUDA_CHECK(cudaGetLastError());
}

__global__ void AppendOneTokenAndAdvanceSequenceKernel(int* const* token_ids_ptrs,
                                                       const int*  selected_tokens,
                                                       int*        sequence_length,
                                                       int         batch_size)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch_size) {
        const int position          = sequence_length[b];
        token_ids_ptrs[b][position] = selected_tokens[b];
        sequence_length[b]          = position + 1;
    }
}

void invokeAppendOneTokenAndAdvanceSequence(
    int* const* token_ids_ptrs, const int* selected_tokens, int* sequence_length, int batch_size, cudaStream_t stream)
{
    if (batch_size == 0) {
        return;
    }

    constexpr int block_size = 128;
    const int     grid_size  = cdiv(batch_size, block_size);

    AppendOneTokenAndAdvanceSequenceKernel<<<grid_size, block_size, 0, stream>>>(
        token_ids_ptrs, selected_tokens, sequence_length, batch_size);
}

namespace {

__global__ void ClearTokenMaskKernel(bool*                      token_mask,
                                     const bool*                finished,
                                     const int*                 q_offsets,
                                     size_t                     rank_stride,
                                     Array<int, kMaxAttnDPSize> token_base)
{
    const int r = blockIdx.y;
    const int s = blockIdx.x;

    // Select this rank's metadata block (no-op when attn_dp == 1: rank_stride is 0).
    finished  = (const bool*)((const char*)finished + r * rank_stride);
    q_offsets = (const int*)((const char*)q_offsets + r * rank_stride);

    if (!finished[s]) {
        return;
    }
    const int begin = __ldg(q_offsets + s);
    const int end   = __ldg(q_offsets + s + 1);
    bool*     out   = token_mask + token_base[r];
    for (int t = begin + threadIdx.x; t < end; t += blockDim.x) {
        out[t] = false;
    }
}

}  // namespace

void invokeBuildTokenMask(bool*        token_mask,
                          const bool*  finished,
                          const int*   q_offsets,
                          size_t       rank_stride,
                          const int*   token_base,
                          int          attn_dp_size,
                          int          batch_size,
                          int          global_token_num,
                          cudaStream_t stream)
{
    if (global_token_num == 0) {
        return;
    }
    // Every forward token is valid by default; invalidate the token ranges owned
    // by finished sequences (`q_offsets` maps slot -> [begin, end) token range).
    TM_CHECK_LE(attn_dp_size, kMaxAttnDPSize);
    Array<int, kMaxAttnDPSize> base{};
    std::copy_n(token_base, attn_dp_size, &base[0]);
    TM_CUDA_CHECK(cudaMemsetAsync(token_mask, 1, global_token_num, stream));
    ClearTokenMaskKernel<<<dim3(batch_size, attn_dp_size), 128, 0, stream>>>(
        token_mask, finished, q_offsets, rank_stride, base);
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace turbomind
