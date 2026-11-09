// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/norm/l2_norm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <type_traits>

#include "src/turbomind/utils/cuda_utils.h"

#include "src/turbomind/kernels/core/array_ops.h"

namespace turbomind {

template<class T>
__global__ void L2NormalizeQKKernel(T*      q,
                                    int64_t q_batch_stride,
                                    int64_t q_token_stride,
                                    T*      k,
                                    int64_t k_batch_stride,
                                    int64_t k_token_stride,
                                    int     token_num,
                                    int     hq,
                                    float   epsilon)
{
    constexpr int kHeadDim   = 128;
    const int     lane       = threadIdx.x;
    const int     work       = int(blockIdx.x);
    const int     batch      = work / (token_num * hq);
    const int     token_head = work % (token_num * hq);
    const int     token      = token_head / hq;
    const int     head       = token_head % hq;
    T* q_ptr = q + int64_t(batch) * q_batch_stride + int64_t(token) * q_token_stride + int64_t(head) * kHeadDim;
    T* k_ptr = k + int64_t(batch) * k_batch_stride + int64_t(token) * k_token_stride + int64_t(head) * kHeadDim;

    float q_values[4];
    float k_values[4];
    float q_sum = 0.f;
    float k_sum = 0.f;
    PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
        const int d = lane + i * WARP_SIZE;
        q_values[i] = static_cast<float>(q_ptr[d]);
        k_values[i] = static_cast<float>(k_ptr[d]);
        q_sum += q_values[i] * q_values[i];
        k_sum += k_values[i] * k_values[i];
    }
    PRAGMA_UNROLL
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        q_sum += __shfl_xor_sync(0xffffffff, q_sum, mask);
        k_sum += __shfl_xor_sync(0xffffffff, k_sum, mask);
    }
    const float q_inv = rsqrtf(q_sum + epsilon);
    const float k_inv = rsqrtf(k_sum + epsilon);
    PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
        const int d = lane + i * WARP_SIZE;
        q_ptr[d]    = T(q_values[i] * q_inv);
        k_ptr[d]    = T(k_values[i] * k_inv);
    }
}

void invokeL2NormalizeQK(core::Tensor& q, core::Tensor& k, float epsilon, cudaStream_t stream)
{
    constexpr int kWarpThreads = WARP_SIZE;
    const int     work         = static_cast<int>(q.shape(0) * q.shape(1) * q.shape(2));
    auto          invoke       = [&](auto t) {
        using T = decltype(t);
        L2NormalizeQKKernel<T><<<work, kWarpThreads, 0, stream>>>(q.data<T>(),
                                                                  q.stride(0),
                                                                  q.stride(1),
                                                                  k.data<T>(),
                                                                  k.stride(0),
                                                                  k.stride(1),
                                                                  static_cast<int>(q.shape(1)),
                                                                  static_cast<int>(q.shape(2)),
                                                                  epsilon);
    };
    TM_DISPATCH_PRIMARY_DTYPES(q.dtype(), invoke);
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace turbomind
