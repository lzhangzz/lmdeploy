// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/linear_attn/gdn_gating.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <type_traits>

#include "src/turbomind/utils/cuda_utils.h"

#include "src/turbomind/kernels/core/array_ops.h"

namespace turbomind {

template<class T>
__global__ void ComputeBetaGKernel(float*   beta,
                                   float*   g,
                                   int64_t  gate_token_stride,
                                   const T* b,
                                   int64_t  b_token_stride,
                                   const T* a,
                                   int64_t  a_token_stride,
                                   const T* A_log,
                                   const T* dt_bias,
                                   int      token_num,
                                   int      hv)
{
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = int64_t(token_num) * gate_token_stride;
    if (index >= total) {
        return;
    }
    const int token = static_cast<int>(index / gate_token_stride);
    const int head  = static_cast<int>(index % gate_token_stride);
    if (head >= hv) {
        beta[index] = 0.f;
        g[index]    = 0.f;
        return;
    }
    const float b_value  = static_cast<float>(b[token * b_token_stride + head]);
    const float a_value  = static_cast<float>(a[token * a_token_stride + head]);
    beta[index]          = 1.f / (1.f + expf(-b_value));
    const float x        = a_value + static_cast<float>(dt_bias[head]);
    const float softplus = x > 20.f ? x : log1pf(expf(x));
    g[index]             = -expf(static_cast<float>(A_log[head])) * softplus;
}

void ComputeBetaG(core::Tensor&       beta,
                  core::Tensor&       g,
                  const core::Tensor& b,
                  const core::Tensor& a,
                  const core::Tensor& A_log,
                  const core::Tensor& dt_bias,
                  cudaStream_t        stream)
{
    const int     token_num     = static_cast<int>(b.shape(0));
    const int     hv            = static_cast<int>(A_log.size());
    constexpr int kBlockThreads = 256;
    const int64_t gate_capacity = int64_t(token_num) * beta.stride(1);
    const int     gate_blocks   = static_cast<int>((gate_capacity + kBlockThreads - 1) / kBlockThreads);

    auto invoke = [&](auto t) {
        using T = decltype(t);
        ComputeBetaGKernel<T><<<gate_blocks, kBlockThreads, 0, stream>>>(beta.data<float>(),
                                                                         g.data<float>(),
                                                                         beta.stride(1),
                                                                         b.data<T>(),
                                                                         b.stride(0),
                                                                         a.data<T>(),
                                                                         a.stride(0),
                                                                         A_log.data<T>(),
                                                                         dt_bias.data<T>(),
                                                                         token_num,
                                                                         hv);
    };
    TM_DISPATCH_PRIMARY_DTYPES(b.dtype(), invoke);
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace turbomind
