/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/activation_kernels.h"
#include "src/turbomind/kernels/activation_ops.h"
#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/utils/cuda_type_utils.cuh"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/memory_utils.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

namespace turbomind {

template<typename T>
__global__ void SigmoidGateMultiplyKernel(T* attn, const T* gate_base, int dim, int gate_stride, int num_tokens)
{
    const int ti = blockIdx.x;
    const int di = threadIdx.x + blockIdx.y * blockDim.x;
    if (ti >= num_tokens || di >= dim) {
        return;
    }
    float g             = (float)gate_base[ti * gate_stride + di];
    float s             = 1.0f / (1.0f + __expf(-g));
    float a             = (float)attn[ti * dim + di];
    attn[ti * dim + di] = (T)(a * s);
}

void invokeSigmoidGateMultiply(
    void* attn, const void* gate_base, int dim, int gate_stride, int num_tokens, DataType dtype, cudaStream_t stream)
{
    constexpr int block = 256;
    const dim3    grid(num_tokens, cdiv(dim, block));

    auto invoke = [&](auto t) {
        using T = decltype(t);
        SigmoidGateMultiplyKernel<<<grid, block, 0, stream>>>(
            (T*)attn, (const T*)gate_base, dim, gate_stride, num_tokens);
    };

    TM_DISPATCH_PRIMARY_DTYPES(dtype, invoke);
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace turbomind
