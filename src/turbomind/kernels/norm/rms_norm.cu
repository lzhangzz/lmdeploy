// Copyright (c) OpenMMLab. All rights reserved.

#include <stdexcept>

#include "cub/block/block_reduce.cuh"

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/core/meta.h"

#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/kernels/norm/rms_norm_utils.cuh"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

namespace kernel {

template<bool ZeroCentered, class T, class Accum, int block_dim, int vec_size>
__global__ void RMSNorm(T*       dst,
                        int      dst_ld,
                        const T* src,
                        int      src_ld,
                        const T* __restrict__ weights,
                        int   dims,
                        int   num,
                        float eps,
                        float inv_dims)
{
    const int ti = blockIdx.x;
    const int di = threadIdx.x * vec_size;

    if (ti >= num) {
        return;
    }

    src += src_ld * ti;

    Array<Accum, vec_size> accum{};
    Array<T, vec_size>     vec;

    for (int i = di; i < dims; i += block_dim * vec_size) {
        Load(vec, &src[i]);
        Array<Accum, vec_size> tmp = cast<Accum>(vec);
        using namespace ops;
        accum = accum + tmp * tmp;
    }

    float sum{};
    PRAGMA_UNROLL
    for (int i = 0; i < vec_size; ++i) {
        sum += accum[i];
    }

    using BlockReduce = cub::BlockReduce<Accum, block_dim>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    sum = BlockReduce{temp_storage}.Sum(sum);

    __shared__ float shared_sum;

    if (threadIdx.x == 0) {
        shared_sum = rsqrtf(sum * inv_dims + eps);
    }

    __syncthreads();

    sum = shared_sum;

    dst += dst_ld * ti;

    Array<T, vec_size> sv;
    for (int i = di; i < dims; i += block_dim * vec_size) {
        Load(vec, &src[i]);
        Ldg(sv, &weights[i]);
        PRAGMA_UNROLL
        for (int c = 0; c < vec_size; ++c) {
            vec[c] = ApplyRMSnorm<ZeroCentered>(vec[c], sum, sv[c]);
        }
        Store(&dst[i], vec);
    }
}

}  // namespace kernel

void invokeRMSNorm(Tensor& out, const Tensor& x, const Tensor& w, float eps, bool zero_centered, cudaStream_t st)
{
    if (x.size() == 0) {
        return;
    }

    TM_CHECK(x.ndim() == 2);
    TM_CHECK(out.shape() == x.shape());
    TM_CHECK(out.dtype() == x.dtype());
    TM_CHECK(w.dtype() == x.dtype() && w.shape(-1) == x.shape(-1));

    auto invoke = [&](auto t) {
        using T = decltype(t);

        const int num = x.shape(0);
        const int dim = x.shape(1);

        const int blocks = num;

        auto launch = [&](auto zero_centered_c) {
            // Redeclare these as constexpr inside this nested lambda. MSVC odr-uses
            // (captures) constexpr locals read from an enclosing lambda and then
            // refuses them as non-type template arguments.
            constexpr int  kThreads     = 512;
            constexpr int  kVecSize     = 16 / sizeof(T);
            constexpr bool ZeroCentered = decltype(zero_centered_c)::value;
            kernel::RMSNorm<ZeroCentered, T, float, kThreads, kVecSize>
                <<<blocks, kThreads, 0, st>>>((T*)out.raw_data(),  //
                                              out.stride(0),
                                              (const T*)x.raw_data(),
                                              x.stride(0),
                                              (const T*)w.raw_data(),
                                              dim,
                                              num,
                                              eps,
                                              1.f / dim);
        };

        if (zero_centered) {
            launch(constant<true>{});
        }
        else {
            launch(constant<false>{});
        }
    };

    TM_DISPATCH_PRIMARY_DTYPES(x.dtype(), invoke);
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeRMSNormConcat(Tensor&       out,
                         const Tensor& left,
                         const Tensor& left_weight,
                         float         left_eps,
                         bool          left_zero_centered,
                         const Tensor& right,
                         const Tensor& right_weight,
                         float         right_eps,
                         bool          right_zero_centered,
                         cudaStream_t  stream)
{
    const auto [token_num, hidden_size] = left.shapes(0, 1);

    if (token_num == 0) {
        return;
    }

    Tensor left_out  = out.slice({0, 0}, {token_num, hidden_size});
    Tensor right_out = out.slice({0, hidden_size}, {token_num, hidden_size});

    invokeRMSNorm(left_out, left, left_weight, left_eps, left_zero_centered, stream);
    invokeRMSNorm(right_out, right, right_weight, right_eps, right_zero_centered, stream);
}

namespace kernel {

template<class T, class A, int vec_size, int max_dim, bool ZeroCentered>
__global__ void QkRMSNorm(T*       qkv,
                          int      ld,
                          const T* q_weight,
                          const T* k_weight,
                          int      dim,
                          int      q_head_num,
                          int      k_head_num,
                          int      token_num,
                          int      q_block_num,
                          float    eps,
                          float    inv_dim,
                          constant<ZeroCentered>)
{
    static_assert((max_dim & (max_dim - 1)) == 0);

    constexpr int threads_per_head = max_dim / vec_size;

    const bool is_k        = blockIdx.x >= q_block_num;
    const int  block_idx   = is_k ? blockIdx.x - q_block_num : blockIdx.x;
    const int  head_num    = is_k ? k_head_num : q_head_num;
    const int  head_offset = is_k ? q_head_num : 0;
    const T*   weight      = is_k ? k_weight : q_weight;

    const int bi = (threadIdx.x + block_idx * blockDim.x) / threads_per_head;
    const int di = threadIdx.x % threads_per_head * vec_size;

    if (bi >= token_num * head_num) {
        return;
    }

    const int ti = bi / head_num;
    const int hi = bi % head_num;

    qkv += ti * ld + (head_offset + hi) * dim;

    Array<T, vec_size> vec{};
    if (di < dim) {
        Load(vec, &qkv[di]);
    }

    using namespace ops;
    auto acc = cast<A>(vec);
    acc      = acc * acc;

    float sum{};
    PRAGMA_UNROLL
    for (int i = 0; i < vec_size; ++i) {
        sum += acc[i];
    }

    PRAGMA_UNROLL
    for (int mask = threads_per_head / 2; mask >= 1; mask /= 2) {
        sum += __shfl_xor_sync((uint32_t)-1, sum, mask);
    }

    sum = rsqrtf(sum * inv_dim + eps);

    Array<T, vec_size> w;
    if (di < dim) {
        Ldg(w, &weight[di]);
        PRAGMA_UNROLL
        for (int i = 0; i < vec_size; ++i) {
            vec[i] = ApplyRMSnorm<ZeroCentered>(vec[i], sum, w[i]);
        }
        Store(&qkv[di], vec);
    }
}

}  // namespace kernel

void invokeQkRMSNorm(Tensor&       qkv,
                     const Tensor& q_weight,
                     const Tensor& k_weight,
                     int           q_head_num,
                     int           k_head_num,
                     float         eps,
                     bool          zero_centered,
                     cudaStream_t  stream)
{
    TM_CHECK(qkv.ndim() == 3);

    const int token_num = qkv.shape(0);
    const int head_dim  = qkv.shape(2);

    TM_CHECK(qkv.stride(1) == head_dim);

    auto data   = qkv.raw_data();
    auto stride = qkv.stride(0);

    auto invoke = [&](auto t) {
        using T = decltype(t);

        auto launch = [&](auto max_dim_c, auto zero_centered_c) {
            constexpr int kMaxDim = std::decay_t<decltype(max_dim_c)>::value;
            TM_CHECK_LE(head_dim, kMaxDim);

            constexpr int vec_size         = sizeof(uint4) / sizeof(T);
            constexpr int threads_per_head = kMaxDim / vec_size;
            constexpr int block_dim        = 512;

            TM_CHECK(head_dim % vec_size == 0);

            const int q_block_num = cdiv(token_num * q_head_num * threads_per_head, block_dim);
            const int k_block_num = cdiv(token_num * k_head_num * threads_per_head, block_dim);
            const int grid_dim    = q_block_num + k_block_num;

            kernel::QkRMSNorm<T, float, vec_size, kMaxDim>
                <<<grid_dim, block_dim, 0, stream>>>((T*)data,
                                                     stride,
                                                     (const T*)q_weight.raw_data(),
                                                     (const T*)k_weight.raw_data(),
                                                     head_dim,
                                                     q_head_num,
                                                     k_head_num,
                                                     token_num,
                                                     q_block_num,
                                                     eps,
                                                     1.f / head_dim,
                                                     zero_centered_c);
        };

        if (head_dim <= 128) {
            if (zero_centered) {
                launch(constant<128>{}, constant<true>{});
            }
            else {
                launch(constant<128>{}, constant<false>{});
            }
        }
        else {
            if (zero_centered) {
                launch(constant<256>{}, constant<true>{});
            }
            else {
                launch(constant<256>{}, constant<false>{});
            }
        }
    };

    TM_DISPATCH_PRIMARY_DTYPES(qkv.dtype(), invoke);
    TM_CUDA_CHECK(cudaGetLastError());
}

// r' <- r + (h + b)
// h' <- norm(r') * w
template<bool ZeroCentered, class T, class Tacc, int block_dim, int vec_size>
__global__ void BiasResidualRMSNormKernel(T* __restrict__ residual,
                                          T* __restrict__ hidden_states,
                                          const T* __restrict__ weights,
                                          const T* __restrict__ bias,
                                          int   dims,
                                          int   num,
                                          float eps,
                                          float inv_dims)
{
    const int ti = blockIdx.x;
    const int di = threadIdx.x * vec_size;

    if (ti >= num) {
        return;
    }

    residual += dims * ti;
    hidden_states += dims * ti;

    Array<Tacc, vec_size> accum{};

    Array<T, vec_size> r_vec;
    Array<T, vec_size> h_vec;
    Array<T, vec_size> b_vec;

    for (int i = di; i < dims; i += block_dim * vec_size) {
        Load(r_vec, &residual[i]);
        Load(h_vec, &hidden_states[i]);

        using namespace ops;
        r_vec = r_vec + h_vec;

        if (bias) {
            Ldg(b_vec, &bias[i]);
            r_vec = r_vec + b_vec;
        }

        Store(&residual[i], r_vec);

        Array<Tacc, vec_size> tmp = cast<Tacc>(r_vec);

        accum = accum + tmp * tmp;
    }

    float sum{};
    PRAGMA_UNROLL
    for (int i = 0; i < vec_size; ++i) {
        sum += accum[i];
    }

    using BlockReduce = cub::BlockReduce<Tacc, block_dim>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    sum = BlockReduce{temp_storage}.Sum(sum);

    __shared__ float shared_sum;

    if (threadIdx.x == 0) {
        shared_sum = rsqrtf(sum * inv_dims + eps);
    }

    __syncthreads();

    sum = shared_sum;

    Array<T, vec_size> w_vec;
    for (int i = di; i < dims; i += block_dim * vec_size) {
        Load(r_vec, &residual[i]);
        Ldg(w_vec, &weights[i]);
        PRAGMA_UNROLL
        for (int c = 0; c < vec_size; ++c) {
            r_vec[c] = kernel::ApplyRMSnorm<ZeroCentered>(r_vec[c], sum, w_vec[c]);
        }
        Store(&hidden_states[i], r_vec);
    }
}

void invokeResidualBiasRMSNorm(void*        hidden_states,
                               void*        residual,
                               const void*  weights,
                               const void*  bias,
                               DataType     dtype,
                               int          dims,
                               int          num,
                               float        eps,
                               bool         zero_centered,
                               cudaStream_t st)
{
    if (num == 0) {
        return;
    }
    auto invoke = [&](auto t) {
        using T          = decltype(t);
        const int blocks = num;
        auto      launch = [&](auto zero_centered_c) {
            // Keep kernel template constants local to the nested lambda for MSVC.
            constexpr int  kThreads     = 512;
            constexpr int  kVecSize     = sizeof(uint4) / sizeof(T);
            constexpr bool ZeroCentered = decltype(zero_centered_c)::value;
            BiasResidualRMSNormKernel<ZeroCentered, T, float, kThreads, kVecSize>
                <<<blocks, kThreads, 0, st>>>((T*)residual,  //
                                              (T*)hidden_states,
                                              (const T*)weights,
                                              (const T*)bias,
                                              dims,
                                              num,
                                              eps,
                                              1.f / dims);
        };

        if (zero_centered) {
            launch(constant<true>{});
        }
        else {
            launch(constant<false>{});
        }
    };

    TM_DISPATCH_PRIMARY_DTYPES(dtype, invoke);
    TM_CUDA_CHECK(cudaGetLastError());
}

template<class T, class B, int vec_size>
__global__ void biasKernel(T* output, T* data, const B* bias, int num, int dim)
{
    int ti = blockIdx.x;
    int di = threadIdx.x * vec_size;

    Array<B, vec_size> b;
    Ldg(b, bias + di);

    Array<T, vec_size> x;
    Load(x, data + ti * dim + di);
    using namespace ops;
    x = x + cast<T>(b);
    Store(output + ti * dim + di, x);
}

template<bool inplace>
void ApplyBias(Tensor& output, Tensor& data, const Tensor& bias, cudaStream_t st)
{
    if constexpr (!inplace) {
        TM_CHECK(output.shape() == data.shape());
        TM_CHECK_EQ(output.dtype(), data.dtype());
        TM_CHECK(output.is_contiguous());
        TM_CHECK(data.is_contiguous());
    }
    if (!bias) {
        if constexpr (!inplace) {
            if (auto size = data.byte_size()) {
                TM_CUDA_CHECK(cudaMemcpyAsync(output.raw_data(), data.raw_data(), size, cudaMemcpyDefault, st));
            }
        }
        return;
    }

    const int num = data.shape(0);
    const int dim = data.shape(1);

    TM_CHECK_EQ(dim, bias.shape(-1));

    auto invoke0 = [&](auto t) {
        using T      = decltype(t);
        auto invoke1 = [&](auto b) {
            using B                = decltype(b);
            constexpr int vec_size = sizeof(uint4) / std::max(sizeof(T), sizeof(B));
            TM_CHECK(dim % vec_size == 0);
            const int blocks  = num;
            const int threads = dim / vec_size;
            TM_CHECK_LE(threads, 1024);
            biasKernel<T, B, vec_size><<<blocks, threads, 0, st>>>(output.data<T>(),  //
                                                                   data.data<T>(),
                                                                   bias.data<B>(),
                                                                   num,
                                                                   dim);
        };
        if constexpr (data_type_v<T> == kFloat) {
            TM_DISPATCH_PRIMARY_DTYPES(bias.dtype(), invoke1);
        }
        else {  // skip mixing half and bf16
            invoke1(t);
        }
    };
    TM_DISPATCH_DTYPES(data.dtype(), invoke0, float, half, nv_bfloat16);
    TM_CUDA_CHECK(cudaGetLastError());
}

void ApplyBias(Tensor& data, const Tensor& bias, cudaStream_t st)
{
    ApplyBias<true>(data, data, bias, st);
}

void ApplyBias(Tensor& output, Tensor& data, const Tensor& bias, cudaStream_t st)
{
    ApplyBias<false>(output, data, bias, st);
}

// =============================================================================
// RMSNorm * SiLU-Gate (fused output normalization)
// =============================================================================
template<typename T>
__global__ void rms_norm_gated_kernel(
    T* hidden, const T* gate, const T* weight, float eps, int N, int head_dim, int gate_stride, int num_heads)
{
    const int row = blockIdx.x;
    if (row >= N)
        return;

    T*        h         = hidden + row * head_dim;
    const int token_idx = row / num_heads;
    const int head_idx  = row % num_heads;
    const T*  g         = gate + token_idx * gate_stride + head_idx * head_dim;

    __shared__ float smem[32];
    float            sum_sq = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float val = static_cast<float>(h[d]);
        sum_sq += val * val;
    }
    for (int mask = 16; mask > 0; mask >>= 1)
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, mask);
    if ((threadIdx.x & 31) == 0)
        smem[threadIdx.x >> 5] = sum_sq;
    __syncthreads();
    if (threadIdx.x >> 5 == 0) {
        sum_sq = (threadIdx.x < (blockDim.x + 31) / 32) ? smem[threadIdx.x] : 0.0f;
        for (int mask = 16; mask > 0; mask >>= 1)
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, mask);
        if (threadIdx.x == 0)
            smem[0] = sum_sq;
    }
    __syncthreads();
    sum_sq = smem[0];

    float inv_rms = rsqrtf(sum_sq / (float)head_dim + eps);
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float h_val  = static_cast<float>(h[d]) * inv_rms * static_cast<float>(weight[d]);
        float g_val  = static_cast<float>(g[d]);
        float silu_g = g_val / (1.0f + expf(-g_val));
        h[d]         = static_cast<T>(h_val * silu_g);
    }
}

void invokeRMSNormGated(Ref<Tensor> hidden_, const Tensor& gate, const Tensor& weight, float eps, cudaStream_t stream)
{
    auto& hidden = hidden_.get();

    const int N           = hidden.shape(0);
    const int head_dim    = hidden.shape(1);
    const int token_num   = gate.shape(0);
    const int gate_stride = gate.stride(0);
    const int num_heads   = N / token_num;

    if (N == 0)
        return;

    const int threads = std::min(256, head_dim);

    auto invoke = [&](auto t) {
        using T = decltype(t);
        rms_norm_gated_kernel<<<N, threads, 0, stream>>>(
            hidden.data<T>(), gate.data<T>(), weight.data<T>(), eps, N, head_dim, gate_stride, num_heads);
    };
    TM_DISPATCH_PRIMARY_DTYPES(hidden.dtype(), invoke);
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace turbomind
