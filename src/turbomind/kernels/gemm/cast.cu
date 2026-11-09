// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/cast.h"

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

template<class Ti, class To>
struct Cast {
    template<int N>
    __device__ static Array<To, N> apply(const Array<Ti, N>& vi)
    {
        Array<To, N> vo;
        PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            vo[i] = static_cast<To>(vi[i]);
        }
        return vo;
    }
};

template<class Ti>
struct Cast<Ti, uint4_t> {
    template<int N>
    __device__ static Array<uint4_t, N> apply(const Array<Ti, N>& vi)
    {
        static_assert(N % 8 == 0);
        Array<uint4_t, N> vo;
        PRAGMA_UNROLL
        for (int i = 0; i < N; i += 8) {
            uint32_t& v = (uint32_t&)vo[i];
            v           = 0;
            PRAGMA_UNROLL
            for (int j = 7; j >= 0; --j) {
                v = (v << 4) | vi[i + j];
            }
        }
        return vo;
    }
};

template<class To>
struct Cast<uint4_t, To> {
    template<int N>
    __device__ static Array<To, N> apply(const Array<uint4_t, N>& vi)
    {
        static_assert(N % 8 == 0);
        Array<To, N> vo;
        PRAGMA_UNROLL
        for (int i = 0; i < N; i += 8) {
            uint32_t v = (const uint32_t&)vi[i];
            PRAGMA_UNROLL
            for (int j = 0; j < 8; ++j) {
                vo[i + j] = (v & 0xf);
                v >>= 4;
            }
        }
        return vo;
    }
};

template<>
struct Cast<uint4_t, uint4_t> {
    template<int N>
    __device__ static Array<uint4_t, N> apply(const Array<uint4_t, N>& vi)
    {
        return vi;
    }
};

template<int VecSize, class Ti, class To>
__global__ void cast_kernel(To* dst, const Ti* src, size_t n)
{
    n /= VecSize;

    auto p_src = (const Array<Ti, VecSize>*)src;
    auto p_dst = (Array<To, VecSize>*)dst;

    for (size_t p = threadIdx.x + blockDim.x * blockIdx.x; p < n; p += blockDim.x * gridDim.x) {
        Array<Ti, VecSize> vi;
        Ldg(vi, (const Ti*)&p_src[p]);

        Array<To, VecSize> vo = Cast<Ti, To>::apply(vi);

        Store((To*)&p_dst[p], vo);
    }
}

template<int VecSize, class Ti, class To>
void invokeCast(To* dst, const Ti* src, size_t n, cudaStream_t st)
{
    cast_kernel<VecSize><<<256, 256, 0, st>>>(dst, src, n);
    TM_CUDA_CHECK(cudaGetLastError());
}

void extend_to_u16(uint16_t* dst, const uint4_t* src, size_t n, cudaStream_t st)
{
    TM_SCOPE_CALL(invokeCast<8>(dst, src, n, st));
}

namespace {

__global__ void extend_u16_u8(uint16_t* dst, const uint8_t* src, size_t n)
{
    int64_t idx = threadIdx.x + (int64_t)blockDim.x * blockIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

}  // namespace

void extend_to_u16(uint16_t* dst, const uint8_t* src, size_t n, cudaStream_t st)
{
    extend_u16_u8<<<(n + 511) / 512, 512, 0, st>>>(dst, src, n);
    TM_CUDA_CHECK(cudaGetLastError());
}

template<int VecSize, class T>
__global__ void fuse_scales_and_zeros_kernel(T* fused, const T* scales, T* zeros, size_t n)
{
    n /= VecSize;

    auto p_scales = (const Array<T, VecSize>*)scales;
    auto p_zeros  = (const Array<T, VecSize>*)zeros;

    auto p_fused = (Array<T, VecSize * 2>*)fused;

    for (size_t p = threadIdx.x + blockDim.x * blockIdx.x; p < n; p += blockDim.x * gridDim.x) {
        Array<T, VecSize> vs;
        Ldg(vs, (const T*)&p_scales[p]);
        Array<T, VecSize> vz{};
        if (zeros) {
            Ldg(vz, (const T*)&p_zeros[p]);
        }
        Array<T, VecSize * 2> vf;
        PRAGMA_UNROLL
        for (int i = 0; i < VecSize; ++i) {
            vf[i * 2]     = vs[i];
            vf[i * 2 + 1] = -vz[i] * vs[i];
        }
        Store((T*)&p_fused[p], vf);
    }
}

void fuse_scales_and_zeros(half* fused, const half* scales, half* zeros, size_t n, cudaStream_t st)
{
    fuse_scales_and_zeros_kernel<4><<<256, 256, 0, st>>>(fused, scales, zeros, n);
    TM_CUDA_CHECK(cudaGetLastError());
}

__global__ void adjust_ue8m0_scale_for_half_kernel(uint8_t* data, int n)
{
    int64_t idx = threadIdx.x + (int64_t)blockDim.x * blockIdx.x;
    if (idx < n) {
        /// TODO: saturate the quantized values accordingly
        data[idx] = max(0, min(30, (int)data[idx] + 15 - 127));  // exponent 31 is INF in half
    }
}

void AdjustUe8m0ScaleForHalf(uint8_t* data, int n, cudaStream_t st)
{
    constexpr int block = 512;
    const int     grid  = cdiv(n, block);
    adjust_ue8m0_scale_for_half_kernel<<<grid, block, 0, st>>>(data, n);
    TM_CUDA_CHECK(cudaGetLastError());
}
}  // namespace turbomind
