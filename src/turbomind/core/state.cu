
#include "src/turbomind/core/state.h"

namespace turbomind {

template<class T>
__global__ void MaskedGather_Kernel(const T* src, const int* perm, const int* mask, int n, T* dst)
{
    int64_t tidx = threadIdx.x + (int64_t)blockDim.x * blockIdx.x;

    if (tidx >= n || mask[tidx] == 0) {
        return;
    }

    dst[tidx] = src[perm[tidx]];
}

template<class T>
__global__ void MaskedGather_Kernel(const T* src, const int* perm, const int* mask, int n, T* dst, int vec_size)
{
    int64_t tidx = threadIdx.x + (int64_t)blockDim.x * blockIdx.x;

    auto vidx = tidx / vec_size;

    if (vidx >= n || mask[vidx] == 0) {
        return;
    }

    auto offset = tidx % vec_size;

    dst[vidx * vec_size + offset] = src[perm[vidx] * vec_size + offset];
}

void MaskedGather(const Tensor& src, const Buffer_<int>& perm, const Buffer_<int>& mask, const Tensor& dst)
{
    TM_CHECK_EQ(src.dtype(), dst.dtype());
    TM_CHECK_EQ(src.ndim(), dst.ndim());
    TM_CHECK_EQ(src.stride(0), dst.stride(0));
    TM_CHECK_EQ(perm.size(), mask.size());

    const auto elem_size = byte_size(src.dtype());

    auto stream = core::Context::stream().handle();

    if (src.ndim() == 1) {
        auto invoke = [&](auto t) {
            using T           = decltype(t);
            const int n       = mask.size();
            const int threads = 256;
            const int blocks  = cdiv(n, threads);
            MaskedGather_Kernel<<<blocks, threads, 0, stream>>>(
                (const T*)src.raw_data(), perm.data(), mask.data(), n, (T*)dst.raw_data());
        };
        if (0) {}
        else if (elem_size == sizeof(uint32_t)) {
            invoke(uint32_t{});
        }
        else if (elem_size == sizeof(uint64_t)) {
            invoke(uint64_t{});
        }
        else if (elem_size == sizeof(uint16_t)) {
            invoke(uint16_t{});
        }
    }
    else {
        const int vec_size = src.stride(0);
        auto      invoke   = [&](auto t) {
            using T           = decltype(t);
            const int n       = mask.size();
            const int threads = 256;
            const int blocks  = cdiv(n * vec_size, threads);
            MaskedGather_Kernel<<<blocks, threads, 0, stream>>>(
                (const T*)src.raw_data(), perm.data(), mask.data(), n, (T*)dst.raw_data(), vec_size);
        };
        if (0) {}
        else if (elem_size == sizeof(uint32_t)) {
            invoke(uint32_t{});
        }
        else if (elem_size == sizeof(uint64_t)) {
            invoke(uint64_t{});
        }
        else if (elem_size == sizeof(uint16_t)) {
            invoke(uint16_t{});
        }
    }
}

}  // namespace turbomind