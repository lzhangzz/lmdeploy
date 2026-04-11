#include "src/turbomind/core/buffer.h"
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/core/meta.h"

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include <numeric>
#include <utility>

namespace turbomind::core {

// ============================================================================
// CuTE helper: convert Array<T, N> to cute::tuple via index_sequence
// ============================================================================
template<typename T, int N, std::size_t... Is>
TM_HOST_DEVICE auto to_cute_tuple(const Array<T, N>& arr, std::index_sequence<Is...>)
{
    return cute::make_tuple(static_cast<T>(arr[Is])...);
}

template<typename T, int N>
TM_HOST_DEVICE auto to_cute_tuple(const Array<T, N>& arr)
{
    return to_cute_tuple(arr, std::make_index_sequence<N>{});
}

// ============================================================================
// CUDA kernel: GenericCopyKernel
// ============================================================================
namespace kernel {

template<typename VecT, int kRank>
__global__ void GenericCopyKernel(const VecT* __restrict__ src_ptr,
                                  VecT* __restrict__       dst_ptr,
                                  Array<int64_t, kRank>    src_strides,
                                  Array<int64_t, kRank>    dst_strides,
                                  Array<int32_t, kRank>    shape,
                                  int64_t                  count)
{
    const int64_t idx = static_cast<int64_t>(threadIdx.x) + static_cast<int64_t>(blockIdx.x) * blockDim.x;

    if (idx >= count) {
        return;
    }

    // Build CuTE tensors with dynamic layouts
    auto src_layout = cute::make_layout(to_cute_tuple(shape), to_cute_tuple(src_strides));
    auto dst_layout = cute::make_layout(to_cute_tuple(shape), to_cute_tuple(dst_strides));

    auto src_tensor = cute::make_tensor(src_ptr, src_layout);
    auto dst_tensor = cute::make_tensor(dst_ptr, dst_layout);

    // Decompose linear index into multi-dim coordinates
    Array<int32_t, kRank> coord;
    int64_t rem = idx;
    PRAGMA_UNROLL
    for (int i = 0; i < kRank; ++i) {
        coord[i] = static_cast<int32_t>(rem % shape[i]);
        rem /= shape[i];
    }

    // Convert coord to cute::tuple for tensor accessor
    dst_tensor(to_cute_tuple(coord)) = src_tensor(to_cute_tuple(coord));
}

}  // namespace kernel

// ============================================================================
// Host function: GenericCopy
// ============================================================================
void GenericCopy(const Tensor& src, Tensor& dst, const Stream& stream)
{
    auto a = src.layout();
    auto b = dst.layout();

    // Sort strides ascending
    vector<int> idxs(a.rank());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [&](int i, int j) {  //
        return a.stride()[i] < a.stride()[j];
    });

    a = a.permute(idxs);
    b = b.permute(idxs);

    a = a.coalesce();
    b = b.coalesce();

    int rank = std::max(a.rank(), b.rank());

    if (a.rank() < rank) {
        a = a.view(b.shape());
    }
    else if (b.rank() < rank) {
        b = b.view(a.shape());
    }

    const DataType dtype = src.dtype();

    int64_t alignment = 16;

    auto align = [&](auto v) { alignment = std::gcd(alignment, v); };

    if (a.stride(0) > 1 || b.stride(0) > 1) {
        alignment = byte_size(dtype);
    }

    align(byte_size(dtype, a.shape(0)));

    auto data_a = src.raw_data();
    auto data_b = dst.raw_data();

    align(reinterpret_cast<uintptr_t>(data_a));
    align(reinterpret_cast<uintptr_t>(data_b));

    for (int i = 1; i < rank; ++i) {
        align(byte_size(dtype, a.stride(i)));
        align(byte_size(dtype, b.stride(i)));
    }

    const int64_t vec_size = alignment / std::max<int64_t>(1, byte_size(dtype));
    const int64_t size     = a.size() / vec_size;

    int device{};
    check_cuda_error(cudaGetDevice(&device));
    int sm_num{};
    check_cuda_error(cudaDeviceGetAttribute(&sm_num, cudaDevAttrMultiProcessorCount, device));

    auto invoke = [&](auto vec_t, auto index_t, auto d) {
        using VecT    = decltype(vec_t);
        using IndexT  = decltype(index_t);
        constexpr int kRank = d.value;

        Array<int32_t, kRank> shape;
        std::fill(shape.begin() + rank, shape.end(), 1);
        std::copy_n(a.shape().data(), rank, shape.data());

        Array<int64_t, kRank> stride_a{};
        Array<int64_t, kRank> stride_b{};
        std::copy_n(a.stride().data(), rank, stride_a.data());
        std::copy_n(b.stride().data(), rank, stride_b.data());

        if (vec_size > 1) {
            shape[0] /= vec_size;
            for (int i = 0; i < rank; ++i) {
                stride_a[i] /= vec_size;
                stride_b[i] /= vec_size;
            }
        }

        auto func = kernel::GenericCopyKernel<VecT, kRank>;

        int min_waves  = INT_MAX;
        int block_size = 0;
        int grid_size  = 0;

        for (int threads = 256; threads <= 1024; threads *= 2) {
            int blocks = cdiv<ssize_t>(size, threads);
            int n_active{};
            check_cuda_error(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n_active, func, threads, 0));
            int waves = cdiv(blocks, n_active * sm_num);
            if (waves < min_waves) {
                min_waves  = waves;
                block_size = threads;
                grid_size  = blocks;
            }
        }

        func<<<grid_size, block_size, 0, stream.handle()>>>(
            reinterpret_cast<const VecT*>(data_a),
            reinterpret_cast<VecT*>(data_b),
            stride_a,
            stride_b,
            shape,
            size);
    };

    auto invoke_d = [&](auto vec_t, auto idx_t) {
        if (rank <= 2) {
            invoke(vec_t, idx_t, constant<2>{});
        }
        else if (rank <= 4) {
            invoke(vec_t, idx_t, constant<4>{});
        }
        else if (rank <= 6) {
            invoke(vec_t, idx_t, constant<6>{});
        }
        else {
            throw std::runtime_error("GenericCopy: rank > 6 not implemented");
        }
    };

    auto invoke_i = [&](auto vec_t) {
        if (size < INT_MAX) {
            invoke_d(vec_t, int{});
        }
        else {
            invoke_d(vec_t, int64_t{});
        }
    };

    switch (alignment) {
        case 16:
            return invoke_i(uint4{});
        case 8:
            return invoke_i(uint2{});
        case 4:
            return invoke_i(uint{});
        case 2:
            return invoke_i(ushort{});
        default:
            return invoke_i(char{});
    }
}

}  // namespace turbomind::core
