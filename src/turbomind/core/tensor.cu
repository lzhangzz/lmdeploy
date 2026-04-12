#include "src/turbomind/core/tensor.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/core/meta.h"

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>

#include <numeric>

namespace turbomind::core {

// ============================================================================
// Helpers: Construct CuTe layouts from runtime shape/stride arrays
// ============================================================================
namespace detail {

template<int kRank>
auto make_cute_shape(const ssize_t* shape_data)
{
    if constexpr (kRank == 1) {
        return cute::make_shape(static_cast<int32_t>(shape_data[0]));
    }
    else if constexpr (kRank == 2) {
        return cute::make_shape(static_cast<int32_t>(shape_data[0]),
                                static_cast<int32_t>(shape_data[1]));
    }
    else if constexpr (kRank == 3) {
        return cute::make_shape(static_cast<int32_t>(shape_data[0]),
                                static_cast<int32_t>(shape_data[1]),
                                static_cast<int32_t>(shape_data[2]));
    }
    else if constexpr (kRank == 4) {
        return cute::make_shape(static_cast<int32_t>(shape_data[0]),
                                static_cast<int32_t>(shape_data[1]),
                                static_cast<int32_t>(shape_data[2]),
                                static_cast<int32_t>(shape_data[3]));
    }
    else if constexpr (kRank == 5) {
        return cute::make_shape(static_cast<int32_t>(shape_data[0]),
                                static_cast<int32_t>(shape_data[1]),
                                static_cast<int32_t>(shape_data[2]),
                                static_cast<int32_t>(shape_data[3]),
                                static_cast<int32_t>(shape_data[4]));
    }
    else if constexpr (kRank == 6) {
        return cute::make_shape(static_cast<int32_t>(shape_data[0]),
                                static_cast<int32_t>(shape_data[1]),
                                static_cast<int32_t>(shape_data[2]),
                                static_cast<int32_t>(shape_data[3]),
                                static_cast<int32_t>(shape_data[4]),
                                static_cast<int32_t>(shape_data[5]));
    }
}

template<int kRank>
auto make_cute_stride(const ssize_t* stride_data)
{
    if constexpr (kRank == 1) {
        return cute::make_stride(static_cast<int64_t>(stride_data[0]));
    }
    else if constexpr (kRank == 2) {
        return cute::make_stride(static_cast<int64_t>(stride_data[0]),
                                 static_cast<int64_t>(stride_data[1]));
    }
    else if constexpr (kRank == 3) {
        return cute::make_stride(static_cast<int64_t>(stride_data[0]),
                                 static_cast<int64_t>(stride_data[1]),
                                 static_cast<int64_t>(stride_data[2]));
    }
    else if constexpr (kRank == 4) {
        return cute::make_stride(static_cast<int64_t>(stride_data[0]),
                                 static_cast<int64_t>(stride_data[1]),
                                 static_cast<int64_t>(stride_data[2]),
                                 static_cast<int64_t>(stride_data[3]));
    }
    else if constexpr (kRank == 5) {
        return cute::make_stride(static_cast<int64_t>(stride_data[0]),
                                 static_cast<int64_t>(stride_data[1]),
                                 static_cast<int64_t>(stride_data[2]),
                                 static_cast<int64_t>(stride_data[3]),
                                 static_cast<int64_t>(stride_data[4]));
    }
    else if constexpr (kRank == 6) {
        return cute::make_stride(static_cast<int64_t>(stride_data[0]),
                                 static_cast<int64_t>(stride_data[1]),
                                 static_cast<int64_t>(stride_data[2]),
                                 static_cast<int64_t>(stride_data[3]),
                                 static_cast<int64_t>(stride_data[4]),
                                 static_cast<int64_t>(stride_data[5]));
    }
}

template<int kRank>
auto make_cute_layout(const ssize_t* shape, const ssize_t* stride)
{
    return cute::make_layout(make_cute_shape<kRank>(shape),
                             make_cute_stride<kRank>(stride));
}

}  // namespace detail

// ============================================================================
// CUDA kernel: GenericCopyKernel (CuTe TiledCopy)
// ============================================================================
namespace kernel {

template<typename VecT, typename SrcLayoutT, typename DstLayoutT>
__global__ void __launch_bounds__(256)
GenericCopyKernel(const VecT* __restrict__ src_ptr,
                  VecT* __restrict__       dst_ptr,
                  SrcLayoutT               src_layout,
                  DstLayoutT               dst_layout,
                  int64_t                  inner_size,
                  int64_t                  outer_total)
{
    constexpr int kRank        = cute::rank_v<SrcLayoutT>;
    constexpr int kBlockThreads = 256;

    // 1. Map CTA to outer-dim coordinates via blockIdx.x
    int64_t outer_idx = blockIdx.x;
    if (outer_idx >= outer_total) {
        return;
    }

    // 2. Extract shape/stride from CuTe layout into local arrays
    int32_t shape[kRank];
    int64_t src_strides[kRank];
    int64_t dst_strides[kRank];

    auto src_shape   = cute::shape(src_layout);
    auto src_stride  = cute::stride(src_layout);
    auto dst_stride  = cute::stride(dst_layout);

    cute::for_each(cute::make_seq<kRank>{}, [&](auto i) {
        shape[i]        = cute::get<decltype(i){}>(src_shape);
        src_strides[i]  = cute::get<decltype(i){}>(src_stride);
        dst_strides[i]  = cute::get<decltype(i){}>(dst_stride);
    });

    // 3. Decompose outer_idx into coordinates for dims 1..kRank-1
    const VecT* my_src = src_ptr;
    VecT*       my_dst = dst_ptr;
    int64_t     rem    = outer_idx;

    PRAGMA_UNROLL
    for (int i = kRank - 1; i >= 1; --i) {
        int64_t c = rem % shape[i];
        rem /= shape[i];
        my_src += c * src_strides[i];
        my_dst += c * dst_strides[i];
    }

    // 4. Build TiledCopy for the innermost dimension
    auto tiled_copy = cute::make_tiled_copy(
        cute::Copy_Atom<cute::UniversalCopy<VecT>, VecT>{},
        cute::make_layout(cute::make_shape(cute::Int<kBlockThreads>{})),
        cute::make_layout(cute::make_shape(cute::Int<1>{})));

    auto thr_copy = tiled_copy.get_slice(threadIdx.x);

    // 5. Loop over inner-dim tiles of kBlockThreads elements each
    PRAGMA_UNROLL
    for (int64_t tile = 0; tile < inner_size; tile += kBlockThreads) {
        int64_t remaining = min(static_cast<int64_t>(kBlockThreads), inner_size - tile);

        auto tile_src = cute::make_tensor(
            my_src + tile * src_strides[0],
            cute::make_layout(cute::make_shape(remaining), cute::make_stride(src_strides[0])));
        auto tile_dst = cute::make_tensor(
            my_dst + tile * dst_strides[0],
            cute::make_layout(cute::make_shape(remaining), cute::make_stride(dst_strides[0])));

        if (threadIdx.x < remaining) {
            auto thr_src = thr_copy.partition_S(tile_src);
            auto thr_dst = thr_copy.partition_D(tile_dst);
            cute::copy(tiled_copy, thr_src, thr_dst);
        }
    }
}

}  // namespace kernel

// ============================================================================
// Host function: GenericCopy
// ============================================================================
void GenericCopy(const Tensor& src, Tensor& dst, cudaStream_t stream)
{
    auto a = src.layout();
    auto b = dst.layout();

    vector<int> idxs(a.rank());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [&](int i, int j) {
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

    auto invoke = [&](auto vec_t, auto d) {
        using VecT    = decltype(vec_t);
        constexpr int kRank = d.value;

        auto src_layout = detail::make_cute_layout<kRank>(a.shape().data(), a.stride().data());
        auto dst_layout = detail::make_cute_layout<kRank>(a.shape().data(), b.stride().data());

        int64_t inner_size  = a.shape(0) / vec_size;
        int64_t outer_total = 1;
        for (int i = 1; i < rank; ++i) {
            outer_total *= a.shape(i);
        }

        // Adjust strides for vectorization (outer dims only)
        ssize_t src_stride_adj[kRank]{};
        ssize_t dst_stride_adj[kRank]{};
        std::copy_n(a.stride().data(), rank, src_stride_adj);
        std::copy_n(b.stride().data(), rank, dst_stride_adj);

        if (vec_size > 1) {
            for (int i = 1; i < rank; ++i) {
                src_stride_adj[i] /= vec_size;
                dst_stride_adj[i] /= vec_size;
            }
            ssize_t shape_adj[kRank];
            std::copy_n(a.shape().data(), rank, shape_adj);
            shape_adj[0] /= vec_size;
            src_layout = detail::make_cute_layout<kRank>(shape_adj, src_stride_adj);
            dst_layout = detail::make_cute_layout<kRank>(shape_adj, dst_stride_adj);
        }

        auto func = kernel::GenericCopyKernel<VecT, decltype(src_layout), decltype(dst_layout)>;

        int grid_size = static_cast<int>(outer_total);

        func<<<grid_size, 256, 0, stream>>>(
            reinterpret_cast<const VecT*>(data_a),
            reinterpret_cast<VecT*>(data_b),
            src_layout,
            dst_layout,
            inner_size,
            outer_total);
    };

    auto dispatch_rank = [&](auto vec_t) {
        switch (rank) {
            case 1: invoke(vec_t, constant<1>{}); break;
            case 2: invoke(vec_t, constant<2>{}); break;
            case 3: invoke(vec_t, constant<3>{}); break;
            case 4: invoke(vec_t, constant<4>{}); break;
            case 5: invoke(vec_t, constant<5>{}); break;
            case 6: invoke(vec_t, constant<6>{}); break;
            default: throw std::runtime_error("GenericCopy: rank > 6 not implemented");
        }
    };

    switch (alignment) {
        case 16: return dispatch_rank(uint4{});
        case 8:  return dispatch_rank(uint2{});
        case 4:  return dispatch_rank(uint{});
        case 2:  return dispatch_rank(ushort{});
        default: return dispatch_rank(char{});
    }
}

}  // namespace turbomind::core
