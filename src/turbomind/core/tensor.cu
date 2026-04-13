#include "src/turbomind/core/tensor.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/core/meta.h"

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>

#include <algorithm>
#include <numeric>
#include <string>
#include <utility>

namespace turbomind::core {

using cute::_;

// CuTe's make_shape/make_stride require compile-time variadic template args,
// but our tensor shapes and strides are runtime values. These helpers bridge
// that gap via std::index_sequence expansion, producing CuTe Layout objects
// from runtime shape/stride arrays. Only the innermost stride is promoted to
// compile-time Int<1> (in make_cute_layout_unit_inner) to enable CuTe's
// vectorized Copy_Atom recast.
namespace detail {

template<size_t... Is>
auto make_cute_shape_impl(const ssize_t* data, std::index_sequence<Is...>)
{
    return cute::make_shape(static_cast<int32_t>(data[Is])...);
}

template<int kRank>
auto make_cute_shape(const ssize_t* data)
{
    return make_cute_shape_impl(data, std::make_index_sequence<kRank>{});
}

template<size_t... Is>
auto make_cute_stride_impl(const ssize_t* data, std::index_sequence<Is...>)
{
    return cute::make_stride(static_cast<int64_t>(data[Is])...);
}

template<int kRank>
auto make_cute_stride(const ssize_t* data)
{
    return make_cute_stride_impl(data, std::make_index_sequence<kRank>{});
}

template<int kRank>
auto make_cute_layout(const ssize_t* shape, const ssize_t* stride)
{
    return cute::make_layout(make_cute_shape<kRank>(shape),
                             make_cute_stride<kRank>(stride));
}

// Layout with compile-time Int<1> inner stride — needed for CuTe's recast
// in wide Copy_Atom (vectorized path). Only valid when inner stride == 1.
template<size_t... Is>
auto make_unit_inner_stride_impl(const ssize_t* stride, std::index_sequence<Is...>)
{
    return cute::make_stride(cute::Int<1>{}, static_cast<int64_t>(stride[Is + 1])...);
}

template<int kRank>
auto make_cute_layout_unit_inner(const ssize_t* shape, const ssize_t* stride)
{
    return cute::make_layout(
        make_cute_shape<kRank>(shape),
        make_unit_inner_stride_impl(stride, std::make_index_sequence<kRank - 1>{}));
}

}  // namespace detail

// ============================================================================
// CUDA kernel: CopyKernelND (rank-1..4 vectorized copy)
// ============================================================================
namespace kernel {
template<typename T, int kVec, typename SrcLayoutT, typename DstLayoutT>
__global__ void __launch_bounds__(256)
CopyKernelND(const T* __restrict__ src_ptr,
             T* __restrict__       dst_ptr,
             SrcLayoutT             src_layout,
             DstLayoutT             dst_layout)
{
    if constexpr (kVec * cute::sizeof_bits_v<T> <= 128)
    {
    constexpr int kBlockThreads = 256;
    constexpr int kRank         = cute::rank_v<SrcLayoutT>;
    static_assert(1 <= kRank && kRank <= 4, "CopyKernelND: rank must be 1..4");
    constexpr int kCopyThreads  = kBlockThreads / kVec;

    auto tiled_copy = cute::make_tiled_copy(
        cute::Copy_Atom<cute::UniversalCopy<cute::uint_bit_t<kVec * cute::sizeof_bits_v<T>>>, T>{},
        cute::make_layout(cute::make_shape(cute::Int<kCopyThreads>{})),
        cute::make_layout(cute::make_shape(cute::Int<kVec>{})));

    if (threadIdx.x >= kCopyThreads) return;

    auto thr_copy = tiled_copy.get_slice(threadIdx.x);

    // Bounds check on outer dimension.
    // Note: group<1,kRank> is recomputed inside the rowSrc/rowDst lambdas below.
    // This is intentional — group is a pure compile-time operation with zero
    // runtime cost. The compiler eliminates the redundancy via CSE.
    if constexpr (kRank > 1) {
        auto src_layout_g = cute::group<1, kRank>(src_layout);
        if (blockIdx.y >= cute::size(cute::get<1>(src_layout_g))) return;
    } else {
        if (blockIdx.y > 0) return;
    }

    // Obtain the 1D row tensor (for rank>1, group outer dims and slice; for rank-1, use as-is)
    auto rowSrc = [&] {
        if constexpr (kRank > 1) {
            auto src_layout_g = cute::group<1, kRank>(src_layout);
            auto gSrc_g = cute::make_tensor(cute::make_gmem_ptr(src_ptr), src_layout_g);
            return gSrc_g(_, blockIdx.y);
        } else {
            return cute::make_tensor(cute::make_gmem_ptr(src_ptr), src_layout);
        }
    }();

    auto rowDst = [&] {
        if constexpr (kRank > 1) {
            auto dst_layout_g = cute::group<1, kRank>(dst_layout);
            auto gDst_g = cute::make_tensor(cute::make_gmem_ptr(dst_ptr), dst_layout_g);
            return gDst_g(_, blockIdx.y);
        } else {
            return cute::make_tensor(cute::make_gmem_ptr(dst_ptr), dst_layout);
        }
    }();

    auto tiler    = cute::Int<kBlockThreads>{};
    auto tiledSrc = cute::zipped_divide(rowSrc, tiler);
    auto tiledDst = cute::zipped_divide(rowDst, tiler);

    if (blockIdx.x >= cute::size<1>(tiledSrc)) return;

    auto ctaSrc = tiledSrc(_, blockIdx.x);
    auto ctaDst = tiledDst(_, blockIdx.x);

    auto thrSrc = thr_copy.partition_S(ctaSrc);
    auto thrDst = thr_copy.partition_D(ctaDst);

    if constexpr (kVec > 1) {
        cute::copy(tiled_copy, thrSrc, thrDst);
    }
    else {
        auto id_row   = cute::make_identity_tensor(cute::shape(rowSrc));
        auto id_tiled = cute::zipped_divide(id_row, tiler);
        auto tile_id  = id_tiled(_, blockIdx.x);

        auto thrId = thr_copy.partition_S(tile_id);

        auto pred = cute::make_tensor<bool>(cute::shape(thrSrc));
        PRAGMA_UNROLL
        for (int i = 0; i < cute::size(pred); ++i) {
            pred(i) = cute::get<0>(thrId(i)) < cute::size(rowSrc);
        }

        cute::copy_if(pred, thrSrc, thrDst);
    }
    }  // end if constexpr (kVec * sizeof_bits_v<T> <= 128)
}

// ============================================================================
// CUDA kernel: TransposeCopyKernel (cooperative_copy 2D layout conversion)
// ============================================================================
// Copies a 2D tensor from src to dst where src and dst have orthogonal
// contiguous dimensions (src contiguous on dim 0, dst contiguous on dim 1).
// Uses smem staging with two views of a flat buffer: row-major write view
// for Phase 1, column-major read view for Phase 2 (implicit transpose).
// Both phases use kMaxVecBits (scalar = 8*sizeof(T) from host).
template<int kTileDim, uint32_t kMaxVecBits,
         typename SrcEngine, typename SrcLayout,
         typename DstEngine, typename DstLayout>
__global__ void __launch_bounds__(256)
TransposeCopyKernel(cute::Tensor<SrcEngine, SrcLayout> src,
                    cute::Tensor<DstEngine, DstLayout> dst)
{
    using T = typename SrcEngine::value_type;
    static_assert(std::is_same_v<T, typename DstEngine::value_type>,
                  "TransposeCopyKernel: src and dst value types must match");

    __shared__ T smem[kTileDim * kTileDim];

    // Write view: row-major (stride-1 on dim 0, matches src)
    auto smem_w = cute::make_tensor(cute::make_smem_ptr(smem),
        cute::make_layout(cute::make_shape(cute::Int<kTileDim>{}, cute::Int<kTileDim>{}),
                          cute::make_stride(cute::Int<1>{}, cute::Int<kTileDim>{})));

    // Read view: column-major (stride-1 on dim 0, matches dst) — transposed
    auto smem_r = cute::make_tensor(cute::make_smem_ptr(smem),
        cute::make_layout(cute::make_shape(cute::Int<kTileDim>{}, cute::Int<kTileDim>{}),
                          cute::make_stride(cute::Int<kTileDim>{}, cute::Int<1>{})));

    // Tile gmem tensors — inner (kTileDim, kTileDim) is static, outer is dynamic
    auto tiler = cute::make_shape(cute::Int<kTileDim>{}, cute::Int<kTileDim>{});
    auto src_tiled = cute::zipped_divide(src, tiler);
    auto dst_tiled = cute::zipped_divide(dst, tiler);

    // Bounds check on tile grid — zipped_divide produces ((inner0,inner1),(outer0,outer1))
    if (blockIdx.y >= cute::size<1, 0>(src_tiled) ||
        blockIdx.x >= cute::size<1, 1>(src_tiled)) return;

    // Per-CTA tile with static shape (kTileDim, kTileDim)
    auto src_tile = src_tiled(cute::_,
                              cute::make_coord(blockIdx.y, blockIdx.x));
    auto dst_tile = dst_tiled(cute::_,
                              cute::make_coord(blockIdx.y, blockIdx.x));

    // Phase 1: gmem(src) -> smem (row-major)
    cute::cooperative_copy<256, kMaxVecBits>(threadIdx.x, src_tile, smem_w);
    __syncthreads();
    // Phase 2: smem -> gmem(dst) (column-major view — transposed)
    cute::cooperative_copy<256, kMaxVecBits>(threadIdx.x, smem_r, dst_tile);
}

}  // namespace kernel

// ============================================================================
// Host function: GenericCopy (alignment-gated vectorization)
// ============================================================================
void GenericCopy(const Tensor& src, Tensor& dst, cudaStream_t stream)
{
    auto a = src.layout();
    auto b = dst.layout();

    TM_CHECK_EQ(a.size(), b.size()) << "GenericCopy: src and dst must have the same number of elements";

    // Sort strides ascending so innermost (fastest-varying) dim is first
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
    constexpr int  kBlockThreads = 256;

    // --- 2D transpose detection ---
    constexpr int kTileDim = 32;
    bool is_2d_transpose = (rank == 2) &&
        (a.stride(0) == 1) && (b.stride(1) == 1) &&
        (a.stride(1) > 1) && (b.stride(0) > 1);

    if (is_2d_transpose &&
        a.shape(0) % kTileDim == 0 && a.shape(1) % kTileDim == 0)
    {
        int64_t tr_alignment = 16;
        auto tr_data_a = src.raw_data();
        auto tr_data_b = dst.raw_data();
        tr_alignment = std::gcd(tr_alignment, reinterpret_cast<uintptr_t>(tr_data_a));
        tr_alignment = std::gcd(tr_alignment, reinterpret_cast<uintptr_t>(tr_data_b));

        int max_vec_bits = std::min(128, static_cast<int>(tr_alignment * 8));

        int32_t M = static_cast<int32_t>(a.shape(0));
        int32_t N = static_cast<int32_t>(a.shape(1));
        dim3 grid(static_cast<uint32_t>(N / kTileDim),
                  static_cast<uint32_t>(M / kTileDim));

        auto tr_dispatch_elem_size = [&](auto t) {
            using T = decltype(t);

            auto src_gmem = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<const T*>(tr_data_a)),
                cute::make_layout(cute::make_shape(M, N),
                                  cute::make_stride(cute::Int<1>{}, a.stride(1))));

            auto dst_gmem = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<T*>(tr_data_b)),
                cute::make_layout(cute::make_shape(M, N),
                                  cute::make_stride(b.stride(0), cute::Int<1>{})));

            auto tr_dispatch_vec = [&](auto v) {
                constexpr uint32_t kVB = v.value;
                using SrcE = typename decltype(src_gmem)::engine_type;
                using SrcL = typename decltype(src_gmem)::layout_type;
                using DstE = typename decltype(dst_gmem)::engine_type;
                using DstL = typename decltype(dst_gmem)::layout_type;
                kernel::TransposeCopyKernel<kTileDim, kVB, SrcE, SrcL, DstE, DstL>
                    <<<grid, 256, 0, stream>>>(src_gmem, dst_gmem);
            };

            switch (max_vec_bits) {
                case 128: tr_dispatch_vec(constant<128>{}); break;
                case 64:  tr_dispatch_vec(constant<64>{}); break;
                case 32:  tr_dispatch_vec(constant<32>{}); break;
                case 16:  tr_dispatch_vec(constant<16>{}); break;
                default:  tr_dispatch_vec(constant<8>{}); break;
            }
        };

        switch (byte_size(dtype)) {
            case 1: return tr_dispatch_elem_size(uint8_t{});
            case 2: return tr_dispatch_elem_size(uint16_t{});
            case 4: return tr_dispatch_elem_size(uint32_t{});
            default:
                TM_CHECK(0) << "GenericCopy: unsupported element size " << byte_size(dtype);
                break;
        }
    }

    // --- Alignment detection ---
    // NOTE: GenericCopy vectorizes along the innermost (stride-sorted) dimension.
    // If neither src nor dst has a stride-1 innermost dim, alignment falls to
    // byte_size(dtype) (vec_size=1), resulting in scalar copies. Vectorizing
    // along a non-contiguous dimension would require a different kernel architecture.
    int64_t alignment = 16;

    auto align = [&](auto v) { alignment = std::gcd(alignment, v); };

    // If the innermost dim is not stride-1, we can't vectorize along it
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

    // --- vec_size computation ---
    const int elem_size = byte_size(dtype);
    int vec_size = static_cast<int>(alignment / std::max<int64_t>(1, elem_size));

    // Cap at 128 bits (16 bytes) — CuTe's max
    if (vec_size * elem_size > 16) {
        vec_size = 16 / elem_size;
    }

    // Shape divisibility: shape[0] must be divisible by vec_size * kBlockThreads
    while (vec_size > 1 && a.shape(0) % (static_cast<int64_t>(vec_size) * kBlockThreads) != 0) {
        vec_size /= 2;
    }

    // --- Dispatch on data type T, vec_size kVec, and rank kRank ---
    auto dispatch_elem_size = [&](auto t) {
        using T = decltype(t);

        auto dispatch_vec = [&](auto v) {
            constexpr int kVec = v.value;

            auto invoke_nd = [&](auto d) {
                constexpr int kRank = d.value;

                int64_t inner_size = a.shape(0);
                int64_t outer_total = 1;
                for (int i = 1; i < rank; ++i) {
                    outer_total *= a.shape(i);
                }

                int64_t num_inner_tiles = (inner_size + kBlockThreads - 1) / kBlockThreads;
                dim3 grid(static_cast<uint32_t>(num_inner_tiles),
                          static_cast<uint32_t>(outer_total));

                if constexpr (kVec > 1) {
                    auto src_layout = detail::make_cute_layout_unit_inner<kRank>(a.shape().data(), a.stride().data());
                    auto dst_layout = detail::make_cute_layout_unit_inner<kRank>(a.shape().data(), b.stride().data());
                    auto func = kernel::CopyKernelND<T, kVec, decltype(src_layout), decltype(dst_layout)>;
                    func<<<grid, 256, 0, stream>>>(
                        reinterpret_cast<const T*>(data_a),
                        reinterpret_cast<T*>(data_b),
                        src_layout, dst_layout);
                }
                else {
                    auto src_layout = detail::make_cute_layout<kRank>(a.shape().data(), a.stride().data());
                    auto dst_layout = detail::make_cute_layout<kRank>(a.shape().data(), b.stride().data());
                    auto func = kernel::CopyKernelND<T, kVec, decltype(src_layout), decltype(dst_layout)>;
                    func<<<grid, 256, 0, stream>>>(
                        reinterpret_cast<const T*>(data_a),
                        reinterpret_cast<T*>(data_b),
                        src_layout, dst_layout);
                }
            };

            switch (rank) {
                case 1: invoke_nd(constant<1>{}); break;
                case 2: invoke_nd(constant<2>{}); break;
                case 3: invoke_nd(constant<3>{}); break;
                case 4: invoke_nd(constant<4>{}); break;
                default: TM_CHECK(0) << "GenericCopy: rank > 4 not implemented"; break;
            }
        };

        switch (vec_size) {
            case 16: dispatch_vec(constant<16>{}); break;
            case 8:  dispatch_vec(constant<8>{}); break;
            case 4:  dispatch_vec(constant<4>{}); break;
            case 2:  dispatch_vec(constant<2>{}); break;
            default: dispatch_vec(constant<1>{}); break;
        }
    };

    switch (byte_size(dtype)) {
        case 1: return dispatch_elem_size(uint8_t{});
        case 2: return dispatch_elem_size(uint16_t{});
        case 4: return dispatch_elem_size(uint32_t{});
        default: TM_CHECK(0) << "GenericCopy: unsupported element size " << byte_size(dtype); break;
    }
}

}  // namespace turbomind::core
