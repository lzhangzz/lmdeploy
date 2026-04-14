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

using namespace cute;

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
    return make_shape(static_cast<int32_t>(data[Is])...);
}

template<int kRank>
auto make_cute_shape(const ssize_t* data)
{
    return make_cute_shape_impl(data, std::make_index_sequence<kRank>{});
}

template<size_t... Is>
auto make_cute_stride_impl(const ssize_t* data, std::index_sequence<Is...>)
{
    return make_stride(static_cast<int64_t>(data[Is])...);
}

template<int kRank>
auto make_cute_stride(const ssize_t* data)
{
    return make_cute_stride_impl(data, std::make_index_sequence<kRank>{});
}

template<int kRank>
auto make_cute_layout(const ssize_t* shape, const ssize_t* stride)
{
    return make_layout(make_cute_shape<kRank>(shape),
                             make_cute_stride<kRank>(stride));
}

// Layout with compile-time Int<1> inner stride — needed for CuTe's recast
// in wide Copy_Atom (vectorized path). Only valid when inner stride == 1.
template<size_t... Is>
auto make_unit_inner_stride_impl(const ssize_t* stride, std::index_sequence<Is...>)
{
    return make_stride(Int<1>{}, static_cast<int64_t>(stride[Is + 1])...);
}

template<int kRank>
auto make_cute_layout_unit_inner(const ssize_t* shape, const ssize_t* stride)
{
    return make_layout(
        make_cute_shape<kRank>(shape),
        make_unit_inner_stride_impl(stride, std::make_index_sequence<kRank - 1>{}));
}

}  // namespace detail

// ============================================================================
// CUDA kernel: CopyKernelND (tutorial-pattern vectorized copy)
// ============================================================================
namespace kernel {
// Matches the copy_kernel_vectorized pattern from
// cutlass/examples/cute/tutorial/tiled_copy.cu:
//   1. Host creates and tiles tensors via tiled_divide
//   2. Kernel slices tile by blockIdx
//   3. Thread partition via TiledCopy
//   4. Register fragment + two-phase copy (gmem→fragment→gmem)
template<bool kPredicated, class TensorS, class TensorD, class TiledCopy>
__global__ void __launch_bounds__(256)
CopyKernelND(TensorS S, TensorD D, TiledCopy tiled_copy, int64_t inner_size)
{
    using namespace cute;

    constexpr int kBlockThreads = 256;

    // Excess threads (kCopyThreads < kBlockThreads when kVec > 1) exit early
    if (threadIdx.x >= size(tiled_copy)) return;

    // Bounds check on tile grid
    if (blockIdx.y >= size<2>(S) || blockIdx.x >= size<1>(S)) return;

    // Slice tile — tutorial pattern
    auto tile_S = S(_, blockIdx.x, blockIdx.y);
    auto tile_D = D(_, blockIdx.x, blockIdx.y);

    // Thread partition — tutorial pattern
    ThrCopy thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
    auto thr_S = thr_copy.partition_S(tile_S);
    auto thr_D = thr_copy.partition_D(tile_D);

    // Register fragment — tutorial pattern
    auto fragment = make_fragment_like(thr_D);

    if constexpr (!kPredicated) {
        // Vectorized: unconditional two-phase copy
        copy(tiled_copy, thr_S, fragment);
        copy(tiled_copy, fragment, thr_D);
    } else {
        // Scalar: predicated copy for partial last tile
        // Use identity tensor to find this thread's element index within the tile,
        // then check against inner_size. Avoid copy_if(TiledCopy, pred, ...)
        // because the Copy_Atom-level copy_if calls pred() expecting a scalar,
        // but our pred tensor is rank-1.
        auto id_tile = make_identity_tensor(make_shape(size<0>(tile_S)));
        auto thr_id  = thr_copy.partition_S(id_tile);

        // Each thread has 0 or 1 elements for scalar TiledCopy
        if (size(thr_id) > 0) {
            bool valid = static_cast<int64_t>(get<0>(thr_id(0)))
                         + static_cast<int64_t>(blockIdx.x) * kBlockThreads
                         < inner_size;
            if (valid) {
                copy(tiled_copy, thr_S, fragment);
                copy(tiled_copy, fragment, thr_D);
            }
        }
    }
}

// ============================================================================
// CUDA kernel: TransposeCopyKernel (TiledCopy 2D layout conversion)
// ============================================================================
// Copies a 2D tensor from src to dst where src and dst have orthogonal
// contiguous dimensions (src contiguous on dim 0, dst contiguous on dim 1).
// Uses smem staging to decouple read/write access patterns for coalesced gmem
// access in both phases. Both phases use explicit TiledCopy (scalar Copy_Atom).
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

    __shared__ T smem[kTileDim * (kTileDim + 1)];

    // Smem view: row-major — stride-1 on dim 0 (matches src contiguous dim)
    auto smem_view = make_tensor(make_smem_ptr(smem),
        make_layout(make_shape(Int<kTileDim>{}, Int<kTileDim>{}),
                          make_stride(Int<1>{}, Int<kTileDim + 1>{})));

    // Tile gmem tensors — tiled_divide produces ((TM,TN), M/TM, N/TN)
    auto tiler = make_shape(Int<kTileDim>{}, Int<kTileDim>{});
    auto src_tiled = tiled_divide(src, tiler);
    auto dst_tiled = tiled_divide(dst, tiler);

    // Bounds check on tile grid
    if (blockIdx.y >= size<1>(src_tiled) ||
        blockIdx.x >= size<2>(src_tiled)) return;

    // Per-CTA tile — make_coord(_,_) unpacks zipped inner mode to rank-2 (TM,TN)
    auto src_tile = src_tiled(make_coord(_, _), blockIdx.y, blockIdx.x);
    auto dst_tile = dst_tiled(make_coord(_, _), blockIdx.y, blockIdx.x);

    // Phase 1: gmem(src) -> smem via TiledCopy
    auto tc1 = make_tiled_copy(
        Copy_Atom<UniversalCopy<T>, T>{},
        make_layout(make_shape(Int<kTileDim>{}, Int<kTileDim / 4>{})),
        make_layout(make_shape(Int<1>{}, Int<1>{})));
    auto thr1 = tc1.get_slice(threadIdx.x);
    copy(tc1, thr1.partition_S(src_tile), thr1.partition_D(smem_view));

    __syncthreads();

    // Phase 2: smem -> gmem(dst) via TiledCopy
    auto tc2 = make_tiled_copy(
        Copy_Atom<UniversalCopy<T>, T>{},
        make_layout(make_shape(Int<kTileDim / 4>{}, Int<kTileDim>{})),
        make_layout(make_shape(Int<1>{}, Int<1>{})));
    auto thr2 = tc2.get_slice(threadIdx.x);
    copy(tc2, thr2.partition_S(smem_view), thr2.partition_D(dst_tile));
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
        auto tr_data_a = src.raw_data();
        auto tr_data_b = dst.raw_data();

        int32_t M = static_cast<int32_t>(a.shape(0));
        int32_t N = static_cast<int32_t>(a.shape(1));
        dim3 grid(static_cast<uint32_t>(N / kTileDim),
                  static_cast<uint32_t>(M / kTileDim));

        auto tr_dispatch_elem_size = [&](auto t) {
            using T = decltype(t);

            auto src_gmem = make_tensor(make_gmem_ptr(reinterpret_cast<const T*>(tr_data_a)),
                make_layout(make_shape(M, N),
                                  make_stride(Int<1>{}, a.stride(1))));

            auto dst_gmem = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(tr_data_b)),
                make_layout(make_shape(M, N),
                                  make_stride(b.stride(0), Int<1>{})));

            kernel::TransposeCopyKernel<kTileDim, 8 * sizeof(T)>
                <<<grid, 256, 0, stream>>>(src_gmem, dst_gmem);
        };

        switch (byte_size(dtype)) {
            case 1: return tr_dispatch_elem_size(uint8_t{});
            case 2: return tr_dispatch_elem_size(uint16_t{});
            case 4: return tr_dispatch_elem_size(uint32_t{});
            case 8: return tr_dispatch_elem_size(uint64_t{});
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
        constexpr int kElemBits = sizeof_bits_v<T>;

        auto dispatch_vec = [&](auto v) {
            constexpr int kVec = v.value;

            // Guard: CuTe Copy_Atom supports up to 128 bits
            if constexpr (kVec * kElemBits <= 128) {

            auto invoke_nd = [&](auto d) {
                constexpr int kRank      = d.value;
                constexpr int kBlockThreads = 256;
                constexpr int kCopyThreads  = kBlockThreads / kVec;

                // 1. Create CuTe layouts (same as before)
                auto src_layout = [&] {
                    if constexpr (kVec > 1)
                        return detail::make_cute_layout_unit_inner<kRank>(
                            a.shape().data(), a.stride().data());
                    else
                        return detail::make_cute_layout<kRank>(
                            a.shape().data(), a.stride().data());
                }();

                auto dst_layout = [&] {
                    if constexpr (kVec > 1)
                        return detail::make_cute_layout_unit_inner<kRank>(
                            a.shape().data(), b.stride().data());
                    else
                        return detail::make_cute_layout<kRank>(
                            a.shape().data(), b.stride().data());
                }();

                // 2. Wrap in CuTe gmem tensors
                auto src_gmem = make_tensor(
                    make_gmem_ptr(reinterpret_cast<const T*>(data_a)), src_layout);
                auto dst_gmem = make_tensor(
                    make_gmem_ptr(reinterpret_cast<T*>(data_b)), dst_layout);

                // 3. Group outer dims → rank-2 (inner, outer_grouped)
                //    For kRank=1, append trivial outer dim (size 1, stride 0).
                //    Use stride<0>() to get scalar inner stride (not .stride()
                //    which returns a tuple even for rank-1).
                auto src_grouped = [&] {
                    if constexpr (kRank > 1) return group_modes<1, kRank>(src_gmem);
                    else return make_tensor(src_gmem.data(),
                        make_layout(make_shape(src_gmem.size(), Int<1>{}),
                                    make_stride(stride<0>(src_gmem.layout()), Int<0>{})));
                }();
                auto dst_grouped = [&] {
                    if constexpr (kRank > 1) return group_modes<1, kRank>(dst_gmem);
                    else return make_tensor(dst_gmem.data(),
                        make_layout(make_shape(dst_gmem.size(), Int<1>{}),
                                    make_stride(stride<0>(dst_gmem.layout()), Int<0>{})));
                }();

                // 4. Construct rank-3 tiled tensor: (tile, num_tiles, outer_grouped)
                //    Manually build the layout to avoid CuTe's logical_divide
                //    flattening multi-mode tensors with scalar tilers.
                auto make_tiled_3d = [&](auto tensor_2d) {
                    auto inner_size_val   = size<0>(tensor_2d);
                    auto inner_stride_val = stride<0>(tensor_2d.layout());
                    int   num_tiles       = (inner_size_val + kBlockThreads - 1) / kBlockThreads;
                    auto outer_shape      = shape<1>(tensor_2d.layout());
                    auto outer_stride     = stride<1>(tensor_2d.layout());
                    // Tile stride = kBlockThreads * inner_stride.
                    // Int<N> * runtime doesn't compile in CuTe, so cast explicitly.
                    auto tile_num_stride  = static_cast<int64_t>(inner_stride_val) * kBlockThreads;
                    return make_tensor(tensor_2d.data(),
                        make_layout(
                            make_shape(Int<kBlockThreads>{}, num_tiles, outer_shape),
                            make_stride(inner_stride_val, tile_num_stride, outer_stride)));
                };

                auto tiled_src = make_tiled_3d(src_grouped);
                auto tiled_dst = make_tiled_3d(dst_grouped);

                // 5. Create TiledCopy
                auto tiled_copy = make_tiled_copy(
                    Copy_Atom<UniversalCopy<uint_bit_t<kVec * kElemBits>>, T>{},
                    make_layout(make_shape(Int<kCopyThreads>{})),
                    make_layout(make_shape(Int<kVec>{})));

                // 6. Grid and launch
                int64_t inner_size = static_cast<int64_t>(a.shape(0));
                dim3    grid(size<1>(tiled_src), size<2>(tiled_src));

                auto func = kernel::CopyKernelND<
                    kVec == 1,
                    decltype(tiled_src),
                    decltype(tiled_dst),
                    decltype(tiled_copy)>;

                func<<<grid, kBlockThreads, 0, stream>>>(
                    tiled_src, tiled_dst, tiled_copy, inner_size);
            };

            switch (rank) {
                case 1: invoke_nd(constant<1>{}); break;
                case 2: invoke_nd(constant<2>{}); break;
                case 3: invoke_nd(constant<3>{}); break;
                case 4: invoke_nd(constant<4>{}); break;
                default: TM_CHECK(0) << "GenericCopy: rank > 4 not implemented"; break;
            }

            }  // end if constexpr (kVec * kElemBits <= 128)
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
        case 8: return dispatch_elem_size(uint64_t{});
        default: TM_CHECK(0) << "GenericCopy: unsupported element size " << byte_size(dtype); break;
    }
}

}  // namespace turbomind::core
