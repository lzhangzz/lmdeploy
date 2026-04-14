#include "src/turbomind/core/tensor.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/core/logger.h"
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>

#include <algorithm>
#include <array>
#include <numeric>
#include <string>
#include <utility>

namespace turbomind::core {

using namespace cute;

// CuTe's make_shape/make_stride require compile-time variadic template args,
// but our tensor shapes and strides are runtime values. These helpers bridge
// that gap via std::index_sequence expansion, producing CuTe tuple types
// from runtime shape/stride arrays.
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


// Construct vec_factors tuple: (kVec, 1, 1, ...) — used for element coord scaling.
// Dim 0 (innermost) scales by kVec; all other dims scale by 1.
template<int kVec, int kRank, size_t... Is>
auto make_vec_factors_impl(std::index_sequence<Is...>)
{
    return make_shape((Is == 0 ? Int<kVec>{} : Int<1>{})...);
}

template<int kVec, int kRank>
auto make_vec_factors()
{
    return make_vec_factors_impl<kVec, kRank>(std::make_index_sequence<kRank>{});
}

// Compute thread partition: (T0, T1, ..., Tk-1) where T0*...*Tk-1 = 256.
// T0 is the largest power-of-2 <= shape[0]/kVec.
// Remaining threads are distributed across outer dims.
template<int kRank>
auto compute_thr_partition(const ssize_t* shape, int kVec)
    -> std::array<ssize_t, kRank>
{
    std::array<ssize_t, kRank> partition{};
    partition.fill(1);

    // Inner dim: largest power-of-2 that divides 256 and <= shape[0]/kVec
    int64_t max_inner = shape[0] / kVec;
    ssize_t T0 = 256;
    while (T0 > 1 && T0 > max_inner) {
        T0 /= 2;
    }
    partition[0] = T0;

    // Distribute remaining threads across outer dims
    ssize_t remaining = 256 / T0;
    for (int i = 1; i < kRank; ++i) {
        partition[i] = std::min<ssize_t>(shape[i], remaining);
        remaining /= partition[i];
        if (remaining < 1) {
            remaining = 1;
        }
    }
    return partition;
}

}  // namespace detail

// ============================================================================
// CUDA kernel: CopyKernelND (full-utilization manual-tiling copy)
// ============================================================================
// All 256 threads participate. Thread-to-element mapping is done manually
// using CuTe tuple operations (idx2crd, crd2idx, transform).
// Copy_Atom is applied to per-thread rank-1 (kVec,) gmem tensors.
namespace kernel {

template<int kVec, class DataShape, class SrcStride, class DstStride,
         class ThrPartition, class TileCounts, class VecFactors, typename T>
__global__ void __launch_bounds__(256)
CopyKernelND(const T* __restrict__ src, T* __restrict__ dst,
             DataShape data_shape,
             SrcStride src_strides, DstStride dst_strides,
             ThrPartition thr_partition, TileCounts tile_counts,
             VecFactors vec_factors)
{
    using namespace cute;

    // 1. Decode threadIdx -> per-dim thread coordinate (colexicographic)
    auto thr_coord = idx2crd(threadIdx.x, thr_partition);

    // 2. Decode blockIdx -> per-dim tile coordinate
    auto tile_coord = idx2crd(int64_t(blockIdx.x), tile_counts);

    // 3. Compute element coordinate
    //    inner_coord[i] = tile_coord[i] * thr_partition[i] + thr_coord[i]
    //    elem_coord[i]  = inner_coord[i] * vec_factors[i]
    auto inner_coord = transform(tile_coord, thr_coord, thr_partition,
        [](auto tc, auto thr, auto tp) { return tc * tp + thr; });
    auto elem_coord = transform(inner_coord, vec_factors,
        [](auto ic, auto vf) { return ic * vf; });

    // 4. Bounds check: elem[i] + vec_factor[i] <= shape[i]
    bool valid = true;
    for_each(transform(elem_coord, vec_factors, data_shape,
        [](auto ec, auto vf, auto s) { return ec + vf <= s; }),
        [&](auto v) { valid = valid && static_cast<bool>(v); });
    if (!valid) return;

    // 5. Compute memory offsets via CuTe's crd2idx
    int64_t src_off = crd2idx(elem_coord, data_shape, src_strides);
    int64_t dst_off = crd2idx(elem_coord, data_shape, dst_strides);

    // 6. Per-thread vectorized copy via Copy_Atom
    auto src_frag = make_tensor(make_gmem_ptr(src + src_off),
        make_layout(make_shape(Int<kVec>{}), make_stride(Int<1>{})));
    auto dst_frag = make_tensor(make_gmem_ptr(dst + dst_off),
        make_layout(make_shape(Int<kVec>{}), make_stride(Int<1>{})));
    copy(Copy_Atom<UniversalCopy<uint_bit_t<kVec * sizeof_bits_v<T>>>, T>{},
         src_frag, dst_frag);
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
// TransposeCopy: 2D transpose via smem-staged TiledCopy
// ============================================================================
static void TransposeCopy(const void* data_a, void* data_b,
                          const Layout& a, const Layout& b,
                          DataType dtype, cudaStream_t stream)
{
    constexpr int kTileDim = 32;

    int32_t M = static_cast<int32_t>(a.shape(0));
    int32_t N = static_cast<int32_t>(a.shape(1));
    dim3 grid(static_cast<uint32_t>(N / kTileDim),
              static_cast<uint32_t>(M / kTileDim));

    auto dispatch = [&](auto t) {
        using T = decltype(t);

        auto src_gmem = make_tensor(make_gmem_ptr(reinterpret_cast<const T*>(data_a)),
            make_layout(make_shape(M, N),
                              make_stride(Int<1>{}, a.stride(1))));

        auto dst_gmem = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(data_b)),
            make_layout(make_shape(M, N),
                              make_stride(b.stride(0), Int<1>{})));

        kernel::TransposeCopyKernel<kTileDim, 8 * sizeof(T)>
            <<<grid, 256, 0, stream>>>(src_gmem, dst_gmem);
    };

    switch (byte_size(dtype)) {
        case 1: return dispatch(uint8_t{});
        case 2: return dispatch(uint16_t{});
        case 4: return dispatch(uint32_t{});
        case 8: return dispatch(uint64_t{});
        default:
            TM_CHECK(0) << "TransposeCopy: unsupported element size " << byte_size(dtype);
            break;
    }
}

// ============================================================================
// VectorizedCopy: alignment-gated vectorized ND copy via CopyKernelND
// ============================================================================
static void VectorizedCopy(const void* data_a, void* data_b,
                           const Layout& a, const Layout& b,
                           int rank, DataType dtype, cudaStream_t stream)
{
    constexpr int kBlockThreads = 256;

    // --- Alignment detection ---
    int64_t alignment = 16;

    auto align = [&](auto v) { alignment = std::gcd(alignment, v); };

    if (a.stride(0) > 1 || b.stride(0) > 1) {
        alignment = byte_size(dtype);
    }

    align(byte_size(dtype, a.shape(0)));
    align(reinterpret_cast<uintptr_t>(data_a));
    align(reinterpret_cast<uintptr_t>(data_b));

    for (int i = 1; i < rank; ++i) {
        align(byte_size(dtype, a.stride(i)));
        align(byte_size(dtype, b.stride(i)));
    }

    // --- vec_size computation ---
    const int elem_size = byte_size(dtype);
    int vec_size = static_cast<int>(alignment / std::max<int64_t>(1, elem_size));

    if (vec_size * elem_size > 16) {
        vec_size = 16 / elem_size;
    }

    while (vec_size > 1 && a.shape(0) % static_cast<int64_t>(vec_size) != 0) {
        vec_size /= 2;
    }

    // --- Dispatch on data type T and vec_size kVec ---
    auto dispatch_elem_size = [&](auto t) {
        using T = decltype(t);
        constexpr int kElemBits = sizeof_bits_v<T>;
        constexpr int kMaxVec = 128 / kElemBits;

        auto dispatch_vec = [&](auto v) {
            constexpr int kVec = v.value;

            auto dispatch_rank = [&](auto d) {
                constexpr int kRank = d.value;

                auto data_shape  = detail::make_cute_shape<kRank>(a.shape().data());
                auto src_strides = detail::make_cute_stride<kRank>(a.stride().data());
                auto dst_strides = detail::make_cute_stride<kRank>(b.stride().data());

                auto partition_arr = detail::compute_thr_partition<kRank>(
                    a.shape().data(), kVec);
                auto thr_partition = detail::make_cute_shape<kRank>(
                    partition_arr.data());

                auto vec_factors = detail::make_vec_factors<kVec, kRank>();
                auto tile_sizes  = transform(thr_partition, vec_factors,
                    [](auto tp, auto vf) { return tp * vf; });
                auto tile_counts = transform(data_shape, tile_sizes,
                    [](auto s, auto ts) -> int64_t {
                        return (static_cast<int64_t>(s) + static_cast<int64_t>(ts) - 1)
                               / static_cast<int64_t>(ts);
                    });

                int64_t total_tiles = product(tile_counts);
                dim3 grid(static_cast<uint32_t>(total_tiles));

                kernel::CopyKernelND<kVec>
                    <<<grid, kBlockThreads, 0, stream>>>(
                        reinterpret_cast<const T*>(data_a),
                        reinterpret_cast<T*>(data_b),
                        data_shape, src_strides, dst_strides,
                        thr_partition, tile_counts, vec_factors);
            };

            switch (rank) {
                case 1: dispatch_rank(constant<1>{}); break;
                case 2: dispatch_rank(constant<2>{}); break;
                case 3: dispatch_rank(constant<3>{}); break;
                case 4: dispatch_rank(constant<4>{}); break;
                default: TM_CHECK(0) << "VectorizedCopy: rank > 4 not implemented"; break;
            }
        };

        switch (vec_size) {
            case 16: if constexpr (16 <= kMaxVec) dispatch_vec(constant<16>{}); break;
            case 8:  if constexpr (8  <= kMaxVec) dispatch_vec(constant<8>{});  break;
            case 4:  if constexpr (4  <= kMaxVec) dispatch_vec(constant<4>{});  break;
            case 2:  if constexpr (2  <= kMaxVec) dispatch_vec(constant<2>{});  break;
            default: dispatch_vec(constant<1>{}); break;
        }
    };

    switch (byte_size(dtype)) {
        case 1: return dispatch_elem_size(uint8_t{});
        case 2: return dispatch_elem_size(uint16_t{});
        case 4: return dispatch_elem_size(uint32_t{});
        case 8: return dispatch_elem_size(uint64_t{});
        default:
            TM_CHECK(0) << "VectorizedCopy: unsupported element size " << byte_size(dtype);
            break;
    }
}

// ============================================================================
// GenericCopy: layout normalization + dispatch
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

    // --- 2D transpose detection ---
    constexpr int kTileDim = 32;
    bool is_2d_transpose = (rank == 2) &&
        (a.stride(0) == 1) && (b.stride(1) == 1) &&
        (a.stride(1) > 1) && (b.stride(0) > 1);

    if (is_2d_transpose &&
        a.shape(0) % kTileDim == 0 && a.shape(1) % kTileDim == 0)
    {
        TransposeCopy(src.raw_data(), dst.raw_data(), a, b, dtype, stream);
        return;
    }

    // --- Vectorized / scalar copy ---
    VectorizedCopy(src.raw_data(), dst.raw_data(), a, b, rank, dtype, stream);
}

}  // namespace turbomind::core
