#include "src/turbomind/core/tensor.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/core/meta.h"

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>

#include <algorithm>
#include <numeric>
#include <string>

namespace turbomind::core {

using cute::_;

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

// Layout with compile-time Int<1> inner stride — needed for CuTe's recast
// in wide Copy_Atom (vectorized path). Only valid when inner stride == 1.
template<int kRank>
auto make_cute_layout_unit_inner(const ssize_t* shape, const ssize_t* stride)
{
    if constexpr (kRank == 1) {
        return cute::make_layout(
            cute::make_shape(static_cast<int32_t>(shape[0])),
            cute::make_stride(cute::Int<1>{}));
    }
    else if constexpr (kRank == 2) {
        return cute::make_layout(
            cute::make_shape(static_cast<int32_t>(shape[0]),
                             static_cast<int32_t>(shape[1])),
            cute::make_stride(cute::Int<1>{},
                              static_cast<int64_t>(stride[1])));
    }
    else if constexpr (kRank == 3) {
        return cute::make_layout(
            cute::make_shape(static_cast<int32_t>(shape[0]),
                             static_cast<int32_t>(shape[1]),
                             static_cast<int32_t>(shape[2])),
            cute::make_stride(cute::Int<1>{},
                              static_cast<int64_t>(stride[1]),
                              static_cast<int64_t>(stride[2])));
    }
    else if constexpr (kRank == 4) {
        return cute::make_layout(
            cute::make_shape(static_cast<int32_t>(shape[0]),
                             static_cast<int32_t>(shape[1]),
                             static_cast<int32_t>(shape[2]),
                             static_cast<int32_t>(shape[3])),
            cute::make_stride(cute::Int<1>{},
                              static_cast<int64_t>(stride[1]),
                              static_cast<int64_t>(stride[2]),
                              static_cast<int64_t>(stride[3])));
    }
    else if constexpr (kRank == 5) {
        return cute::make_layout(
            cute::make_shape(static_cast<int32_t>(shape[0]),
                             static_cast<int32_t>(shape[1]),
                             static_cast<int32_t>(shape[2]),
                             static_cast<int32_t>(shape[3]),
                             static_cast<int32_t>(shape[4])),
            cute::make_stride(cute::Int<1>{},
                              static_cast<int64_t>(stride[1]),
                              static_cast<int64_t>(stride[2]),
                              static_cast<int64_t>(stride[3]),
                              static_cast<int64_t>(stride[4])));
    }
    else if constexpr (kRank == 6) {
        return cute::make_layout(
            cute::make_shape(static_cast<int32_t>(shape[0]),
                             static_cast<int32_t>(shape[1]),
                             static_cast<int32_t>(shape[2]),
                             static_cast<int32_t>(shape[3]),
                             static_cast<int32_t>(shape[4]),
                             static_cast<int32_t>(shape[5])),
            cute::make_stride(cute::Int<1>{},
                              static_cast<int64_t>(stride[1]),
                              static_cast<int64_t>(stride[2]),
                              static_cast<int64_t>(stride[3]),
                              static_cast<int64_t>(stride[4]),
                              static_cast<int64_t>(stride[5])));
    }
}

}  // namespace detail

// ============================================================================
// CUDA kernel: GenericCopyKernel (CuTe layout algebra + vectorization)
// ============================================================================
namespace kernel {

template<typename T, int kVec, typename SrcLayoutT, typename DstLayoutT>
__global__ void __launch_bounds__(256)
GenericCopyKernel(const T* __restrict__ src_ptr,
                  T* __restrict__       dst_ptr,
                  SrcLayoutT             src_layout,
                  DstLayoutT             dst_layout)
{
    // Guard: kVec * sizeof(T) must not exceed 16 bytes (128 bits).
    // Invalid combos (e.g. kVec=16 with int32) are never dispatched at runtime,
    // but the compiler instantiates all template combos. Discard the body.
    if constexpr (kVec * cute::sizeof_bits_v<T> <= 128)
    {
    constexpr int kBlockThreads = 256;
    constexpr int kRank         = cute::rank_v<SrcLayoutT>;
    constexpr int kCopyThreads  = kBlockThreads / kVec;

    // 1. Create CuTe tensors from pointers + layouts
    auto gSrc = cute::make_tensor(cute::make_gmem_ptr(src_ptr), src_layout);
    auto gDst = cute::make_tensor(cute::make_gmem_ptr(dst_ptr), dst_layout);

    // Build TiledCopy: kCopyThreads threads, kVec elements per thread
    auto tiled_copy = cute::make_tiled_copy(
        cute::Copy_Atom<cute::UniversalCopy<cute::uint_bit_t<kVec * cute::sizeof_bits_v<T>>>, T>{},
        cute::make_layout(cute::make_shape(cute::Int<kCopyThreads>{})),
        cute::make_layout(cute::make_shape(cute::Int<kVec>{})));

    // Threads beyond kCopyThreads must not access the TiledCopy slice
    if (threadIdx.x >= kCopyThreads) return;

    auto thr_copy = tiled_copy.get_slice(threadIdx.x);

    if constexpr (kRank == 1) {
        // No outer dims — the whole tensor is the inner dim
        if (blockIdx.y > 0) return;

        // 2. Tile the inner dim with zipped_divide
        auto tiler    = cute::Int<kBlockThreads>{};
        auto tiledSrc = cute::zipped_divide(gSrc, tiler);
        auto tiledDst = cute::zipped_divide(gDst, tiler);

        if (blockIdx.x >= cute::size<1>(tiledSrc)) return;

        auto ctaSrc = tiledSrc(_, blockIdx.x);
        auto ctaDst = tiledDst(_, blockIdx.x);

        auto thrSrc = thr_copy.partition_S(ctaSrc);
        auto thrDst = thr_copy.partition_D(ctaDst);

        if constexpr (kVec > 1) {
            // Full tiles guaranteed by host — no predication needed
            cute::copy(tiled_copy, thrSrc, thrDst);
        }
        else {
            // Scalar path with identity tensor predication
            auto id_row   = cute::make_identity_tensor(cute::shape(gSrc));
            auto id_tiled = cute::zipped_divide(id_row, tiler);
            auto tile_id  = id_tiled(_, blockIdx.x);

            auto thrId = thr_copy.partition_S(tile_id);

            auto pred = cute::make_tensor<bool>(cute::shape(thrSrc));
            PRAGMA_UNROLL
            for (int i = 0; i < cute::size(pred); ++i) {
                pred(i) = cute::get<0>(thrId(i)) < cute::size(gSrc);
            }

            cute::copy_if(pred, thrSrc, thrDst);
        }
    }
    else {
        // 2. Group outer dims (modes 1..kRank-1) into a single mode
        //    rank-k layout -> rank-2 (inner, outer_flat)
        auto src_layout_g = cute::group<1, kRank>(src_layout);
        auto dst_layout_g = cute::group<1, kRank>(dst_layout);
        auto gSrc_g = cute::make_tensor(cute::make_gmem_ptr(src_ptr), src_layout_g);
        auto gDst_g = cute::make_tensor(cute::make_gmem_ptr(dst_ptr), dst_layout_g);

        if (blockIdx.y >= cute::size<1>(gSrc_g)) return;

        // 3. Slice to get this CTA's 1D inner tensor
        auto rowSrc = gSrc_g(_, blockIdx.y);
        auto rowDst = gDst_g(_, blockIdx.y);

        // 4. Tile the inner dim with zipped_divide
        auto tiler    = cute::Int<kBlockThreads>{};
        auto tiledSrc = cute::zipped_divide(rowSrc, tiler);
        auto tiledDst = cute::zipped_divide(rowDst, tiler);

        if (blockIdx.x >= cute::size<1>(tiledSrc)) return;

        auto ctaSrc = tiledSrc(_, blockIdx.x);
        auto ctaDst = tiledDst(_, blockIdx.x);

        auto thrSrc = thr_copy.partition_S(ctaSrc);
        auto thrDst = thr_copy.partition_D(ctaDst);

        if constexpr (kVec > 1) {
            // Full tiles guaranteed by host — no predication needed
            cute::copy(tiled_copy, thrSrc, thrDst);
        }
        else {
            // Scalar path with identity tensor predication
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
    }
    }  // end if constexpr (kVec * sizeof_bits_v<T> <= 128)
}

// ============================================================================
// CUDA kernel: TransposeCopyKernel (SMEM tiled 2D transpose)
// ============================================================================
template<typename T, int kVec, int kTileDim>
__global__ void __launch_bounds__(256)
TransposeCopyKernel(const T* __restrict__ src_ptr,
                    T* __restrict__       dst_ptr,
                    int64_t               src_stride_outer,
                    int64_t               dst_stride_outer,
                    int32_t               M,
                    int32_t               N)
{
    // Guard: kVec * sizeof(T) must not exceed 16 bytes (128 bits).
    // Also require kVec >= 2 — kVec=1 would need 1024 threads (kTileDim*kTileDim)
    // but we only launch 256, so the template body must not be instantiated.
    if constexpr (kVec * cute::sizeof_bits_v<T> <= 128 && kVec >= 2)
    {
    // Thread bounds: for kTileDim=32, kVec=4: 8*32=256 (all threads).
    // For kVec=8: 4*32=128 (half threads). Compute min across both phases.
    constexpr int kThrLoad  = (kTileDim / kVec) * kTileDim;
    constexpr int kThrStore = kTileDim * (kTileDim / kVec);
    constexpr int kCopyThreads = kThrLoad < kThrStore ? kThrLoad : kThrStore;
    if (threadIdx.x >= kCopyThreads) return;

    // --- Shared memory with padding for bank conflict avoidance ---
    __shared__ T smem[kTileDim * (kTileDim + 1)];

    // Single smem view: row-major padded layout.
    // Phase 1 writes through this view (coalesced along mode 0).
    // Phase 2 reads through this SAME view (NOT a transposed view — we
    // want smem(i,j) for both phases, not smem(j,i)).
    auto smem_w = cute::make_tensor(cute::make_smem_ptr(smem),
        cute::make_layout(cute::make_shape(cute::Int<kTileDim>{}, cute::Int<kTileDim>{}),
                          cute::make_stride(cute::Int<1>{}, cute::Int<kTileDim + 1>{})));

    // --- Tile coordinates ---
    int m0 = blockIdx.y * kTileDim;
    int n0 = blockIdx.x * kTileDim;

    // --- Gmem tile tensors with Int<1> on contiguous modes ---
    // Src tile: contiguous along mode 0 (Int<1> stride)
    auto src_tile = cute::make_tensor(cute::make_gmem_ptr(src_ptr + m0 + n0 * src_stride_outer),
        cute::make_layout(cute::make_shape(cute::Int<kTileDim>{}, cute::Int<kTileDim>{}),
                          cute::make_stride(cute::Int<1>{}, src_stride_outer)));

    // Dst tile: contiguous along mode 1 (Int<1> stride).
    // Same logical position (m0,n0): dst[m0+i, n0+j] = dst_ptr + (m0+i)*dst_stride_outer + (n0+j)
    auto dst_tile = cute::make_tensor(cute::make_gmem_ptr(dst_ptr + n0 + m0 * dst_stride_outer),
        cute::make_layout(cute::make_shape(cute::Int<kTileDim>{}, cute::Int<kTileDim>{}),
                          cute::make_stride(dst_stride_outer, cute::Int<1>{})));

    // --- Phase 1: gmem(src) → registers → smem ---
    // Thread layout: (kTileDim/kVec, kTileDim) — vectorize along mode 0
    auto load_copy = cute::make_tiled_copy(
        cute::Copy_Atom<cute::UniversalCopy<cute::uint_bit_t<kVec * cute::sizeof_bits_v<T>>>, T>{},
        cute::make_layout(cute::make_shape(cute::Int<kTileDim / kVec>{},
                                           cute::Int<kTileDim>{})),
        cute::make_layout(cute::make_shape(cute::Int<kVec>{}, cute::Int<1>{})));

    auto thr_load = load_copy.get_slice(threadIdx.x);
    auto thr_src  = thr_load.partition_S(src_tile);
    auto thr_smw  = thr_load.partition_D(smem_w);
    auto rmem_ld  = cute::make_fragment_like(thr_smw);

    cute::copy(load_copy, thr_src, rmem_ld);    // vectorized gmem → registers

    // Manual rmem → smem transfer (avoids CuTe auto-vectorization on smem)
    CUTE_UNROLL
    for (int i = 0; i < cute::size(thr_smw); ++i) {
        thr_smw(i) = rmem_ld(i);
    }

    __syncthreads();

    // --- Phase 2: smem → registers → gmem(dst) ---
    // Thread layout: (kTileDim, kTileDim/kVec) — vectorize along mode 1
    auto store_copy = cute::make_tiled_copy(
        cute::Copy_Atom<cute::UniversalCopy<cute::uint_bit_t<kVec * cute::sizeof_bits_v<T>>>, T>{},
        cute::make_layout(cute::make_shape(cute::Int<kTileDim>{},
                                           cute::Int<kTileDim / kVec>{})),
        cute::make_layout(cute::make_shape(cute::Int<1>{}, cute::Int<kVec>{})));

    auto thr_store = store_copy.get_slice(threadIdx.x);
    auto thr_smw2  = thr_store.partition_S(smem_w);   // same view, NOT transposed
    auto thr_dst   = thr_store.partition_D(dst_tile);
    auto rmem_st   = cute::make_fragment_like(thr_smw2);

    // Manual smem → rmem transfer (avoids CuTe auto-vectorization on smem)
    CUTE_UNROLL
    for (int i = 0; i < cute::size(thr_smw2); ++i) {
        rmem_st(i) = thr_smw2(i);
    }

    cute::copy(store_copy, rmem_st, thr_dst);    // vectorized registers → gmem

    }  // end if constexpr (kVec * sizeof_bits_v<T> <= 128)
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
        // NOTE: The transpose kernel requires kVec >= 2 because kTileDim=32 with
        // kVec=1 would need 1024 threads (32*32) but we only have 256.
        // If pointer alignment gives vec_size=1, fall through to scalar GenericCopy.

        // Transpose alignment: only pointer alignment matters
        int64_t tr_alignment = 16;
        auto tr_data_a = src.raw_data();
        auto tr_data_b = dst.raw_data();
        tr_alignment = std::gcd(tr_alignment, reinterpret_cast<uintptr_t>(tr_data_a));
        tr_alignment = std::gcd(tr_alignment, reinterpret_cast<uintptr_t>(tr_data_b));

        const int tr_elem_size = byte_size(dtype);
        int tr_vec_size = static_cast<int>(tr_alignment / std::max<int64_t>(1, tr_elem_size));

        // Cap at 128 bits (16 bytes)
        if (tr_vec_size * tr_elem_size > 16) {
            tr_vec_size = 16 / tr_elem_size;
        }

        // kTileDim must be divisible by vec_size for TiledCopy thread layout
        while (tr_vec_size > 1 && kTileDim % tr_vec_size != 0) {
            tr_vec_size /= 2;
        }

        // Require vec_size >= 2 for transpose kernel (otherwise fall through to scalar)
        if (tr_vec_size >= 2) {
            int32_t M = static_cast<int32_t>(a.shape(0));
            int32_t N = static_cast<int32_t>(a.shape(1));
            dim3 grid(static_cast<uint32_t>(N / kTileDim),
                      static_cast<uint32_t>(M / kTileDim));

            auto tr_dispatch_dtype = [&](auto t) {
                using T = decltype(t);

                auto tr_dispatch_vec = [&](auto v) {
                    constexpr int kVec = v.value;
                    auto func = kernel::TransposeCopyKernel<T, kVec, kTileDim>;
                    func<<<grid, 256, 0, stream>>>(
                        reinterpret_cast<const T*>(tr_data_a),
                        reinterpret_cast<T*>(tr_data_b),
                        a.stride(1), b.stride(0),
                        M, N);
                };

                switch (tr_vec_size) {
                    case 16: tr_dispatch_vec(constant<16>{}); break;
                    case 8:  tr_dispatch_vec(constant<8>{}); break;
                    case 4:  tr_dispatch_vec(constant<4>{}); break;
                    case 2:  tr_dispatch_vec(constant<2>{}); break;
                    default: tr_dispatch_vec(constant<1>{}); break;
                }
            };

            switch (dtype) {
                case DataType::kFloat32:  return tr_dispatch_dtype(float{});
                case DataType::kFloat16:  return tr_dispatch_dtype(half_t{});
                case DataType::kBfloat16: return tr_dispatch_dtype(bfloat16_t{});
                case DataType::kInt8:     return tr_dispatch_dtype(int8_t{});
                case DataType::kInt32:    return tr_dispatch_dtype(int32_t{});
                case DataType::kBool:     return tr_dispatch_dtype(uint8_t{});
                case DataType::kUint8:    return tr_dispatch_dtype(uint8_t{});
                default:
                    throw std::runtime_error(std::string("GenericCopy: unsupported data type ") + to_string(dtype));
            }
        }
    }

    // --- Alignment detection ---
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
    auto dispatch_dtype = [&](auto t) {
        using T = decltype(t);

        auto dispatch_vec = [&](auto v) {
            constexpr int kVec = v.value;

            auto invoke = [&](auto d) {
                constexpr int kRank = d.value;

                // Compute 2D grid: (num_inner_tiles, outer_total)
                // inner_size is NOT divided by kVec — the layout describes
                // individual elements, val_layout handles vector grouping.
                int64_t inner_size = a.shape(0);
                int64_t outer_total = 1;
                for (int i = 1; i < rank; ++i) {
                    outer_total *= a.shape(i);
                }

                int64_t num_inner_tiles = (inner_size + kBlockThreads - 1) / kBlockThreads;
                dim3 grid(static_cast<uint32_t>(num_inner_tiles),
                          static_cast<uint32_t>(outer_total));

                // Layout types differ between vectorized (Int<1> inner stride)
                // and scalar (dynamic strides), so dispatch in two branches.
                if constexpr (kVec > 1) {
                    // No shape adjustment — val_layout groups kVec elements per thread.
                    // Just use compile-time Int<1> inner stride for CuTe recast.
                    auto src_layout = detail::make_cute_layout_unit_inner<kRank>(a.shape().data(), a.stride().data());
                    auto dst_layout = detail::make_cute_layout_unit_inner<kRank>(a.shape().data(), b.stride().data());

                    auto func = kernel::GenericCopyKernel<T, kVec, decltype(src_layout), decltype(dst_layout)>;
                    func<<<grid, 256, 0, stream>>>(
                        reinterpret_cast<const T*>(data_a),
                        reinterpret_cast<T*>(data_b),
                        src_layout,
                        dst_layout);
                }
                else {
                    auto src_layout = detail::make_cute_layout<kRank>(a.shape().data(), a.stride().data());
                    auto dst_layout = detail::make_cute_layout<kRank>(a.shape().data(), b.stride().data());

                    auto func = kernel::GenericCopyKernel<T, kVec, decltype(src_layout), decltype(dst_layout)>;
                    func<<<grid, 256, 0, stream>>>(
                        reinterpret_cast<const T*>(data_a),
                        reinterpret_cast<T*>(data_b),
                        src_layout,
                        dst_layout);
                }
            };

            // Dispatch on exact rank (1-6)
            switch (rank) {
                case 1: invoke(constant<1>{}); break;
                case 2: invoke(constant<2>{}); break;
                case 3: invoke(constant<3>{}); break;
                case 4: invoke(constant<4>{}); break;
                case 5: invoke(constant<5>{}); break;
                case 6: invoke(constant<6>{}); break;
                default: throw std::runtime_error("GenericCopy: rank > 6 not implemented");
            }
        };

        // Dispatch on vec_size (powers of 2, max 128 bits)
        switch (vec_size) {
            case 16: dispatch_vec(constant<16>{}); break;
            case 8:  dispatch_vec(constant<8>{}); break;
            case 4:  dispatch_vec(constant<4>{}); break;
            case 2:  dispatch_vec(constant<2>{}); break;
            default: dispatch_vec(constant<1>{}); break;
        }
    };

    // Dispatch on data type
    switch (dtype) {
        case DataType::kFloat32:  return dispatch_dtype(float{});
        case DataType::kFloat16:  return dispatch_dtype(half_t{});
        case DataType::kBfloat16: return dispatch_dtype(bfloat16_t{});
        case DataType::kInt8:     return dispatch_dtype(int8_t{});
        case DataType::kInt32:    return dispatch_dtype(int32_t{});
        case DataType::kBool:     return dispatch_dtype(uint8_t{});
        case DataType::kUint8:    return dispatch_dtype(uint8_t{});
        default:
            throw std::runtime_error(std::string("GenericCopy: unsupported data type ") + to_string(dtype));
    }
}

}  // namespace turbomind::core
