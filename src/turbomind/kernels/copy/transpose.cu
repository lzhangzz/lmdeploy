#include "src/turbomind/kernels/copy/copy.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/logger.h"
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>

namespace turbomind::core {

using namespace cute;

// ============================================================================
// CUDA kernel: TransposeCopyKernel (vectorized 3-phase smem-staged transpose)
// ============================================================================
namespace kernel {

extern __shared__ char smem_buf[];

template<int kTileDim, int kVec,
         typename SrcEngine, typename SrcLayout,
         typename DstEngine, typename DstLayout>
__global__ void __launch_bounds__(256)
TransposeCopyKernel(cute::Tensor<SrcEngine, SrcLayout> src,
                    cute::Tensor<DstEngine, DstLayout> dst)
{
    using T = typename SrcEngine::value_type;
    static_assert(std::is_same_v<T, typename DstEngine::value_type>,
                  "TransposeCopyKernel: src and dst value types must match");

    constexpr int kPad      = kVec;
    constexpr int kStride   = kTileDim + kPad;
    constexpr int kThrRows  = 256 / kTileDim;  // threads along dim 0 in phase 1
    using VecT = uint_bit_t<kVec * sizeof_bits_v<T>>;

    T* smem_base = reinterpret_cast<T*>(smem_buf);

    // Smem1: row-major — contiguous dim 0 (for phase 1 vectorization)
    auto smem1 = make_tensor(make_smem_ptr(smem_base),
        make_layout(make_shape(Int<kTileDim>{}, Int<kTileDim>{}),
                          make_stride(Int<1>{}, Int<kStride>{})));

    // Smem2: col-major — contiguous dim 1 (for phase 2 vectorization)
    auto smem2 = make_tensor(make_smem_ptr(smem_base + kTileDim * kStride),
        make_layout(make_shape(Int<kTileDim>{}, Int<kTileDim>{}),
                          make_stride(Int<kStride>{}, Int<1>{})));

    // Tile gmem tensors
    auto tiler = make_shape(Int<kTileDim>{}, Int<kTileDim>{});
    auto src_tiled = tiled_divide(src, tiler);
    auto dst_tiled = tiled_divide(dst, tiler);

    // Bounds check on tile grid
    if (blockIdx.y >= size<1>(src_tiled) ||
        blockIdx.x >= size<2>(src_tiled)) return;

    auto src_tile = src_tiled(make_coord(_, _), blockIdx.y, blockIdx.x);
    auto dst_tile = dst_tiled(make_coord(_, _), blockIdx.y, blockIdx.x);

    // Phase 1: gmem(src) -> smem1, vectorize along dim 0
    auto tc1 = make_tiled_copy(
        Copy_Atom<UniversalCopy<VecT>, T>{},
        make_layout(make_shape(Int<kThrRows>{}, Int<kTileDim>{})),
        make_layout(make_shape(Int<kVec>{}, Int<1>{})));
    auto thr1 = tc1.get_slice(threadIdx.x);
    copy(tc1, thr1.partition_S(src_tile), thr1.partition_D(smem1));

    __syncthreads();

    // In-smem: smem1 -> smem2 (physical layout conversion, same logical data)
    auto tc_s = make_tiled_copy(
        Copy_Atom<UniversalCopy<T>, T>{},
        make_layout(make_shape(Int<16>{}, Int<16>{})),
        make_layout(make_shape(Int<1>{}, Int<1>{})));
    auto thr_s = tc_s.get_slice(threadIdx.x);
    copy(tc_s, thr_s.partition_S(smem1), thr_s.partition_D(smem2));

    __syncthreads();

    // Phase 2: smem2 -> gmem(dst), vectorize along dim 1
    auto tc2 = make_tiled_copy(
        Copy_Atom<UniversalCopy<VecT>, T>{},
        make_layout(make_shape(Int<kTileDim>{}, Int<kThrRows>{}),
                       make_stride(Int<kThrRows>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<kVec>{})));
    auto thr2 = tc2.get_slice(threadIdx.x);
    copy(tc2, thr2.partition_S(smem2), thr2.partition_D(dst_tile));
}

}  // namespace kernel

// ============================================================================
// TransposeCopy: 2D transpose via vectorized smem-staged TiledCopy
// ============================================================================
void TransposeCopy(const void* data_a, void* data_b,
                   const Layout& a, const Layout& b,
                   DataType dtype, cudaStream_t stream)
{
    int32_t M = static_cast<int32_t>(a.shape(0));
    int32_t N = static_cast<int32_t>(a.shape(1));

    auto dispatch = [&](auto t, auto kvec, auto ktiledim) {
        using T = decltype(t);
        constexpr int kVec = decltype(kvec)::value;
        constexpr int kTileDim = decltype(ktiledim)::value;

        if (M % kTileDim || N % kTileDim) {
            TM_CHECK(0) << "TransposeCopy: shape not divisible by tile " << kTileDim;
            return;
        }

        constexpr int smem_bytes = 2 * kTileDim * (kTileDim + kVec) * sizeof(T);

        dim3 grid(static_cast<uint32_t>(N / kTileDim),
                  static_cast<uint32_t>(M / kTileDim));

        auto src_gmem = make_tensor(make_gmem_ptr(reinterpret_cast<const T*>(data_a)),
            make_layout(make_shape(M, N),
                              make_stride(Int<1>{}, a.stride(1))));

        auto dst_gmem = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(data_b)),
            make_layout(make_shape(M, N),
                              make_stride(b.stride(0), Int<1>{})));

        kernel::TransposeCopyKernel<kTileDim, kVec>
            <<<grid, 256, smem_bytes, stream>>>(src_gmem, dst_gmem);
    };

    switch (byte_size(dtype)) {
        case 1: return dispatch(uint8_t{},  Int<4>{}, Int<64>{});
        case 2: return dispatch(uint16_t{}, Int<4>{}, Int<64>{});
        case 4: return dispatch(uint32_t{}, Int<2>{}, Int<32>{});
        case 8: return dispatch(uint64_t{}, Int<1>{}, Int<32>{});
        default:
            TM_CHECK(0) << "TransposeCopy: unsupported element size " << byte_size(dtype);
            break;
    }
}

}  // namespace turbomind::core
