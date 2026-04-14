#include "src/turbomind/kernels/copy/copy.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/logger.h"
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>

namespace turbomind::core {

using namespace cute;

// ============================================================================
// CUDA kernel: TransposeCopyKernel (TiledCopy 2D layout conversion)
// ============================================================================
namespace kernel {

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
void TransposeCopy(const void* data_a, void* data_b,
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

}  // namespace turbomind::core
