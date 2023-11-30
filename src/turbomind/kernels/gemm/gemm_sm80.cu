// clang-format off

#include "cute/arch/copy_sm80.hpp"
#include "cute/numeric/math.hpp"
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/layout.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/reduction/kernel/reduce_split_k.h"

// clang-format on

#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

using namespace cutlass;
using namespace cute;

template<class Tacc, class Tin, class Tout>
__global__ void reduce_split_k(const Tin* src, Tout* dst, int m, int n, int l)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < m * n; i += gridDim.x * blockDim.x) {
        Tacc acc{};
        for (int j = 0; j < l; ++j) {
            acc += (Tacc)src[i + m * n * j];
        }
        dst[i] = (Tout)acc;
    }
}

// template<bool IsSplitK, class T, class M, class N, class _K, class L>
template<bool IsSplitK, class T>
void gemm_sm80_impl(const T* A, const T* b, T* c, int m, int n, int k, int l, cudaStream_t stream)
{
    using namespace cutlass::gemm;

    // using K = conditional_t<is_static<_K>{} && is_static<L>{}, decltype(safe_div(_K{}, L{})), int>;
    // using K = int;

    using LayoutA   = cutlass::layout::ColumnMajor;
    using LayoutB   = cutlass::layout::ColumnMajor;
    using LayoutC   = cutlass::layout::ColumnMajor;
    using TileShape = Shape<_128, _8, _32>;

    using DispatchPolicy = gemm::MainloopSm80CpAsync<8>;
    using TiledMma       = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                              Layout<Shape<_4, _1, _1>>,   // 4x1x1 thread group
                              Layout<Shape<_1, _1, _2>>>;  // 1x1x2 value group for 16x8x32 MMA and LDSM

    // A
    using SmemLayoutAtomA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_64, _8>, Stride<_1, _64>>{}));
    using SmemCopyAtomA   = Copy_Atom<SM75_U16x8_LDSM_T, half_t>;
    using GmemTiledCopyA  = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, half_t>{},
                                                    Layout<Shape<_16, _8>, Stride<_1, _16>>{},
                                                    Layout<Shape<_8, _1>>{}));

    // B
    using SmemLayoutAtomB = decltype(composition(Swizzle<2, 3, 3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{}));
    using SmemCopyAtomB   = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;
    using GmemTiledCopyB  = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint32_t>, half_t>{},
                                                    Layout<Shape<_8, _16>, Stride<_16, _1>>{},
                                                    Layout<Shape<_1, _2>>{}));

    static int init = [&] {
        print("TiledMMA:\n"), print(TiledMma{}), print('\n');

        print("GmemTiledCopyA:\n"), print(GmemTiledCopyA{}), print('\n');
        print("SmemTiledCopyA:\n"), print(decltype(make_tiled_copy_A(SmemCopyAtomA{}, std::declval<TiledMma>())){}),
            print('\n');

        print("GmemTiledCopyB:\n"), print(GmemTiledCopyB{}), print('\n');
        print("SmemTiledCopyB:\n"), print(decltype(make_tiled_copy_B(SmemCopyAtomB{}, std::declval<TiledMma>())){}),
            print('\n');
        return 0;
    }();

    // Mainloop
    using CollectiveMainloop = collective::CollectiveMma<DispatchPolicy,
                                                         TileShape,
                                                         half_t,
                                                         TagToStrideA_t<LayoutA>,
                                                         half_t,
                                                         TagToStrideB_t<LayoutB>,
                                                         TiledMma,
                                                         GmemTiledCopyA,
                                                         SmemLayoutAtomA,
                                                         SmemCopyAtomA,
                                                         cute::identity,  // A
                                                         GmemTiledCopyB,
                                                         SmemLayoutAtomB,
                                                         SmemCopyAtomB,
                                                         cute::identity  // B
                                                         >;

    // static int init2 = [&] {
    //     print("SmemLayoutA:\n"), print(typename CollectiveMainloop::SmemLayoutA{}), print('\n');
    //     print("SmemLayoutB:\n"), print(typename CollectiveMainloop::SmemLayoutB{}), print('\n');
    //     return 0;
    // }();

    using ElementC = std::conditional_t<IsSplitK, float, half_t>;

    static constexpr int kAlignmentD = 16 / sizeof(ElementC);

    // Epilogue
    using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
        TagToStrideC_t<LayoutC>,
        TagToStrideC_t<LayoutC>,
        epilogue::thread::LinearCombination<ElementC, kAlignmentD, float, float, epilogue::thread::ScaleType::Nothing>,
        cutlass::gemm::EpilogueDefault>;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    TagToStrideA_t<LayoutA> stride_a{};
    get<1>(stride_a) = m;
    get<2>(stride_a) = m * k / l;

    TagToStrideB_t<LayoutB> stride_b{};
    get<0>(stride_b) = k;
    get<2>(stride_b) = k / l;

    TagToStrideC_t<LayoutB> stride_c{};
    get<1>(stride_c) = m;
    get<2>(stride_c) = m * n;

    Shape<int, int, int, int> problem_size{m, n, k / l, l};
    // Shape<M, N, K, L> problem_size{};
    // if constexpr (!is_static<M>{}) {
    //     get<0>(problem_size) = m;
    // }
    // if constexpr (!is_static<N>{}) {
    //     get<1>(problem_size) = n;
    // }
    // if constexpr (!is_static<K>{}) {
    //     get<2>(problem_size) = k / l;
    // }
    // if constexpr (!is_static<L>{}) {
    //     get<3>(problem_size) = l;
    // }

    constexpr size_t WORKSPACE_SIZE = 32 << 20;

    static void* workspace = [&] {
        void* ptr{};
        cudaMalloc(&ptr, WORKSPACE_SIZE);
        cudaMemset(ptr, 0, WORKSPACE_SIZE);
        cudaDeviceSynchronize();
        return ptr;
    }();

    auto dst = IsSplitK ? (ElementC*)workspace : (ElementC*)c;

    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kBatched,
                                  problem_size,
                                  {
                                      (const half_t*)A,
                                      stride_a,
                                      (const half_t*)b,
                                      stride_b,
                                  },
                                  {{}, (const ElementC*)dst, stride_c, (ElementC*)dst, stride_c}};

    Gemm gemm_op;
    auto status = gemm_op.run(args, nullptr, stream);

    turbomind::FT_CHECK(status == cutlass::Status::kSuccess);

    if (IsSplitK) {
        reduce_split_k<float><<<128, 512, 0, stream>>>((const ElementC*)workspace, c, m, n, l);
    }
}

template<typename T>
void gemm_sm80(const T* A, const T* b, T* c, int m, int n, int k, cudaStream_t stream)
{
    // using cute::Int;
    // if (m == 11008 && n == 1 && k == 4096) {
    //     gemm_sm80_impl<0>(A, b, c, Int<11008>{}, Int<1>{}, Int<4096>{}, cute::Int<1>{}, stream);
    //     return;
    // }
    // if (m == 12288 && n == 1 && k == 4096) {
    //     gemm_sm80_impl<0>(A, b, c, Int<12288>{}, Int<1>{}, Int<4096>{}, cute::Int<1>{}, stream);
    //     return;
    // }
    // if (m == 4096 && n == 1 && k == 11008) {
    //     gemm_sm80_impl<1>(A, b, c, Int<4096>{}, Int<1>{}, Int<11008>{}, cute::Int<2>{}, stream);
    //     return;
    // }
    // if (m == 4096 && n == 1 && k == 4096) {
    //     gemm_sm80_impl<1>(A, b, c, Int<4096>{}, Int<1>{}, Int<4096>{}, cute::Int<2>{}, stream);
    //     return;
    // }

    if (k >= m) {
        gemm_sm80_impl<1>(A, b, c, m, n, k, 2, stream);
    }
    else {
        gemm_sm80_impl<0>(A, b, c, m, n, k, 1, stream);
    }
}

template void gemm_sm80(const half* A, const half* b, half* c, int m, int n, int k, cudaStream_t stream);

}  // namespace turbomind