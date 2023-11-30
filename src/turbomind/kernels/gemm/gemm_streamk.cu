#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/layout/matrix.h"

// A matrix configuration
using ElementA           = cutlass::half_t;                              // Element type for A matrix operand
using LayoutA            = cutlass::layout::ColumnMajor;                 // Layout type for A matrix operand
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // Memory access granularity/alignment of A
                                                                         // matrix in units of elements (up to 16 bytes)

// B matrix configuration
using ElementB           = cutlass::half_t;                              // Element type for B matrix operand
using LayoutB            = cutlass::layout::ColumnMajor;                 // Layout type for B matrix operand
constexpr int AlignmentB = 64 / cutlass::sizeof_bits<ElementB>::value;  // Memory access granularity/alignment of B
                                                                         // matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using ElementC = cutlass::half_t;               // Element type for C and D matrix operands
using LayoutC  = cutlass::layout::ColumnMajor;  // Layout type for C and D matrix operands
constexpr int AlignmentC =
    128 / cutlass::sizeof_bits<ElementC>::value;  // Memory access granularity/alignment of C/D matrices in units of
                                                  // elements (up to 16 bytes)

// Multiply-accumulate blocking/pipelining details
using ElementAccumulator = float;                // Element type for internal accumulation
using ArchTag            = cutlass::arch::Sm80;  // Tag indicating the minimum SM that supports the intended feature
using OperatorClass      = cutlass::arch::OpClassTensorOp;          // Operator class tag
using ThreadblockShape   = cutlass::gemm::GemmShape<64, 64, 32>;  // Threadblock-level tile size (concept: GemmShape)
using WarpShape          = cutlass::gemm::GemmShape<32, 32, 32>;    // Warp-level tile size (concept: GemmShape)
using InstructionShape   = cutlass::gemm::GemmShape<16, 8, 16>;     // Instruction-level tile size (concept: GemmShape)
constexpr int NumStages  = 5;  // Number of global->shared pipeline stages used in the GEMM mainloop

// Epilogue output operator
using EpilogueOp =
    cutlass::epilogue::thread::LinearCombination<ElementC,    // Element type for C and D matrix operands
                                                 AlignmentC,  // Memory access granularity of C and D matrix in units of
                                                              // elements
                                                 ElementAccumulator,   // Element type from internal accumaccumulation
                                                 ElementAccumulator>;  // Data type used to compute linear combination

using DeviceGemmStreamK =
    cutlass::gemm::device::GemmUniversal<ElementA,
                                         LayoutA,
                                         ElementB,
                                         LayoutB,
                                         ElementC,
                                         LayoutC,
                                         ElementAccumulator,
                                         OperatorClass,
                                         ArchTag,
                                         ThreadblockShape,
                                         WarpShape,
                                         InstructionShape,
                                         EpilogueOp,
                                         cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,  // <-- Only difference
                                         NumStages,
                                         AlignmentA,
                                         AlignmentB>;

namespace turbomind {

template<typename T>
void gemm_streamk(const T* A, const T* b, T* c, int m, int n, int k, cudaStream_t stream)
{
    typename DeviceGemmStreamK::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
                                               {m, n, k},
                                               1,
                                               {ElementAccumulator(1.f), ElementAccumulator(0.f)},
                                               A,
                                               b,
                                               c,
                                               c,
                                               m * k,
                                               n * k,
                                               m * n,
                                               m * n,
                                               m,
                                               k,
                                               m,
                                               m);

    auto workspace_size = DeviceGemmStreamK::get_workspace_size(args);

    constexpr size_t WORKSPACE_SIZE = 32 << 20;

    static void* workspace = [&] {
        void* ptr{};
        cudaMalloc(&ptr, WORKSPACE_SIZE);
        cudaMemset(ptr, 0, WORKSPACE_SIZE);
        cudaDeviceSynchronize();
        return ptr;
    }();

    if (WORKSPACE_SIZE < workspace_size) {
        std::abort();
    }

    DeviceGemmStreamK gemm_op;
    gemm_op(args, workspace, stream);
}

template void gemm_streamk(const half* A, const half* b, half* c, int m, int n, int k, cudaStream_t stream);

}  // namespace turbomind