

#include <cstdint>
#include <iostream>

// #include "src/turbomind/kernels/core/array_ops.h"
#include <thrust/universal_vector.h>

#include <cudaTypedefs.h>
#include <cuda_runtime.h>

PFN_cuTensorMapEncodeTiled get_cuTensorMapEncodeTiled()
{
    static void* cuTensorMapEncodeTiled_ptr = [] {
        cudaDriverEntryPointQueryResult driver_status;
        void*                           ptr{};
        auto status = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &ptr, cudaEnableDefault, &driver_status);
        if (status != cudaSuccess && driver_status != cudaDriverEntryPointSuccess) {
            std::cerr << "failed to get cuTensorMapEncodeTiled\n";
            std::abort();
        }
        std::cout << ptr << "\n";
        return ptr;
    }();
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled>(cuTensorMapEncodeTiled_ptr);
}

#if 0
__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map)
{
    __shared__ int smem_buf[16][16];

    constexpr int tma_tx_bytes = sizeof(smem_buf);

    __shared__ uint64_t barrier;

    if (threadIdx.x == 0) {
        // init barrier
        asm volatile("mbarrier.init.b64 [%0], %1;\n" ::"r"(barrier), "n"(1));

        // set barrier tx bytes
        // asm volatile("mbarrier.expect_tx.b64 [%0], %1;\n" ::"r"(barrier), "n"(tma_tx_bytes));

        // arrive & set expect_tx
        asm volatile("mbarrier.arrive.expect_tx.b64 _, [%0], %1;\n"::"r"(barrier), "n"(tma_tx_bytes));
    }

    __syncthreads();

    // wait barrier
}
#endif

__global__ void kernel_1d(const int* input, int* output)
{
    __shared__ int smem_buf[256];

    constexpr int tma_tx_bytes = sizeof(smem_buf);

    __shared__ uint64_t barrier;
    uint64_t            state;

    uint64_t smem_int64_ptr;
    asm volatile("cvta.shared::cta.u64 %0, %1;\n" : "=l"(smem_int64_ptr) : "l"(smem_buf));
    uint32_t smem_int_ptr = smem_int64_ptr;

    if (threadIdx.x == 0) {
        // init barrier
        asm volatile("mbarrier.init.b64 [%0], %1;\n" ::"l"(&barrier), "n"(1));

        // make barrier visible to async proxy
        asm volatile("fence.proxy.async;\n");

        // set barrier tx count
        asm volatile("mbarrier.expect_tx.b64 [%0], %1;\n" ::"l"(&barrier), "n"(tma_tx_bytes));

        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::  //
                     "r"(smem_int_ptr),
                     "l"(input),
                     "n"(tma_tx_bytes),
                     "l"(&barrier));

        // arrive
        asm volatile("mbarrier.arrive.b64 %0, [%1];\n" : "=l"(state) : "l"(&barrier));

        asm volatile("{\n"
                     "  .reg.pred complete;\n"
                     "  waitLoop:\n"
                     "  mbarrier.try_wait.b64 complete, [%0], %1;\n"
                     "  @!complete bra waitLoop;\n"
                     "}\n" ::"l"(&barrier),
                     "l"(state));
    }

    __syncthreads();

    // output[threadIdx.x] = smem_buf[threadIdx.x] * 2;

    smem_buf[threadIdx.x] *= 2;

    // make `smem_buf` modification visible to async proxy
    asm volatile("fence.proxy.async;\n");

    __syncthreads();

    if (threadIdx.x == 0) {
        // int offset = threadIdx.x * 128;
        asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n" ::  //
                     "l"(output),
                     "r"(smem_int_ptr),
                     "n"(tma_tx_bytes));
        asm volatile("cp.async.bulk.commit_group;\n");
        asm volatile("cp.async.bulk.wait_group 0;\n");
    }
}

void test_tma_1d()
{
    thrust::universal_vector<int> input(256);
    thrust::universal_vector<int> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = i;
    }
    kernel_1d<<<1, 256>>>(input.data().get(), output.data().get());
    cudaDeviceSynchronize();
    for (size_t i = 0; i < output.size(); ++i) {
        std::cerr << output[i] << " ";
    }
    std::cerr << "\n";
}

int main(int argc, char* argv[])
{
    test_tma_1d();
    /*
    CUtensorMap tensor_map{};
    uint64_t    size[]        = {16, 16};
    uint64_t    stride[]      = {16 * sizeof(int)};  // in bytes
    uint32_t    box_size[]    = {16, 16};
    uint32_t    elem_stride[] = {1, 1};

    auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

    CUresult res = cuTensorMapEncodeTiled(&tensor_map,  //
                                          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
                                          2,
                                          0,
                                          size,
                                          stride,
                                          box_size,
                                          elem_stride,
                                          CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                                          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
                                          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) {
        std::cerr << "cuTensorMapEncodeTiled failed: " << res << "\n";
        std::abort();
    }
*/
    return 0;
}