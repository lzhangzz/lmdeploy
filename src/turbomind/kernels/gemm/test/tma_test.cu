

#include <cstdint>
#include <iostream>

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"

#include <thrust/universal_vector.h>

#include <cudaTypedefs.h>
#include <cuda_runtime.h>

#include <cuda/barrier>
// #include <cuda/ptx>

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

template<int N, class T>
__global__ void kernel_1d(const T* input, T* output)
{
    __shared__ int smem_buf[N];

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

        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::  //
                     "r"(smem_int_ptr),
                     "l"(input),
                     "n"(tma_tx_bytes),
                     "l"(&barrier));

        asm volatile("mbarrier.arrive.expect_tx.b64 %0, [%1], %2;\n" : "=l"(state) : "l"(&barrier), "n"(tma_tx_bytes));

        asm volatile("{\n"
                     "  .reg.pred complete;\n"
                     "  waitLoop:\n"
                     "  mbarrier.try_wait.b64 complete, [%0], %1;\n"
                     "  @!complete bra waitLoop;\n"
                     "}\n" ::"l"(&barrier),
                     "l"(state));
    }

    __syncthreads();

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        smem_buf[i] *= 2;
    }

    // make `smem_buf` modification visible to async proxy
    asm volatile("fence.proxy.async;\n");

    __syncthreads();

    constexpr int tma_thrs   = 1;
    constexpr int split_size = N / tma_thrs;

    // This will become a loop if `tma_thrs > 1`
    if (threadIdx.x < tma_thrs) {
        int offset = threadIdx.x * split_size;
        asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n" ::  //
                     "l"(output + offset),
                     "r"(smem_int_ptr + offset * (int)sizeof(int)),
                     "n"(split_size * sizeof(int)));
        asm volatile("cp.async.bulk.commit_group;\n");
        asm volatile("cp.async.bulk.wait_group 0;\n");
    }
}

template<int chunk_size, int iter, class T>
__global__ void batch_1d(const T* input, T* output, size_t n)
{
    __shared__ __align__(16) T smem_buf[iter][chunk_size];
    __shared__ __align__(8) uint64_t barrier[iter];

    uint32_t smem_int_ptr[iter];
    PRAGMA_UNROLL
    for (int i = 0; i < iter; ++i) {
        uint64_t tmp;
        asm volatile("cvta.shared::cta.u64 %0, %1;\n" : "=l"(tmp) : "l"(smem_buf[i]));
        smem_int_ptr[i] = tmp;
    }

    uint64_t state[iter];

    if (threadIdx.x == 0) {
        PRAGMA_UNROLL
        for (int i = 0; i < iter; ++i) {
            asm volatile("mbarrier.init.b64 [%0], %1;\n" ::"l"(&barrier[i]), "n"(1));
        }
        asm volatile("fence.proxy.async;\n");
    }

    for (int base = 0; base < n; base += gridDim.x * iter * chunk_size) {

        if (threadIdx.x == 0) {
            PRAGMA_UNROLL
            for (int i = 0; i < iter; ++i) {
                const int offset       = base + blockIdx.x * iter * chunk_size + i * chunk_size;
                const int tma_tx_bytes = min(max(n - offset, 0UL), (size_t)chunk_size) * sizeof(T);
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::  //
                    "r"(smem_int_ptr[i]),
                    "l"(input + offset),
                    "r"(tma_tx_bytes),
                    "l"(&barrier[i]));

                // This must come after `cp.async.bulk` to ensure memory ordering
                asm volatile("mbarrier.arrive.expect_tx.b64 %0, [%1], %2;\n"
                             : "=l"(state[i])
                             : "l"(&barrier[i]), "r"(tma_tx_bytes));
            }
        }

        PRAGMA_UNROLL
        for (int i = 0; i < iter; ++i) {
            if (threadIdx.x == 0) {
                asm volatile("{\n"
                             "  .reg.pred complete;\n"
                             "  waitLoop:\n"
                             "  mbarrier.try_wait.b64 complete, [%0], %1;\n"
                             "  @!complete bra waitLoop;\n"
                             "}\n" ::"l"(&barrier[i]),
                             "l"(state[i]));
            }
            __syncthreads();

            for (int p = threadIdx.x; p < chunk_size; p += blockDim.x) {
                smem_buf[i][p] *= 2;
            }

            asm volatile("fence.proxy.async;\n");

            __syncthreads();

            if (threadIdx.x == 0) {
                const int offset       = base + blockIdx.x * iter * chunk_size + i * chunk_size;
                const int tma_tx_bytes = min(max(n - offset, 0UL), (size_t)chunk_size) * sizeof(T);
                asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n" ::  //
                             "l"(output + offset),
                             "r"(smem_int_ptr[i]),
                             "r"(tma_tx_bytes));
                asm volatile("cp.async.bulk.commit_group;\n");
            }
        }

        if (threadIdx.x == 0) {
            asm volatile("cp.async.bulk.wait_group 0;\n");
        }

        __syncthreads();  // wait for the smem being used
    }
}

__device__ uint32_t cvta_shared_cta(void* p)
{
    uint64_t tmp;
    asm volatile("cvta.shared::cta.u64 %0, %1;\n" : "=l"(tmp) : "l"(p));
    return static_cast<uint32_t>(tmp);
}

template<int chunk_size, int stages, class T>
__global__ void batch_1d_multistage(const T* input, T* output, int64_t n)
{
    // __shared__ __align__(16) T smem_buf[stages][chunk_size];

    constexpr int stage_size = chunk_size * sizeof(T);

    __shared__ extern T smem_buf[];

    __shared__ __align__(8) uint64_t barrier[stages];

    auto smem_int_ptr = cvta_shared_cta(smem_buf);

    if (threadIdx.x == 0) {
        PRAGMA_UNROLL
        for (int s = 0; s < stages; ++s) {
            asm volatile("mbarrier.init.b64 [%0], %1;\n" ::"l"(&barrier[s]), "r"(1));
        }
        asm volatile("fence.proxy.async;\n");
    }

    int64_t load_ptr  = 0;
    int64_t store_ptr = 0;

    auto load = [&](int& s) {
        uint64_t token{};
        if (threadIdx.x == 0) {
            const int offset       = load_ptr + blockIdx.x * chunk_size;
            const int tma_tx_bytes = min(max(n - offset, 0L), (long)chunk_size) * sizeof(T);

            if (tma_tx_bytes) {
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::  //
                    "r"(smem_int_ptr + s * stage_size),
                    "l"(input + offset),
                    "r"(tma_tx_bytes),
                    "r"(cvta_shared_cta(&barrier[s])));
            }

            // This must come after `cp.async.bulk` to ensure memory ordering
            asm volatile("mbarrier.arrive.expect_tx.b64 %0, [%1], %2;\n"
                         : "=l"(token)
                         : "r"(cvta_shared_cta(&barrier[s])), "r"(tma_tx_bytes));
        }
        s = (s + 1) % stages;
        load_ptr += gridDim.x * chunk_size;
        return token;
    };

    auto compute = [&](int& s, uint64_t token) {
        if (threadIdx.x == 0) {
            asm volatile("{\n"
                         "  .reg.pred complete;\n"
                         "  waitLoop:\n"
                         "  mbarrier.try_wait.b64 complete, [%0], %1;\n"
                         "  @!complete bra waitLoop;\n"
                         "}\n" ::"r"(cvta_shared_cta(&barrier[s])),
                         "l"(token));
        }
        __syncthreads();
        for (int p = threadIdx.x; p < chunk_size; p += blockDim.x) {
            smem_buf[s * chunk_size + p] *= 2;
        }
        asm volatile("fence.proxy.async;\n");
        __syncthreads();
        s = (s + 1) % stages;
    };

    auto store = [&](int& s) {
        if (threadIdx.x == 0) {
            const int offset       = store_ptr + blockIdx.x * chunk_size;
            const int tma_tx_bytes = min(max(n - offset, 0L), (long)chunk_size) * sizeof(T);
            if (tma_tx_bytes) {
                asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n" ::  //
                             "l"(output + offset),
                             "r"(smem_int_ptr + s * stage_size),
                             "r"(tma_tx_bytes));
            }
            asm volatile("cp.async.bulk.commit_group;\n");
            asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(stages - 3));
        }
        s = (s + 1) % stages;
        store_ptr += gridDim.x * chunk_size;
    };

    uint64_t token0;
    uint64_t token1;

    int ri = 0;
    int ci = 0;
    int wi = 0;

    token0 = load(ri);

    token1 = load(ri);
    compute(ci, token0);

    while (store_ptr < n) {
        token0 = load(ri);
        compute(ci, token1);
        store(wi);
        token1 = token0;
    }

    if (threadIdx.x == 0) {
        asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(0));
    }
    __syncthreads();
}

template<class T>
__global__ void reference_1d(const T* input, T* output, size_t n)
{
    n /= 4;
    using namespace turbomind;
    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        Array<T, 4> tmp;
        Ldg(tmp, &input[i * 4]);
        for (auto& x : tmp) {
            x *= 2;
        }
        Store(&output[i * 4], tmp);
    }
}

void test_tma_1d()
{
    constexpr int                 N = 1024 * 1024 * 1024;
    thrust::universal_vector<int> input(N);
    thrust::universal_vector<int> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = i;
    }
    thrust::fill(output.begin(), output.end(), -1);

    cudaMemPrefetchAsync_v2(input.data().get(), sizeof(int) * input.size(), {cudaMemLocationTypeDevice, 0}, 0);
    cudaMemPrefetchAsync_v2(output.data().get(), sizeof(int) * output.size(), {cudaMemLocationTypeDevice, 0}, 0);

    // kernel_1d<N><<<1, 256>>>(input.data().get(), output.data().get());

    // batch_1d<8192, 1><<<256, 128>>>(input.data().get(), output.data().get(), input.size());

    auto func = &batch_1d_multistage<6144, 4, int>;

    constexpr int smem_size = sizeof(int) * 6144 * 4;

    cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    func<<<256, 128, smem_size>>>(input.data().get(), output.data().get(), input.size());
    // reference_1d<<<512, 512>>>(input.data().get(), output.data().get(), input.size());

    cudaMemPrefetchAsync_v2(output.data().get(), sizeof(int) * output.size(), {cudaMemLocationTypeHost, 0}, 0);

    cudaDeviceSynchronize();

    size_t fail = 0;
    for (size_t i = 0; i < output.size(); ++i) {
        if (output[i] != input[i] * 2) {
            ++fail;
        }
    }
    std::cerr << "failed: " << fail << "\n";

    // for (size_t i = 0; i < output.size(); ++i) {
    //     std::cerr << output[i] << " ";
    // }
    // std::cerr << "\n";
}

void test_tma_2d()
{
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
}

int main(int argc, char* argv[])
{
    test_tma_1d();
    return 0;
}