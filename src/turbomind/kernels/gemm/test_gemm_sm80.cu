
#include <thrust/universal_vector.h>

namespace turbomind {

template<typename T>
void gemm_sm80(const T* A, const T* b, T* c, int m, int n, int k, cudaStream_t stream);

}  // namespace turbomind

int main()
{
    int M = 11008;
    int N = 1;
    int K = 4096;

    // int M = 12288;
    // int N = 1;
    // int K = 4096;

    // int M = 4096;
    // int N = 1;
    // int K = 11008;

    // int M = 4096;
    // int N = 1;
    // int K = 4096;

    thrust::universal_vector<half> A(M * K);
    thrust::universal_vector<half> b(N * K);
    thrust::universal_vector<half> c(M * N);

    cudaStream_t stream{};
    cudaStreamCreate(&stream);

    for (int i = 0; i < 100; ++i) {
        turbomind::gemm_sm80(A.data().get(), b.data().get(), c.data().get(), M, N, K, stream);
    }

    cudaStreamSynchronize(stream);

    return 0;
}