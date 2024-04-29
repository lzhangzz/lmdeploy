#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/gemm_template.h"

namespace turbomind::gemm {

template void invoke(half*        C,
                     const half*  A,
                     const half*  B,
                     const half*  Q,
                     int          m,
                     int          n,
                     int          k,
                     int          splits,
                     void*        workspace,
                     cudaStream_t st);
template void invoke(half*          C,
                     const half*    A,
                     const uint4_t* B,
                     const half*    Q,
                     int            m,
                     int            n,
                     int            k,
                     int            splits,
                     void*          workspace,
                     cudaStream_t   st);
template void invoke(half*          C,
                     const half*    A,
                     const uint8_t* B,
                     const half*    Q,
                     int            m,
                     int            n,
                     int            k,
                     int            splits,
                     void*          workspace,
                     cudaStream_t   st);

}  // namespace turbomind::gemm