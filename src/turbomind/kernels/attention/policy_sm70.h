#include "array_ops.h"
#include "policy.h"
#include "src/turbomind/kernels/custom_ar_kernels.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"
#include <cmath>

namespace turbomind {

__inline__ __device__ void
mma_m8n8k4_row_col(Array<float, 8>& d, const Array<half, 4>& a, const Array<half, 4>& b, Array<float, 8>& c)
{
#if TURBOMIND_ARCH_SM70
    uint32_t const* A = reinterpret_cast<uint32_t const*>(&a);
    uint32_t const* B = reinterpret_cast<uint32_t const*>(&b);
    // clang-format off
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32"
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11},"
        "{%12, %13, %14, %15, %16, %17, %18, %19};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
        : "r"(A[0]), "r"(A[1]), 
          "r"(B[0]), "r"(B[1]), 
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "f"(c[4]), "f"(c[5]), "f"(c[6]), "f"(c[7]));
// clang-format on
#endif
}

__inline__ __device__ void
mma_m8n8k4_row_row(Array<float, 8>& d, const Array<half, 4>& a, const Array<half, 4>& b, Array<float, 8>& c)
{
#if TURBOMIND_ARCH_SM70
    uint32_t const* A = reinterpret_cast<uint32_t const*>(&a);
    uint32_t const* B = reinterpret_cast<uint32_t const*>(&b);
    // clang-format off
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32"
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11},"
        "{%12, %13, %14, %15, %16, %17, %18, %19};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
        : "r"(A[0]), "r"(A[1]), 
          "r"(B[0]), "r"(B[1]), 
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "f"(c[4]), "f"(c[5]), "f"(c[6]), "f"(c[7]));
// clang-format on
#endif
}

template<class T, class Tkv, int CTA_Q, int CTA_S, int HeadDim>
struct AttentionPolicy<sm70_t, T, Tkv, CTA_Q, CTA_S, HeadDim> {
    static constexpr int kPadQ = 4;
    static constexpr int kPadK = 4;
    static constexpr int kPadP = 4;
    static constexpr int kPadV = 0;

    static constexpr bool kUseSmemQ = false;
    static constexpr bool kUseSmemP = false;

    static constexpr int WARP_S = CTA_S;
    static constexpr int WARP_Q = 16;

    static constexpr int kWarpCount = CTA_Q / WARP_Q;

    static constexpr int OP_M = 16;
    static constexpr int OP_N = 16;
    static constexpr int OP_K = 4;

    static constexpr int kM = WARP_Q / OP_M;   // 1
    static constexpr int kN = WARP_S / OP_N;   // 4
    static constexpr int kK = HeadDim / OP_K;  // 32

    static constexpr int vM = WARP_Q / OP_M;   // 1
    static constexpr int vN = HeadDim / OP_N;  // 8
    static constexpr int vK = WARP_S / OP_K;   // 16

    //  +---+---+
    //  | 0 | 1 |
    //  +---+---+
    //  | 2 | 3 |
    //  +---+---+
    using FragQ = Array<half, 4>[kK][kM];   //    (q2,q2,x2,q4) (Dk,Qm) (d4)
                                            //      4  8  0  1    4 16    1
    using FragK = Array<half, 4>[kK][kN];   //    (s2,x2,s2,s4) (Dk,Sn) (d4)
                                            //      4  0  8  1    4 16    1
    using FragS = Array<float, 8>[kM][kN];  // (q2,q2,s2,s2,q2) (Qm,Sn) (s2,q2,s2)
                                            //   4  8  8  2  1   16 16    4  2  1
    using FragP = Array<half, 4>[vK][vM];   //    (q2,q2,x2,q4) (Sk,Qm) (s4)
                                            //      4  8  0  1    4 16    1
    using FragV = Array<half, 4>[vK][vN];   //    (d2,x2,d2,s4) (Sk,Dn) (d4)       [row major]
                                            //      4  0  8  1    4 16    1
    using FragO = Array<float, 8>[vM][vN];  // (q2,q2,d2,d2,q2) (Qm,Dn) (d2,q2,d2)
                                            //   4  8  8  2  1   16 16    4  2  1
    using FragM = Array<float, 2>[vM];      // (q2,q2,_2,_2,q2) (Qm)    (q2))
    using FragL = FragM;

    using Swizzle = Identity;

    struct SwizzleV {
        __device__ int operator()(int offset)
        {
            // Rearrange for LDS.128 (also avoid bank-conflict along C)
            // 6543210
            // dDDDDdd
            offset = ((offset & 8) << 2) ^ offset;                                     // x[5] ^= x[3]
            offset = ((offset & ~20) | (((offset & 16) >> 2) | ((offset & 4) << 2)));  // swap(x[4], x[2])

            // Shuffle C according S to avoid bank-conflict
            // ssssSSddDDddd
            offset = ((offset & (0x3 << 7)) >> 4) ^ offset;
            return offset;
        }

        template<int D>
        __device__ int AdvanceS(int offset, int s0, int s1)
        {
            if constexpr (D % 4 == 0) {
                return offset;
            }
            else if constexpr (D % 2 == 0) {
                return offset ^ (((s0 ^ s1) & 0x2) << 3);
            }
            else {
                return offset ^ (((s0 ^ s1) & 0x3) << 3);
            }
        }
    };

    template<class Fragment, class Func>
    static __device__ void ForeachS(Fragment& S, Func&& func)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int m = 0; m < kM; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < kN; ++n) {
                PRAGMA_UNROLL
                for (int s1 = 0; s1 < 2; ++s1) {
                    PRAGMA_UNROLL
                    for (int q = 0; q < 2; ++q) {
                        PRAGMA_UNROLL
                        for (int s0 = 0; s0 < 2; ++s0) {
                            const int qi = m * OP_M + (lane_id & 8) + (lane_id & 1) + lane_id / 16 * 4 + q * 2;
                            const int si = n * OP_N + (lane_id & 4) * 2 + (lane_id & 2) + s1 * 4 + s0;
                            ((Func&&)func)(warp_id * WARP_Q + qi, si, S[m][n][s1 * 4 + q * 2 + s0]);
                        }
                    }
                }
            }
        }
    }

    __device__ void TransformQ(const T* smem_Q, FragQ& frag_Q)
    {
        if constexpr (!kUseSmemQ) {
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;
            PRAGMA_UNROLL
            for (int k = 0; k < kK; ++k) {
                PRAGMA_UNROLL
                for (int m = 0; m < kM; ++m) {
                    const int mm = m * OP_M + (lane_id & 8) + lane_id % 4 + lane_id / 16 * 4 + warp_id * WARP_Q;
                    const int kk = k * 4 + 0;
                    Lds(frag_Q[k][m], &smem_Q[mm * (HeadDim + kPadQ) + kk]);
                }
            }
        }
    }

    template<class SmemQ, class SmemK>
    static __device__ void ComputeQK(SmemQ& smem_Q, SmemK& smem_K, FragQ& frag_Q, FragS& frag_S)
    {
        if constexpr (kUseSmemQ) {
            smem_Q.LoadQ_sm70(frag_Q[0], 0);
        }
        FragK frag_K;
        smem_K.LoadK_sm70(frag_K[0], 0);

        PRAGMA_UNROLL
        for (int k = 0; k < kK; ++k) {
            if (k < kK - 1) {
                if constexpr (kUseSmemQ) {
                    smem_Q.LoadQ_sm70(frag_Q[k + 1], k + 1);
                }
                smem_K.LoadK_sm70(frag_K[k + 1], k + 1);
            }
            PRAGMA_UNROLL
            for (int m = 0; m < kM; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < kN; ++n) {
                    mma_m8n8k4_row_col(frag_S[m][n], frag_Q[k][m], frag_K[k][n], frag_S[m][n]);
                }
            }
        }
    }

    template<class SmemP, class Smem>
    __device__ void ComputePV(SmemP& smem_P, Smem& smem_V, FragP& frag_P, FragO& frag_O)
    {
        if constexpr (kUseSmemP) {
            smem_P.LoadP_sm70(frag_P[0], 0);
        }
        FragV frag_V;
        smem_V.LoadV_sm70(frag_V[0], 0);

        PRAGMA_UNROLL
        for (int k = 0; k < vK; ++k) {
            if (k < vK - 1) {
                if constexpr (kUseSmemP) {
                    smem_P.LoadP_sm70(frag_P[k + 1], k + 1);
                }
                smem_V.LoadV_sm70(frag_V[k + 1], k + 1);
            }
            PRAGMA_UNROLL
            for (int m = 0; m < vM; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < vN; ++n) {
                    mma_m8n8k4_row_row(frag_O[m][n], frag_P[k][m], frag_V[k][n], frag_O[m][n]);
                }
            }
        }
    }

    template<bool is_residue>
    static __device__ void
    Softmax(FragS& frag_S, FragM& frag_M, FragM& frag_L, FragP& frag_P, FragO& frag_O, float qk_scale, T* smem_P)
    {
        FragM prev_M;
        PRAGMA_UNROLL
        for (int m = 0; m < kM; ++m) {
            prev_M[m] = frag_M[m];
        }

        PRAGMA_UNROLL
        for (int m = 0; m < kM; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < kN; ++n) {
                PRAGMA_UNROLL
                for (int s1 = 0; s1 < 2; ++s1) {
                    PRAGMA_UNROLL
                    for (int q = 0; q < 2; ++q) {
                        PRAGMA_UNROLL
                        for (int s0 = 0; s0 < 2; ++s0) {
                            frag_M[m][q] =
                                fmaxf(frag_M[m][q], frag_S[m][n][s1 * 4 + q * 2 + s0]);  // reduce over local quad
                        }
                    }
                }
            }
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {  // reduce over thread group within warp (within warp tiles)
                frag_M[m][q] = fmaxf(frag_M[m][q], __shfl_xor_sync(uint32_t(-1), frag_M[m][q], 2));
                frag_M[m][q] = fmaxf(frag_M[m][q], __shfl_xor_sync(uint32_t(-1), frag_M[m][q], 4));
            }
        }

        PRAGMA_UNROLL
        for (int m = 0; m < kM; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                // exp(M - M'), isinf(frag_M) => isnan(expdiff_M)
                float expdiff_M = exp2f((prev_M[m][q] - frag_M[m][q]) * qk_scale);
                if (is_residue && frag_M[m][q] == -std::numeric_limits<float>::infinity()) {
                    expdiff_M = 0.f;
                }
                PRAGMA_UNROLL
                for (int n = 0; n < vN; ++n) {
                    PRAGMA_UNROLL
                    for (int s1 = 0; s1 < 2; ++s1) {
                        PRAGMA_UNROLL
                        for (int s0 = 0; s0 < 2; ++s0) {
                            frag_O[m][n][s1 * 4 + q * 2 + s0] *= expdiff_M;  // Rescale previous output
                        }
                    }
                }

                frag_L[m][q] *= expdiff_M;
            }
        }

        PRAGMA_UNROLL
        for (int m = 0; m < kM; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                float tmp_L{};
                PRAGMA_UNROLL
                for (int n = 0; n < kN; ++n) {
                    PRAGMA_UNROLL
                    for (int s1 = 0; s1 < 2; ++s1) {
                        PRAGMA_UNROLL
                        for (int s0 = 0; s0 < 2; ++s0) {
                            // unnormalized prob, optimized to FFMA
                            float p = exp2f(frag_S[m][n][s1 * 4 + q * 2 + s0] * qk_scale - frag_M[m][q] * qk_scale);
                            if (is_residue && frag_M[m][q] == -std::numeric_limits<float>::infinity()) {
                                p = 0.f;
                            }
                            tmp_L += p;
                            frag_S[m][n][s1 * 4 + q * 2 + s0] = p;
                        }
                    }
                }
                tmp_L += __shfl_xor_sync(uint32_t(-1), tmp_L, 2);
                tmp_L += __shfl_xor_sync(uint32_t(-1), tmp_L, 4);
                frag_L[m][q] = frag_L[m][q] + tmp_L;  // update L
            }
        }

        ForeachS(frag_S, [&](int qi, int si, float p) { smem_P[qi * (CTA_S + kPadP) + si] = half(p); });

        if constexpr (!kUseSmemP) {
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;
            PRAGMA_UNROLL
            for (int k = 0; k < vK; ++k) {
                PRAGMA_UNROLL
                for (int m = 0; m < vM; ++m) {
                    const int qi = m * OP_M + lane_id / 16 * 4 + (lane_id & 8) + lane_id % 4 + warp_id * WARP_Q;
                    const int si = k * OP_K;
                    Lds(frag_P[k][m], &smem_P[qi * (CTA_S + kPadP) + si]);
                }
            }
        }
    }

    template<class Func>
    static __device__ void StoreO(FragO& frag_O, const FragL& frag_L, Func&& func)
    {
        FragL tmp_L;
        PRAGMA_UNROLL
        for (int m = 0; m < vM; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                tmp_L[m][q] = fdividef(1.f, frag_L[m][q] + 1e-8f);
            }
        }

        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        const int mm = lane_id / 16 * 4 + (lane_id & 8) + (lane_id & 1);
        const int nn = (lane_id & 4) * 2 + (lane_id & 2);

        PRAGMA_UNROLL
        for (int m = 0; m < vM; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < vN; ++n) {
                PRAGMA_UNROLL
                for (int d1 = 0; d1 < 2; ++d1) {
                    PRAGMA_UNROLL
                    for (int q = 0; q < 2; ++q) {
                        const int qi = m * OP_M + mm + q * 2 + warp_id * WARP_Q;
                        const int di = n * OP_N + nn + d1 * 4;
                        //
                        Array<half, 2> tmp_O;
                        PRAGMA_UNROLL
                        for (int d0 = 0; d0 < 2; ++d0) {
                            tmp_O[d0] = (T)(frag_O[m][n][d1 * 4 + q * 2 + d0] * tmp_L[m][q]);
                        }
                        ((Func&&)func)(qi, di, tmp_O);
                    }
                }
            }
        }
    }
};

}  // namespace turbomind