#include "policy.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"

namespace turbomind {

template<class T, class Tkv, int CTA_Q, int CTA_S, int HeadDim>
struct AttentionPolicy<sm80_t, T, Tkv, CTA_Q, CTA_S, HeadDim> {
    static constexpr int kSmemPadding = 0;
    static constexpr int kWarpCount   = 4;

    static constexpr int OP_M = 16;
    static constexpr int OP_N = 8;
    static constexpr int OP_K = 16;

    static constexpr int WARP_S = CTA_S;
    static constexpr int WARP_Q = CTA_Q / kWarpCount;

    static constexpr int K_ITER_M = WARP_Q / OP_M;   //  16 / 16 = 1
    static constexpr int K_ITER_N = WARP_S / OP_N;   //  64 /  8 = 8
    static constexpr int K_ITER_K = HeadDim / OP_K;  // 128 / 16 = 8
    static constexpr int V_ITER_M = WARP_Q / OP_M;   //  16 / 16 = 1
    static constexpr int V_ITER_N = HeadDim / OP_N;  // 128 /  8 = 16 -> D16
    static constexpr int V_ITER_K = WARP_S / OP_K;   //  64 / 16 = 4  -> S4

    using FragQ  = Array<T, 8>[K_ITER_K][K_ITER_M];      // ((q8, d4), (D8, Q1), (d2, q2, d2))
    using FragK  = Array<Tkv, 4>[K_ITER_K][K_ITER_N];    // ((s8, d4), (D8, S8), (d2, d2))
    using FragS  = Array<float, 4>[K_ITER_M][K_ITER_N];  // ((q8, s4), (Q1, S8), (q2, s2))
    using FragPs = Array<T, 4>[K_ITER_M][K_ITER_N];      // ((q8, s4), (Q1, S8), (q2, s2))
    using FragP  = Array<T, 8>[V_ITER_M][V_ITER_K];      // ((q8, s4), (Q1, S4), (s2, q2, s2))
    using FragV  = Array<Tkv, 4>[V_ITER_K][V_ITER_N];    // ((d8, s4), (S4, D16), (s2, s2))
    using FragO  = Array<float, 4>[V_ITER_M][V_ITER_N];  // ((q8, d4), (Q1, D16), (q2, d2))
    using FragM  = Array<float, 2>[V_ITER_M];            // ((q8, _4), Q1, q2) => fragS with all S dim reduced
    using FragL  = FragM;

    using TransformedK = Array<T, 4>[K_ITER_K][K_ITER_N];  // ((s8, d4), (D8, S8), (d2, d2))
    using TransformedV = Array<T, 4>[V_ITER_K][V_ITER_N];  // ((d8, s4), (S4, D16), (s2, s2))

    template<class Func>
    static __device__ void ApplyCasualMask(FragS& frag_S, Func&& func)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int m = 0; m < K_ITER_M; ++m) {  // Q
            PRAGMA_UNROLL
            for (int n = 0; n < K_ITER_N; ++n) {  // K
                PRAGMA_UNROLL
                for (int q = 0; q < 2; ++q) {
                    PRAGMA_UNROLL
                    for (int s = 0; s < 2; ++s) {
                        const int qi = m * OP_M + lane_id / 4 + q * 8 + warp_id * WARP_Q;
                        const int ki = n * OP_N + lane_id % 4 * 2 + s;
                        ((Func&&)func)(qi, ki, frag_S[m][n][q * 2 + s]);
                    }
                }
            }
        }
    }

    template<class Func>
    static __device__ void StoreS(const FragS& frag_S, Func&& func)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int m = 0; m < K_ITER_M; ++m) {  // Q
            PRAGMA_UNROLL
            for (int n = 0; n < K_ITER_N; ++n) {  // KV
                PRAGMA_UNROLL
                for (int q = 0; q < 2; ++q) {
                    PRAGMA_UNROLL
                    for (int s = 0; s < 2; ++s) {
                        const int qi = m * OP_M + lane_id / 4 + q * 8 + warp_id * WARP_Q;
                        const int ki = n * OP_N + lane_id % 4 * 2 + s;
                        ((Func&&)func)(qi, ki, frag_S[m][n][q * 2 + s]);
                    }
                }
            }
        }
    }

    template<class Func>
    static __device__ void StoreP(const FragP& frag_P, Func&& func)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        for (int m = 0; m < V_ITER_M; ++m) {
            for (int k = 0; k < V_ITER_K; ++k) {
                for (int s1 = 0; s1 < 2; ++s1) {
                    for (int q = 0; q < 2; ++q) {
                        for (int s0 = 0; s0 < 2; ++s0) {
                            const int qi = m * OP_M + lane_id / 4 + q * 8 + warp_id * WARP_Q;
                            const int vi = k * OP_K + lane_id % 4 * 2 + s1 * 8 + s0;
                            ((Func&&)func)(qi, vi, frag_P[m][k][s1 * 4 + q * 2 + s0]);
                        }
                    }
                }
            }
        }
    }

    template<bool is_residue>
    static __device__ void
    Softmax(FragS& frag_S, FragM& frag_M, FragM& frag_L, FragPs& frag_Ps, FragO& frag_O, float qk_scale)
    {
        FragM prev_M;
        PRAGMA_UNROLL
        for (int m = 0; m < K_ITER_M; ++m) {
            prev_M[m] = frag_M[m];
        }

        // maximum
        PRAGMA_UNROLL
        for (int m = 0; m < K_ITER_M; ++m) {  // Q
            auto& row_M = frag_M[m];
            PRAGMA_UNROLL
            for (int n = 0; n < K_ITER_N; ++n) {  // KV
                auto& C = frag_S[m][n];
                PRAGMA_UNROLL
                for (int q = 0; q < 2; ++q) {
                    row_M[q] = fmaxf(row_M[q], fmaxf(C[q * 2 + 0], C[q * 2 + 1]));  // reduce over local pair
                }
            }
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {  // reduce over thread group within warp (within warp tiles)
                row_M[q] = fmaxf(row_M[q], __shfl_xor_sync(uint32_t(-1), row_M[q], 1));
                row_M[q] = fmaxf(row_M[q], __shfl_xor_sync(uint32_t(-1), row_M[q], 2));
            }
        }

        PRAGMA_UNROLL
        for (int m = 0; m < K_ITER_M; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                // exp(M - M'), isinf(frag_M) => isnan(expdiff_M)
                float expdiff_M = exp2f((prev_M[m][q] - frag_M[m][q]) * qk_scale);
                if (is_residue && frag_M[m][q] == -std::numeric_limits<float>::infinity()) {
                    expdiff_M = 0.f;
                }
                for (int n = 0; n < V_ITER_N; ++n) {
                    for (int d = 0; d < 2; ++d) {
                        frag_O[m][n][q * 2 + d] = frag_O[m][n][q * 2 + d] * expdiff_M;  // Rescale previous output
                    }
                }
                frag_L[m][q] *= expdiff_M;
            }
        }

        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;

        PRAGMA_UNROLL
        for (int m = 0; m < K_ITER_M; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                float tmp_L{};
                PRAGMA_UNROLL
                for (int n = 0; n < K_ITER_N; ++n) {
                    PRAGMA_UNROLL
                    for (int s = 0; s < 2; ++s) {
                        // unnormalized prob, optimized to FFMA
                        float p = exp2f(frag_S[m][n][q * 2 + s] * qk_scale - frag_M[m][q] * qk_scale);
                        if (is_residue && frag_M[m][q] == -std::numeric_limits<float>::infinity()) {
                            p = 0.f;
                        }
                        tmp_L += p;
                        frag_S[m][n][q * 2 + s] = p;
                    }
                }
                tmp_L += __shfl_xor_sync(uint32_t(-1), tmp_L, 1);
                tmp_L += __shfl_xor_sync(uint32_t(-1), tmp_L, 2);
                frag_L[m][q] = frag_L[m][q] + tmp_L;  // update L
            }
        }

        PRAGMA_UNROLL
        for (int m = 0; m < K_ITER_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < K_ITER_N; ++n) {
                PRAGMA_UNROLL
                for (int q = 0; q < 2; ++q) {
                    PRAGMA_UNROLL
                    for (int s = 0; s < 2; ++s) {
                        frag_Ps[m][n][q * 2 + s] = static_cast<T>(frag_S[m][n][q * 2 + s]);
                    }
                }
            }
        }
    }

    template<class Func>
    static __device__ void StoreO(FragO& frag_O, const FragL& frag_L, Func&& func)
    {
        FragL tmp_L;
        PRAGMA_UNROLL
        for (int m = 0; m < V_ITER_M; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                tmp_L[m][q] = fdividef(1.f, frag_L[m][q] + 1e-8f);
            }
        }

        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        PRAGMA_UNROLL
        for (int m = 0; m < V_ITER_M; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                const int qi = m * OP_M + q * 8 + lane_id / 4 + warp_id * WARP_Q;
                PRAGMA_UNROLL
                for (int n = 0; n < V_ITER_N; ++n) {
                    Array<T, 2> tmp_O;
                    PRAGMA_UNROLL
                    for (int d = 0; d < 2; ++d) {
                        tmp_O[d] = (T)(frag_O[m][n][q * 2 + d] * tmp_L[m][q]);
                    }
                    const int di = n * 8 + lane_id % 4 * 2;
                    ((Func&&)func)(qi, di, tmp_O);
                }
            }
        }
    }
};

}  // namespace turbomind