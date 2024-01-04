// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "impl.h"
#include "iterator_sm80.h"
#include "src/turbomind/kernels/attention/thread_map.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"

namespace turbomind::attention {

template<class T_, int CTA_Q_, int CTA_S_, int WARP_Q, int WARP_S, int HeadDim>
struct Impl<Sm80_16816, T_, T_, CTA_Q_, CTA_S_, WARP_Q, WARP_S, HeadDim> {

    using T   = T_;
    using Tkv = T_;

    static constexpr int CTA_Q    = CTA_Q_;
    static constexpr int CTA_S    = CTA_S_;
    static constexpr int kHeadDim = HeadDim;

    static constexpr int kWarpCntQ  = CTA_Q / WARP_Q;
    static constexpr int kWarpCntS  = CTA_S / WARP_S;
    static constexpr int kWarpCount = kWarpCntQ * kWarpCntS;

    static constexpr int OP_M = 16;
    static constexpr int OP_N = 8;
    static constexpr int OP_K = 16;

    static constexpr int K_M = WARP_Q / OP_M;   //  16 / 16 = 1
    static constexpr int K_N = WARP_S / OP_N;   //  64 /  8 = 8
    static constexpr int K_K = HeadDim / OP_K;  // 128 / 16 = 8

    static constexpr int V_M = WARP_Q / OP_M;   //  16 / 16 = 1
    static constexpr int V_N = HeadDim / OP_N;  // 128 /  8 = 16 -> D16
    static constexpr int V_K = WARP_S / OP_K;   //  64 / 16 = 4  -> S4

    using FragQ = Array<T, 8>[K_K][K_M];      // ((q8, d4), (Dk, Qm), (d2, q2, d2))
                                              //    1   2    16  16     8   8   1
    using FragK = Array<T, 4>[K_K][K_N];      // ((s8, d4), (Dk, Sn), (d2, d2))
                                              //    1   2    16   8     8   1
    using FragS = Array<float, 4>[K_M][K_N];  // ((q8, s4), (Qm, Sn), (q2, s2))
                                              //    1   2    16   8     8   1
    using FragPs = Array<T, 4>[K_M][K_N];     // ((q8, s4), (Qm, Sn), (q2, s2))
                                              //    1   2    16   8     8   1
    using FragP = Array<T, 8>[V_M][V_K];      // ((q8, s4), (Qm, Sk), (s2, q2, s2))
                                              //    1   2    16  16     8   8   1
    using FragV = Array<T, 4>[V_K][V_N];      // ((d8, s4), (Sk, Dn), (s2, s2))
                                              //    1   2    16   8     8   1
    using FragO = Array<float, 4>[V_M][V_N];  // ((q8, d4), (Qm, Dn), (q2, d2))
                                              //    1   2    16   8     8   1
    using FragM = Array<float, 2>[V_M];       // ((q8, x4), Qm, q2) => FragS with all S dim reduced
                                              //    1   0   16   8
    using FragL = FragM;

    static_assert(sizeof(FragPs) == sizeof(FragP));

    struct Swizzle {
        __device__ int operator()(int index)
        {
            // sssSSSdDDDddd
            // DDD ^= SSS
            constexpr int mask = 0x7 << 7;
            return index ^ ((index & mask) >> 4);
        }
    };

    using SmemLayoutQ = SmemLayout<HeadDim, Swizzle>;
    using SmemLayoutK = SmemLayout<HeadDim, Swizzle>;
    using SmemLayoutP = Identity;
    using SmemLayoutV = SmemLayout<HeadDim, Swizzle>;

    struct SharedStorage {
        union {
            __align__(16) T Q[CTA_Q * SmemLayoutQ::kStride];
            struct {
                __align__(16) T K[CTA_S * SmemLayoutK::kStride];
                __align__(16) T V[CTA_S * SmemLayoutV::kStride];
            };
            struct {
                T P[1];
            };
        };
    };

    using SmemIterQ = NullSmemIter<T>;
    using SmemIterP = NullSmemIter<T>;

    using SmemIterK = Sm80SmemIterK<T, SmemLayoutK, K_N>;
    using SmemIterV = Sm80SmemIterV<T, SmemLayoutV, V_N>;

    static constexpr bool kUseSmemQ = false;

    using ThreadMapQ  = RakedThreadMap<HeadDim, CTA_Q, 8, kWarpCount>;
    using ThreadMapKV = RakedThreadMap<HeadDim, CTA_S, 8, kWarpCount>;

    template<class Fragment, class Func>
    __device__ static void ForeachS(Fragment& S, Func&& func)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {  // Q
            PRAGMA_UNROLL
            for (int n = 0; n < K_N; ++n) {  // KV
                PRAGMA_UNROLL
                for (int q = 0; q < 2; ++q) {
                    PRAGMA_UNROLL
                    for (int s = 0; s < 2; ++s) {
                        const int qi = m * OP_M + lane_id / 4 + q * 8 + warp_id * WARP_Q;
                        const int ki = n * OP_N + lane_id % 4 * 2 + s;
                        ((Func&&)func)(qi, ki, S[m][n][q * 2 + s]);
                    }
                }
            }
        }
    }

    template<class Func>
    __device__ static void ForeachP(FragP& P, Func&& func)
    {
        ForeachS((FragPs&)P, (Func&&)func);
    }

    __device__ static void TransformQ(const T* smem_Q, FragQ& frag_Q)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        __syncwarp();

        uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_Q);
        if constexpr (!kUseSmemQ) {
            // Load from shared memory using LDSM, rearrange to m16n8k16 atom layout
            PRAGMA_UNROLL
            for (int m = 0; m < K_M; ++m) {
                PRAGMA_UNROLL
                for (int k = 0; k < K_K; ++k) {
                    auto&     Q      = (Array<uint32_t, 4>&)frag_Q[k][m];
                    const int qi     = m * 16 + lane_id % 16 + warp_id * WARP_Q;
                    const int di     = k * 16 + lane_id / 16 * 8;
                    const int offset = sizeof(T) * SmemLayoutQ::swizzle(qi, di);
                    ldmatrix_m8n8_x4_b16(Q[0], Q[1], Q[2], Q[3], smem_int_ptr + offset);
                }
            }
        }
        else {
            // Rearrange Q in smem so that swizzling is not needed for later LDSMs
            const int group_id      = lane_id / 16;
            const int group_lane_id = lane_id % 16;
            PRAGMA_UNROLL
            for (int k = 0; k < K_K; ++k) {
                PRAGMA_UNROLL
                for (int m = 0; m < K_M; ++m) {
                    auto&     r      = (Array<uint32_t, 4>&)frag_Q[k][m];
                    const int qi     = m * 16 + group_lane_id % 8 + group_id * 8 + warp_id * WARP_Q;
                    const int di     = k * 16 + group_lane_id / 8 * 8;
                    const int offset = sizeof(T) * SmemLayoutQ::swizzle(qi, di);
                    ldmatrix_m8n8_x4_b16(r[0], r[2], r[1], r[3], smem_int_ptr + offset);
                }
            }

            __syncthreads();

            constexpr int THREADS = kWarpCount * WARP_SIZE;
            PRAGMA_UNROLL
            for (int k = 0; k < K_K; ++k) {
                PRAGMA_UNROLL
                for (int m = 0; m < K_M; ++m) {
                    constexpr int kVecSize = 8;
                    Store(&smem_Q[(k * K_M * THREADS + m * THREADS + threadIdx.x) * kVecSize], frag_Q[k][m]);
                }
            }
        }
    }

    template<class SmemQ, class SmemK>
    __device__ static void ComputeQK(SmemQ& smem_Q, SmemK& smem_K, FragQ& frag_Q, FragS& frag_S)
    {
        FragK frag_K;

        smem_K.Load(frag_K[0], 0);
        if constexpr (kUseSmemQ) {
            smem_Q.Load(frag_Q[0], 0);
        }

        PRAGMA_UNROLL
        for (int k = 0; k < K_K; ++k) {
            if (k < K_K - 1) {
                smem_K.Load(frag_K[k + 1], k + 1);
                if constexpr (kUseSmemQ) {
                    smem_Q.Load(frag_Q[k + 1], k + 1);
                }
            }
            PRAGMA_UNROLL
            for (int m = 0; m < K_M; ++m) {
                PRAGMA_UNROLL
                for (int n = K_N - 1; n >= 0; --n) {
                    mma_m16n8k16_row_col(frag_S[m][n], frag_Q[k][m], frag_K[k][n], frag_S[m][n]);
                }
            }
        }
    }

    template<class SmemP, class SmemV>
    __device__ static void ComputePV(SmemP&, SmemV& smem_V, const FragP& frag_P, FragO& frag_O)
    {
        FragV frag_V;

        smem_V.Load(frag_V[0], 0);

        PRAGMA_UNROLL
        for (int k = 0; k < V_K; ++k) {
            if (k < V_K - 1) {
                smem_V.Load(frag_V[k + 1], k + 1);
            }
            PRAGMA_UNROLL
            for (int m = 0; m < V_M; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < V_N; ++n) {
                    mma_m16n8k16_row_col(frag_O[m][n], frag_P[m][k], frag_V[k][n], frag_O[m][n]);
                }
            }
        }
    }

    template<bool is_residue>
    __device__ static void Softmax(FragS& frag_S, FragM& frag_M, FragM& frag_L, FragO& frag_O, float qk_scale)
    {
        FragM prev_M;
        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            prev_M[m] = frag_M[m];
        }

        // maximum
        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {  // Q
            auto& row_M = frag_M[m];
            PRAGMA_UNROLL
            for (int n = 0; n < K_N; ++n) {  // KV
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
        for (int m = 0; m < K_M; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                // exp(M - M'), isinf(frag_M) => isnan(expdiff_M)
                float expdiff_M = exp2f((prev_M[m][q] - frag_M[m][q]) * qk_scale);
                if (is_residue && frag_M[m][q] == -std::numeric_limits<float>::infinity()) {
                    expdiff_M = 0.f;
                }
                for (int n = 0; n < V_N; ++n) {
                    for (int d = 0; d < 2; ++d) {
                        frag_O[m][n][q * 2 + d] = frag_O[m][n][q * 2 + d] * expdiff_M;  // Rescale previous output
                    }
                }
                frag_L[m][q] *= expdiff_M;
            }
        }

        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                float tmp_L{};
                PRAGMA_UNROLL
                for (int n = 0; n < K_N; ++n) {
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
    }

    __device__ static void ConvertStoP(FragS& frag_S, FragP& frag_P, T* smem_P)
    {
        FragPs& frag_Ps = (FragPs&)frag_P;
        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < K_N; ++n) {
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
    __device__ static void StoreO(FragO& frag_O, const FragL& frag_L, Func&& func)
    {
        FragL tmp_L;
        PRAGMA_UNROLL
        for (int m = 0; m < V_M; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                tmp_L[m][q] = fdividef(1.f, frag_L[m][q] + 1e-8f);
            }
        }

        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        PRAGMA_UNROLL
        for (int m = 0; m < V_M; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                const int qi = m * OP_M + q * 8 + lane_id / 4 + warp_id * WARP_Q;
                PRAGMA_UNROLL
                for (int n = 0; n < V_N; ++n) {
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

}  // namespace turbomind::attention