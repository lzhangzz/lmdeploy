// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "array_ops.h"
#include "impl.h"
#include "iterator.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"
#include "thread_map.h"
#include <limits>

namespace turbomind::attention {

template<class T, class Layout, int WARP_S, int N>
struct SimtSmemIterK: BaseSmemIterator<T, Layout> {
    using Base = BaseSmemIterator<T, Layout>;
    using Base::Base;
    using Base::swizzle_ptr;

    __device__ void Load(Array<T, 8> (&frag_K)[N], int k, int offset)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int n = 0; n < N; ++n) {
            const int si = n * +4 + lane_id / 8 + warp_id * WARP_S;
            const int di = k * 64 + lane_id % 8 * 8;
            Lds(frag_K[n], offset + swizzle_ptr(si, di));
        }
    }
};

template<class T, class Layout, int WARP_S, int N>
struct SimtSmemIterV: BaseSmemIterator<T, Layout> {
    using Base = BaseSmemIterator<T, Layout>;
    using Base::Base;
    using Base::swizzle_ptr;

    __device__ void Load(Array<T, 8> (&frag_V)[N], int k, int offset)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int n = 0; n < N; ++n) {
            const int si = k * +4 + lane_id / 8 + warp_id * WARP_S;
            const int di = n * 64 + lane_id % 8 * 8;
            Lds(frag_V[n], offset + swizzle_ptr(si, di));
        }
    }
};

template<class T_, int CTA_Q_, int CTA_S_, int WARP_Q, int WARP_S, int HeadDim, int Stages>
struct Impl<Sm70_Simt, T_, T_, CTA_Q_, CTA_S_, WARP_Q, WARP_S, HeadDim, Stages> {
    using T   = T_;
    using Tkv = T_;

    static constexpr int CTA_Q    = CTA_Q_;
    static constexpr int CTA_S    = CTA_S_;
    static constexpr int kHeadDim = HeadDim;

    static constexpr int kWarpCntQ  = CTA_Q / WARP_Q;
    static constexpr int kWarpCntS  = CTA_S / WARP_S;
    static constexpr int kWarpCount = kWarpCntQ * kWarpCntS;

    static_assert(kWarpCntQ == 1);

    static constexpr int OP_Q = 1;
    static constexpr int OP_S = 4;
    static constexpr int OP_D = 64;

    static constexpr int K_M = WARP_Q / OP_Q;   // 1
    static constexpr int K_N = WARP_S / OP_S;   // 4
    static constexpr int K_K = HeadDim / OP_D;  // 2

    static_assert(WARP_Q % OP_Q == 0 && K_M > 0);
    static_assert(WARP_S % OP_S == 0 && K_N > 0);
    static_assert(HeadDim % OP_D == 0 && K_K > 0);

    static constexpr int V_M = WARP_Q / OP_Q;   // 1
    static constexpr int V_N = HeadDim / OP_D;  // 2
    static constexpr int V_K = WARP_S / OP_S;   // 4

    using Tqk = float;
    using Tpv = float;

    using FragQ = Array<T, 8>[K_K][K_M];      // (q4, d8), (Dk, Qm), (d8)
                                              //   0   8    64   1     1
    using FragK = Array<T, 8>[K_K][K_N];      // (s4, d8), (Dk, Sn), (d8)
                                              //   1   8    64   4     1
    using FragS = Array<float, 1>[K_M][K_N];  // (s4, d8), (Qm, Sn)
                                              //   1   8     1   4
                                              // (s4, _8), (Qm, Sn)       [after redsum]
                                              //   1   0     1   4
    using FragM = Array<float, 1>[K_M];       // (s4, _8), (Qm)
                                              //   1   0     1
    using FragP = Array<T, 1>[V_M][V_K];      // (s4, _8), (Qm, Sk), (s1)
                                              //   1   0     1   4     1
    using FragV = Array<T, 8>[V_K][V_N];      // (s4, d8), (Sk, Dn), (d8)
                                              //   1   8     4  64     1
    using FragO = Array<float, 8>[V_M][V_N];  // (s4, d8), (Qm, Dn), (d8)
                                              //   1   8     1  64     1
    using FragSp = Array<T, 1>[K_M][K_N];
    using FragL  = FragM;

    using SmemLayoutQ = SmemLayout<HeadDim, Identity>;
    using SmemLayoutK = SmemLayout<HeadDim, Identity>;
    using SmemLayoutP = SmemLayout<CTA_S, Identity>;
    using SmemLayoutV = SmemLayout<HeadDim, Identity>;

    using SmemM = float[K_M][kWarpCntS];
    using SmemL = float[K_M][kWarpCntS];
    using SmemO = Array<float, 4>[V_M][V_N][2][kWarpCntS][8];  // (Qm, Dn, d2, Sw, d8), (d4)
                                                               //   1  64   4  WS   8     1

    union SharedStorage {
        __align__(16) T Q[CTA_Q * SmemLayoutQ::kStride];

        __align__(16) T KV[Stages * CTA_S * (SmemLayoutK::kStride + SmemLayoutV::kStride) / 2];
        struct {
            __align__(16) T K[Stages == 2 ? CTA_S * SmemLayoutK::kStride : 1];
            __align__(16) T V[Stages == 2 ? CTA_S * SmemLayoutV::kStride : 1];
        };

        struct {
            __align__(16) SmemM M;
            __align__(16) SmemL L;
            __align__(16) SmemO O;
        };
        T P[1];
    };

    using SmemIterQ = NullSmemIter<T>;
    using SmemIterP = NullSmemIter<T>;

    using SmemIterK = SimtSmemIterK<T, SmemLayoutK, WARP_S, K_N>;
    using SmemIterV = SimtSmemIterV<T, SmemLayoutV, WARP_S, V_N>;

    static constexpr bool kUseSmemQ = false;
    static constexpr bool kUseSmemP = false;

    using ThreadMapQ  = RakedThreadMap<HeadDim, CTA_Q, 8, kWarpCount>;
    using ThreadMapKV = RakedThreadMap<HeadDim, CTA_S, 8, kWarpCount>;

    __device__ static void Sync() {}

    template<class Fragment, class Func>
    __device__ static void ForeachS(Fragment& S, Func&& func)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < K_N; ++n) {
                const int qi = m * OP_Q;
                const int si = n * OP_S + lane_id / 8 + warp_id * WARP_S;
                ((Func&&)func)(qi, si, S[m][n][0]);
            }
        }
    }

    __device__ static void TransformQ(const T* smem_Q, FragQ& frag_Q)
    {
        const int lane_id = threadIdx.x % WARP_SIZE;

        __syncthreads();

        PRAGMA_UNROLL
        for (int k = 0; k < K_K; ++k) {
            PRAGMA_UNROLL
            for (int m = 0; m < K_M; ++m) {
                const int qi = m;
                const int di = k * 64 + lane_id % 8 * 8;
                Lds(frag_Q[k][m], &smem_Q[SmemLayoutQ::swizzle(qi, di)]);
            }
        }
    }

    template<class SmemQ, class SmemK>
    __device__ static void ComputeQK(SmemQ&, SmemK& smem_K, FragQ& frag_Q, FragS& frag_S, int offset)
    {
        FragK frag_K;
        smem_K.Load(frag_K[0], 0, offset);
        PRAGMA_UNROLL
        for (int k = 0; k < K_K; ++k) {
            if (k < K_K - 1) {
                smem_K.Load(frag_K[k + 1], k + 1, offset);
            }
            PRAGMA_UNROLL
            for (int m = 0; m < K_M; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < K_N; ++n) {
                    PRAGMA_UNROLL
                    for (int c = 0; c < 8; ++c) {
                        frag_S[m][n][0] += static_cast<float>((Tqk)frag_Q[k][m][c] * (Tqk)frag_K[k][n][c]);
                    }
                }
            }
        }

        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < K_N; ++n) {
                frag_S[m][n][0] += __shfl_xor_sync(uint32_t(-1), frag_S[m][n][0], 1);
                frag_S[m][n][0] += __shfl_xor_sync(uint32_t(-1), frag_S[m][n][0], 2);
                frag_S[m][n][0] += __shfl_xor_sync(uint32_t(-1), frag_S[m][n][0], 4);
            }
        }
    }

    template<class SmemP, class SmemV>
    __device__ static void ComputePV(SmemP&, SmemV& smem_V, const FragP& frag_P, FragO& frag_O, int offset)
    {
        FragV frag_V;
        smem_V.Load(frag_V[0], 0, offset);

        PRAGMA_UNROLL
        for (int k = 0; k < V_K; ++k) {
            if (k < V_K - 1) {
                smem_V.Load(frag_V[k + 1], k + 1, offset);
            }
            PRAGMA_UNROLL
            for (int m = 0; m < V_M; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < V_N; ++n) {
                    PRAGMA_UNROLL
                    for (int d = 0; d < 8; ++d) {
                        frag_O[m][n][d] += static_cast<float>((Tpv)frag_P[m][k][0] * (Tpv)frag_V[k][n][d]);
                    }
                }
            }
        }
    }

    template<bool is_residue>
    __device__ static void Softmax(FragS& frag_S, FragM& frag_M, FragL& frag_L, FragO& frag_O, float qk_scale)
    {

        FragM prev_M;
        copy(frag_M, prev_M);

        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < K_N; ++n) {
                frag_M[m][0] = fmaxf(frag_M[m][0], frag_S[m][n][0]);
            }
        }

        PRAGMA_UNROLL
        for (int m = 0; m < V_M; ++m) {
            float expdiff_M = exp2f((prev_M[m][0] - frag_M[m][0]) * qk_scale);
            if (is_residue && frag_M[m][0] == -std::numeric_limits<float>::infinity()) {
                expdiff_M = 0.f;
            }
            PRAGMA_UNROLL
            for (int n = 0; n < V_N; ++n) {
                using namespace ops;
                frag_O[m][n] = frag_O[m][n] * expdiff_M;
            }
            frag_L[m][0] *= expdiff_M;
        }

        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            float tmp_L{};
            PRAGMA_UNROLL
            for (int n = 0; n < K_N; ++n) {
                float p = exp2f(frag_S[m][n][0] * qk_scale - frag_M[m][0] * qk_scale);
                if (is_residue && frag_M[m][0] == -std::numeric_limits<float>::infinity()) {
                    p = 0.f;
                }
                tmp_L += p;
                frag_S[m][n][0] = p;
            }
            frag_L[m][0] += tmp_L;
        }
    }

    __device__ static void ConvertStoP(FragS& frag_S, FragP& frag_P, T* smem_P)
    {
        FragSp& frag_Sp = (FragSp&)frag_P;
        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < K_N; ++n) {
                frag_Sp[m][n][0] = static_cast<T>(frag_S[m][n][0]);
            }
        }
    }

    __device__ static void Merge(FragO& frag_O, FragM& frag_M, FragL& frag_L, float qk_scale, SharedStorage& storage)
    {
        const int warp_id_s = threadIdx.x / WARP_SIZE;
        const int lane_id   = threadIdx.x % WARP_SIZE;

        FragM prev_M;
        copy(frag_M, prev_M);

        __syncthreads();

        /////////////////////////////////////////////////////////////////////////
        //  global max
        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            frag_M[m][0] = fmaxf(frag_M[m][0], __shfl_xor_sync(uint32_t(-1), frag_M[m][0], 8));
            frag_M[m][0] = fmaxf(frag_M[m][0], __shfl_xor_sync(uint32_t(-1), frag_M[m][0], 16));
            if (lane_id == 0) {
                storage.M[m][warp_id_s] = frag_M[m][0];
            }
        }

        __syncthreads();

        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            PRAGMA_UNROLL
            for (int w = 0; w < kWarpCntS - 1; ++w) {
                frag_M[m][0] = fmaxf(frag_M[m][0], storage.M[m][(warp_id_s + w + 1) % kWarpCntS]);
            }
        }

        ///////////////////////////////////////////////////////////////////////////
        //  rescale & global sum
        PRAGMA_UNROLL
        for (int m = 0; m < V_M; ++m) {
            float expdiff_M = exp2f((prev_M[m][0] - frag_M[m][0]) * qk_scale);
            if (frag_M[m][0] == -std::numeric_limits<float>::infinity()) {
                expdiff_M = 0.f;
            }
            PRAGMA_UNROLL
            for (int n = 0; n < V_N; ++n) {
                PRAGMA_UNROLL
                for (int d = 0; d < 8; ++d) {
                    frag_O[m][n][d] = frag_O[m][n][d] * expdiff_M;
                    frag_O[m][n][d] += __shfl_xor_sync(uint32_t(-1), frag_O[m][n][d], 8);
                    frag_O[m][n][d] += __shfl_xor_sync(uint32_t(-1), frag_O[m][n][d], 16);
                }
                PRAGMA_UNROLL
                for (int d = 0; d < 8; d += 4) {
                    if (lane_id < 8) {
                        Store(storage.O[m][n][d / 4][warp_id_s][lane_id].data(), (Array<float, 4>&)frag_O[m][n][d]);
                    }
                }
            }
            frag_L[m][0] *= expdiff_M;
            frag_L[m][0] += __shfl_xor_sync(uint32_t(-1), frag_L[m][0], 8);
            frag_L[m][0] += __shfl_xor_sync(uint32_t(-1), frag_L[m][0], 16);
            if (lane_id == 0) {
                storage.L[m][warp_id_s] = frag_L[m][0];
            }
        }

        __syncthreads();

        clear(frag_O);

        PRAGMA_UNROLL
        for (int m = 0; m < V_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < V_N; ++n) {
                static_assert(kWarpCntS % 4 == 0);
                PRAGMA_UNROLL
                for (int s = 0; s < kWarpCntS; s += 4) {
                    PRAGMA_UNROLL
                    for (int d = 0; d < 8; d += 4) {
                        Array<float, 4> tmp_O;
                        Lds(tmp_O, storage.O[m][n][d / 4][s + lane_id / 8][lane_id % 8].data());
                        using namespace ops;
                        (Array<float, 4>&)frag_O[m][n][d] = (Array<float, 4>&)frag_O[m][n][d] + tmp_O;
                    }
                }
                PRAGMA_UNROLL
                for (int d = 0; d < 8; ++d) {
                    frag_O[m][n][d] += __shfl_xor_sync(uint32_t(-1), frag_O[m][n][d], 8);
                    frag_O[m][n][d] += __shfl_xor_sync(uint32_t(-1), frag_O[m][n][d], 16);
                }
            }
            PRAGMA_UNROLL
            for (int w = 0; w < kWarpCntS - 1; ++w) {
                frag_L[m][0] += storage.L[m][(warp_id_s + w + 1) % kWarpCntS];
            }
        }
    }

    template<class Func>
    __device__ static void StoreO(FragO& frag_O, const FragL& frag_L, Func&& func)
    {
        FragL inv_L;

        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            inv_L[m][0] = fdividef(1.f, frag_L[m][0]);
        }

        const int warp_id_s = threadIdx.x / WARP_SIZE;
        const int lane_id   = threadIdx.x % WARP_SIZE;

        if (warp_id_s != 0) {
            return;
        }

        PRAGMA_UNROLL
        for (int m = 0; m < V_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < V_N; ++n) {
                Array<T, 8> tmp_O;
                PRAGMA_UNROLL
                for (int d = 0; d < 8; ++d) {
                    tmp_O[d] = static_cast<T>(frag_O[m][n][d] * inv_L[m][0]);
                }
                if (lane_id < 8) {
                    const int qi = m;
                    const int di = n * OP_D + lane_id % 8 * 8;
                    ((Func&&)func)(qi, di, tmp_O);
                }
            }
        }
    }
};

}  // namespace turbomind::attention