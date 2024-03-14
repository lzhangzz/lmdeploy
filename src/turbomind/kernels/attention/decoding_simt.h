// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "array_ops.h"
#include "impl.h"
#include "iterator.h"
#include "src/turbomind/kernels/attention/data_type.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"
#include "thread_map.h"
#include <limits>

namespace turbomind::attention {

template<class T, class Layout, int M>
struct SimtSmemIterQ: BaseSmemIterator<T, Layout> {
    using Base = BaseSmemIterator<T, Layout>;
    using Base::Base;
    using Base::swizzle_ptr;

    __device__ void Load(Array<T, 8> (&frag_Q)[M], int k)
    {
        const int lane_id = threadIdx.x % WARP_SIZE;

        PRAGMA_UNROLL
        for (int m = 0; m < M; ++m) {
            const int hi = m;
            const int di = k * 64 + lane_id % 8 * 8;
            Lds(frag_Q[m], swizzle_ptr(hi, di));
        }
    }
};

template<class T, class Layout, int WARP_S, int N>
struct SimtSmemIterK: BaseSmemIterator<T, Layout> {
    using Base = BaseSmemIterator<T, Layout>;
    using Base::Base;
    using Base::smem_;

    __device__ void Load(Array<T, 8> (&frag_K)[N], int k, int offset)
    {
        const int warp_id  = threadIdx.x / WARP_SIZE;
        const int lane_id  = threadIdx.x % WARP_SIZE;
        const int offset_s = lane_id / 8 + warp_id * WARP_S;
        const int offset_c = lane_id % 8 * 8;
        PRAGMA_UNROLL
        for (int n = 0; n < N; ++n) {
            const int si = n * 4 + offset_s;
            const int di = k * 64 + offset_c;
            Lds(frag_K[n], &smem_[offset + Layout::apply(si, di)]);
        }
    }
};

template<class T, class Layout, int WARP_S, int N>
struct SimtSmemIterV: BaseSmemIterator<T, Layout> {
    using Base = BaseSmemIterator<T, Layout>;
    using Base::Base;
    using Base::smem_;

    __device__ void Load(Array<T, 8> (&frag_V)[N], int k, int offset)
    {
        const int warp_id  = threadIdx.x / WARP_SIZE;
        const int lane_id  = threadIdx.x % WARP_SIZE;
        const int offset_s = lane_id / 8 + warp_id * WARP_S;
        const int offset_c = lane_id % 8 * 8;
        PRAGMA_UNROLL
        for (int n = 0; n < N; ++n) {
            const int si = k * 4 + offset_s;
            const int di = n * 64 + offset_c;
            Lds(frag_V[n], &smem_[offset + Layout::apply(si, di)]);
        }
    }
};

template<class T_,
         class Tkv_,
         int CTA_H_,
         int CTA_Q_,
         int CTA_S_,
         int WARP_H_,
         int WARP_Q,
         int WARP_S,
         int HeadDim,
         int Stages>
struct Impl<Sm70_Simt, T_, Tkv_, CTA_H_, CTA_Q_, CTA_S_, WARP_H_, WARP_Q, WARP_S, HeadDim, Stages> {
    using T   = T_;
    using Tkv = Tkv_;

    using Arch = Sm70_Simt;

    static constexpr int CTA_H = CTA_H_;
    static constexpr int CTA_Q = CTA_Q_;
    static constexpr int CTA_S = CTA_S_;

    static constexpr int WARP_H = WARP_H_;

    static constexpr int kHeadDim = HeadDim;

    static constexpr int kWarpCntH = CTA_H / WARP_H;
    static constexpr int kWarpCntQ = CTA_Q / WARP_Q;
    static constexpr int kWarpCntS = CTA_S / WARP_S;

    static constexpr int kWarpCount = kWarpCntQ * kWarpCntS;

    static_assert(CTA_H_ == WARP_H_);

    static_assert(kWarpCntQ == 1);

    static constexpr int OP_H = 1;
    static constexpr int OP_S = 4;
    static constexpr int OP_D = 64;

    static constexpr int K_M = WARP_H / OP_H;   // 1
    static constexpr int K_N = WARP_S / OP_S;   // 4
    static constexpr int K_K = HeadDim / OP_D;  // 2

    static_assert(WARP_H % OP_H == 0 && K_M > 0);
    static_assert(WARP_S % OP_S == 0 && K_N > 0);
    static_assert(HeadDim % OP_D == 0 && K_K > 0);

    static constexpr int V_M = WARP_H / OP_H;   // 1
    static constexpr int V_N = HeadDim / OP_D;  // 2
    static constexpr int V_K = WARP_S / OP_S;   // 4

    using Tqk = T;
    using Tpv = T;

    using FragQ = Array<T, 8>[K_K][K_M];      // (q4, d8), (Dk, Qm), (d8)
                                              //   0  16     8   1     1
    template<class Tk>                        //
    using FragK_ = Array<Tk, 8>[K_N][K_K];    // (s4, d8), (Sn, Dk), (d8)
                                              //   1  16     4   8     1
    using FragS = Array<float, 1>[K_M][K_N];  // (s4, d8), (Qm, Sn)
                                              //   1  16     1   4
                                              // (s4, _8), (Qm, Sn)       [after redsum]
                                              //   1   0     1   4
    using FragM = Array<float, 1>[K_M];       // (s4, _8), (Qm)
                                              //   1   0     1
    using FragP = Array<Tpv, 1>[V_M][V_K];    // (s4, _8), (Qm, Sk), (s1)
                                              //   1   0     1   4     1
    template<class Tv>                        //
    using FragV_ = Array<Tv, 8>[V_K][V_N];    // (s4, d8), (Sk, Dn), (d8)
                                              //   1  16     4   8     1
    using FragO = Array<float, 8>[V_M][V_N];  // (s4, d8), (Qm, Dn), (d8)
                                              //   1  16     1   8     1
    using ParamK = Array<T, 2>[K_N];          // (s4, x8), (Sn)
                                              //   1   0     4
    using ParamV = Array<T, 2>[V_K];          // (s4, x8), (Sk)
                                              //   1   0     4
    using FragSp = Array<Tpv, 1>[K_M][K_N];

    static_assert(sizeof(FragP) == sizeof(FragSp));

    using DataK = FragK_<Tkv>;
    using DataV = FragV_<Tkv>;

    using FragK = FragK_<Tqk>;
    using FragV = FragV_<Tpv>;

    using FragL = FragM;

    using SmemLayoutQ = SmemLayoutV2<CTA_S, HeadDim, 1, 1, Identity>;
    using SmemLayoutP = SmemLayoutV2<CTA_H, CTA_S, 1, 1, Identity>;
    using SmemLayoutK = SmemLayoutV2<CTA_S, HeadDim, CTA_S, HeadDim, Identity>;
    using SmemLayoutV = SmemLayoutV2<CTA_S, HeadDim, CTA_S, HeadDim, Identity>;

    using SmemLayoutKVp = SmemLayoutV2<CTA_S, 2, CTA_S, 2, Identity>;

    using SmemM = float[K_M][kWarpCntH][kWarpCntS];
    using SmemL = float[K_M][kWarpCntH][kWarpCntS];
    using SmemO = Array<float, 4>[V_M][V_N][2][kWarpCntH][kWarpCntS][8];  // (Qm, Dn, d2, Hw, Sw, d8), (d4)
                                                                          //   1  64   4  WH  WS   8     1

    using PointerKV = get_pointer_type<Tkv>;

    union SharedStorage {
        __align__(16) T Q[SmemLayoutQ::kSize];

        struct {
            // __align__(16) Tkv KV[Stages * SmemLayoutK::kSize];
            __align__(16) Array<Tkv, Stages * SmemLayoutK::kSize> KV;
            __align__(16) T KVp[Stages * SmemLayoutKVp::kSize];
        };

        struct {
            __align__(16) SmemM M;
            __align__(16) SmemL L;
            __align__(16) SmemO O;
        };
        T P[1];
    };

    static constexpr bool kUseSmemQ = false;
    static constexpr bool kUseSmemP = false;

    using ThreadMapQ  = RakedThreadMap<HeadDim, CTA_H, 8, kWarpCount>;
    using ThreadMapKV = RakedThreadMap<HeadDim, CTA_S, 128 / bitsof<Tkv>, kWarpCount>;
    // `WARP_SIZE / WARP_S` is chosen to achieve minimum kIterS w/o introducing partial S iter
    using ThreadMapKVp = RakedThreadMap<2, CTA_S, 2, kWarpCount, WARP_SIZE / WARP_S>;

    static constexpr int kBatchK = ThreadMapKV::kIterS;
    static constexpr int kBatchV = ThreadMapKV::kIterS;

    __device__ static void Sync()
    {
        if constexpr (!std::is_same_v<T, Tkv>) {
            __syncwarp();
            // __syncthreads();
        }
    }

    template<class GmemIterK, class GmemIterV>
    __device__ static void SetSmemKV(GmemIterK& gmem_K, GmemIterV& gmem_V, SharedStorage& storage, bool offset_kv)
    {
        int pred = offset_kv;
        if constexpr (std::is_same_v<T, Tkv>) {
            gmem_K.SetSmem(storage.KV.data());
            gmem_V.SetSmem(storage.KV.data() + pred * SmemLayoutK::kSize);
        }
        else {
            gmem_K.SetSmem(storage.KV.data(), storage.KVp);
            gmem_V.SetSmem(storage.KV.data() + pred * SmemLayoutK::kSize, storage.KVp + pred * SmemLayoutKVp::kSize);
        }
    }

    template<class Func>
    __device__ static void ForeachML(FragM& frag_M, FragL& frag_L, Func&& func)
    {
        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {  // Q
            const int hi = m * OP_H;
            const int ri = threadIdx.x;
            ((Func&&)func)(hi, 0, ri, frag_M[m][0], frag_L[m][0]);
        }
    }

    template<class Fragment, class Func>
    __device__ static void ForeachS(Fragment& S, Func&& func)
    {
        const int warp_id   = threadIdx.x / WARP_SIZE;
        const int lane_id   = threadIdx.x % WARP_SIZE;
        const int warp_id_s = warp_id % kWarpCntS;
        const int warp_id_h = warp_id / kWarpCntS;
        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < K_N; ++n) {
                const int hi = m * OP_H + warp_id_h * WARP_H;
                const int si = n * OP_S + lane_id / 8 + warp_id_s * WARP_S;
                const int ri = lane_id % 8;
                ((Func&&)func)(hi, /*qi*/ 0, si, ri, S[m][n][0]);
            }
        }
    }

    __device__ static void TransformQ(T* smem_Q, FragQ& frag_Q)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        const int warp_id_h = warp_id / kWarpCntS;

        __syncthreads();

        SmemAccessor<T, SmemLayoutQ> sQ{smem_Q};

        PRAGMA_UNROLL
        for (int k = 0; k < K_K; ++k) {
            PRAGMA_UNROLL
            for (int m = 0; m < K_M; ++m) {
                const int hi = m + warp_id_h * WARP_H;
                const int di = k * 8 + lane_id % 8 * 16;
                Lds(frag_Q[k][m], &sQ(hi, di));
            }
        }
    }

    struct StateQK {
        PointerKV smem_K;
        T*        smem_K_param;
        FragQ     frag_Q;
        FragK     frag_K;
        DataK     data_K;
        ParamK    param_K;

        __device__ StateQK(SharedStorage& storage, FragQ frag_Q_)
        {
            smem_K       = storage.KV.data();
            smem_K_param = storage.KVp;
            if constexpr (!kUseSmemQ) {
                PRAGMA_UNROLL
                for (int k = 0; k < K_K; ++k) {
                    PRAGMA_UNROLL
                    for (int n = 0; n < K_N; ++n) {
                        frag_Q[n][k] = frag_Q_[n][k];
                    }
                }
            }
        }

#if 0
        __device__ void Load(int k, int pipe_iter)
        {
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;

            if (!std::is_same_v<T, Tkv> && k == 0) {
                const int offset_s = lane_id / 8 + warp_id * WARP_S;
                PRAGMA_UNROLL
                for (int n = 0; n < K_N; ++n) {
                    const int si = n * 4 + offset_s;
                    Lds(param_K[n], &smem_K_param[pipe_iter * SmemLayoutKVp::kSize + SmemLayoutKVp::apply(si, 0)]);
                }
            }

            {
                const int offset_s = lane_id / 8 + warp_id * WARP_S;
                const int offset_c = lane_id % 8 * 16;
                if constexpr (bitsof<Tkv> != 4) {
                    PRAGMA_UNROLL
                    for (int n = 0; n < K_N; ++n) {
                        const int si = n * 4 + offset_s;
                        const int di = k * 8 + offset_c;
                        Lds(data_K[n][k], &smem_K[pipe_iter * SmemLayoutK::kSize + SmemLayoutK::apply(si, di)]);
                    }
                }
                else {
                    if (k % 2 == 0) {
                        PRAGMA_UNROLL
                        for (int n = 0; n < K_N; ++n) {
                            const int si = n * 4 + offset_s;
                            const int di = k * 8 + offset_c;
                            Lds((Array<Tkv, 16>&)data_K[n][k],
                                &smem_K[pipe_iter * SmemLayoutK::kSize + SmemLayoutK::apply(si, di)]);
                        }
                    }
                }
            }
        }
#else
        __device__ void Load(int n, int pipe_iter)
        {
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;

            if (!std::is_same_v<T, Tkv>) {
                const int offset_s = lane_id / 8 + warp_id * WARP_S;
                PRAGMA_UNROLL
                for (int n = 0; n < K_N; ++n) {
                    const int si = n * 4 + offset_s;
                    Lds(param_K[n], &smem_K_param[pipe_iter * SmemLayoutKVp::kSize + SmemLayoutKVp::apply(si, 0)]);
                }
            }

            {
                const int offset_s = lane_id / 8 + warp_id * WARP_S;
                const int offset_c = lane_id % 8 * 16;
                if constexpr (bitsof<Tkv> != 4) {
                    PRAGMA_UNROLL
                    for (int k = 0; k < K_K; ++k) {
                        const int si = n * 4 + offset_s;
                        const int di = k * 8 + offset_c;
                        Lds(data_K[n][k], &smem_K[pipe_iter * SmemLayoutK::kSize + SmemLayoutK::apply(si, di)]);
                    }
                }
                else {
                    PRAGMA_UNROLL
                    for (int k = 0; k < K_K; k += 2) {
                        const int si = n * 4 + offset_s;
                        const int di = k * 8 + offset_c;
                        Lds((Array<Tkv, 16>&)data_K[n][k],
                            &smem_K[pipe_iter * SmemLayoutK::kSize + SmemLayoutK::apply(si, di)]);
                    }
                }
            }
        }
#endif

        __device__ void Transform(int n)
        {
            PRAGMA_UNROLL
            for (int k = 0; k < K_K; ++k) {
                ConvertKvCache<Tkv, Tqk> convert(param_K[n][0], param_K[n][1]);
                frag_K[n][k] = convert(data_K[n][k]);
            }
        }
    };

    template<class Prefetch, class Preload>
    __device__ static void
    ComputeQK(StateQK state_QK, FragS& frag_S, int offset, Prefetch&& prefetch, Preload&& preload)
    {
#if 0
        PRAGMA_UNROLL
        for (int k = 0; k < K_K; ++k) {
            if (k < K_K - 1) {
                state_QK.Load(k + 1, offset);
            }
            else {
                ((Preload&&)preload)();
            }

            state_QK.Transform(k);

            PRAGMA_UNROLL
            for (int m = 0; m < K_M; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < K_N; ++n) {
                    PRAGMA_UNROLL
                    for (int c = 0; c < 8; ++c) {
                        frag_S[m][n][0] += static_cast<float>((Tqk)state_QK.frag_Q[k][m][c] * state_QK.frag_K[n][k][c]);
                    }
                }
            }

            if (k < K_K - 1) {
                ((Prefetch&&)prefetch)(k);
            }
            if (k == K_K - 2) {
                ((Prefetch&&)prefetch)(K_K - 1);
            }
        }
#else
        PRAGMA_UNROLL
        for (int n = 0; n < K_N; ++n) {
            if (n < K_N - 1) {
                state_QK.Load(n + 1, offset);
            }
            else {
                ((Preload&&)preload)();
            }

            state_QK.Transform(n);

            PRAGMA_UNROLL
            for (int m = 0; m < K_M; ++m) {
                PRAGMA_UNROLL
                for (int k = 0; k < K_K; ++k) {
                    PRAGMA_UNROLL
                    for (int c = 0; c < 8; ++c) {
                        frag_S[m][n][0] += static_cast<float>((Tqk)state_QK.frag_Q[k][m][c] * state_QK.frag_K[n][k][c]);
                    }
                }
            }

            if (n < K_N - 1) {
                ((Prefetch&&)prefetch)(n);
            }
            if (n == K_N - 2) {
                ((Prefetch&&)prefetch)(K_N - 1);
            }
        }
#endif

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

    struct StatePV {
        PointerKV smem_V;
        T*        smem_V_param;
        FragP     frag_P;
        FragV     frag_V;
        DataV     data_V;
        ParamV    param_V;

        __device__ StatePV(SharedStorage& storage)
        {
            smem_V       = storage.KV.data();
            smem_V_param = storage.KVp;
        }

        __device__ void Load(int k, int pipe_iter)
        {
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;

            if (!std::is_same_v<T, Tkv>) {
                const int offset_s = lane_id / 8 + warp_id * WARP_S;
                PRAGMA_UNROLL
                for (int k = 0; k < V_K; ++k) {
                    const int si = k * 4 + offset_s;
                    Lds(param_V[k], &smem_V_param[pipe_iter * SmemLayoutKVp::kSize + SmemLayoutKVp::apply(si, 0)]);
                }
            }

            {
                const int offset_s = lane_id / 8 + warp_id * WARP_S;
                const int offset_c = lane_id % 8 * 16;
                if constexpr (bitsof<Tkv> != 4) {
                    PRAGMA_UNROLL
                    for (int n = 0; n < V_N; ++n) {
                        const int si = k * 4 + offset_s;
                        const int di = n * 8 + offset_c;
                        Lds(data_V[k][n], &smem_V[pipe_iter * SmemLayoutV::kSize + SmemLayoutV::apply(si, di)]);
                    }
                }
                else {
                    PRAGMA_UNROLL
                    for (int n = 0; n < V_N; n += 2) {
                        const int si = k * 4 + offset_s;
                        const int di = n * 8 + offset_c;
                        Lds((Array<Tkv, 16>&)data_V[k][n],
                            &smem_V[pipe_iter * SmemLayoutV::kSize + SmemLayoutV::apply(si, di)]);
                    }
                }
            }
        }

        __device__ void Transform(int k)
        {
            PRAGMA_UNROLL
            for (int n = 0; n < V_N; ++n) {
                ConvertKvCache<Tkv, Tpv> convert(param_V[k][0], param_V[k][1]);
                frag_V[k][n] = convert(data_V[k][n]);
                // frag_V[k][n] = cast<Tpv>(data_V[k][n]);
            }
        }
    };

    template<class Prefetch, class Preload>
    __device__ static void
    ComputePV(StatePV state_PV, FragO& frag_O, int offset, Prefetch&& prefetch, Preload&& preload)
    {
        PRAGMA_UNROLL
        for (int k = 0; k < V_K; ++k) {
            if (k < V_K - 1) {
                state_PV.Load(k + 1, offset);
            }
            else {
                ((Preload&&)preload)();
            }

            state_PV.Transform(k);

            PRAGMA_UNROLL
            for (int m = 0; m < V_M; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < V_N; ++n) {
                    PRAGMA_UNROLL
                    for (int d = 0; d < 8; ++d) {
                        frag_O[m][n][d] += static_cast<float>((Tpv)state_PV.frag_P[m][k][0] * state_PV.frag_V[k][n][d]);
                    }
                }
            }

            if (k < V_K - 1) {
                ((Prefetch&&)prefetch)(k);
            }
            if (k == V_K - 2) {
                ((Prefetch&&)prefetch)(V_K - 1);
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
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        const int warp_id_s = warp_id % kWarpCntS;
        const int warp_id_h = warp_id / kWarpCntS;

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
                storage.M[m][warp_id_h][warp_id_s] = frag_M[m][0];
            }
        }

        __syncthreads();

        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            PRAGMA_UNROLL
            for (int w = 0; w < kWarpCntS - 1; ++w) {
                frag_M[m][0] = fmaxf(frag_M[m][0], storage.M[m][warp_id_h][(warp_id_s + w + 1) % kWarpCntS]);
            }
            // if (threadIdx.x == 0) {
            //     printf("M %d %f\n", m * OP_H + blockIdx.x * CTA_H, frag_M[m][0]);
            // }
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
                        Store(storage.O[m][n][d / 4][warp_id_h][warp_id_s][lane_id].data(),
                              (Array<float, 4>&)frag_O[m][n][d]);
                    }
                }
            }
            frag_L[m][0] *= expdiff_M;
            frag_L[m][0] += __shfl_xor_sync(uint32_t(-1), frag_L[m][0], 8);
            frag_L[m][0] += __shfl_xor_sync(uint32_t(-1), frag_L[m][0], 16);
            if (lane_id == 0) {
                storage.L[m][warp_id_h][warp_id_s] = frag_L[m][0];
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
                        Lds(tmp_O, storage.O[m][n][d / 4][warp_id_h][s + lane_id / 8][lane_id % 8].data());
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
                frag_L[m][0] += storage.L[m][warp_id_h][(warp_id_s + w + 1) % kWarpCntS];
            }
            // if (threadIdx.x == 0) {
            //     printf("L %d %f\n", m * OP_H + blockIdx.x * CTA_H, frag_L[m][0]);
            // }
        }
    }

    template<bool is_norm, class Func>
    __device__ static void StoreO(FragO& frag_O, const FragL& frag_L, SharedStorage& storage, Func&& func)
    {
        FragL inv_L;

        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            inv_L[m][0] = fdividef(1.f, frag_L[m][0]);
        }

        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        const int warp_id_s = warp_id % kWarpCntS;
        const int warp_id_h = warp_id / kWarpCntS;

        if (warp_id_s != 0) {
            return;
        }

        PRAGMA_UNROLL
        for (int m = 0; m < V_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < V_N; ++n) {
                if constexpr (is_norm) {
                    PRAGMA_UNROLL
                    for (int d = 0; d < 8; ++d) {
                        frag_O[m][n][d] *= inv_L[m][0];
                    }
                }

                if (lane_id < 8) {
                    const int hi = m * OP_H + warp_id_h * WARP_H;
                    const int di = n * 8 + lane_id % 8 * 16;
                    // for (int i = 0; i < 8; ++i) {
                    //     printf("O %4d %4d %f\n", hi + blockIdx.x * CTA_H, di + i, frag_O[m][n][i]);
                    // }
                    ((Func&&)func)(hi, 0, di, frag_O[m][n]);
                }
            }
        }
    }
};

}  // namespace turbomind::attention
