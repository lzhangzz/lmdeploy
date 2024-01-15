// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "impl.h"
#include "impl_m16n8.h"
#include "iterator.h"
#include "src/turbomind/kernels/attention/thread_map.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"
#include <cuda_pipeline_primitives.h>

namespace turbomind::attention {

template<class T, class Layout, int M, int WARPS>
struct Sm80SmemIterQ: BaseSmemIterator<T, Layout> {

    using Base = BaseSmemIterator<T, Layout>;
    using Base::Base;
    using Base::smem_;

    __device__ void Load(Array<T, 8> (&frag_Q)[M], int k)
    {
        PRAGMA_UNROLL
        for (int m = 0; m < M; ++m) {
            Lds(frag_Q[m], &smem_[((k * M + m) * WARPS * WARP_SIZE + threadIdx.x) * 8]);
        }
    }
};

template<class T, class Layout, int N>
struct Sm80SmemIterK: BaseSmemIterator<T, Layout> {

    using Base = BaseSmemIterator<T, Layout>;

    using Base::Base;
    using Base::swizzle_uint_ptr;

    static_assert(N % 2 == 0);

    __device__ void Load(Array<T, 4> (&frag_K)[N], int k, int offset)
    {
        const int lane_id       = threadIdx.x % WARP_SIZE;
        const int group_id      = lane_id / 16;
        const int group_lane_id = lane_id % 16;
        PRAGMA_UNROLL
        for (int n = 0; n < N; n += 2) {  // Load (s16,d16) tiles
            const int s = n * +8 + group_lane_id % 8 + group_id * 8;
            const int c = k * 16 + group_lane_id / 8 * 8;
            ldsm_x4((Array<uint32_t, 4>&)frag_K[n], offset + swizzle_uint_ptr(s, c));
        }
    }
};

template<class T, class Layout, int N>
struct Sm80SmemIterV: BaseSmemIterator<T, Layout> {

    using Base = BaseSmemIterator<T, Layout>;

    using Base::swizzle_uint_ptr;

    static_assert(N % 2 == 0);

    Array<int, N / 2> offsets_;

    static constexpr bool kFlag = false;

    __device__ Sm80SmemIterV(const T* smem): Base{smem}
    {
        if constexpr (kFlag) {
            const int lane_id = threadIdx.x % WARP_SIZE;
            PRAGMA_UNROLL
            for (int n = 0; n < N; n += 2) {  //  (d16,s16) tiles
                const int si    = 0 * 16 + lane_id % 16;
                const int di    = n * +8 + lane_id / 16 * 8;
                offsets_[n / 2] = swizzle_uint_ptr(si, di);
            }
        }
    }

    __device__ void Load(Array<T, 4> (&frag_V)[N], int k, int base)
    {
        if constexpr (kFlag) {
            PRAGMA_UNROLL
            for (int n = 0; n < N; n += 2) {  //  (d16,s16) tiles
                const int offset = offsets_[n / 2] + sizeof(T) * k * 16 * Layout::kStride;
                ldsm_x4_trans((Array<uint32_t, 4>&)frag_V[n], base + offset);
            }
        }
        else {
            const int lane_id = threadIdx.x % WARP_SIZE;
            PRAGMA_UNROLL
            for (int n = 0; n < N; n += 2) {  // Load (d16,s16) tiles
                const int si = k * 16 + lane_id % 16;
                const int di = n * +8 + lane_id / 16 * 8;
                ldsm_x4_trans((Array<uint32_t, 4>&)frag_V[n], base + swizzle_uint_ptr(si, di));
            }
        }
    }
};

template<class Layout, int N>
struct Sm80SmemIterK<uint8_t, Layout, N>: BaseSmemIterator<uint8_t, Layout> {

    using Base = BaseSmemIterator<uint8_t, Layout>;

    using Base::Base;
    using Base::swizzle_uint_ptr;

    static_assert(N % 4 == 0);

    __device__ void Load(Array<uint8_t, 4> (&frag_K)[N], int k)
    {
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int n = 0; n < N; n += 4) {  // K16,N32 (1x4) per LDSM.x4
            auto&     r = (Array<uint32_t, 4>&)frag_K[n];
            const int s = n * 8 + lane_id;
            const int c = k * 16 + 0;
            ldmatrix_m8n8_x4_b16(r[0], r[1], r[2], r[3], swizzle_uint_ptr(s, c));
        }
    }
};

template<class Layout, int N>
struct Sm80SmemIterV<uint8_t, Layout, N>: BaseSmemIterator<uint8_t, Layout> {

    using Base = BaseSmemIterator<uint8_t, Layout>;

    using Base::Base;
    using Base::swizzle_uint_ptr;

    static_assert(N % 4 == 0);

    __device__ void Load(Array<uint8_t, 4> (&frag_V)[N], int k, int offset)
    {
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int n = 0; n < N; n += 4) {  // K16,N32 (2x2) per LDSM.x4
            auto&     r = (Array<uint32_t, 4>&)frag_V[n];
            const int s = lane_id % 16 + k * 16;      // v
            const int c = lane_id / 16 * 16 + n * 8;  // d
            ldsm_x4_trans(r[0], r[1], r[2], r[3], swizzle_uint_ptr(s, c));
        }
    }
};

template<class T_, int CTA_H_, int CTA_Q_, int CTA_S_, int WARP_H, int WARP_Q, int WARP_S, int HeadDim, int Stages>
struct Impl<Sm80_16816, T_, T_, CTA_H_, CTA_Q_, CTA_S_, WARP_H, WARP_Q, WARP_S, HeadDim, Stages>:
    Impl_m16k8<T_, CTA_Q_, WARP_Q, WARP_S, HeadDim> {

    using Base = Impl_m16k8<T_, CTA_Q_, WARP_Q, WARP_S, HeadDim>;

    using Base::OP_M;
    using Base::OP_N;
    using Base::K_M;
    using Base::K_N;
    using Base::V_M;
    using Base::V_N;

    using typename Base::FragS;
    using typename Base::FragO;
    using typename Base::FragM;
    using typename Base::FragL;

    using Base::ForeachS;
    using Base::Softmax;
    using Base::ConvertStoP;
    using Base::StoreO;

    using T   = T_;
    using Tkv = T_;

    static constexpr int kHeadDim = HeadDim;

    static constexpr int CTA_H = CTA_H_;
    static constexpr int CTA_Q = CTA_Q_;
    static constexpr int CTA_S = CTA_S_;

    static constexpr int kWarpCntH  = CTA_H / WARP_H;
    static constexpr int kWarpCntQ  = CTA_Q / WARP_Q;
    static constexpr int kWarpCntS  = CTA_S / WARP_S;
    static constexpr int kWarpCount = kWarpCntH * kWarpCntQ * kWarpCntS;

    static constexpr int OP_K = 16;

    static constexpr int K_K = HeadDim / OP_K;  // 128 / 16 = 8
    static constexpr int V_K = WARP_S / OP_K;   //  64 / 16 = 4  -> S4

    using FragQ = Array<T, 8>[K_K][K_M];  // ((q8, d4), (Dk, Qm), (d2, q2, d2))
                                          //    1   2    16  16     8   8   1
    using FragK = Array<T, 4>[K_K][K_N];  // ((s8, d4), (Dk, Sn), (d2, d2))
                                          //    1   2    16   8     8   1
    using FragP = Array<T, 8>[V_M][V_K];  // ((q8, s4), (Qm, Sk), (s2, q2, s2))
                                          //    1   2    16  16     8   8   1
    using FragV = Array<T, 4>[V_K][V_N];  // ((d8, s4), (Sk, Dn), (s2, s2))
                                          //    1   2    16   8     8   1

    static_assert(sizeof(FragS) / 2 == sizeof(FragP));

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
    using SmemLayoutV = SmemLayout<HeadDim, Swizzle>;
    // using SmemLayoutK = SmemLayout<HeadDim + 8, Identity>;
    // using SmemLayoutV = SmemLayout<HeadDim + 8, Identity>;
    using SmemLayoutP = Identity;

    union SharedStorage {
        __align__(16) T Q[CTA_H * CTA_Q * SmemLayoutQ::kStride];

        __align__(16) T KV[Stages * CTA_S * (SmemLayoutK::kStride + SmemLayoutV::kStride) / 2];
        struct {
            __align__(16) T K[CTA_S * SmemLayoutK::kStride];
            __align__(16) T V[CTA_S * SmemLayoutV::kStride];
        };

        struct {
            T P[1];
        };
    };

    using SmemIterQ = NullSmemIter<T>;
    using SmemIterP = NullSmemIter<T>;

    using SmemIterK = Sm80SmemIterK<T, SmemLayoutK, K_N>;
    using SmemIterV = Sm80SmemIterV<T, SmemLayoutV, V_N>;

    static constexpr bool kUseSmemQ = false;

    using ThreadMapQ  = RakedThreadMap<HeadDim, CTA_H * CTA_Q, 8, kWarpCount>;
    using ThreadMapKV = RakedThreadMap<HeadDim, CTA_S, 8, kWarpCount>;

    __device__ static void Sync()
    {
        __syncthreads();
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
                    const int hi     = warp_id / kWarpCntQ;
                    const int qi     = m * 16 + lane_id % 16 + (warp_id % kWarpCntQ) * WARP_Q;
                    const int di     = k * 16 + lane_id / 16 * 8;
                    const int offset = sizeof(T) * SmemLayoutQ::swizzle(hi * CTA_Q + qi, di);
                    ldsm_x4((Array<uint32_t, 4>&)frag_Q[k][m], smem_int_ptr + offset);
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
                    const int hi     = warp_id / kWarpCntQ;
                    const int qi     = m * 16 + group_lane_id % 8 + group_id * 8 + (warp_id % kWarpCntQ) * WARP_Q;
                    const int di     = k * 16 + group_lane_id / 8 * 8;
                    const int offset = sizeof(T) * SmemLayoutQ::swizzle(hi * CTA_Q + qi, di);
                    ldsm_x4((Array<uint32_t, 4>&)frag_Q[k][m], smem_int_ptr + offset);
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

    template<class SmemQ, class SmemK, class Func>
    __device__ static void
    ComputeQK(SmemQ& smem_Q, SmemK& smem_K, FragQ& frag_Q, FragK& frag_K, FragS& frag_S, int offset, Func&& func)
    {
        // FragK frag_K;

        // smem_K.Load(frag_K[0], 0, offset);
        if constexpr (kUseSmemQ) {
            smem_Q.Load(frag_Q[0], 0);
        }

        PRAGMA_UNROLL
        for (int k = 0; k < K_K; ++k) {
            if (k < K_K - 1) {
                smem_K.Load(frag_K[k + 1], k + 1, offset);
                if constexpr (kUseSmemQ) {
                    smem_Q.Load(frag_Q[k + 1], k + 1);
                }
            }
            else {
                ((Func&&)func)();
            }
            PRAGMA_UNROLL
            for (int m = 0; m < K_M; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < K_N; ++n) {
                    const int nn = n ^ 1;
                    mma_m16n8k16_row_col(frag_S[m][nn], frag_Q[k][m], frag_K[k][nn], frag_S[m][nn]);
                }
            }
        }
    }

    template<class SmemP, class SmemV, class Func>
    __device__ static void
    ComputePV(SmemP&, SmemV& smem_V, const FragP& frag_P, FragV& frag_V, FragO& frag_O, int offset, Func&& func)
    {
        // FragV frag_V;

        // smem_V.Load(frag_V[0], 0, offset);

        PRAGMA_UNROLL
        for (int k = 0; k < V_K; ++k) {
            if (k < V_K - 1) {
                smem_V.Load(frag_V[k + 1], k + 1, offset);
            }
            else {
                ((Func&&)func)();
            }
            PRAGMA_UNROLL
            for (int m = 0; m < V_M; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < V_N; ++n) {
                    const int nn = n ^ 0;
                    mma_m16n8k16_row_col(frag_O[m][nn], frag_P[m][k], frag_V[k][nn], frag_O[m][nn]);
                }
            }
        }
    }
};

}  // namespace turbomind::attention