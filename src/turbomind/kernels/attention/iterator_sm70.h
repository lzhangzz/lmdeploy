// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "array_ops.h"
#include "iterator.h"

namespace turbomind {

template<class T, class Map, class BlockSeqLen, class SmemLayout>
struct Sm70GmemIterator: BaseGmemIterator<T, Map, BlockSeqLen, SmemLayout> {
    using Base = BaseGmemIterator<T, Map, BlockSeqLen, SmemLayout>;

    using typename Base::AccessType;
    using typename Base::Fragment;

    using Base::block_;
    using Base::local_offset_;
    using Base::init_offset_;
    using Base::dst_offset_;
    using Base::smem_;

    using Base::Base;

    template<bool is_residue>
    __device__ void Load(Fragment& rmem, std::bool_constant<is_residue>, int max_s)
    {
        auto      src      = block_ + local_offset_ + init_offset_;
        const int offset_s = Map::get_offset(threadIdx.x / WARP_SIZE, threadIdx.x % WARP_SIZE).y;
        PRAGMA_UNROLL
        for (int s = 0; s < Map::kIterS; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < Map::kIterC; ++c) {
                if constexpr (!is_residue) {
                    Ldg(rmem[s][c], &src[s * Map::kDeltaS * Map::kDimC + c * Map::kDeltaC]);
                }
                else if (offset_s + s * Map::kDeltaS < max_s) {
                    Ldg(rmem[s][c], &src[s * Map::kDeltaS * Map::kDimC + c * Map::kDeltaC]);
                }
                else {
                    clear(rmem[s][c]);
                }
            }
        }
    }

    __device__ void Save(const Fragment& rmem)
    {
        // typename SmemLayout::Swizzle swizzle{};

        // Array<int, Map::kIterC> idxs;
        // PRAGMA_UNROLL
        // for (int c = 0; c < Map::kIterC; ++c) {
        //     const int idx0 = swizzle(dst_offset_ + c * Map::kDeltaC);
        //     idxs[c]        = idx0;
        // }
        // const int offset_s = Map::get_offset(threadIdx.x / WARP_SIZE, threadIdx.x % WARP_SIZE).y;
        // PRAGMA_UNROLL
        // for (int s = 0; s < Map::kIterS; ++s) {
        //     PRAGMA_UNROLL
        //     for (int c = 0; c < Map::kIterC; ++c) {
        //         Store(&smem_[idxs[c]], rmem[s][c]);
        //     }
        //     PRAGMA_UNROLL
        //     for (int c = 0; c < Map::kIterC; ++c) {
        //         const int s0 = offset_s + s * Map::kDeltaS;
        //         const int s1 = s0 + Map::kDeltaS;
        //         idxs[c]      = swizzle.AdvanceS<Map::kDeltaS>(idxs[c], s0, s1) + Map::kDeltaS * SmemLayout::kStride;
        //     }
        // }

        PRAGMA_UNROLL
        for (int s = 0; s < Map::kIterS; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < Map::kIterC; ++c) {
                Store(&smem_[SmemLayout::swizzle(dst_offset_ + s * Map::kDeltaS * SmemLayout::kStride
                                                 + c * Map::kDeltaC)],
                      rmem[s][c]);
            }
        }
    }
};

template<class T, class Layout, int M>
struct Sm70SmemIterQ: BaseSmemIterator<T, Layout> {

    using Base = BaseSmemIterator<T, Layout>;

    using Base::Base;
    using Base::ptr;

    __device__ void Load(Array<half, 4> (&frag_Q)[M], int k)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int m = 0; m < M; ++m) {
            const int qi = m * 16 + (lane_id & 8) + lane_id % 4 + lane_id / 16 * 4 + warp_id * 16;
            const int di = k * 4;
            Lds(frag_Q[m], ptr(qi, di));
        }
    }
};

template<class T, class Layout, int N>
struct Sm70SmemIterK: BaseSmemIterator<T, Layout> {
    using Base = BaseSmemIterator<T, Layout>;

    using Base::Base;
    using Base::ptr;

    __device__ void Load(Array<half, 4> (&frag_K)[N], int k)
    {
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int n = 0; n < N; ++n) {
            const int s = n * 16 + lane_id / 16 * 4 + (lane_id & 4) * 2 + lane_id % 4;
            const int c = k * 4;
            Lds(frag_K[n], ptr(s, c));
        }
    }
};

template<class T, class Layout, int N>
struct Sm70SmemIterV: BaseSmemIterator<T, Layout> {
    using Base = BaseSmemIterator<T, Layout>;
    using Base::ptr;

    static_assert(N % 2 == 0);

    Array<int, N / 2> idxs_;

    __device__ explicit Sm70SmemIterV(const T* smem): Base{smem}
    {
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int n = 0; n < 8; n += 2) {
            const int s  = 0 * 4 + lane_id % 4;
            const int c  = n * 16 + lane_id / 16 * 4 + (lane_id & 4) * 2;
            idxs_[n / 2] = Layout::swizzle(s, c);
        }
    }

    __device__ void Load(Array<half, 4> (&frag_V)[N], int k)
    {
        PRAGMA_UNROLL
        for (int n = 0; n < N; n += 2) {
            const int idx = idxs_[n / 2] + k * 4 * Layout::kStride;
            Lds((Array<half, 8>&)frag_V[n], ptr(idx));
        }
    }
};

template<class T, class Layout, int M>
struct Sm70SmemIterP: BaseSmemIterator<T, Layout> {

    using Base = BaseSmemIterator<T, Layout>;

    using Base::Base;
    using Base::ptr;

    __device__ void Load(Array<half, 4> (&frag_P)[M], int k)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int m = 0; m < M; ++m) {
            const int qi = m * 16 + lane_id / 16 * 4 + (lane_id & 8) + lane_id % 4 + warp_id * 16;
            const int si = k * 4;
            Lds(frag_P[m], ptr(qi, si));
        }
    }
};

}  // namespace turbomind