// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "iterator.h"

namespace turbomind {

template<class T, class Layout, int N>
struct Sm75SmemIterK: BaseSmemIterator<T, Layout> {

    using Base = BaseSmemIterator<T, Layout>;

    using Base::Base;
    using Base::swizzle_uint_ptr;

    static_assert(N % 4 == 0);

    __device__ void Load(Array<T, 2> (&frag_K)[N], int k)
    {
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int n = 0; n < N; n += 4) {  // Load (s32,d8) tiles
            const int s = n * 8 + lane_id;
            const int c = k * 8;
            ldsm_x4((Array<uint32_t, 4>&)frag_K[n], swizzle_uint_ptr(s, c));
        }
    }
};

template<class T, class Layout, int N>
struct Sm75SmemIterV0: BaseSmemIterator<T, Layout> {

    using Base = BaseSmemIterator<T, Layout>;

    using Base::Base;
    using Base::swizzle_uint_ptr;

    static_assert(N % 4 == 0);

    __device__ void Load(Array<T, 2> (&frag_V)[N], int k)
    {
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int n = 0; n < N; n += 4) {  // Load (d32,s8) tiles
            const int si = k * 8 + lane_id % 8;
            const int di = n * 8 + lane_id / 8 * 8;
            ldsm_x4_trans((Array<uint32_t, 4>&)frag_V[n], swizzle_uint_ptr(si, di));
        }
    }
};

template<class T, class Layout, int N>
struct Sm75SmemIterV: BaseSmemIterator<T, Layout> {

    using Base = BaseSmemIterator<T, Layout>;
    using Base::Base;
    using Base::swizzle_uint_ptr;

    static_assert(N % 4 == 0);

    Array<int, N / 4> idxs_;

    __device__ Sm75SmemIterV(T* smem): Base{smem}
    {
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int n = 0; n < N; n += 4) {  // Load (d32,s8) tiles
            const int si = 0 * 8 + lane_id % 8;
            const int di = n * 8 + lane_id / 8 * 8;
            idxs_[n / 4] = swizzle_uint_ptr(si, di);
        }
    }

    __device__ void Load(Array<T, 2> (&frag_V)[N], int k)
    {
        PRAGMA_UNROLL
        for (int n = 0; n < N; n += 4) {
            const int idx = idxs_[n / 4] + sizeof(T) * k * 8 * Layout::kStride;
            ldsm_x4_trans((Array<uint32_t, 4>&)frag_V[n], idx);
        }
    }
};

}  // namespace turbomind