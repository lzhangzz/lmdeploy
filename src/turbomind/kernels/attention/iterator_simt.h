// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "array_ops.h"
#include "iterator.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"

namespace turbomind {

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

}  // namespace turbomind