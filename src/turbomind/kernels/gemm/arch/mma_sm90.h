// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/gmma.h"
#include "src/turbomind/kernels/gemm/desc.h"

namespace turbomind::gemm {

struct MatrixDescriptor {
    struct BitField {
        unsigned start : 14;
        unsigned leading_byte_offset : 14;
        unsigned stride_byte_offset : 14;
        unsigned offset : 3;
        unsigned swizzle : 2;
    };
    union {
        BitField bitfield;
        uint64_t desc;
    };
};

static_assert(sizeof(MatrixDescriptor) == sizeof(uint64_t));

template<int N_>
struct SM90_GMMA_64xNx16_F32_F16_F16 {
    static constexpr int M = 64;
    static constexpr int N = N_;
    static constexpr int K = 16;

    static constexpr int kThreadCount = 128;

    static constexpr auto kOpClass = OpClass::kGMMA_s64n16;

    using FragA = Array<half, 8>;
    using FragC = Array<float, N / 2>;

    using OffsetC = Array<int2, N / 4>;  // 1 idx for 2 float
    using FragC_  = Array<float, 2>[N / 4];

    __device__ static void fma(FragC& d, const FragA& a, uint64_t desc_b)
    {
        return gmma_m64k16_rs(constant<N>{}, d, a, desc_b);
    }

    __device__ static constexpr OffsetC static_offset_C()
    {
        OffsetC offset_C{};
        for (int n = 0; n < N / 8; ++n) {
            for (int m = 0; m < 2; ++m) {
                offset_C[n * 2 + m] = int2{m * 8, n * 8};
            }
        }
        return offset_C;
    }

    __device__ static int2 thread_offset_C()
    {
        const int thread_idx = threadIdx.x;
        const int lane_id    = thread_idx % WARP_SIZE;
        return {(thread_idx & 96) / 2 + lane_id / 4, lane_id % 4 * 2};
    }

    __device__ static void ReshapeC(const FragC& c, FragC_& c_)
    {
        PRAGMA_UNROLL
        for (int n = 0; n < N / 8; ++n) {
            PRAGMA_UNROLL
            for (int m = 0; m < 2; ++m) {
                c_[n * 2 + m] = (Array<float, 2>&)c[(n * 2 + m) * 2];
            }
        }
    }

    __device__ static int get_group_id(int thread_idx)
    {
        return thread_idx / 128;
    }
};

}  // namespace turbomind::gemm