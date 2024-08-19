// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/kernels/gemm/arch/operand_sm80_s16816.h"
#include "src/turbomind/kernels/gemm/arch/smem_copy_sm80.h"
#include "src/turbomind/kernels/gemm/iterator.h"
#include "src/turbomind/kernels/gemm/operand.h"
#include "src/turbomind/kernels/gemm/smem_copy.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

namespace sm90_s64n16 {

using sm80_s16816::Operand_C;

// (m, k)
template<class T, Order order>
struct Operand_A {
    using Dtype = T;

    static constexpr Pack  kPack  = 0;
    static constexpr Order kOrder = order;

    using SmemCopyAtom = LDSM_SM75_8x8<T, 64, 16, ~order, order, 4>;

    using GetSmemLayout = sm80_s16816::GetSmemLayoutV2<kOrder>;
    using GetGmemIter   = GetGmemIter;
};


template<class T, Order order, int Pack_M>
struct Operand_A_Pack {
    using Dtype = T;

    static constexpr Pack  kPack  = HGMMA_64n16 | OPERAND_A | Pack_M;
    static constexpr Order kOrder = order;

    using _SCp         = typename Operand_A<T, order>::SmemCopyAtom;
    using SmemCopyAtom = SmemCopyAtom_Pack_v3<T, _SCp, order, Pack_M>;

    using GetSmemLayout = sm80_s16816::GetSmemLayout_Pack<kOrder>;
    using GetGmemIter   = GetGmemIter;
};


template<class T>
struct Operand_UV {
    using Dtype = T;

    static constexpr Pack  kPack  = 0;
    static constexpr Order kOrder = kColMajor;

    using SmemCopyAtom = SmemCopy_MMA_16816_U<T, 4>;

    struct GetSmemLayout {
        template<int M, int K>
        static constexpr auto apply(pair<M, K>)
        {
            return SmemLayoutV2<K, M>{};
        }
    };
    using GetGmemIter = GetGmemIter;
};

template<class T, bool is_V>
struct Operand_UV_Pack {
    using Dtype = T;

    static constexpr int Pack_M = 1;

    static constexpr Pack  kPack  = HGMMA_64n16 | (is_V ? OPERAND_V : OPERAND_U) | Pack_M;
    static constexpr Order kOrder = Order::kColMajor;

    using _SCp         = typename Operand_UV<T>::SmemCopyAtom;
    using SmemCopyAtom = SmemCopyAtom_Pack_v3<T, _SCp, kOrder, Pack_M>;

    using GetSmemLayout = sm80_s16816::GetSmemLayout_Pack<kOrder>;
    using GetGmemIter   = GetGmemIter;
};


template<class T, Order order, int N>
struct Operand_B {
    using Dtype = T;

    static constexpr Pack  kPack  = 0;
    static constexpr Order kOrder = order;

    using SmemCopyAtom = void;

    struct GetSmemLayout {
        template<int S, int C>
        static constexpr auto apply(pair<S, C>)
        {
            // No swizzle
            // LBO: 16
            // SBO: 128
            return SmemLayoutV2<S, C, 8, 8>{};
        }
    };
    using GetGmemIter = GetGmemIter;
};

}  // namespace sm90_s64n16

template<class T, Order order>
struct GetOperand<HGMMA_64n16, OPERAND_A, T, order, false>: std::true_type {
    using Operand = sm90_s64n16::Operand_A<T, order>;
};

template<class T>
struct GetOperand<HGMMA_64n16, OPERAND_U, T, kColMajor, false>: std::true_type {
    using Operand = sm90_s64n16::Operand_UV<T>;
};


}  // namespace turbomind::gemm