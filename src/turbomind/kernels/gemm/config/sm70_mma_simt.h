// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/gemm_universal.h"
#include "src/turbomind/kernels/gemm/impl.h"
#include "src/turbomind/kernels/gemm/impl_simt.h"
#include "src/turbomind/kernels/gemm/iterator.h"
#include "src/turbomind/kernels/gemm/kernel_impl.h"
#include "src/turbomind/kernels/gemm/mainloop_sm80_v2.h"
#include "src/turbomind/kernels/gemm/operand.h"
#include "src/turbomind/kernels/gemm/smem_copy.h"
#include "src/turbomind/kernels/gemm/smem_copy_simt.h"
#include "src/turbomind/kernels/gemm/thread_group_map.h"
#include "src/turbomind/kernels/gemm/tiled_mma.h"
#include "src/turbomind/kernels/gemm/transform.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

namespace sm70_mma_simt {

struct GetSmemLayout {
    template<int M, int K>
    static constexpr auto apply(pair<M, K>)
    {
        return SmemLayoutV2<M, K>{};
    }
};

template<class T, int K>
struct OperandA {
    using Dtype = T;

    static constexpr Pack  kPack  = 0;
    static constexpr Order kOrder = kRowMajor;

    using SmemCopyAtom = SmemCopy_MMA_SIMT_A<T, K>;

    using GetSmemLayout = GetSmemLayout;
    using GetGmemIter   = GetGmemIter;
};

template<class T, int K>
struct OperandB {
    using Dtype = T;

    static constexpr Pack  kPack  = 0;
    static constexpr Order kOrder = kRowMajor;

    using SmemCopyAtom = SmemCopy_MMA_SIMT_B<T, K>;

    using GetSmemLayout = GetSmemLayout;
    using GetGmemIter   = GetGmemIter;
};

struct GetSmemLayout_Pack {
    template<int M, int K>
    static constexpr auto apply(pair<M, K>)
    {
        return SmemLayoutV2<M, K>{};
    }
};

template<class T, int K>
struct Operand_B_Pack {
    using Dtype = T;

    static constexpr int Pack_M = 1;

    static constexpr Pack  kPack  = HMMA_SIMT | OPERAND_B | Pack_M;
    static constexpr Order kOrder = kRowMajor;

    using SmemCopyAtom  = SmemCopyAtom_Pack_v3<T, typename OperandB<T, K>::SmemCopyAtom, kRowMajor, Pack_M>;
    using GetSmemLayout = GetSmemLayout_Pack;
    using GetGmemIter   = GetGmemIter;
};

template<class A, class TransformA, class U, class B, class TransformB, class V, class Tc>
struct SM70_MMA_F32 {
    template<int  CTA_M,
             int  CTA_N,
             int  CTA_K,
             int  WARP_CNT_M,
             int  WARP_CNT_N,
             int  WARP_CNT_K,
             int  Stages,
             bool SplitK,
             bool AlignedM,
             bool AlignedN,
             int  GroupSizeU = 1,
             int  GroupSizeV = 1>
    struct Type {

        // (TM, TN, TK) = R(MMA_Atom, SmemCopy_Atom)
        using MMA_Atom = SM70_MMA_SIMT<half>;

        static constexpr int TM = MMA_Atom::M;
        static constexpr int TN = MMA_Atom::N;
        static constexpr int TK = MMA_Atom::K;

        using MMA_Map = RakedThreadGroupMap<CTA_M, CTA_N, CTA_K, TM, TN, TK, WARP_CNT_M, WARP_CNT_N, WARP_CNT_K>;

        using MMA = Tiled_MMA_v2<MMA_Atom, MMA_Map>;

        using Mainloop = MainloopSm80_v2<CTA_M,
                                         CTA_N,
                                         CTA_K,
                                         MMA,
                                         A,
                                         TransformA,
                                         U,
                                         GroupSizeU,
                                         B,
                                         TransformB,
                                         V,
                                         GroupSizeV,
                                         Stages>;

        using Kernel = GemmUniversal<void, Mainloop, Tc, CtaMap, AlignedM, AlignedN, SplitK>;
    };
};

}  // namespace sm70_mma_simt

template<class T>
struct GetOperand<HMMA_SIMT, OPERAND_A, T, kRowMajor, false>: std::true_type {
    using Operand = sm70_mma_simt::OperandA<T, 4>;
};

template<class T>
struct GetOperand<HMMA_SIMT, OPERAND_B, T, kRowMajor, false>: std::true_type {
    using Operand = sm70_mma_simt::OperandB<T, 4>;
};

template<class T>
struct GetOperand<HMMA_SIMT, OPERAND_B, T, kRowMajor, true>: std::true_type {
    using Operand = sm70_mma_simt::Operand_B_Pack<T, 4>;
};

}  // namespace turbomind::gemm