// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/arch/mma_sm90.h"
#include "src/turbomind/kernels/gemm/arch/operand_sm90_s64n16.h"
#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/epilogue.h"
#include "src/turbomind/kernels/gemm/gemm_universal.h"
#include "src/turbomind/kernels/gemm/iterator_sm80.h"
#include "src/turbomind/kernels/gemm/mainloop_sm90.h"
#include "src/turbomind/kernels/gemm/thread_group_map.h"
#include "src/turbomind/kernels/gemm/tiled_mma.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm::sm90_s64n16 {

template<class Arch,
         class A,
         class TransformA,
         class U,
         class B,
         class TransformB,
         class V,
         Order order_c,
         class Tc,
         class CtaMap_ = CtaMap>
struct Sm90_s64n16 {

    // static_assert(A::SmemCopyAtom::K == B::SmemCopyAtom::K);

    static constexpr int SMEM_M = A::SmemCopyAtom::M / A::SmemCopyAtom::kFragNum;
    // static constexpr int SMEM_N = B::SmemCopyAtom::M / B::SmemCopyAtom::kFragNum;
    static constexpr int SMEM_K = A::SmemCopyAtom::K;

    template<int CTA_M,
             int CTA_N,
             int CTA_K,
             int TG_M,
             int TG_N,
             int TG_K,
             class PolicyA,
             class PolicyB,
             int  Stages,
             bool SplitK,
             int  GroupSizeU   = 1,
             int  GroupSizeV   = 1,
             int  TILE_C_M_    = -1,
             int  TILE_C_N_    = -1>

    struct Type {

        static constexpr int MMA_N = 8;

        // Raked partition dont support `Pack_M > 1`
        using Partition = Raked<TG_M, TG_N, kColMajor>;
        //                          128     16     16      16     16      16
        using MMA_Map   = MMA_Map<CTA_M, CTA_N, CTA_K, SMEM_M, MMA_N, SMEM_K, Partition, TG_K>;
        using MMA       = Tiled_MMA_v2<SM90_GMMA_64xNx16_F32_F16_F16<MMA_N>, MMA_Map>;

        using Mainloop = MainloopSm90<MMA,
                                      A,
                                      IteratorSm80<PolicyA>,
                                      TransformA,
                                      U,
                                      GroupSizeU,
                                      B,
                                      IteratorSm80<PolicyB>,
                                      TransformB,
                                      V,
                                      GroupSizeV,
                                      Stages>;

        static constexpr int TILE_C_M = TILE_C_M_ == -1 ? CTA_M : TILE_C_M_;
        static constexpr int TILE_C_N = TILE_C_N_ == -1 ? CTA_N : TILE_C_N_;

        using Epilogue = gemm::Epilogue_<Tc,
                                         CTA_M,
                                         CTA_N,
                                         TILE_C_M,
                                         TILE_C_N,
                                         MMA::kThreadCount,
                                         Rearrange<MMA>,
                                         Operand_C<float, order_c>,
                                         SplitK>;

        using Kernel = GemmUniversal<Arch, Mainloop, Epilogue, CtaMap_>;
    };
};

}  // namespace turbomind::gemm::sm90_s64n16