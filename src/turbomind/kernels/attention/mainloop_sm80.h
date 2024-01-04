// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "iterator_sm80.h"
#include "mainloop.h"
#include <cuda_pipeline_primitives.h>

namespace turbomind::attention {

template<class Impl_>
struct Mainloop<Sm80_CpAsync, Impl_> {

    using Impl = Impl_;

    using T   = typename Impl::T;
    using Tkv = typename Impl::Tkv;

    using SmemIterQ = typename Impl::SmemIterQ;
    using SmemIterK = typename Impl::SmemIterK;
    using SmemIterP = typename Impl::SmemIterP;
    using SmemIterV = typename Impl::SmemIterV;

    using ThreadMapKV = typename Impl::ThreadMapKV;
    template<class BlockSeqLen>
    using GmemIterK = Sm80GmemIterator<T, ThreadMapKV, BlockSeqLen, typename Impl::SmemLayoutK, 2>;
    template<class BlockSeqLen>
    using GmemIterV = Sm80GmemIterator<T, ThreadMapKV, BlockSeqLen, typename Impl::SmemLayoutV, 2>;

    using FragQ = typename Impl::FragQ;
    using FragS = typename Impl::FragS;
    using FragO = typename Impl::FragO;
    using FragP = typename Impl::FragP;
    using FragM = typename Impl::FragM;
    using FragL = typename Impl::FragL;

    using SharedStorage = typename Impl::SharedStorage;

    static constexpr int CTA_S = Impl::CTA_S;

    template<class GmemIterK, class GmemIterV, class StoreS>
    __device__ void operator()(FragQ&         frag_Q,
                               GmemIterK&     gmem_K,
                               GmemIterV&     gmem_V,
                               FragO&         frag_O,
                               FragM&         frag_M,
                               FragL&         frag_L,
                               int            offset_Q,
                               int            max_step,
                               int            tile_iter,
                               int            mask_iter,
                               float          qk_scale,
                               SharedStorage& storage,
                               const StoreS&  store_S)
    {
        SmemIterQ smem_Q{storage.Q};
        SmemIterK smem_K{storage.K};
        SmemIterP smem_P{storage.P};
        SmemIterV smem_V{storage.V};

        gmem_K.AdjustBlockTileIdx(tile_iter);
        gmem_K.Prefetch(std::true_type{}, max_step - tile_iter * CTA_S);
        CpAsyncCommit();

        auto loop = [&](auto is_residue, auto is_mask) {
            const int offset_K = tile_iter * CTA_S;

            CpAsyncWait();
            __syncthreads();

            gmem_V.AdjustBlockTileIdx(tile_iter);
            gmem_V.Prefetch(is_residue, is_residue ? max_step - offset_K : CTA_S);
            CpAsyncCommit();

            __align__(16) FragS frag_S{};

            Impl::ComputeQK(smem_Q, smem_K, frag_Q, frag_S);

            CpAsyncWait();
            __syncthreads();

            if (tile_iter > 0) {
                gmem_K.AdjustBlockTileIdx(tile_iter - 1);
                gmem_K.Prefetch(std::false_type{}, CTA_S);
                CpAsyncCommit();
            }

            // store_S(frag_S, offset_K);

            if constexpr (is_mask) {
                ApplyCasualMask(frag_S, offset_Q, offset_K);
            }

            Impl::Softmax<is_residue>(frag_S, frag_M, frag_L, frag_O, qk_scale);

            __align__(16) FragP frag_P;

            Impl::ConvertStoP(frag_S, frag_P, storage.P);

            Impl::ComputePV(smem_P, smem_V, frag_P, frag_O);
        };

        PRAGMA_UNROLL
        for (; tile_iter >= 0 && mask_iter != 0; --tile_iter, --mask_iter) {
            loop(std::true_type{}, std::true_type{});
        }

        for (; tile_iter >= 0; --tile_iter) {
            loop(std::false_type{}, std::false_type{});
        }
    }

    __device__ void ApplyCasualMask(FragS& frag_S, int offset_Q, int offset_K)
    {
        Impl::ForeachS(frag_S, [&](int qi, int si, float& score) {
            if (offset_Q + qi < offset_K + si) {
                score -= std::numeric_limits<float>::infinity();
            }
        });
    }

    __device__ void CpAsyncWait()
    {
        if constexpr (TURBOMIND_ARCH_SM80) {
            __pipeline_wait_prior(0);
        }
    }

    __device__ void CpAsyncCommit()
    {
        if constexpr (TURBOMIND_ARCH_SM80) {
            __pipeline_commit();
        }
    }

    __device__ void CpAsyncFlush()
    {
        if constexpr (TURBOMIND_ARCH_SM80) {
            __pipeline_commit();
            __pipeline_wait_prior(0);
        }
    }
};

}  // namespace turbomind::attention