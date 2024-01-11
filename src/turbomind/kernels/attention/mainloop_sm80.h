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
    using GmemIterK = Sm80GmemIterator<T, ThreadMapKV, BlockSeqLen, typename Impl::SmemLayoutK>;
    template<class BlockSeqLen>
    using GmemIterV = Sm80GmemIterator<T, ThreadMapKV, BlockSeqLen, typename Impl::SmemLayoutV>;

    using FragQ = typename Impl::FragQ;
    using FragS = typename Impl::FragS;
    using FragO = typename Impl::FragO;
    using FragP = typename Impl::FragP;
    using FragM = typename Impl::FragM;
    using FragL = typename Impl::FragL;

    using SharedStorage = typename Impl::SharedStorage;

    static constexpr int CTA_S = Impl::CTA_S;

    static constexpr int Stages = 2;

    static constexpr int kTileSizeKV = CTA_S * Impl::SmemLayoutK::kStride;
    static constexpr int kSmemSizeKV = Stages * kTileSizeKV;

    __device__ int SmemKVStep(int& offset)
    {
        auto ret = offset;
        offset += kTileSizeKV;
        if (offset >= kSmemSizeKV) {
            offset -= kSmemSizeKV;
        }
        return ret;
    }

    template<class GmemIterK, class GmemIterV, class BlockIter, class StoreS>
    __device__ void operator()(FragQ&         frag_Q,
                               GmemIterK&     gmem_K,
                               GmemIterV&     gmem_V,
                               BlockIter&     block_iter,
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

        gmem_K.ClearSmem(0);
        gmem_V.ClearSmem(0);

        auto block_ptrs = gmem_K.block_ptrs_;

        int block_id;
        int local_id;
        block_iter.GetTile(tile_iter, block_id, local_id);

        const T* block = block_ptrs[block_id];

        gmem_K.Prefetch(block, local_id, std::true_type{}, max_step - tile_iter * CTA_S, 0);
        __pipeline_commit();

        auto loop = [&](auto is_residue, auto is_mask) {
            const int offset_K = tile_iter * CTA_S;

            __pipeline_wait_prior(0);
            Impl::Sync();

            __align__(16) FragS frag_S{};

            gmem_V.Prefetch(block, local_id, is_residue, is_residue ? max_step - offset_K : CTA_S, 0);
            __pipeline_commit();

            block_iter.NextTile(block_id, local_id);
            if (block_id >= 0) {
                block = block_ptrs[block_id];
            }

            Impl::ComputeQK(smem_Q, smem_K, frag_Q, frag_S, 0);

            __pipeline_wait_prior(0);
            Impl::Sync();

            if (tile_iter > 0) {
                gmem_K.Prefetch(block, local_id, std::false_type{}, CTA_S, 0);
                __pipeline_commit();
            }

            // store_S(frag_S, offset_K);

            if constexpr (is_mask) {
                ApplyCasualMask(frag_S, offset_Q, offset_K);
            }

            Impl::Softmax<is_residue>(frag_S, frag_M, frag_L, frag_O, qk_scale);

            __align__(16) FragP frag_P;

            Impl::ConvertStoP(frag_S, frag_P, storage.P);

            Impl::ComputePV(smem_P, smem_V, frag_P, frag_O, 0);
        };

        PRAGMA_UNROLL
        for (; tile_iter >= 0 && mask_iter != 0; --tile_iter, --mask_iter) {
            loop(std::true_type{}, std::true_type{});
        }

        for (; tile_iter >= 0; --tile_iter) {
            loop(std::false_type{}, std::false_type{});
        }

        __pipeline_commit();
        __pipeline_wait_prior(0);
    }

    __device__ void ApplyCasualMask(FragS& frag_S, int offset_Q, int offset_K)
    {
        Impl::ForeachS(frag_S, [&](int qi, int si, float& score) {
            if (offset_Q + qi < offset_K + si) {
                score -= std::numeric_limits<float>::infinity();
            }
        });
    }
};

}  // namespace turbomind::attention