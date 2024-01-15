// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "iterator_sm80.h"
#include "mainloop.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"
#include <cuda_pipeline_primitives.h>

namespace turbomind::attention {

template<int Stages>
struct Sm80_CpAsync {};

template<int Stages, class Impl_>
struct Mainloop<Sm80_CpAsync<Stages>, Impl_> {

    using Impl = Impl_;

    using T   = typename Impl::T;
    using Tkv = typename Impl::Tkv;

    using SmemIterQ = typename Impl::SmemIterQ;
    using SmemIterK = typename Impl::SmemIterK;
    using SmemIterP = typename Impl::SmemIterP;
    using SmemIterV = typename Impl::SmemIterV;

    using ThreadMapKV = typename Impl::ThreadMapKV;
    using GmemIterK   = Sm80GmemIterator<T, ThreadMapKV, typename Impl::SmemLayoutK>;
    using GmemIterV   = Sm80GmemIterator<T, ThreadMapKV, typename Impl::SmemLayoutV>;

    using FragQ = typename Impl::FragQ;
    using FragK = typename Impl::FragK;
    using FragV = typename Impl::FragV;
    using FragS = typename Impl::FragS;
    using FragO = typename Impl::FragO;
    using FragP = typename Impl::FragP;
    using FragM = typename Impl::FragM;
    using FragL = typename Impl::FragL;

    using SharedStorage = typename Impl::SharedStorage;

    static constexpr int CTA_S = Impl::CTA_S;

    static constexpr int kTileSizeKV = CTA_S * Impl::SmemLayoutK::kStride;
    static constexpr int kSmemSizeKV = Stages * kTileSizeKV;

    __device__ int SmemKVStep(int& offset)
    {
        auto ret = offset;
        offset += sizeof(Tkv) * kTileSizeKV;
        if (offset >= sizeof(Tkv) * kSmemSizeKV) {
            offset -= sizeof(Tkv) * kSmemSizeKV;
        }
        return ret;
    }

    template<class... Args>
    __device__ void operator()(Args&&... args)
    {
        Run(Sm80_CpAsync<Stages>{}, ((Args&&)args)...);
    }

    template<class GmemIterK, class GmemIterV, class BlockIter, class StoreS, int Stages_>
    __device__ void Run(Sm80_CpAsync<Stages_>,
                        FragQ&         frag_Q,
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
        gmem_K.SetSmem(storage.KV);
        gmem_V.SetSmem(storage.KV);

        SmemIterQ smem_Q{storage.Q};
        SmemIterP smem_P{storage.P};
        SmemIterK smem_K{storage.KV};
        SmemIterV smem_V{storage.KV};

        PRAGMA_UNROLL
        for (int i = 0; i < Stages; ++i) {
            gmem_K.ClearSmem(i * kTileSizeKV);
        }

        int kv_offset_r = 0;
        int kv_offset_w = 0;

        block_iter.SetTile(tile_iter);

        gmem_K.Prefetch<true>(
            block_iter, 0, ThreadMapKV::kIterS, max_step - tile_iter * CTA_S, SmemKVStep(kv_offset_w));
        __pipeline_commit();

        gmem_V.Prefetch<true>(
            block_iter, 0, ThreadMapKV::kIterS, max_step - tile_iter * CTA_S, SmemKVStep(kv_offset_w));
        __pipeline_commit();

        block_iter.Advance();

        PRAGMA_UNROLL
        for (int stages = 2; stages < Stages - 1; stages += 2) {
            gmem_K.Prefetch<false>(block_iter, 0, ThreadMapKV::kIterS, CTA_S, SmemKVStep(kv_offset_w));
            __pipeline_commit();

            gmem_V.Prefetch<false>(block_iter, 0, ThreadMapKV::kIterS, CTA_S, SmemKVStep(kv_offset_w));
            __pipeline_commit();

            block_iter.Advance();
        }

        __pipeline_wait_prior(Stages - 2);
        Impl::Sync();

        FragK frag_K;
        FragV frag_V;
        auto  rk = SmemKVStep(kv_offset_r);
        smem_K.Load(frag_K[0], 0, rk);

        auto loop = [&](auto is_residue, auto is_mask) {
            const int offset_K = tile_iter * CTA_S;

            __align__(16) FragS frag_S{};

            auto rv = SmemKVStep(kv_offset_r);
            auto wk = SmemKVStep(kv_offset_w);

            FragM prev_M;
            PRAGMA_UNROLL
            for (int m = 0; m < Impl::K_M; ++m) {
                prev_M[m] = frag_M[m];
            }

            Impl::ComputeQK(
                smem_Q,
                smem_K,
                frag_Q,
                frag_K,
                frag_S,
                frag_M,
                rk,
                [&](int k) {
                    // constexpr int kBatch = (ThreadMapKV::kIterS + Impl::K_K - 1) / Impl::K_K;
                    constexpr int kBatch = ThreadMapKV::kIterS / 2;
                    if (k * kBatch < ThreadMapKV::kIterS) {
                        gmem_K.Prefetch<false>(block_iter, k * kBatch, (k + 1) * kBatch, CTA_S, wk);
                    }
                    if ((k + 1) * kBatch == ThreadMapKV::kIterS) {
                        __pipeline_commit();
                    }
                },
                [&] { smem_V.Load(frag_V[0], 0, rv); },
                [&] {
                    // __pipeline_commit();
                    __pipeline_wait_prior(Stages - 2);
                    Impl::Sync();
                });

            if constexpr (is_mask) {
                ApplyCasualMask(frag_S, offset_Q, offset_K);
            }

            Impl::Softmax<is_residue>(frag_S, frag_M, prev_M, frag_L, frag_O, qk_scale);

            __align__(16) FragP frag_P;

            Impl::ConvertStoP(frag_S, frag_P, storage.P);

            rk      = SmemKVStep(kv_offset_r);
            auto wv = SmemKVStep(kv_offset_w);
            Impl::ComputePV(
                smem_P,
                smem_V,
                frag_P,
                frag_V,
                frag_O,
                rv,
                [&](int k) {
                    // constexpr int kBatch = (ThreadMapKV::kIterS + Impl::V_K - 1) / Impl::V_K;
                    constexpr int kBatch = ThreadMapKV::kIterS / 2;
                    if (k * kBatch < ThreadMapKV::kIterS) {
                        gmem_V.Prefetch<false>(block_iter, k * kBatch, (k + 1) * kBatch, CTA_S, wv);
                    }
                    if ((k + 1) * kBatch == ThreadMapKV::kIterS) {
                        block_iter.Advance();
                        __pipeline_commit();
                    }
                },
                [&] { smem_K.Load(frag_K[0], 0, rk); },
                [&] {
                    // block_iter.Advance();
                    // __pipeline_commit();
                    __pipeline_wait_prior(Stages - 2);
                    Impl::Sync();
                });
        };

        PRAGMA_UNROLL
        for (; tile_iter >= 0 && mask_iter != 0; --tile_iter, --mask_iter) {
            loop(std::true_type{}, std::true_type{});
        }

        PRAGMA_NO_UNROLL
        for (; tile_iter >= 0; --tile_iter) {
            loop(std::false_type{}, std::false_type{});
        }

        __pipeline_commit();
        __pipeline_wait_prior(0);
    }

    template<class GmemIterK, class GmemIterV, class BlockIter, class StoreS>
    __device__ void Run(Sm80_CpAsync<2>,
                        FragQ&         frag_Q,
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
        gmem_K.SetSmem(storage.K);
        gmem_V.SetSmem(storage.V);

        SmemIterQ smem_Q{storage.Q};
        SmemIterK smem_K{storage.K};
        SmemIterP smem_P{storage.P};
        SmemIterV smem_V{storage.V};

        gmem_K.ClearSmem(0);
        gmem_V.ClearSmem(0);

        block_iter.SetTile(tile_iter);

        gmem_K.Prefetch<true>(block_iter, max_step - tile_iter * CTA_S, 0);
        __pipeline_commit();

        __pipeline_wait_prior(0);
        Impl::Sync();

        FragK frag_K;
        FragV frag_V;
        smem_K.Load(frag_K[0], 0, 0);

        auto loop = [&](auto is_residue, auto is_mask) {
            const int offset_K = tile_iter * CTA_S;

            // __pipeline_wait_prior(0);
            // Impl::Sync();

            __align__(16) FragS frag_S{};

            gmem_V.Prefetch<is_residue>(block_iter, is_residue ? max_step - offset_K : CTA_S, 0);
            __pipeline_commit();

            block_iter.Advance();

            Impl::ComputeQK(smem_Q, smem_K, frag_Q, frag_K, frag_S, 0, [&] {
                __pipeline_wait_prior(0);
                Impl::Sync();
                smem_V.Load(frag_V[0], 0, 0);
            });

            // __pipeline_wait_prior(0);
            // Impl::Sync();

            if (tile_iter > 0) {
                gmem_K.Prefetch<false>(block_iter, CTA_S, 0);
                __pipeline_commit();
            }

            // store_S(frag_S, offset_K);

            if constexpr (is_mask) {
                ApplyCasualMask(frag_S, offset_Q, offset_K);
            }

            Impl::Softmax<is_mask>(frag_S, frag_M, frag_L, frag_O, qk_scale);

            __align__(16) FragP frag_P;

            Impl::ConvertStoP(frag_S, frag_P, storage.P);

            Impl::ComputePV(smem_P, smem_V, frag_P, frag_V, frag_O, 0, [&] {
                __pipeline_wait_prior(0);
                Impl::Sync();
                smem_K.Load(frag_K[0], 0, 0);
            });
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
        Impl::ForeachS(frag_S, [&](int hi, int qi, int si, float& score) {
            if (offset_Q + qi < offset_K + si) {
                score -= std::numeric_limits<float>::infinity();
            }
        });
    }

    __device__ void CommitAndWait()
    {
        __pipeline_commit();
        __pipeline_wait_prior(Stages - 2);
        Impl::Sync();
    }
};

}  // namespace turbomind::attention