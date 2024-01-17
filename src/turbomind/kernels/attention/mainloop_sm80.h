// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "iterator_sm80.h"
#include "mainloop.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"
#include <cuda_pipeline_primitives.h>
#include <limits>

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

    static constexpr int CTA_Q = Impl::CTA_Q;
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

    // template<class GmemIterK, class GmemIterV, class BlockIter, class StoreS, int Stages_>
    // __device__ void Run(Sm80_CpAsync<Stages_>,
    //                     FragQ&         frag_Q,
    //                     GmemIterK&     gmem_K,
    //                     GmemIterV&     gmem_V,
    //                     BlockIter&     block_iter,
    //                     FragO&         frag_O,
    //                     FragM&         frag_M,
    //                     FragL&         frag_L,
    //                     int            offset_Q,
    //                     int            max_step,
    //                     int            tile_iter,
    //                     int            mask_iter,
    //                     float          qk_scale,
    //                     SharedStorage& storage,
    //                     const StoreS&  store_S)
    // {
    //     gmem_K.SetSmem(storage.KV);
    //     gmem_V.SetSmem(storage.KV);

    //     SmemIterQ smem_Q{storage.Q};
    //     SmemIterP smem_P{storage.P};
    //     SmemIterK smem_K{storage.KV};
    //     SmemIterV smem_V{storage.KV};

    //     PRAGMA_UNROLL
    //     for (int i = 0; i < Stages; ++i) {
    //         gmem_K.ClearSmem(i * kTileSizeKV);
    //     }

    //     int kv_offset_r = 0;
    //     int kv_offset_w = 0;

    //     block_iter.SetTile(tile_iter);

    //     gmem_K.Prefetch<true>(
    //         block_iter, 0, ThreadMapKV::kIterS, max_step - tile_iter * CTA_S, SmemKVStep(kv_offset_w));
    //     __pipeline_commit();

    //     gmem_V.Prefetch<true>(
    //         block_iter, 0, ThreadMapKV::kIterS, max_step - tile_iter * CTA_S, SmemKVStep(kv_offset_w));
    //     __pipeline_commit();

    //     block_iter.Advance();

    //     PRAGMA_UNROLL
    //     for (int stages = 2; stages < Stages - 1; stages += 2) {
    //         gmem_K.Prefetch<false>(block_iter, 0, ThreadMapKV::kIterS, CTA_S, SmemKVStep(kv_offset_w));
    //         __pipeline_commit();

    //         gmem_V.Prefetch<false>(block_iter, 0, ThreadMapKV::kIterS, CTA_S, SmemKVStep(kv_offset_w));
    //         __pipeline_commit();

    //         block_iter.Advance();
    //     }

    //     FragK frag_K;
    //     FragV frag_V;

    //     __pipeline_wait_prior(Stages - 2);
    //     Impl::Sync();

    //     auto rk = SmemKVStep(kv_offset_r);
    //     smem_K.Load(frag_K[0], 0, rk);

    //     auto loop = [&](auto is_residue, auto is_mask) {
    //         const int offset_K = tile_iter * CTA_S;

    //         __align__(16) FragS frag_S[1]{};

    //         auto rv = SmemKVStep(kv_offset_r);
    //         auto wk = SmemKVStep(kv_offset_w);

    //         Impl::ComputeQK(
    //             smem_Q,
    //             smem_K,
    //             frag_Q,
    //             frag_K,
    //             frag_S[0],
    //             rk,
    //             [&](int k) {
    //                 constexpr int kBatch = ThreadMapKV::kIterS / 2;
    //                 if (k * kBatch < ThreadMapKV::kIterS) {
    //                     gmem_K.Prefetch<false>(block_iter, k * kBatch, (k + 1) * kBatch, CTA_S, wk);
    //                 }
    //                 if ((k + 1) * kBatch == ThreadMapKV::kIterS) {
    //                     __pipeline_commit();
    //                 }
    //             },
    //             [&] {
    //                 __pipeline_wait_prior(Stages - 2);
    //                 Impl::Sync();
    //                 smem_V.Load(frag_V[0], 0, rv);
    //             });

    //         if constexpr (is_mask) {
    //             ApplyCasualMask(frag_S[0], offset_Q, offset_K);
    //         }

    //         Impl::Softmax<is_mask>(frag_S, frag_M, frag_L, frag_O, qk_scale);

    //         __align__(16) FragP frag_P;

    //         Impl::ConvertStoP(frag_S[0], frag_P, storage.P);

    //         rk      = SmemKVStep(kv_offset_r);
    //         auto wv = SmemKVStep(kv_offset_w);
    //         Impl::ComputePV(
    //             smem_P,
    //             smem_V,
    //             frag_P,
    //             frag_V,
    //             frag_O,
    //             rv,
    //             [&](int k) {
    //                 constexpr int kBatch = ThreadMapKV::kIterS / 2;
    //                 if (k * kBatch < ThreadMapKV::kIterS) {
    //                     gmem_V.Prefetch<false>(block_iter, k * kBatch, (k + 1) * kBatch, CTA_S, wv);
    //                 }
    //                 if ((k + 1) * kBatch == ThreadMapKV::kIterS) {
    //                     block_iter.Advance();
    //                     __pipeline_commit();
    //                 }
    //             },
    //             [&] {
    //                 __pipeline_wait_prior(Stages - 2);
    //                 Impl::Sync();
    //                 smem_K.Load(frag_K[0], 0, rk);
    //             });
    //     };

    //     // PRAGMA_UNROLL
    //     for (; tile_iter >= 0 && mask_iter != 0; --tile_iter, --mask_iter) {
    //         loop(std::true_type{}, std::true_type{});
    //     }

    //     PRAGMA_NO_UNROLL
    //     for (; tile_iter >= 0; --tile_iter) {
    //         loop(std::false_type{}, std::false_type{});
    //     }

    //     __pipeline_commit();
    //     __pipeline_wait_prior(0);
    // }

    template<class GmemIterK, class GmemIterV, class BlockIter, class StoreS>
    __device__ void Run(Sm80_CpAsync<3>,
                        FragQ&         frag_Q,
                        GmemIterK&     gmem_K,
                        GmemIterV&     gmem_V,
                        BlockIter&     block_iter_,
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

        block_iter_.SetTile(tile_iter);

        auto block_iter_K = block_iter_;
        auto block_iter_V = block_iter_;

        gmem_K.Prefetch<true>(block_iter_K, max_step - tile_iter * CTA_S, SmemKVStep(kv_offset_w));
        block_iter_K.Advance();
        __pipeline_commit();

        gmem_K.Prefetch<false>(block_iter_K, CTA_S, SmemKVStep(kv_offset_w));
        block_iter_K.Advance();
        __pipeline_commit();

        FragK frag_K;
        FragV frag_V;

        __pipeline_wait_prior(Stages - 2);
        Impl::Sync();

        auto rk1 = SmemKVStep(kv_offset_r);
        smem_Q.Load(frag_Q[0], 0);
        smem_K.Load(frag_K[0], 0, rk1);

        auto loop = [&](auto is_residue, auto is_mask) {
            __align__(16) FragS frag_S[2]{};

            auto wv = SmemKVStep(kv_offset_w);

            if constexpr (is_residue) {
                gmem_V.Prefetch<true>(block_iter_V, max_step - tile_iter * CTA_S, wv);
                block_iter_V.Advance();
                __pipeline_commit();
            }

            const auto rk2 = SmemKVStep(kv_offset_r);

            Impl::ComputeQK(
                smem_Q,
                smem_K,
                frag_Q,
                frag_K,
                frag_S[0],
                rk1,
                [&](int k) {
                    if constexpr (is_residue.value) {
                        return;
                    }
                    constexpr int kBatch = ThreadMapKV::kIterS / 2;
                    if (k * kBatch < ThreadMapKV::kIterS) {
                        gmem_V.Prefetch<false>(block_iter_V, k * kBatch, (k + 1) * kBatch, CTA_S, wv);
                    }
                    if ((k + 1) * kBatch == ThreadMapKV::kIterS) {
                        block_iter_V.Advance();
                        __pipeline_commit();
                    }
                },
                [&] {
                    smem_Q.Load(frag_Q[0], 0);
                    __pipeline_wait_prior(Stages - 2);
                    Impl::Sync();
                    smem_K.Load(frag_K[0], 0, rk2);
                });

            if constexpr (is_mask) {
                ApplyCasualMask(frag_S[0], offset_Q, tile_iter * CTA_S);
            }

            wv = SmemKVStep(kv_offset_w);

            const auto rv1 = SmemKVStep(kv_offset_r);

            Impl::ComputeQK(
                smem_Q,
                smem_K,
                frag_Q,
                frag_K,
                frag_S[1],
                rk2,
                [&](int k) {
                    constexpr int kBatch = ThreadMapKV::kIterS / 2;
                    if (k * kBatch < ThreadMapKV::kIterS) {
                        gmem_V.Prefetch<false>(block_iter_V, k * kBatch, (k + 1) * kBatch, CTA_S, wv);
                    }
                    if ((k + 1) * kBatch == ThreadMapKV::kIterS) {
                        block_iter_V.Advance();
                        __pipeline_commit();
                    }
                },
                [&] {
                    __pipeline_wait_prior(Stages - 2);
                    Impl::Sync();
                    smem_V.Load(frag_V[0], 0, rv1);
                });

            if constexpr (is_mask) {
                ApplyCasualMask(frag_S[1], offset_Q, tile_iter * CTA_S - CTA_S);
            }

            Impl::Softmax<is_mask>(frag_S, frag_M, frag_L, frag_O, qk_scale);

            __align__(16) FragP frag_P[2];

            Impl::ConvertStoP(frag_S[0], frag_P[0], storage.P);

            auto       wk  = SmemKVStep(kv_offset_w);
            const auto rv2 = SmemKVStep(kv_offset_r);

            Impl::ComputePV(
                smem_P,
                smem_V,
                frag_P[0],
                frag_V,
                frag_O,
                rv1,
                [&](int k) {
                    constexpr int kBatch = ThreadMapKV::kIterS / 2;
                    if (k * kBatch < ThreadMapKV::kIterS) {
                        gmem_K.Prefetch<false>(block_iter_K, k * kBatch, (k + 1) * kBatch, CTA_S, wk);
                    }
                    if ((k + 1) * kBatch == ThreadMapKV::kIterS) {
                        block_iter_K.Advance();
                        __pipeline_commit();
                    }
                },
                [&] {
                    __pipeline_wait_prior(Stages - 2);
                    Impl::Sync();
                    smem_V.Load(frag_V[0], 0, rv2);
                });

            wk  = SmemKVStep(kv_offset_w);
            rk1 = SmemKVStep(kv_offset_r);

            Impl::ConvertStoP(frag_S[1], frag_P[1], storage.P);

            Impl::ComputePV(
                smem_P,
                smem_V,
                frag_P[1],
                frag_V,
                frag_O,
                rv2,
                [&](int k) {
                    constexpr int kBatch = ThreadMapKV::kIterS / 2;
                    if (k * kBatch < ThreadMapKV::kIterS) {
                        gmem_K.Prefetch<false>(block_iter_K, k * kBatch, (k + 1) * kBatch, CTA_S, wk);
                    }
                    if ((k + 1) * kBatch == ThreadMapKV::kIterS) {
                        block_iter_K.Advance();
                        __pipeline_commit();
                    }
                },
                [&] {
                    smem_Q.Load(frag_Q[0], 0);
                    __pipeline_wait_prior(Stages - 2);
                    Impl::Sync();
                    smem_K.Load(frag_K[0], 0, rk1);
                });
        };

        static_assert(CTA_Q / CTA_S == 2);

        PRAGMA_UNROLL
        for (; tile_iter >= 0 && mask_iter > 0; tile_iter -= 2, --mask_iter -= 2) {
            loop(std::true_type{}, std::true_type{});
        }

        PRAGMA_NO_UNROLL
        for (; tile_iter >= 0; tile_iter -= 2) {
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
        gmem_K.SetSmem(storage.KV);
        gmem_V.SetSmem(storage.KV);

        SmemIterK smem_K{storage.KV};
        SmemIterV smem_V{storage.KV};
        SmemIterQ smem_Q{storage.Q};
        SmemIterP smem_P{storage.P};

        PRAGMA_UNROLL
        for (int i = 0; i < Stages; ++i) {
            gmem_K.ClearSmem(i * kTileSizeKV);
        }

        block_iter.SetTile(tile_iter);

        auto block_iter_K = block_iter;
        auto block_iter_V = block_iter;

        int kv_offset_r = 0;
        int kv_offset_w = 0;

        gmem_K.Prefetch<true>(block_iter_K, max_step - tile_iter * CTA_S, SmemKVStep(kv_offset_w));
        block_iter_K.Advance();
        __pipeline_commit();

        FragK frag_K;
        FragV frag_V;

        int rk1 = SmemKVStep(kv_offset_r);

        __pipeline_wait_prior(0);
        Impl::Sync();
        smem_Q.Load(frag_Q[0], 0);
        smem_K.Load(frag_K[0], 0, rk1);

        constexpr auto nop = [](int) {};

        auto loop = [&](auto is_residue, auto is_mask) {
            __align__(16) FragS frag_S[2]{};

            gmem_K.Prefetch<false>(block_iter_K, CTA_S, SmemKVStep(kv_offset_w));
            block_iter_K.Advance();
            __pipeline_commit();

            const int rk2 = SmemKVStep(kv_offset_r);

            Impl::ComputeQK(smem_Q, smem_K, frag_Q, frag_K, frag_S[0], rk1, nop, [&] {
                smem_Q.Load(frag_Q[0], 0);
                __pipeline_wait_prior(0);
                Impl::Sync();
                smem_K.Load(frag_K[0], 0, rk2);
            });

            gmem_V.Prefetch<is_residue>(block_iter_V, max_step - tile_iter * CTA_S, SmemKVStep(kv_offset_w));
            block_iter_V.Advance();
            __pipeline_commit();

            const int rv1 = SmemKVStep(kv_offset_r);

            Impl::ComputeQK(smem_Q, smem_K, frag_Q, frag_K, frag_S[1], rk2, nop, [&] {
                __pipeline_wait_prior(0);
                Impl::Sync();
                smem_V.Load(frag_V[0], 0, rv1);
            });

            gmem_V.Prefetch<false>(block_iter_V, CTA_S, SmemKVStep(kv_offset_w));
            block_iter_V.Advance();
            __pipeline_commit();

            if constexpr (is_mask) {
                ApplyCasualMask(frag_S[0], offset_Q, tile_iter * CTA_S);
                ApplyCasualMask(frag_S[1], offset_Q, tile_iter * CTA_S - CTA_S);
            }

            Impl::Softmax<is_mask>(frag_S, frag_M, frag_L, frag_O, qk_scale);

            __align__(16) FragP frag_P[2];

            const int rv2 = SmemKVStep(kv_offset_r);

            Impl::ConvertStoP(frag_S[0], frag_P[0], storage.P);
            Impl::ConvertStoP(frag_S[1], frag_P[1], storage.P);

            Impl::ComputePV(smem_P, smem_V, frag_P[0], frag_V, frag_O, rv1, nop, [&] {
                __pipeline_wait_prior(0);
                Impl::Sync();
                smem_V.Load(frag_V[0], 0, rv2);
            });

            gmem_K.Prefetch<false>(block_iter_K, CTA_S, SmemKVStep(kv_offset_w));
            block_iter_K.Advance();
            __pipeline_commit();

            rk1 = SmemKVStep(kv_offset_r);

            Impl::ComputePV(smem_P, smem_V, frag_P[1], frag_V, frag_O, rv2, nop, [&] {
                smem_Q.Load(frag_Q[0], 0);
                __pipeline_wait_prior(0);
                Impl::Sync();
                smem_K.Load(frag_K[0], 0, rk1);
            });
        };

        PRAGMA_UNROLL
        for (; tile_iter >= 0 && mask_iter > 0; tile_iter -= 2, mask_iter -= 2) {
            loop(std::true_type{}, std::true_type{});
        }

        PRAGMA_NO_UNROLL
        for (; tile_iter >= 0; tile_iter -= 2) {
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
};

}  // namespace turbomind::attention