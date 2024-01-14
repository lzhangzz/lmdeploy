// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "array_ops.h"

#include "iterator.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"
#include "thread_map.h"
#include <cstdint>
#include <limits>
#include <type_traits>

#include "attention_params.h"

namespace turbomind {

template<class Mainloop, class BlockSeqLen, class CtaMap_>
struct AttentionUniversal {

    using T   = typename Mainloop::T;
    using Tkv = typename Mainloop::Tkv;

    using Impl   = typename Mainloop::Impl;
    using CtaMap = CtaMap_;

    static constexpr int kWarpCount = Impl::kWarpCount;

    using ParamType = AttentionParams<T>;

    static constexpr int kHeadDim = Impl::kHeadDim;

    using FragQ = typename Impl::FragQ;
    using FragO = typename Impl::FragO;
    using FragM = typename Impl::FragM;
    using FragL = typename Impl::FragL;

    using GmemIterK = typename Mainloop::GmemIterK;
    using GmemIterV = typename Mainloop::GmemIterV;

    static constexpr int CTA_H = Impl::CTA_H;
    static constexpr int CTA_Q = Impl::CTA_Q;
    static constexpr int CTA_S = Impl::CTA_S;

    using SharedStorage = typename Mainloop::SharedStorage;

    __device__ void LoadQ(const ParamType& params,
                          T*               smem_Q,
                          FragQ&           frag_Q,
                          int              qi_begin,
                          int              qi_end,
                          int              query_idx,
                          int              head_idx,
                          int              batch_idx,
                          int              warp_id,
                          int              lane_id)
    {
        using Map = typename Impl::ThreadMapQ;

        constexpr int kVecSize = Map::kAccessC;

        using Vec = Array<T, kVecSize>;

        constexpr int ITER_C = Map::kIterC;
        constexpr int ITER_S = Map::kIterS;

        Vec vec_Q[ITER_S][ITER_C];

        const int2 offset = Map::get_offset(warp_id, lane_id);

        // Load Q
        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            const int si = offset.y + s * Map::kDeltaS;
            const int qi = si % CTA_Q + qi_begin;
            const int hi = si / CTA_Q + head_idx;
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                const int di = offset.x + c * Map::kDeltaC;
                if (qi < qi_end) {
                    Ldg(vec_Q[s][c], &params.q[qi * params.stride + hi * kHeadDim + di]);
                }
                else {
                    clear(vec_Q[s][c]);
                }
            }
        }

        // Optionally apply bias to Q
        if (params.q_bias) {
            Vec bias_Q[ITER_C];
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                const int di = offset.x + c * Map::kDeltaC;
                Ldg(bias_Q[c], &params.q_bias[head_idx * kHeadDim + di]);
            }
            using namespace ops;
            PRAGMA_UNROLL
            for (int s = 0; s < ITER_S; ++s) {
                PRAGMA_UNROLL
                for (int c = 0; c < ITER_C; ++c) {
                    vec_Q[s][c] = vec_Q[s][c] + bias_Q[c];
                }
            }
        }

        if constexpr (0) {
            PRAGMA_UNROLL
            for (int s = 0; s < ITER_S; ++s) {
                PRAGMA_UNROLL
                for (int c = 0; c < ITER_C; ++c) {
                    const int qi = offset.y + s * Map::kDeltaS + query_idx;
                    const int di = offset.x + c * Map::kDeltaC;
                    //
                    RotaryEmbedding<kVecSize> rope(10000.f, kHeadDim, qi, {di, 0});
                    rope.apply(vec_Q[s][c]);
                }
            }
        }

        using SmemLayoutQ = typename Impl::SmemLayoutQ;

        // Store to shared memory
        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            const int si = offset.y + s * Map::kDeltaS;
            const int qi = si % CTA_Q;
            const int hi = si / CTA_Q;
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                const int di = offset.x + c * Map::kDeltaC;
                if (qi < CTA_Q && hi < CTA_H) {
                    Store(&smem_Q[SmemLayoutQ::swizzle(hi * CTA_Q + qi, di)], vec_Q[s][c]);
                }
            }
        }

        Impl::TransformQ(smem_Q, frag_Q);
    }

    __device__ void operator()(const ParamType& params, char* smem_buf)
    {
        // [q, h, b]
        const int query_idx = CtaMap::query_idx() * CTA_Q;  // local offset into `input_length`
        const int head_idx  = CtaMap::head_idx() * CTA_H;
        const int batch_idx = CtaMap::batch_idx();

        // early exit if finished flag is set
        if (params.finished[batch_idx]) {
            return;
        }

        if (query_idx >= params.input_length[batch_idx]) {
            return;
        }

        const BlockSeqLen block_seq_len = [&]() -> BlockSeqLen {
            if constexpr (std::is_integral_v<BlockSeqLen>) {
                return params.kv_cache_block_size;
            }
            else {
                return {};
            }
        }();

        SharedStorage& storage = *(SharedStorage*)smem_buf;

        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        const int qi_begin = params.cu_seqlens[batch_idx] + query_idx;  // global offset into `cu_seqlens`
        const int qi_end   = params.cu_seqlens[batch_idx + 1];

        FragQ frag_Q;
        LoadQ(params, storage.Q, frag_Q, qi_begin, qi_end, query_idx, head_idx, batch_idx, warp_id, lane_id);

        const auto k_cache_ptrs = (const Tkv**)params.k_cache_block_ptrs + params.cu_block_cnts[batch_idx];

        const int kv_head_idx = head_idx * params.num_kv_heads / params.num_heads;

        // [L, 2, H, s, D]
        const int local_key_offset = params.key_offset + kv_head_idx * block_seq_len * kHeadDim;
        const int local_val_offset = params.val_offset + kv_head_idx * block_seq_len * kHeadDim;

        GmemIterK gmem_K{local_key_offset, warp_id, lane_id};
        GmemIterV gmem_V{local_val_offset, warp_id, lane_id};

        Block<Tkv, CTA_S, BlockSeqLen> block_iter(k_cache_ptrs, block_seq_len);

        const int input_len   = params.input_length[batch_idx];
        const int context_len = params.context_length[batch_idx];
        const int history_len = context_len - input_len;
        const int offset_Q    = history_len + query_idx;

        const float qk_scale = params.inv_sqrt_dh;

        int tile_iter = (history_len + min(query_idx + CTA_Q, input_len) + CTA_S - 1) / CTA_S - 1;
        int mask_iter = 2;

        __align__(16) FragO frag_O{};

        FragL frag_L{};
        FragM frag_M;
        fill(frag_M, -std::numeric_limits<float>::infinity());

        __syncthreads();

        Mainloop mainloop;
        mainloop(frag_Q,
                 gmem_K,
                 gmem_V,
                 block_iter,
                 frag_O,
                 frag_M,
                 frag_L,
                 offset_Q,
                 context_len,
                 tile_iter,
                 mask_iter,
                 qk_scale,
                 storage,
                 StoreS(params, query_idx, head_idx, batch_idx, context_len));

        if constexpr (Impl::kWarpCntS > 1) {
            Impl::Merge(frag_O, frag_M, frag_L, qk_scale, storage);
        }

        StoreO(frag_O, frag_L, qi_begin, qi_end, head_idx, params);
    }

    __device__ void
    StoreO(FragO& frag_O, const FragL& frag_L, int qi_begin, int qi_end, int head_idx, const ParamType& params)
    {
        Impl::StoreO(frag_O, frag_L, [&](int hi, int qi, int di, const auto& vec) {
            if (qi_begin + qi < qi_end) {
                Store(&params.out[(qi_begin + qi) * params.num_heads * kHeadDim + (head_idx + hi) * kHeadDim + di],
                      vec);
            }
        });
    }

    __device__ auto StoreS(const ParamType& params,
                           const int&       query_idx,
                           const int&       head_idx,
                           const int&       batch_idx,
                           const int&       max_context_len)
    {
        return [&](auto& frag_S, int offset_K) {
            Impl::ForeachS(frag_S, [&](int hi, int qi, int si, float score) {
                qi += query_idx;
                si += offset_K;
                if (qi < params.max_input_len && si < max_context_len) {
                    params.qk[batch_idx * params.num_heads * params.max_input_len * max_context_len
                              + (head_idx + hi) * params.max_input_len * max_context_len + qi * max_context_len + si] =
                        score;
                }
            });
        };
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

extern __shared__ char dynamic_smem[];

template<class AttentionType, class ParamType = typename AttentionType::ParamType>
__global__ void attention_kernel(ParamType params)
{
    AttentionType{}(params, dynamic_smem);
}

template<typename MHAType, typename ParamType = typename MHAType::ParamType>
__global__ void attention_reduction_kernel(ParamType params)
{
    MHAType::Reduce(params);
}

}  // namespace turbomind
