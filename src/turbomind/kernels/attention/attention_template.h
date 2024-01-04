// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "array_ops.h"
#include "quantization.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"
#include "thread_map.h"
#include <cstdint>
#include <cuda_pipeline_primitives.h>
#include <limits>
#include <type_traits>

#include "attention_params.h"

// template<int Stages>
// struct Sm80 {};
// template<class OpClass, int Stages>
// struct Sm80Decoding {};

// struct Sm75 {};
// template<class OpClass>
// struct Sm75Decoding {};

// struct Sm70 {};
// template<class OpClass>
// struct Sm70Decoding {};

namespace turbomind {

template<class Mainloop, class BlockSeqLen>
struct Attention {

    using T   = typename Mainloop::T;
    using Tkv = typename Mainloop::Tkv;

    using Impl = typename Mainloop::Impl;

    static constexpr int kWarpCount = Impl::kWarpCount;

    using ParamType = AttentionParams<T>;

    static constexpr int kHeadDim = Impl::kHeadDim;

    using FragQ = typename Impl::FragQ;
    using FragO = typename Impl::FragO;
    using FragM = typename Impl::FragM;
    using FragL = typename Impl::FragL;

    using GmemIterK = typename Mainloop::template GmemIterK<BlockSeqLen>;
    using GmemIterV = typename Mainloop::template GmemIterV<BlockSeqLen>;

    static constexpr int CTA_Q = Impl::CTA_Q;
    static constexpr int CTA_S = Impl::CTA_S;

    using SharedStorage = typename Impl::SharedStorage;

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
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                const int qi = offset.y + s * Map::kDeltaS + qi_begin;
                const int di = offset.x + c * Map::kDeltaC;
                if (qi < qi_end) {
                    Ldg(vec_Q[s][c], &params.q[qi * params.stride + head_idx * kHeadDim + di]);
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
            const int qi = offset.y + s * Map::kDeltaS;
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                const int di = offset.x + c * Map::kDeltaC;
                Store(&smem_Q[SmemLayoutQ::swizzle(qi, di)], vec_Q[s][c]);
            }
        }

        Impl::TransformQ(smem_Q, frag_Q);

        __syncthreads();
    }

    __device__ void operator()(const ParamType& params, char* smem_buf)
    {
        // [q, h, b]
        const int query_idx = blockIdx.x * CTA_Q;  // local offset into `input_length`
        const int head_idx  = blockIdx.z;
        const int batch_idx = blockIdx.y;

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

        // [L, 2, H, s, D]
        const int local_key_offset = params.key_offset + head_idx * block_seq_len * kHeadDim;
        const int local_val_offset = params.val_offset + head_idx * block_seq_len * kHeadDim;

        GmemIterK gmem_K{k_cache_ptrs, block_seq_len, local_key_offset, storage.K, warp_id, lane_id};
        GmemIterV gmem_V{k_cache_ptrs, block_seq_len, local_val_offset, storage.V, warp_id, lane_id};

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

        gmem_K.ClearSmem();
        gmem_V.ClearSmem();

        __syncthreads();

        Mainloop mainloop;
        mainloop(frag_Q,
                 gmem_K,
                 gmem_V,
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

        StoreO(frag_O, frag_L, qi_begin, qi_end, head_idx, params);
    }

    __device__ void
    StoreO(FragO& frag_O, const FragL& frag_L, int qi_begin, int qi_end, int head_idx, const ParamType& params)
    {
        Impl::StoreO(frag_O, frag_L, [&](int qi, int di, const auto& vec) {
            if (qi_begin + qi < qi_end) {
                Store(&params.out[(qi_begin + qi) * params.num_heads * kHeadDim + head_idx * kHeadDim + di], vec);
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
            Impl::ForeachS(frag_S, [&](int qi, int si, float score) {
                qi += query_idx;
                si += offset_K;
                if (qi < params.max_input_len && si < max_context_len) {
                    params.qk[batch_idx * params.num_heads * params.max_input_len * max_context_len
                              + head_idx * params.max_input_len * max_context_len + qi * max_context_len + si] = score;
                }
            });
        };
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<class T, int N>
__device__ void CpAsync(T* dst, const Array<T, N>* __restrict__ src, bool mask)
{
    const int     smem_int_ptr = cast_smem_ptr_to_uint(dst);
    constexpr int cp_size      = sizeof(Array<T, N>);
#if TURBOMIND_ARCH_SM80
    // clang-format off
        asm volatile("{\n"
                     "  .reg .pred p;\n"
                     "  setp.ne.b32 p, %0, 0;\n"
                     "  @p cp.async.cg.shared.global " L2_CACHEHINT(128) " [%1], [%2], %3;\n"
                     "}\n" ::"r"((int)mask),
                     "r"(smem_int_ptr),
                     "l"(src),
                     "n"(cp_size));
    // clang-format on
    // " L2_CACHEHINT(128) "
#else
    assert(TURBOMIND_ARCH_SM80);
#endif
}

template<class T, class Tkv, int CTA_Q, int kHeadDim, int kWarpCount, class ParamType>
__global__ void __launch_bounds__(128, 8) ProcessKV(ParamType params)
{
    constexpr int kVecSize = sizeof(uint4) / sizeof(T);

    using Vec = Array<T, kVecSize>;
    using Map = RakedThreadMap<kHeadDim, CTA_Q, kVecSize, kWarpCount>;

    constexpr int ITER_C = Map::kIterC;
    constexpr int ITER_S = Map::kIterS;

    const int token_idx = blockIdx.x * CTA_Q;  // local offset into `input_length`
    const int head_idx  = blockIdx.y;
    const int batch_idx = blockIdx.z;

    const int qi_beg = params.cu_seqlens[batch_idx] + token_idx;  // global offset into `cu_seqlens`
    const int qi_end = params.cu_seqlens[batch_idx + 1];

    const int input_len   = params.input_length[batch_idx];
    const int history_len = params.context_length[batch_idx] - input_len;

    if (token_idx >= input_len) {  // empty tile
        return;
    }

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int2 offset = Map::get_offset(warp_id, lane_id);

    Vec __align__(16) vec_K[ITER_S][ITER_C];
    Vec __align__(16) vec_V[ITER_S][ITER_C];

    Vec bias_V[ITER_C];
    Vec bias_K[ITER_C];

    if (params.k_bias) {
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int di = offset.x + c * Map::kDeltaC;
            Ldg(bias_K[c], &params.k_bias[head_idx * kHeadDim + di]);
        }
    }
    if (params.v_bias) {
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int di = offset.x + c * Map::kDeltaC;
            Ldg(bias_V[c], &params.v_bias[head_idx * kHeadDim + di]);
        }
    }

    PRAGMA_UNROLL
    for (int s = 0; s < ITER_S; ++s) {
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int qi = offset.y + s * Map::kDeltaS + qi_beg;
            const int di = offset.x + c * Map::kDeltaC;
            if (qi < qi_end) {
                Ldg(vec_K[s][c], &params.k[qi * params.stride + head_idx * kHeadDim + di]);
                Ldg(vec_V[s][c], &params.v[qi * params.stride + head_idx * kHeadDim + di]);
            }
        }
    }

    if (params.k_bias) {
        using namespace ops;
        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                vec_K[s][c] = vec_K[s][c] + bias_K[c];
            }
        }
    }
    if (params.v_bias) {
        using namespace ops;
        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                vec_V[s][c] = vec_V[s][c] + bias_V[c];
            }
        }
    }

    Tkv** k_cache_block_ptrs = (Tkv**)params.k_cache_block_ptrs + params.cu_block_cnts[batch_idx];

    Array<Tkv, kVecSize> out_K[ITER_S][ITER_C];
    Array<Tkv, kVecSize> out_V[ITER_S][ITER_C];

    // quant param
    using PType = T;
    Array<PType, 2> param_K[ITER_S];
    Array<PType, 2> param_V[ITER_S];

    if constexpr (std::is_same_v<T, Tkv>) {
        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                out_K[s][c] = vec_K[s][c];
                out_V[s][c] = vec_V[s][c];
            }
        }
    }
    else if constexpr (1) {
        constexpr std::integral_constant<int, sizeof(Tkv) * 8> n_bits{};
        warp_stats<Map::kWarpThreadC>(param_K, vec_K, n_bits);
        warp_stats<Map::kWarpThreadC>(param_V, vec_V, n_bits);
        quantize(out_K, vec_K, param_K, n_bits);
        quantize(out_V, vec_V, param_V, n_bits);
        fuse_magic(param_K);
        fuse_magic(param_V);
    }
    else {
        using QType = uint8_t;
        constexpr std::integral_constant<int, sizeof(QType) * 8> n_bits{};
        // quant data
        Array<QType, kVecSize> quant_K[ITER_S][ITER_C];
        Array<QType, kVecSize> quant_V[ITER_S][ITER_C];
        warp_stats<Map::kWarpThreadC>(param_K, vec_K, n_bits);
        warp_stats<Map::kWarpThreadC>(param_V, vec_V, n_bits);
        quantize(quant_K, vec_K, param_K, n_bits);
        quantize(quant_V, vec_V, param_V, n_bits);
        dequantize(out_K, quant_K, param_K, n_bits);
        dequantize(out_V, quant_V, param_V, n_bits);
    }

    // if constexpr (std::is_same_v<Tkv, uint8_t>) {
    //     PRAGMA_UNROLL
    //     for (int s = 0; s < ITER_S; ++s) {
    //         PRAGMA_UNROLL
    //         for (int c = 0; c < ITER_C; ++c) {
    //             permute_K(out_K[s][c]);
    //         }
    //     }
    //     permute_V<Map>(out_V);
    // }

    PRAGMA_UNROLL
    for (int s = 0; s < ITER_S; ++s) {
        const int si = offset.y + s * Map::kDeltaS;
        const int qi = si + token_idx;  // local offset into `input_length`

        if (qi < input_len) {
            const int ti = history_len + qi;  // timestep

            const int block_seqlen = params.kv_cache_block_size;
            // block index and local offsets
            const int cache_block_index  = ti / block_seqlen;
            const int cache_block_offset = ti % block_seqlen;
            // [H, s, D]
            Tkv* k_cache = k_cache_block_ptrs[cache_block_index] + params.key_offset
                           + head_idx * block_seqlen * kHeadDim + cache_block_offset * kHeadDim;
            Tkv* v_cache = k_cache_block_ptrs[cache_block_index] + params.val_offset
                           + head_idx * block_seqlen * kHeadDim + cache_block_offset * kHeadDim;
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                int di = offset.x + c * Map::kDeltaC;
                // di ^= ((si & 0x7) << 3);
                Stcs(&k_cache[di], out_K[s][c]);
                Stcs(&v_cache[di], out_V[s][c]);
            }

            if (std::is_same_v<Tkv, uint8_t>) {
                int max_context_len = params.max_input_len + params.max_seq_len;
                // [B, H, 2, S, 2]
                auto k_cache_quant_data = params.kv_cache_quant_data
                                          + batch_idx * params.num_kv_heads * 2 * max_context_len * 2
                                          + head_idx * 2 * max_context_len * 2  //
                                          + (history_len + qi) * 2;
                auto v_cache_quant_data = k_cache_quant_data + max_context_len * 2;

                if (offset.x == 0) {  // thread group leader stores
                    Stcs(k_cache_quant_data, param_K[s]);
                    Stcs(v_cache_quant_data, param_V[s]);
                }
            }
        }
    }
}

extern __shared__ char dynamic_smem[];

template<typename AttentionType, typename ParamType = typename AttentionType::ParamType>
__global__ void __launch_bounds__(256, 1) attention_kernel(ParamType params)
{
    AttentionType{}(params, dynamic_smem);
    // MHAType{params, dynamic_smem}.Run();
}

template<typename MHAType, typename ParamType = typename MHAType::ParamType>
__global__ void attention_reduction_kernel(ParamType params)
{
    MHAType::Reduce(params);
}

}  // namespace turbomind
