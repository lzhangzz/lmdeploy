// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "array_ops.h"
#include "iterator.h"
#include "quantization.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"
#include "thread_map.h"
#include <cstdint>
#include <cuda_pipeline_primitives.h>
#include <limits>
#include <type_traits>

#include "attention_params.h"
#include "policy_sm70.h"
#include "policy_sm80.h"

namespace turbomind {

template<class T, class Tkv, class BlockSeqLen, int CTA_Q, int CTA_S, int HeadDim, int Stages>
struct Attention: AttentionPolicy<sm70_t, T, Tkv, CTA_Q, CTA_S, HeadDim> {
    using Policy    = AttentionPolicy<sm70_t, T, Tkv, CTA_Q, CTA_S, HeadDim>;
    using ParamType = AttentionParams<T>;

    using Policy::kPadQ;
    using Policy::kPadK;
    using Policy::kPadV;
    using Policy::kWarpCount;

    using Policy::WARP_S;
    using Policy::WARP_Q;

    using typename Policy::FragQ;
    using typename Policy::FragK;
    using typename Policy::FragS;
    using typename Policy::FragP;
    using typename Policy::FragV;
    using typename Policy::FragO;
    using typename Policy::FragM;
    using typename Policy::FragL;

    using typename Policy::Swizzle;

    using typename Policy::SwizzleV;

    static constexpr int kHeadDim = HeadDim;
    static constexpr int kStages  = Stages;

    static_assert(kStages == 2);

    struct SharedStorage {
        // __align__(16) T smem_Q[CTA_Q][kHeadDim + kPadQ];
        // __align__(16) Tkv smem_K[CTA_S][kHeadDim + kPadK];
        // __align__(16) Tkv smem_V[CTA_S][kHeadDim + kPadV];
        // __align__(16) T smem_P[CTA_Q][CTA_S + kPadQ];
        union {
            __align__(16) T smem_Q[CTA_Q][kHeadDim + kPadQ];
            struct {
                __align__(16) Tkv smem_K[CTA_S][kHeadDim + kPadK];
                __align__(16) Tkv smem_V[CTA_S][kHeadDim + kPadV];
                __align__(16) T smem_P[CTA_Q][CTA_S + kPadQ];
            };
        };
    };

    const ParamType& params_;

    int query_idx_;
    int head_idx_;
    int batch_idx_;
    int warp_id_;
    int lane_id_;

    BlockSeqLen block_seqlen_;

    const Tkv** k_cache_ptrs_;

    Tkv* smem_K_;
    Tkv* smem_V_;
    T*   smem_Q_;
    T*   smem_P_;

    int max_context_len_{};

    template<int I>
    using Int = std::integral_constant<int, I>;

    __device__ bool thread0()
    {
        return blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0;
    }

    __device__ Attention(const ParamType& params, uint8_t* dsmem): params_(params)
    {
        SharedStorage* shared = (SharedStorage*)dsmem;

        smem_Q_ = (T*)&shared->smem_Q;
        smem_K_ = (Tkv*)&shared->smem_K;
        smem_V_ = (Tkv*)&shared->smem_V;
        smem_P_ = (T*)&shared->smem_P;

        // [q, h, b]
        query_idx_ = blockIdx.x * CTA_Q;  // local offset into `input_length`
        head_idx_  = blockIdx.z;
        batch_idx_ = blockIdx.y;

        if constexpr (std::is_integral_v<BlockSeqLen>) {
            block_seqlen_ = params.kv_cache_block_size;
        }

        max_context_len_ = params_.max_input_len + params_.max_seq_len;

        warp_id_ = threadIdx.x / WARP_SIZE;
        lane_id_ = threadIdx.x % WARP_SIZE;

        k_cache_ptrs_ = (const Tkv**)params_.k_cache_block_ptrs + params_.cu_block_cnts[batch_idx_];
    }

    __device__ void LoadQ(FragQ& frag_Q)
    {
        constexpr int kVecSize = 8;  // sizeof(uint4) / sizeof(T);

        using Vec = Array<T, kVecSize>;
        using Map = RakedThreadMap<kHeadDim, CTA_Q, kVecSize, kWarpCount>;

        constexpr int ITER_C = Map::kIterC;
        constexpr int ITER_S = Map::kIterS;

        Vec vec_Q[ITER_S][ITER_C];

        const int qi_beg = params_.cu_seqlens[batch_idx_] + query_idx_;  // global offset into `cu_seqlens`
        const int qi_end = params_.cu_seqlens[batch_idx_ + 1];

        const int2 offset = Map::get_offset(warp_id_, lane_id_);

        // Load Q
        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                const int qi = offset.y + s * Map::kDeltaS + qi_beg;
                const int di = offset.x + c * Map::kDeltaC;
                if (qi < qi_end) {
                    Ldg(vec_Q[s][c], &params_.q[qi * params_.stride + head_idx_ * kHeadDim + di]);
                }
                else {
                    clear(vec_Q[s][c]);
                }
            }
        }

        // Optionally apply bias to Q
        if (params_.q_bias) {
            Vec bias_Q[ITER_C];
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                const int di = offset.x + c * Map::kDeltaC;
                Ldg(bias_Q[c], &params_.q_bias[head_idx_ * kHeadDim + di]);
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
                    const int qi = offset.y + s * Map::kDeltaS + query_idx_;
                    const int di = offset.x + c * Map::kDeltaC;
                    //
                    RotaryEmbedding<kVecSize> rope(10000.f, kHeadDim, qi, {di, 0});
                    rope.apply(vec_Q[s][c]);
                }
            }
        }

        // Store to shared memory
        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            const int qi = offset.y + s * Map::kDeltaS;
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                const int di = offset.x + c * Map::kDeltaC;
                // Store(&smem_Q_[Swizzle{}(qi * (kHeadDim + kPadQ) + di)], vec_Q[s][c]);
                Store(&smem_Q_[qi * (kHeadDim + kPadQ) + di + 0], (Array<T, 4>&)vec_Q[s][c][0]);
                Store(&smem_Q_[qi * (kHeadDim + kPadQ) + di + 4], (Array<T, 4>&)vec_Q[s][c][4]);
            }
        }

        Policy::TransformQ(smem_Q_, frag_Q);

        __syncthreads();
    }

    __device__ void ApplyCasualMask(FragS& frag_S, int offset_Q, int offset_K)
    {
        Policy::ForeachS(frag_S, [&](int qi, int si, float& score) {
            if (offset_Q + qi < offset_K + si) {
                score = -std::numeric_limits<float>::infinity();
            }
        });
    }

    __device__ void StoreS(const FragS& frag_S, int offset_K)
    {
        Policy::ForeachS(frag_S, [&](int qi, int si, float score) {
            qi += query_idx_;
            si += offset_K;
            if (qi < params_.max_input_len && si < max_context_len_) {
                params_.qk[batch_idx_ * params_.num_heads * params_.max_input_len * max_context_len_
                           + head_idx_ * params_.max_input_len * max_context_len_ + qi * max_context_len_ + si] = score;
            }
        });
    }

    __device__ void StoreP(FragP& frag_P, int offset_K)
    {
        Policy::ForeachP(frag_P, [&](int qi, int vi, float score) {
            qi += query_idx_;
            vi += offset_K;
            if (qi < params_.max_input_len && vi < max_context_len_) {
                params_.pr[batch_idx_ * params_.num_heads * params_.max_input_len * max_context_len_
                           + head_idx_ * params_.max_input_len * max_context_len_ + qi * max_context_len_ + vi] = score;
            }
        });
    }

    __device__ void StoreO(FragO& frag_O, const FragL& frag_L)
    {
        const int qi_beg = params_.cu_seqlens[batch_idx_] + query_idx_;  // global offset into `cu_seqlens`
        const int qi_end = params_.cu_seqlens[batch_idx_ + 1];
        Policy::StoreO(frag_O, frag_L, [&](int qi, int di, const auto& vec) {
            if (qi_beg + qi < qi_end) {
                Store(&params_.out[(qi_beg + qi) * params_.num_heads * kHeadDim + head_idx_ * kHeadDim + di], vec);
            }
        });
    }

    __device__ void Run()
    {
        // early exit if finished flag is set
        if (params_.finished[batch_idx_]) {
            return;
        }

        if (query_idx_ >= params_.input_length[batch_idx_]) {
            return;
        }
        using ThrMap = RakedThreadMap<kHeadDim, CTA_S, 8 /*sizeof(uint4) / sizeof(T)*/, kWarpCount>;
        using GmemK  = GmemIterator<Tkv, ThrMap, BlockSeqLen, Swizzle, kPadK, kStages>;
        using GmemV  = GmemIterator<Tkv, ThrMap, BlockSeqLen, SwizzleV, kPadV, kStages>;
        using SmemQ  = SmemIterator<T, kHeadDim, Swizzle, kPadQ>;
        using SmemK  = SmemIterator<T, kHeadDim, Swizzle, kPadK>;
        using SmemP  = SmemIterator<T, CTA_S, SwizzleV, kPadQ>;
        using SmemV  = SmemIterator<T, kHeadDim, SwizzleV, kPadV>;

        FragQ frag_Q;
        LoadQ(frag_Q);

        // [L, 2, H, s, D]
        int local_key_offset = params_.key_offset + head_idx_ * block_seqlen_ * kHeadDim;
        int local_val_offset = params_.val_offset + head_idx_ * block_seqlen_ * kHeadDim;

        GmemK gmem_K{k_cache_ptrs_, block_seqlen_, local_key_offset, smem_K_, warp_id_, lane_id_};
        GmemV gmem_V{k_cache_ptrs_, block_seqlen_, local_val_offset, smem_V_, warp_id_, lane_id_};

        gmem_K.ClearSmem(Int<0>{});
        gmem_V.ClearSmem(Int<0>{});

        __syncthreads();

        SmemQ smem_Q{smem_Q_};
        SmemK smem_K{smem_K_};
        SmemV smem_V{smem_V_};
        SmemP smem_P{smem_P_};
        // SmemIteratorQ<T, kHeadDim, Swizzle> smem_Q{smem_Q_};
        // SmemIteratorK<Tkv, kHeadDim, Swizzle> smem_K{smem_KV_};
        // SmemIteratorV<Tkv, kHeadDim, Swizzle> smem_V{smem_KV_ + CTA_S * (kHeadDim + SMEM_PAD)};
        // SmemIteratorK<Tkv, kHeadDim, Swizzle> smem_R{smem_R_};

        const int input   = params_.input_length[batch_idx_];
        const int context = params_.context_length[batch_idx_];
        const int history = context - input;

        const int offset_Q = history + query_idx_;

        const float qk_scale = params_.inv_sqrt_dh;

        // ceil(tiles) - 1
        int iter = (history + min(query_idx_ + CTA_Q, input) + CTA_S - 1) / CTA_S - 1;

        typename GmemK::Fragment fragment_K;
        typename GmemV::Fragment fragment_V;

        gmem_K.AdjustBlockTileIdx(iter);
        gmem_K.PrefetchStage(std::true_type{}, context - iter * CTA_S, fragment_K);
        CpAsyncCommit();
        gmem_K.Save(fragment_K);
        // gmem_K.Load<true>(fragment_K, context - iter * CTA_S);
        // gmem_K.Save(fragment_K);
        // // CpAsyncCommit();

        __align__(16) FragO frag_O{};

        FragL frag_L{};
        FragM frag_M;
        fill(frag_M, -std::numeric_limits<float>::infinity());

        auto loop = [&](auto is_residue, auto is_mask) {
            const int offset_K = iter * CTA_S;

            __align__(16) FragS frag_S{};

            CpAsyncWait();
            __syncthreads();  // Wait while smem_K being filled & smem_V being used

            // Prefetch for V, gmem_V[i-1] -> smem_V
            gmem_V.AdjustBlockTileIdx(iter);
            gmem_V.PrefetchStage(is_residue, is_residue ? context - offset_K : CTA_S, fragment_V);
            CpAsyncCommit();

            // gmem_V.Load<is_residue>(fragment_V, is_residue ? context - offset_K : CTA_S);

            Policy::ComputeQK(smem_Q, smem_K, frag_Q, frag_S);

            gmem_V.Save(fragment_V);

            CpAsyncWait();
            __syncthreads();  // Wait while trans_K being used & smem_K being filled

            // Prefetch for next K, gmem_K[i - 1] -> smem_K
            if (iter > 0) {
                gmem_K.AdjustBlockTileIdx(iter - 1);
                gmem_K.PrefetchStage(std::false_type{}, CTA_S, fragment_K);
                CpAsyncCommit();
            }

            if constexpr (is_mask) {
                // StoreS(frag_S, offset_K);
                ApplyCasualMask(frag_S, offset_Q, offset_K);
                // StoreS(frag_S, offset_K);
            }

            __align__(16) FragP frag_P;

            Policy::Softmax<is_residue>(frag_S, frag_M, frag_L, frag_P, frag_O, qk_scale, smem_P_);
            // StoreS(frag_S, offset_K);
            // StoreP(frag_P, offset_K);

            Policy::ComputePV(smem_P, smem_V, frag_P, frag_O);

            gmem_K.Save(fragment_K);
        };

        int mask_iter = 4;

        PRAGMA_UNROLL
        for (; iter >= 0 && mask_iter != 0; --iter, --mask_iter) {
            loop(std::true_type{}, std::true_type{});
        }

        for (; iter >= 0; --iter) {
            loop(std::false_type{}, std::false_type{});
        }

        StoreO(frag_O, frag_L);
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

extern __shared__ uint8_t dynamic_smem[];

template<typename MHAType, typename ParamType = typename MHAType::ParamType>
__global__ void __launch_bounds__(256, 1) attention_kernel(ParamType params)
{
    MHAType{params, dynamic_smem}.Run();
}

template<typename MHAType, typename ParamType = typename MHAType::ParamType>
__global__ void attention_reduction_kernel(ParamType params)
{
    MHAType::Reduce(params);
}

}  // namespace turbomind
