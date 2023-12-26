// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "array_ops.h"
#include "iterator.h"
#include "quantization.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"
#include "thread_map.h"
#include <climits>
#include <cmath>
#include <cstdint>
#include <cuda_pipeline_primitives.h>
#include <limits>
#include <type_traits>

#include "attention_params.h"
#include "policy_sm80.h"

namespace turbomind {

__inline__ __device__ void
mma_m8n8k4_row_col(Array<float, 8>& d, const Array<half, 4>& a, const Array<half, 4>& b, Array<float, 8>& c)
{
#if TURBOMIND_ARCH_SM70
    uint32_t const* A = reinterpret_cast<uint32_t const*>(&a);
    uint32_t const* B = reinterpret_cast<uint32_t const*>(&b);
    // clang-format off
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32"
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11},"
        "{%12, %13, %14, %15, %16, %17, %18, %19};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
        : "r"(A[0]), "r"(A[1]), 
          "r"(B[0]), "r"(B[1]), 
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "f"(c[4]), "f"(c[5]), "f"(c[6]), "f"(c[7]));
// clang-format on
#endif
}

template<class T, class Tkv, class BlockSeqLen, int CTA_Q, int CTA_S, int HeadDim, int Stages>
struct Attention: AttentionPolicy<sm80_t, T, Tkv, CTA_Q, CTA_S, HeadDim> {

    using Policy    = AttentionPolicy<sm80_t, T, Tkv, CTA_Q, CTA_S, HeadDim>;
    using ParamType = AttentionParams<T>;

    using Policy::kSmemPadding;
    using Policy::kWarpCount;

    using Policy::WARP_S;
    using Policy::WARP_Q;

    using Policy::K_ITER_M;
    using Policy::K_ITER_N;
    using Policy::K_ITER_K;
    using Policy::V_ITER_M;
    using Policy::V_ITER_N;
    using Policy::V_ITER_K;

    using typename Policy::FragQ;
    using typename Policy::FragK;
    using typename Policy::FragS;
    using typename Policy::FragPs;
    using typename Policy::FragP;
    using typename Policy::FragV;
    using typename Policy::FragO;
    using typename Policy::FragM;
    using typename Policy::FragL;
    using typename Policy::TransformedK;
    using typename Policy::TransformedV;

    static constexpr int kHeadDim = HeadDim;
    static constexpr int kStages  = Stages;

    static_assert(kStages == 2);

    struct SharedStorage {
        T   smem_Q[CTA_Q][kHeadDim + SMEM_PAD];
        Tkv smem_K[CTA_S][kHeadDim + SMEM_PAD];
        Tkv smem_V[CTA_S][kHeadDim + SMEM_PAD];
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

    int max_context_len_{};

    template<int I>
    using Int = std::integral_constant<int, I>;

    struct Swizzle {
        __device__ int operator()(int index)
        {
            // sssSSSdDDDddd
            // DDD ^= SSS
            constexpr int mask = 0x7 << 7;
            return index ^ ((index & mask) >> 4);
        }
    };

    struct Identity {
        template<class X>
        __device__ X operator()(X x)
        {
            return x;
        }
    };

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

    static constexpr bool kUseSmemQ = false;

    //

    static_assert(sizeof(FragPs) == sizeof(FragP));

    __device__ void LoadQ(FragQ& frag_Q)
    {
        constexpr int kVecSize = sizeof(uint4) / sizeof(T);

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
                Store(&smem_Q_[Swizzle{}(qi * (kHeadDim + SMEM_PAD) + di)], vec_Q[s][c]);
            }
        }

        __syncwarp();

        uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_Q_);
        if constexpr (!kUseSmemQ) {
            // Load from shared memory using LDSM, rearrange to m16n8k16 atom layout
            PRAGMA_UNROLL
            for (int m = 0; m < K_ITER_M; ++m) {
                PRAGMA_UNROLL
                for (int k = 0; k < K_ITER_K; ++k) {
                    auto&     Q   = (Array<uint32_t, 4>&)frag_Q[k][m];
                    const int mm  = m * 16 + lane_id_ % 16 + warp_id_ * WARP_Q;
                    const int kk  = k * 16 + lane_id_ / 16 * 8;
                    const int idx = Swizzle{}(mm * (kHeadDim + SMEM_PAD) + kk);
                    ldmatrix_m8n8_x4_b16(Q[0], Q[1], Q[2], Q[3], smem_int_ptr + sizeof(T) * idx);
                }
            }
        }
        else {
            // Rearrange Q in smem so that swizzling is not needed for later LDSMs
            const int lane_id       = threadIdx.x % WARP_SIZE;
            const int group_id      = lane_id / 16;
            const int group_lane_id = lane_id % 16;
            PRAGMA_UNROLL
            for (int k = 0; k < K_ITER_K; ++k) {
                PRAGMA_UNROLL
                for (int m = 0; m < K_ITER_M; ++m) {
                    auto&     r   = (Array<uint32_t, 4>&)frag_Q[k][m];
                    const int s   = m * 16 + group_lane_id % 8 + group_id * 8 + warp_id_ * WARP_Q;
                    const int c   = k * 16 + group_lane_id / 8 * 8;
                    const int idx = Swizzle{}(s * (kHeadDim + SMEM_PAD) + c);
                    ldmatrix_m8n8_x4_b16(r[0], r[2], r[1], r[3], smem_int_ptr + sizeof(T) * idx);
                }
            }

            __syncthreads();

            constexpr int THREADS = kWarpCount * WARP_SIZE;
            PRAGMA_UNROLL
            for (int k = 0; k < K_ITER_K; ++k) {
                PRAGMA_UNROLL
                for (int m = 0; m < K_ITER_M; ++m) {
                    constexpr int kVecSize = 8;
                    Store(&smem_Q_[(k * K_ITER_M * THREADS + m * THREADS + threadIdx.x) * kVecSize], frag_Q[k][m]);
                }
            }
        }
    }

    template<bool is_residue, class SmemQ, class SmemK>
    __device__ void ComputeQK(SmemQ& smem_Q, SmemK& smem_K, FragQ& frag_Q, FragS& frag_S, int offset_K)
    {
        TransformedK frag_K;

        smem_K.LoadK(frag_K[0], 0);
        if constexpr (kUseSmemQ) {
            smem_Q.LoadQ_(frag_Q[0], 0);
        }

        PRAGMA_UNROLL
        for (int k = 0; k < K_ITER_K; ++k) {
            if (k < K_ITER_K - 1) {
                smem_K.LoadK(frag_K[k + 1], k + 1);
                if constexpr (kUseSmemQ) {
                    smem_Q.LoadQ_(frag_Q[k + 1], k + 1);
                }
            }
            PRAGMA_UNROLL
            for (int m = 0; m < K_ITER_M; ++m) {
                PRAGMA_UNROLL
                for (int n = K_ITER_N - 1; n >= 0; --n) {
                    mma_m16n8k16_row_col(frag_S[m][n], frag_Q[k][m], frag_K[k][n], frag_S[m][n]);
                }
            }
        }
    }

    template<class Smem>
    __device__ void ComputePV(Smem& smem, const FragP& frag_P, FragO& frag_O)
    {
        TransformedV frag_V;

        smem.LoadV(frag_V[0], 0);

        PRAGMA_UNROLL
        for (int k = 0; k < V_ITER_K; ++k) {
            if (k < V_ITER_K - 1) {
                smem.LoadV(frag_V[k + 1], k + 1);
            }
            PRAGMA_UNROLL
            for (int m = 0; m < V_ITER_M; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < V_ITER_N; ++n) {
                    mma_m16n8k16_row_col(frag_O[m][n], frag_P[m][k], frag_V[k][n], frag_O[m][n]);
                }
            }
        }
        // smem.Advance(-V_ITER_K);
    }

    __device__ void ApplyCasualMask(FragS& frag_S, int offset_Q, int offset_K)
    {
        Policy::ApplyCasualMask(frag_S, [&](int qi, int ki, float& score) {
            if (offset_Q + qi < offset_K + ki) {
                score = -std::numeric_limits<float>::infinity();
            }
        });
    }

    __device__ void StoreS(const FragS& frag_S, int offset_K)
    {
        Policy::StoreS(frag_S, [&](int qi, int ki, float score) {
            qi += query_idx_;
            ki += offset_K;
            if (qi < params_.max_input_len && ki < max_context_len_) {
                params_.qk[batch_idx_ * params_.num_heads * params_.max_input_len * max_context_len_
                           + head_idx_ * params_.max_input_len * max_context_len_ + qi * max_context_len_ + ki] = score;
            }
        });
    }

    __device__ void StoreP(const FragP& frag_P, int offset_K)
    {
        Policy::StoreP(frag_P, [&](int qi, int vi, float score) {
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

    template<bool is_residue>
    __device__ void Softmax(FragS& frag_S, FragM& frag_M, FragM& frag_L, FragPs& frag_Ps, FragO& frag_O, float qk_scale)
    {
        Policy::Softmax<is_residue>(frag_S, frag_M, frag_L, frag_Ps, frag_O, qk_scale);
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
        using ThrMap = RakedThreadMap<kHeadDim, CTA_S, sizeof(uint4) / sizeof(T), kWarpCount>;
        using GmemK  = GmemIterator<Tkv, ThrMap, BlockSeqLen, Swizzle, kStages>;
        using GmemV  = GmemIterator<Tkv, ThrMap, BlockSeqLen, Swizzle, kStages>;
        using SmemQ  = SmemIterator<T, kHeadDim, Swizzle>;
        using SmemK  = SmemIterator<T, kHeadDim, Swizzle>;
        using SmemV  = SmemIterator<T, kHeadDim, Swizzle>;

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
        int iter = (history + min(query_idx_ + CTA_S, input) + CTA_S - 1) / CTA_S - 1;

        gmem_K.AdjustBlockTileIdx(iter);
        gmem_K.PrefetchStage(Int<0>{}, std::true_type{}, context - iter * CTA_S);
        CpAsyncCommit();

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
            gmem_V.PrefetchStage(Int<1>{}, is_residue, is_residue ? context - offset_K : CTA_S);
            CpAsyncCommit();

            ComputeQK<is_residue>(smem_Q, smem_K, frag_Q, frag_S, offset_K);

            CpAsyncWait();
            __syncthreads();  // Wait while trans_K being used & smem_K being filled

            // Prefetch for next K, gmem_K[i - 1] -> smem_K
            if (iter > 0) {
                gmem_K.AdjustBlockTileIdx(iter - 1);
                gmem_K.PrefetchStage(Int<0>{}, std::false_type{}, CTA_S);
                CpAsyncCommit();
            }

            if constexpr (is_mask) {
                ApplyCasualMask(frag_S, offset_Q, offset_K);
                // StoreS(frag_S, offset_K);
            }

            __align__(16) FragP frag_P;

            Softmax<is_residue>(frag_S, frag_M, frag_L, (FragPs&)frag_P, frag_O, qk_scale);
            // StoreP(frag_P, offset_K);

            ComputePV(smem_V, frag_P, frag_O);
        };

        int mask_iter = 2;

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
        __pipeline_wait_prior(0);
    }

    __device__ void CpAsyncCommit()
    {
        __pipeline_commit();
    }

    __device__ void CpAsyncFlush()
    {
        __pipeline_commit();
        __pipeline_wait_prior(0);
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
__global__ void __launch_bounds__(128, 2) attention_kernel(ParamType params)
{
    MHAType{params, dynamic_smem}.Run();
}

template<typename MHAType, typename ParamType = typename MHAType::ParamType>
__global__ void attention_reduction_kernel(ParamType params)
{
    MHAType::Reduce(params);
}

}  // namespace turbomind
