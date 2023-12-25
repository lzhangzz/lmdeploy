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

namespace turbomind {

template<class T, class Tkv, class BlockSeqLen, int CTA_Q, int CTA_S, int HeadDim, int Stages>
struct Attention {
    using ParamType = AttentionParams<T>;

    static constexpr int kWarpCount = 4;
    static constexpr int kHeadDim   = HeadDim;
    static constexpr int kStages    = Stages;

    static_assert(kStages == 2);

    struct SharedStorage {
        T smem_Q[CTA_Q][kHeadDim + SMEM_PAD];

        Tkv smem_K[CTA_S][kHeadDim + SMEM_PAD];
        Tkv smem_V[CTA_S][kHeadDim + SMEM_PAD];

        T smem_trans_K[CTA_S][kHeadDim + SMEM_PAD];
        T smem_trans_V[CTA_S][kHeadDim + SMEM_PAD];

        Array<T, 2> param_KV[2][CTA_S];
    };

    const ParamType& params_;

    int query_idx_;
    int head_idx_;
    int batch_idx_;
    int warp_id_;
    int lane_id_;

    BlockSeqLen block_seqlen_;

    // int  kv_head_idx_;
    // bool is_gqa_leader_;

    // int timestep_;

    const Tkv** k_cache_ptrs_;
    // const Tkv** v_cache_ptrs_;

    Tkv* smem_K_;
    Tkv* smem_V_;
    T*   smem_trans_K_;
    T*   smem_trans_V_;
    T*   smem_Q_;
    T*   smem_R_;

    int max_context_len_{};

    Array<T, 2>* smem_param_K_;
    Array<T, 2>* smem_param_V_;
    // float* smem_O_;

    template<int I>
    using Int = std::integral_constant<int, I>;

    struct Swizzle {
        // __device__ int operator()(int index)
        // {
        //     // sssSSSDDDdddd
        //     // DDD ^= SSS
        //     constexpr int mask = 0x7 << 7;
        //     return index ^ ((index & mask) >> 3);
        // }

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

        smem_Q_       = (T*)&shared->smem_Q;
        smem_K_       = (Tkv*)&shared->smem_K;
        smem_V_       = (Tkv*)&shared->smem_V;
        smem_trans_K_ = (T*)&shared->smem_trans_K;
        smem_trans_V_ = (T*)&shared->smem_trans_V;
        smem_param_K_ = shared->param_KV[0];
        smem_param_V_ = shared->param_KV[1];

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

        // warp_id_ = __shfl_sync(uint32_t(-1), warp_id_, 0);

        // const int gqa_group_size = params.num_heads / params.num_kv_heads;
        // kv_head_idx_             = head_idx_ / gqa_group_size;
        // is_gqa_leader_           = head_idx_ % gqa_group_size == 0;

        k_cache_ptrs_ = (const Tkv**)params_.k_cache_block_ptrs + params_.cu_block_cnts[batch_idx_];
        // v_cache_ptrs_ = (const Tkv**)params_.v_cache_block_ptrs + params_.cu_block_cnts[batch_idx_];
    }

    static constexpr int OP_M = 16;
    static constexpr int OP_N = 8;
    static constexpr int OP_K = 16;

    static constexpr int WARP_S = CTA_S;
    static constexpr int WARP_Q = CTA_Q / kWarpCount;

    static constexpr int K_ITER_M = WARP_Q / OP_M;    //  16 / 16 = 1
    static constexpr int K_ITER_N = WARP_S / OP_N;    //  64 /  8 = 8
    static constexpr int K_ITER_K = kHeadDim / OP_K;  // 128 / 16 = 8
    static constexpr int V_ITER_M = WARP_Q / OP_M;    //  16 / 16 = 1
    static constexpr int V_ITER_N = kHeadDim / OP_N;  // 128 /  8 = 16 -> D16
    static constexpr int V_ITER_K = WARP_S / OP_K;    //  64 / 16 = 4  -> S4
    //

    static constexpr bool kUseSmemQ = false;

    using FragQ  = Array<T, 8>[K_ITER_K][K_ITER_M];      // ((q8, d4), (D8, Q1), (d2, q2, d2))
    using FragK  = Array<Tkv, 4>[K_ITER_K][K_ITER_N];    // ((s8, d4), (D8, S8), (d2, d2))
    using FragS  = Array<float, 4>[K_ITER_M][K_ITER_N];  // ((q8, s4), (Q1, S8), (q2, s2))
    using FragPs = Array<T, 4>[K_ITER_M][K_ITER_N];      // ((q8, s4), (Q1, S8), (q2, s2))
    using FragP  = Array<T, 8>[V_ITER_M][V_ITER_K];      // ((q8, s4), (Q1, S4), (s2, q2, s2))
    using FragV  = Array<Tkv, 4>[V_ITER_K][V_ITER_N];    // ((d8, s4), (S4, D16), (s2, s2))
    using FragO  = Array<float, 4>[V_ITER_M][V_ITER_N];  // ((q8, d4), (Q1, D16), (q2, d2))
    using FragM  = Array<float, 2>[V_ITER_M];            // ((q8, _4), Q1, q2) => fragS with all S dim reduced
    using FragL  = FragM;

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

        // Load from shared memory using LDSM, rearrange to m16n8k16 atom layout
        if constexpr (!kUseSmemQ) {
            uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_Q_);
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

        // const int lane_id       = threadIdx.x % WARP_SIZE;
        // const int group_id      = lane_id / 16;
        // const int group_lane_id = lane_id % 16;
        // PRAGMA_UNROLL
        // for (int k = 0; k < K_ITER_K; ++k) {
        //     PRAGMA_UNROLL
        //     for (int m = 0; m < K_ITER_M; ++m) {
        //         auto&     r   = (Array<uint32_t, 4>&)frag_Q[k][m];
        //         const int s   = m * 16 + group_lane_id % 8 + group_id * 8 + warp_id_ * WARP_Q;
        //         const int c   = k * 16 + group_lane_id / 8 * 8;
        //         const int idx = Swizzle{}(s * (kHeadDim + SMEM_PAD) + c);
        //         ldmatrix_m8n8_x4_b16(r[0], r[2], r[1], r[3], smem_int_ptr + sizeof(T) * idx);
        //     }
        // }

        // __syncthreads();

        // constexpr int THREADS = kWarpCount * WARP_SIZE;
        // PRAGMA_UNROLL
        // for (int k = 0; k < K_ITER_K; ++k) {
        //     PRAGMA_UNROLL
        //     for (int m = 0; m < K_ITER_M; ++m) {
        //         constexpr int kVecSize = 8;
        //         Store(&smem_Q_[(k * K_ITER_M * THREADS + m * THREADS + threadIdx.x) * kVecSize], frag_Q[k][m]);
        //     }
        // }
    }

    __device__ void PrecomputeRotaryEmbeddings(int offset_K)
    {
        //         dim: (128, 64)
        //      access: ( 2,  1)
        // warp thread: (32,  1)
        // warp access: (64,  1)
        //   warp iter: ( 2, 64)
        //       warps: ( 2,  2)
        //       iters: ( 1, 32)
        //   footprint: (64, 32)
        //       delta: ( 0,  1)

        constexpr int S = 32;

        const int warp_offset_c = warp_id_ % 2;
        const int warp_offset_s = warp_id_ / 2;

        const int warp_thread_offset_c = lane_id_;

        int       offset_c = 64 * warp_offset_c + warp_thread_offset_c * 2;
        const int offset_s = 32 * warp_offset_s;

        // instead of swizzling the data, we swizzle the computation index
        // offset_c = offset_c ^ ((offset_s & 0x7) << 3);

        const float base     = 10000.f;
        const float offset_S = offset_K + offset_s;
        const float inv_freq = fdividef(1.f, powf(base, (float)offset_c / kHeadDim));

        Swizzle swizzle;

        PRAGMA_UNROLL
        for (int s = 0; s < S; ++s) {
            Array<float, 2> cs;
            sincosf((offset_S + (float)s) * inv_freq, &cs[1], &cs[0]);
            Store(&smem_R_[swizzle((offset_s + s) * kHeadDim + offset_c)], cast<T>(cs));
        }
    }

    using TransformedK = Array<T, 4>[K_ITER_K][K_ITER_N];  // ((s8, d4), (D8, S8), (d2, d2))

    __device__ void
    TransformK(TransformedK& transformed_K, FragK& frag_K, const Array<T, 2> (&param_K)[K_ITER_N], int k)
    {
        dequantize_K(transformed_K[k], frag_K[k], param_K);
    }

    template<bool is_residue, class SmemQ, class SmemK>
    __device__ void ComputeQK(SmemQ& smem_Q, SmemK& smem_K, FragQ& frag_Q, FragS& frag_S, int offset_K)
    {
        TransformedK frag_K;

        smem_K.LoadK(frag_K[0], 0);
        if constexpr (kUseSmemQ) {
            smem_Q.LoadQ(frag_Q[0], 0);
        }

        PRAGMA_UNROLL
        for (int k = 0; k < K_ITER_K; ++k) {
            if (k < K_ITER_K - 1) {
                smem_K.LoadK(frag_K[k + 1], k + 1);
                if constexpr (kUseSmemQ) {
                    smem_Q.LoadQ(frag_Q[k + 1], k + 1);
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

    __device__ void ApplyCasualMask(FragS& frag_S, int offset_Q, int offset_K)
    {
        PRAGMA_UNROLL
        for (int m = 0; m < K_ITER_M; ++m) {  // Q
            PRAGMA_UNROLL
            for (int n = 0; n < K_ITER_N; ++n) {  // K
                PRAGMA_UNROLL
                for (int q = 0; q < 2; ++q) {
                    PRAGMA_UNROLL
                    for (int s = 0; s < 2; ++s) {
                        const int qi = offset_Q + m * OP_M + lane_id_ / 4 + q * 8 + warp_id_ * WARP_Q;
                        const int ki = offset_K + n * OP_N + lane_id_ % 4 * 2 + s;
                        if (ki > qi) {
                            frag_S[m][n][q * 2 + s] = -std::numeric_limits<float>::infinity();
                        }
                    }
                }
            }
        }
    }

    __device__ void OutputQk(const FragS& frag_S, int offset_Q, int offset_K)
    {
        PRAGMA_UNROLL
        for (int m = 0; m < K_ITER_M; ++m) {  // Q
            PRAGMA_UNROLL
            for (int n = 0; n < K_ITER_N; ++n) {  // KV
                PRAGMA_UNROLL
                for (int q = 0; q < 2; ++q) {
                    PRAGMA_UNROLL
                    for (int s = 0; s < 2; ++s) {
                        const int qi = offset_Q + m * OP_M + lane_id_ / 4 + q * 8 + warp_id_ * WARP_Q;
                        const int ki = offset_K + n * OP_N + lane_id_ % 4 * 2 + s;
                        if (qi < params_.max_input_len && ki < max_context_len_) {
                            params_.qk[batch_idx_ * params_.num_heads * params_.max_input_len * max_context_len_
                                       + head_idx_ * params_.max_input_len * max_context_len_ + qi * max_context_len_
                                       + ki] = frag_S[m][n][q * 2 + s];
                        }
                    }
                }
            }
        }
    }

    template<bool is_residue>
    __device__ void Softmax(FragS& frag_S, FragM& frag_M, FragM& frag_L, FragPs& frag_Ps, FragO& frag_O, float qk_scale)
    {
        FragM prev_M;
        PRAGMA_UNROLL
        for (int m = 0; m < K_ITER_M; ++m) {
            prev_M[m] = frag_M[m];
        }

        // maximum
        PRAGMA_UNROLL
        for (int m = 0; m < K_ITER_M; ++m) {  // Q
            auto& row_M = frag_M[m];
            PRAGMA_UNROLL
            for (int n = 0; n < K_ITER_N; ++n) {  // KV
                auto& C = frag_S[m][n];
                PRAGMA_UNROLL
                for (int q = 0; q < 2; ++q) {
                    row_M[q] = fmaxf(row_M[q], fmaxf(C[q * 2 + 0], C[q * 2 + 1]));  // reduce over local pair
                }
            }
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {  // reduce over thread group within warp (within warp tiles)
                row_M[q] = fmaxf(row_M[q], __shfl_xor_sync(uint32_t(-1), row_M[q], 1));
                row_M[q] = fmaxf(row_M[q], __shfl_xor_sync(uint32_t(-1), row_M[q], 2));
            }
        }

        PRAGMA_UNROLL
        for (int m = 0; m < K_ITER_M; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                // exp(M - M'), isinf(frag_M) => isnan(expdiff_M)
                float expdiff_M = exp2f((prev_M[m][q] - frag_M[m][q]) * qk_scale);
                if (is_residue && frag_M[m][q] == -std::numeric_limits<float>::infinity()) {
                    expdiff_M = 0.f;
                }
                for (int n = 0; n < V_ITER_N; ++n) {
                    for (int d = 0; d < 2; ++d) {
                        frag_O[m][n][q * 2 + d] = frag_O[m][n][q * 2 + d] * expdiff_M;  // Rescale previous output
                    }
                }
                frag_L[m][q] *= expdiff_M;
            }
        }

        PRAGMA_UNROLL
        for (int m = 0; m < K_ITER_M; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                float tmp_L{};
                PRAGMA_UNROLL
                for (int n = 0; n < K_ITER_N; ++n) {
                    PRAGMA_UNROLL
                    for (int s = 0; s < 2; ++s) {
                        // unnormalized prob, optimized to FFMA
                        float p = exp2f(frag_S[m][n][q * 2 + s] * qk_scale - frag_M[m][q] * qk_scale);
                        if (is_residue && frag_M[m][q] == -std::numeric_limits<float>::infinity()) {
                            p = 0.f;
                        }
                        tmp_L += p;
                        frag_S[m][n][q * 2 + s] = p;
                    }
                }
                tmp_L += __shfl_xor_sync(uint32_t(-1), tmp_L, 1);
                tmp_L += __shfl_xor_sync(uint32_t(-1), tmp_L, 2);
                frag_L[m][q] = frag_L[m][q] + tmp_L;  // update L
            }
        }

        PRAGMA_UNROLL
        for (int m = 0; m < K_ITER_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < K_ITER_N; ++n) {
                PRAGMA_UNROLL
                for (int q = 0; q < 2; ++q) {
                    PRAGMA_UNROLL
                    for (int s = 0; s < 2; ++s) {
                        frag_Ps[m][n][q * 2 + s] = static_cast<T>(frag_S[m][n][q * 2 + s]);
                    }
                }
            }
        }
    }

    using TransformedV = Array<T, 4>[V_ITER_K][V_ITER_N];  // ((d8, s4), (S4, D16), (s2, s2))

    __device__ void TransformV(TransformedV& transformed_V, FragV& frag_V, const Array<T, 2> (&param_V)[4], int k)
    {
        dequantize_V(transformed_V[k], frag_V[k], param_V);
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

    __device__ void StoreO(FragO& frag_O, const FragL& frag_L)
    {
        FragL tmp_L;
        PRAGMA_UNROLL
        for (int m = 0; m < V_ITER_M; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                tmp_L[m][q] = fdividef(1.f, frag_L[m][q] + 1e-8f);
            }
        }

        const int qi_beg   = params_.cu_seqlens[batch_idx_] + query_idx_;  // global offset into `cu_seqlens`
        const int qi_end   = params_.cu_seqlens[batch_idx_ + 1];
        const int offset_Q = qi_beg + warp_id_ * WARP_Q;

        PRAGMA_UNROLL
        for (int m = 0; m < V_ITER_M; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                const int qi = offset_Q + m * OP_M + q * 8 + lane_id_ / 4;
                PRAGMA_UNROLL
                for (int n = 0; n < V_ITER_N; ++n) {
                    Array<T, 2> tmp_O;
                    PRAGMA_UNROLL
                    for (int d = 0; d < 2; ++d) {
                        tmp_O[d] = frag_O[m][n][q * 2 + d] * tmp_L[m][q];
                    }
                    if (qi < qi_end) {
                        const int di = n * 8 + lane_id_ % 4 * 2;
                        // [(b, s), h, d]
                        Store(&params_.out[qi * params_.num_heads * kHeadDim + head_idx_ * kHeadDim + di], tmp_O);
                    }
                }
            }
        }
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

    template<class MaxS>
    __device__ void LoadQuantParamsK(int offset_K, MaxS max_s)
    {
        auto quant_data = params_.kv_cache_quant_data                                     //
                          + batch_idx_ * params_.num_kv_heads * 2 * max_context_len_ * 2  //
                          + head_idx_ * 2 * max_context_len_ * 2                          //
                          + offset_K * 2;                                                 //

        if (threadIdx.x < max_s && threadIdx.x < CTA_S) {
            smem_param_K_[threadIdx.x] = (Array<T, 2>&)quant_data[threadIdx.x * 2];
        }
    }

    template<class MaxS>
    __device__ void LoadQuantParamsV(int offset_K, MaxS max_s)
    {
        auto quant_data = params_.kv_cache_quant_data                                     //
                          + batch_idx_ * params_.num_kv_heads * 2 * max_context_len_ * 2  //
                          + head_idx_ * 2 * max_context_len_ * 2                          //
                          + offset_K * 2;                                                 //
        quant_data += max_context_len_ * 2;

        if (threadIdx.x < max_s && threadIdx.x < CTA_S) {
            smem_param_V_[threadIdx.x] = (Array<T, 2>&)quant_data[threadIdx.x * 2];
        }
    }

    // k0v0k1v1k2v2
    //   K0V0K1V1K2
    //     Q0P0Q1P1

    template<class Map>
    __device__ void Dequantize(T* smem_trans, Tkv* smem, const Array<T, 2>* param)
    {
        const int2    offset = Map::get_offset(threadIdx.x / WARP_SIZE, threadIdx.x % WARP_SIZE);
        constexpr int N      = Map::kAccessC;
        PRAGMA_UNROLL
        for (int s = 0; s < Map::kIterS; ++s) {
            const int si = offset.y + s * Map::kDeltaS;
            PRAGMA_UNROLL
            for (int c = 0; c < Map::kIterC; ++c) {
                const int ci  = offset.x + c * Map::kDeltaC;
                // const int idx = Swizzle{}(si * (Map::kDimC + SMEM_PAD) + ci);
                const int idx = si * (Map::kDimC + SMEM_PAD) + ci;
                dequantize((Array<T, N>(&)[1][1])smem_trans[Swizzle{}(idx)],
                           (Array<uint8_t, N>(&)[1][1])smem[idx],
                           (Array<T, 2>(&)[1])param[si],
                           std::integral_constant<int, 8>{});
            }
        }
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
        using GmemK  = GmemIterator<Tkv, ThrMap, BlockSeqLen, Identity, kStages>;
        using GmemV  = GmemIterator<Tkv, ThrMap, BlockSeqLen, Identity, kStages>;
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

        SmemQ smem_Q{smem_Q_};
        SmemK smem_K{smem_trans_K_};
        SmemV smem_V{smem_trans_V_};
        // SmemIteratorQ<T, kHeadDim, Swizzle> smem_Q{smem_Q_};
        // SmemIteratorK<Tkv, kHeadDim, Swizzle> smem_K{smem_KV_};
        // SmemIteratorV<Tkv, kHeadDim, Swizzle> smem_V{smem_KV_ + CTA_S * (kHeadDim + SMEM_PAD)};
        // SmemIteratorK<Tkv, kHeadDim, Swizzle> smem_R{smem_R_};

        const int input   = params_.input_length[batch_idx_];
        const int context = params_.context_length[batch_idx_];
        const int history = context - input;

        const int offset_Q = history + query_idx_;

        // ceil(tiles) - 1
        int iter = (history + min(query_idx_ + CTA_S, input) + CTA_S - 1) / CTA_S - 1;

        LoadQuantParamsK(iter * CTA_S, context - iter * CTA_S);
        gmem_K.AdjustBlockTileIdx(iter);
        gmem_K.PrefetchStage(Int<0>{}, std::true_type{}, context - iter * CTA_S);
        CpAsyncCommit();

        LoadQuantParamsV(iter * CTA_S, context - iter * CTA_S);
        gmem_V.AdjustBlockTileIdx(iter);
        gmem_V.PrefetchStage(Int<0>{}, std::true_type{}, context - iter * CTA_S);
        CpAsyncCommit();

        CpAsyncWait();
        __syncthreads();

        Dequantize<ThrMap>(smem_trans_K_, smem_K_, smem_param_K_);

        __align__(16) FragO frag_O{};

        FragL frag_L{};
        FragM frag_M;
        fill(frag_M, -std::numeric_limits<float>::infinity());

        auto loop = [&](auto is_residue, auto is_mask) {
            const int offset_K = iter * CTA_S;

            __align__(16) FragS frag_S{};

            CpAsyncWait();
            __syncthreads();  // Wait while smem_V being filled & trans_V being used

            // Prefetch for next K, gmem_K[i-1] -> smem_K
            if (iter > 0) {
                LoadQuantParamsK(offset_K - CTA_S, CTA_S);
                gmem_K.AdjustBlockTileIdx(iter - 1);
                gmem_K.PrefetchStage(Int<1>{}, std::false_type{}, CTA_S);
                CpAsyncCommit();
            }

            Dequantize<ThrMap>(smem_trans_V_, smem_V_, smem_param_V_);  // (smem_V, param_V) -> smem_trans_V

            ComputeQK<is_residue>(smem_Q, smem_K, frag_Q, frag_S, offset_K);

            // if (params_.qk) {
            //     OutputQk(frag_S, offset_Q, offset_K);
            // }

            CpAsyncWait();
            __syncthreads();  // Wait while trans_K being used & smem_K being filled

            // Prefetch for next V, gmem_V[i - 1] -> smem_V
            if (iter > 0) {
                LoadQuantParamsV(offset_K - CTA_S, CTA_S);
                gmem_V.AdjustBlockTileIdx(iter - 1);
                gmem_V.PrefetchStage(Int<0>{}, std::false_type{}, CTA_S);
                CpAsyncCommit();
            }

            if constexpr (is_mask) {
                ApplyCasualMask(frag_S, offset_Q, offset_K);
            }

            const float         qk_scale = params_.inv_sqrt_dh;
            __align__(16) FragP frag_P;
            Softmax<is_residue>(frag_S, frag_M, frag_L, (FragPs&)frag_P, frag_O, qk_scale);

            Dequantize<ThrMap>(smem_trans_K_, smem_K_, smem_param_K_);  // (smem_K, param_K) -> smem_trans_K

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

    // __device__ void
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
        const int qi = offset.y + s * Map::kDeltaS + token_idx;  // local offset into `input_length`
        const int ti = history_len + qi;

        if (qi < input_len) {
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
                const int di = offset.x + c * Map::kDeltaC;
                Stcs(&k_cache[di], out_K[s][c]);
                Stcs(&v_cache[di], out_V[s][c]);
            }

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
