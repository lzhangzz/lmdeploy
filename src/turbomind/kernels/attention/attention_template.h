// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "array_ops.h"
#include "iterator.h"
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

template<typename T, typename Tkv, int CTA_Q, int CTA_S, int HeadDim, int Stages>
struct Attention {
    using ParamType = AttentionParams<T>;

    static constexpr int kWarpCount = 4;
    static constexpr int kHeadDim   = HeadDim;
    static constexpr int kStages    = Stages;

    static_assert(kStages == 2);

    struct SharedStorage {
        T   smem_Q[CTA_Q][kHeadDim + SMEM_PAD];
        Tkv smem_KV[2][CTA_S][kHeadDim + SMEM_PAD];
    };

    const ParamType& params_;

    int query_idx_;
    int head_idx_;
    int batch_idx_;
    int warp_id_;
    int lane_id_;

    // int  kv_head_idx_;
    // bool is_gqa_leader_;

    // int timestep_;

    const Tkv** k_cache_ptrs_;
    // const Tkv** v_cache_ptrs_;

    Tkv* smem_KV_;
    T*   smem_Q_;
    // float* smem_O_;

    template<int I>
    using Int = std::integral_constant<int, I>;

    struct Swizzle {
        __device__ int operator()(int index)
        {
            return index;

            // sssSSSdDDDddd
            // DDD ^= SSS
            constexpr int mask = 0x7f << 7;
            return index ^ ((index & mask) >> 4);
        }
    };

    __device__ bool thread0()
    {
        return blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0;
    }

    __device__ Attention(const ParamType& params, uint8_t* dsmem): params_(params)
    {
        SharedStorage* shared = (SharedStorage*)dsmem;

        smem_Q_  = (T*)&shared->smem_Q;
        smem_KV_ = (Tkv*)&shared->smem_KV;

        // [q, h, b]
        query_idx_ = blockIdx.x * CTA_Q;  // local offset into `input_length`
        head_idx_  = blockIdx.y;
        batch_idx_ = blockIdx.z;

        // if (query_idx_ >= params.input_length[batch_idx_]) {
        //     return;
        // }

        warp_id_ = threadIdx.x / WARP_SIZE;
        lane_id_ = threadIdx.x % WARP_SIZE;

        // const int gqa_group_size = params.num_heads / params.num_kv_heads;
        // kv_head_idx_             = head_idx_ / gqa_group_size;
        // is_gqa_leader_           = head_idx_ % gqa_group_size == 0;

        // timestep_ = params_.context_length[batch_idx_] - 1;

        k_cache_ptrs_ = (const Tkv**)params_.k_cache_block_ptrs + params_.cu_block_cnts[batch_idx_];
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
    static constexpr int V_ITER_N = kHeadDim / OP_N;  // 128 /  8 = 16
    static constexpr int V_ITER_K = WARP_S / OP_K;    //  64 / 16 = 4
                                                      //
    // using FragQ = Array<T, 8>[K_ITER_K][K_ITER_M];      // ((q8, d4), (D8, Q1), (d2, q2, d2))
    using FragQ = Array<T, 8>[K_ITER_M];                // ((q8, d4), (D8, Q1), (d2, q2, d2))
    using FragK = Array<T, 4>[K_ITER_N];                // ((s8, d4), (D8, S8), (d2, d2))
    using FragS = Array<float, 4>[K_ITER_N][K_ITER_M];  // ((q8, s4), (S8, Q1), (q2, s2))
                                                        //
    using FragP = Array<T, 8>[V_ITER_K][V_ITER_M];      // ((q8, s4), (S4, Q1), (s2, q2, s2))
    using FragV = Array<T, 4>[V_ITER_N];                // ((d8, s4), (S4, D16), (s2, s2))
    using FragO = Array<float, 4>[V_ITER_N][V_ITER_M];  // ((q8, d4), (D16, Q1), (q2, d2))
                                                        //
    using FragM = Array<float, 2>[K_ITER_M];            // ((q8, _4), Q1, q2) => fragS with all S dim reduced
    using FragL = FragM;

    using SmemO = Array<float, 4>[V_ITER_N][V_ITER_M][kWarpCount][WARP_SIZE];  // ((D16, Q1, Q4, (q8, d4)), (q2, d2))

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

        // Apply rotary embedding
        // float rope_theta = params_.rope_theta ? params_.rope_theta[batch_idx_] : params_.rotary_embedding_base;
        // PRAGMA_UNROLL
        // for (int c = 0; c < ITER_C; ++c) {
        //     FastRoPE rope(offset.x + c * Map::kDeltaC, Int<kHeadDim>{}, rope_theta, Int<kVecSize>{});
        //     PRAGMA_UNROLL
        //     for (int s = 0; s < ITER_S; ++s) {
        //         rope.apply(vec_Q[s][c], timestep_ + query_idx_ + offset.y + s * Map::kDeltaS);
        //     }
        // }

        Swizzle swizzle;

        // Store to shared memory
        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            const int qi = offset.y + s * Map::kDeltaS;
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                const int di = offset.x + c * Map::kDeltaC;
                Store(&smem_Q_[swizzle(qi * (kHeadDim + SMEM_PAD) + di)], vec_Q[s][c]);
            }
        }

        // __syncwarp();
        __syncthreads();

        // auto get_qi = [&](int m, int q) { return m * OP_M + lane_id_ / 4 * 1 + q * 8 + warp_id_ * WARP_Q; };
        // auto get_di = [&](int k, int d1, int d0) { return k * 16 + d1 * 8 + lane_id_ * 2 + d0; };
        // Load from shared memory using LDSM, rearrange to m16n8k16 atom layout
        // uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_Q_);
        // PRAGMA_UNROLL
        // for (int m = 0; m < K_ITER_M; ++m) {
        //     PRAGMA_UNROLL
        //     for (int k = 0; k < K_ITER_K; ++k) {
        //         auto&     Q   = (Array<uint32_t, 4>&)frag_Q[k][m];
        //         const int mm  = m * 16 + lane_id_ % 16 + warp_id_ * WARP_Q;
        //         const int kk  = k * 16 + lane_id_ / 16 * 8;
        //         const int idx = swizzle(mm * kHeadDim + kk);
        //         ldmatrix_m8n8_x4_b16(Q[0], Q[1], Q[2], Q[3], smem_int_ptr + sizeof(T) * idx);
        //     }
        // }
    }

    template<class SmemQ, class SmemK>
    __device__ void ComputeQK(SmemQ& smem_Q, SmemK& smem_K, const FragQ& _frag_Q, FragS& frag_S)
    {
        // FragQ frag_Q_buf[2];
        // FragK frag_K_buf[2];
        FragQ frag_Q_buf[K_ITER_K];
        FragK frag_K_buf[K_ITER_K];
        smem_K.LoadK(frag_K_buf[0], 0);
        smem_Q.LoadQ(frag_Q_buf[0], 0);
        PRAGMA_UNROLL
        for (int k = 0; k < K_ITER_K; ++k) {  //  reuse `inv_freqs` for dims
            if (k < K_ITER_K - 1) {
                smem_K.LoadK(frag_K_buf[k + 1], k + 1);
                smem_Q.LoadQ(frag_Q_buf[k + 1], k + 1);
            }
            auto& frag_K = frag_K_buf[k];
            auto& frag_Q = frag_Q_buf[k];
            PRAGMA_UNROLL
            for (int n = 0; n < K_ITER_N; ++n) {  // reuse rope coeff for steps
                PRAGMA_UNROLL
                for (int m = 0; m < K_ITER_M; ++m) {
                    mma_m16n8k16_row_col(frag_S[n][m], frag_Q[m], frag_K[n], frag_S[n][m]);
                }
            }
        }
    }

    __device__ void ApplyCasualMask(FragS& frag_S, int offset_Q, int offset_K, int input_len, int context_len)
    {
        PRAGMA_UNROLL
        for (int m = 0; m < K_ITER_M; ++m) {  // Q
            PRAGMA_UNROLL
            for (int n = 0; n < K_ITER_N; ++n) {  // K
                PRAGMA_UNROLL
                for (int q = 0; q < 2; ++q) {
                    PRAGMA_UNROLL
                    for (int s = 0; s < 2; ++s) {
                        const int qi = offset_Q + m * OP_M + lane_id_ / 4 * 1 + q * 8 + warp_id_ * WARP_Q;
                        const int ki = offset_K + n * OP_N + lane_id_ % 4 * 2 + s * 1;
                        if (ki > qi) {
                            frag_S[n][m][q * 2 + s] = -std::numeric_limits<float>::infinity();
                        }
                    }
                }
            }
        }
    }

    template<bool is_residue>
    __device__ void
    Softmax(FragS& frag_S, FragM& frag_M, FragM& expdiff_M, FragM& frag_L, FragP& frag_P, int offset_Q, int offset_K)
    {
        FragM tmp_M;
        PRAGMA_UNROLL
        for (int m = 0; m < K_ITER_M; ++m) {
            tmp_M[m] = frag_M[m];
        }

        auto get_qi = [&](int m, int q) { return m * OP_M + lane_id_ / 4 * 1 + q * 8 + warp_id_ * WARP_Q; };
        auto get_ki = [&](int n, int s) { return n * OP_N + lane_id_ % 4 * 2 + s * 1 + offset_K; };

        // maximum
        PRAGMA_UNROLL
        for (int m = 0; m < K_ITER_M; ++m) {  // Q
            auto& row_M = tmp_M[m];
            PRAGMA_UNROLL
            for (int n = 0; n < K_ITER_N; ++n) {  // KV
                auto& C = frag_S[n][m];
                PRAGMA_UNROLL
                for (int q = 0; q < 2; ++q) {
                    C[q * 2 + 0] *= params_.inv_sqrt_dh;
                    C[q * 2 + 1] *= params_.inv_sqrt_dh;
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
                // isinf(frag_M) => isnan(expdiff_M)
                expdiff_M[m][q] = exp2f(frag_M[m][q] - tmp_M[m][q]);  // exp(M - M')

                if constexpr (is_residue) {
                    if (tmp_M[m][q] == -std::numeric_limits<float>::infinity()) {
                        expdiff_M[m][q] = 0.f;
                    }
                }

                frag_M[m][q] = tmp_M[m][q];  // update M
            }
        }

        FragL tmp_L{};
        PRAGMA_UNROLL
        for (int m = 0; m < K_ITER_M; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                PRAGMA_UNROLL
                for (int n = 0; n < K_ITER_N; ++n) {
                    PRAGMA_UNROLL
                    for (int s = 0; s < 2; ++s) {
                        float p = exp2f(frag_S[n][m][q * 2 + s] - frag_M[m][q]);  // unnormalized prob
                        if constexpr (is_residue) {
                            if (frag_M[m][q] == -std::numeric_limits<float>::infinity()) {
                                p = 0.f;
                            }
                        }
                        tmp_L[m][q] += p;
                        frag_P[n / 2][m][n % 2 * 4 + q * 2 + s] = static_cast<T>(p);
                    }
                }
                tmp_L[m][q] += __shfl_xor_sync(uint32_t(-1), tmp_L[m][q], 1);
                tmp_L[m][q] += __shfl_xor_sync(uint32_t(-1), tmp_L[m][q], 2);
                frag_L[m][q] = tmp_L[m][q] + expdiff_M[m][q] * frag_L[m][q];  // update L
            }
        }
    }

    __device__ int get_vi(int k, int s1, int s0)
    {
        return k * OP_K + lane_id_ % 4 * 2 + s1 * 8 + s0;
    }

    template<class Smem>
    __device__ void ComputePV(Smem& smem, const FragP& frag_P, const FragM& expdiff_M, FragO& frag_O)
    {
        PRAGMA_UNROLL
        for (int n = 0; n < V_ITER_N; ++n) {
            PRAGMA_UNROLL
            for (int m = 0; m < V_ITER_M; ++m) {
                PRAGMA_UNROLL
                for (int q = 0; q < 2; ++q) {
                    PRAGMA_UNROLL
                    for (int d = 0; d < 2; ++d) {
                        frag_O[n][m][q * 2 + d] = frag_O[n][m][q * 2 + d] * expdiff_M[m][q];  // Rescale previous output
                    }
                }
            }
        }

        FragV frag_V_buf[2];
        smem.LoadV(frag_V_buf[0], 0);
        PRAGMA_UNROLL
        for (int k = 0; k < V_ITER_K; ++k) {
            if (k < V_ITER_K - 1) {
                smem.LoadV(frag_V_buf[(k + 1) % 2], (k + 1) % V_ITER_K);
            }
            FragV& frag_V = frag_V_buf[k % 2];
            PRAGMA_UNROLL
            for (int n = 0; n < V_ITER_N; ++n) {
                PRAGMA_UNROLL
                for (int m = 0; m < V_ITER_M; ++m) {
                    mma_m16n8k16_row_col(frag_O[n][m], frag_P[k][m], frag_V[n], frag_O[n][m]);
                }
            }
        }
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
                        tmp_O[d] = frag_O[n][m][q * 2 + d] * tmp_L[m][q];
                    }
                    const int di = n * 8 + lane_id_ % 4 * 2;
                    if (qi < qi_end) {
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

    __device__ void Run()
    {
        // early exit if finished flag is set
        if (params_.finished[batch_idx_]) {
            return;
        }

        if (query_idx_ >= params_.input_length[batch_idx_]) {
            return;
        }

        using ThrMap = RakedThreadMap<kHeadDim, CTA_S, sizeof(uint4) / sizeof(Tkv), kWarpCount>;
        using Gmem   = GmemIterator<Tkv, ThrMap, Swizzle, kStages>;
        using SmemQ  = SmemIterator<Tkv, kHeadDim, Swizzle>;
        using SmemK  = SmemIterator<Tkv, kHeadDim, Swizzle>;
        using SmemV  = SmemIterator<Tkv, kHeadDim, Swizzle>;

        FragQ frag_Q;
        LoadQ(frag_Q);

        const int block_seqlen = params_.kv_cache_block_size;
        // [L, 2, H, s, D]
        const int local_key_offset = params_.key_offset + head_idx_ * block_seqlen * kHeadDim;
        const int local_val_offset = params_.val_offset + head_idx_ * block_seqlen * kHeadDim;

        Gmem gmem{k_cache_ptrs_, block_seqlen, {local_key_offset, local_val_offset}, smem_KV_, warp_id_, lane_id_};

        gmem.ClearSmem(Int<0>{});
        gmem.ClearSmem(Int<1>{});

        SmemQ smem_Q{smem_Q_};
        SmemK smem_K{smem_KV_};
        static_assert(CTA_S * (kHeadDim + SMEM_PAD) == Gmem::kSizePerTile);
        SmemV smem_V{smem_KV_ + CTA_S * (kHeadDim + SMEM_PAD)};

        const int input   = params_.input_length[batch_idx_];
        const int context = params_.context_length[batch_idx_];
        const int history = context - input;

        const int offset_Q = history + query_idx_;

        // ceil(tiles) - 1
        int iter = (history + min(query_idx_ + CTA_S, input) + CTA_S - 1) / CTA_S - 1;

        gmem.AdjustBlockTileIdx(iter);
        gmem.PrefetchStage(Int<0>{}, std::true_type{}, context - iter * CTA_S);
        CpAsyncCommit();  // commit for K

        FragO frag_O{};
        FragL frag_L{};
        FragM frag_M;
        fill(frag_M, -std::numeric_limits<float>::infinity());

        auto loop = [&](auto is_residue, auto is_mask) {
            const int offset_K = iter * CTA_S;

            FragS frag_S{};

            CpAsyncWait();
            __syncthreads();  // Wait while K being filled & V being used

            // Prefetch for V
            gmem.PrefetchStage(Int<1>{}, is_residue, is_residue ? context - offset_K : CTA_S);
            CpAsyncCommit();

            ComputeQK(smem_Q, smem_K, frag_Q, frag_S);

            CpAsyncWait();
            __syncthreads();  // Wait while K being used & V being filled

            // Prefetch K for next iter (always full tile here)
            if (iter > 0) {
                gmem.AdjustBlockTileIdx(iter - 1);
                gmem.PrefetchStage(Int<0>{}, std::false_type{}, CTA_S);
            }
            CpAsyncCommit();

            if constexpr (is_mask) {
                ApplyCasualMask(frag_S, offset_Q, offset_K, input, context);
            }

            FragP frag_P;
            FragM expdiff_M;
            Softmax<is_residue>(frag_S, frag_M, expdiff_M, frag_L, frag_P, offset_Q, offset_K);

            ComputePV(smem_V, frag_P, expdiff_M, frag_O);
        };

        PRAGMA_UNROLL
        for (int mask_iter = 2; iter >= 0 && mask_iter != 0; --iter, --mask_iter) {
            loop(std::true_type{}, std::true_type{});
        }

        for (; iter >= 0; --iter) {
            loop(std::false_type{}, std::false_type{});
        }

        StoreO(frag_O, frag_L);
    }
};

template<class T, class Tkv, int CTA_Q, int kHeadDim, int kWarpCount, class ParamType>
__global__ void ProcessKV(ParamType params)
{
    constexpr int kVecSize = sizeof(uint4) / sizeof(T);

    using Vec = Array<T, kVecSize>;
    using Map = RakedThreadMap<kHeadDim, CTA_Q, kVecSize, kWarpCount>;

    constexpr int ITER_C = Map::kIterC;
    constexpr int ITER_S = Map::kIterS;

    Vec vec_K[ITER_S][ITER_C];
    Vec vec_V[ITER_S][ITER_C];

    const int token_idx = blockIdx.x * CTA_Q;  // local offset into `input_length`
    const int head_idx  = blockIdx.y;
    const int batch_idx = blockIdx.z;

    const int qi_beg = params.cu_seqlens[batch_idx] + token_idx;  // global offset into `cu_seqlens`
    const int qi_end = params.cu_seqlens[batch_idx + 1];

    const int input_len   = params.input_length[batch_idx];
    const int history_len = params.context_length[batch_idx] - input_len;

    if (qi_beg >= qi_end) {
        return;
    }

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int2 offset = Map::get_offset(warp_id, lane_id);

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

    auto apply_bias = [&](const T* g_bias, Vec(&vec)[ITER_S][ITER_C]) {
        Vec bias[ITER_C];
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int di = offset.x + c * Map::kDeltaC;
            Ldg(bias[c], &g_bias[head_idx * kHeadDim + di]);
        }
        using namespace ops;
        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                vec[s][c] = vec[s][c] + bias[c];
            }
        }
    };

    if (params.k_bias) {
        apply_bias(params.k_bias, vec_K);
    }
    if (params.v_bias) {
        apply_bias(params.v_bias, vec_V);
    }

    Tkv** k_cache_block_ptrs = (Tkv**)params.k_cache_block_ptrs + params.cu_block_cnts[batch_idx];

    PRAGMA_UNROLL
    for (int s = 0; s < ITER_S; ++s) {
        const int qi = offset.y + s * Map::kDeltaS + token_idx;  // local offset into `input_length`
        const int ti = history_len + qi;
        // block index and local offsets
        const int cache_block_index  = ti / params.kv_cache_block_size;
        const int cache_block_offset = ti % params.kv_cache_block_size;
        // [H, s, D]
        Tkv* k_cache = k_cache_block_ptrs[cache_block_index] + params.key_offset
                       + head_idx * params.kv_cache_block_size * kHeadDim + cache_block_offset * kHeadDim;
        Tkv* v_cache = k_cache_block_ptrs[cache_block_index] + params.val_offset
                       + head_idx * params.kv_cache_block_size * kHeadDim + cache_block_offset * kHeadDim;
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int di = offset.x + c * Map::kDeltaC;
            if (qi < input_len) {
                Store(&k_cache[di], vec_K[s][c]);
                Store(&v_cache[di], vec_V[s][c]);
            }
        }
    }
}

extern __shared__ uint8_t dynamic_smem[];

template<typename MHAType, typename ParamType = typename MHAType::ParamType>
__global__ void attention_kernel(ParamType params)
{
    MHAType{params, dynamic_smem}.Run();
}

template<typename MHAType, typename ParamType = typename MHAType::ParamType>
__global__ void attention_reduction_kernel(ParamType params)
{
    MHAType::Reduce(params);
}

}  // namespace turbomind
