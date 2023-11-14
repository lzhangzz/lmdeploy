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
#include <type_traits>

#include "decoder_multihead_attention_params.h"

namespace turbomind {

template<typename T,
         typename Tkv,
         int  HeadPerCta,
         int  MaxHeadDim,
         int  KeyPerIter,
         int  HeadDim,
         int  SliceLen,
         int  Stages,
         bool SplitK>
struct DecoderMultiHeadAttentionKernel {
    using ParamType = DecoderMultiHeadAttentionParams<T>;

    static constexpr int  kWarpCount  = 4;
    static constexpr int  kHeadPerCta = HeadPerCta;
    static constexpr int  kMaxHeadDim = MaxHeadDim;
    static constexpr int  kKeyPerIter = KeyPerIter;
    static constexpr int  kHeadDim    = HeadDim;
    static constexpr int  kStages     = Stages;
    static constexpr bool kSplitK     = SplitK;

    static constexpr int kSliceLen     = SliceLen;
    static constexpr int kIterPerSlice = kSliceLen / kKeyPerIter;

    static constexpr int kVecKvSize = 16 / sizeof(Tkv);
    // static constexpr int kThreadPerKey = 8;

    using VecKv      = Array<T, kVecKvSize>;
    using VecKvFloat = Array<float, kVecKvSize>;

    using GmemMap  = RakedThreadMap<kHeadDim, kKeyPerIter, kVecKvSize, kWarpCount>;
    using GmemIter = GmemIterator<Tkv, GmemMap, kStages>;

    using SmemMap  = RakedThreadMap<kHeadDim, kKeyPerIter, kVecKvSize, kWarpCount, 8>;
    using SmemIter = SmemIterator<Tkv, SmemMap, kStages>;

    static constexpr size_t GetDynamicSmemSize()
    {
        size_t smem_kv_cache = GmemIter::kSmemByteSize;
        size_t smem_qk       = sizeof(float) * kHeadPerCta * kSliceLen;
        size_t smem_pr       = sizeof(float) * kHeadPerCta * kSliceLen;
        size_t smem_kv_align = 0;
        return smem_kv_cache + std::max(smem_qk, smem_pr) + smem_kv_align;
    }

    using QkAccumType   = float;
    using QkComputeType = float;

    using PvAccumType   = float;
    using PvComputeType = float;

    struct SharedStorage {
        __align__(16) T Q[kHeadPerCta * kMaxHeadDim];
        __align__(16) float O[kHeadPerCta * kMaxHeadDim];
        float M[kHeadPerCta];  // max{dot(Q,  K^T  )}
        float L[kHeadPerCta];  // sum{exp(s - S_max)}
        float red_max[kHeadPerCta * kWarpCount];
        float red_sum[kHeadPerCta * kWarpCount];
    };

    const ParamType& params_;

    int head_idx_;
    int batch_idx_;
    int warp_id_;
    int lane_id_;

    int  kv_head_idx_;
    bool is_gqa_leader_;

    int step_begin_;
    int step_end_;

    int timestep_;
    Tkv* __restrict__ k_cache_;  // [S, D]
    Tkv* __restrict__ v_cache_;  // [S, D]

    const Tkv** __restrict__ k_cache_ptrs_;
    const Tkv** __restrict__ v_cache_ptrs_;

    Tkv* __restrict__ smem_Kv_;
    float* __restrict__ smem_S_;
    float* __restrict__ smem_P_;
    T* __restrict__ smem_Q_;
    float* __restrict__ smem_M_;
    float* __restrict__ smem_L_;
    float* __restrict__ smem_O_;
    float* __restrict__ smem_red_max_;
    float* __restrict__ smem_red_sum_;

    // avoid redundant type cast for KV8
    using KLoadType = std::conditional_t<std::is_same_v<Tkv, int8_t>, float, T>;
    using VLoadType = std::conditional_t<std::is_same_v<Tkv, int8_t>, float, T>;

    ConvertKvCache<T, Tkv>         conv_k_store_;
    ConvertKvCache<T, Tkv>         conv_v_store_;
    ConvertKvCache<Tkv, KLoadType> conv_k_;
    ConvertKvCache<Tkv, VLoadType> conv_v_;

    __device__ bool thread0()
    {
        return blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0;
    }

    __device__ DecoderMultiHeadAttentionKernel(const ParamType& params, SharedStorage& smem, uint8_t* dsmem):
        params_(params),
        conv_k_store_{params_.kv_quant_params[0], params_.kv_quant_params[1]},
        conv_v_store_{params_.kv_quant_params[2], params_.kv_quant_params[3]},
        conv_k_{params_.kv_quant_params[0], params_.kv_quant_params[1]},
        conv_v_{params_.kv_quant_params[2], params_.kv_quant_params[3]}
    {
        smem_Kv_      = (Tkv*)dsmem;
        smem_S_       = (float*)(smem_Kv_ + GmemIter::kSizePerTile * kStages);  // [HeadPerCta * kSliceLen]
        smem_P_       = smem_S_;  // ! reusing only works when S and P has same dtype
        smem_Q_       = smem.Q;
        smem_M_       = smem.M;
        smem_L_       = smem.L;
        smem_O_       = smem.O;
        smem_red_max_ = smem.red_max;
        smem_red_sum_ = smem.red_sum;

        head_idx_  = get_head_idx() * kHeadPerCta;
        batch_idx_ = get_batch_idx();
        warp_id_   = threadIdx.x / WARP_SIZE;
        lane_id_   = threadIdx.x % WARP_SIZE;

        const int gqa_group_size = params.num_heads / params.num_kv_heads;
        kv_head_idx_             = head_idx_ / gqa_group_size;
        is_gqa_leader_           = head_idx_ % gqa_group_size == 0;

        timestep_ = params_.per_sample_length[batch_idx_];

        if (kSplitK && params.max_split_k > 1) {
            const int slice_count     = (timestep_ + kSliceLen - 1) / kSliceLen;
            const int slice_per_split = (slice_count + params_.max_split_k - 1) / params_.max_split_k;

            step_begin_ = slice_per_split * get_split_k_idx() * kSliceLen;
            step_end_   = min(timestep_, step_begin_ + slice_per_split * kSliceLen);
        }
        else {
            step_begin_ = 0;
            step_end_   = timestep_;
        }

        k_cache_ptrs_ = (const Tkv**)params_.k_cache_block_ptrs + params_.cu_block_cnts[batch_idx_];
        v_cache_ptrs_ = (const Tkv**)params_.v_cache_block_ptrs + params_.cu_block_cnts[batch_idx_];
    }

    __device__ void Prolugue()
    {
        // - Each warp is handling a row of Q
        // - K/V are loaded redundantly only for the current step
        static_assert(kMaxHeadDim % WARP_SIZE == 0);
        static constexpr int kVecQSize = kMaxHeadDim / WARP_SIZE;

        using VecQ      = Array<T, kVecQSize>;
        using VecQFloat = Array<float, kVecQSize>;

        using MapQ = ThreadMapQ<kMaxHeadDim, kHeadPerCta, kVecQSize, kWarpCount>;

        static constexpr int kQVecPerThread  = MapQ::kIterC;
        static constexpr int kQHeadPerThread = MapQ::kIterS;  // > 1 when #warp < kCtaPerHead

        static_assert(kQVecPerThread == 1);

        int2 offset   = MapQ::get_offset(warp_id_, lane_id_);
        bool is_valid = offset.x < kMaxHeadDim && offset.y < kHeadPerCta;

        if (!is_valid) {
            return;
        }

        VecQ frag_Q[kQHeadPerThread];
        VecQ frag_K;
        VecQ frag_V;

        // load qkv
        PRAGMA_UNROLL
        for (int s = 0; s < kQHeadPerThread; ++s) {
            int di = offset.x;
            int qi = offset.y + s;
            Ldg(frag_Q[s], &params_.q[batch_idx_ * params_.stride + (head_idx_ + qi) * kHeadDim + di]);
        }
        Ldg(frag_K, &params_.k[batch_idx_ * params_.stride + kv_head_idx_ * kHeadDim + offset.x]);
        Ldg(frag_V, &params_.v[batch_idx_ * params_.stride + kv_head_idx_ * kHeadDim + offset.x]);

        if (params_.q_bias) {
            // load biases
            VecQ bias_Q[kQHeadPerThread];
            PRAGMA_UNROLL
            for (int s = 0; s < kQHeadPerThread; ++s) {
                int di = offset.x;
                int qi = offset.y + s;
                Ldg(bias_Q[s], &params_.q_bias[(head_idx_ + qi) * kHeadDim + di]);
            }
            VecQ bias_K;
            VecQ bias_V;
            Ldg(bias_K, &params_.k_bias[kv_head_idx_ * kHeadDim + offset.x]);
            Ldg(bias_V, &params_.v_bias[kv_head_idx_ * kHeadDim + offset.x]);

            using namespace ops;
            // apply biases
            PRAGMA_UNROLL
            for (int s = 0; s < kQHeadPerThread; ++s) {
                frag_Q[s] = frag_Q[s] + bias_Q[s];
            }
            frag_K = frag_K + bias_K;
            frag_V = frag_V + bias_V;
        }

        // for (int i = 0; i < kVecQSize; ++i) {
        //     printf("q[%2d][%3d] = %f\n", (int)head_idx_, (int)(offset.x + i), (float)frag_Q[0][i]);
        // }

        float rotary_embedding_base =
            params_.rope_theta ? params_.rope_theta[batch_idx_] : params_.rotary_embedding_base;

        // Apply rotary embedding
        RotaryEmbedding<kVecQSize> rotary_emb(rotary_embedding_base, params_.rotary_embedding_dim, timestep_, offset);

        PRAGMA_UNROLL
        for (int s = 0; s < kQHeadPerThread; ++s) {
            rotary_emb.apply(frag_Q[s]);
        }
        rotary_emb.apply(frag_K);

        if (params_.use_logn_attn) {
            LogNScaling logn_scaling(timestep_ + 1, params_.max_position_embeddings);
            PRAGMA_UNROLL
            for (int s = 0; s < kQHeadPerThread; ++s) {
                logn_scaling.apply(frag_Q[s]);
            }
        }

        if (kSplitK && step_begin_) {  // Split idx > 0
            PRAGMA_UNROLL
            for (int s = 0; s < kQHeadPerThread; ++s) {
                int qi = offset.y + s;
                if (lane_id_ == 0) {
                    smem_M_[qi] = -std::numeric_limits<float>::infinity();
                    smem_L_[qi] = 0.f;
                }
                Store(&smem_Q_[qi * kMaxHeadDim + offset.x], frag_Q[s]);
                Store(&smem_O_[qi * kMaxHeadDim + offset.x], VecQFloat{});
            }
            return;
        }

        ////////////////////////////////////////////////////////
        // Split 0 computes last step and stores to k/v cache
        PRAGMA_UNROLL
        for (int s = 0; s < kQHeadPerThread; ++s) {
            int         qi = offset.y + s;
            QkAccumType qk = qk_dot<QkAccumType, QkComputeType, WARP_SIZE>(frag_Q[s], frag_K);
            if (lane_id_ == 0) {
                qk *= params_.inv_sqrt_dh;
                smem_M_[qi] = qk;
                smem_L_[qi] = 1.f;
                // printf("qk[%2d] = %f\n", head_idx_, qk);
            }
            // write Q and O
            Store(&smem_Q_[qi * kMaxHeadDim + offset.x], frag_Q[s]);
            Store(&smem_O_[qi * kMaxHeadDim + offset.x], cast<float>(frag_V));
        }

        auto frag_K_store = conv_k_store_(frag_K);
        auto frag_V_store = conv_v_store_(frag_V);

        // store
        if (warp_id_ == 0 && is_gqa_leader_) {
            int block_index  = timestep_ / params_.kv_cache_block_size;
            int block_offset = timestep_ % params_.kv_cache_block_size;

            k_cache_ = (Tkv*)k_cache_ptrs_[block_index] + params_.layer_offset
                       + kv_head_idx_ * params_.kv_cache_block_size * kHeadDim;
            v_cache_ = (Tkv*)v_cache_ptrs_[block_index] + params_.layer_offset
                       + kv_head_idx_ * params_.kv_cache_block_size * kHeadDim;

            Store(&k_cache_[block_offset * kHeadDim + offset.x], frag_K_store);
            Store(&v_cache_[block_offset * kHeadDim + offset.x], frag_V_store);
        }
    }

    __device__ void PrefetchKvCache(GmemIter& iter)
    {
        PRAGMA_UNROLL
        for (int stage = 0; stage < kStages - 1; ++stage) {
            iter.PrefetchStage();
            CpAsyncCommit();
        }
    }

    __device__ void CpAsyncWait()
    {
        __pipeline_wait_prior(kStages - 2);
        // if constexpr (!std::is_same_v<GmemMap, SmemMap>) {
        //     __syncthreads();
        // }
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

    struct FragmentQ {
        Array<QkComputeType, kVecKvSize> data[kHeadPerCta][SmemMap::kIterC];
    };

    struct State {
        // Double buffering to hide smem/dequant latency
        Array<Tkv, kVecKvSize> frag_Kv_tmp_buf[2][SmemMap::kIterC];
    };

    State state;

    template<bool IsResidue, class IterLength>
    __device__ void ComputeSlice(FragmentQ& frag_Q, const int2& offset, int step, IterLength iter_length)
    {
        // static constexpr int kPrefetchCount = (IterKv<IsResidue>::kIterCount + MapKv::kIterS - 1) / MapKv::kIterS;

#if 0
        // use `KeyPerIter` as min cache block seq len
        __shared__ const void* smem_k_ptrs[kSliceLen / KeyPerIter];
        __shared__ const void* smem_v_ptrs[kSliceLen / KeyPerIter];

        {
            int    block_seq_len      = params_.kv_cache_block_size;
            int    block_beg          = step / block_seq_len;
            int    block_cnt          = (iter_length + block_seq_len - 1) / block_seq_len;
            size_t block_local_offset = params_.layer_offset + head_idx_ * block_seq_len * kHeadDim;
            for (int i = threadIdx.x; i < block_cnt; i += blockDim.x) {
                smem_k_ptrs[i] = (const Tkv*)k_cache_ptrs_[block_beg + i] + block_local_offset;
                smem_v_ptrs[i] = (const Tkv*)v_cache_ptrs_[block_beg + i] + block_local_offset;
            }
            __syncthreads();
        }
#endif

        static constexpr int kPrefetchCount = (GmemIter::kIterCount + SmemMap::kIterS - 1) / SmemMap::kIterS;

        Array<float, kHeadPerCta> frag_M;
        PRAGMA_UNROLL
        for (int i = 0; i < kHeadPerCta; ++i) {
            frag_M[i] = smem_M_[i];
        }

        const int block_seq_len      = params_.kv_cache_block_size;
        const int block_local_offset = (int)params_.layer_offset + head_idx_ * block_seq_len * kHeadDim;
        const int block_shifter      = 31 - __clz(block_seq_len);  // faster than `__ffs`
        const int block_beg          = step >> block_shifter;
        const int max_iter           = (iter_length + KeyPerIter - 1) / KeyPerIter;

        GmemIter gmem_iter_K{
            k_cache_ptrs_ + block_beg, block_seq_len, block_local_offset, smem_Kv_, max_iter, warp_id_, lane_id_};

        PrefetchKvCache(gmem_iter_K);

        SmemIter smem_iter_K{smem_Kv_, warp_id_, lane_id_};
        CpAsyncWait();

        smem_iter_K.Load(state.frag_Kv_tmp_buf[0]);

        // Start prefetching next stage
        gmem_iter_K.PrefetchBatch(0, kPrefetchCount);

        static_assert(SmemMap::kIterS >= 2);

        ///////////////////////////////////////////////////////////////////////////////////////////
        /// Compute QK(Q, S) = Q(Q, D) * K^T(D, S)
        PRAGMA_NO_UNROLL
        for (int _it = 0; _it < iter_length; _it += kKeyPerIter) {

            PRAGMA_UNROLL
            for (int si = 0; si < SmemMap::kIterS; ++si) {
                // smem -> rmem for next iter
                smem_iter_K.Load(state.frag_Kv_tmp_buf[(si + 1) % 2]);

                gmem_iter_K.PrefetchBatch((si + 1) % SmemMap::kIterS, kPrefetchCount);
                if (si == SmemMap::kIterS - 2) {
                    CpAsyncCommit();
                    CpAsyncWait();
                    gmem_iter_K.AdvanceStage();
                    smem_iter_K.AdvanceStage();
                }

                auto& frag_K_tmp = state.frag_Kv_tmp_buf[si % 2];

                Array<KLoadType, kVecKvSize> frag_K[SmemMap::kIterC];
                PRAGMA_UNROLL
                for (int vi = 0; vi < SmemMap::kIterC; ++vi) {
                    frag_K[vi] = conv_v_(frag_K_tmp[vi]);
                }
                const int local_offset = offset.y + _it + si * SmemMap::kDeltaS;
                PRAGMA_UNROLL
                for (int qi = 0; qi < kHeadPerCta; ++qi) {
                    auto qk = qk_dot<QkAccumType, QkComputeType, SmemMap::kWarpThreadC>(frag_Q.data[qi], frag_K);
                    qk *= params_.inv_sqrt_dh;
                    if (!IsResidue || step + local_offset < timestep_) {
                        // group leader writes to smem
                        if (threadIdx.x % SmemMap::kWarpThreadC == 0) {
                            // printf("QK[%04d] = %f\n", step + local_offset, (float)qk);
                            smem_S_[kSliceLen * qi + local_offset] = qk;
                            // local max
                            frag_M[qi] = fmaxf(frag_M[qi], qk);
                        }
                    }
                }
            }
        }

        CpAsyncFlush();

        // Start prefetching of V
        GmemIter gmem_iter_V{
            v_cache_ptrs_ + block_beg, block_seq_len, block_local_offset, smem_Kv_, max_iter, warp_id_, lane_id_};
        PrefetchKvCache(gmem_iter_V);

        __syncthreads();

        Array<float, kHeadPerCta> exp_M_diff;
        PRAGMA_UNROLL
        for (int i = 0; i < kHeadPerCta; ++i) {
            exp_M_diff[i] = smem_M_[i];
        }

        /// block synchronization
        frag_M = qk_max<SmemMap>(frag_M, smem_red_max_, warp_id_, lane_id_);

        PRAGMA_UNROLL
        for (int i = 0; i < kHeadPerCta; ++i) {
            exp_M_diff[i] = __expf(exp_M_diff[i] - frag_M[i]);

            if (threadIdx.x == 0) {
                smem_M_[i] = frag_M[i];
            }
        }

        // if (threadIdx.x == 0 && step + iter_length == timestep_) {
        //     printf("frag_M[%2d] = %f\n", head_idx_, (float)frag_M[0]);
        // }

        // __syncthreads();  // DEBUG

        /////////////////////////////////////////////////////////////////////////////////////////
        // / Compute softmax P(Q, S)
        Array<float, kHeadPerCta> frag_L{};

        for (int ti = threadIdx.x; ti < iter_length; ti += kWarpCount * WARP_SIZE) {
            PRAGMA_UNROLL
            for (int qi = 0; qi < kHeadPerCta; ++qi) {
                int   idx = qi * kSliceLen + ti;
                float qk  = smem_S_[idx];
                float pr  = expf(qk - frag_M[qi]);
                // printf("smem_P[%d] = %f\n", ti, pr);
                smem_P_[idx] = pr;
                frag_L[qi] += pr;
            }
        }

        /// block synchronization
        frag_L = blockSum<kWarpCount>(frag_L, smem_red_sum_, warp_id_, lane_id_);

        for (int qi = 0; qi < kHeadPerCta; ++qi) {
            // exp(m1 - m2) * l1
            frag_L[qi] += exp_M_diff[qi] * smem_L_[qi];
        }

        __syncthreads();

        for (int qi = 0; qi < kHeadPerCta; ++qi) {
            if (threadIdx.x == 0) {
                smem_L_[qi] = frag_L[qi];
            }
        }

        // if (threadIdx.x == 0 && step == timestep_ - kSliceLen) {
        //     printf("frag_L'[%d] = %f\n", head_idx_, (float)frag_L[0]);
        // }

        /////////////////////////////////////////////////////////////////////////////////////////
        // / Compute O[H,D] = P[H,S] * V[S,D]
        VecKvFloat frag_O[kHeadPerCta][SmemMap::kIterC]{};  // value initialize

        SmemIter smem_iter_V{smem_Kv_, warp_id_, lane_id_};

        CpAsyncWait();
        smem_iter_V.Load(state.frag_Kv_tmp_buf[0]);

        gmem_iter_V.PrefetchBatch(0, kPrefetchCount);

        PRAGMA_NO_UNROLL
        for (int _it = 0; _it < iter_length; _it += kKeyPerIter) {
            PRAGMA_UNROLL
            for (int si = 0; si < SmemMap::kIterS; ++si) {
                const int next = (si + 1) % 2;
                // Load value cache for next warp iter
                smem_iter_V.Load(state.frag_Kv_tmp_buf[next]);

                gmem_iter_V.PrefetchBatch((si + 1) % SmemMap::kIterS, kPrefetchCount);

                if (si == SmemMap::kIterS - 2) {
                    CpAsyncCommit();
                    CpAsyncWait();
                    gmem_iter_V.AdvanceStage();
                    smem_iter_V.AdvanceStage();
                }

                auto& frag_V_tmp = state.frag_Kv_tmp_buf[si % 2];

                Array<VLoadType, kVecKvSize> frag_V[SmemMap::kIterC];
                PRAGMA_UNROLL
                for (int vi = 0; vi < SmemMap::kIterC; ++vi) {
                    frag_V[vi] = conv_v_(frag_V_tmp[vi]);
                }

                const int local_offset = offset.y + _it + si * SmemMap::kDeltaS;

                float frag_P[kHeadPerCta];
                PRAGMA_UNROLL
                for (int qi = 0; qi < kHeadPerCta; ++qi) {
                    frag_P[qi] = smem_P_[qi * kSliceLen + local_offset];
                }

                if (!IsResidue || step + local_offset < timestep_) {
                    PRAGMA_UNROLL
                    for (int qi = 0; qi < kHeadPerCta; ++qi) {
                        fma_pv<PvComputeType>(frag_P[qi], frag_V, frag_O[qi]);
                    }
                }
            }
        }

        /// warp reduce over S dim
        PRAGMA_UNROLL
        for (int qi = 0; qi < kHeadPerCta; ++qi) {
            PRAGMA_UNROLL
            for (int vi = 0; vi < SmemMap::kIterC; ++vi) {
                PRAGMA_UNROLL
                for (int i = 0; i < kVecKvSize; ++i) {
                    // reduce over warp thread S
                    PRAGMA_UNROLL
                    for (int mask = WARP_SIZE / 2; mask >= SmemMap::kWarpThreadC; mask /= 2) {
                        frag_O[qi][vi][i] += __shfl_xor_sync(uint32_t(-1), frag_O[qi][vi][i], mask);
                    }
                }
            }
        }

        // __syncthreads();

        /// rescale output & block reduce
        PRAGMA_UNROLL
        for (int wi = 0; wi < SmemMap::kWarpCount; ++wi) {
            PRAGMA_UNROLL
            for (int qi = 0; qi < kHeadPerCta; ++qi) {
                PRAGMA_UNROLL
                for (int vi = 0; vi < SmemMap::kIterC; ++vi) {
                    // first partition of corresponding warp
                    if (warp_id_ == wi && lane_id_ < SmemMap::kWarpThreadC) {
                        // bank conflict?
                        auto& smem_O = (VecKvFloat&)smem_O_[qi * kMaxHeadDim + offset.x + vi * SmemMap::kDeltaC];
                        using namespace ops;
                        auto tmp_O = smem_O;
                        if (warp_id_ == 0) {
                            tmp_O = tmp_O * exp_M_diff[qi];
                        }
                        // bank conflict?
                        smem_O = tmp_O + frag_O[qi][vi];
                    }
                }
            }
            __syncthreads();
        }

        CpAsyncFlush();
    }

    __device__ void LoopKv()
    {
        const int2 offset = SmemMap::get_offset(warp_id_, lane_id_);

        ///////////////////////////////////////////////////////////////////////////////////////////
        /// Load Q from shared memory.
        /// NOTE: There will be bank-conflict when sizeof(VecKv) > 16 (e.g. KV is quantized)
        FragmentQ frag_Q;

        PRAGMA_UNROLL
        for (int qi = 0; qi < kHeadPerCta; ++qi) {
            PRAGMA_UNROLL
            for (int c = 0; c < SmemMap::kIterC; ++c) {
                const int di       = offset.x + SmemMap::kDeltaC * c;
                frag_Q.data[qi][c] = cast<QkComputeType>((VecKv&)smem_Q_[qi * kMaxHeadDim + di]);
            }
        }

        int step_end = step_end_;
        if (step_end % kSliceLen) {
            int residue_start = step_end / kSliceLen * kSliceLen;
            int iter_count    = min(step_end - residue_start, kSliceLen);
            ComputeSlice<true>(frag_Q, offset, residue_start, iter_count);
            step_end = residue_start;
        }

        PRAGMA_NO_UNROLL
        for (int step = step_begin_; step < step_end; step += kSliceLen) {
            ComputeSlice<false>(frag_Q, offset, step, std::integral_constant<int, kSliceLen>{});
        }
    }

    __device__ void Run()
    {

        // early exit if split if out of bound
        if (kSplitK && step_begin_ >= step_end_) {
            return;
        }

        // early exit if finished flag is set
        if (params_.finished[batch_idx_]) {
            return;
        }

        // Compute attention for current step
        Prolugue();

        __syncthreads();

        // Iterate over K/V
        LoopKv();

        __syncthreads();

        // Normalize outputs & write to device memory
        Epilogue();
    }

    __device__ void Epilogue()
    {
        static constexpr int kVecQSize = kMaxHeadDim / WARP_SIZE;

        using VecQFloat = Array<float, kVecQSize>;

        using MapQ = ThreadMapQ<kMaxHeadDim, kHeadPerCta, kVecQSize, kWarpCount>;

        static constexpr int kQkvHeadPerThread = MapQ::kIterS;

        int2 offset = MapQ::get_offset(warp_id_, lane_id_);

        using namespace ops;

        if (step_begin_ == 0 && step_end_ == timestep_) {  // non-split-k
            if (offset.x >= kMaxHeadDim || offset.y >= kHeadPerCta) {
                return;
            }
            PRAGMA_UNROLL
            for (int s = 0; s < kQkvHeadPerThread; ++s) {
                const int di = offset.x;
                const int qi = offset.y + s;

                const float     scale  = __fdividef(1.f, smem_L_[qi] + 1e-8f);
                const VecQFloat frag_O = (VecQFloat&)smem_O_[qi * kMaxHeadDim + di] * scale;

                Store(&params_.out[batch_idx_ * params_.num_heads * kHeadDim + (head_idx_ + qi) * kHeadDim + di],
                      cast<T>(frag_O));
            }
        }
        else {
            StorePartial();

            const auto index       = (batch_idx_ * params_.num_heads + head_idx_) * params_.max_split_k;
            const auto locks       = params_.locks + index;
            const int  split_k_idx = get_split_k_idx();
            const int  thread_idx  = threadIdx.x;

            if (step_end_ != timestep_) {
                sem_post(&locks[split_k_idx], 1, thread_idx == 0);
            }
            else {
                sem_wait_many(&locks[threadIdx.x], split_k_idx, thread_idx < split_k_idx);

                ReduceLastSplit();

                if (thread_idx < split_k_idx) {
                    locks[thread_idx] = 0;
                }
            }
        }
    }

    __device__ void StorePartial()
    {
        static constexpr int kVecQSize = kMaxHeadDim / WARP_SIZE;

        using VecQFloat = Array<float, kVecQSize>;
        using MapQ      = ThreadMapQ<kMaxHeadDim, kHeadPerCta, kVecQSize, kWarpCount>;

        int2 offset = MapQ::get_offset(warp_id_, lane_id_);

        if (offset.x >= kMaxHeadDim || offset.y >= kHeadPerCta) {
            return;
        }

        PRAGMA_UNROLL
        for (int s = 0; s < MapQ::kIterS; ++s) {  // split-k
            const int di = offset.x;
            const int qi = offset.y + s;

            const VecQFloat frag_O = (VecQFloat&)smem_O_[qi * kMaxHeadDim + di];

            // [B, H, k, D]
            const int index = batch_idx_ * params_.num_heads * params_.max_split_k
                              + (head_idx_ + qi) * params_.max_split_k + get_split_k_idx();
            Store(&params_.partial_O[index * kHeadDim + di], cast<float>(frag_O));

            if (di == 0) {
                params_.partial_M[index] = smem_M_[qi];
                params_.partial_L[index] = smem_L_[qi];
            }
        }
    }

    __device__ void ReduceLastSplit()
    {
        const int split_k_idx = get_split_k_idx();
        const int lane_id     = threadIdx.x % WARP_SIZE;

        const int index =
            batch_idx_ * params_.num_heads * params_.max_split_k + head_idx_ * params_.max_split_k + lane_id;

        const int split_k = split_k_idx + 1;

        __shared__ __align__(16) float smem_scale[WARP_SIZE];

        float global_M = lane_id < split_k ? params_.partial_M[index] : -std::numeric_limits<float>::infinity();
        PRAGMA_UNROLL
        for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
            global_M = fmaxf(global_M, __shfl_xor_sync((uint32_t)-1, global_M, mask));
        }

        float global_L   = 0.f;
        float exp_M_diff = 1.f;

        if (lane_id < split_k) {
            exp_M_diff = expf(params_.partial_M[index] - global_M);
            global_L   = exp_M_diff * params_.partial_L[index];
        }

        PRAGMA_UNROLL
        for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
            global_L += __shfl_xor_sync((uint32_t)-1, global_L, mask);
        }

        if (threadIdx.x < split_k) {
            smem_scale[threadIdx.x] = exp_M_diff / (global_L + 1e-8f);
        }

        __syncthreads();

        int idx = (batch_idx_ * params_.num_heads * params_.max_split_k + head_idx_ * params_.max_split_k) * kHeadDim
                  + threadIdx.x;
        float accum_O{};

        const bool mask = threadIdx.x < kHeadDim;

        for (int k = 0; k < split_k; ++k) {
            if (mask) {
                accum_O += smem_scale[k] * params_.partial_O[idx];
            }
            idx += kHeadDim;
        }
        if (mask) {
            params_.out[batch_idx_ * params_.num_heads * kHeadDim + head_idx_ * kHeadDim + threadIdx.x] = (T)accum_O;
        }
    }

    __device__ void ReduceIterative()
    {
        auto lock = &params_.locks[batch_idx_ * params_.num_heads + head_idx_];

        sem_wait(lock, get_split_k_idx(), threadIdx.x);

        for (int i = threadIdx.x; i < kHeadPerCta * kHeadDim; i += kWarpCount * WARP_SIZE) {

            int qi = i / kHeadDim;
            int di = i % kHeadDim;

            float frag_O2 = smem_O_[qi * kHeadDim + di];

            // [B, H, k, D]
            const int index = batch_idx_ * params_.num_heads * params_.max_split_k
                              + (head_idx_ + qi) * params_.max_split_k + get_split_k_idx();

            float frag_M2 = smem_M_[qi];
            float frag_L2 = smem_L_[qi];

            using namespace ops;

            if (step_begin_ != 0) {
                float frag_M1     = params_.partial_M[index - 1];
                float frag_L1     = params_.partial_L[index - 1];
                float frag_M      = fmaxf(frag_M1, frag_M2);
                float exp_M1_diff = expf(frag_M1 - frag_M);
                float exp_M2_diff = expf(frag_M2 - frag_M);
                frag_M2           = frag_M;
                frag_L2           = exp_M1_diff * frag_L1 + exp_M2_diff * frag_L2;
                float frag_O1     = params_.partial_O[(index - 1) * kHeadDim + di];
                frag_O2           = frag_O1 * exp_M1_diff + frag_O2 * exp_M2_diff;
            }

            if (step_end_ == timestep_) {
                const float scale = __fdividef(1.f, frag_L2 + 1e-8f);
                frag_O2           = frag_O2 * scale;
                params_.out[batch_idx_ * params_.num_heads * kHeadDim + (head_idx_ + qi) * kHeadDim + di] = (T)frag_O2;
            }
            else {
                params_.partial_O[index * kHeadDim + di] = (float)frag_O2;
                if (di == 0) {
                    params_.partial_M[index] = frag_M2;
                    params_.partial_L[index] = frag_L2;
                }
            }
        }

        auto val = step_end_ == timestep_ ? 0 : get_split_k_idx() + 1;

        sem_post(lock, val, threadIdx.x);
    }

    static __device__ void Reduce(const ParamType& params)
    {
        const int batch_idx       = get_batch_idx();
        const int head_idx        = get_head_idx();
        const int timestep        = params.per_sample_length[batch_idx];
        const int max_split_k     = params.max_split_k;
        const int slice_count     = get_slice_count(timestep);
        const int slice_per_split = (slice_count + max_split_k - 1) / max_split_k;
        const int split_k         = (slice_count + slice_per_split - 1) / slice_per_split;

        if (split_k == 1) {
            return;
        }

        // [B, H, k, D]
        const int index = batch_idx * params.num_heads * max_split_k + head_idx * max_split_k + threadIdx.x;

        __shared__ float smem_global_M;
        __shared__ float smem_global_L;
        __shared__ __align__(16) float smem_expdiff_M[WARP_SIZE];
        __shared__ __align__(16) float smem_scale_O[WARP_SIZE];

        {
            float global_M = threadIdx.x < split_k ? params.partial_M[index] : -std::numeric_limits<float>::infinity();
            PRAGMA_UNROLL
            for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
                global_M = fmaxf(global_M, __shfl_xor_sync((uint32_t)-1, global_M, mask));
            }

            if (threadIdx.x == 0) {
                smem_global_M = global_M;
            }
        }

        __syncthreads();

        {
            float global_L = threadIdx.x < split_k ? params.partial_L[index] : 0.f;

            if (threadIdx.x < split_k) {
                auto expdiff_M = expf(params.partial_M[index] - smem_global_M);
                global_L *= expdiff_M;
                smem_expdiff_M[threadIdx.x] = expdiff_M;
            }

            PRAGMA_UNROLL
            for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
                global_L += __shfl_xor_sync((uint32_t)-1, global_L, mask);
            }

            if (threadIdx.x == 0) {
                smem_global_L = global_L;
            }
        }

        __syncthreads();

        if (threadIdx.x < split_k) {
            smem_scale_O[threadIdx.x] = smem_expdiff_M[threadIdx.x] / (smem_global_L + 1e-8f);
        }

        __syncthreads();

        int   idx = (batch_idx * params.num_heads * max_split_k + head_idx * max_split_k) * kHeadDim + threadIdx.x;
        float accum_O{};

        const bool is_valid = threadIdx.x < kHeadDim;

        for (int k = 0; k < split_k; ++k) {
            if (is_valid) {
                accum_O += smem_scale_O[k] * params.partial_O[idx];
            }
            idx += kHeadDim;
        }
        if (is_valid) {
            params.out[batch_idx * params.num_heads * kHeadDim + head_idx * kHeadDim + threadIdx.x] = (T)accum_O;
        }
    }

    static __device__ int get_slice_count(int timestep)
    {
        return (timestep + kSliceLen - 1) / kSliceLen;
    }

    static __device__ int get_head_idx()
    {
        return blockIdx.x;
    }

    static __device__ int get_batch_idx()
    {
        return blockIdx.y;
    }

    static __device__ int get_split_k_idx()
    {
        return blockIdx.z;
    }
};

extern __shared__ uint8_t dynamic_smem[];

template<typename MHAType, typename ParamType = typename MHAType::ParamType>
__global__ void decoder_multihead_attention(ParamType params)
{
    __shared__ typename MHAType::SharedStorage shared_storage;

    uint8_t* smem_ptr = dynamic_smem;

    MHAType{params, shared_storage, smem_ptr}.Run();
}

template<typename MHAType, typename ParamType = typename MHAType::ParamType>
__global__ void decoder_multihead_attention_reduce(ParamType params)
{
    MHAType::Reduce(params);
}

}  // namespace turbomind
