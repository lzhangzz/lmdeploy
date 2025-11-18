/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Modified from
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/layers/attention_layers/GptContextAttentionLayer.cc

#include <algorithm>
#include <functional>
#include <math.h>
#include <numeric>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/core.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/exchange.h"
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/engine/request.h"

#include "src/turbomind/kernels/attention/attention.h"
#include "src/turbomind/kernels/attention/decoding.h"
#include "src/turbomind/kernels/attention/kv_cache_utils_v2.h"
#include "src/turbomind/kernels/norm/rms_norm.h"

#include "src/turbomind/macro.h"

#include "src/turbomind/models/llama/llama_rope.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/mla_utils.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"

#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind {

UnifiedAttentionLayer::~UnifiedAttentionLayer()
{
    for (auto& s : streams_) {
        s = {};
    }

    check_cuda_error(cudaEventDestroy(aux_event_));
    check_cuda_error(cudaEventDestroy(qkv_event_));
    check_cuda_error(cudaStreamDestroy(aux_stream_));

    aux_event_ = qkv_event_ = {};
    aux_stream_             = {};
}

UnifiedAttentionLayer::UnifiedAttentionLayer(const ModelParam&     model,
                                             const AttentionParam& attn,
                                             const EngineParam&    engine,
                                             const LoraParam&      lora,
                                             int                   tp_size,
                                             const Context&        ctx,
                                             int                   phases):
    head_num_(model.head_num),
    kv_head_num_(model.kv_head_num),
    size_per_head_(model.head_dim),
    hidden_units_(model.hidden_units),
    local_head_num_(head_num_ / tp_size),
    local_kv_head_num_(model.kv_head_num / tp_size),
    param_(attn),
    model_param_(model),
    lora_param_(lora),
    context_(ctx),
    stream_(ctx.stream),
    linear_(*ctx.linear),
    arch_(getSMVersion())
{
    TM_CHECK_EQ(head_num_ % tp_size, 0) << head_num_ << " " << tp_size;
    TM_CHECK_EQ(head_num_ % kv_head_num_, 0) << head_num_ << " " << kv_head_num_;

    check_cuda_error(cudaStreamCreateWithFlags(&aux_stream_, cudaStreamNonBlocking));
    check_cuda_error(cudaEventCreateWithFlags(&qkv_event_, cudaEventDisableTiming));
    check_cuda_error(cudaEventCreateWithFlags(&aux_event_, cudaEventDisableTiming));

    streams_[0] = stream_;
    streams_[1] = aux_stream_;

    init_rope_kernel_param(param_.rope, rope_param_);

    partial_M_ = Tensor_<float>({kMaxWorkspaceTokens, local_head_num_}, kDEVICE);
    partial_L_ = Tensor_<float>({kMaxWorkspaceTokens, local_head_num_}, kDEVICE);
    partial_O_ = Tensor_<float>({kMaxWorkspaceTokens, local_head_num_, size_per_head_}, kDEVICE);
    split_cnt_ = Tensor_<int>({kMaxWorkspaceTokens}, kDEVICE);
    barriers_  = Tensor_<int>({kMaxWorkspaceTokens, local_head_num_}, kDEVICE);

    Clear(split_cnt_.buffer());
    Clear(barriers_.buffer());

    rope_base_buf_        = {engine.max_batch_size + 1, kCPUpinned};
    decode_token_pos_buf_ = {engine.max_batch_size, kCPUpinned};

    const int max_blocks = engine.max_batch_size * cdiv(engine.session_len, param_.cache_block_seq_len);
    for (int i = 0; i < phases; ++i) {
        data_.push_back(std::make_shared<AttentionData>(engine.max_batch_size, max_blocks, rope_param_));
    }
}

struct AttentionData {

    struct Stat {
        int n;
        int q_sum;
        int q_max;
        int k_sum;
        int k_max;
    } decode, prefill;

    // Buffer_<int> h_offset_q;
    // Buffer_<int> h_offset_k;

    // Buffer_<int> d_offset_q;
    // Buffer_<int> d_offset_k;

    Buffer_<void*> block_ptrs;
    Buffer_<int>   block_ptrs_offsets;

    // Buffer_<int> decode_token_pos;

    Buffer_<float> rope_base;

    Tensor_<int> mrope_position_ids;
    Buffer_<int> mrope_position_delta;
    Buffer_<int> mrope_length;

    // borrowed from env
    Buffer_<bool> finished;
    Buffer_<int>  offset_q;
    Buffer_<int>  offset_k;

    AttentionData(int bsz, int max_blocks, RopeKernelParam& rope)
    {
        // h_offset_q = {bsz + 1, kCPUpinned};
        // h_offset_k = {bsz + 1, kCPUpinned};
        // d_offset_q = {bsz + 1, kDEVICE};
        // d_offset_k = {bsz + 1, kDEVICE};

        block_ptrs         = {max_blocks + 16, kDEVICE};
        block_ptrs_offsets = {bsz + 1, kDEVICE};

        // decode_token_pos = {bsz, kDEVICE};

        if (rope.type == RopeType::kDynamic) {
            rope_base = {bsz, kDEVICE};
        }
    }
};

static void init_dynamic_ntk(RequestCache& cache, const RopeParam& rope)
{
    cache.rope_base = rope.base;
    if (auto scaling_factor = rope.factor; scaling_factor > 1.f) {
        const auto max_seq_len = cache.prompt_len;
        const auto max_pos_emb = rope.max_position_embeddings;
        if (max_seq_len > max_pos_emb) {
            scaling_factor = scaling_factor * max_seq_len / max_pos_emb - (scaling_factor - 1);
            cache.rope_base *= powf(scaling_factor, rope.dim / (rope.dim - 2.f));
            // clang-format off
            TM_LOG_INFO("[ProcessInferRequests] %ld rope_scaling_factor: %f, rope_theta = %f",
                        (long)cache.request->id, scaling_factor, cache.rope_base);
            // clang-format on
        }
    }
}

void UnifiedAttentionLayer::Run(ExchOp op, int phase, TensorMap& env)
{
    if (op == ExchOp::kAdd) {
        Buffer_<RequestCache*> rc = env.at("requests").buffer();
        if (rope_param_.type == RopeType::kDynamic) {
            for (int i = 0; i < rc.size(); ++i) {
                init_dynamic_ntk(*rc[i], param_.rope);
            }
        }
    }
    else if (op == ExchOp::kSetup) {
        Setup(phase, env);
    }
    else if (op == ExchOp::kPrepare) {
        data_.at(phase)->finished = env.at("finished").buffer().borrow();
        data_.at(phase)->offset_q = env.at("offset_q").buffer().borrow();
        data_.at(phase)->offset_k = env.at("offset_k").buffer().borrow();
        // env["decode_token_pos"]   = data_.at(phase)->decode_token_pos;
    }
}

void UnifiedAttentionLayer::Setup(int phase, TensorMap& env)
{
    Buffer_<RequestCache*> rc  = env.at("requests").buffer();
    const int              bsz = rc.size();

    auto& d = *data_.at(phase);

    {  /// Upload KV cache ptrs
        const Buffer_<int> offsets = env.at("block_ptrs_offsets").buffer();
        Copy(env.at("block_ptrs").buffer(), offsets[bsz], d.block_ptrs);
        Copy(offsets, bsz + 1, d.block_ptrs_offsets);
    }

    /// prepare Q/K stats for decode/prefill
    d.decode = d.prefill = {};

    d.decode.n  = std::find_if(rc.begin(), rc.end(), [](auto r) { return r->input_len > 1; }) - rc.begin();
    d.prefill.n = bsz - d.decode.n;

    // d.h_offset_q[0] = d.h_offset_k[0] = 0;
    for (int i = 0; i < bsz; ++i) {
        const auto& c = *rc[i];

        // d.h_offset_q[i + 1] = d.h_offset_q[i] + c.input_len;
        // d.h_offset_k[i + 1] = d.h_offset_k[i] + c.context_len;

        auto& s = i < d.decode.n ? d.decode : d.prefill;
        s.q_sum += c.input_len;
        s.k_sum += c.context_len;
        s.q_max = std::max(s.q_max, c.input_len);
        s.k_max = std::max(s.k_max, c.context_len);

        // last input token
        // decode_token_pos_buf_[i] = d.h_offset_q[i + 1] - 1;
        // TM_LOG_ERROR("%d deocde pos %d", i, decode_token_pos_buf_[i]);
    }

    // Copy_(d.h_offset_q, bsz + 1, d.d_offset_q);
    // Copy_(d.h_offset_k, bsz + 1, d.d_offset_k);
    // Copy_(decode_token_pos_buf_, bsz, d.decode_token_pos);

    /// handling different RoPE types
    if (rope_param_.type == RopeType::kDynamic) {
        for (int i = 0; i < bsz; ++i) {
            rope_base_buf_[i] = rc[i]->rope_base;
        }
        Copy_(rope_base_buf_, bsz, d.rope_base);
    }
    else if (rope_param_.type == RopeType::kMrope) {
        TM_CHECK(0) << "not implemented.";
    }
}

void UnifiedAttentionLayer::Forward(ForwardParam p)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    /**
     * input_tensors:
     *   \param input_query [token_num, hidden_dim]
     *   \param cu_q_len [batch_size+1], int
     *   \param cu_k_len [batch_size+1], int
     *   \param cu_block_counts [batch_size+1], int
     *   \param finished [batch_size], bool
     *   \param rope_theta [batch_size], float
     *   \param h_q_len [batch_size], int on cpu
     *   \param h_k_len [batch_size], int on cpu
     *   \param h_cu_q_len [batch_size+1], int on cpu
     *   \param h_cu_k_len [batch_size+1], int on cpu
     *   \param dc_batch_size [1], int on cpu
     *   \param pf_batch_size [1], int on cpu
     *   \param layer_id [1], int on cpu
     *
     * output_tensors:
     *   \param hidden_features [token_num, hidden_dim], float
     *   \param block_ptrs [total_block_counts], void*
     */

    /////////////////////////////////////////////
    /// parse inputs
    const int token_num = p.input.shape(0);

    if (token_num == 0) {
        return;
    }

    const int layer_id = p.layer_id;

    const auto& weights = *p.weights;

    // [L, 2, H, s, D]
    const size_t layer_offset = layer_id * 2 * local_kv_head_num_ * param_.cache_block_seq_len * size_per_head_;

    Tensor qkv;

    if (weights.qkv.output_dim) {
        // [token_num, hidden_dim] -> [token_num, local_q_kv_head_num, head_dim]
        qkv = linear_.Forward(p.input, weights.qkv);
        sync_check_cuda_error();

        if (model_param_.qk_norm) {
            qk_norm(qkv, weights);
        }
    }
    else {
        qkv = forward_mla(p.input, weights);
    }

    TM_DEBUG_TENSOR(qkv, Concat("qkv", layer_id), 3);

    auto invoke = [&](auto t) -> Tensor {
        using T = decltype(t);
        return core_attention<T>(qkv, p, weights);
    };

    Tensor attn = [&]() -> Tensor { TM_DISPATCH_PRIMARY_DTYPES_RET(qkv.dtype(), invoke); }();

    TM_DEBUG_TENSOR(attn, Concat("attn", layer_id), 3);

    //////////////////////////////////////////////
    /// output gemm <Bs,HD> -> <Bs,HD>
    (void)linear_.Forward(attn, weights.output, p.output);
    sync_check_cuda_error();
}

template<class T>
Tensor UnifiedAttentionLayer::core_attention(Tensor& qkv, const ForwardParam& p, const WeightType& weights)
{
    const auto device = qkv.device();
    const auto dtype  = qkv.dtype();

    auto& d = *data_.at(p.phase);

    const int batch_size = d.decode.n + d.prefill.n;
    const int q_count    = qkv.shape(0);

    const int local_q_kv_head_num = local_head_num_ + 2 * local_kv_head_num_;

    Tensor attn{{q_count, (int)local_head_num_ * (int)size_per_head_}, dtype, device};
    Tensor tmp_kv{{2, (int)local_kv_head_num_, d.prefill.k_sum + MAX_CTA_S, (int)size_per_head_}, dtype, device};

    auto stream_ptr = streams_.data();

    auto CreateParams = [&](int offset, AttentionData::Stat stat, int max_kv_splits, cudaStream_t stream) {
        AttentionParams<T> params{};

        // Batch offset for `out` and `q` are computed inside the kernel
        params.out = (T*)attn.raw_data();

        params.q      = (T*)qkv.raw_data();
        params.k      = params.q + local_head_num_ * size_per_head_;
        params.v      = params.k + local_kv_head_num_ * size_per_head_;
        params.stride = (local_head_num_ + 2 * local_kv_head_num_) * size_per_head_;

        if (weights.qkv.bias) {
            params.q_bias = (T*)weights.qkv.bias.data_or<T>(nullptr);
            params.k_bias = params.q_bias + local_head_num_ * size_per_head_;
            params.v_bias = params.k_bias + local_kv_head_num_ * size_per_head_;
        }

        params.batch_size = batch_size;

        params.token_num = stat.q_sum;
        params.max_q_len = stat.q_max;
        params.max_k_len = stat.k_max;

        // decode only
        params.block_iter_params = BlockIteratorParams{(char**)d.block_ptrs.data(),  //
                                                       d.block_ptrs_offsets.data() + offset,
                                                       p.layer_id,
                                                       (int)param_.cache_block_seq_len};

        // prefill only
        params.linear_iter_params = LinearIteratorParams{tmp_kv.raw_data(),  //
                                                         int(2 * stat.k_sum * size_per_head_),
                                                         int(stat.k_sum * size_per_head_)};

        params.finished = d.finished.data() + offset;
        params.cu_q_len = d.offset_q.data() + offset;
        params.cu_k_len = d.offset_k.data() + offset;

        params.num_heads     = local_head_num_;
        params.num_kv_heads  = local_kv_head_num_;
        params.size_per_head = size_per_head_;

        double scaling = 1.;
        if (param_.softmax_scale) {  // model predefined softmax scale
            scaling *= param_.softmax_scale;
        }
        else {  // default value
            scaling /= std::sqrt((float)params.size_per_head);
        }
        params.inv_sqrt_dh = scaling * std::log2(std::exp(1.));

        params.sinks       = weights.sinks.data_or((T*)nullptr);
        params.scale_sinks = scaling;

        params.window_size = weights.window_size;
        if (!params.window_size) {
            params.window_size = 256 << 20;  // 256 M
        }

        // add offset to rope
        params.rope_param = rope_param_;
        if (rope_param_.type == RopeType::kDynamic) {
            params.rope_param.base += offset;
        }
        else if (rope_param_.type == RopeType::kMrope) {
            params.rope_param.mrope.position_ids += offset * rope_param_.mrope.stride;
            params.rope_param.mrope.position_delta += offset;
            params.rope_param.mrope.length += offset;
        }

        // logn attn
        params.use_logn_attn           = param_.use_logn_attn;
        params.max_position_embeddings = param_.max_position_embeddings;

        // Decoding use only for now
        params.split_cnt   = split_cnt_.data();
        params.partial_L   = partial_L_.data();
        params.partial_M   = partial_M_.data();
        params.partial_O   = partial_O_.data();
        params.locks       = barriers_.data();
        params.max_split_k = std::min(std::max(1, kMaxWorkspaceTokens / params.token_num), max_kv_splits);

        params.arch   = arch_;
        params.stream = stream;

        params.quant_policy = model_param_.quant_policy;
        return params;
    };

    cudaStream_t pf_stream = stream_;
    cudaStream_t dc_stream = stream_;

    if (d.decode.n && d.prefill.n) {
        pf_stream = aux_stream_;
        check_cuda_error(cudaEventRecord(qkv_event_, stream_));
        check_cuda_error(cudaStreamWaitEvent(aux_stream_, qkv_event_));
    }

    if (d.prefill.n && !isTuning()) {
        const int offset = d.decode.n;
        // We are executing prefill & decoding kernels concurrently, but only have 1 workspace
        // disable split kv for prefill for now
        auto params = CreateParams(offset, d.prefill, 1, pf_stream);
        if constexpr (sizeof(T) == 2) {
            invokeProcessKV_v2_(params);
            sync_check_cuda_error();

            /// TODO: skip flattening for `sm_80`
            invokeFlattenKV_v2_(params, d.prefill.k_sum);
            sync_check_cuda_error();

            dispatchAttention(params);
            sync_check_cuda_error();
        }
    }

    if (d.decode.n && !isTuning()) {
        auto params = CreateParams(0, d.decode, kMaxKVSplits, dc_stream);
        if constexpr (sizeof(T) == 2) {
            dispatchDecoding<T>(params);
            sync_check_cuda_error();
        }
    }

    if (d.decode.n && d.prefill.n) {
        check_cuda_error(cudaEventRecord(aux_event_, aux_stream_));
        check_cuda_error(cudaStreamWaitEvent(stream_, aux_event_));
    }

    if (isTuning()) {
        rng_.set_stream(stream_);
        rng_.GenerateUniform(attn.data<T>(), attn.size(), .02f, -.01f);
    }

    return attn;
}

Tensor UnifiedAttentionLayer::forward_mla(const Tensor& hidden_state, const WeightType& w)
{
    const int q_lora_rank  = w.q_a_proj.output_dim;
    const int kv_lora_rank = w.kv_b_proj.input_dim;
    const int qk_rope_dim  = w.kv_a_proj.output_dim - kv_lora_rank;
    const int qk_nope_dim  = std::max(w.q_b_proj.output_dim, w.q_proj.output_dim) / local_head_num_ - qk_rope_dim;
    const int v_head_dim   = w.kv_b_proj.output_dim / local_head_num_ - qk_nope_dim;

    const auto token_num = hidden_state.shape(0);
    const auto dtype     = hidden_state.dtype();

    Tensor q;

    if (w.q_proj.weight) {
        q = linear_.Forward(hidden_state, w.q_proj);
        sync_check_cuda_error();
    }
    else {
        Tensor q_a = linear_.Forward(hidden_state, w.q_a_proj);
        sync_check_cuda_error();

        invokeRMSNorm(q_a, q_a, w.q_a_layernorm, model_param_.norm_eps, stream_);
        sync_check_cuda_error();

        q = linear_.Forward(q_a, w.q_b_proj);
        sync_check_cuda_error();
    }

    Tensor kv_a_k_pe = linear_.Forward(hidden_state, w.kv_a_proj);
    sync_check_cuda_error();

    auto kv_a = kv_a_k_pe.slice({0, 0}, {-1, kv_lora_rank});
    invokeRMSNorm(kv_a, kv_a, w.kv_a_layernorm, model_param_.norm_eps, stream_);
    sync_check_cuda_error();

    Tensor kv_b = linear_.Forward(kv_a, w.kv_b_proj);
    sync_check_cuda_error();

    const int local_q_kv_head_num = local_head_num_ + 2 * local_kv_head_num_;

    Tensor qkv{{token_num, local_q_kv_head_num, (int)size_per_head_}, dtype, hidden_state.device()};
    MLACopyQKV(dtype,
               qkv.raw_data(),
               q.raw_data(),
               kv_a.raw_data(),
               kv_b.raw_data(),
               token_num,
               local_head_num_,
               qk_nope_dim,
               qk_rope_dim,
               kv_lora_rank,
               v_head_dim,
               stream_);
    sync_check_cuda_error();

    return qkv;
}

void UnifiedAttentionLayer::qk_norm(Tensor& qkv, const WeightType& weights)
{
    check_cuda_error(cudaEventRecord(qkv_event_, stream_));
    check_cuda_error(cudaStreamWaitEvent(aux_stream_, qkv_event_));

    TM_CHECK(model_param_.attn_bias == false) << "not implemented";

    const auto token_num = qkv.shape(0);

    auto qkv3 = qkv.view({token_num, -1, (int)size_per_head_});

    auto q = qkv3.slice({0, 0, 0}, {-1, (int)local_head_num_, -1});
    invokeRMSNormQK(q, weights.q_a_layernorm, model_param_.norm_eps, stream_);
    sync_check_cuda_error();

    auto k = qkv3.slice({0, (int)local_head_num_, 0}, {-1, (int)local_kv_head_num_, -1});
    invokeRMSNormQK(k, weights.kv_a_layernorm, model_param_.norm_eps, aux_stream_);
    sync_check_cuda_error();

    check_cuda_error(cudaEventRecord(aux_event_, aux_stream_));
    check_cuda_error(cudaStreamWaitEvent(stream_, aux_event_));
}

}  // namespace turbomind
