// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/attention.h"

#include <algorithm>
#include <functional>
#include <math.h>
#include <numeric>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/core.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/registry.h"
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/engine/block.h"
#include "src/turbomind/engine/cache_registry.h"
#include "src/turbomind/engine/request.h"

#include "src/turbomind/kernels/attention/attention.h"
#include "src/turbomind/kernels/attention/block.h"
#include "src/turbomind/kernels/attention/decoding.h"
#include "src/turbomind/kernels/attention/kv_cache_utils_v2.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/norm/rms_norm.h"

#include "src/turbomind/macro.h"

#include "src/turbomind/memory/object.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/mla_utils.h"

#include "src/turbomind/core/logger.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

namespace {
// clang-format off
struct BlockConfig {
    int head_dim_;
    int head_num_;
    int block_len_;
    int t_bits_;
    int q_bits_;
    bool share_kv_;
    int t_bits() const { return t_bits_; }
    int q_bits() const { return q_bits_; }
    int head_dim() const { return head_dim_; }
    int head_num() const { return head_num_; }
    int block_len() const { return block_len_; }
    bool is_share_kv() const { return share_kv_; }
    auto as_tuple() const noexcept {
        return std::tie(head_dim_, head_num_, block_len_, t_bits_, q_bits_, share_kv_);
    }
    friend bool operator==(const BlockConfig& a, const BlockConfig& b) {
        return a.as_tuple() == b.as_tuple();
    }
};
// clang-format on
}  // namespace

Attention::Attention(const core::AttentionConfig& cfg):
    hidden_dim(cfg.hidden_dim),
    head_dim(cfg.head_dim),
    head_num(cfg.head_num),
    kv_head_num(cfg.kv_head_num),
    kv_lora_rank(cfg.kv_lora_rank),
    q_lora_rank(cfg.q_lora_rank),
    qk_rope_dim(cfg.qk_rope_dim),
    v_head_dim(cfg.v_head_dim),
    tp_size(cfg.tp_size),
    tp_rank(cfg.tp_rank),
    data_type(cfg.data_type),
    window_size(cfg.window_size),
    causal(cfg.causal),
    output_gate(cfg.output_gate),
    softmax_scale(cfg.softmax_scale),
    use_logn_attn(cfg.use_logn_attn),
    rope(cfg.rope)
{
    init_rope_kernel_param(rope, rope_param_);
}

void Attention::register_cache(CacheRegistry& registry)
{
    Module::register_cache(registry);

    const int dtype_bits = byte_size(data_type, 8);
    const int quant_bits = registry.quant_policy() ? registry.quant_policy() : dtype_bits;

    BlockConfig   b{head_dim,
                  kv_head_num / tp_size,
                  registry.cache_block_seq_len(),
                  dtype_bits == quant_bits ? 0 : dtype_bits,
                  quant_bits,
                  head_dim == 576};
    block::Layout layout{b};

    prefix_offset_     = registry.prefix().Register(layout.layer_size(), /*alignment=*/1);
    prefix_registered_ = true;
}

void Attention::bind(Context& ctx)
{
    Module::bind(ctx);
    state_ = TM_CHECK_NOTNULL(ctx.attention_state());
    if (prefix_registered_) {
        state_->prefix_base = std::min(state_->prefix_base, prefix_offset_);
    }
}

// ---------------------------------------------------------------------------
// AttentionState
// ---------------------------------------------------------------------------

AttentionState::AttentionState(const EngineParam& engine, const Context& ctx, int phases, const Attention& exemplar):
    quant_policy{engine.quant_policy},
    cache_block_seq_len{engine.cache_block_seq_len},
    attn_cp_size{engine.attn_cp_size},
    attn_cp_rank{engine.attn_cp_rank},
    attn_dp_size{engine.attn_dp_size},
    attn_dp_rank{engine.attn_dp_rank},
    max_forward_token_num{engine.max_forward_token_num},
    is_warm_up(*ctx.is_warm_up),
    arch{getSMVersion()},
    cp_fn_ctx_(ctx.comm.d_comm, ctx.comm.d_cp_group, engine.attn_cp_size),
    rope_param_(exemplar.rope_kernel_param())
{
    const auto max_block_num = engine.max_batch_size * cdiv(engine.session_len, engine.cache_block_seq_len);

    block_ptrs_buf_         = {max_block_num, kCPUpinned};
    block_ptrs_offsets_buf_ = {engine.max_batch_size + 1, kCPUpinned};

    TM_CUDA_CHECK(cudaStreamCreateWithFlags(&aux_stream_, cudaStreamNonBlocking));
    TM_CUDA_CHECK(cudaEventCreateWithFlags(&qkv_event_, cudaEventDisableTiming));
    TM_CUDA_CHECK(cudaEventCreateWithFlags(&aux_event_, cudaEventDisableTiming));

    const int bsz = engine.max_batch_size;

    if (rope_param_.type == RopeType::kDynamic) {
        rope_base_buf_ = {bsz + 1, kCPUpinned};
    }
    if (rope_param_.mrope_mode != MropeMode::kNone) {
        mrope_default_buf_ = Buffer_<int>{std::max(bsz, 3), kDEVICE};
        Clear(mrope_default_buf_);
    }
    const int max_blocks = bsz * cdiv(engine.session_len, cache_block_seq_len);
    batches_.resize(phases);
    for (auto& d : batches_) {
        d.block_ptrs         = {max_blocks + 16, kDEVICE};
        d.block_ptrs_offsets = {bsz + 1, kDEVICE};
        if (rope_param_.type == RopeType::kDynamic) {
            d.rope_base = empty_like(rope_base_buf_, kDEVICE);
        }
    }

    // Eagerly initialize workspace buffers (was previously lazy in Init())
    {
        const int tp_size        = exemplar.tp_size;
        const int local_head_num = exemplar.head_num / tp_size;
        const int size_per_head  = exemplar.head_dim;

        TM_CHECK_EQ(exemplar.head_num % tp_size, 0) << exemplar.head_num << " " << tp_size;
        TM_CHECK_EQ(exemplar.head_num % exemplar.kv_head_num, 0) << exemplar.head_num << " " << exemplar.kv_head_num;

        ssize_t   workspace_tokens = kMaxWorkspaceTokens;
        Allocator alloc            = core::Context::device_alloc();
        if (attn_cp_size > 1) {
            alloc = GetSymmAllocator(ctx.comm.d_comm);
            workspace_tokens += max_forward_token_num;
        }

        partial_O_  = Tensor_<float>({workspace_tokens, local_head_num, size_per_head}, kDEVICE);
        partial_ML_ = Tensor_<float>({attn_cp_size, workspace_tokens, local_head_num, 2}, alloc);
        split_cnt_  = Tensor_<int>({workspace_tokens}, kDEVICE);

        Clear(split_cnt_.buffer());
    }
}

AttentionState::~AttentionState()
{
    TM_CUDA_CHECK(cudaEventDestroy(aux_event_));
    TM_CUDA_CHECK(cudaEventDestroy(qkv_event_));
    TM_CUDA_CHECK(cudaStreamDestroy(aux_stream_));

    aux_event_ = qkv_event_ = {};
    aux_stream_             = {};
}

// ---------------------------------------------------------------------------
// BatchOp handling
// ---------------------------------------------------------------------------

static void init_dynamic_ntk(Sequence& cache, const core::RopeConfig& rope)
{
    cache.rope_base = rope.base;
    if (auto scaling_factor = rope.factor; scaling_factor > 1.f) {
        const auto max_seq_len = cache.prompt_len;
        const auto max_pos_emb = rope.max_position_embeddings;
        if (max_seq_len > max_pos_emb) {
            scaling_factor = scaling_factor * max_seq_len / max_pos_emb - (scaling_factor - 1);
            cache.rope_base *= powf(scaling_factor, rope.dim / (rope.dim - 2.f));
            // clang-format off
            TM_LOG_INFO("{} rope_scaling_factor: {}, rope_theta = {}",
                        cache.req->id, scaling_factor, cache.rope_base);
            // clang-format on
        }
    }
}

void Attention::Run(BatchOp op, int phase, TensorMap& env)
{
    if (op != BatchOp::kSetup) {
        // Engine progress since the last fill: the batch-level slot is stale.
        state_->AdvanceWave();
    }

    if (op == BatchOp::kAdd) {
        Buffer_<Sequence*> rc = env.at("requests").buffer();
        if (rope_param_.type == RopeType::kDynamic) {
            for (int i = 0; i < rc.size(); ++i) {
                init_dynamic_ntk(*rc[i], rope);
            }
        }
    }
    else if (op == BatchOp::kSetup) {
        Setup(phase, env);
    }
    else if (op == BatchOp::kPrepare) {
        auto& d              = state_->batch(phase);
        d.finished           = env.at("finished").buffer().borrow();
        d.q_offsets          = env.at("q_offsets").buffer().borrow();
        d.k_offsets          = env.at("k_offsets").buffer().borrow();
        d.readonly_block_num = env.at("readonly_block_num").buffer().borrow();

        // Borrow the global mask owned by LanguageModel (pointer only; its content is
        // built at Forward time) and resolve this rank's token offset within it.
        d.token_mask      = env.at("token_mask").buffer().borrow();
        d.token_mask_base = 0;
        if (state_->attn_dp_size > 1) {
            const auto& local_token_num = env.at("batch").data<BatchData*>()[0]->local_token_num;
            TM_CHECK_EQ((int)local_token_num.size(), state_->attn_dp_size);
            d.token_mask_base =
                std::accumulate(local_token_num.begin(), local_token_num.begin() + state_->attn_dp_rank, 0);
        }
    }
}

void Attention::Setup(int phase, TensorMap& env)
{
    // Versioned fill guard: the first module of this kSetup wave fills the
    // batch-level slot; the rest see the stamp and skip.
    auto& d = state_->batch(phase);
    if (d.wave == state_->wave()) {
        return;
    }

    Buffer_<Sequence*> rc = env.at("requests").buffer();

    const int bsz = rc.size();

    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    {  /// Upload KV cache ptrs
        auto blocks  = state_->block_ptrs_buf_.data();
        auto offsets = state_->block_ptrs_offsets_buf_.data();

        offsets[0] = 0;
        for (int i = 0; i < rc.size(); ++i) {
            const auto& r = *rc[i];
            for (const auto& h : r.block_ids) {
                const CacheBlock& cb = *h->prefix;
                TM_CHECK_NOTNULL(cb.allocation.a);
                *blocks++ = cb.base(0) + state_->prefix_base;
            }
            offsets[i + 1] = offsets[i] + r.block_ids.size();
        }

        copy(state_->block_ptrs_buf_, state_->block_ptrs_offsets_buf_[bsz], d.block_ptrs);
        copy(state_->block_ptrs_offsets_buf_, bsz + 1, d.block_ptrs_offsets);
    }

    /// prepare Q/K stats for decode/prefill
    d.decode = d.prefill = {};

    d.decode.n  = std::find_if(rc.begin(), rc.end(), [](auto r) { return r->input_len > 1; }) - rc.begin();
    d.prefill.n = bsz - d.decode.n;

    for (int i = 0; i < bsz; ++i) {
        const auto& c = *rc[i];

        auto& s = i < d.decode.n ? d.decode : d.prefill;
        s.q_sum += c.input_len;
        s.k_sum += c.history_len + c.inflight_input_len + c.input_len;
        s.q_max = std::max(s.q_max, c.input_len);
        s.k_max = std::max(s.k_max, c.history_len + c.inflight_input_len + c.input_len);
    }

    /// handling different RoPE types
    if (state_->rope_param_.type == RopeType::kDynamic) {
        for (int i = 0; i < bsz; ++i) {
            state_->rope_base_buf_[i] = rc[i]->rope_base;
        }
        copy(state_->rope_base_buf_, bsz, d.rope_base);
    }
    else if (state_->rope_param_.mrope_mode != MropeMode::kNone) {
        auto* mrope_length           = env.try_("mrope_length");
        auto* mrope_position_delta   = env.try_("mrope_position_delta");
        auto* mrope_position_offsets = env.try_("mrope_position_offsets");
        auto* mrope_position_ids     = env.try_("mrope_position_ids");
        if (mrope_length || mrope_position_delta || mrope_position_offsets || mrope_position_ids) {
            TM_CHECK(mrope_length) << "MRoPE requires native vision-produced mrope_length";
            TM_CHECK(mrope_position_delta) << "MRoPE requires native vision-produced mrope_position_delta";
            TM_CHECK(mrope_position_offsets) << "MRoPE requires native vision-produced mrope_position_offsets";
            TM_CHECK(mrope_position_ids) << "MRoPE requires native vision-produced mrope_position_ids";

            d.mrope_length           = mrope_length->buffer().borrow();
            d.mrope_position_delta   = mrope_position_delta->buffer().borrow();
            d.mrope_position_offsets = mrope_position_offsets->buffer().borrow();
            d.mrope_position_ids     = mrope_position_ids->buffer().borrow();
        }
        else {
            d.mrope_length           = state_->mrope_default_buf_.borrow();
            d.mrope_position_delta   = state_->mrope_default_buf_.borrow();
            d.mrope_position_offsets = state_->mrope_default_buf_.borrow();
            d.mrope_position_ids     = state_->mrope_default_buf_.borrow();
        }
    }

    d.wave = state_->wave();
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

void Attention::Forward(int phase, const Tensor& input, Ref<Tensor> output)
{
    TM_FUNCTION_SCOPE();

    /////////////////////////////////////////////
    /// parse inputs
    const int token_num = input.shape(0);

    if (token_num == 0) {
        return;
    }

    TM_LOG_DEBUG("token_num=%d", token_num);

    Tensor qkv;

    if (w_qkv && w_qkv->output_dim) {
        // [token_num, hidden_dim] -> [token_num, local_q_kv_head_num, head_dim]
        TM_SCOPE_CALL(w_qkv->Forward(input, qkv));

        qk_norm(qkv);
    }
    else {
        qkv = forward_mla(input);
    }

    TM_DEBUG_TENSOR(qkv, "qkv", 3);

    auto invoke = [&](auto t) -> Tensor {
        using T = decltype(t);
        return core_attention<T>(qkv, phase);
    };

    Tensor attn = [&]() -> Tensor { TM_DISPATCH_PRIMARY_DTYPES_RET(qkv.dtype(), invoke); }();

    // Apply sigmoid gating: attn *= sigmoid(gate)
    // Gate is stored at the end of each token's QKV: [Q|K|V|Gate]
    if (output_gate) {
        const int  tp_size           = this->tp_size;
        const int  local_head_num    = head_num / tp_size;
        const int  local_kv_head_num = kv_head_num / tp_size;
        const int  size_per_head     = head_dim;
        const int  q_count           = qkv.shape(0);
        const int  attn_dim          = local_head_num * size_per_head;
        const int  gate_offset       = (local_head_num + 2 * local_kv_head_num) * size_per_head;
        const int  qkv_stride        = (2 * local_head_num + 2 * local_kv_head_num) * size_per_head;
        const auto stream            = core::Context::stream().handle();
        invokeSigmoidGateMultiply(attn.raw_data(),
                                  (const char*)qkv.raw_data() + gate_offset * byte_size(qkv.dtype(), 1),
                                  attn_dim,
                                  qkv_stride,
                                  q_count,
                                  qkv.dtype(),
                                  stream);
        TM_CUDA_CHECK(cudaGetLastError());
    }

    TM_DEBUG_TENSOR(attn, "attn", 3);

    //////////////////////////////////////////////
    /// output gemm <Bs,HD> -> <Bs,HD>
    TM_SCOPE_CALL(wo->Forward(attn, output));
}

template<class T>
Tensor Attention::core_attention(Tensor& qkv, int phase)
{
    TM_FUNCTION_SCOPE();
    const int tp_size           = this->tp_size;
    const int local_head_num    = head_num / tp_size;
    const int local_kv_head_num = kv_head_num / tp_size;
    const int size_per_head     = head_dim;

    const auto device = qkv.device();
    const auto dtype  = qkv.dtype();

    auto& st = *state_;
    auto& d  = st.batch(phase);

    const int batch_size = d.decode.n + d.prefill.n;
    const int q_count    = qkv.shape(0);

    TM_CHECK_EQ(d.prefill.q_sum + d.decode.n, q_count);

    const int local_q_kv_head_num = local_head_num + 2 * local_kv_head_num;

    Tensor attn{{q_count, local_head_num * size_per_head}, dtype, device};

    const bool is_mla = this->is_mla();

    Tensor tmp_kv{{local_kv_head_num, is_mla ? 1 : 2, d.prefill.k_sum + MAX_CTA_S, size_per_head}, dtype, device};

    auto CreateParams = [&](int offset, AttentionBatch::Stat stat, int max_kv_splits, cudaStream_t stream) {
        AttentionParams<T> params{};

        // Batch offset for `out` and `q` are computed inside the kernel
        params.out = (T*)attn.raw_data();

        params.q = (T*)qkv.raw_data();
        params.k = params.q + local_head_num * size_per_head;
        if (is_mla) {
            params.v      = params.k;
            params.stride = (local_head_num + 1 * local_kv_head_num) * size_per_head;
        }
        else {
            params.v = params.k + local_kv_head_num * size_per_head;
            // When attn_output_gate, QKV layout is [Q|K|V|Gate] per token
            // stride must account for the extra gate portion at the end
            if (output_gate) {
                params.stride = (2 * local_head_num + 2 * local_kv_head_num) * size_per_head;
            }
            else {
                params.stride = (local_head_num + 2 * local_kv_head_num) * size_per_head;
            }
        }

        if (!is_mla && w_qkv && w_qkv->bias) {
            params.q_bias = (T*)w_qkv->bias.data_or<T>(nullptr);
            params.k_bias = params.q_bias + local_head_num * size_per_head;
            params.v_bias = params.k_bias + local_kv_head_num * size_per_head;
        }

        params.batch_size = stat.n;

        params.token_num = stat.q_sum;
        params.max_q_len = stat.q_max;
        params.max_k_len = stat.k_max;

        TM_CHECK_LE(prefix_offset_, INT_MAX);

        // decode only
        params.block_iter_params = BlockIteratorParams{(char**)d.block_ptrs.data(),  //
                                                       d.block_ptrs_offsets.data() + offset,
                                                       (int)(prefix_offset_ - st.prefix_base),
                                                       st.cache_block_seq_len};

        // prefill only
        if (is_mla) {
            params.linear_iter_params = LinearIteratorParams{
                tmp_kv.raw_data(),           // flattened KV
                stat.k_sum * size_per_head,  // stride to next head
                0                            // stride from K to V
            };
        }
        else {
            params.linear_iter_params = LinearIteratorParams{
                tmp_kv.raw_data(),               // flattened KV
                stat.k_sum * size_per_head * 2,  // stride to next head
                stat.k_sum * size_per_head       // stride from K to V
            };
        }

        params.finished = d.finished.data() + offset;
        // decode rows: base; prefill rows: + decode.n (this rank's slice of the global mask)
        params.token_mask         = d.token_mask.data() + d.token_mask_base + offset;
        params.cu_q_len           = d.q_offsets.data() + offset;
        params.cu_k_len           = d.k_offsets.data() + offset;
        params.readonly_block_num = d.readonly_block_num.data() + offset;

        params.num_heads     = local_head_num;
        params.num_kv_heads  = local_kv_head_num;
        params.size_per_head = size_per_head;

        double scaling = 1.;
        if (softmax_scale) {  // model predefined softmax scale
            scaling *= softmax_scale;
        }
        else {  // default value
            scaling /= std::sqrt((float)params.size_per_head);
        }
        params.inv_sqrt_dh = scaling * std::log2(std::exp(1.));

        params.sinks       = sinks ? sinks.data_or((T*)nullptr) : (T*)nullptr;
        params.scale_sinks = scaling;

        params.window_size = window_size;
        if (!params.window_size) {
            params.window_size = 256 << 20;  // 256 M
        }

        params.rope_param = rope_param_;
        if (rope_param_.type == RopeType::kDynamic) {
            params.rope_param.base = d.rope_base.data() + offset;
        }
        if (rope_param_.mrope_mode != MropeMode::kNone) {
            params.rope_param.mrope.position_delta   = d.mrope_position_delta.data() + offset;
            params.rope_param.mrope.position_offsets = d.mrope_position_offsets.data() + offset;
            params.rope_param.mrope.length           = d.mrope_length.data() + offset;
            params.rope_param.mrope.position_ids     = d.mrope_position_ids.data();
        }

        // logn attn
        params.use_logn_attn           = use_logn_attn;
        params.max_position_embeddings = rope.max_position_embeddings;

        // Decoding use only for now
        params.split_cnt   = st.split_cnt_.data();
        params.partial_ML  = st.partial_ML_.data();
        params.partial_O   = st.partial_O_.data();
        params.max_split_k = std::min(std::max(1, st.kMaxWorkspaceTokens / params.token_num), max_kv_splits);

        // context parallel
        params.cp_rank = st.attn_cp_rank;
        params.cp_size = st.attn_cp_size;
        if (params.cp_size > 1) {
            params.cp_size = cutlass::FastDivmod(params.cp_size);

            // update ML,O offset if both prefill and decode present
            const int offset_ML_stage = st.attn_cp_size * (offset ? st.kMaxWorkspaceTokens * local_head_num * 2 : 0);
            const int offset_ML_rank  = params.cp_rank * params.token_num * local_head_num * params.max_split_k * 2;
            const int offset_O        = offset ? st.kMaxWorkspaceTokens * local_head_num * size_per_head : 0;

            params.partial_ML = st.partial_ML_.data() + offset_ML_stage + offset_ML_rank;
            params.partial_O  = st.partial_O_.data() + offset_O;
            params.offset_q   = offset;

            // postprocess func
            params.cp_fn             = CpPost;
            params.cp_fn_ctx         = (void*)&st.cp_fn_ctx_;
            st.cp_fn_ctx_.cp_rank    = params.cp_rank;
            st.cp_fn_ctx_.count      = params.token_num * local_head_num * params.max_split_k * 2;
            st.cp_fn_ctx_.partial_ML = st.partial_ML_.data() + offset_ML_stage;
            st.cp_fn_ctx_.stream     = stream;
        }

        params.arch   = st.arch;
        params.stream = stream;

        params.quant_policy = st.quant_policy;
        return params;
    };

    const cudaStream_t stream = core::Context::stream().handle();

    cudaStream_t pf_stream = stream;
    cudaStream_t dc_stream = stream;

    if (d.decode.n && d.prefill.n) {
        pf_stream = st.aux_stream_;
        TM_CUDA_CHECK(cudaEventRecord(st.qkv_event_, stream));
        TM_CUDA_CHECK(cudaStreamWaitEvent(pf_stream, st.qkv_event_));
    }

    if (d.prefill.n && !st.is_warm_up) {
        const int offset = d.decode.n;
        // We are executing prefill & decoding kernels concurrently, but only have 1 workspace
        // disable split kv for prefill for now
        auto params = CreateParams(offset, d.prefill, 1, pf_stream);
        if constexpr (sizeof(T) == 2) {
            invokeProcessKV_v2_(params);
            TM_CUDA_CHECK(cudaGetLastError());

            /// TODO: skip flattening for `sm_80`
            invokeFlattenKV_v2_(params, d.prefill.k_sum);
            TM_CUDA_CHECK(cudaGetLastError());

            dispatchAttention(params);
            TM_CUDA_CHECK(cudaGetLastError());
        }
    }

    if (d.decode.n && !st.is_warm_up) {
        auto params = CreateParams(0, d.decode, st.kMaxKVSplits, dc_stream);
        if constexpr (sizeof(T) == 2) {
            dispatchDecoding<T>(params);
            TM_CUDA_CHECK(cudaGetLastError());
        }
    }

    if (d.decode.n && d.prefill.n) {
        TM_CUDA_CHECK(cudaEventRecord(st.aux_event_, pf_stream));
        TM_CUDA_CHECK(cudaStreamWaitEvent(stream, st.aux_event_));
    }

    if (st.is_warm_up) {
        st.rng_.set_stream(stream);
        st.rng_.GenerateUniform(attn.data<T>(), attn.size(), .02f, -.01f);
    }

    return attn;
}

Tensor Attention::forward_mla(const Tensor& hidden_state)
{
    TM_FUNCTION_SCOPE();

    const int tp_size           = this->tp_size;
    const int local_head_num    = head_num / tp_size;
    const int local_kv_head_num = kv_head_num / tp_size;
    const int size_per_head     = head_dim;

    const auto token_num = hidden_state.shape(0);
    const auto dtype     = hidden_state.dtype();

    const int q_lora_rank  = q_a_proj->output_dim;
    const int kv_lora_rank = kv_a_layernorm->weight.size();
    const int qk_rope_dim  = kv_a_proj->output_dim - kv_lora_rank;

    Tensor q;

    const auto stream = core::Context::stream().handle();

    if (q_proj && q_proj->weight) {
        TM_SCOPE_CALL(q_proj->Forward(hidden_state, q));
    }
    else {
        Tensor q_a;
        TM_SCOPE_CALL(q_a_proj->Forward(hidden_state, q_a));

        invokeRMSNorm(q_a, q_a, q_a_layernorm->weight, q_a_layernorm->norm_eps_, q_a_layernorm->zero_centered_, stream);
        TM_CUDA_CHECK(cudaGetLastError());

        TM_SCOPE_CALL(q_b_proj->Forward(q_a, q));
    }

    Tensor kv_a_k_pe;
    TM_SCOPE_CALL(kv_a_proj->Forward(hidden_state, kv_a_k_pe));

    auto kv_a = kv_a_k_pe.slice({0, 0}, {-1, kv_lora_rank});
    invokeRMSNorm(
        kv_a, kv_a, kv_a_layernorm->weight, kv_a_layernorm->norm_eps_, kv_a_layernorm->zero_centered_, stream);
    TM_CUDA_CHECK(cudaGetLastError());

    const int local_q_kv_head_num = local_head_num + 1 * local_kv_head_num;

    Tensor qkv{{token_num, local_q_kv_head_num, size_per_head}, dtype, hidden_state.device()};
    MLACopyQKV(dtype,
               qkv.raw_data(),
               q.raw_data(),
               kv_a_k_pe.raw_data(),
               token_num,
               local_head_num,
               kv_lora_rank,
               qk_rope_dim,
               stream);
    TM_CUDA_CHECK(cudaGetLastError());

    return qkv;
}

void Attention::qk_norm(Tensor& qkv)
{
    if (!(q_norm || k_norm)) {
        return;
    }

    TM_CHECK(q_norm && k_norm);

    const int tp_size           = this->tp_size;
    const int local_head_num    = head_num / tp_size;
    const int local_kv_head_num = kv_head_num / tp_size;
    const int size_per_head     = head_dim;

    const auto stream = core::Context::stream().handle();

    const auto token_num = qkv.shape(0);

    auto qkv3 = qkv.view({token_num, -1, size_per_head});

    invokeQkRMSNorm(qkv3,
                    q_norm->weight,
                    k_norm->weight,
                    local_head_num,
                    local_kv_head_num,
                    q_norm->norm_eps_,
                    q_norm->zero_centered_,
                    stream);
    TM_CUDA_CHECK(cudaGetLastError());
}

void init_rope_kernel_param(const core::RopeConfig& rope, RopeKernelParam& rope_kernel)
{
    auto rope_type = static_cast<RopeType>(rope.type);

    rope_kernel.type         = rope_type;
    rope_kernel.dim          = rope.dim;
    rope_kernel.scale_factor = -std::log2(rope.base) / rope.dim;
    if (rope_type == RopeType::kDynamic) {
        rope_kernel.inv_factor = 1.f;
    }
    else {
        rope_kernel.inv_factor = (rope.factor != 0.f) ? 1.0 / rope.factor : 1.f;
    }

    if (rope_type == RopeType::kYarn) {
        auto&        dst = rope_kernel.yarn;
        const double PI  = 3.14159265358979323846;

        auto find_correction_dim = [&](float num_rotations) {
            return (rope.dim * std::log(rope.max_position_embeddings / (num_rotations * 2 * PI)))
                   / (2 * std::log(rope.base));
        };

        auto find_correction_range = [&](float low_rot, float high_rot, float& low, float& high) {
            low  = std::floor(find_correction_dim(low_rot));
            high = std::ceil(find_correction_dim(high_rot));
            low  = std::max(low, 0.f);
            high = std::min(high, rope.dim - 1.f);
        };

        float low, high;
        find_correction_range(rope.yarn_beta_fast, rope.yarn_beta_slow, low, high);
        if (low == high) {
            high += 0.001;
        }
        dst.ramp_inv_factor_div_2   = 1.0 / (high - low) / 2.0;
        dst.ramp_inv_factor_mul_min = 1.0 / (high - low) * low;
        dst.attention_factor        = rope.yarn_attention_factor;
    }
    else if (rope_type == RopeType::kLlama3) {
        auto& dst = rope_kernel.llama3;

        float inv_diff_freq_factor = 1.0 / (rope.llama3_high_freq_factor - rope.llama3_low_freq_factor);
        dst.alpha = rope.llama3_original_max_position_embeddings / (2 * 3.14159265358979323846) * inv_diff_freq_factor;
        dst.beta  = rope.llama3_low_freq_factor * inv_diff_freq_factor;
    }

    // mrope is layered on top of the base rope: orthogonal to rope_type.
    rope_kernel.mrope_mode = MropeMode::kNone;
    const auto& ms         = rope.mrope_section;
    const bool  has_mrope  = ms[0] > 0 || ms[1] > 0 || ms[2] > 0;
    if (has_mrope) {
        rope_kernel.mrope_mode = rope.mrope_interleaved ? MropeMode::kInterleaved : MropeMode::kChunked;
        auto& dst              = rope_kernel.mrope;
        dst.section            = {ms[0], ms[1], ms[2]};
    }
}

TM_MODULE_REGISTER_AS("AttentionWeight", Attention, core::AttentionConfig);

TM_MODULE_METHODS(Attention, ATTENTION_CHILDREN, ATTENTION_PARAMS)

}  // namespace turbomind
