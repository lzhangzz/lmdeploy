# Decouple UnifiedAttentionLayer from ModelParam and AttentionParam

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove UnifiedAttentionLayer's dependency on `ModelParam` and `AttentionParam`. Dimensional and behavioral fields are read from `AttentionWeight` at runtime. Non-dimensional fields are passed as individual constructor params. Head-count-dependent buffers use lazy initialization.

**Architecture:** `AttentionWeight` exposes all config fields as public. `UnifiedAttentionLayer` constructor shrinks to individual params (no `ModelParam`, no `AttentionParam`, no redundant `tp_size`). On first `Forward`, lazy `Init` allocates head-count-dependent buffers from the weight's fields. `Forward`, `core_attention`, and helpers read head counts, MLA params, and behavioral config from the weight.

**Tech Stack:** C++17, CUDA, ninja build, Python (spec updates)

---

### Task 1: Make AttentionWeight fields public, add softmax_scale / use_logn_attn / max_position_embeddings

**Files:**
- Modify: `src/turbomind/models/attention_weight.h`
- Modify: `src/turbomind/models/attention_weight.cc`

- [ ] **Step 1: Add fields to AttentionConfig X-macro and make AttentionWeight fields public**

Replace lines 14-91 in `src/turbomind/models/attention_weight.h` (the full struct + class body):

```cpp
struct AttentionConfig: ModuleConfig {
    AttentionConfig(): ModuleConfig{"AttentionWeight"} {}

    #define ATTENTION_FIELDS(X) \
        X(int,      hidden_dim) \
        X(int,      head_dim) \
        X(int,      head_num) \
        X(int,      kv_head_num) \
        X(int,      kv_lora_rank) \
        X(int,      q_lora_rank) \
        X(int,      qk_rope_dim) \
        X(int,      v_head_dim) \
        X(bool,     has_bias) \
        X(bool,     qk_norm) \
        X(int,      tp_size) \
        X(int,      tp_rank) \
        X(DataType, data_type) \
        X(int,      window_size, -1) \
        X(bool,     attn_sink) \
        X(bool,     attn_output_gate) \
        X(int,      rope_dim) \
        X(int,      repeat_kv) \
        X(int,      qk_nope_dim) \
        X(float,    softmax_scale, 0.f) \
        X(bool,     use_logn_attn, false) \
        X(int,      max_position_embeddings, 0)

    ATTENTION_FIELDS(TM_MEMBER)
    TM_FOR_EACH(AttentionConfig, ATTENTION_FIELDS)

    #undef ATTENTION_FIELDS
};

}  // namespace turbomind::core

namespace turbomind {

class AttentionWeight: public core::Module {
public:
    const char* type() const override { return "AttentionWeight"; }

    AttentionWeight() = default;

    AttentionWeight(const core::AttentionConfig& cfg);

    void prepare() override;

    // --- X-macro field lists ---
#define ATTENTION_WEIGHT_CHILDREN(X) \
    X(LinearWeight, w_qkv)          \
    X(LinearWeight, wo)             \
    X(LinearWeight, q_proj)         \
    X(LinearWeight, q_a_proj)       \
    X(LinearWeight, q_b_proj)       \
    X(LinearWeight, kv_a_proj)      \
    X(NormWeight,   q_norm)         \
    X(NormWeight,   k_norm)         \
    X(NormWeight,   q_a_layernorm)  \
    X(NormWeight,   kv_a_layernorm)

#define ATTENTION_WEIGHT_PARAMS(X) \
    X(sinks)

    TM_MODULE_DECLARE(AttentionWeight, ATTENTION_WEIGHT_CHILDREN, ATTENTION_WEIGHT_PARAMS)

    bool is_mla() const { return mla_.kv_lora_rank > 0; }

    // --- Config fields (public for runtime access) ---
    int      hidden_dim_{};
    int      head_dim_{};
    int      head_num_{};
    int      kv_head_num_{};
    MLAParam mla_{};
    bool     bias_{};
    bool     qk_norm_{};
    int      tp_size_{};
    int      tp_rank_{};
    DataType data_type_{};
    int      window_size_{};
    bool     sink_{};
    bool     attn_output_gate_{};
    float    softmax_scale_{};
    bool     use_logn_attn_{};
    int      max_position_embeddings_{};
};

}  // namespace turbomind
```

Changes: `window_size()` accessor removed (field is public). All private fields moved to public section. Three new fields added: `softmax_scale_`, `use_logn_attn_`, `max_position_embeddings_`. `is_mla()` kept as convenience method.

- [ ] **Step 2: Populate new fields in AttentionWeight constructor**

Replace the full constructor in `src/turbomind/models/attention_weight.cc` (lines 10-25):

```cpp
AttentionWeight::AttentionWeight(const core::AttentionConfig& cfg)
    : hidden_dim_(cfg.hidden_dim)
    , head_dim_(cfg.head_dim)
    , head_num_(cfg.head_num)
    , kv_head_num_(cfg.kv_head_num)
    , mla_(MLAParam{cfg.q_lora_rank, cfg.kv_lora_rank, cfg.qk_rope_dim, cfg.v_head_dim})
    , bias_(cfg.has_bias)
    , qk_norm_(cfg.qk_norm)
    , tp_size_(cfg.tp_size)
    , tp_rank_(cfg.tp_rank)
    , data_type_(cfg.data_type)
    , window_size_(cfg.window_size)
    , sink_(cfg.attn_sink)
    , attn_output_gate_(cfg.attn_output_gate)
    , softmax_scale_(cfg.softmax_scale)
    , use_logn_attn_(cfg.use_logn_attn)
    , max_position_embeddings_(cfg.max_position_embeddings)
{
}
```

- [ ] **Step 3: Build to verify no breakage**

Run: `cd build && ninja`
Expected: clean build (no code reads these fields externally yet, `window_size()` callers still compile because the header is only included by unified_attention_layer.cc which we haven't changed)

Wait — `weights.window_size()` is called in `unified_attention_layer.cc`. Removing the accessor breaks the build. This is expected; we fix in Task 2.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/attention_weight.h src/turbomind/models/attention_weight.cc
git commit -m "refactor(attn): make AttentionWeight fields public, add softmax_scale/use_logn_attn/max_position_embeddings"
```

---

### Task 2: Rewrite UnifiedAttentionLayer header and implementation

**Files:**
- Modify: `src/turbomind/models/llama/unified_attention_layer.h`
- Modify: `src/turbomind/models/llama/unified_attention_layer.cc`

- [ ] **Step 1: Rewrite UnifiedAttentionLayer header**

Replace the full class body in `src/turbomind/models/llama/unified_attention_layer.h` (lines 39-121):

```cpp
class UnifiedAttentionLayer {
public:
    using WeightType = AttentionWeight;

    static constexpr int kMaxKVSplits        = 128;
    static constexpr int kMaxWorkspaceTokens = 4096;

    struct ForwardParam {
        int               phase;
        Tensor            input;
        Tensor            output;
        const WeightType* weights;
        int               layer_id;
    };

    ~UnifiedAttentionLayer();

    UnifiedAttentionLayer(float                   norm_eps,
                          int                     quant_policy,
                          const std::vector<int>& layer_types,
                          int                     layer_num,
                          const RopeParam&        rope,
                          int                     cache_block_seq_len,
                          const EngineParam&      engine,
                          const Context&          context,
                          int                     phases,
                          bool                    init);

    void Run(BatchOp op, int phase, TensorMap& env);

    void Forward(ForwardParam p);

private:
    void Init(const ForwardParam& p);

    void Setup(int phase, TensorMap& env);

    Tensor forward_mla(const Tensor& hidden_state, const WeightType& weights);

    /// TODO: dropping the `T` here requires deep refactor of attention dispatch
    template<class T>
    Tensor core_attention(Tensor& qkv, const ForwardParam& p, const WeightType& weights);

    void qk_norm(Tensor& qkv, const WeightType& weights);

private:
    // Stored config (not in weight)
    float       norm_eps_;
    int         quant_policy_;
    RopeParam   rope_;
    int         cache_block_seq_len_;

    const EngineParam engine_param_;
    const Context&    context_;

    int& is_warm_up_;
    bool init_{};
    bool initialized_ = false;

    LlamaLinear& linear_;
    const int    arch_{};

    cudaStream_t aux_stream_;
    cudaStream_t qkv_event_;
    cudaStream_t aux_event_;

    RNG rng_;

    RopeKernelParam rope_param_{};

    std::vector<std::shared_ptr<AttentionData>> data_;

    std::vector<int> cache_layer_ids_;

    ///////////////////////////////////////////////////////
    /// temp runtime buffers (lazily initialized)
    Tensor_<float> partial_O_;
    Tensor_<float> partial_ML_;
    Tensor_<int>   split_cnt_;
    Tensor         tmp_attn_;
    ///////////////////////////////////////////////////////

    Buffer_<float> rope_base_buf_;
    Buffer_<int>   mrope_position_delta_buf_;
    Buffer_<int>   mrope_length_buf_;

    CpPostContext cp_fn_ctx_;  // context parallel
};
```

Changes from current:
- Constructor: removes `ModelParam`, `AttentionParam`, redundant `tp_size`; adds individual params
- Added `Init(const ForwardParam& p)` private method
- Removed: `head_num_`, `kv_head_num_`, `size_per_head_`, `hidden_units_`, `local_head_num_`, `local_kv_head_num_`, `model_param_`, `param_`
- Added: `norm_eps_`, `quant_policy_`, `rope_`, `cache_block_seq_len_`, `init_`, `initialized_`

- [ ] **Step 2: Rewrite the constructor**

Replace the full constructor in `src/turbomind/models/llama/unified_attention_layer.cc` (lines 94-181):

```cpp
UnifiedAttentionLayer::UnifiedAttentionLayer(float                   norm_eps,
                                             int                     quant_policy,
                                             const std::vector<int>& layer_types,
                                             int                     layer_num,
                                             const RopeParam&        rope,
                                             int                     cache_block_seq_len,
                                             const EngineParam&      engine,
                                             const Context&          ctx,
                                             int                     phases,
                                             bool                    init):
    norm_eps_(norm_eps),
    quant_policy_(quant_policy),
    rope_(rope),
    cache_block_seq_len_(cache_block_seq_len),
    engine_param_(engine),
    cp_fn_ctx_(ctx.comm.d_comm, ctx.comm.d_cp_group),
    is_warm_up_{*ctx.is_warm_up},
    init_(init),
    context_(ctx),
    linear_(*ctx.linear),
    arch_(getSMVersion())
{
    init_rope_kernel_param(rope_, rope_param_);

    std::vector<int> types = layer_types;
    types.resize(layer_num);
    cache_layer_ids_.resize(types.size(), -1);
    int next_cache_id = 0;
    for (size_t i = 0; i < types.size(); ++i) {
        if (types[i] == 0) {
            cache_layer_ids_[i] = next_cache_id++;
        }
    }

    check_cuda_error(cudaStreamCreateWithFlags(&aux_stream_, cudaStreamNonBlocking));
    check_cuda_error(cudaEventCreateWithFlags(&qkv_event_, cudaEventDisableTiming));
    check_cuda_error(cudaEventCreateWithFlags(&aux_event_, cudaEventDisableTiming));

    const int bsz = engine.max_batch_size;

    if (rope_param_.type == RopeType::kDynamic) {
        rope_base_buf_ = {bsz + 1, kCPUpinned};
    }
    else if (rope_param_.type == RopeType::kMrope) {
        mrope_position_delta_buf_ = {bsz, kCPUpinned};
        mrope_length_buf_         = {bsz, kCPUpinned};
    }
    const int max_blocks = bsz * cdiv(engine.session_len, cache_block_seq_len_);
    for (int i = 0; i < phases; ++i) {
        auto& d               = data_.emplace_back(std::make_shared<AttentionData>());
        d->block_ptrs         = {max_blocks + 16, kDEVICE};
        d->block_ptrs_offsets = {bsz + 1, kDEVICE};
        if (rope_param_.type == RopeType::kDynamic) {
            d->rope_base = empty_like(rope_base_buf_, kDEVICE);
        }
        else if (rope_param_.type == RopeType::kMrope) {
            /// TODO: total space for `mrope_position_ids` can be reduced to (max_fwd_tokens, 3)
            d->mrope_position_ids    = {{bsz, engine.session_len, 3}, kDEVICE};
            d->mrope_position_delta  = empty_like(mrope_position_delta_buf_, kDEVICE);
            d->mrope_length          = empty_like(mrope_length_buf_, kDEVICE);
            rope_param_.mrope.stride = d->mrope_position_ids.stride(0);
        }
    }
}
```

- [ ] **Step 3: Add lazy Init method**

Add after the constructor, before `Run`:

```cpp
void UnifiedAttentionLayer::Init(const ForwardParam& p)
{
    const auto& w = *p.weights;

    const int tp_size          = w.tp_size_;
    const int local_head_num   = w.head_num_ / tp_size;
    const int local_kv_head_num = w.kv_head_num_ / tp_size;
    const int size_per_head    = w.head_dim_;

    TM_CHECK_EQ(w.head_num_ % tp_size, 0) << w.head_num_ << " " << tp_size;
    TM_CHECK_EQ(w.head_num_ % w.kv_head_num_, 0) << w.head_num_ << " " << w.kv_head_num_;

    ssize_t   workspace_tokens = kMaxWorkspaceTokens;
    Allocator alloc            = core::Context::device_alloc();
    if (engine_param_.attn_cp_size > 1) {
        alloc = GetSymmAllocator(context_.comm.d_comm);
        workspace_tokens += engine_param_.max_forward_token_num;
    }
    // partial_O layout:
    //   w/  cp, decode(q, h, k, 2) + prefill(q, h, 1, 2)
    //   w/o cp, decode(q, h, k, 2)
    partial_O_  = Tensor_<float>({workspace_tokens, local_head_num, size_per_head}, kDEVICE);
    partial_ML_ = Tensor_<float>({engine_param_.attn_cp_size, workspace_tokens, local_head_num, 2}, alloc);
    split_cnt_  = Tensor_<int>({workspace_tokens}, kDEVICE);
    if (init_) {
        const int dim = local_head_num * size_per_head;
        tmp_attn_     = Tensor{{engine_param_.max_forward_token_num, dim}, w.data_type_, kDEVICE};
    }

    Clear(split_cnt_.buffer());

    initialized_ = true;
}
```

- [ ] **Step 4: Update Run — replace param_.rope with rope_**

In `Run`, replace:
```cpp
                init_dynamic_ntk(*rc[i], param_.rope);
```
With:
```cpp
                init_dynamic_ntk(*rc[i], rope_);
```

This is the only change in `Run`. The `kSetup` and `kPrepare` branches are unchanged.

- [ ] **Step 5: Rewrite Forward**

Replace the `Forward` method (lines 300-378):

```cpp
void UnifiedAttentionLayer::Forward(ForwardParam p)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!initialized_) {
        Init(p);
    }

    /////////////////////////////////////////////
    /// parse inputs
    const int token_num = p.input.shape(0);

    if (token_num == 0) {
        return;
    }

    const int layer_id = p.layer_id;

    const auto& weights = *p.weights;

    TM_LOG_DEBUG("layer=%d, token_num=%d", layer_id, token_num);

    Tensor qkv;


    auto& d = *data_.at(p.phase);

    if (weights.w_qkv && weights.w_qkv->output_dim) {
        // [token_num, hidden_dim] -> [token_num, local_q_kv_head_num, head_dim]
        qkv = linear_.Forward(p.input, *weights.w_qkv);
        sync_check_cuda_error();

        if (weights.qk_norm_) {
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


    // Apply sigmoid gating: attn *= sigmoid(gate)
    // Gate is stored at the end of each token's QKV: [Q|K|V|Gate]
    if (weights.attn_output_gate_) {
        const int  tp_size          = weights.tp_size_;
        const int  local_head_num   = weights.head_num_ / tp_size;
        const int  local_kv_head_num = weights.kv_head_num_ / tp_size;
        const int  size_per_head    = weights.head_dim_;

        const int  q_count     = qkv.shape(0);
        const int  attn_dim    = local_head_num * size_per_head;
        const int  gate_offset = (local_head_num + 2 * local_kv_head_num) * size_per_head;
        const int  qkv_stride  = (2 * local_head_num + 2 * local_kv_head_num) * size_per_head;
        const auto stream      = core::Context::stream().handle();
        invokeSigmoidGateMultiply(attn.raw_data(),
                                  (const char*)qkv.raw_data() + gate_offset * byte_size(qkv.dtype(), 1),
                                  attn_dim,
                                  qkv_stride,
                                  q_count,
                                  qkv.dtype(),
                                  stream);
        sync_check_cuda_error();
    }

    TM_DEBUG_TENSOR(attn, Concat("attn", layer_id), 3);

    //////////////////////////////////////////////
    /// output gemm <Bs,HD> -> <Bs,HD>
    (void)linear_.Forward(attn, *weights.wo, p.output);
    sync_check_cuda_error();
}
```

Changes: `model_param_.qk_norm` -> `weights.qk_norm_`, `model_param_.attn_output_gate` -> `weights.attn_output_gate_`. Gate block computes local head counts from weight.

- [ ] **Step 6: Rewrite core_attention**

Replace the full `core_attention` method (lines 380-594). This is the largest method. Every `local_head_num_` becomes local `local_head_num`, `local_kv_head_num_` becomes local `local_kv_head_num`, `size_per_head_` becomes local `size_per_head`. `model_param_.*` and `param_.*` references are replaced per the table in the spec.

```cpp
template<class T>
Tensor UnifiedAttentionLayer::core_attention(Tensor& qkv, const ForwardParam& p, const WeightType& weights)
{
    const auto device = qkv.device();
    const auto dtype  = qkv.dtype();

    auto& d = *data_.at(p.phase);

    const int tp_size          = weights.tp_size_;
    const int local_head_num   = weights.head_num_ / tp_size;
    const int local_kv_head_num = weights.kv_head_num_ / tp_size;
    const int size_per_head    = weights.head_dim_;

    const int batch_size = d.decode.n + d.prefill.n;
    const int q_count    = qkv.shape(0);

    TM_CHECK_EQ(d.prefill.q_sum + d.decode.n, q_count);

    const int local_q_kv_head_num = local_head_num + 2 * local_kv_head_num;

    Tensor attn;
    if (tmp_attn_) {
        attn = tmp_attn_.slice(0, q_count);
    }
    else {
        attn = {{q_count, local_head_num * size_per_head}, dtype, device};
    }

    const bool is_mla = weights.is_mla();

    Tensor tmp_kv{
        {local_kv_head_num, is_mla ? 1 : 2, d.prefill.k_sum + MAX_CTA_S, size_per_head}, dtype, device};

    const int cache_layer_id = cache_layer_ids_[p.layer_id];

    auto CreateParams = [&](int offset, AttentionData::Stat stat, int max_kv_splits, cudaStream_t stream) {
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
            if (weights.attn_output_gate_) {
                params.stride = (2 * local_head_num + 2 * local_kv_head_num) * size_per_head;
            }
            else {
                params.stride = (local_head_num + 2 * local_kv_head_num) * size_per_head;
            }
        }

        if (!is_mla && weights.w_qkv && weights.w_qkv->bias) {
            params.q_bias = (T*)weights.w_qkv->bias.data_or<T>(nullptr);
            params.k_bias = params.q_bias + local_head_num * size_per_head;
            params.v_bias = params.k_bias + local_kv_head_num * size_per_head;
        }

        params.batch_size = stat.n;

        params.token_num = stat.q_sum;
        params.max_q_len = stat.q_max;
        params.max_k_len = stat.k_max;

        // decode only
        params.block_iter_params = BlockIteratorParams{(char**)d.block_ptrs.data(),  //
                                                       d.block_ptrs_offsets.data() + offset,
                                                       cache_layer_id,
                                                       cache_block_seq_len_};

        // prefill only
        if (is_mla) {
            params.linear_iter_params = LinearIteratorParams{
                tmp_kv.raw_data(),            // flattened KV
                stat.k_sum * size_per_head,  // stride to next head
                0                             // stride from K to V
            };
        }
        else {
            params.linear_iter_params = LinearIteratorParams{
                tmp_kv.raw_data(),                // flattened KV
                stat.k_sum * size_per_head * 2,  // stride to next head
                stat.k_sum * size_per_head       // stride from K to V
            };
        }

        params.finished = d.finished.data() + offset;
        params.cu_q_len = d.q_offsets.data() + offset;
        params.cu_k_len = d.k_offsets.data() + offset;

        params.num_heads     = local_head_num;
        params.num_kv_heads  = local_kv_head_num;
        params.size_per_head = size_per_head;
        params.layer_id      = cache_layer_id;

        double scaling = 1.;
        if (weights.softmax_scale_) {  // model predefined softmax scale
            scaling *= weights.softmax_scale_;
        }
        else {  // default value
            scaling /= std::sqrt((float)params.size_per_head);
        }
        params.inv_sqrt_dh = scaling * std::log2(std::exp(1.));

        params.sinks       = weights.sinks ? weights.sinks.data_or((T*)nullptr) : (T*)nullptr;
        params.scale_sinks = scaling;

        params.window_size = weights.window_size_;
        if (!params.window_size) {
            params.window_size = 256 << 20;  // 256 M
        }

        params.rope_param = rope_param_;
        if (rope_param_.type == RopeType::kDynamic) {
            params.rope_param.base = d.rope_base.data() + offset;
        }
        else if (rope_param_.type == RopeType::kMrope) {
            params.rope_param.mrope.position_ids   = d.mrope_position_ids.data() + offset * rope_param_.mrope.stride;
            params.rope_param.mrope.position_delta = d.mrope_position_delta.data() + offset;
            params.rope_param.mrope.length         = d.mrope_length.data() + offset;
        }

        // logn attn
        params.use_logn_attn           = weights.use_logn_attn_;
        params.max_position_embeddings = weights.max_position_embeddings_;

        // Decoding use only for now
        params.split_cnt   = split_cnt_.data();
        params.partial_ML  = partial_ML_.data();
        params.partial_O   = partial_O_.data();
        params.max_split_k = std::min(std::max(1, kMaxWorkspaceTokens / params.token_num), max_kv_splits);

        // context parallel
        params.cp_rank = engine_param_.attn_cp_rank;
        params.cp_size = engine_param_.attn_cp_size;
        if (params.cp_size > 1) {
            params.cp_size = cutlass::FastDivmod(params.cp_size);

            // update ML,O offset if both prefill and decode present
            const int offset_ML_stage =
                engine_param_.attn_cp_size * (offset ? kMaxWorkspaceTokens * local_head_num * 2 : 0);
            const int offset_ML_rank = params.cp_rank * params.token_num * local_head_num * params.max_split_k * 2;
            const int offset_O       = offset ? kMaxWorkspaceTokens * local_head_num * size_per_head : 0;

            params.partial_ML = partial_ML_.data() + offset_ML_stage + offset_ML_rank;
            params.partial_O  = partial_O_.data() + offset_O;
            params.offset_q   = offset;

            // postprocess func
            params.cp_fn          = CpPost;
            params.cp_fn_ctx      = (void*)&cp_fn_ctx_;
            cp_fn_ctx_.cp_rank    = params.cp_rank;
            cp_fn_ctx_.count      = params.token_num * local_head_num * params.max_split_k * 2;
            cp_fn_ctx_.partial_ML = partial_ML_.data() + offset_ML_stage;
            cp_fn_ctx_.stream     = stream;
        }

        params.arch   = arch_;
        params.stream = stream;

        params.quant_policy = quant_policy_;
        return params;
    };

    const cudaStream_t stream = core::Context::stream().handle();

    cudaStream_t pf_stream = stream;
    cudaStream_t dc_stream = pf_stream;

    if (d.decode.n && d.prefill.n) {
        pf_stream = aux_stream_;
        check_cuda_error(cudaEventRecord(qkv_event_, stream));
        check_cuda_error(cudaStreamWaitEvent(aux_stream_, qkv_event_));
    }

    if (d.prefill.n && !is_warm_up_) {
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

    if (d.decode.n && !is_warm_up_) {
        auto params = CreateParams(0, d.decode, kMaxKVSplits, dc_stream);
        if constexpr (sizeof(T) == 2) {
            dispatchDecoding<T>(params);
            sync_check_cuda_error();
        }
    }

    if (d.decode.n && d.prefill.n) {
        check_cuda_error(cudaEventRecord(aux_event_, aux_stream_));
        check_cuda_error(cudaStreamWaitEvent(stream, aux_event_));
    }

    if (is_warm_up_) {
        rng_.set_stream(stream);
        rng_.GenerateUniform(attn.data<T>(), attn.size(), .02f, -.01f);
    }

    return attn;
}
```

- [ ] **Step 7: Rewrite forward_mla**

Replace the `forward_mla` method (lines 596-647):

```cpp
Tensor UnifiedAttentionLayer::forward_mla(const Tensor& hidden_state, const WeightType& w)
{

    const auto token_num = hidden_state.shape(0);
    const auto dtype     = hidden_state.dtype();

    const int q_lora_rank  = w.q_a_proj->output_dim;
    const int kv_lora_rank = w.kv_a_layernorm->weight.size();
    const int qk_rope_dim  = w.kv_a_proj->output_dim - kv_lora_rank;

    Tensor q;

    const auto stream = core::Context::stream().handle();

    if (w.q_proj && w.q_proj->weight) {
        q = linear_.Forward(hidden_state, *w.q_proj);
        sync_check_cuda_error();
    }
    else {
        Tensor q_a = linear_.Forward(hidden_state, *w.q_a_proj);
        sync_check_cuda_error();

        invokeRMSNorm(q_a, q_a, w.q_a_layernorm->weight, norm_eps_, stream);
        sync_check_cuda_error();

        q = linear_.Forward(q_a, *w.q_b_proj);
        sync_check_cuda_error();
    }

    Tensor kv_a_k_pe = linear_.Forward(hidden_state, *w.kv_a_proj);
    sync_check_cuda_error();

    auto kv_a = kv_a_k_pe.slice({0, 0}, {-1, kv_lora_rank});
    invokeRMSNorm(kv_a, kv_a, w.kv_a_layernorm->weight, norm_eps_, stream);
    sync_check_cuda_error();

    const int tp_size          = w.tp_size_;
    const int local_head_num   = w.head_num_ / tp_size;
    const int local_kv_head_num = w.kv_head_num_ / tp_size;
    const int size_per_head    = w.head_dim_;

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
    sync_check_cuda_error();

    return qkv;
}
```

Changes: `model_param_.norm_eps` -> `norm_eps_`. Local head counts computed from weight.

- [ ] **Step 8: Rewrite qk_norm**

Replace the `qk_norm` method (lines 649-672):

```cpp
void UnifiedAttentionLayer::qk_norm(Tensor& qkv, const WeightType& weights)
{
    const auto stream = core::Context::stream().handle();

    check_cuda_error(cudaEventRecord(qkv_event_, stream));
    check_cuda_error(cudaStreamWaitEvent(aux_stream_, qkv_event_));

    TM_CHECK(weights.bias_ == false) << "not implemented";

    const int tp_size          = weights.tp_size_;
    const int local_head_num   = weights.head_num_ / tp_size;
    const int local_kv_head_num = weights.kv_head_num_ / tp_size;
    const int size_per_head    = weights.head_dim_;

    const auto token_num = qkv.shape(0);

    auto qkv3 = qkv.view({token_num, -1, size_per_head});

    auto q = qkv3.slice({0, 0, 0}, {-1, local_head_num, -1});
    invokeRMSNormQK(q, weights.q_norm->weight, norm_eps_, stream);
    sync_check_cuda_error();

    auto k = qkv3.slice({0, local_head_num, 0}, {-1, local_kv_head_num, -1});
    invokeRMSNormQK(k, weights.k_norm->weight, norm_eps_, aux_stream_);
    sync_check_cuda_error();

    check_cuda_error(cudaEventRecord(aux_event_, aux_stream_));
    check_cuda_error(cudaStreamWaitEvent(stream, aux_event_));
}
```

Changes: `model_param_.attn_bias` -> `weights.bias_`, `model_param_.norm_eps` -> `norm_eps_`, `size_per_head_` -> local `size_per_head`.

- [ ] **Step 9: Build verification skipped**

Call sites not yet updated — will fail. Expected; we fix in Task 3.

- [ ] **Step 10: Commit**

```bash
git add src/turbomind/models/llama/unified_attention_layer.h src/turbomind/models/llama/unified_attention_layer.cc
git commit -m "refactor(attn): decouple UnifiedAttentionLayer from ModelParam/AttentionParam, lazy buffer init"
```

---

### Task 3: Update call site in unified_decoder.cc

**Files:**
- Modify: `src/turbomind/models/llama/unified_decoder.cc`

- [ ] **Step 1: Update UnifiedAttentionLayer construction (line 55-56)**

Replace:
```cpp
    attn_layer_ =
        std::make_unique<UnifiedAttentionLayer>(model, attn, engine, attn_tp_size_, ctx, phases, (bool)moe_ffn_layer_);
```

With:
```cpp
    attn_layer_ = std::make_unique<UnifiedAttentionLayer>(
        model.norm_eps,
        model.quant_policy,
        model.layer_types,
        model.layer_num,
        attn.rope,
        attn.cache_block_seq_len,
        engine,
        ctx,
        phases,
        (bool)moe_ffn_layer_);
```

- [ ] **Step 2: Build to verify clean compile**

Run: `cd build && ninja`
Expected: clean build with no warnings

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/models/llama/unified_decoder.cc
git commit -m "refactor(decoder): update UnifiedAttentionLayer call site for decomposed params"
```

---

### Task 4: Update Python specs to set new AttentionConfig fields

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py`

These 4 files create `_tm.AttentionConfig()` instances. After adding `softmax_scale`, `use_logn_attn`, `max_position_embeddings` to the C++ X-macro, the Python specs need to set these fields on `_attn_cfg` so the `AttentionWeight` instances get the correct values at model conversion time.

The base class `TextModelSpec` already provides `self._softmax_scale` (default 0.0) and `self._max_position_embeddings` (parsed from HF config by `parse_rope_param`). `use_logn_attn` has no HF config field in any spec and defaults to false — safe to leave unset.

- [ ] **Step 1: Update qwen3_spec.py**

In `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py`, after line 57 (`self._attn_cfg.data_type = dtype`), add:

```python
        self._attn_cfg.softmax_scale          = self._softmax_scale
        self._attn_cfg.max_position_embeddings = self._max_position_embeddings
```

- [ ] **Step 2: Update qwen3_5_spec.py**

In `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py`, after line 72 (`self._attn_cfg.data_type = dtype`), add:

```python
        self._attn_cfg.softmax_scale          = self._softmax_scale
        self._attn_cfg.max_position_embeddings = self._max_position_embeddings
```

- [ ] **Step 3: Update glm4_moe_lite_spec.py**

In `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py`, after line 91 (`self._attn_cfg.data_type = dtype`), add:

```python
        self._attn_cfg.softmax_scale          = self._softmax_scale
        self._attn_cfg.max_position_embeddings = self._max_position_embeddings
```

Note: `self._softmax_scale` is conditionally overridden earlier in this spec (lines 46-71) for YaRN. The value is correct when it reaches this line.

- [ ] **Step 4: Update gpt_oss_spec.py**

In `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py`, after line 63 (`self._attn_cfg.data_type = dtype`), add:

```python
        self._attn_cfg.softmax_scale          = self._softmax_scale
        self._attn_cfg.max_position_embeddings = self._max_position_embeddings
```

- [ ] **Step 5: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/qwen3_spec.py \
        lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py \
        lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py \
        lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py
git commit -m "refactor(specs): set softmax_scale and max_position_embeddings on AttentionConfig"
```

---

### Task 5: Smoke test with non-MoE and MoE models

**Files:**
- None (verification only)

- [ ] **Step 1: Check GPU availability**

Run the `get_gpu_usage` MCP tool to confirm a free GPU.

- [ ] **Step 2: Test with a non-MoE model**

Run:
```bash
cd /data/lmdeploy-modeling
python scripts/test_turbomind_model.py <non-moe-model-id> --prompt "Hello, how are you?" --request-output-len 128
```

Use a model from the model-server registry. Expected: Model produces meaningful human-language response, no CUDA errors, no assertion failures.

- [ ] **Step 3: Test with a MoE model**

Run:
```bash
python scripts/test_turbomind_model.py <moe-model-id> --prompt "Hello, how are you?" --request-output-len 128
```

Expected: Same as Step 2.

- [ ] **Step 4: Commit (if any fixup was needed)**

Only commit if fixes were required during testing.
