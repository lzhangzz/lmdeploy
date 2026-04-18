# Decouple LlamaFfnLayer and MoeFfnLayer from ModelParam/MoeParam Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove both layers' dependency on ModelParam and MoeParam. LlamaFfnLayer loses its only (dead) ModelParam field. MoeFfnLayer reads dimensional fields and routing config from MoeWeight at runtime, with lazy buffer initialization on first Forward call.

**Architecture:** MoeWeight exposes `hidden_dim_` and new `inter_size_` (TP-split) as public fields. MoeFfnLayer constructor shrinks to `EngineParam` + `Context` only. On first Forward call, a lazy `Init()` allocates all expert-dependent buffers from the weight's MoeParam. LlamaFfnLayer constructor becomes `Context` only (dead member removal).

**Tech Stack:** C++17, CUDA, ninja build

---

### Task 1: Make MoeWeight fields public, add inter_size_

**Files:**
- Modify: `src/turbomind/models/moe_weight.h`
- Modify: `src/turbomind/models/moe_weight.cc`

- [ ] **Step 1: Add public fields to MoeWeight header**

In `src/turbomind/models/moe_weight.h`, add `hidden_dim_` and `inter_size_` to the public section (after the `moe_param()` accessor, before the `private:` label).

Replace lines 69–88 (the public accessors through end of class):

```cpp
    // --- Typed accessors ---
    FfnWeight*    expert(int i) const;
    FfnWeight*    block() const { return block_.get(); }
    MoeParam::Method method() const { return moe_param_.method; }
    const MoeParam& moe_param() const { return moe_param_; }

    // --- Config fields (public for runtime access) ---
    int hidden_dim_{};
    int inter_size_{};

private:
    int            layer_id_{};
    MoeParam       moe_param_{};
    bool           mlp_bias_{};
    DataType       data_type_{};
    int            tp_size_{};
    int            tp_rank_{};
    ActivationType act_type_{};
    bool           fuse_silu_act_{};
    int            expert_num_{};

    mutable std::unique_ptr<FfnWeight> block_;
```

Note: `hidden_dim_` was previously private — it moves to public. `inter_size_` is new. The other private fields (`layer_id_`, `moe_param_`, `mlp_bias_`, `data_type_`, `tp_size_`, `tp_rank_`, `act_type_`, `fuse_silu_act_`, `expert_num_`, `block_`) remain private.

- [ ] **Step 2: Populate inter_size_ in MoeWeight constructor**

In `src/turbomind/models/moe_weight.cc`, add `inter_size_` initialization in the constructor body. After line 27 (`hidden_dim_ = cfg.hidden_dim;`), add:

```cpp
    inter_size_ = cfg.inter_size / cfg.tp_size;
```

The constructor should now contain (lines 11–37, with the addition):

```cpp
MoeWeight::MoeWeight(const core::MoeConfig& cfg)
{
    layer_id_ = cfg.layer_id;
    moe_param_.method = static_cast<MoeParam::Method>(cfg.method);
    moe_param_.experts_per_token = cfg.experts_per_token;
    moe_param_.inter_size = cfg.inter_size;
    moe_param_.norm_topk_prob = cfg.norm_topk_prob;
    moe_param_.shared_gate = cfg.shared_gate;
    moe_param_.routed_scale = static_cast<float>(cfg.routed_scale);
    moe_param_.router_bias = cfg.router_bias;
    moe_param_.topk_group = cfg.topk_group;
    moe_param_.topk_method = cfg.topk_method;
    moe_param_.n_group = cfg.n_group;
    moe_param_.scoring_func = cfg.scoring_func;
    moe_param_.router_n_groups = cfg.router_n_groups;
    moe_param_.expert_num.assign(1, cfg.expert_num);
    hidden_dim_ = cfg.hidden_dim;
    inter_size_ = cfg.inter_size / cfg.tp_size;
    mlp_bias_ = cfg.mlp_bias;
    data_type_ = cfg.data_type;
    tp_size_ = cfg.tp_size;
    tp_rank_ = cfg.tp_rank;
    act_type_ = static_cast<ActivationType>(cfg.act_type);
    fuse_silu_act_ = cfg.fuse_silu;
    // The expert_num vector always has 1 element for per-layer instances,
    // so always use index 0 regardless of layer_id.
    expert_num_ = cfg.expert_num;
}
```

- [ ] **Step 3: Build to verify no breakage**

Run: `cd build && ninja`
Expected: clean build (no code reads these fields externally yet)

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/moe_weight.h src/turbomind/models/moe_weight.cc
git commit -m "refactor(moe): make MoeWeight fields public, add inter_size_"
```

---

### Task 2: Remove ModelParam from LlamaFfnLayer

**Files:**
- Modify: `src/turbomind/models/llama/LlamaFfnLayer.h`
- Modify: `src/turbomind/models/llama/LlamaFfnLayer.cc`

- [ ] **Step 1: Update LlamaFfnLayer header**

Replace the full class body in `src/turbomind/models/llama/LlamaFfnLayer.h` (lines 30–48):

```cpp
class LlamaFfnLayer {
public:
    LlamaFfnLayer(const Context& ctx): linear_(*ctx.linear)
    {
    }

    struct ForwardParam {
        Tensor                input;
        Tensor                output;
        const FfnWeight* weights;
        int                   layer_id;
    };

    void forward(ForwardParam param);

private:
    LlamaLinear& linear_;
};
```

Changes: constructor takes `const Context& ctx` only, dead `hidden_units_` member removed.

- [ ] **Step 2: Build verification skipped**

The call site in `unified_decoder.cc` still passes `model` — will fail. This is expected; we fix in Task 4.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/models/llama/LlamaFfnLayer.h src/turbomind/models/llama/LlamaFfnLayer.cc
git commit -m "refactor(ffn): remove ModelParam from LlamaFfnLayer, delete dead hidden_units_"
```

---

### Task 3: Rewrite MoeFfnLayer constructor, add lazy Init, update Forward/Combine

**Files:**
- Modify: `src/turbomind/models/llama/moe_ffn_layer.h`
- Modify: `src/turbomind/models/llama/moe_ffn_layer.cc`

- [ ] **Step 1: Update MoeFfnLayer header**

Replace the full class body in `src/turbomind/models/llama/moe_ffn_layer.h` (lines 13–61):

```cpp
class MoeFfnLayer {
public:
    MoeFfnLayer(const EngineParam& engine, const Context& ctx);

    struct ForwardParam {
        Tensor              input;
        Tensor              output;
        const MoeWeight* weights;
        float               scale;
        int                 layer_id;
    };

    void Forward(ForwardParam& p);

    void Combine(ForwardParam& p);

private:
    void Init(ForwardParam& p);

    Tensor_<float> Gate(const Tensor& input, const LinearWeight& gate);

    void dump_logits(int token_num, int layer_id, int expert_num);

    const int tp_size_;
    const int max_token_num_;
    int&      is_warm_up_;

    LlamaLinear& linear_;

    std::unique_ptr<LlamaFfnLayer> expert_ffn_;

    bool initialized_ = false;

    ///////////////////////////////////////////////////////
    /// runtime states
    Buffer_<int> h_offsets_;

    Buffer_<int>   masks_;
    Buffer_<int>   f2n_;
    Buffer_<int>   f2E_;
    Buffer_<int>   en2f_;
    Buffer_<float> scales_;
    Buffer_<int>   accum_;
    Buffer_<int>   offsets_;

    Tensor         temp_;
    Tensor_<float> shared_scales_;
    ///////////////////////////////////////////////////////
};
```

Changes from current:
- Constructor: `(const EngineParam& engine, const Context& ctx)` — removed `ModelParam`, `MoeParam`
- Added `Init(ForwardParam& p)` private method
- Removed: `inter_size_`, `hidden_dim_`, `param_`
- Added: `max_token_num_`, `initialized_`

- [ ] **Step 2: Rewrite MoeFfnLayer constructor**

Replace the constructor in `src/turbomind/models/llama/moe_ffn_layer.cc` (lines 21–52):

```cpp
MoeFfnLayer::MoeFfnLayer(const EngineParam& engine, const Context& ctx):
    tp_size_(engine.mlp_tp_size),
    max_token_num_(engine.max_forward_token_num * engine.attn_dp_size),
    is_warm_up_(*ctx.is_warm_up),
    linear_(*ctx.linear),
    expert_ffn_(std::make_unique<LlamaFfnLayer>(ctx))
{
}
```

- [ ] **Step 3: Add Init method**

Add the `Init` method after the constructor in `moe_ffn_layer.cc`:

```cpp
void MoeFfnLayer::Init(ForwardParam& p)
{
    const auto& moe_param = p.weights->moe_param();
    const int   expert_num       = p.weights->num_experts();
    const int   experts_per_token = moe_param.experts_per_token;

    h_offsets_ = {expert_num + 1, kCPUpinned};

    const int pad_token_num =
        (max_token_num_ + kMoeGateVecSize - 1) / kMoeGateVecSize * kMoeGateVecSize;

    masks_   = {expert_num * pad_token_num, kDEVICE};
    f2n_     = {experts_per_token * max_token_num_, kDEVICE};
    f2E_     = {experts_per_token * max_token_num_, kDEVICE};
    en2f_    = {experts_per_token * max_token_num_, kDEVICE};
    scales_  = {experts_per_token * max_token_num_, kDEVICE};
    offsets_ = {expert_num + 1, kDEVICE};
    accum_   = {expert_num * kMoeGateMaxTiles, kDEVICE};

    initialized_ = true;
}
```

- [ ] **Step 4: Rewrite Forward to read from weight**

Replace the `Forward` method in `src/turbomind/models/llama/moe_ffn_layer.cc` (lines 66–215).

Every `param_.*` reference becomes `moe_param.*` (local from `p.weights->moe_param()`). `hidden_dim_` becomes local `hidden_dim`. `inter_size_` becomes local `inter_size`.

```cpp
void MoeFfnLayer::Forward(ForwardParam& p)
{
    if (!initialized_) {
        Init(p);
    }

    const int   tokens    = p.input.shape(0);
    const auto& moe       = *p.weights;
    const auto& moe_param = p.weights->moe_param();

    const int hidden_dim = p.weights->hidden_dim_;
    const int inter_size = p.weights->inter_size_;

    const size_t padded     = (tokens + kMoeGateVecSize - 1) / kMoeGateVecSize * kMoeGateVecSize;
    const int    expert_num = moe.num_experts();

    FT_CHECK(expert_num);

    auto logits = Gate(p.input, *moe.gate.get());

    TM_DEBUG_TENSOR(logits, "logits", 2);

    const auto st = core::Context::stream().handle();

    if (moe_param.topk_method == "noaux_tc") {
        // invokeMoeGate_NoAuxTC clears accum and masks internally
        TM_CHECK_EQ(moe_param.n_group, 1);
        TM_CHECK_EQ(moe_param.topk_group, 1);
        const float* correction_bias = nullptr;
        if (moe.score_correction_bias) {
            correction_bias = moe.score_correction_bias.size() > 0 ? moe.score_correction_bias.data<float>()
                                                                    : nullptr;
        }
        invokeMoeGate_NoAuxTC(f2n_.data(),
                              f2E_.data(),
                              en2f_.data(),
                              offsets_.data(),
                              scales_.data(),
                              masks_.data(),
                              accum_.data(),
                              logits.data(),
                              correction_bias,
                              tokens,
                              padded,
                              expert_num,
                              moe_param.experts_per_token,
                              moe_param.norm_topk_prob,
                              moe_param.routed_scale,
                              moe_param.scoring_func == "sigmoid",
                              st);
    }
    else {
        // V2: accum must be cleared by caller; masks cleared internally
        check_cuda_error(cudaMemsetAsync(accum_.data(), 0, sizeof(int) * expert_num * kMoeGateMaxTiles, st));

        bool softmax = true;
        if (moe_param.topk_method == "group_limited_greedy") {
            invokeMoeSoftmaxMaskTopKGroups(
                logits.data(), tokens, expert_num, expert_num / moe_param.n_group, moe_param.topk_group, st);
            sync_check_cuda_error();
            softmax = false;
        }

        /// TODO: fix illegal memory access even if NaN are present in logits
        invokeMoeGate_V2(f2n_.data(),
                         f2E_.data(),
                         en2f_.data(),
                         offsets_.data(),
                         scales_.data(),
                         masks_.data(),
                         accum_.data(),
                         logits.data(),
                         tokens,
                         padded,
                         expert_num,
                         moe_param.experts_per_token,
                         softmax,
                         moe_param.norm_topk_prob,
                         moe_param.routed_scale,
                         st);
    }
    sync_check_cuda_error();

    if (is_warm_up_) {
        std::mt19937     g;
        const auto       expert_ids = SampleUniform(tokens, expert_num, moe_param.experts_per_token, g);
        std::vector<int> cnt(expert_num);
        for (const auto& x : expert_ids) {
            ++cnt[x];
        }
        h_offsets_[0] = 0;
        for (int i = 0; i < expert_num; ++i) {
            h_offsets_[i + 1] = h_offsets_[i] + cnt[i];
        }
        check_cuda_error(
            cudaMemcpyAsync(offsets_.data(), h_offsets_.data(), sizeof(int) * (expert_num + 1), cudaMemcpyDefault, st));
    }

    temp_ = Tensor{{moe_param.experts_per_token * tokens, hidden_dim}, p.input.dtype(), p.input.device()};

    if (moe_param.method == MoeParam::kNaive) {

        invokeMoeDispatch(temp_, p.input, f2n_.data(), moe_param.experts_per_token, st);
        sync_check_cuda_error();

        check_cuda_error(
            cudaMemcpyAsync(h_offsets_.data(), offsets_.data(), sizeof(int) * (expert_num + 1), cudaMemcpyDefault, st));

        check_cuda_error(cudaStreamSynchronize(st));

        TM_CHECK_EQ(h_offsets_[expert_num], tokens * moe_param.experts_per_token);

        for (int i = 0; i < expert_num; ++i) {
            if (int count = h_offsets_[i + 1] - h_offsets_[i]) {
                auto io = temp_.slice({h_offsets_[i], 0}, {count, -1});
                expert_ffn_->forward({io, io, moe.expert(i), p.layer_id});
            }
        }
    }
    else {

        auto* block = moe.block();

        auto indices = f2n_.slice(0, tokens * moe_param.experts_per_token);
        auto offsets = offsets_.slice(0, expert_num + 1);

        if (block->w1w3 && block->w1w3->weight) {
            // Fused w1w3 path
            Tensor inter = linear_.Forward(p.input, *block->w1w3, indices, offsets_);
            sync_check_cuda_error();

            if (!block->is_fused_silu()) {
                Activation(inter, block->w1w3->bias, f2E_, block->act_type(), st);
                sync_check_cuda_error();
            }

            linear_.Forward(inter.slice({0, 0}, {-1, inter_size}), *block->w2, {}, offsets, temp_);
            sync_check_cuda_error();
        }
        else {
            // Separate w1/w3 path
            Tensor gating = linear_.Forward(p.input, *block->w1, indices, offsets_);
            sync_check_cuda_error();

            Tensor up = linear_.Forward(p.input, *block->w3, indices, offsets_);
            sync_check_cuda_error();

            Activation(gating, up, block->act_type(), st);
            sync_check_cuda_error();

            linear_.Forward(gating, *block->w2, {}, offsets, temp_);
            sync_check_cuda_error();
        }
    }

    if (moe.shared_gate && moe.shared_gate->weight) {
        shared_scales_ = Gate(p.input, *moe.shared_gate.get());
    }
}
```

Replacement table for `param_.*` → `moe_param.*`:

| Old | New |
|---|---|
| `param_.topk_method` | `moe_param.topk_method` |
| `param_.n_group` | `moe_param.n_group` |
| `param_.topk_group` | `moe_param.topk_group` |
| `param_.experts_per_token` | `moe_param.experts_per_token` |
| `param_.norm_topk_prob` | `moe_param.norm_topk_prob` |
| `param_.routed_scale` | `moe_param.routed_scale` |
| `param_.scoring_func` | `moe_param.scoring_func` |
| `param_.method` | `moe_param.method` |
| `hidden_dim_` | `hidden_dim` (local from `p.weights->hidden_dim_`) |
| `inter_size_` | `inter_size` (local from `p.weights->inter_size_`) |

- [ ] **Step 5: Update Combine to read from weight**

Replace the `Combine` method in `src/turbomind/models/llama/moe_ffn_layer.cc` (lines 217–239):

```cpp
void MoeFfnLayer::Combine(ForwardParam& p)
{
    auto&       moe       = *p.weights;
    const auto& moe_param = p.weights->moe_param();

    const Tensor& block_bias = moe.block() && moe.block()->w2 ? moe.block()->w2->bias : Tensor{};

    invokeMoeCombine(p.output,
                     temp_,
                     block_bias,
                     scales_.data(),
                     en2f_.data(),
                     f2E_.data(),
                     shared_scales_.data_or((float*)nullptr),
                     moe_param.experts_per_token,
                     1.f / tp_size_,
                     p.scale,
                     core::Context::stream().handle());
    sync_check_cuda_error();

    temp_          = {};
    shared_scales_ = {};
}
```

- [ ] **Step 6: Build verification skipped**

Call sites not yet updated — will fail. Expected; we fix in Task 4.

- [ ] **Step 7: Commit**

```bash
git add src/turbomind/models/llama/moe_ffn_layer.h src/turbomind/models/llama/moe_ffn_layer.cc
git commit -m "refactor(moe): read dims from MoeWeight at runtime, lazy buffer init, drop ModelParam/MoeParam"
```

---

### Task 4: Update call sites in unified_decoder.cc

**Files:**
- Modify: `src/turbomind/models/llama/unified_decoder.cc`

- [ ] **Step 1: Update MoeFfnLayer construction (line 52)**

Replace:
```cpp
        moe_ffn_layer_ = std::make_unique<MoeFfnLayer>(model, moe, engine, ctx);
```

With:
```cpp
        moe_ffn_layer_ = std::make_unique<MoeFfnLayer>(engine, ctx);
```

- [ ] **Step 2: Update LlamaFfnLayer construction (line 65)**

Replace:
```cpp
        ffn_layer_ = std::make_unique<LlamaFfnLayer>(model, ctx);
```

With:
```cpp
        ffn_layer_ = std::make_unique<LlamaFfnLayer>(ctx);
```

- [ ] **Step 3: Build to verify clean compile**

Run: `cd build && ninja`
Expected: clean build with no warnings

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/llama/unified_decoder.cc
git commit -m "refactor(decoder): update LlamaFfnLayer and MoeFfnLayer call sites"
```

---

### Task 5: Smoke test with a MoE model

**Files:**
- None (verification only)

- [ ] **Step 1: Check GPU availability**

Run the `get_gpu_usage` MCP tool to confirm a free GPU.

- [ ] **Step 2: Test with a MoE model**

Run:
```bash
cd /data/lmdeploy-modeling
python scripts/test_turbomind_model.py <moe-model-id> --prompt "Hello, how are you?" --request-output-len 128
```

Use a model from the model-server registry that uses MoE layers (e.g., Mixtral, Qwen MoE variant). Check `list_models` MCP tool for available models.

Expected: Model produces meaningful human-language response, no CUDA errors, no assertion failures.

- [ ] **Step 3: Commit (if any fixup was needed)**

Only commit if fixes were required during testing.
