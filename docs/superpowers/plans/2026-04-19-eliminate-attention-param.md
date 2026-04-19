# Eliminate AttentionParam Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the `AttentionParam` struct and thread `AttentionWeight*` through `UnifiedDecoder → UnifiedAttentionLayer` so attention params come from weights instead of a separate config path.

**Architecture:** `cache_block_seq_len` moves to `EngineParam`. `UnifiedDecoder` extracts `AttentionWeight*` from `DecoderLayerWeight*` and passes to `UnifiedAttentionLayer` constructor. `UnifiedAttentionLayer` reads rope config from weights at construction time, eliminating the lazy `Init()` pattern.

**Tech Stack:** C++17, CUDA, YAML-cpp

---

### Task 1: Add `cache_block_seq_len` to `EngineParam` and update YAML parsing

**Files:**
- Modify: `src/turbomind/models/llama/llama_params.h:112-116` (EngineParam cache params section)
- Modify: `src/turbomind/turbomind.cc:167-374` (TurboMind::Impl members and YAML parsing)

- [ ] **Step 1: Add field to EngineParam**

In `src/turbomind/models/llama/llama_params.h`, add `cache_block_seq_len` to `EngineParam` after `cache_chunk_size` (line 114):

```cpp
// cache params
float cache_max_block_count;
int   cache_chunk_size;
int   cache_block_seq_len;   // moved from AttentionParam
bool  enable_prefix_caching;
bool  enable_metrics;
```

- [ ] **Step 2: Update YAML parsing in turbomind.cc**

In `src/turbomind/turbomind.cc`:

a) Remove `AttentionParam attn_param_;` member (line 170).

b) Change line 320 from:
```cpp
attn_param_.cache_block_seq_len = attention["cache_block_seq_len"].as<int>(0);
```
to:
```cpp
engine_param_.cache_block_seq_len = attention["cache_block_seq_len"].as<int>(0);
```

c) Remove lines 368-373 (dead `AttentionParam` YAML parsing):
```cpp
// DELETE these lines:
attn_param_.softmax_scale = attention["softmax_scale"].as<float>(0);
// logn attn for qwen model
attn_param_.use_logn_attn           = attention["use_logn_attn"].as<int>(0);
attn_param_.max_position_embeddings = attention["max_position_embeddings"].as<int>(0);
// rotary embedding parameters
parse_rope_param(attention["rope_param"], attn_param_.rope);
```

d) Remove `attn_param_` from the `LanguageModel` constructor call (line 552). Change:
```cpp
LanguageModel model{data_type_,  //
                    model_param_,
                    param,
                    attn_param_,
                    moe_param_,
                    ctx,
                    *weights_[index],
                    phases_};
```
to:
```cpp
LanguageModel model{data_type_,  //
                    model_param_,
                    param,
                    moe_param_,
                    ctx,
                    *weights_[index],
                    phases_};
```

- [ ] **Step 3: Build to verify compilation**

Run: `cd /data/lmdeploy-modeling/build && ninja`
Expected: Compile errors in files still referencing `AttentionParam` — that's expected, we'll fix them in subsequent tasks.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/llama/llama_params.h src/turbomind/turbomind.cc
git commit -m "refactor(config): move cache_block_seq_len to EngineParam, remove AttentionParam from TurboMind::Impl"
```

---

### Task 2: Remove `AttentionParam` from `Engine`

**Files:**
- Modify: `src/turbomind/engine/engine.cc:207-215,238`

- [ ] **Step 1: Replace model_.attn_param() with param_**

In `src/turbomind/engine/engine.cc`:

a) Line 208 — change:
```cpp
const ssize_t max_batch_block_num =
    param.max_batch_size * cdiv(session_len_trunc_, model_.attn_param().cache_block_seq_len);
```
to:
```cpp
const ssize_t max_batch_block_num =
    param.max_batch_size * cdiv(session_len_trunc_, param.cache_block_seq_len);
```

b) Line 215 — change:
```cpp
const auto cache_block_seq_len = model_.attn_param().cache_block_seq_len;
```
to:
```cpp
const auto cache_block_seq_len = param_.cache_block_seq_len;
```

Line 238 uses the local `cache_block_seq_len` variable — no change needed.

- [ ] **Step 2: Commit**

```bash
git add src/turbomind/engine/engine.cc
git commit -m "refactor(engine): read cache_block_seq_len from EngineParam instead of LanguageModel"
```

---

### Task 3: Remove `AttentionParam` from `LanguageModel`

**Files:**
- Modify: `src/turbomind/models/language_model.h:27-39`
- Modify: `src/turbomind/models/language_model.cc:35-38,105-166,495-519`

- [ ] **Step 1: Update LanguageModel header**

In `src/turbomind/models/language_model.h`:

a) Remove `const AttentionParam& attn` from the constructor (line 30):
```cpp
LanguageModel(DataType              dtype,
              const ModelParam&     model,
              const EngineParam&    engine,
              const MoeParam&       moe,
              const Context&        ctx,
              const ModelWeight&    weights,
              int                   phases);
```

b) Remove the `attn_param()` accessor (line 39):
```cpp
const ModelParam& model_param() const noexcept;
// remove: const AttentionParam& attn_param() const noexcept;
```

- [ ] **Step 2: Update LanguageModel implementation**

In `src/turbomind/models/language_model.cc`:

a) Remove `const AttentionParam attn_param_;` member from `Impl` (line 38).

b) Remove `const AttentionParam& attn` from `Impl` constructor (line 108) and remove `attn_param_{attn}` from the initializer list (line 134).

c) Update `Impl` constructor body line 166 — pass `weights_.layers_list()` to `UnifiedDecoder`:
```cpp
unified_decoder_ = std::make_unique<UnifiedDecoder>(model, engine, weights_.layers_list(), moe, ctx, phases);
```

d) Remove the `LanguageModel` constructor's `const AttentionParam& attn` parameter (line 498) and remove `attn` from the `Impl` construction call (line 504):
```cpp
impl_ = std::make_unique<Impl>(dtype, model, engine, moe, ctx, weights, phases);
```

e) Delete the `attn_param()` accessor implementation (lines 517-519):
```cpp
// DELETE:
const AttentionParam& LanguageModel::attn_param() const noexcept
{
    return TM_CHECK_NOTNULL(impl_)->attn_param_;
}
```

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/models/language_model.h src/turbomind/models/language_model.cc
git commit -m "refactor(LanguageModel): remove AttentionParam, pass layer weights to UnifiedDecoder"
```

---

### Task 4: Remove `AttentionParam` from `UnifiedDecoder`

**Files:**
- Modify: `src/turbomind/models/llama/unified_decoder.h:18-23`
- Modify: `src/turbomind/models/llama/unified_decoder.cc:33-74`

- [ ] **Step 1: Update UnifiedDecoder header**

In `src/turbomind/models/llama/unified_decoder.h`:

Replace the constructor declaration:
```cpp
UnifiedDecoder(const ModelParam&     model,
               const EngineParam&    engine,
               const AttentionParam& attn,
               const MoeParam&       moe,
               const Context&        ctx,
               int                   phases);
```
with:
```cpp
UnifiedDecoder(const ModelParam&                      model,
               const EngineParam&                     engine,
               const std::vector<DecoderLayerWeight*>& layer_weights,
               const MoeParam&                        moe,
               const Context&                         ctx,
               int                                    phases);
```

Add the forward declaration for `AttentionWeight` if not already present (check includes):
```cpp
class AttentionWeight;  // forward declare if not already included
```

Note: `attention_weight.h` is already included transitively via `unified_attention_layer.h`.

- [ ] **Step 2: Update UnifiedDecoder implementation**

In `src/turbomind/models/llama/unified_decoder.cc`:

a) Update constructor signature and extract `AttentionWeight*` vector:
```cpp
UnifiedDecoder::UnifiedDecoder(const ModelParam&                      model,
                               const EngineParam&                     engine,
                               const std::vector<DecoderLayerWeight*>& layer_weights,
                               const MoeParam&                        moe,
                               const Context&                         ctx,
                               int                                    phases):
    layer_num_(model.layer_num),
    hidden_units_(model.hidden_units),
    attn_tp_size_(engine.attn_tp_size),
    attn_dp_size_(engine.attn_dp_size),
    attn_dp_rank_(engine.attn_dp_rank),
    mlp_tp_size_(engine.mlp_tp_size),
    attn_tp_group_(ctx.comm.d_tp_group),
    d_comm_(ctx.comm.d_comm),
    tune_layer_num_(model.tune_layer_num),
    is_warm_up_{*ctx.is_warm_up}
{
    if (std::accumulate(moe.expert_num.begin(), moe.expert_num.end(), 0LL)) {
        moe_ffn_layer_ = std::make_unique<MoeFfnLayer>(engine, ctx);
    }

    // Extract AttentionWeight* from each DecoderLayerWeight
    std::vector<AttentionWeight*> attn_weights;
    attn_weights.reserve(layer_weights.size());
    for (auto* lw : layer_weights) {
        attn_weights.push_back(lw->attention.get());
    }

    attn_layer_ = std::make_unique<UnifiedAttentionLayer>(
        model.quant_policy,
        model.layer_types,
        model.layer_num,
        attn_weights,
        engine,
        ctx,
        phases,
        (bool)moe_ffn_layer_);

    if (std::find(model.layer_types.begin(), model.layer_types.end(), 1) != model.layer_types.end()) {
        linear_attn_layer_ = std::make_unique<GatedDeltaNetLayer>(
            model.linear_state_dtype, model.layer_types,
            engine, ctx, phases);
    }

    if (std::accumulate(model.inter_size.begin(), model.inter_size.end(), 0LL)) {
        ffn_layer_ = std::make_unique<LlamaFfnLayer>(ctx);
    }
}
```

Note: `core::to_rope_config(attn.rope)` and `attn.cache_block_seq_len` are gone — `UnifiedAttentionLayer` reads these from `attn_weights[0]` and `engine_param_`.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/models/llama/unified_decoder.h src/turbomind/models/llama/unified_decoder.cc
git commit -m "refactor(UnifiedDecoder): replace AttentionParam with AttentionWeight extraction"
```

---

### Task 5: Rewrite `UnifiedAttentionLayer` constructor — remove lazy init

**Files:**
- Modify: `src/turbomind/models/llama/unified_attention_layer.h:56-86,91,104-112`
- Modify: `src/turbomind/models/llama/unified_attention_layer.cc:95-189,306-326,448`

This is the largest change. The constructor absorbs all `Init()` logic, and several members/parameters are removed.

- [ ] **Step 1: Update the header**

In `src/turbomind/models/llama/unified_attention_layer.h`:

a) Replace constructor declaration (lines 56-64):
```cpp
UnifiedAttentionLayer(int                     quant_policy,
                      const std::vector<int>& layer_types,
                      int                     layer_num,
                      std::vector<AttentionWeight*> attn_weights,
                      const EngineParam&      engine,
                      const Context&          context,
                      int                     phases,
                      bool                    init);
```

b) Remove `Init` declaration (line 71):
```cpp
// DELETE: void Init(const ForwardParam& p);
```

c) Remove `cache_block_seq_len_` member (line 86):
```cpp
// DELETE: const int           cache_block_seq_len_;
```

d) Remove `initialized_` flag (line 91):
```cpp
// DELETE: bool                initialized_ = false;
```

e) Update comment on temp buffers (line 108):
```cpp
///////////////////////////////////////////////////////
/// temp runtime buffers (allocated in constructor)
```

- [ ] **Step 2: Update the implementation**

In `src/turbomind/models/llama/unified_attention_layer.cc`:

a) Replace the constructor (lines 95-157) with the merged version that includes Init logic:
```cpp
UnifiedAttentionLayer::UnifiedAttentionLayer(int                     quant_policy,
                                             const std::vector<int>& layer_types,
                                             int                     layer_num,
                                             std::vector<AttentionWeight*> attn_weights,
                                             const EngineParam&      engine,
                                             const Context&          ctx,
                                             int                     phases,
                                             bool                    init):
    quant_policy_{quant_policy},
    rope_{attn_weights[0]->rope_},
    engine_param_{engine},
    cp_fn_ctx_{ctx.comm.d_comm, ctx.comm.d_cp_group},
    is_warm_up_{*ctx.is_warm_up},
    context_{ctx},
    init_{init},
    linear_(*ctx.linear),
    arch_{getSMVersion()}
{
    check_cuda_error(cudaStreamCreateWithFlags(&aux_stream_, cudaStreamNonBlocking));
    check_cuda_error(cudaEventCreateWithFlags(&qkv_event_, cudaEventDisableTiming));
    check_cuda_error(cudaEventCreateWithFlags(&aux_event_, cudaEventDisableTiming));

    init_rope_kernel_param(rope_, rope_param_);

    // Skip other attention layer types
    std::vector<int> types = layer_types;
    types.resize(layer_num);
    cache_layer_ids_.resize(types.size(), -1);
    int next_cache_id = 0;
    for (size_t i = 0; i < types.size(); ++i) {
        if (types[i] == 0) {
            cache_layer_ids_[i] = next_cache_id++;
        }
    }

    const int bsz = engine.max_batch_size;

    if (rope_param_.type == RopeType::kDynamic) {
        rope_base_buf_ = {bsz + 1, kCPUpinned};
    }
    else if (rope_param_.type == RopeType::kMrope) {
        mrope_position_delta_buf_ = {bsz, kCPUpinned};
        mrope_length_buf_         = {bsz, kCPUpinned};
    }
    const int max_blocks = bsz * cdiv(engine.session_len, engine_param_.cache_block_seq_len);
    for (int i = 0; i < phases; ++i) {
        auto& d               = data_.emplace_back(std::make_shared<AttentionData>());
        d->block_ptrs         = {max_blocks + 16, kDEVICE};
        d->block_ptrs_offsets = {bsz + 1, kDEVICE};
        if (rope_param_.type == RopeType::kDynamic) {
            d->rope_base = empty_like(rope_base_buf_, kDEVICE);
        }
        else if (rope_param_.type == RopeType::kMrope) {
            d->mrope_position_ids    = {{bsz, engine.session_len, 3}, kDEVICE};
            d->mrope_position_delta  = empty_like(mrope_position_delta_buf_, kDEVICE);
            d->mrope_length          = empty_like(mrope_length_buf_, kDEVICE);
            rope_param_.mrope.stride = d->mrope_position_ids.stride(0);
        }
    }

    // --- Former Init() logic ---
    const auto& w = *attn_weights[0];
    const int   tp_size         = w.tp_size_;
    const int   local_head_num  = w.head_num_ / tp_size;
    const int   local_kv_head_num = w.kv_head_num_ / tp_size;
    const int   size_per_head   = w.head_dim_;

    TM_CHECK_EQ(w.head_num_ % tp_size, 0) << w.head_num_ << " " << tp_size;
    TM_CHECK_EQ(w.head_num_ % w.kv_head_num_, 0) << w.head_num_ << " " << w.kv_head_num_;

    ssize_t   workspace_tokens = kMaxWorkspaceTokens;
    Allocator alloc            = core::Context::device_alloc();
    if (engine_param_.attn_cp_size > 1) {
        alloc = GetSymmAllocator(context_.comm.d_comm);
        workspace_tokens += engine_param_.max_forward_token_num;
    }

    partial_O_  = Tensor_<float>({workspace_tokens, local_head_num, size_per_head}, kDEVICE);
    partial_ML_ = Tensor_<float>({engine_param_.attn_cp_size, workspace_tokens, local_head_num, 2}, alloc);
    split_cnt_  = Tensor_<int>({workspace_tokens}, kDEVICE);
    if (init_) {
        const int dim = local_head_num * size_per_head;
        tmp_attn_     = Tensor{{engine_param_.max_forward_token_num, dim}, w.data_type_, kDEVICE};
    }

    Clear(split_cnt_.buffer());
}
```

b) Delete the `Init()` method entirely (lines 159-189).

c) Remove the lazy init check from `Forward()` (lines 319-321):
```cpp
// DELETE:
    if (!initialized_) {
        Init(p);
    }
```

d) Update `cache_block_seq_len_` references in `core_attention` (line 448):
Change:
```cpp
cache_block_seq_len_};
```
to:
```cpp
engine_param_.cache_block_seq_len};
```

- [ ] **Step 3: Build and fix any remaining compilation errors**

Run: `cd /data/lmdeploy-modeling/build && ninja`

Expected: Clean build. If there are errors about `cache_block_seq_len_` elsewhere in the file, replace with `engine_param_.cache_block_seq_len`.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/llama/unified_attention_layer.h src/turbomind/models/llama/unified_attention_layer.cc
git commit -m "refactor(UnifiedAttentionLayer): read rope from AttentionWeight, remove lazy Init()"
```

---

### Task 6: Delete `AttentionParam` struct

**Files:**
- Modify: `src/turbomind/models/llama/llama_params.h:96-104`

- [ ] **Step 1: Delete the struct**

In `src/turbomind/models/llama/llama_params.h`, delete lines 96-104:
```cpp
// DELETE:
struct AttentionParam {
    float softmax_scale;
    int   cache_block_seq_len;
    // logn attention
    bool use_logn_attn;
    int  max_position_embeddings;
    // rotary embedding
    RopeParam rope;
};
```

- [ ] **Step 2: Build to verify**

Run: `cd /data/lmdeploy-modeling/build && ninja`
Expected: Clean build. All references to `AttentionParam` should already be removed from earlier tasks.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/models/llama/llama_params.h
git commit -m "refactor: delete AttentionParam struct"
```

---

### Task 7: Clean up dead code

**Files:**
- Modify: `src/turbomind/models/llama/llama_rope.h:35-64,101-157`
- Modify: `src/turbomind/models/attention_weight.h:37-58`
- Modify: `src/turbomind/turbomind.cc:64-138`

- [ ] **Step 1: Remove dead rope types from llama_rope.h**

In `src/turbomind/models/llama/llama_rope.h`, delete:
- `YarnRopeParam` struct (lines 35-39)
- `Llama3RopeParam` struct (lines 41-45)
- `MropeRopeParam` struct (lines 47-49)
- `RopeParam` struct (lines 51-64)
- `init_rope_kernel_param(const RopeParam&, ...)` function (lines 101-157)

Keep: `RopeType` enum, `GetRoPEType`, all `*KernelParam` structs (lines 66-99).

- [ ] **Step 2: Remove `to_rope_config` from attention_weight.h**

In `src/turbomind/models/attention_weight.h`, delete lines 37-58:
```cpp
// DELETE:
inline RopeConfig to_rope_config(const ::turbomind::RopeParam& p) {
    ...
}
```

- [ ] **Step 3: Remove dead YAML parsing helpers from turbomind.cc**

In `src/turbomind/turbomind.cc`, delete the rope parsing functions (lines 64-138):
- `parse_default_rope_param`
- `parse_linear_rope_param`
- `parse_dynamic_rope_param`
- `parse_yarn_rope_param`
- `parse_llama3_rope_param`
- `parse_mrope_rope_param`
- `parse_rope_param`

- [ ] **Step 4: Build to verify**

Run: `cd /data/lmdeploy-modeling/build && ninja`
Expected: Clean build.

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/models/llama/llama_rope.h src/turbomind/models/attention_weight.h src/turbomind/turbomind.cc
git commit -m "refactor: remove dead RopeParam, to_rope_config, and YAML rope parsing"
```

---

### Task 8: Build, test, and verify

- [ ] **Step 1: Full clean build**

Run: `cd /data/lmdeploy-modeling/build && ninja`
Expected: Clean build with no warnings related to the changes.

- [ ] **Step 2: Check GPU availability**

Use MCP tool `get_gpu_usage` to find an empty GPU.

- [ ] **Step 3: Run model test**

Run: `python scripts/test_turbomind_model.py --model <model_id> --tp 1`
(use a model from `list_models` MCP tool)

Expected: Model loads, generates meaningful text response to the test prompt. Gibberish indicates a bug.

- [ ] **Step 4: Fix any issues**

If the model produces gibberish or crashes, debug and fix before proceeding. The most likely issues:
- Missing `cache_block_seq_len` propagation to per-device `engine_params_` (check `turbomind.cc` where `engine_params_` are created from `engine_param_`)
- `attn_weights` vector is empty or contains null pointers

- [ ] **Step 5: Final commit if fixes were needed**

```bash
git add -u
git commit -m "fix: address test failures from AttentionParam removal"
```
