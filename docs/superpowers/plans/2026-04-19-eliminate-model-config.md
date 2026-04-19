# Eliminate model_config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete ModelParam, MLAParam, MoeParam. Stop parsing model_config YAML. ModelWeight derives all fields from children.

**Architecture:** ModelWeight gains public fields populated in prepare() after children load. Consumers read from ModelWeight directly. turbomind.cc stops parsing model_config entirely. Two fields (quant_policy, tune_layer_num) move to EngineParam. MoeParam::Method enum moves to MoeWeight header.

**Tech Stack:** C++ (CUDA), YAML-cpp (parsing removed), ninja build system

---

### Task 1: Add quant_policy and tune_layer_num to EngineParam

**Files:**
- Modify: `src/turbomind/models/llama/llama_params.h:95-131`

- [ ] **Step 1: Add fields to EngineParam**

Add after `session_len` (line 98):

```cpp
    int   session_len;
    int   step_length;
+   int   quant_policy   = 0;
+   int   tune_layer_num = 1;
```

- [ ] **Step 2: Update turbomind.cc to populate from engine_config**

In `src/turbomind/turbomind.cc`, change line 244 from:
```cpp
    model_param_.quant_policy       = engine["quant_policy"].as<int>(0);
```
to:
```cpp
    engine_param_.quant_policy      = engine["quant_policy"].as<int>(0);
```

Add after line 310 (phases_ assignment):
```cpp
    engine_param_.tune_layer_num = model["tune_layer_num"].as<int>(1);
```

Note: `tune_layer_num` still reads from model_config temporarily. It moves to engine_config when model_config parsing is fully deleted in Task 12.

- [ ] **Step 3: Build**

Run: `cd build && ninja`
Expected: Clean build (no behavioral change yet).

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/llama/llama_params.h src/turbomind/turbomind.cc
git commit -m "refactor: add quant_policy and tune_layer_num to EngineParam"
```

---

### Task 2: Move MoeMethod enum to MoeWeight header

**Files:**
- Modify: `src/turbomind/models/moe_weight.h:11,72-73,81`
- Modify: `src/turbomind/turbomind.cc:40-62,361-364`
- Modify: `src/turbomind/models/llama/moe_ffn_layer.cc` (includes)

- [ ] **Step 1: Define MoeMethod enum in moe_weight.h**

Add before `struct MoeConfig`:
```cpp
enum class MoeMethod {
    kNaive,
    kFused,
};
```

- [ ] **Step 2: Update MoeWeight to use MoeMethod**

Change `moe_weight.h` line 72 from:
```cpp
    MoeParam::Method method() const { return moe_param_.method; }
```
to:
```cpp
    MoeMethod method() const { return method_; }
```

Add private member:
```cpp
    MoeMethod method_{MoeMethod::kFused};
```

Update `moe_weight.cc` constructor to initialize `method_` from `MoeConfig::method` (the int field maps to the enum). In the constructor body or init list:
```cpp
    method_ = static_cast<MoeMethod>(cfg.method);
```

- [ ] **Step 3: Update get_moe_method() in turbomind.cc**

Change return type from `std::optional<MoeParam::Method>` to `std::optional<MoeMethod>`. Include `moe_weight.h` instead of (or in addition to) using MoeParam types. Update `MoeParam::kNaive` → `MoeMethod::kNaive`, `MoeParam::kFused` → `MoeMethod::kFused`.

Update lines 361-364:
```cpp
    moe_param_.method = *method;
```
stays the same for now (MoeParam still exists temporarily).

- [ ] **Step 4: Update MoeFfnLayer references**

In `src/turbomind/models/llama/moe_ffn_layer.cc`, change `MoeParam::kNaive` → `MoeMethod::kNaive` and `MoeParam::kFused` → `MoeMethod::kFused`.

- [ ] **Step 5: Build**

Run: `cd build && ninja`
Expected: Clean build.

- [ ] **Step 6: Commit**

```bash
git add src/turbomind/models/moe_weight.h src/turbomind/models/moe_weight.cc src/turbomind/turbomind.cc src/turbomind/models/llama/moe_ffn_layer.cc
git commit -m "refactor: move MoeMethod enum to MoeWeight header"
```

---

### Task 3: Eliminate MLAParam from AttentionWeight

**Files:**
- Modify: `src/turbomind/models/attention_weight.h:104,111`
- Modify: `src/turbomind/models/attention_weight.cc:16`

- [ ] **Step 1: Replace MLAParam mla_ with individual fields**

In `src/turbomind/models/attention_weight.h`:

Replace line 104:
```cpp
    bool is_mla() const { return mla_.kv_lora_rank > 0; }
```
with:
```cpp
    bool is_mla() const { return kv_lora_rank_ > 0; }
```

Replace line 111:
```cpp
    MLAParam mla_{};
```
with:
```cpp
    int kv_lora_rank_{};
    int q_lora_rank_{};
    int qk_rope_dim_{};
    int v_head_dim_{};
```

Remove `#include "src/turbomind/models/llama/llama_params.h"` if no other types from it are needed (check: `DataType` comes from `core/data_type.h`, `RopeConfig` comes from the same header). If MLAParam was the only reason for the include, remove it.

- [ ] **Step 2: Update AttentionWeight constructor**

In `src/turbomind/models/attention_weight.cc`, change line 16 from:
```cpp
    , mla_(MLAParam{cfg.q_lora_rank, cfg.kv_lora_rank, cfg.qk_rope_dim, cfg.v_head_dim})
```
to:
```cpp
    , kv_lora_rank_(cfg.kv_lora_rank)
    , q_lora_rank_(cfg.q_lora_rank)
    , qk_rope_dim_(cfg.qk_rope_dim)
    , v_head_dim_(cfg.v_head_dim)
```

- [ ] **Step 3: Build**

Run: `cd build && ninja`
Expected: Clean build. `forward_mla()` in unified_attention_layer.cc does not read `mla_` directly (it derives dims from weight sub-modules).

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/attention_weight.h src/turbomind/models/attention_weight.cc
git commit -m "refactor: eliminate MLAParam, use individual fields on AttentionWeight"
```

---

### Task 4: Add derivation logic to ModelWeight::prepare()

**Files:**
- Modify: `src/turbomind/models/model_weight.h:53,63-84`
- Modify: `src/turbomind/models/model_weight.cc:8-30`

- [ ] **Step 1: Add public derived fields to ModelWeight**

In `src/turbomind/models/model_weight.h`, remove accessors (lines 59-61):
```cpp
    int   hidden_units() const { return hidden_units_; }
    int   vocab_size_padded() const { return vocab_size_padded_; }
    int   tp_size() const { return tp_size_; }
```

Make the fields public. Change the private section (lines 63-84) -- move derived fields to public, remove dead storage:

```cpp
public:
    // Derived in prepare() from children
    DataType    data_type_{};
    int         hidden_units_{};
    int         vocab_size_{};
    int         vocab_size_padded_{};
    int         embedding_size_{};
    int         num_layer_{};
    int         head_dim_{};
    int         kv_head_num_{};
    std::vector<int> layer_types_;

    // From EngineParam at construction
    int         tp_size_{};
    int         tp_rank_{};

private:
    core::Stream    stream_{};
    core::Allocator alloca_{};

    mutable std::vector<DecoderLayerWeight*> layers_cache_;
```

Remove `model_param_`, `engine_param_`, `moe_param_` dead storage. Keep `initialized_` only if used.

- [ ] **Step 2: Update constructor to take only EngineParam**

Change constructor signature in header and .cc from:
```cpp
ModelWeight(DataType data_type, const ModelParam& model_param, const EngineParam& engine_param, const MoeParam& moe_param)
```
to:
```cpp
ModelWeight(const EngineParam& engine_param)
```

Update .cc constructor:
```cpp
ModelWeight::ModelWeight(const EngineParam& engine_param)
    : tp_size_(engine_param.attn_tp_size * engine_param.attn_cp_size)
    , tp_rank_(engine_param.attn_tp_rank)
{
    stream_ = core::Stream::create();
    alloca_ = core::Allocator{stream_, /*use_default_pool=*/true};
}
```

- [ ] **Step 3: Add derivation in prepare()**

Update `model_weight.cc` prepare():
```cpp
void ModelWeight::prepare()
{
    for_each_child([](const char* /*name*/, Module* child) {
        if (child) child->prepare();
    });

    // Derive model-level fields from first layer's attention weight
    auto* layer0 = layer(0);
    TM_CHECK(layer0 && layer0->attention);
    data_type_    = layer0->attention->data_type_;
    hidden_units_ = layer0->attention->hidden_dim_;
    head_dim_     = layer0->attention->head_dim_;
    kv_head_num_  = layer0->attention->kv_head_num_;

    vocab_size_        = tok_embeddings->weight.shape(0);
    embedding_size_    = vocab_size_;
    num_layer_         = layers->size();
    vocab_size_padded_ = round_up((size_t)vocab_size_, (size_t)tp_size_);

    // Derive layer_types from child structure
    layer_types_.resize(num_layer_);
    for (int i = 0; i < num_layer_; ++i) {
        layer_types_[i] = layer(i)->linear_attn ? 1 : 0;
    }
}
```

Note: Remove `#include "src/turbomind/models/llama/llama_params.h"` from model_weight.h since ModelParam/MoeParam are no longer used. Keep it only if EngineParam is still needed from that header (it is -- EngineParam still lives there).

- [ ] **Step 4: Update turbomind.cc call site**

In `src/turbomind/turbomind.cc`, change `CreateWeights()` line 143-146 from:
```cpp
    weights_[index] = std::make_shared<ModelWeight>(data_type_,
                                                    model_param_,
                                                    engine_params_.at(index),
                                                    moe_param_);
```
to:
```cpp
    weights_[index] = std::make_shared<ModelWeight>(engine_params_.at(index));
```

- [ ] **Step 5: Build**

Run: `cd build && ninja`
Expected: Clean build. `num_layers()` accessor removed -- update any call sites. LanguageModel still compiles because it reads via old accessor pattern temporarily... actually, we removed the accessors. Check if LanguageModel reads `hidden_units()` or `vocab_size_padded()` -- if so, update to direct field access.

Specifically, in `src/turbomind/turbomind.cc` CreateRequest() reads `model_param_.vocab_size` and `model_param_.hidden_units`. These must now come from `weights_[index]->vocab_size_` and `weights_[index]->hidden_units_`. But CreateRequest() is called from WarmUp (after weights loaded), so this works. Update:
```cpp
    return std::make_unique<ModelRequest>(gateway_.get(),
                                          data_type_,
                                          engine_param_.session_len,
                                          model_param_.vocab_size,
                                          model_param_.hidden_units);
```
to:
```cpp
    return std::make_unique<ModelRequest>(gateway_.get(),
                                          weights_[0]->data_type_,
                                          engine_param_.session_len,
                                          weights_[0]->vocab_size_,
                                          weights_[0]->hidden_units_);
```

Note: `weights_[0]` is valid because CreateRequest is called during WarmUp, after CreateWeights/ProcessWeights.

Similarly update WarmUp line 558:
```cpp
    std::uniform_int_distribution<int> d{0, (int)model_param_.vocab_size - 1};
```
to:
```cpp
    std::uniform_int_distribution<int> d{0, (int)weights_[index]->vocab_size_ - 1};
```

- [ ] **Step 6: Commit**

```bash
git add src/turbomind/models/model_weight.h src/turbomind/models/model_weight.cc src/turbomind/turbomind.cc
git commit -m "refactor: ModelWeight derives fields from children in prepare()"
```

---

### Task 5: Update InputProcessor -- remove ModelParam

**Files:**
- Modify: `src/turbomind/models/input_processor.h:12,17-18`
- Modify: `src/turbomind/models/input_processor.cc:17,34,244-245`

- [ ] **Step 1: Change InputProcessor constructor signature**

In `input_processor.h`, change constructor from:
```cpp
    InputProcessor(const EngineParam& engine, const ModelParam& model, int phases);
```
to:
```cpp
    InputProcessor(const EngineParam& engine, int hidden_units, DataType data_type, int phases);
```

Update `Impl` constructor similarly (line 17):
```cpp
    Impl(const EngineParam& engine, int hidden_units, DataType data_type, int phases);
```

- [ ] **Step 2: Update InputProcessor implementation**

In `input_processor.cc`, update Impl constructor. Change line 34:
```cpp
    d.input_embeds_buf = {{max_forward_token_num_, (int)model.hidden_units}, model.data_type, kCPUpinned};
```
to:
```cpp
    d.input_embeds_buf = {{max_forward_token_num_, hidden_units}, data_type, kCPUpinned};
```

Update public constructor (line 244):
```cpp
InputProcessor::InputProcessor(const EngineParam& engine, int hidden_units, DataType data_type, int phases):
    impl_{std::make_unique<Impl>(engine, hidden_units, data_type, phases)}
```

Remove `#include "src/turbomind/models/llama/llama_params.h"` if ModelParam was the only reason.

- [ ] **Step 3: Update LanguageModel call site**

In `src/turbomind/models/language_model.cc`, update line 160 from:
```cpp
    input_processor_.emplace(engine, param_, phases);
```
to:
```cpp
    input_processor_.emplace(engine, weights_.hidden_units_, weights_.data_type_, phases);
```

- [ ] **Step 4: Build**

Run: `cd build && ninja`
Expected: Clean build.

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/models/input_processor.h src/turbomind/models/input_processor.cc src/turbomind/models/language_model.cc
git commit -m "refactor(input-processor): remove ModelParam, take hidden_units and data_type"
```

---

### Task 6: Update OutputProcessor -- remove ModelParam

**Files:**
- Modify: `src/turbomind/models/output_processor.h:12`
- Modify: `src/turbomind/models/output_processor.cc:25-30,289-290`

- [ ] **Step 1: Change OutputProcessor constructor signature**

In `output_processor.h`, change from:
```cpp
    OutputProcessor(const ModelParam& model, int max_logits_len, int tp_rank, int phases, ...);
```
to:
```cpp
    OutputProcessor(int vocab_size, int max_logits_len, int tp_rank, int phases, ...);
```

Update `Impl` constructor similarly (line 25):
```cpp
    Impl(int vocab_size, int max_logits_len, int tp_rank, int phases, ...);
```

- [ ] **Step 2: Update OutputProcessor implementation**

In `output_processor.cc`, change line 30 from:
```cpp
    vocab_size_{(int)model.vocab_size},
```
to:
```cpp
    vocab_size_{vocab_size},
```

Update public constructor (line 289):
```cpp
OutputProcessor::OutputProcessor(
    int vocab_size, int max_logits_len, int tp_rank, int phases, std::function<Tensor(const Tensor&)> lm_head):
    impl_{std::make_unique<Impl>(vocab_size, max_logits_len, tp_rank, phases, std::move(lm_head))}
```

Remove `#include "src/turbomind/models/llama/llama_params.h"` if ModelParam was the only reason.

- [ ] **Step 3: Update LanguageModel call site**

In `language_model.cc`, update line 192 from:
```cpp
    output_processor_.emplace(param_, max_logits_len_, tp_rank_, phases, [this](const Tensor& hstate) {
```
to:
```cpp
    output_processor_.emplace(weights_.vocab_size_, max_logits_len_, tp_rank_, phases, [this](const Tensor& hstate) {
```

Note: `vocab_size_` on ModelWeight is the raw vocab size. The OutputProcessor's `vocab_size_` was `model.vocab_size` from ModelParam, which was also raw. So this matches.

- [ ] **Step 4: Build**

Run: `cd build && ninja`
Expected: Clean build.

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/models/output_processor.h src/turbomind/models/output_processor.cc src/turbomind/models/language_model.cc
git commit -m "refactor(output-processor): remove ModelParam, take vocab_size"
```

---

### Task 7: Update UnifiedDecoder -- remove ModelParam and MoeParam

**Files:**
- Modify: `src/turbomind/models/llama/unified_decoder.h:18-20`
- Modify: `src/turbomind/models/llama/unified_decoder.cc:33-79`

- [ ] **Step 1: Change UnifiedDecoder constructor signature**

In `unified_decoder.h`, change from:
```cpp
    UnifiedDecoder(const ModelParam& model,
                   const EngineParam& engine,
                   const MoeParam& moe,
                   const Context& ctx,
                   int phases,
                   const std::vector<DecoderLayerWeight*>& layer_weights);
```
to:
```cpp
    UnifiedDecoder(const EngineParam& engine,
                   const Context& ctx,
                   int phases,
                   const ModelWeight& model_weight);
```

- [ ] **Step 2: Update UnifiedDecoder constructor body**

In `unified_decoder.cc`, update the constructor:

Replace all `model.xxx` reads with `model_weight.xxx_`:
- `model.layer_num` → `model_weight.num_layer_`
- `model.hidden_units` → `model_weight.hidden_units_`
- `model.tune_layer_num` → `engine.tune_layer_num`

Replace `model.quant_policy` → `engine.quant_policy`.

Replace `model.layer_types` → `model_weight.layer_types_`.

Replace `model.linear_state_dtype` → `model_weight.data_type_`.

Replace MoE check:
```cpp
    if (std::accumulate(moe.expert_num.begin(), moe.expert_num.end(), 0LL)) {
```
with:
```cpp
    bool has_moe = false;
    for (int i = 0; i < model_weight.num_layer_; ++i) {
        if (model_weight.layer(i)->moe_ffn) {
            has_moe = true;
            break;
        }
    }
    if (has_moe) {
```

Replace FFN check:
```cpp
    if (std::accumulate(model.inter_size.begin(), model.inter_size.end(), 0LL)) {
```
with:
```cpp
    bool has_ffn = false;
    for (int i = 0; i < model_weight.num_layer_; ++i) {
        if (model_weight.layer(i)->feed_forward) {
            has_ffn = true;
            break;
        }
    }
    if (has_ffn) {
```

Replace attention weights extraction:
```cpp
    std::vector<AttentionWeight*> attn_weights;
    attn_weights.reserve(layer_weights.size());
    for (auto* lw : layer_weights) {
        attn_weights.push_back(lw->attention.get());
    }
```
with:
```cpp
    std::vector<AttentionWeight*> attn_weights;
    attn_weights.reserve(model_weight.num_layer_);
    for (int i = 0; i < model_weight.num_layer_; ++i) {
        attn_weights.push_back(model_weight.layer(i)->attention.get());
    }
```

Replace linear attention layer check:
```cpp
    if (std::find(model.layer_types.begin(), model.layer_types.end(), 1) != model.layer_types.end()) {
```
with:
```cpp
    bool has_linear_attn = false;
    for (auto t : model_weight.layer_types_) {
        if (t == 1) { has_linear_attn = true; break; }
    }
    if (has_linear_attn) {
```

Remove `#include "src/turbomind/models/llama/llama_params.h"` if no longer needed. Remove `#include <numeric>` if `accumulate` was the only use.

- [ ] **Step 3: Update LanguageModel call site**

In `language_model.cc`, update line 162 from:
```cpp
    unified_decoder_ = std::make_unique<UnifiedDecoder>(model, engine, moe, ctx, phases, weights_.layers_list());
```
to:
```cpp
    unified_decoder_ = std::make_unique<UnifiedDecoder>(engine, ctx, phases, weights_);
```

Note: This requires `weights_` to be a `const ModelWeight&` in LanguageModel::Impl, which it already is (line 39).

- [ ] **Step 4: Build**

Run: `cd build && ninja`
Expected: Clean build.

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/models/llama/unified_decoder.h src/turbomind/models/llama/unified_decoder.cc src/turbomind/models/language_model.cc
git commit -m "refactor(decoder): remove ModelParam and MoeParam, read from ModelWeight"
```

---

### Task 8: Update LanguageModel -- remove ModelParam, MoeParam, model_param()

**Files:**
- Modify: `src/turbomind/models/language_model.h:28-30,37`
- Modify: `src/turbomind/models/language_model.cc:104-195,491-510`

- [ ] **Step 1: Change LanguageModel constructor signature**

In `language_model.h`, change from:
```cpp
    LanguageModel(DataType dtype,
                  const ModelParam& model,
                  const EngineParam& engine,
                  const MoeParam& moe,
                  const Context& ctx,
                  const ModelWeight& weights,
                  int phases);
```
to:
```cpp
    LanguageModel(const EngineParam& engine,
                  const Context& ctx,
                  const ModelWeight& weights,
                  int phases);
```

Remove `const ModelParam& model_param() const noexcept;` accessor.

- [ ] **Step 2: Remove param_ and dtype_ from Impl**

In `language_model.cc`, remove from Impl struct:
- `const ModelParam param_;` (line 37)
- `const DataType dtype_;` (line 36)

Replace `param_.hidden_units` (line 201) with `weights_.hidden_units_`.

Replace `model.vocab_size` (line 167, local variable from param_) with `weights_.vocab_size_`.

Replace `model.hidden_units` in buffer sizing (line 181) with `weights_.hidden_units_`.

Replace `model.hidden_units` at line 189 with `weights_.hidden_units_`.

Replace `dtype_` references with `weights_.data_type_`.

- [ ] **Step 3: Update Impl constructor**

Change Impl constructor to not take ModelParam/MoeParam:
```cpp
Impl::Impl(const EngineParam& engine,
           const Context& ctx,
           const ModelWeight& weights,
           int phases):
    comm_{ctx.comm},
    weights_{weights},
    linear_{*ctx.linear},
    tp_size_{comm_.h_tp_group->n_ranks()},
    tp_rank_{comm_.h_tp_group->rank()},
    use_ag2d_{comm_.d_comm && comm_.d_comm->Query(comm::kHasAllGather2D)},
    debug_{isDebug()}
```

Update the forward declarations to match.

Update the public LanguageModel constructor similarly.

- [ ] **Step 4: Update turbomind.cc call site**

In `turbomind.cc`, update CreateEngine() lines 465-471 from:
```cpp
    LanguageModel model{data_type_,
                        model_param_,
                        param,
                        moe_param_,
                        ctx,
                        *weights_[index],
                        phases_};
```
to:
```cpp
    LanguageModel model{param,
                        ctx,
                        *weights_[index],
                        phases_};
```

- [ ] **Step 5: Build**

Run: `cd build && ninja`
Expected: Clean build. engine.cc still calls `model_.model_param()` -- will break. Fix by adding `ModelWeight&` to Engine first (Task 9).

Actually, engine.cc line 217 calls `model_.model_param()`. We just deleted that accessor. So we must fix engine.cc in this same task or Task 9 must happen simultaneously. Fix: in engine.cc, the two call sites that use `model_.model_param()` need to be updated now. But Engine doesn't have ModelWeight yet. So we must pass ModelWeight to Engine in this task too.

Update engine.cc `CreateSequenceManager()` -- temporarily inline the extraction from LanguageModel's weights... actually, the cleanest fix is to do Task 9 (Engine changes) in the same commit as this task.

**Revised: Combine Tasks 8 and 9 into one commit.**

- [ ] **Step 6: Commit (combined with Task 9)**

See Task 9 for the Engine changes. Commit together:
```bash
git add src/turbomind/models/language_model.h src/turbomind/models/language_model.cc src/turbomind/engine/engine.h src/turbomind/engine/engine.cc src/turbomind/turbomind.cc
git commit -m "refactor: remove ModelParam from LanguageModel, Engine reads from ModelWeight"
```

---

### Task 9: Update Engine and SequenceManager -- remove ModelParam

**Files:**
- Modify: `src/turbomind/engine/engine.h:29-36`
- Modify: `src/turbomind/engine/engine.cc:105-106,175-211,213-244,246-260`
- Modify: `src/turbomind/models/llama/SequenceManager.h:92`
- Modify: `src/turbomind/models/llama/SequenceManager.cc:34-107`

This task is committed together with Task 8.

- [ ] **Step 1: Pass ModelWeight to Engine constructor**

In `engine.h`, change constructor from:
```cpp
    Engine(DataType dtype, EngineParam param, LanguageModel model, Context& ctx, Gateway& gateway, int device_id, int queue_id, int phases);
```
to:
```cpp
    Engine(EngineParam param, LanguageModel model, const ModelWeight& weights, Context& ctx, Gateway& gateway, int device_id, int queue_id, int phases);
```

Remove `const DataType dtype_;` from Impl (line 105). Replace with `const ModelWeight& weights_;`.

- [ ] **Step 2: Update Engine constructor**

In `engine.cc`, update Impl constructor init list to include `weights_` and remove `dtype_`:
```cpp
    weights_{weights},
```

Update the forwarding constructor to match.

Update turbomind.cc CreateEngine() call:
```cpp
    engines_[index] = Engine{param,
                             std::move(model),
                             *weights_[index],
                             ctx,
                             *gateway_,
                             engine_param_.devices[index],
                             queue_id_[index],
                             phases_};
```

- [ ] **Step 3: Change SequenceManager constructor**

In `SequenceManager.h`, change from:
```cpp
    SequenceManager(const ModelParam& model_param, ...)
```
to take individual params:
```cpp
    SequenceManager(int head_dim, int kv_head_num, int num_layer,
                    const std::vector<int>& layer_types,
                    int quant_policy, DataType data_type, DataType runtime_dtype,
                    int linear_key_head_dim, int linear_value_head_dim,
                    int linear_conv_kernel_dim, int linear_num_key_heads,
                    int linear_num_value_heads,
                    int cache_block_seq_len, int attn_tp_size,
                    int max_batch_size, double block_count, int chunk_size,
                    bool enable_prefix_caching, int rank, int attn_cp_size,
                    core::Allocator allocator, GetFreeMemSize get_free_size);
```

Update `SequenceManager.cc` constructor body: replace `model_param.xxx` with the corresponding parameter name.

- [ ] **Step 4: Update Engine::CreateSequenceManager()**

Replace `model_.model_param()` extraction with `weights_` reads:

```cpp
void Engine::Impl::CreateSequenceManager()
{
    const auto cache_block_seq_len = param_.cache_block_seq_len;

    // Derive DeltaNet fields if linear attention exists
    int linear_key_head_dim = 0, linear_value_head_dim = 0;
    int linear_conv_kernel_dim = 0, linear_num_key_heads = 0, linear_num_value_heads = 0;
    for (int i = 0; i < weights_.num_layer_; ++i) {
        if (auto* dn = weights_.layer(i)->linear_attn.get()) {
            linear_key_head_dim   = dn->key_head_dim_;
            linear_value_head_dim = dn->value_head_dim_;
            linear_conv_kernel_dim = dn->d_conv_;
            linear_num_key_heads  = dn->num_k_heads_ * param_.attn_tp_size;
            linear_num_value_heads = dn->num_v_heads_ * param_.attn_tp_size;
            break;
        }
    }

    const auto get_free_size = [&] {
        size_t free{}, total{};
        check_cuda_error(cudaMemGetInfo(&free, &total));
        return AllReduce(tp_group_, free, comm::RedOp::kMin);
    };

    seq_mgr_ = std::make_unique<SequenceManager>(
        weights_.head_dim_, weights_.kv_head_num_ / param_.attn_tp_size,
        weights_.num_layer_, weights_.layer_types_,
        param_.quant_policy, weights_.data_type_, dtype_,
        linear_key_head_dim, linear_value_head_dim,
        linear_conv_kernel_dim, linear_num_key_heads, linear_num_value_heads,
        cache_block_seq_len, param_.attn_tp_size,
        param_.max_batch_size, param_.cache_max_block_count, param_.cache_chunk_size,
        param_.enable_prefix_caching, tp_rank_, param_.attn_cp_size,
        core::Context::alloc(kDEVICE), get_free_size);
    // ... rest stays the same
}
```

Note: `dtype_` (Engine's runtime dtype) is now `weights_.data_type_`. But wait, Engine's `dtype_` was the same as the model data_type. Use `weights_.data_type_` for `runtime_dtype` param.

Actually, Engine's `dtype_` was the constructor param. Now we derive it: `weights_.data_type_`.

- [ ] **Step 5: Update Engine::Validate()**

Replace:
```cpp
    const bool has_linear_attention = HasLinearAttention(model_.model_param());
```
with:
```cpp
    bool has_linear_attention = false;
    for (auto t : weights_.layer_types_) {
        if (t == 1) { has_linear_attention = true; break; }
    }
```

- [ ] **Step 6: Build**

Run: `cd build && ninja`
Expected: May have compilation errors from remaining references to ModelParam/MoeParam in language_model.cc/engine.cc. Fix iteratively.

- [ ] **Step 7: Commit (combined with Task 8)**

```bash
git add src/turbomind/models/language_model.h src/turbomind/models/language_model.cc src/turbomind/engine/engine.h src/turbomind/engine/engine.cc src/turbomind/turbomind.cc src/turbomind/models/llama/SequenceManager.h src/turbomind/models/llama/SequenceManager.cc
git commit -m "refactor: remove ModelParam from LanguageModel and Engine, SequenceManager takes individual params"
```

---

### Task 10: Delete ModelParam, MoeParam, MLAParam, HasLinearAttention

**Files:**
- Modify: `src/turbomind/models/llama/llama_params.h`
- Modify: `src/turbomind/turbomind.cc`

- [ ] **Step 1: Delete structs from llama_params.h**

Remove from `llama_params.h`:
- `struct MLAParam` (lines 15-20)
- `struct ModelParam` (lines 22-58)
- `HasLinearAttention` function (lines 60-68)
- `struct MoeParam` including `Method` enum (lines 71-93)

Keep only `EngineParam` (now with `quant_policy` and `tune_layer_num` added in Task 1).

- [ ] **Step 2: Delete model_config parsing in turbomind.cc**

Remove all parsing of `node["model_config"]`:
- Lines 222-289 (model_config field parsing)
- Line 296 (`session_len` from model -- already reading from engine_config in EngineParam)
- Lines 336-365 (MoE field parsing from model_config)

Remove `model_param_` and `moe_param_` from TurboMind::Impl (line 93-94).

Remove `model_param_{}` and `moe_param_{}` from constructor init list (line 209).

Remove `data_type_` assignment from model_config (line 226). Parse from engine_config instead:
```cpp
    data_type_ = data_type_from_string(engine["data_type"].as<std::string>());
```

Remove `model_name_` and its parsing (line 229).

Remove `HasLinearAttention` check (line 303) -- already moved to CreateEngine in Task 4.

Update `session_len` to read from engine_config:
```cpp
    engine_param_.session_len = engine["session_len"].as<int>(0);
```

Update `tune_layer_num` to read from engine_config:
```cpp
    engine_param_.tune_layer_num = engine["tune_layer_num"].as<int>(1);
```

Remove `get_moe_method()` function's references to `moe_param_`. The MoE method is now derived from MoeWeight at runtime, not from config. The env var override `TM_MOE_METHOD` can still work -- apply it when creating MoeFfnLayer or ignore it (MoeWeight already has the method from its config).

Actually, `get_moe_method()` was used to override `moe_param_.method`. Since we no longer have `moe_param_`, and MoeWeight gets its method from MoeConfig (set by Python), we can delete `get_moe_method()` entirely. The env var override is a development tool that's no longer needed since the method is set per-model in Python.

- [ ] **Step 3: Clean up includes**

Remove `#include "src/turbomind/models/llama/llama_params.h"` from files that only needed ModelParam/MoeParam:
- `src/turbomind/models/model_weight.h` -- still needs EngineParam
- `src/turbomind/models/language_model.h` -- no longer needs it
- `src/turbomind/models/output_processor.h` -- no longer needs it
- `src/turbomind/models/input_processor.h` -- no longer needs it
- `src/turbomind/models/llama/unified_decoder.h` -- no longer needs it
- `src/turbomind/models/llama/SequenceManager.h` -- no longer needs it
- `src/turbomind/models/llama/unified_attention_layer.h` -- check if still needed
- `src/turbomind/models/llama/GatedDeltaNetLayer.h` -- check if still needed

Note: EngineParam still lives in `llama_params.h`, so files that need EngineParam must still include it.

- [ ] **Step 4: Build**

Run: `cd build && ninja`
Expected: Fix any remaining compilation errors iteratively. All ModelParam/MoeParam references should be gone.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: delete ModelParam, MoeParam, MLAParam; stop parsing model_config YAML"
```

---

### Task 11: Test all models

**Files:** None (testing only)

- [ ] **Step 1: Check GPU availability**

Use `get_gpu_usage` MCP tool to find an empty GPU.

- [ ] **Step 2: Run a quick model test**

```bash
python scripts/test_turbomind_model.py <model-name> --gpu <gpu-id> --prompt "Hello, how are you?" --max-new-tokens 128
```

Pick a standard model (e.g., Qwen2.5-7B-Instruct or similar from `list_models`). Verify the response contains meaningful words.

- [ ] **Step 3: Test a MoE model** (if available)

```bash
python scripts/test_turbomind_model.py <moe-model-name> --gpu <gpu-id> --prompt "Explain quantum computing" --max-new-tokens 128
```

- [ ] **Step 4: Test an MLA model** (if available, e.g., DeepSeek-V2-Lite)

```bash
python scripts/test_turbomind_model.py <mla-model-name> --gpu <gpu-id> --prompt "What is machine learning?" --max-new-tokens 128
```

- [ ] **Step 5: Commit if any fixes were needed**

If bugs were found and fixed during testing, commit the fixes.
