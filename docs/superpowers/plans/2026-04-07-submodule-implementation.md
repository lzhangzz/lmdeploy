# Submodule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the CRTP `Module<Derived>` pattern with auto-registering `Submodule<T>`, reducing each child from 3 declarations (member pointer, kChildren entry, accessor) to 1 (Submodule member).

**Architecture:** `Submodule<T>` stores a `const char* name` and `Module* pointer`. Its constructor registers a slot with the parent Module. `Module::add_child` iterates the child's slots, matches by name against existing children, and assigns the typed pointer. The CRTP layer and kChildren are eliminated entirely.

**Tech Stack:** C++17, pybind11

---

## File Structure

| File | Change |
|------|--------|
| `src/turbomind/core/module.h` | Add `Submodule<T>`, `add_slot`, `slots_`; delete `Module<Derived>` CRTP; rename `ModuleBase` → `Module` |
| `src/turbomind/core/module.cc` | Rename all `ModuleBase` → `Module` |
| `src/turbomind/models/attention_weight.h` | Replace 3-part pattern with Submodule members |
| `src/turbomind/models/attention_weight.cc` | Update convenience accessors |
| `src/turbomind/models/delta_net_weight.h` | Replace 3-part pattern with Submodule members |
| `src/turbomind/models/ffn_weight.h` | Replace 3-part pattern with Submodule members |
| `src/turbomind/models/ffn_weight.cc` | Remove `mutable`, update accessor usage |
| `src/turbomind/models/decoder_layer_weight.h` | Replace 3-part pattern with Submodule members |
| `src/turbomind/models/decoder_layer_weight.cc` | Update verify() to use Submodule |
| `src/turbomind/models/model_weight.h` | Replace 3-part pattern with Submodule members |
| `src/turbomind/models/model_weight.cc` | Update layer()/layers()/verify() to use Submodule |
| `src/turbomind/models/moe_weight.h` | Replace 3-part pattern with Submodule members |
| `src/turbomind/models/moe_weight.cc` | Update expert(), alloc(), prepare() |
| `src/turbomind/python/bind.cpp` | Update `Module<MoeWeight>::alloc` reference, any `ModuleBase` references |
| All files using `ModuleBase` | Rename to `Module` |

---

### Task 1: Add Submodule\<T\> to module.h and restructure Module base

**Files:**
- Modify: `src/turbomind/core/module.h`

- [ ] **Step 1: Add Submodule\<T\> template and update Module class**

In `module.h`, make these changes:

1. Add the `Submodule<T>` template (before the `Module` class):

```cpp
template<class T>
struct Submodule {
    const char* name;
    Module*     pointer = nullptr;

    Submodule(Module& parent, const char* n): name(n)
    {
        parent.add_slot(n, &pointer);
    }

    explicit operator bool() const { return pointer != nullptr; }

    operator T*() const
    {
        return static_cast<T*>(TM_CHECK_NOTNULL(pointer));
    }

    T* operator->() const
    {
        return static_cast<T*>(TM_CHECK_NOTNULL(pointer));
    }
};
```

Note: `Submodule` references `Module` — this requires the forward declaration of `Module` to exist before `Submodule`, and `Module`'s definition to be visible before any `Submodule` member is instantiated (which happens in derived classes, so this is fine).

2. Rename `ModuleBase` → `Module` throughout the file.

3. Add to `Module`'s protected section:

```cpp
std::vector<std::pair<const char*, Module**>> slots_;

void add_slot(const char* name, Module** pp)
{
    slots_.emplace_back(name, pp);
}
```

4. Update `Module::add_child` (was `ModuleBase::add_child`) to wire slots before storing:

```cpp
virtual Module* add_child(std::string name, std::unique_ptr<Module> child)
{
    TM_CHECK(child != nullptr);
    TM_CHECK(child->parent_ == nullptr) << "module already has a parent";

    // Wire child's slots against existing children
    for (auto& [slot_name, pp] : child->slots_) {
        for (auto& [cname, cptr] : children_) {
            if (cname == slot_name) {
                *pp = cptr;
                break;
            }
        }
    }

    child->parent_ = this;
    child->name_   = name;

    Module* raw = child.get();
    children_.emplace_back(std::move(name), std::move(child));
    return raw;
}
```

5. Delete the entire `Module<Derived>` CRTP template (lines 174-238).

6. Delete the `detail` namespace (lines 21-36) — it's only used by the CRTP.

7. Delete the forward declaration `template<typename Derived> class Module;` (line 45-46) — no longer needed.

8. Update `ModuleList` to inherit from `Module` instead of `ModuleBase`:

```cpp
class ModuleList: public Module {
```

9. Update all `ModuleBase` references in the file to `Module`:
   - `virtual ~Module()` instead of `virtual ~ModuleBase()`
   - `Module()` instead of `ModuleBase()`
   - `Module* parent_` instead of `ModuleBase* parent_`
   - Return types, parameter types, etc.

- [ ] **Step 2: Update module.cc — rename ModuleBase → Module**

In `src/turbomind/core/module.cc`, replace all occurrences of `ModuleBase` with `Module`:
- `ModuleBase::ModuleBase()` → `Module::Module()`
- `ModuleBase::~ModuleBase()` → `Module::~Module()`
- `ModuleBase::add_child` → `Module::add_child`
- `ModuleBase::type()` → `Module::type()`
- All other method definitions
- `ModuleList::add_child` — `ModuleBase::add_child` call → `Module::add_child`
- The registrar factory returning `std::make_unique<core::ModuleList>()` — no change needed (ModuleList already correct)

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/core/module.h src/turbomind/core/module.cc
git commit -m "refactor(module): add Submodule<T>, delete CRTP, rename ModuleBase → Module"
```

---

### Task 2: Migrate AttentionWeight

**Files:**
- Modify: `src/turbomind/models/attention_weight.h`
- Modify: `src/turbomind/models/attention_weight.cc`

- [ ] **Step 1: Rewrite attention_weight.h**

Replace the class body. Full new class definition:

```cpp
class AttentionWeight: public core::Module {
public:
    static constexpr const char* kTypeName = "AttentionWeight";
    const char* type() const override { return kTypeName; }

    AttentionWeight() = default;

    AttentionWeight(int          hidden_dim,
                    int          head_dim,
                    int          head_num,
                    int          kv_head_num,
                    MLAParam     mla,
                    bool         bias,
                    bool         qk_norm,
                    int          tp_size,
                    int          tp_rank,
                    DataType     data_type,
                    int          window_size,
                    bool         sink,
                    bool         attn_output_gate);

    void prepare() override;

    // --- Typed child members ---
    Submodule<LinearWeight> w_qkv             {*this, "w_qkv"};
    Submodule<LinearWeight> wo                {*this, "wo"};
    Submodule<LinearWeight> q_proj            {*this, "q_proj"};
    Submodule<LinearWeight> q_a_proj          {*this, "q_a_proj"};
    Submodule<LinearWeight> q_b_proj          {*this, "q_b_proj"};
    Submodule<LinearWeight> kv_a_proj         {*this, "kv_a_proj"};
    Submodule<NormWeight>   q_norm_mod        {*this, "q_norm"};
    Submodule<NormWeight>   k_norm_mod        {*this, "k_norm"};
    Submodule<NormWeight>   q_a_layernorm_mod {*this, "q_a_layernorm"};
    Submodule<NormWeight>   kv_a_layernorm_mod{*this, "kv_a_layernorm"};
    Submodule<NormWeight>   sinks_mod         {*this, "sinks"};

    // Convenience tensor accessors
    Tensor* q_norm() const;
    Tensor* k_norm() const;
    Tensor* q_a_layernorm() const;
    Tensor* kv_a_layernorm() const;
    Tensor* sinks() const;

    int  window_size() const { return window_size_; }
    bool is_mla() const { return mla_.kv_lora_rank > 0; }

private:
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
};
```

Deleted: 11 member pointers, kChildren tuple, `friend class`, 11 typed accessors.

- [ ] **Step 2: Update attention_weight.cc convenience accessors**

The convenience tensor accessors currently use `q_norm_mod_` (old member pointer). Update to use the Submember member names:

```cpp
Tensor* AttentionWeight::q_norm() const
{
    return q_norm_mod ? &q_norm_mod->weight() : nullptr;
}
Tensor* AttentionWeight::k_norm() const
{
    return k_norm_mod ? &k_norm_mod->weight() : nullptr;
}
Tensor* AttentionWeight::q_a_layernorm() const
{
    return q_a_layernorm_mod ? &q_a_layernorm_mod->weight() : nullptr;
}
Tensor* AttentionWeight::kv_a_layernorm() const
{
    return kv_a_layernorm_mod ? &kv_a_layernorm_mod->weight() : nullptr;
}
Tensor* AttentionWeight::sinks() const
{
    return sinks_mod ? &sinks_mod->weight() : nullptr;
}
```

Note: `q_norm_mod` (Submodule) implicitly converts to `NormWeight*` via `operator bool()` check and `operator T*()` for `->weight()`.

- [ ] **Step 3: Build and commit**

```bash
cd build && ninja src/turbomind/models/CMakeFiles/turbomind.dir/attention_weight.cc.o
```

```bash
git add src/turbomind/models/attention_weight.h src/turbomind/models/attention_weight.cc
git commit -m "refactor(AttentionWeight): migrate to Submodule<T>"
```

---

### Task 3: Migrate DeltaNetWeight

**Files:**
- Modify: `src/turbomind/models/delta_net_weight.h`

- [ ] **Step 1: Rewrite delta_net_weight.h**

```cpp
class DeltaNetWeight: public core::Module {
public:
    static constexpr const char* kTypeName = "DeltaNetWeight";
    const char* type() const override { return kTypeName; }

    DeltaNetWeight() = default;

    DeltaNetWeight(int      hidden_dim,
                   int      num_k_heads,
                   int      num_v_heads,
                   int      key_head_dim,
                   int      value_head_dim,
                   int      d_conv,
                   bool     bias,
                   int      tp_size,
                   int      tp_rank,
                   DataType data_type);

    void prepare() override;

    // --- Typed child members ---
    Submodule<LinearWeight> in_proj_all {*this, "in_proj_all"};
    Submodule<LinearWeight> out_proj    {*this, "out_proj"};
    Submodule<NormWeight>   conv1d      {*this, "conv1d"};
    Submodule<NormWeight>   A_log       {*this, "A_log"};
    Submodule<NormWeight>   dt_bias     {*this, "dt_bias"};
    Submodule<NormWeight>   norm        {*this, "norm"};

private:
    int      hidden_dim_{};
    int      num_k_heads_{};
    int      num_v_heads_{};
    int      key_head_dim_{};
    int      value_head_dim_{};
    int      d_conv_{};
    bool     bias_{};
    int      tp_size_{};
    int      tp_rank_{};
    DataType data_type_{};
};
```

No .cc changes needed — delta_net_weight.cc only uses `children_` iteration.

- [ ] **Step 2: Build and commit**

```bash
cd build && ninja src/turbomind/models/CMakeFiles/turbomind.dir/delta_net_weight.cc.o
```

```bash
git add src/turbomind/models/delta_net_weight.h
git commit -m "refactor(DeltaNetWeight): migrate to Submodule<T>"
```

---

### Task 4: Migrate FfnWeight

**Files:**
- Modify: `src/turbomind/models/ffn_weight.h`
- Modify: `src/turbomind/models/ffn_weight.cc`

- [ ] **Step 1: Rewrite ffn_weight.h**

```cpp
class FfnWeight: public core::Module {
public:
    static constexpr const char* kTypeName = "FfnWeight";
    const char* type() const override { return kTypeName; }

    FfnWeight() = default;

    FfnWeight(int hidden_dim, int inter_size, bool bias, int tp_size, int tp_rank,
              DataType data_type, ActivationType act_type, bool fuse_silu_act);

    void prepare() override;

    // --- Typed child members ---
    Submodule<LinearWeight> w1  {*this, "w1"};
    Submodule<LinearWeight> w3  {*this, "w3"};
    Submodule<LinearWeight> w2  {*this, "w2"};
    Submodule<LinearWeight> w1w3{*this, "w1w3"};

    int            inter_size() const { return inter_size_; }
    ActivationType act_type() const { return act_type_; }
    bool           is_fused_silu() const { return is_fused_silu_; }

    /// Set grouped-GEMM mode for MoE (affects weight conversion layout).
    void set_fused_moe(bool fused_moe) { is_fused_moe_ = fused_moe; }

    /// Override is_fused_silu_ (used by MoE block view after linking experts).
    void set_fused_silu(bool val) { is_fused_silu_ = val; }

private:
    int           hidden_dim_{};
    int           inter_size_{};
    bool          bias_{};
    int           tp_size_{};
    int           tp_rank_{};
    DataType      data_type_{};
    ActivationType act_type_{};
    bool          is_fused_silu_{};
    bool          is_fused_moe_{};
};
```

Deleted: `mutable` keyword, kChildren, friend, 4 accessors.

- [ ] **Step 2: Update ffn_weight.cc**

Line 29: `auto* fused = w1w3()` → `auto* fused = w1w3;` (implicit conversion)

```cpp
void FfnWeight::prepare()
{
    // Set epilogue on existing w1w3 child if fused silu is active.
    if (auto* fused = w1w3) {
        if (is_fused_silu_) {
            fused->epilogue = gemm::Epilogue::kGatedSilu;
        }
    }

    // Prepare (format conversion) for all children
    for (auto& [name, child] : children_) {
        if (is_fused_moe_) {
            if (auto* lw = dynamic_cast<LinearWeight*>(child.get())) {
                lw->set_grouped(true);
            }
        }
        child->prepare();
    }
}
```

- [ ] **Step 3: Build and commit**

```bash
cd build && ninja src/turbomind/models/CMakeFiles/turbomind.dir/ffn_weight.cc.o
```

```bash
git add src/turbomind/models/ffn_weight.h src/turbomind/models/ffn_weight.cc
git commit -m "refactor(FfnWeight): migrate to Submodule<T>"
```

---

### Task 5: Migrate DecoderLayerWeight

**Files:**
- Modify: `src/turbomind/models/decoder_layer_weight.h`
- Modify: `src/turbomind/models/decoder_layer_weight.cc`

- [ ] **Step 1: Rewrite decoder_layer_weight.h**

```cpp
class DecoderLayerWeight: public core::Module {
public:
    static constexpr const char* kTypeName = "DecoderLayerWeight";
    const char* type() const override { return kTypeName; }

    DecoderLayerWeight() = default;

    bool verify(std::vector<std::string>& missing) override;

    // --- Typed child members ---
    Submodule<AttentionWeight> attention    {*this, "attention"};
    Submodule<DeltaNetWeight>  linear_attn  {*this, "linear_attn"};
    Submodule<FfnWeight>       feed_forward {*this, "feed_forward"};
    Submodule<MoeWeight>       moe_ffn      {*this, "moe_ffn"};
    Submodule<NormWeight>      attn_norm    {*this, "attention_norm"};
    Submodule<NormWeight>      ffn_norm     {*this, "ffn_norm"};
};
```

Note: `attn_norm` Submodule stores `"attention_norm"` as the child name (matching what Python creates), but the member is named `attn_norm`.

Deleted: kChildren, friend, 6 accessors.

- [ ] **Step 2: Update decoder_layer_weight.cc**

The verify() method accesses `attention_`, `linear_attn_`, `feed_forward_`, `moe_ffn_`, `attn_norm_` directly. Update to use Submodule members:

```cpp
bool DecoderLayerWeight::verify(std::vector<std::string>& missing)
{
    Module::verify(missing);
    // At least one of attention or linear_attn must exist
    if (!attention && !linear_attn) {
        missing.push_back(full_path() + ": missing attention or linear_attn");
    }
    // At least one of feed_forward or moe_ffn must exist
    if (!feed_forward && !moe_ffn) {
        missing.push_back(full_path() + ": missing feed_forward or moe_ffn");
    }
    // attention_norm must exist
    if (!attn_norm) {
        missing.push_back(full_path() + ": missing attention_norm");
    }
    return missing.empty();
}
```

Note: `ModuleBase::verify(missing)` → `Module::verify(missing)` (renamed).

- [ ] **Step 3: Build and commit**

```bash
cd build && ninja src/turbomind/models/CMakeFiles/turbomind.dir/decoder_layer_weight.cc.o
```

```bash
git add src/turbomind/models/decoder_layer_weight.h src/turbomind/models/decoder_layer_weight.cc
git commit -m "refactor(DecoderLayerWeight): migrate to Submodule<T>"
```

---

### Task 6: Migrate ModelWeight

**Files:**
- Modify: `src/turbomind/models/model_weight.h`
- Modify: `src/turbomind/models/model_weight.cc`

- [ ] **Step 1: Rewrite model_weight.h**

```cpp
class ModelWeight: public core::Module {
public:
    static constexpr const char* kTypeName = "ModelWeight";
    const char* type() const override { return kTypeName; }

    ModelWeight() = default;

    ModelWeight(DataType          data_type,
                const ModelParam&  model_param,
                const EngineParam& engine_param,
                const MoeParam&    moe_param);

    void                    prepare() override;
    bool                    verify(std::vector<std::string>& missing) override;

    core::ContextGuard context() const
    {
        return core::ContextGuard{stream_, alloca_};
    }

    // --- Typed child members ---
    Submodule<LinearWeight>  tok_embeddings {*this, "tok_embeddings"};
    Submodule<LinearWeight>  output         {*this, "output"};
    Submodule<NormWeight>    norm           {*this, "norm"};
    Submodule<ModuleList>    layers         {*this, "layers"};

    DecoderLayerWeight*  layer(int i) const;
    std::vector<DecoderLayerWeight*> layers_list() const;
    int                  num_layers() const { return num_layer_; }

    // --- Lifecycle ---
    bool is_initialized() const { return initialized_; }

    // --- Model config accessors ---
    int   hidden_units() const { return hidden_units_; }
    int   vocab_size_padded() const { return vocab_size_padded_; }
    int   tp_size() const { return tp_size_; }

private:
    DataType    data_type_{};
    ModelParam  model_param_{};
    EngineParam engine_param_{};
    MoeParam    moe_param_{};

    size_t hidden_units_{};
    size_t vocab_size_{};
    size_t vocab_size_padded_{};
    size_t embedding_size_{};
    size_t num_layer_{};

    int  tp_size_{};
    int  tp_rank_{};

    bool initialized_{false};

    core::Stream    stream_{};
    core::Allocator alloca_{};

    mutable std::vector<DecoderLayerWeight*> layers_cache_;
};
```

Note: The old `layers()` method returns `std::vector<DecoderLayerWeight*>`. The Submodule member is also named `layers`. Rename the method to `layers_list()` to avoid name collision with the Submodule member.

Deleted: kChildren, friend, 4 accessors.

- [ ] **Step 2: Update model_weight.cc**

```cpp
void ModelWeight::prepare()
{
    for (auto& [name, child] : children_) {
        child->prepare();
    }
}

DecoderLayerWeight* ModelWeight::layer(int i) const
{
    if (!layers) {
        return nullptr;
    }
    return static_cast<DecoderLayerWeight*>(layers->child(std::to_string(i)));
}

std::vector<DecoderLayerWeight*> ModelWeight::layers_list() const
{
    if (!layers_cache_.empty()) {
        return layers_cache_;
    }
    if (!layers) {
        return {};
    }
    layers_cache_.resize(layers->size());
    for (int i = 0; i < layers->size(); ++i) {
        layers_cache_[i] = static_cast<DecoderLayerWeight*>(layers->child(std::to_string(i)));
    }
    return layers_cache_;
}

bool ModelWeight::verify(std::vector<std::string>& missing)
{
    Module::verify(missing);
    if (!tok_embeddings) {
        missing.push_back(full_path() + ": missing tok_embeddings");
    }
    if (!norm) {
        missing.push_back(full_path() + ": missing norm");
    }
    return missing.empty();
}
```

- [ ] **Step 3: Update callers of ModelWeight::layers()**

Search for calls to `ModelWeight::layers()` that return a vector and update to `layers_list()`:

```bash
grep -rn "\.layers()" src/turbomind/ --include="*.cc" --include="*.h"
```

- [ ] **Step 4: Build and commit**

```bash
cd build && ninja
```

```bash
git add src/turbomind/models/model_weight.h src/turbomind/models/model_weight.cc
git commit -m "refactor(ModelWeight): migrate to Submodule<T>"
```

---

### Task 7: Migrate MoeWeight

**Files:**
- Modify: `src/turbomind/models/moe_weight.h`
- Modify: `src/turbomind/models/moe_weight.cc`

- [ ] **Step 1: Rewrite moe_weight.h**

```cpp
class MoeWeight: public core::Module {
public:
    static constexpr const char* kTypeName = "MoeWeight";
    const char* type() const override { return kTypeName; }

    MoeWeight() = default;

    MoeWeight(int              layer_id,
              const MoeParam&  param,
              int              hidden_dim,
              bool             mlp_bias,
              DataType         data_type,
              int              tp_size,
              int              tp_rank,
              ActivationType   act_type,
              bool             fuse_silu_act);

    Tensor  alloc(const std::string& param_name, const core::WeightSpec& spec) override;
    void prepare() override;
    int num_experts() const { return expert_num_; }

    // --- Typed child members ---
    Submodule<LinearWeight>  gate        {*this, "gate"};
    Submodule<LinearWeight>  shared_gate {*this, "shared_gate"};
    Submodule<ModuleList>    experts     {*this, "experts"};

    FfnWeight*    expert(int i) const;
    FfnWeight*    block() const { return block_.get(); }
    Tensor*       score_correction_bias() const { return const_cast<Tensor*>(param("score_correction_bias")); }
    MoeParam::Method method() const { return moe_param_.method; }
    const MoeParam& moe_param() const { return moe_param_; }

private:
    // ... same private members ...
};
```

- [ ] **Step 2: Update moe_weight.cc**

Key changes:

1. Line 44: `return Module<MoeWeight>::alloc(param_name, spec);` → `return Module::alloc(param_name, spec);`

2. Lines 107-110: `expert()` uses `experts_` → `experts`:
```cpp
FfnWeight* MoeWeight::expert(int i) const
{
    if (!experts) {
        return nullptr;
    }
    return static_cast<FfnWeight*>(experts->child(std::to_string(i)));
}
```

3. Lines 132-147: Lambda captures use accessors like `exp->w1w3()` → `exp->w1w3`:
```cpp
auto get_expert_w1w3 = [this](int i) -> LinearWeight* {
    auto* exp = expert(i);
    return exp ? static_cast<LinearWeight*>(exp->w1w3) : nullptr;
};
```

Wait — `exp->w1w3` triggers implicit conversion to `LinearWeight*`. The static_cast is not needed. But we need to be careful: `exp` is `FfnWeight*`, and `exp->w1w3` is a `Submodule<LinearWeight>`. The implicit `operator LinearWeight*()` does the conversion. So:

```cpp
auto get_expert_w1w3 = [this](int i) -> LinearWeight* {
    auto* exp = expert(i);
    return exp ? exp->w1w3 : nullptr;
};
```

Same for `get_expert_w1`, `get_expert_w3`, `get_expert_w2`.

4. Lines 151-168: `block_->w1w3()` → `block_->w1w3`:
```cpp
if (get_expert_w1w3(0)) {
    block_->add_child("w1w3", std::make_unique<LinearWeight>());
    LinkLinearExperts(get_expert_w1w3, expert_num_, *block_->w1w3);
}
```

But wait — `block_->w1w3` triggers `operator->()` on Submodule, which returns `LinearWeight*`. Then `*` dereferences to `LinearWeight&`. So `*block_->w1w3` is `LinearWeight&`. This should work.

Actually, `block_->w1w3` — `block_` is `unique_ptr<FfnWeight>`. `block_->w1w3` accesses the Submodule member on the FfnWeight. The `->` here is `unique_ptr::operator->`, not `Submodule::operator->`. So `block_->w1w3` is `Submodule<LinearWeight>&`. Then `*block_->w1w3` — `operator*` on Submodule isn't defined, only `operator T*()`. So this would be `*static_cast<LinearWeight*>(block_->w1w3)`, which needs to be explicit.

The fix: use `*block_->w1w3.operator LinearWeight*()` or just store in a local:

```cpp
if (get_expert_w1w3(0)) {
    block_->add_child("w1w3", std::make_unique<LinearWeight>());
    LinkLinearExperts(get_expert_w1w3, expert_num_, *block_->w1w3);
}
```

Actually, `*block_->w1w3` — C++ will try to apply `operator*` to `Submodule<LinearWeight>`. Submodule doesn't define `operator*`. It defines `operator T*()`. So `block_->w1w3` converts to `LinearWeight*`, then `*` dereferences. Wait, does implicit conversion participate in `operator*` resolution?

In C++, `*expr` requires `expr` to be a pointer type. The implicit conversion `operator LinearWeight*()` returns `LinearWeight*`, which IS a pointer. So `*block_->w1w3` should work: first convert to `LinearWeight*` via implicit conversion, then dereference.

Let me double-check: `block_->w1w3` has type `Submodule<LinearWeight>`. `*block_->w1w3` — the compiler looks for `operator*` on `Submodule<LinearWeight>`. Not found. Then it tries implicit conversions: `operator LinearWeight*()` → `LinearWeight*`. Then `*LinearWeight*` → `LinearWeight&`. Yes, this works.

OK so `*block_->w1w3` is fine.

- [ ] **Step 3: Build and commit**

```bash
cd build && ninja src/turbomind/models/CMakeFiles/turbomind.dir/moe_weight.cc.o
```

```bash
git add src/turbomind/models/moe_weight.h src/turbomind/models/moe_weight.cc
git commit -m "refactor(MoeWeight): migrate to Submodule<T>"
```

---

### Task 8: Update callers of removed accessors

**Files:**
- Modify: `src/turbomind/models/llama/unified_decoder.cc` (lines 186-270)
- Modify: `src/turbomind/models/language_model.cc` (line 444)

- [ ] **Step 1: Update unified_decoder.cc**

Replace accessor calls with Submodule member access. All changes are in the `Forward` method (lines 186-270):

| Line | Old | New |
|------|-----|-----|
| 186 | `weights.at(0)->attn_norm()->weight()` | `weights.at(0)->attn_norm->weight()` |
| 209 | `if (weights.at(layer)->linear_attn())` | `if (weights.at(layer)->linear_attn)` |
| 211 | `weights.at(layer)->linear_attn()` | `weights.at(layer)->linear_attn` |
| 214 | `auto* attn = weights.at(layer)->attention()` | `auto* attn = weights.at(layer)->attention` |
| 224 | `if (weights.at(layer)->linear_attn())` | `if (weights.at(layer)->linear_attn)` |
| 225 | `weights.at(layer)->linear_attn()->out_proj()->bias` | `weights.at(layer)->linear_attn->out_proj->bias` |
| 228 | `weights.at(layer)->attention()->wo()->bias` | `weights.at(layer)->attention->wo->bias` |
| 234 | `weights.at(layer)->ffn_norm()->weight()` | `weights.at(layer)->ffn_norm->weight()` |
| 248 | `if (weights.at(layer)->moe())` | `if (weights.at(layer)->moe_ffn)` |
| 251 | `weights.at(layer)->moe()` | `weights.at(layer)->moe_ffn` |
| 257 | `if (ffn_layer_ && weights.at(layer)->ffn())` | `if (ffn_layer_ && weights.at(layer)->feed_forward)` |
| 259 | `weights.at(layer)->ffn()` | `weights.at(layer)->feed_forward` |
| 270 | `weights.at(layer + 1)->attn_norm()->weight()` | `weights.at(layer + 1)->attn_norm->weight()` |

Note: accessor `moe()` → Submodule member `moe_ffn`, accessor `ffn()` → Submodule member `feed_forward`. These names differ because the old accessors had short names but the Submodule members use the same name as the child name in the parent.

- [ ] **Step 2: Update language_model.cc**

Line 444: `weights_.layers()` → `weights_.layers_list()`

- [ ] **Step 3: Build and commit**

```bash
cd build && ninja
```

```bash
git add src/turbomind/models/llama/unified_decoder.cc src/turbomind/models/language_model.cc
git commit -m "refactor: update callers for Submodule migration"
```

---

### Task 9: Update remaining ModuleBase references and pybind

**Files:**
- Modify: `src/turbomind/python/bind.cpp`
- Modify: Any other files referencing `ModuleBase`

- [ ] **Step 1: Find all remaining ModuleBase references**

```bash
grep -rn "ModuleBase" src/ --include="*.h" --include="*.cc" --include="*.cpp"
```

- [ ] **Step 2: Update each reference to Module**

In `bind.cpp`, update:
- `dynamic_cast<ft::ModelWeight*>(root)` — unchanged (concrete type)
- `dynamic_cast<turbomind::FfnWeight*>(&m)` — unchanged
- Any `core::ModuleBase` references → `core::Module`

In any other files, rename `ModuleBase` → `Module`.

- [ ] **Step 3: Build full project**

```bash
cd build && ninja
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: rename remaining ModuleBase references to Module"
```

---

### Task 10: Full build verification

- [ ] **Step 1: Clean build**

```bash
cd build && ninja -j$(nproc) 2>&1 | tail -20
```

Expected: No errors.

- [ ] **Step 2: Run any existing tests**

```bash
cd build && ctest --output-on-failure 2>&1 | tail -20
```

- [ ] **Step 3: Final commit if needed**

```bash
git add -A
git commit -m "chore: final cleanup for Submodule migration"
```

---

## Self-Review

**Spec coverage:**
- Submodule\<T\> with add_slot — Task 1 ✓
- Module::add_child wiring — Task 1 ✓
- ModuleBase → Module rename — Tasks 1, 9 ✓
- CRTP deletion — Task 1 ✓
- All 6 weight classes migrated — Tasks 2-7 ✓
- kChildren/accessors/friend deleted — Tasks 2-7 ✓
- Caller updates (unified_decoder.cc, language_model.cc) — Task 8 ✓
- Python bindings updated — Task 9 ✓
- mutable removed — Task 4 ✓

**No placeholders:** All steps have exact code, file paths, and commands.

**Type consistency:** `Submodule<T>` template is consistent across all tasks. Module naming is consistent.
