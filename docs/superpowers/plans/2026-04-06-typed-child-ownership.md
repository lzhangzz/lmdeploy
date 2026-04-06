# Typed Child Ownership Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace string-based child lookup in TurboMind weight modules with compile-time typed member pointers via CRTP + static constexpr tuple.

**Architecture:** Split `Module` into `ModuleBase` (non-template, holds `children_` vector for generic traversal) and `Module<Derived>` (CRTP template, populates typed member pointers from a static `kChildren` tuple on `add_child`). Ownership stays in `ModuleBase::children_` via `unique_ptr`. Each concrete module declares its children as `Module*` members and exposes them through typed accessors.

**Tech Stack:** C++17, CRTP, `std::apply`, `constexpr` tuple, pybind11

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/turbomind/core/module.h` | `ModuleBase` (non-template) + `Module<Derived>` (CRTP template) |
| `src/turbomind/core/module.cc` | `ModuleBase::` implementations (rename from `Module::`) |
| `src/turbomind/core/registry.h` | `Factory` / `create()` use `ModuleBase` |
| `src/turbomind/core/registry.cc` | Same |
| `src/turbomind/models/attention_weight.h` | `Module<AttentionWeight>`, `kChildren` tuple, member pointers |
| `src/turbomind/models/ffn_weight.h` | `Module<FfnWeight>`, `kChildren` tuple, member pointers |
| `src/turbomind/models/norm_weight.h` | `Module<NormWeight>`, empty tuple (leaf) |
| `src/turbomind/models/linear_weight.h` | `Module<LinearWeight>`, empty tuple (leaf) |
| `src/turbomind/models/decoder_layer_weight.h` + `.cc` | `Module<DecoderLayerWeight>`, `kChildren` tuple, member pointers |
| `src/turbomind/models/delta_net_weight.h` + `.cc` | `Module<DeltaNetWeight>`, `kChildren` tuple, member pointers |
| `src/turbomind/models/moe_weight.h` + `.cc` | `Module<MoeWeight>`, `kChildren` tuple, member pointers |
| `src/turbomind/models/model_weight.h` + `.cc` | `Module<ModelWeight>`, `kChildren` tuple, member pointers |
| `src/turbomind/python/bind.cpp` | `Module` → `ModuleBase` references |
| `src/turbomind/turbomind.h` + `.cc` | `Module*` → `ModuleBase*` return types |

---

### Task 1: Split Module into ModuleBase + Module\<Derived\>

**Files:**
- Modify: `src/turbomind/core/module.h`
- Modify: `src/turbomind/core/module.cc`

- [ ] **Step 1: Split the Module class in module.h**

Rename the existing `Module` class to `ModuleBase`. Everything stays the same except the name. Then add a CRTP template below it:

```cpp
// Forward declare the CRTP template
template<typename Derived>
class Module;

/// Non-template base: owns children, params, lifecycle.
class ModuleBase {
public:
    virtual ~ModuleBase();
    ModuleBase();
    ModuleBase(const ModuleBase&)            = delete;
    ModuleBase& operator=(const ModuleBase&) = delete;
    ModuleBase(ModuleBase&&)                 = delete;
    ModuleBase& operator=(ModuleBase&&)      = delete;

    virtual ModuleBase* add_child(std::string name, std::unique_ptr<ModuleBase> child);
    void add_alias(std::string name, ModuleBase& target);
    void add_param(std::string name, Tensor& tensor);
    virtual const char* type() const;
    virtual Tensor alloc(const std::string& param_name, const WeightSpec& spec);
    virtual void prepare();
    virtual void release();
    virtual void to_device(DeviceType dev);
    virtual bool verify(std::vector<std::string>& missing);
    ModuleBase* create_child(const std::string& name,
                              const std::string& type_name,
                              const ModuleConfig& config = {});

    template<typename T>
    T* get(const std::string& name) const {
        auto* c = child(name);
        TM_CHECK(c != nullptr) << "child '" << name << "' not found in " << type();
        return static_cast<T*>(c);
    }

    const auto& children() const { return children_; }
    ModuleBase* child(const std::string& name) const;
    ModuleBase* get(const std::string& segment);
    Tensor* param(const std::string& name) const;
    std::unordered_map<std::string, Tensor*> params() const;
    std::string full_path() const;
    ModuleBase* parent() const noexcept { return parent_; }
    const std::string& name() const noexcept { return name_; }

protected:
    ModuleBase*       parent_ = nullptr;
    std::string       name_;
    std::vector<std::pair<std::string, std::unique_ptr<ModuleBase>>> children_;
    std::vector<std::pair<std::string, ModuleBase*>>                 aliases_;
    std::vector<std::pair<std::string, Tensor*>>                     params_;

private:
    void collect_params(const std::string& prefix, std::unordered_map<std::string, Tensor*>& out) const;
};

/// CRTP template: populates typed member pointers from Derived::kChildren.
template<typename Derived>
class Module: public ModuleBase {
public:
    ModuleBase* add_child(std::string name, std::unique_ptr<ModuleBase> child) override {
        ModuleBase* raw = child.get();

        bool matched = false;
        std::apply([&](auto&... entry) {
            matched = (try_match(static_cast<Derived*>(this), raw, name, entry.first, entry.second) || ...);
        }, Derived::kChildren);
        TM_CHECK(matched) << "child '" << name << "' not recognized in " << type();

        raw->parent_ = this;
        raw->name_   = name;
        children_.emplace_back(name, raw);
        children_owned_.emplace_back(std::move(name), std::move(child));
        return raw;
    }

protected:
    // Non-owning vector kept in sync for generic traversal (release, to_device, verify).
    // children_ in ModuleBase holds raw pointers; children_owned_ holds unique_ptrs.
    std::vector<std::pair<std::string, std::unique_ptr<ModuleBase>>> children_owned_;

private:
    template<typename T, typename MemberPtr>
    static bool try_match(T* self, ModuleBase* raw,
                          const std::string& name, const char* target, MemberPtr member) {
        if (name == target) {
            self->*member = static_cast<std::remove_pointer_t<MemberPtr>>(raw);
            return true;
        }
        return false;
    }
};
```

Key design decisions:
- `children_` in `ModuleBase` holds non-owning `pair<string, ModuleBase*>` — used by `child()`, `prepare()`, `release()`, `to_device()`, `verify()`, `collect_params()`
- `children_owned_` in `Module<Derived>` holds `unique_ptr` — actual ownership
- `ModuleBase::add_child` (non-CRTP path, used by `ModuleList`) keeps the old ownership-in-children_ behavior
- Both `ModuleBase` and `Module<Derived>` define `add_child` — the non-template base's version stores ownership in `children_` (for ModuleList), the template version stores ownership in `children_owned_`

Wait — this creates two different ownership models. Let me simplify.

**Revised approach:** `ModuleBase` always owns via `children_` (`vector<pair<string, unique_ptr<ModuleBase>>>`). The CRTP template's `add_child` calls `ModuleBase::add_child` for ownership, then additionally populates the typed member pointers from the tuple. No separate `children_owned_`.

```cpp
template<typename Derived>
class Module: public ModuleBase {
public:
    ModuleBase* add_child(std::string name, std::unique_ptr<ModuleBase> child) override {
        ModuleBase* raw = child.get();

        bool matched = false;
        std::apply([&](auto&... entry) {
            matched = (try_match(static_cast<Derived*>(this), raw, name, entry.first, entry.second) || ...);
        }, Derived::kChildren);
        TM_CHECK(matched) << "child '" << name << "' not recognized in " << type();

        return ModuleBase::add_child(std::move(name), std::move(child));
    }

private:
    template<typename T, typename MemberPtr>
    static bool try_match(T* self, ModuleBase* raw,
                          const std::string& name, const char* target, MemberPtr member) {
        if (name == target) {
            self->*member = static_cast<std::remove_pointer_t<MemberPtr>>(raw);
            return true;
        }
        return false;
    }
};
```

Much cleaner — `ModuleBase::add_child` does all the ownership/parent/name work. The CRTP override just populates the typed members and validates, then delegates. `ModuleList` overrides `add_child` and calls `ModuleBase::add_child` directly.

- [ ] **Step 2: Rename all `Module::` to `ModuleBase::` in module.cc**

Every method definition: `Module::Module()` → `ModuleBase::ModuleBase()`, `Module::~Module()` → `ModuleBase::~ModuleBase()`, etc. Also update `ModuleList` to inherit from `ModuleBase` instead of `Module`:

```cpp
class ModuleList: public ModuleBase {
    // ...
    ModuleBase* add_child(std::string name, std::unique_ptr<ModuleBase> child) override;
};
```

- [ ] **Step 3: Update module.cc ModuleList implementation**

Change `ModuleList::add_child` to use `ModuleBase` types:

```cpp
ModuleBase* ModuleList::add_child(std::string name, std::unique_ptr<ModuleBase> child)
{
    int index = -1;
    {
        std::istringstream iss(name);
        iss >> index;
        if (!iss.eof()) {
            index = -1;
        }
    }
    auto* raw = ModuleBase::add_child(std::move(name), std::move(child));
    if (index >= 0) {
        if (index >= static_cast<int>(indexed_.size())) {
            indexed_.resize(index + 1, nullptr);
        }
        indexed_[index] = raw;
    }
    return raw;
}
```

Change `indexed_` from `vector<Module*>` to `vector<ModuleBase*>`.

- [ ] **Step 4: Build and verify**

```bash
cd /data/lmdeploy-modeling/build && ninja
```

At this point the build will fail because all concrete weight modules still inherit from `Module` (which no longer exists as a standalone class — only `ModuleBase` and `Module<Derived>`). That's expected — the next tasks fix each module.

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/core/module.h src/turbomind/core/module.cc
git commit -m "refactor(core): split Module into ModuleBase + Module<Derived> CRTP template"
```

---

### Task 2: Update registry to use ModuleBase

**Files:**
- Modify: `src/turbomind/core/registry.h`
- Modify: `src/turbomind/core/registry.cc`

- [ ] **Step 1: Update registry.h**

```cpp
// Change forward declaration
class ModuleBase;  // was: class Module

// Change Factory type
using Factory = std::function<std::unique_ptr<ModuleBase>(const ModuleConfig&)>;

// Change create() return type
std::unique_ptr<ModuleBase> create(const std::string& type_name, const ModuleConfig& config = {}) const;
```

- [ ] **Step 2: Update registry.cc**

```cpp
std::unique_ptr<ModuleBase> ModuleRegistry::create(const std::string& type_name, const ModuleConfig& config) const {
    // ... same body, return type changed
}
```

- [ ] **Step 3: Build and verify**

```bash
cd /data/lmdeploy-modeling/build && ninja
```

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/core/registry.h src/turbomind/core/registry.cc
git commit -m "refactor(core): update registry to use ModuleBase"
```

---

### Task 3: Update leaf modules (LinearWeight, NormWeight)

These are leaf modules with no children — minimal change: just inherit from `ModuleBase` and declare an empty `kChildren`.

**Files:**
- Modify: `src/turbomind/models/linear_weight.h`
- Modify: `src/turbomind/models/linear_weight.cc`
- Modify: `src/turbomind/models/norm_weight.h`
- Modify: `src/turbomind/models/norm_weight.cc`

- [ ] **Step 1: Update linear_weight.h**

```cpp
class LinearWeight: public core::ModuleBase {  // was: core::Module
    // ... everything else unchanged
    // No kChildren needed — leaf node, add_child never called on it
};
```

- [ ] **Step 2: Update linear_weight.cc registrar**

```cpp
// Change factory return type
[](const core::ModuleConfig& cfg) -> std::unique_ptr<core::ModuleBase> {  // was: core::Module
```

- [ ] **Step 3: Update norm_weight.h**

```cpp
class NormWeight: public core::ModuleBase {  // was: core::Module
    // ... everything else unchanged
};
```

- [ ] **Step 4: Update norm_weight.cc registrar**

Same factory return type change.

- [ ] **Step 5: Build and verify**

```bash
cd /data/lmdeploy-modeling/build && ninja
```

- [ ] **Step 6: Commit**

```bash
git add src/turbomind/models/linear_weight.h src/turbomind/models/linear_weight.cc \
        src/turbomind/models/norm_weight.h src/turbomind/models/norm_weight.cc
git commit -m "refactor(models): update LinearWeight and NormWeight to ModuleBase"
```

---

### Task 4: Convert FfnWeight to CRTP

**Files:**
- Modify: `src/turbomind/models/ffn_weight.h`
- Modify: `src/turbomind/models/ffn_weight.cc`

- [ ] **Step 1: Update ffn_weight.h**

```cpp
class FfnWeight: public core::Module<FfnWeight> {
public:
    const char* type() const override { return "FfnWeight"; }

    FfnWeight() = default;
    // ... existing constructors unchanged

    void prepare() override;

    // --- Typed child members ---
    LinearWeight* w1_   = nullptr;
    LinearWeight* w3_   = nullptr;
    LinearWeight* w2_   = nullptr;
    LinearWeight* w1w3_ = nullptr;

    static constexpr auto kChildren = std::make_tuple(
        std::pair{"w1",   &FfnWeight::w1_},
        std::pair{"w3",   &FfnWeight::w3_},
        std::pair{"w2",   &FfnWeight::w2_},
        std::pair{"w1w3", &FfnWeight::w1w3_}
    );
    friend class core::Module<FfnWeight>;

    // --- Typed accessors (now just return the member) ---
    LinearWeight* w1()   { return w1_; }
    LinearWeight* w3()   { return w3_; }
    LinearWeight* w2()   { return w2_; }
    LinearWeight* w1w3() { return w1w3_; }

    // ... rest of class unchanged
};
```

- [ ] **Step 2: Update ffn_weight.cc registrar**

Change factory return type to `std::unique_ptr<core::ModuleBase>`.

- [ ] **Step 3: Build and verify**

```bash
cd /data/lmdeploy-modeling/build && ninja
```

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/ffn_weight.h src/turbomind/models/ffn_weight.cc
git commit -m "refactor(models): convert FfnWeight to CRTP Module<FfnWeight>"
```

---

### Task 5: Convert AttentionWeight to CRTP

**Files:**
- Modify: `src/turbomind/models/attention_weight.h`
- Modify: `src/turbomind/models/attention_weight.cc`

- [ ] **Step 1: Update attention_weight.h**

```cpp
class AttentionWeight: public core::Module<AttentionWeight> {
public:
    const char* type() const override { return "AttentionWeight"; }

    AttentionWeight() = default;
    // ... existing constructors unchanged

    void prepare() override;

    // --- Typed child members ---
    LinearWeight* w_qkv_              = nullptr;
    LinearWeight* wo_                 = nullptr;
    LinearWeight* q_proj_             = nullptr;
    LinearWeight* q_a_proj_           = nullptr;
    LinearWeight* q_b_proj_           = nullptr;
    LinearWeight* kv_a_proj_          = nullptr;
    NormWeight*   q_norm_mod_         = nullptr;
    NormWeight*   k_norm_mod_         = nullptr;
    NormWeight*   q_a_layernorm_mod_  = nullptr;
    NormWeight*   kv_a_layernorm_mod_ = nullptr;

    static constexpr auto kChildren = std::make_tuple(
        std::pair{"w_qkv",          &AttentionWeight::w_qkv_},
        std::pair{"wo",             &AttentionWeight::wo_},
        std::pair{"q_proj",         &AttentionWeight::q_proj_},
        std::pair{"q_a_proj",       &AttentionWeight::q_a_proj_},
        std::pair{"q_b_proj",       &AttentionWeight::q_b_proj_},
        std::pair{"kv_a_proj",      &AttentionWeight::kv_a_proj_},
        std::pair{"q_norm",         &AttentionWeight::q_norm_mod_},
        std::pair{"k_norm",         &AttentionWeight::k_norm_mod_},
        std::pair{"q_a_layernorm",  &AttentionWeight::q_a_layernorm_mod_},
        std::pair{"kv_a_layernorm", &AttentionWeight::kv_a_layernorm_mod_}
    );
    friend class core::Module<AttentionWeight>;

    // --- Typed accessors ---
    LinearWeight* w_qkv() const { return w_qkv_; }
    LinearWeight* wo() const { return wo_; }
    LinearWeight* q_proj() const { return q_proj_; }
    LinearWeight* q_a_proj() const { return q_a_proj_; }
    LinearWeight* q_b_proj() const { return q_b_proj_; }
    LinearWeight* kv_a_proj() const { return kv_a_proj_; }
    NormWeight*   q_norm_mod() const { return q_norm_mod_; }
    NormWeight*   k_norm_mod() const { return k_norm_mod_; }
    NormWeight*   q_a_layernorm_mod() const { return q_a_layernorm_mod_; }
    NormWeight*   kv_a_layernorm_mod() const { return kv_a_layernorm_mod_; }

    // Convenience tensor accessors — update to use members instead of child()
    Tensor* q_norm() const;
    Tensor* k_norm() const;
    Tensor* q_a_layernorm() const;
    Tensor* kv_a_layernorm() const;
    Tensor* sinks() const;

    // ... rest unchanged
};
```

- [ ] **Step 2: Update attention_weight.cc**

Update the convenience tensor accessors:
```cpp
Tensor* AttentionWeight::q_norm() const {
    return q_norm_mod_ ? &q_norm_mod_->weight() : nullptr;
}
Tensor* AttentionWeight::k_norm() const {
    return k_norm_mod_ ? &k_norm_mod_->weight() : nullptr;
}
Tensor* AttentionWeight::q_a_layernorm() const {
    return q_a_layernorm_mod_ ? &q_a_layernorm_mod_->weight() : nullptr;
}
Tensor* AttentionWeight::kv_a_layernorm() const {
    return kv_a_layernorm_mod_ ? &kv_a_layernorm_mod_->weight() : nullptr;
}
```

For `sinks()` — check if it uses `child("sinks")` or `param("sinks")`. If `param`, it stays as-is.

Update registrar factory return type.

- [ ] **Step 3: Build and verify**

```bash
cd /data/lmdeploy-modeling/build && ninja
```

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/attention_weight.h src/turbomind/models/attention_weight.cc
git commit -m "refactor(models): convert AttentionWeight to CRTP Module<AttentionWeight>"
```

---

### Task 6: Convert DeltaNetWeight to CRTP

**Files:**
- Modify: `src/turbomind/models/delta_net_weight.h`
- Modify: `src/turbomind/models/delta_net_weight.cc`

- [ ] **Step 1: Update delta_net_weight.h**

```cpp
class DeltaNetWeight: public core::Module<DeltaNetWeight> {
public:
    const char* type() const override { return "DeltaNetWeight"; }

    DeltaNetWeight() = default;
    // ... existing constructors unchanged

    void prepare() override;

    // --- Typed child members ---
    LinearWeight* in_proj_all_ = nullptr;
    LinearWeight* out_proj_    = nullptr;

    static constexpr auto kChildren = std::make_tuple(
        std::pair{"in_proj_all", &DeltaNetWeight::in_proj_all_},
        std::pair{"out_proj",    &DeltaNetWeight::out_proj_}
    );
    friend class core::Module<DeltaNetWeight>;

    // --- Typed accessors ---
    LinearWeight* in_proj_all() const { return in_proj_all_; }
    LinearWeight* out_proj() const { return out_proj_; }

    // Convenience tensor accessors — update to use param() or child()
    Tensor* conv1d() const;
    Tensor* A_log() const;
    Tensor* dt_bias() const;
    Tensor* norm() const;

    // ... rest unchanged
};
```

- [ ] **Step 2: Update delta_net_weight.cc**

Update convenience tensor accessors to use `param()` directly (they currently use `child()` + `static_cast<NormWeight*>` to get the weight tensor). Update registrar factory return type.

- [ ] **Step 3: Build and verify**

```bash
cd /data/lmdeploy-modeling/build && ninja
```

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/delta_net_weight.h src/turbomind/models/delta_net_weight.cc
git commit -m "refactor(models): convert DeltaNetWeight to CRTP Module<DeltaNetWeight>"
```

---

### Task 7: Convert DecoderLayerWeight to CRTP

**Files:**
- Modify: `src/turbomind/models/decoder_layer_weight.h`
- Modify: `src/turbomind/models/decoder_layer_weight.cc`

- [ ] **Step 1: Update decoder_layer_weight.h**

```cpp
class DecoderLayerWeight: public core::Module<DecoderLayerWeight> {
public:
    const char* type() const override { return "DecoderLayerWeight"; }

    DecoderLayerWeight() = default;

    bool verify(std::vector<std::string>& missing) override;

    // --- Typed child members ---
    AttentionWeight* attention_    = nullptr;
    DeltaNetWeight*  linear_attn_  = nullptr;
    FfnWeight*       feed_forward_ = nullptr;
    MoeWeight*       moe_ffn_      = nullptr;
    NormWeight*      attn_norm_    = nullptr;
    NormWeight*      ffn_norm_     = nullptr;

    static constexpr auto kChildren = std::make_tuple(
        std::pair{"attention",     &DecoderLayerWeight::attention_},
        std::pair{"linear_attn",   &DecoderLayerWeight::linear_attn_},
        std::pair{"feed_forward",  &DecoderLayerWeight::feed_forward_},
        std::pair{"moe_ffn",       &DecoderLayerWeight::moe_ffn_},
        std::pair{"attention_norm",&DecoderLayerWeight::attn_norm_},
        std::pair{"ffn_norm",      &DecoderLayerWeight::ffn_norm_}
    );
    friend class core::Module<DecoderLayerWeight>;

    // --- Typed accessors ---
    AttentionWeight* attention() const { return attention_; }
    DeltaNetWeight*  linear_attn() const { return linear_attn_; }
    FfnWeight*       ffn() const { return feed_forward_; }
    MoeWeight*       moe() const { return moe_ffn_; }
    NormWeight*      attn_norm() const { return attn_norm_; }
    NormWeight*      ffn_norm() const { return ffn_norm_; }
};
```

- [ ] **Step 2: Update decoder_layer_weight.cc**

Remove the typed accessor implementations (now inline in header). Update `verify()` to use member pointers:

```cpp
bool DecoderLayerWeight::verify(std::vector<std::string>& missing)
{
    ModuleBase::verify(missing);  // was: Module::verify(missing)
    if (!attention_ && !linear_attn_) {
        missing.push_back(full_path() + ": missing attention or linear_attn");
    }
    if (!feed_forward_ && !moe_ffn_) {
        missing.push_back(full_path() + ": missing feed_forward or moe_ffn");
    }
    if (!attn_norm_) {
        missing.push_back(full_path() + ": missing attention_norm");
    }
    return missing.empty();
}
```

Update registrar factory return type.

- [ ] **Step 3: Build and verify**

```bash
cd /data/lmdeploy-modeling/build && ninja
```

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/decoder_layer_weight.h src/turbomind/models/decoder_layer_weight.cc
git commit -m "refactor(models): convert DecoderLayerWeight to CRTP Module<DecoderLayerWeight>"
```

---

### Task 8: Convert MoeWeight to CRTP

**Files:**
- Modify: `src/turbomind/models/moe_weight.h`
- Modify: `src/turbomind/models/moe_weight.cc`

- [ ] **Step 1: Update moe_weight.h**

```cpp
class MoeWeight: public core::Module<MoeWeight> {
public:
    const char* type() const override { return "MoeWeight"; }

    MoeWeight() = default;
    // ... existing constructors unchanged

    Tensor alloc(const std::string& param_name, const core::WeightSpec& spec) override;
    void prepare() override;
    int num_experts() const { return expert_num_; }

    // --- Typed child members ---
    LinearWeight* gate_        = nullptr;
    LinearWeight* shared_gate_ = nullptr;
    core::ModuleList* experts_ = nullptr;

    static constexpr auto kChildren = std::make_tuple(
        std::pair{"gate",        &MoeWeight::gate_},
        std::pair{"shared_gate", &MoeWeight::shared_gate_},
        std::pair{"experts",     &MoeWeight::experts_}
    );
    friend class core::Module<MoeWeight>;

    // --- Typed accessors ---
    LinearWeight* gate() const { return gate_; }
    LinearWeight* shared_gate() const { return shared_gate_; }
    FfnWeight*    expert(int i) const;
    FfnWeight*    block() const { return block_.get(); }
    Tensor*       score_correction_bias() const { return const_cast<Tensor*>(param("score_correction_bias")); }
    MoeParam::Method method() const { return moe_param_.method; }
    const MoeParam& moe_param() const { return moe_param_; }

    // ... rest unchanged (block_, score_correction_bias_, etc.)
};
```

- [ ] **Step 2: Update moe_weight.cc**

Update `expert(i)` to use `experts_` member:
```cpp
FfnWeight* MoeWeight::expert(int i) const {
    if (!experts_) {
        return nullptr;
    }
    return static_cast<FfnWeight*>(experts_->child(std::to_string(i)));
}
```

Update `prepare()` — the `block_->add_child(...)` calls work naturally because `FfnWeight` is now `Module<FfnWeight>` with its own `kChildren` tuple. The `add_child` will populate `w1_`, `w3_`, `w2_`, `w1w3_` automatically.

Update registrar factory return type.

- [ ] **Step 3: Build and verify**

```bash
cd /data/lmdeploy-modeling/build && ninja
```

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/moe_weight.h src/turbomind/models/moe_weight.cc
git commit -m "refactor(models): convert MoeWeight to CRTP Module<MoeWeight>"
```

---

### Task 9: Convert ModelWeight to CRTP

**Files:**
- Modify: `src/turbomind/models/model_weight.h`
- Modify: `src/turbomind/models/model_weight.cc`

- [ ] **Step 1: Update model_weight.h**

```cpp
class ModelWeight: public core::Module<ModelWeight> {
public:
    const char* type() const override { return "ModelWeight"; }

    ModelWeight() = default;
    // ... existing constructors unchanged

    void prepare() override;
    bool verify(std::vector<std::string>& missing) override;

    // --- Typed child members ---
    LinearWeight*        tok_embeddings_ = nullptr;
    LinearWeight*        output_         = nullptr;
    NormWeight*          norm_           = nullptr;
    core::ModuleList*    layers_         = nullptr;

    static constexpr auto kChildren = std::make_tuple(
        std::pair{"tok_embeddings", &ModelWeight::tok_embeddings_},
        std::pair{"output",         &ModelWeight::output_},
        std::pair{"norm",           &ModelWeight::norm_},
        std::pair{"layers",         &ModelWeight::layers_}
    );
    friend class core::Module<ModelWeight>;

    // --- Typed accessors ---
    LinearWeight*        tok_embeddings() const { return tok_embeddings_; }
    LinearWeight*        output() const { return output_; }
    NormWeight*          norm() const { return norm_; }
    DecoderLayerWeight*  layer(int i) const;
    std::vector<DecoderLayerWeight*> layers() const;
    int                  num_layers() const { return num_layer_; }

    // ... rest unchanged
};
```

- [ ] **Step 2: Update model_weight.cc**

Update `layer(i)`:
```cpp
DecoderLayerWeight* ModelWeight::layer(int i) const {
    if (!layers_) {
        return nullptr;
    }
    return static_cast<DecoderLayerWeight*>(layers_->child(std::to_string(i)));
}
```

Update `layers()` similarly. Update `verify()` to use member pointers. Update registrar factory return type.

- [ ] **Step 3: Build and verify**

```bash
cd /data/lmdeploy-modeling/build && ninja
```

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/model_weight.h src/turbomind/models/model_weight.cc
git commit -m "refactor(models): convert ModelWeight to CRTP Module<ModelWeight>"
```

---

### Task 10: Update Python bindings and public API

**Files:**
- Modify: `src/turbomind/python/bind.cpp`
- Modify: `src/turbomind/turbomind.h`
- Modify: `src/turbomind/turbomind.cc`

- [ ] **Step 1: Update bind.cpp**

Replace all `ft::core::Module` with `ft::core::ModuleBase`:
- `py::class_<ft::core::Module, std::shared_ptr<ft::core::Module>>` → `py::class_<ft::core::ModuleBase, std::shared_ptr<ft::core::ModuleBase>>`
- Lambda parameters `ft::core::Module&` → `ft::core::ModuleBase&`
- Return types `ft::core::Module*` → `ft::core::ModuleBase*`

The `with_context` helper and `dynamic_cast` calls stay the same (they cast to concrete types like `ModelWeight*` which still inherit from `ModuleBase`).

- [ ] **Step 2: Update turbomind.h**

```cpp
core::ModuleBase* root(int index);  // was: core::Module*
```

- [ ] **Step 3: Update turbomind.cc**

Update `root()` return type. The body stays the same (upcasts from `shared_ptr<ModelWeight>` to `ModuleBase*`).

- [ ] **Step 4: Build and verify**

```bash
cd /data/lmdeploy-modeling/build && ninja
```

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/python/bind.cpp src/turbomind/turbomind.h src/turbomind/turbomind.cc
git commit -m "refactor(bind): update Python bindings and public API to ModuleBase"
```

---

### Task 11: End-to-end model verification

- [ ] **Step 1: Build final**

```bash
cd /data/lmdeploy-modeling/build && ninja
```

- [ ] **Step 2: Verify model inference across all model classes**

Use the turbomind-tester agent to test one model from each class with TP=1. Verify each response is meaningful (not gibberish) and at least 128 tokens.

Test matrix:

| Model | Architecture | Covers |
|-------|-------------|--------|
| `Qwen/Qwen3-4B` | Dense GQA | `AttentionWeight` (standard), `FfnWeight`, `DecoderLayerWeight`, `ModelWeight` |
| `Qwen/Qwen3-30B-A3B` | MoE | Adds `MoeWeight` with `ModuleList` experts |
| `zai-org/GLM-4.7-Flash` | MoE + MLA | Adds low-rank attention projections (q_lora_rank, kv_lora_rank) |
| `unsloth/gpt-oss-20b-BF16` | MoE + sliding window + bias | Adds attention bias, sliding window |

- [ ] **Step 3: Verify no regressions in child access**

If inference works correctly with meaningful output, the typed child access is functioning properly.

- [ ] **Step 4: Commit any fixes (if needed)**

If any issues are found during testing, fix and commit separately.

---

## Spec Coverage Check

| Spec requirement | Task |
|-----------------|------|
| Split `Module` into `ModuleBase` + `Module<Derived>` | Task 1 |
| `kChildren` tuple declaration per module | Tasks 4-9 |
| CRTP `add_child` with `try_match` + `TM_CHECK` | Task 1 |
| Typed member pointers replacing `child()` calls | Tasks 4-9 |
| Registry uses `ModuleBase` | Task 2 |
| Python bindings use `ModuleBase` | Task 10 |
| `ModuleList` inherits `ModuleBase` | Task 1 |
| `block_` special case in MoeWeight | Task 8 |
| Closed-set validation on `add_child` | Task 1 (in CRTP template) |
| End-to-end verification | Task 11 |
