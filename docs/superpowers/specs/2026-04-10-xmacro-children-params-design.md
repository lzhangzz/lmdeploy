# X-Macro Children and Parameters Design

Date: 2026-04-10

## Motivation

Submodule<T> and Parameter self-register with the parent Module via runtime vectors (slots_, params_, children_). This creates:
- Name duplication (`w_qkv` appears as both a member name and `"w_qkv"` string literal)
- Inconsistent ownership (children_ owns child modules, but params own their tensors directly)
- Dead code (add_alias, params(), release/to_device/persist already broken)

X-macros already work well for config fields. Extending them to children and params gives DRY declarations plus compile-time introspection, while removing the Submodule<T> and Parameter abstractions entirely.

## Design Decisions

### D1: Direct ownership for both children and params

Children: `std::unique_ptr<T>` members (e.g., `unique_ptr<LinearWeight> w_qkv;`)
Params: `Tensor` members (e.g., `mutable Tensor weight_{};`)

Both are directly owned by the class. No `children_` vector, no `params_` vector, no `slots_` vector in Module base.

### D2: Submodule<T> and Parameter removed entirely

No wrapper classes. Members are plain `unique_ptr<T>` and `Tensor`.

### D3: No children_ vector in Module base

Module base has no generic child storage. Static children are `unique_ptr<T>` members discovered via X-macro generated virtual methods. ModuleList provides its own indexed storage for dynamic children.

### D4: 2-arg field list form only

```cpp
#define ATTENTION_WEIGHT_CHILDREN(X) \
    X(LinearWeight, w_qkv)          \
    X(LinearWeight, wo)             \
    ...
```

Member name = wire name. No 3-arg aliasing form. Old `_mod` suffix members renamed (e.g., `q_norm_mod` -> `q_norm`). Redundant accessor methods (e.g., `Tensor* q_norm() const`) removed; callers use `q_norm->weight()` directly.

### D5: Big bang refactor

Convert all ~8 weight classes in one shot. No mixed old/new patterns.

### D6: Macros not undef'd

Field list macros persist from header to .cc. No duplication. Naming convention: `<CLASS_NAME>_CHILDREN(X)` and `<CLASS_NAME>_PARAMS(X)`.

### D7: Dropped APIs

| Removed | Reason |
|---|---|
| `Submodule<T>` class | Replaced by plain `unique_ptr<T>` |
| `Parameter` class | Replaced by plain `Tensor` |
| `add_alias()` | Dead code (never called) |
| `params()` | Dead code (never called) |
| `add_param()` | Only used by Parameter ctor |
| `add_slot()` | Only used by Submodule ctor |
| `slots_` vector | Replaced by generated code |
| `children_` vector | Replaced by direct ownership |
| `params_` vector | Replaced by direct ownership |
| `aliases_` vector | Dead code |
| `release()` | Sleep/wakeup already broken |
| `to_device()` | Sleep/wakeup already broken |
| `persist()` | Sleep/wakeup already broken |

### D8: create_child() takes config only

```cpp
Module* create_child(const std::string& name, const ModuleConfig& config);
```

`config.module_type` already carries the type name. Redundant `type_name` parameter removed.

## Module Base Class

```cpp
class Module {
public:
    virtual ~Module() = default;

    // Hierarchy (virtual — X-macro generated in concrete classes, no-op default)
    virtual Module* add_child(std::string name, std::unique_ptr<Module> child);
    virtual Module* child(const std::string& name) const;

    // Parameters (virtual — X-macro generated in concrete classes)
    virtual Tensor* param(const std::string& name) const;

    // Introspection hooks (default no-op)
    virtual void for_each_child(const std::function<void(const char*, Module*)>&) const {}
    virtual void for_each_param(const std::function<void(const char*, Tensor&)>&) const {}

    // Lifecycle (default: recurse via for_each hooks)
    virtual void prepare();
    virtual bool verify(std::vector<std::string>& missing);

    // Tensor allocation
    virtual Tensor alloc(const std::string& param_name, const WeightSpec& spec);

    // Registry-driven child creation
    Module* create_child(const std::string& name, const ModuleConfig& config);

    // Explicit-shape param creation (Python-facing)
    Tensor create_param(const std::string& name,
                        const std::vector<size_t>& shape,
                        DataType dtype,
                        int group_size = 0);

    // Convenience
    Module* get(const std::string& segment);

    // Introspection
    virtual const char* type() const;
    std::string full_path() const;
    Module* parent() const noexcept;
    const std::string& name() const noexcept;

protected:
    Module*    parent_ = nullptr;
    std::string name_;
};
```

Default implementations:

```cpp
void Module::prepare() {
    for_each_child([](const char*, Module* m) { if (m) m->prepare(); });
}

bool Module::verify(std::vector<std::string>& missing) {
    for_each_child([&](const char*, Module* m) { if (m) m->verify(missing); });
    for_each_param([&](const char* name, Tensor& t) {
        if (!t) missing.push_back(full_path() + "." + name);
    });
    return missing.empty();
}

Tensor Module::alloc(const std::string& param_name, const WeightSpec& spec) {
    if (auto* t = param(param_name)) return *t;
    return {};
}

Tensor Module::create_param(const std::string& name,
                            const std::vector<size_t>& shape,
                            DataType dtype, int group_size) {
    if (auto* t = param(name)) {
        *t = Tensor{Layout{shape}, dtype, kDEVICE};
        return *t;
    }
    return {};
}
```

## X-Macro Field Lists and Expansion

### Header pattern

```cpp
// attention_weight.h
class AttentionWeight: public core::Module {
public:
    const char* type() const override { return "AttentionWeight"; }
    AttentionWeight() = default;
    explicit AttentionWeight(const core::AttentionConfig& cfg);

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
    X(sinks_)

    // Member declarations
    ATTENTION_WEIGHT_CHILDREN(TM_CHILD_MEMBER)
    ATTENTION_WEIGHT_PARAMS(TM_PARAM_MEMBER)

    // Generated virtual overrides
    Module* add_child(std::string name, std::unique_ptr<Module> child) override;
    Module* child(const std::string& name) const override;
    Tensor* param(const std::string& name) const override;
    void for_each_child(const std::function<void(const char*, Module*)>&) const override;
    void for_each_param(const std::function<void(const char*, Tensor&)>&) const override;

    // Custom overrides
    void prepare() override;
    Tensor alloc(const std::string& param_name, const WeightSpec& spec) override;

    // Accessors
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

### Source pattern

Each generated method has a handwritten skeleton with the field list macro expanding the repetitive body. See Expansion Macros section for the case/visit macros.

```cpp
// attention_weight.cc
#include "attention_weight.h"

// Generated methods — skeleton is handwritten, body expanded by macros
Module* AttentionWeight::add_child(std::string name, std::unique_ptr<Module> child) {
    std::string name_ = std::move(name);
    ATTENTION_WEIGHT_CHILDREN(TM_ADD_CHILD_CASE)
    return nullptr;
}
// ... child(), param(), for_each_child(), for_each_param() similarly (return nullptr)

// Custom overrides
AttentionWeight::AttentionWeight(const core::AttentionConfig& cfg) { ... }
void AttentionWeight::prepare() { ... }
Tensor AttentionWeight::alloc(...) { ... }
```

## Expansion Macros

Macros generate **body fragments** (if-chains, visitor calls), not complete method definitions. The method skeleton is handwritten once per class, with the macro expanding the repetitive part.

### TM_CHILD_MEMBER — member declaration (used in header class body)

```cpp
#define TM_CHILD_MEMBER(Type, name) std::unique_ptr<Type> name;
```

### TM_PARAM_MEMBER — member declaration (used in header class body)

```cpp
#define TM_PARAM_MEMBER(name) mutable Tensor name{};
```

### TM_ADD_CHILD_CASE — single if-case for add_child body

```cpp
#define TM_ADD_CHILD_CASE(Type, name)                                          \
    if (name_ == #name) {                                                      \
        name.reset(static_cast<Type*>(child.release()));                       \
        name->parent_ = this;                                                  \
        name->name_   = std::move(name_);                                      \
        return name.get();                                                     \
    }
```

Usage in .cc:
```cpp
Module* AttentionWeight::add_child(std::string name, std::unique_ptr<Module> child) {
    std::string name_ = std::move(name);  // consumed by cases
    ATTENTION_WEIGHT_CHILDREN(TM_ADD_CHILD_CASE)
    return Module::add_child(std::move(name_), std::move(child));
}
```

### TM_CHILD_CASE — single if-case for child() body

```cpp
#define TM_CHILD_CASE(Type, name) \
    if (name == #name) return name.get();
```

### TM_PARAM_CASE — single if-case for param() body

```cpp
#define TM_PARAM_CASE(name) \
    if (name == #name) return &name;
```

### TM_VISIT_CHILD — single visitor call for for_each_child body

```cpp
#define TM_VISIT_CHILD(Type, name) \
    visitor(#name, name.get());
```

### TM_VISIT_PARAM — single visitor call for for_each_param body

```cpp
#define TM_VISIT_PARAM(name) \
    visitor(#name, name);
```

### Source pattern (revised)

```cpp
// attention_weight.cc

// add_child — method skeleton with macro-expanded body
Module* AttentionWeight::add_child(std::string name, std::unique_ptr<Module> child) {
    std::string name_ = std::move(name);
    ATTENTION_WEIGHT_CHILDREN(TM_ADD_CHILD_CASE)
    return nullptr;
}

// child — method skeleton with macro-expanded body
Module* AttentionWeight::child(const std::string& name) const {
    ATTENTION_WEIGHT_CHILDREN(TM_CHILD_CASE)
    return nullptr;
}

// param — method skeleton with macro-expanded body
Tensor* AttentionWeight::param(const std::string& name) const {
    ATTENTION_WEIGHT_PARAMS(TM_PARAM_CASE)
    return nullptr;
}

// for_each_child — method skeleton with macro-expanded body
void AttentionWeight::for_each_child(
    const std::function<void(const char*, Module*)>& visitor) const {
    ATTENTION_WEIGHT_CHILDREN(TM_VISIT_CHILD)
}

// for_each_param — method skeleton with macro-expanded body
void AttentionWeight::for_each_param(
    const std::function<void(const char*, Tensor&)>& visitor) const {
    ATTENTION_WEIGHT_PARAMS(TM_VISIT_PARAM)
}
```

## ModuleList

No X-macro. Overrides virtual hooks with its own dynamic storage:

```cpp
class ModuleList: public Module {
public:
    Module* add_child(std::string name, std::unique_ptr<Module> child) override;
    Module* child(const std::string& name) const override;
    void for_each_child(const std::function<void(const char*, Module*)>&) const override;
    int size() const;

private:
    std::vector<std::pair<std::string, std::unique_ptr<Module>>> items_;
    std::vector<Module*> indexed_;
};
```

## MoeWeight Block View

`block_` is a `unique_ptr<FfnWeight>` private member (not a child of MoeWeight). Calling `add_child` on it invokes the X-macro generated code in FfnWeight, setting the `unique_ptr<LinearWeight>` members. No special treatment needed.

## Python Bindings

Changes:
- `persist()` binding removed
- `create_child()` binding simplified (no explicit type_name)
- Everything else unchanged — same Python-facing API

## Weight Classes Summary

| Class | Children | Params | Notes |
|---|---|---|---|
| `LinearWeight` | — | weight_, bias_, scales_, zeros_ | alloc() is custom |
| `NormWeight` | — | weight_ | alloc() is custom |
| `AttentionWeight` | 10 LinearWeight/NormWeight | sinks_ | alloc(), prepare() custom |
| `FfnWeight` | w1, w3, w2, w1w3 | — | prepare() custom |
| `MoeWeight` | gate, shared_gate, experts | score_correction_bias_ | prepare() custom, block_ is private |
| `DeltaNetWeight` | in_proj_all, out_proj, norm | conv1d_, A_log_, dt_bias_ | alloc() custom |
| `DecoderLayerWeight` | attention, linear_attn, feed_forward, moe_ffn, attn_norm, ffn_norm | — | verify() custom |
| `ModelWeight` | tok_embeddings, output, norm, layers | — | verify(), prepare() custom |
