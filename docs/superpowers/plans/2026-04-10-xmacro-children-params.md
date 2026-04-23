# X-Macro Children and Parameters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Submodule<T> and Parameter with X-macro generated code, using direct ownership (unique_ptr<T> for children, Tensor for params).

**Architecture:** Each weight class declares field list macros (e.g., `ATTENTION_WEIGHT_CHILDREN(X)`) inside the class body. These expand to member declarations, and in the .cc file expand to if-chain method bodies for `add_child()`, `child()`, `param()`, `for_each_child()`, `for_each_param()`. Module base class is stripped down to virtual hooks only — no storage vectors. ModuleList provides its own dynamic storage.

**Tech Stack:** C++17, X-macros, pybind11

**Spec:** `docs/superpowers/specs/2026-04-10-xmacro-children-params-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/turbomind/core/module.h` | **Rewrite** | Module base class, expansion macros, ModuleList |
| `src/turbomind/core/module.cc` | **Rewrite** | Module base implementations, ModuleList implementations |
| `src/turbomind/models/linear_weight.h` | **Modify** | Replace Parameter with Tensor + X-macro params |
| `src/turbomind/models/linear_weight.cc` | **Modify** | Replace `*weight_` with `weight_`, generate param/for_each |
| `src/turbomind/models/norm_weight.h` | **Modify** | Replace Parameter with Tensor + X-macro params |
| `src/turbomind/models/norm_weight.cc` | **Modify** | Replace `*weight_` with `weight_`, generate param/for_each |
| `src/turbomind/models/ffn_weight.h` | **Modify** | Replace Submodule with unique_ptr + X-macro children |
| `src/turbomind/models/ffn_weight.cc` | **Modify** | Replace `children()` iteration, generate methods |
| `src/turbomind/models/attention_weight.h` | **Modify** | Replace both Submodule and Parameter + X-macro |
| `src/turbomind/models/attention_weight.cc` | **Modify** | Replace all patterns, generate methods, remove accessors |
| `src/turbomind/models/delta_net_weight.h` | **Modify** | Replace both + X-macro |
| `src/turbomind/models/delta_net_weight.cc` | **Modify** | Replace patterns, generate methods |
| `src/turbomind/models/moe_weight.h` | **Modify** | Replace both + X-macro |
| `src/turbomind/models/moe_weight.cc` | **Modify** | Replace `children()`, block view, generate methods |
| `src/turbomind/models/decoder_layer_weight.h` | **Modify** | Replace Submodule + X-macro children |
| `src/turbomind/models/decoder_layer_weight.cc` | **Modify** | Replace verify pattern, generate methods |
| `src/turbomind/models/model_weight.h` | **Modify** | Replace Submodule + X-macro children |
| `src/turbomind/models/model_weight.cc` | **Modify** | Replace layer()/layers_list(), generate methods |
| `src/turbomind/python/bind.cpp` | **Modify** | Remove persist binding, simplify create_child |
| `src/turbomind/turbomind.cc` | **Modify** | Remove release/to_device calls |
| `src/turbomind/models/llama/unified_attention_layer.cc` | **Modify** | Update `sinks()` → `sinks_` access |
| `src/turbomind/models/llama/unified_decoder.cc` | **Modify** | Submodule→unique_ptr is transparent |
| `src/turbomind/models/llama/moe_ffn_layer.cc` | **Modify** | Block view access pattern |
| `src/turbomind/models/llama/GatedDeltaNetLayer.cc` | **Modify** | `A_log()`/`conv1d()`/`dt_bias()` → direct |
| `src/turbomind/kernels/gemm/test/testbed_v3.h` | **Modify** | Remove Parameter/Submodule includes if needed |

---

### Task 1: Rewrite Module base class (module.h + module.cc)

**Files:**
- Modify: `src/turbomind/core/module.h` (full rewrite)
- Modify: `src/turbomind/core/module.cc` (full rewrite)

This is the foundation — all other tasks depend on this.

- [ ] **Step 1: Rewrite module.h**

Replace the entire file with the new Module base class, expansion macros, and ModuleList. Key changes:
- Remove `Submodule<T>`, `Parameter`, `PersistOp`
- Remove `children_`, `params_`, `slots_`, `aliases_` vectors
- Remove `add_param()`, `add_slot()`, `add_alias()`, `release()`, `to_device()`, `persist()`, `params()`, `collect_params()`
- Add `for_each_child()`, `for_each_param()` virtual hooks (default no-op)
- Add `create_param()` to public API
- Simplify `create_child()` to take config only
- Define expansion macros: `TM_CHILD_MEMBER`, `TM_PARAM_MEMBER`, `TM_ADD_CHILD_CASE`, `TM_CHILD_CASE`, `TM_PARAM_CASE`, `TM_VISIT_CHILD`, `TM_VISIT_PARAM`
- ModuleList gets own `items_` + `indexed_` storage, overrides `add_child()`, `child()`, `for_each_child()`

New `module.h`:
```cpp
// Copyright (c) OpenMMLab. All rights reserved.
#ifndef TURBOMIND_CORE_MODULE_H
#define TURBOMIND_CORE_MODULE_H

#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/module_config.h"
#include "src/turbomind/core/registry.h"
#include "src/turbomind/core/tensor.h"

namespace turbomind::core {

/// Quantization metadata passed to ``Module::alloc``.
struct WeightSpec {
    DataType dtype{};        // storage dtype of the weight
    int      group_size = 0; // quantization group size (0 = not quantized)
};

// ======================================================================
// Expansion macros for X-macro children/params
// ======================================================================

// --- Member declarations (used in class body) ---

/// Declare a child member: std::unique_ptr<Type> name;
#define TM_CHILD_MEMBER(Type, name) std::unique_ptr<Type> name;

/// Declare a param member: mutable Tensor name{};
#define TM_PARAM_MEMBER(name) mutable Tensor name{};

// --- add_child() body fragments ---

/// Single if-case for add_child: match name, move child into unique_ptr.
/// Uses `name_` (local variable) to avoid consuming the parameter.
#define TM_ADD_CHILD_CASE(Type, name)                                          \
    if (name_ == #name) {                                                      \
        TM_CHECK(child != nullptr);                                            \
        TM_CHECK(!name || name->parent_ == nullptr);                           \
        name.reset(static_cast<Type*>(child.release()));                       \
        name->parent_ = this;                                                  \
        name->name_   = std::move(name_);                                      \
        return name.get();                                                     \
    }

// --- child() body fragments ---

/// Single if-case for child lookup.
#define TM_CHILD_CASE(Type, name)                                              \
    if (n == #name) return name.get();

// --- param() body fragments ---

/// Single if-case for param lookup.
#define TM_PARAM_CASE(name)                                                    \
    if (n == #name) return &name;

// --- for_each_child() body fragments ---

/// Single visitor call for child iteration.
#define TM_VISIT_CHILD(Type, name)                                             \
    visitor(#name, name.get());

// --- for_each_param() body fragments ---

/// Single visitor call for param iteration.
#define TM_VISIT_PARAM(name)                                                   \
    visitor(#name, name);

// ======================================================================
// Module — type-erased hierarchical module with virtual lifecycle
// ======================================================================

class Module {
public:
    virtual ~Module();

    Module();

    Module(const Module&)            = delete;
    Module& operator=(const Module&) = delete;
    Module(Module&&)                 = delete;
    Module& operator=(Module&&)      = delete;

    // ----- Hierarchy (virtual — X-macro overrides in concrete classes) -----

    /// Add a child module by name. Returns pointer to the child, or nullptr.
    virtual Module* add_child(std::string name, std::unique_ptr<Module> child);

    /// Find a direct child by name. Returns nullptr if not found.
    virtual Module* child(const std::string& name) const;

    // ----- Parameters (virtual — X-macro overrides in concrete classes) -----

    /// Find a parameter tensor by name. Returns nullptr if not found.
    virtual Tensor* param(const std::string& name) const;

    // ----- Introspection hooks (default no-op) -----

    /// Iterate over named children: visitor(const char* name, Module* child).
    virtual void for_each_child(const std::function<void(const char*, Module*)>& visitor) const;

    /// Iterate over named params: visitor(const char* name, Tensor& tensor).
    virtual void for_each_param(const std::function<void(const char*, Tensor&)>& visitor) const;

    // ----- Lifecycle (default: recurse via for_each hooks) -----

    /// Post-load processing: weight format conversion, fusion.
    virtual void prepare();

    /// Walk subtree, collect paths of uninitialized params/modules.
    virtual bool verify(std::vector<std::string>& missing);

    // ----- Tensor allocation -----

    /// Allocate tensor for a named parameter and return for data copy.
    virtual Tensor alloc(const std::string& param_name, const WeightSpec& spec);

    /// Create a named parameter tensor with explicit shape/dtype.
    Tensor create_param(const std::string& name,
                        const std::vector<size_t>& shape,
                        DataType dtype,
                        int group_size = 0);

    // ----- Registry-driven child creation -----

    /// Create a child module using the type registry and attach it.
    Module* create_child(const std::string& name, const ModuleConfig& config = {});

    // ----- Convenience -----

    /// Typed child accessor. Aborts if child not found.
    template<typename T>
    T* get(const std::string& name) const {
        auto* c = child(name);
        TM_CHECK(c != nullptr) << "child '" << name << "' not found in " << type();
        return static_cast<T*>(c);
    }

    /// Find a child by name (aborts if not found).
    Module* get(const std::string& segment);

    // ----- Introspection -----

    virtual const char* type() const;
    std::string full_path() const;
    Module* parent() const noexcept { return parent_; }
    const std::string& name() const noexcept { return name_; }

protected:
    Module*     parent_ = nullptr;
    std::string name_;
};

// ======================================================================
// ModuleList — indexed container for dynamic child sequences
// ======================================================================

class ModuleList: public Module {
public:
    const char* type() const override { return "ModuleList"; }

    ModuleList() = default;

    explicit ModuleList(const ModuleListConfig&) {}

    Module* add_child(std::string name, std::unique_ptr<Module> child) override;
    Module* child(const std::string& name) const override;
    void for_each_child(const std::function<void(const char*, Module*)>& visitor) const override;
    int size() const;

private:
    std::vector<std::pair<std::string, std::unique_ptr<Module>>> items_;
    std::vector<Module*> indexed_;
};

}  // namespace turbomind::core

#endif  // TURBOMIND_CORE_MODULE_H
```

- [ ] **Step 2: Rewrite module.cc**

New `module.cc`:
```cpp
// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/core/module.h"

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/registry.h"

namespace turbomind::core {

// ======================================================================
// Module
// ======================================================================

Module::Module() = default;
Module::~Module() = default;

// ----- Hierarchy -----

Module* Module::add_child(std::string /*name*/, std::unique_ptr<Module> /*child*/) {
    return nullptr;
}

Module* Module::child(const std::string& /*name*/) const {
    return nullptr;
}

Tensor* Module::param(const std::string& /*name*/) const {
    return nullptr;
}

// ----- Introspection -----

void Module::for_each_child(const std::function<void(const char*, Module*)>&) const {}

void Module::for_each_param(const std::function<void(const char*, Tensor&)>&) const {}

// ----- Type info -----

const char* Module::type() const { return "Module"; }

// ----- Lifecycle -----

void Module::prepare() {
    for_each_child([](const char*, Module* m) {
        if (m) m->prepare();
    });
}

bool Module::verify(std::vector<std::string>& missing) {
    for_each_child([&](const char*, Module* m) {
        if (m) m->verify(missing);
    });
    for_each_param([&](const char* name, Tensor& t) {
        if (!t) missing.push_back(full_path() + "." + name);
    });
    return missing.empty();
}

// ----- Tensor allocation -----

Tensor Module::alloc(const std::string& param_name, const WeightSpec& /*spec*/) {
    if (auto* t = param(param_name)) return *t;
    return {};
}

Tensor Module::create_param(const std::string& name,
                            const std::vector<size_t>& shape,
                            DataType dtype,
                            int /*group_size*/) {
    if (auto* t = param(name)) {
        *t = Tensor{Layout{std::vector<ssize_t>(shape.begin(), shape.end())}, dtype, kDEVICE};
        return *t;
    }
    return {};
}

// ----- Registry-driven child creation -----

Module* Module::create_child(const std::string& name, const ModuleConfig& config) {
    auto mod = ModuleRegistry::instance().create(std::string(config.module_type), config);
    if (!mod) return nullptr;
    return add_child(name, std::move(mod));
}

// ----- Convenience -----

Module* Module::get(const std::string& segment) {
    auto* c = child(segment);
    TM_CHECK(c != nullptr) << "child '" << segment << "' not found in " << type();
    return c;
}

// ----- Introspection -----

std::string Module::full_path() const {
    if (!parent_) return name_;
    std::string pp = parent_->full_path();
    if (pp.empty()) return name_;
    return pp + "." + name_;
}

// ======================================================================
// ModuleList
// ======================================================================

Module* ModuleList::add_child(std::string name, std::unique_ptr<Module> child) {
    TM_CHECK(child != nullptr);
    TM_CHECK(child->parent_ == nullptr) << "module already has a parent";

    int index = -1;
    {
        std::istringstream iss(name);
        iss >> index;
        if (!iss.eof()) index = -1;
    }

    child->parent_ = this;
    child->name_   = name;

    Module* raw = child.get();
    items_.emplace_back(std::move(name), std::move(child));

    if (index >= 0) {
        if (index >= static_cast<int>(indexed_.size())) {
            indexed_.resize(index + 1, nullptr);
        }
        indexed_[index] = raw;
    }
    return raw;
}

Module* ModuleList::child(const std::string& name) const {
    for (auto& [n, c] : items_) {
        if (n == name) return c.get();
    }
    return nullptr;
}

void ModuleList::for_each_child(const std::function<void(const char*, Module*)>& visitor) const {
    for (auto& [n, c] : items_) {
        visitor(n.c_str(), c.get());
    }
}

int ModuleList::size() const {
    int n = 0;
    for (auto* p : indexed_) {
        if (p) ++n;
    }
    return n;
}

namespace {
struct ModuleListRegistrar {
    ModuleListRegistrar() {
        core::ModuleRegistry::instance().register_type(
            "ModuleList",
            [](const core::ModuleConfig&) -> std::unique_ptr<core::Module> {
                return std::make_unique<core::ModuleList>();
            });
    }
};
static ModuleListRegistrar _module_list_reg;
}  // anonymous namespace

}  // namespace turbomind::core
```

- [ ] **Step 3: Build to verify Module base compiles**

Run: `cd build && ninja 2>&1 | head -80`
Expected: Compile errors in weight classes (expected — they still use old API). But module.h and module.cc themselves should parse cleanly. Look for errors in those two files specifically.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/core/module.h src/turbomind/core/module.cc
git commit -m "refactor(module): rewrite Module base with X-macro hooks, remove Submodule/Parameter/vectors"
```

---

### Task 2: Convert LinearWeight (params only, simplest case)

**Files:**
- Modify: `src/turbomind/models/linear_weight.h`
- Modify: `src/turbomind/models/linear_weight.cc`

- [ ] **Step 1: Rewrite linear_weight.h**

Changes:
- Add `LINEAR_WEIGHT_PARAMS(X)` macro inside class body
- Replace `mutable core::Parameter weight_{*this, "weight"};` with `LINEAR_WEIGHT_PARAMS(TM_PARAM_MEMBER)` expansion
- Declare `param()`, `for_each_param()` overrides
- No `add_child()`, `child()`, `for_each_child()` overrides needed (no children)

New members section:
```cpp
#define LINEAR_WEIGHT_PARAMS(X) \
    X(weight_) \
    X(bias_)   \
    X(scales_) \
    X(zeros_)

    LINEAR_WEIGHT_PARAMS(TM_PARAM_MEMBER)

    // Generated overrides
    Tensor* param(const std::string& name) const override;
    void for_each_param(const std::function<void(const char*, Tensor&)>&) const override;
```

Keep the `weight()`, `bias()`, `scales()`, `zeros()` accessor methods — they return `Tensor&` and are used by execution layers. Change implementation from `return *weight_;` to `return weight_;`.

- [ ] **Step 2: Rewrite linear_weight.cc**

Key pattern changes:
- `*weight_ = Tensor{...}` → `weight_ = Tensor{...}` (no dereference — Tensor is direct member)
- `*bias_ = Tensor{...}` → `bias_ = Tensor{...}`
- `*scales_ = Tensor{...}` → `scales_ = Tensor{...}`
- `*zeros_ = Tensor{...}` → `zeros_ = Tensor{...}`
- `return *weight_;` → `return weight_;`
- Add generated method bodies:
```cpp
Tensor* LinearWeight::param(const std::string& n) const {
    LINEAR_WEIGHT_PARAMS(TM_PARAM_CASE)
    return nullptr;
}

void LinearWeight::for_each_param(
    const std::function<void(const char*, Tensor&)>& visitor) const {
    LINEAR_WEIGHT_PARAMS(TM_VISIT_PARAM)
}
```

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/models/linear_weight.h src/turbomind/models/linear_weight.cc
git commit -m "refactor(linear_weight): replace Parameter with X-macro Tensor members"
```

---

### Task 3: Convert NormWeight (1 param)

**Files:**
- Modify: `src/turbomind/models/norm_weight.h`
- Modify: `src/turbomind/models/norm_weight.cc`

- [ ] **Step 1: Rewrite norm_weight.h**

```cpp
#define NORM_WEIGHT_PARAMS(X) \
    X(weight_)

    NORM_WEIGHT_PARAMS(TM_PARAM_MEMBER)

    Tensor* param(const std::string& name) const override;
    void for_each_param(const std::function<void(const char*, Tensor&)>&) const override;
```

Change `Tensor& weight() { return *weight_; }` → `Tensor& weight() { return weight_; }`

- [ ] **Step 2: Rewrite norm_weight.cc**

Same pattern as LinearWeight:
- `*weight_ = Tensor{...}` → `weight_ = Tensor{...}`
- `return *weight_;` → `return weight_;`
- Add generated `param()` and `for_each_param()` bodies

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/models/norm_weight.h src/turbomind/models/norm_weight.cc
git commit -m "refactor(norm_weight): replace Parameter with X-macro Tensor member"
```

---

### Task 4: Convert FfnWeight (children only)

**Files:**
- Modify: `src/turbomind/models/ffn_weight.h`
- Modify: `src/turbomind/models/ffn_weight.cc`

- [ ] **Step 1: Rewrite ffn_weight.h**

```cpp
#define FFN_WEIGHT_CHILDREN(X) \
    X(LinearWeight, w1)   \
    X(LinearWeight, w3)   \
    X(LinearWeight, w2)   \
    X(LinearWeight, w1w3)

    FFN_WEIGHT_CHILDREN(TM_CHILD_MEMBER)

    Module* add_child(std::string name, std::unique_ptr<Module> child) override;
    Module* child(const std::string& name) const override;
    void for_each_child(const std::function<void(const char*, Module*)>&) const override;
```

No `param()` or `for_each_param()` overrides — no params.

- [ ] **Step 2: Rewrite ffn_weight.cc**

Key changes:
- Replace `for (auto& [name, child] : children()) { child->prepare(); }` with `Module::prepare();`
- Replace `dynamic_cast<LinearWeight*>(child.get())` in the MoE grouped loop — now iterate via `for_each_child`
- Add generated method bodies:
```cpp
Module* FfnWeight::add_child(std::string name, std::unique_ptr<Module> child) {
    std::string name_ = std::move(name);
    FFN_WEIGHT_CHILDREN(TM_ADD_CHILD_CASE)
    return nullptr;
}

Module* FfnWeight::child(const std::string& n) const {
    FFN_WEIGHT_CHILDREN(TM_CHILD_CASE)
    return nullptr;
}

void FfnWeight::for_each_child(
    const std::function<void(const char*, Module*)>& visitor) const {
    FFN_WEIGHT_CHILDREN(TM_VISIT_CHILD)
}
```

In `FfnWeight::prepare()`, the current code does:
```cpp
for (auto& [name, child] : children_) {
    if (is_fused_moe_) {
        if (auto* lw = dynamic_cast<LinearWeight*>(child.get())) {
            lw->set_grouped(true);
        }
    }
    child->prepare();
}
```

Replace with direct member access + base prepare:
```cpp
void FfnWeight::prepare() {
    if (w1w3 && is_fused_silu_) {
        w1w3->epilogue = gemm::Epilogue::kGatedSilu;
    }
    if (is_fused_moe_) {
        auto set_grouped = [this](const char*, Module* m) {
            if (auto* lw = dynamic_cast<LinearWeight*>(m)) {
                lw->set_grouped(true);
            }
        };
        for_each_child(set_grouped);
    }
    Module::prepare();
}
```

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/models/ffn_weight.h src/turbomind/models/ffn_weight.cc
git commit -m "refactor(ffn_weight): replace Submodule with X-macro unique_ptr children"
```

---

### Task 5: Convert AttentionWeight (children + params + accessors to remove)

**Files:**
- Modify: `src/turbomind/models/attention_weight.h`
- Modify: `src/turbomind/models/attention_weight.cc`
- Modify: `src/turbomind/models/llama/unified_attention_layer.cc` (caller of `sinks()`, `q_norm()`, etc.)

- [ ] **Step 1: Rewrite attention_weight.h**

Key changes:
- Add `ATTENTION_WEIGHT_CHILDREN(X)` and `ATTENTION_WEIGHT_PARAMS(X)` macros
- Replace 10 Submodule declarations with `ATTENTION_WEIGHT_CHILDREN(TM_CHILD_MEMBER)`
- Replace `Parameter sinks_` with `ATTENTION_WEIGHT_PARAMS(TM_PARAM_MEMBER)`
- Declare all 5 generated overrides
- **Remove** accessor methods: `q_norm()`, `k_norm()`, `q_a_layernorm()`, `kv_a_layernorm()`, `sinks()`
- Rename old `q_norm_mod` → `q_norm` (the wire name in the X-macro already handles this)

```cpp
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

    ATTENTION_WEIGHT_CHILDREN(TM_CHILD_MEMBER)
    ATTENTION_WEIGHT_PARAMS(TM_PARAM_MEMBER)

    Module* add_child(std::string name, std::unique_ptr<Module> child) override;
    Module* child(const std::string& name) const override;
    Tensor* param(const std::string& name) const override;
    void for_each_child(const std::function<void(const char*, Module*)>&) const override;
    void for_each_param(const std::function<void(const char*, Tensor&)>&) const override;
```

- [ ] **Step 2: Rewrite attention_weight.cc**

Remove the 5 accessor method definitions (`q_norm()`, `k_norm()`, `q_a_layernorm()`, `kv_a_layernorm()`, `sinks()`).

Change `alloc()`:
- `*sinks_ = Tensor{...}` → `sinks_ = Tensor{...}`
- `return *sinks_;` → `return sinks_;`

Change `prepare()` — remove `q_norm_mod` → `q_norm` references.

Add generated method bodies (all 5 overrides).

- [ ] **Step 3: Update unified_attention_layer.cc**

The callers of the removed accessor methods need updating:

In `unified_attention_layer.cc`:
- `weights.sinks()` → `(weights.sinks_ ? &weights.sinks_ : nullptr)` or use `weights.sinks_ ? weights.sinks_.data_or<T>(nullptr) : nullptr`
- `weights.q_norm()` → `(weights.q_norm ? &weights.q_norm->weight() : nullptr)`
- `weights.k_norm()` → `(weights.k_norm ? &weights.k_norm->weight() : nullptr)`
- `*weights.q_norm()` → `weights.q_norm->weight()` (null check already done)
- `*weights.k_norm()` → `weights.k_norm->weight()` (null check already done)

In `unified_attention_layer.cc` line 486:
```cpp
// Before: params.sinks = weights.sinks() ? weights.sinks()->data_or((T*)nullptr) : (T*)nullptr;
// After:
params.sinks = weights.sinks_ ? weights.sinks_.data_or((T*)nullptr) : (T*)nullptr;
```

In `unified_attention_layer.cc` lines 663-667:
```cpp
// Before: invokeRMSNormQK(q, *weights.q_norm(), ...);
// After:  invokeRMSNormQK(q, weights.q_norm->weight(), ...);
```

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/attention_weight.h src/turbomind/models/attention_weight.cc \
        src/turbomind/models/llama/unified_attention_layer.cc
git commit -m "refactor(attention_weight): replace Submodule/Parameter with X-macro, remove accessors"
```

---

### Task 6: Convert DeltaNetWeight (children + params)

**Files:**
- Modify: `src/turbomind/models/delta_net_weight.h`
- Modify: `src/turbomind/models/delta_net_weight.cc`
- Modify: `src/turbomind/models/llama/GatedDeltaNetLayer.cc`

- [ ] **Step 1: Rewrite delta_net_weight.h**

```cpp
#define DELTA_NET_WEIGHT_CHILDREN(X) \
    X(LinearWeight, in_proj_all) \
    X(LinearWeight, out_proj)    \
    X(NormWeight,   norm)

#define DELTA_NET_WEIGHT_PARAMS(X) \
    X(conv1d_) \
    X(A_log_)  \
    X(dt_bias_)

    DELTA_NET_WEIGHT_CHILDREN(TM_CHILD_MEMBER)
    DELTA_NET_WEIGHT_PARAMS(TM_PARAM_MEMBER)

    Module* add_child(...) override;
    Module* child(...) const override;
    Tensor* param(...) const override;
    void for_each_child(...) const override;
    void for_each_param(...) const override;
```

Remove accessor methods `conv1d()`, `A_log()`, `dt_bias()`.

- [ ] **Step 2: Rewrite delta_net_weight.cc**

Same patterns as AttentionWeight:
- `*conv1d_ = Tensor{...}` → `conv1d_ = Tensor{...}`
- `return *conv1d_;` → `return conv1d_;`
- Remove accessor definitions
- Add generated method bodies

- [ ] **Step 3: Update GatedDeltaNetLayer.cc**

In `GatedDeltaNetLayer.cc`:
```cpp
// Before: *weights.A_log(), *weights.dt_bias()
// After:  weights.A_log_, weights.dt_bias_

// Before: *weights.conv1d()
// After:  weights.conv1d_
```

Lines 197 and 214:
```cpp
// Before: ComputeBetaG_v2(beta, g, b, a, *weights.A_log(), *weights.dt_bias(), stream);
// After:  ComputeBetaG_v2(beta, g, b, a, weights.A_log_, weights.dt_bias_, stream);

// Before: *weights.conv1d()
// After:  weights.conv1d_
```

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/delta_net_weight.h src/turbomind/models/delta_net_weight.cc \
        src/turbomind/models/llama/GatedDeltaNetLayer.cc
git commit -m "refactor(delta_net_weight): replace Submodule/Parameter with X-macro"
```

---

### Task 7: Convert MoeWeight (children + params + block view)

**Files:**
- Modify: `src/turbomind/models/moe_weight.h`
- Modify: `src/turbomind/models/moe_weight.cc`

- [ ] **Step 1: Rewrite moe_weight.h**

```cpp
#define MOE_WEIGHT_CHILDREN(X)           \
    X(LinearWeight, gate)                \
    X(LinearWeight, shared_gate)         \
    X(core::ModuleList, experts)

#define MOE_WEIGHT_PARAMS(X) \
    X(score_correction_bias_)

    MOE_WEIGHT_CHILDREN(TM_CHILD_MEMBER)
    MOE_WEIGHT_PARAMS(TM_PARAM_MEMBER)

    Module* add_child(...) override;
    Module* child(...) const override;
    Tensor* param(...) const override;
    void for_each_child(...) const override;
    void for_each_param(...) const override;
```

Remove the `score_correction_bias()` accessor (it used `param("score_correction_bias")` which goes through the now-removed vector). Replace with `score_correction_bias_ ? &score_correction_bias_ : nullptr` if needed externally, or keep a simple inline accessor.

- [ ] **Step 2: Rewrite moe_weight.cc**

Key changes:
- `*score_correction_bias_ = Tensor{...}` → `score_correction_bias_ = Tensor{...}`
- `return *score_correction_bias_;` → `return score_correction_bias_;`
- Replace `for (auto& [name, child] : children()) { child->prepare(); }` with `Module::prepare();`
- Block view `block_->add_child(...)` calls still work — they call FfnWeight's generated `add_child`
- `exp->w1w3.get()` → `exp->w1w3.get()` (already uses `.get()`, no change)
- `block_->w1w3` dereferences unique_ptr via `operator*` (same as Submodule)
- Add generated method bodies

For `expert(int i)`, change from:
```cpp
return static_cast<FfnWeight*>(experts->child(std::to_string(i)));
```
to:
```cpp
return static_cast<FfnWeight*>(experts ? experts->child(std::to_string(i)) : nullptr);
```
Note: `experts` is now `unique_ptr<ModuleList>`, not `Submodule<ModuleList>`. Use `experts ? experts->child(...) : nullptr` instead of relying on Submodule's operator bool + implicit conversion.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/models/moe_weight.h src/turbomind/models/moe_weight.cc
git commit -m "refactor(moe_weight): replace Submodule/Parameter with X-macro, fix block view"
```

---

### Task 8: Convert DecoderLayerWeight (children only)

**Files:**
- Modify: `src/turbomind/models/decoder_layer_weight.h`
- Modify: `src/turbomind/models/decoder_layer_weight.cc`

- [ ] **Step 1: Rewrite decoder_layer_weight.h**

```cpp
#define DECODER_LAYER_WEIGHT_CHILDREN(X) \
    X(AttentionWeight, attention)    \
    X(DeltaNetWeight,  linear_attn)  \
    X(FfnWeight,       feed_forward) \
    X(MoeWeight,       moe_ffn)      \
    X(NormWeight,      attn_norm)    \
    X(NormWeight,      ffn_norm)

    DECODER_LAYER_WEIGHT_CHILDREN(TM_CHILD_MEMBER)

    Module* add_child(...) override;
    Module* child(...) const override;
    void for_each_child(...) const override;
```

- [ ] **Step 2: Rewrite decoder_layer_weight.cc**

Update `verify()` — currently checks `!attention && !linear_attn`, etc. The Submodule implicit bool conversion `if (!attention)` becomes `if (!attention)` — unique_ptr also has operator bool, so this is transparent.

Add generated method bodies.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/models/decoder_layer_weight.h src/turbomind/models/decoder_layer_weight.cc
git commit -m "refactor(decoder_layer_weight): replace Submodule with X-macro unique_ptr children"
```

---

### Task 9: Convert ModelWeight (children only, with ModuleList)

**Files:**
- Modify: `src/turbomind/models/model_weight.h`
- Modify: `src/turbomind/models/model_weight.cc`

- [ ] **Step 1: Rewrite model_weight.h**

```cpp
#define MODEL_WEIGHT_CHILDREN(X)         \
    X(LinearWeight,     tok_embeddings)  \
    X(LinearWeight,     output)          \
    X(NormWeight,       norm)            \
    X(core::ModuleList, layers)

    MODEL_WEIGHT_CHILDREN(TM_CHILD_MEMBER)

    Module* add_child(...) override;
    Module* child(...) const override;
    void for_each_child(...) const override;
```

Remove `layers_cache_` if no longer needed (the `layers->child(i)` pattern still works via ModuleList's override).

- [ ] **Step 2: Rewrite model_weight.cc**

Update `layer(int i)`:
```cpp
// Before: return static_cast<DecoderLayerWeight*>(layers->child(std::to_string(i)));
// After:  same! layers is now unique_ptr<ModuleList>, layers->child() still works
DecoderLayerWeight* ModelWeight::layer(int i) const {
    if (!layers) return nullptr;
    return static_cast<DecoderLayerWeight*>(layers->child(std::to_string(i)));
}
```

Update `layers_list()`:
```cpp
// Before: layers->size() goes through ModuleList's size()
// After:  same — layers->size() still works
```

Update `verify()` — `if (!tok_embeddings)` works with unique_ptr operator bool (same as Submodule).

Update `prepare()` — currently calls `Module::prepare()` already, no change needed.

Add generated method bodies.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/models/model_weight.h src/turbomind/models/model_weight.cc
git commit -m "refactor(model_weight): replace Submodule with X-macro unique_ptr children"
```

---

### Task 10: Update Python bindings and remaining callers

**Files:**
- Modify: `src/turbomind/python/bind.cpp`
- Modify: `src/turbomind/turbomind.cc`
- Modify: `src/turbomind/models/llama/unified_decoder.cc`
- Modify: `src/turbomind/models/llama/moe_ffn_layer.cc`
- Modify: `src/turbomind/models/language_model.cc` (if needed)
- Modify: `src/turbomind/kernels/gemm/test/testbed_v3.h` (if it includes module.h)

- [ ] **Step 1: Update bind.cpp**

Remove:
- `PersistOp` enum binding
- `persist()` method binding
- `create_child()` explicit type_name extraction

Change `create_child` binding from:
```cpp
return m.create_child(name, std::string(config.module_type), config);
```
to:
```cpp
return m.create_child(name, config);
```

Remove the `ft::core::PersistOp` enum binding.

- [ ] **Step 2: Update turbomind.cc**

Remove `Sleep()` body calls to `weights_[index]->release()` and `weights_[index]->to_device(kCPU)`.
Remove `WakeUp()` body call to `weights_[index]->to_device(kDEVICE)`.

Replace with no-ops or log warnings, since sleep/wakeup is broken:
```cpp
void Sleep(int index, int level) {
    TM_LOG_WARNING("Sleep/wakeup is not supported in this version");
}
void WakeUp(int index, const std::vector<std::string>& tags) {
    TM_LOG_WARNING("Sleep/wakeup is not supported in this version");
}
```

Or remove the methods entirely if the Python-side callers are also being updated.

- [ ] **Step 3: Check unified_decoder.cc**

The Submodule→unique_ptr transition is transparent for `->` access:
- `weights.at(layer)->attn_norm->weight()` — works with both
- `weights.at(layer)->attention` as bool check — works with both
- `weights.at(layer)->linear_attn` as bool check — works with both

No changes expected unless the `Submodule` implicit conversion was used in a way that unique_ptr doesn't support.

- [ ] **Step 4: Check moe_ffn_layer.cc**

- `block->w1w3 && block->w1w3->weight()` — works with unique_ptr
- `*block->w1w3` — works with unique_ptr operator*
- `moe.shared_gate && moe.shared_gate->weight()` — works with unique_ptr
- `moe.block() && moe.block()->w2` — `block()` returns `FfnWeight*`, `->w2` is unique_ptr. Works.
- `moe.block()->w2->bias()` — works

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/python/bind.cpp src/turbomind/turbomind.cc
git commit -m "refactor(bindings): remove persist/sleep, simplify create_child"
```

---

### Task 11: Build and test

**Files:** None (verification only)

- [ ] **Step 1: Full build**

Run: `cd build && ninja 2>&1 | tail -40`
Expected: Clean build with no errors.

Fix any remaining compile errors. Common issues:
- Missing `#include <functional>` for `std::function`
- `Submodule` or `Parameter` references in files not yet updated
- `operator bool()` differences between Submodule and unique_ptr in edge cases

- [ ] **Step 2: Test with turbomind-tester agent**

Run the turbomind-tester agent to verify a model loads and produces correct output.

- [ ] **Step 3: Commit any remaining fixes**

```bash
git add -A
git commit -m "fix: resolve remaining compile issues from X-macro refactor"
```

---

## Self-Review

**1. Spec coverage:**
- D1 (direct ownership): Tasks 1-9 ✓
- D2 (Submodule/Parameter removed): Task 1 defines base, Tasks 2-9 convert classes ✓
- D3 (no children_ vector): Task 1 ✓
- D4 (2-arg form): Tasks 2-9 ✓
- D5 (big bang): All tasks sequential ✓
- D6 (macros not undef'd): Tasks 2-9 define macros in class body ✓
- D7 (dropped APIs): Task 1 (base), Task 10 (bindings, turbomind.cc) ✓
- D8 (create_child config only): Task 1 (base), Task 10 (binding) ✓

**2. Placeholder scan:** No TBDs or TODOs. All code shown.

**3. Type consistency:**
- `TM_CHILD_MEMBER(Type, name)` → `std::unique_ptr<Type> name;` — consistent with `TM_ADD_CHILD_CASE`, `TM_CHILD_CASE`, `TM_VISIT_CHILD` all using `(Type, name)` signature
- `TM_PARAM_MEMBER(name)` → `mutable Tensor name{};` — consistent with `TM_PARAM_CASE`, `TM_VISIT_PARAM` all using `(name)` signature
- `unique_ptr<T>::operator bool()` replaces `Submodule<T>::operator bool()` — both check non-null
- `unique_ptr<T>::operator->()` replaces `Submodule<T>::operator->()` — both return T*
- `Tensor` replaces `Parameter` — `*param_` becomes just `param_` for assignment and return
