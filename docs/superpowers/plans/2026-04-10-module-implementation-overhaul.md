# Module Implementation Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce boilerplate in module classes via new macros, fix param naming bugs, remove `mutable`, and clean dead code.

**Architecture:** Replace per-class boilerplate method bodies with two new macros (`TM_MODULE_DECLARE` for headers, `TM_MODULE_METHODS` for sources). Fix param X-macro lists to drop trailing underscores. Make `param()`/`for_each_param()` non-const. Remove dead code and redundant accessors.

**Tech Stack:** C++ (X-macros), pybind11, Python

---

### Task 1: Update macros and base class

**Files:**
- Modify: `src/turbomind/core/module.h`
- Modify: `src/turbomind/core/module.cc`

- [ ] **Step 1: Update TM_PARAM_MEMBER and add new macros in module.h**

In `src/turbomind/core/module.h`, make these changes:

**a)** Drop `mutable` from TM_PARAM_MEMBER (line 59):

```cpp
// Before:
#define TM_PARAM_MEMBER(name) mutable Tensor name{};

// After:
#define TM_PARAM_MEMBER(name) Tensor name{};
```

**b)** Add TM_MODULE_DECLARE macro after the existing CASE macros (after line 89):

```cpp
/// Declares data members (children + params) and virtual method overrides.
/// Used in the public section of a derived class.
#define TM_MODULE_DECLARE(Class, ChildrenX, ParamsX)                         \
    ChildrenX(TM_CHILD_MEMBER)                                                \
    ParamsX(TM_PARAM_MEMBER)                                                  \
    core::Module* add_child(std::string name,                                 \
                            std::unique_ptr<Module> child) override;          \
    core::Module* child(const std::string& name) const override;              \
    Tensor*       param(const std::string& name) override;                    \
    void          for_each_child(std::function<void(const char*, Module*)>    \
                                    visitor) const override;                  \
    void          for_each_param(std::function<void(const char*, Tensor&)>    \
                                    visitor) override;
```

**c)** Add TM_MODULE_METHODS macro after TM_MODULE_DECLARE:

```cpp
/// Defines all X-macro generated method bodies for a derived module class.
/// Used in the .cc file.  ChildrenX/ParamsX may be empty macros.
#define TM_MODULE_METHODS(Class, ChildrenX, ParamsX)                          \
    core::Module* Class::add_child(std::string name,                          \
                                   std::unique_ptr<core::Module> child) {     \
        std::string name_str = std::move(name);                                \
        ChildrenX(TM_ADD_CHILD_CASE)                                            \
        return nullptr;                                                         \
    }                                                                           \
    core::Module* Class::child(const std::string& name_str) const {            \
        ChildrenX(TM_CHILD_CASE)                                                \
        return nullptr;                                                         \
    }                                                                           \
    Tensor* Class::param(const std::string& name_str) {                        \
        ParamsX(TM_PARAM_CASE)                                                  \
        return nullptr;                                                         \
    }                                                                           \
    void Class::for_each_child(                                                 \
        std::function<void(const char*, core::Module*)> visitor) const {       \
        ChildrenX(TM_VISIT_CHILD)                                               \
    }                                                                           \
    void Class::for_each_param(                                                 \
        std::function<void(const char*, Tensor&)> visitor) {                   \
        ParamsX(TM_VISIT_PARAM)                                                 \
    }
```

- [ ] **Step 2: Make param() and for_each_param() non-const in module.h**

In the `Module` base class, drop `const` from the two virtual declarations:

```cpp
// Line 150 — before:
virtual Tensor* param(const std::string& name) const;
// After:
virtual Tensor* param(const std::string& name);

// Line 153 — before:
virtual void for_each_param(std::function<void(const char*, Tensor&)> visitor) const;
// After:
virtual void for_each_param(std::function<void(const char*, Tensor&)> visitor);
```

- [ ] **Step 3: Update default implementations in module.cc**

In `src/turbomind/core/module.cc`, drop `const` from the two default implementations:

```cpp
// Line 46 — before:
Tensor* Module::param(const std::string& /*name*/) const
// After:
Tensor* Module::param(const std::string& /*name*/)

// Line 51 — before:
void Module::for_each_param(std::function<void(const char*, Tensor&)> /*visitor*/) const
// After:
void Module::for_each_param(std::function<void(const char*, Tensor&)> /*visitor*/)
```

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/core/module.h src/turbomind/core/module.cc
git commit -m "refactor(module): add TM_MODULE_DECLARE/TM_MODULE_METHODS macros, drop mutable, make param() non-const"
```

---

### Task 2: Update LinearWeight (params-only, has accessors to remove)

**Files:**
- Modify: `src/turbomind/models/linear_weight.h`
- Modify: `src/turbomind/models/linear_weight.cc`

- [ ] **Step 1: Update linear_weight.h**

**a)** Add empty children X-macro list and rename param entries (drop trailing `_`):

Replace the X-macro block (lines 86-92):

```cpp
// Before:
#define LINEAR_WEIGHT_PARAMS(X) \
    X(weight_) \
    X(bias_)   \
    X(scales_) \
    X(zeros_)

    LINEAR_WEIGHT_PARAMS(TM_PARAM_MEMBER)

    // Generated overrides
    Tensor* param(const std::string& name_str) const override;
    void    for_each_param(std::function<void(const char*, Tensor&)>) const override;

// After:
#define LINEAR_WEIGHT_CHILDREN(X)

#define LINEAR_WEIGHT_PARAMS(X) \
    X(weight) \
    X(bias)   \
    X(scales) \
    X(zeros)

    TM_MODULE_DECLARE(LinearWeight, LINEAR_WEIGHT_CHILDREN, LINEAR_WEIGHT_PARAMS)
```

**b)** Remove the 8 accessor methods (lines 55-63):

Delete these lines entirely:

```cpp
    // Accessors for execution layers (LlamaLinear, etc.)
    Tensor&       weight()       { return weight_; }
    Tensor&       bias()         { return bias_; }
    Tensor&       scales()       { return scales_; }
    Tensor&       zeros()        { return zeros_; }
    const Tensor& weight() const { return weight_; }
    const Tensor& bias()   const { return bias_; }
    const Tensor& scales() const { return scales_; }
    const Tensor& zeros()  const { return zeros_; }
```

**c)** Update `operator bool` (line 53) — `weight_` → `weight`:

```cpp
// Before:
explicit operator bool() const noexcept { return static_cast<bool>(weight_); }
// After:
explicit operator bool() const noexcept { return static_cast<bool>(weight); }
```

**d)** Move the class to have no `private:` section for X-macro members. The `TM_MODULE_DECLARE` goes in the public section. Keep existing private members (`do_allocate`, `has_bias_`, `is_grouped_`).

- [ ] **Step 2: Update linear_weight.cc — internal references**

**a)** Replace the X-macro method bodies at the end of the file (lines 379-388) with:

```cpp
// X-macro generated methods
TM_MODULE_METHODS(LinearWeight, LINEAR_WEIGHT_CHILDREN, LINEAR_WEIGHT_PARAMS)
```

**b)** Rename all `weight_` → `weight`, `bias_` → `bias`, `scales_` → `scales`, `zeros_` → `zeros` throughout the file. Key locations:

- `do_allocate()` (line 102): `weight_ = Tensor(...)` → `weight = Tensor(...)`
- `do_allocate()` (lines 104-106): `bias_ = Tensor(...)` → `bias = Tensor(...)`
- `do_allocate()` (line 108-109): `scales_ = {};` → `scales = {};`, `zeros_ = {};` → `zeros = {};`
- `do_allocate()` (lines 113, 117, 122, 128): `scales_ = Tensor(...)` → `scales = Tensor(...)`, `zeros_ = Tensor(...)` → `zeros = Tensor(...)`
- `alloc()`: replace accessor calls with direct member access:
  - `weight()` → `weight`, `bias()` → `bias`, `scales()` → `scales`, `zeros()` → `zeros`

**c)** In `prepare()` (line 206+), replace all accessor calls:
- `weight()` → `weight`, `bias()` → `bias`, `scales()` → `scales`, `zeros()` → `zeros`

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/models/linear_weight.h src/turbomind/models/linear_weight.cc
git commit -m "refactor(linear_weight): use TM_MODULE_DECLARE, remove accessors, rename params"
```

---

### Task 3: Update NormWeight (params-only)

**Files:**
- Modify: `src/turbomind/models/norm_weight.h`
- Modify: `src/turbomind/models/norm_weight.cc`

- [ ] **Step 1: Update norm_weight.h**

Replace the X-macro block and accessors (lines 36-42):

```cpp
// Before:
#define NORM_WEIGHT_PARAMS(X) \
    X(weight_)

    NORM_WEIGHT_PARAMS(TM_PARAM_MEMBER)

    Tensor* param(const std::string& name_str) const override;
    void    for_each_param(std::function<void(const char*, Tensor&)>) const override;

// After:
#define NORM_WEIGHT_CHILDREN(X)

#define NORM_WEIGHT_PARAMS(X) \
    X(weight)

    TM_MODULE_DECLARE(NormWeight, NORM_WEIGHT_CHILDREN, NORM_WEIGHT_PARAMS)
```

Remove the `weight()` accessors (lines 32-33):

```cpp
// Delete these:
Tensor&       weight()       { return weight_; }
const Tensor& weight() const { return weight_; }
```

- [ ] **Step 2: Update norm_weight.cc**

Rename `weight_` → `weight` throughout:
- Constructors (lines 14, 21): `weight_ = Tensor{...}` → `weight = Tensor{...}`
- `alloc()` (lines 46, 53): `weight_` → `weight`

Replace X-macro method bodies (lines 77-86) with:

```cpp
TM_MODULE_METHODS(NormWeight, NORM_WEIGHT_CHILDREN, NORM_WEIGHT_PARAMS)
```

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/models/norm_weight.h src/turbomind/models/norm_weight.cc
git commit -m "refactor(norm_weight): use TM_MODULE_DECLARE, remove accessors, rename params"
```

---

### Task 4: Update AttentionWeight (mixed: children + params)

**Files:**
- Modify: `src/turbomind/models/attention_weight.h`
- Modify: `src/turbomind/models/attention_weight.cc`

- [ ] **Step 1: Update attention_weight.h**

**a)** Rename param entry (drop trailing `_`):

```cpp
// Before:
#define ATTENTION_WEIGHT_PARAMS(X) \
    X(sinks_)
// After:
#define ATTENTION_WEIGHT_PARAMS(X) \
    X(sinks)
```

**b)** Replace the manual member declarations and override declarations (lines 40-48) with:

```cpp
// Before:
    ATTENTION_WEIGHT_CHILDREN(TM_CHILD_MEMBER)
    ATTENTION_WEIGHT_PARAMS(TM_PARAM_MEMBER)

    // Generated overrides
    Module* add_child(std::string name, std::unique_ptr<Module> child) override;
    Module* child(const std::string& name) const override;
    Tensor* param(const std::string& name) const override;
    void    for_each_child(std::function<void(const char*, Module*)> visitor) const override;
    void    for_each_param(std::function<void(const char*, Tensor&)> visitor) const override;

// After:
    TM_MODULE_DECLARE(AttentionWeight, ATTENTION_WEIGHT_CHILDREN, ATTENTION_WEIGHT_PARAMS)
```

- [ ] **Step 2: Update attention_weight.cc**

Rename `sinks_` → `sinks` in `alloc()` (lines 34-38):

```cpp
// Before:
    if (param_name == "sinks" && !sinks_) {
        sinks_ = Tensor{{head_num_ / tp_size_}, spec.dtype, kDEVICE};
    }
    if (param_name == "sinks") {
        return sinks_;
    }
// After:
    if (param_name == "sinks" && !sinks) {
        sinks = Tensor{{head_num_ / tp_size_}, spec.dtype, kDEVICE};
    }
    if (param_name == "sinks") {
        return sinks;
    }
```

Replace the X-macro method bodies (lines 45-74) with:

```cpp
TM_MODULE_METHODS(AttentionWeight, ATTENTION_WEIGHT_CHILDREN, ATTENTION_WEIGHT_PARAMS)
```

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/models/attention_weight.h src/turbomind/models/attention_weight.cc
git commit -m "refactor(attention_weight): use TM_MODULE_DECLARE, rename sinks_ to sinks"
```

---

### Task 5: Update DeltaNetWeight (mixed: children + params)

**Files:**
- Modify: `src/turbomind/models/delta_net_weight.h`
- Modify: `src/turbomind/models/delta_net_weight.cc`

- [ ] **Step 1: Update delta_net_weight.h**

**a)** Rename param entries (drop trailing `_`):

```cpp
// Before:
#define DELTA_NET_WEIGHT_PARAMS(X) \
    X(conv1d_) \
    X(A_log_)  \
    X(dt_bias_)
// After:
#define DELTA_NET_WEIGHT_PARAMS(X) \
    X(conv1d) \
    X(A_log)  \
    X(dt_bias)
```

**b)** Replace member declarations + overrides (lines 36-44) with:

```cpp
    TM_MODULE_DECLARE(DeltaNetWeight, DELTA_NET_WEIGHT_CHILDREN, DELTA_NET_WEIGHT_PARAMS)
```

- [ ] **Step 2: Update delta_net_weight.cc**

Rename all param member references in `alloc()` (lines 30-44):
- `A_log_` → `A_log` (lines 30, 31, 33)
- `dt_bias_` → `dt_bias` (lines 35, 36, 38)
- `conv1d_` → `conv1d` (lines 40, 41, 43)

Replace X-macro method bodies (lines 51-80) with:

```cpp
TM_MODULE_METHODS(DeltaNetWeight, DELTA_NET_WEIGHT_CHILDREN, DELTA_NET_WEIGHT_PARAMS)
```

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/models/delta_net_weight.h src/turbomind/models/delta_net_weight.cc
git commit -m "refactor(delta_net_weight): use TM_MODULE_DECLARE, rename params"
```

---

### Task 6: Update MoeWeight (mixed: children + params)

**Files:**
- Modify: `src/turbomind/models/moe_weight.h`
- Modify: `src/turbomind/models/moe_weight.cc`

- [ ] **Step 1: Update moe_weight.h**

**a)** Rename param entry (drop trailing `_`):

```cpp
// Before:
#define MOE_WEIGHT_PARAMS(X) \
    X(score_correction_bias_)
// After:
#define MOE_WEIGHT_PARAMS(X) \
    X(score_correction_bias)
```

**b)** Replace member declarations + overrides (lines 33-41) with:

```cpp
    TM_MODULE_DECLARE(MoeWeight, MOE_WEIGHT_CHILDREN, MOE_WEIGHT_PARAMS)
```

**c)** Remove the `score_correction_bias()` accessor (line 46):

```cpp
// Delete:
Tensor*       score_correction_bias() const { return score_correction_bias_ ? &score_correction_bias_ : nullptr; }
```

- [ ] **Step 2: Update moe_weight.cc**

**a)** Rename `score_correction_bias_` → `score_correction_bias` in `alloc()` (lines 41-46):

```cpp
// Before:
    if (param_name == "score_correction_bias" && expert_num_ > 0) {
        if (!score_correction_bias_) {
            score_correction_bias_ = Tensor{{expert_num_}, spec.dtype, kDEVICE};
        }
        return score_correction_bias_;
    }
// After:
    if (param_name == "score_correction_bias" && expert_num_ > 0) {
        if (!score_correction_bias) {
            score_correction_bias = Tensor{{expert_num_}, spec.dtype, kDEVICE};
        }
        return score_correction_bias;
    }
```

**b)** Replace accessor calls in `LinkLinearExperts` and `prepare`:
- `e0.bias()` → `e0.bias` (lines 59, 60)
- `e.weight()` → `e.weight` (line 68)
- `e.scales()` → `e.scales` (lines 69, 70)
- `e.bias()` → `e.bias` (lines 72, 73)
- `d.bias()` → `d.bias` (lines 60, 73)
- `d.weight()` → `d.weight` (lines 83, 91)
- `d.scales()` → `d.scales` (lines 84, 93)
- `e0.weight()` → `e0.weight` (line 83)
- `e0.scales()` → `e0.scales` (lines 69, 84, 92)
- `e0.bias()` → `e0.bias` (lines 59, 60)

**c)** Replace X-macro method bodies (lines 191-218) with:

```cpp
TM_MODULE_METHODS(MoeWeight, MOE_WEIGHT_CHILDREN, MOE_WEIGHT_PARAMS)
```

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/models/moe_weight.h src/turbomind/models/moe_weight.cc
git commit -m "refactor(moe_weight): use TM_MODULE_DECLARE, remove accessors, rename params"
```

---

### Task 7: Update FfnWeight (children-only + dead code removal)

**Files:**
- Modify: `src/turbomind/models/ffn_weight.h`
- Modify: `src/turbomind/models/ffn_weight.cc`

- [ ] **Step 1: Update ffn_weight.h**

**a)** Add empty params X-macro list:

```cpp
#define FFN_WEIGHT_PARAMS(X)
```

**b)** Replace member declarations + overrides (lines 29-34) with:

```cpp
    TM_MODULE_DECLARE(FfnWeight, FFN_WEIGHT_CHILDREN, FFN_WEIGHT_PARAMS)
```

**c)** Remove dead code — delete `set_fused_moe` (line 41):

```cpp
// Delete:
void set_fused_moe(bool fused_moe) { is_fused_moe_ = fused_moe; }
```

- [ ] **Step 2: Update ffn_weight.cc**

Replace X-macro method bodies (lines 51-67) with:

```cpp
TM_MODULE_METHODS(FfnWeight, FFN_WEIGHT_CHILDREN, FFN_WEIGHT_PARAMS)
```

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/models/ffn_weight.h src/turbomind/models/ffn_weight.cc
git commit -m "refactor(ffn_weight): use TM_MODULE_DECLARE, remove set_fused_moe"
```

---

### Task 8: Update DecoderLayerWeight (children-only)

**Files:**
- Modify: `src/turbomind/models/decoder_layer_weight.h`
- Modify: `src/turbomind/models/decoder_layer_weight.cc`

- [ ] **Step 1: Update decoder_layer_weight.h**

**a)** Add empty params X-macro list:

```cpp
#define DECODER_LAYER_WEIGHT_PARAMS(X)
```

**b)** Replace member declarations + overrides (lines 35-40) with:

```cpp
    TM_MODULE_DECLARE(DecoderLayerWeight, DECODER_LAYER_WEIGHT_CHILDREN, DECODER_LAYER_WEIGHT_PARAMS)
```

- [ ] **Step 2: Update decoder_layer_weight.cc**

Replace X-macro method bodies (lines 38-55) with:

```cpp
TM_MODULE_METHODS(DecoderLayerWeight, DECODER_LAYER_WEIGHT_CHILDREN, DECODER_LAYER_WEIGHT_PARAMS)
```

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/models/decoder_layer_weight.h src/turbomind/models/decoder_layer_weight.cc
git commit -m "refactor(decoder_layer_weight): use TM_MODULE_DECLARE"
```

---

### Task 9: Update ModelWeight (children-only)

**Files:**
- Modify: `src/turbomind/models/model_weight.h`
- Modify: `src/turbomind/models/model_weight.cc`

- [ ] **Step 1: Update model_weight.h**

**a)** Add empty params X-macro list:

```cpp
#define MODEL_WEIGHT_PARAMS(X)
```

**b)** Replace member declarations + overrides (lines 43-47) with:

```cpp
    TM_MODULE_DECLARE(ModelWeight, MODEL_WEIGHT_CHILDREN, MODEL_WEIGHT_PARAMS)
```

- [ ] **Step 2: Update model_weight.cc**

Replace X-macro method bodies (lines 74-91) with:

```cpp
TM_MODULE_METHODS(ModelWeight, MODEL_WEIGHT_CHILDREN, MODEL_WEIGHT_PARAMS)
```

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/models/model_weight.h src/turbomind/models/model_weight.cc
git commit -m "refactor(model_weight): use TM_MODULE_DECLARE"
```

---

### Task 10: Update C++ call sites

**Files:**
- Modify: `src/turbomind/models/llama/unified_attention_layer.cc`
- Modify: `src/turbomind/models/llama/GatedDeltaNetLayer.cc`
- Modify: `src/turbomind/models/llama/moe_ffn_layer.cc`
- Modify: `src/turbomind/models/llama/LlamaFfnLayer.cc`

- [ ] **Step 1: Fix unified_attention_layer.cc**

At line 486, rename `sinks_` → `sinks`:

```cpp
// Before:
params.sinks = weights.sinks_ ? weights.sinks_.data_or((T*)nullptr) : (T*)nullptr;
// After:
params.sinks = weights.sinks ? weights.sinks.data_or((T*)nullptr) : (T*)nullptr;
```

- [ ] **Step 2: Fix GatedDeltaNetLayer.cc**

At line 197, rename `A_log_` → `A_log`, `dt_bias_` → `dt_bias`:

```cpp
// Before:
ComputeBetaG_v2(beta, g, b, a, weights.A_log_, weights.dt_bias_, stream);
// After:
ComputeBetaG_v2(beta, g, b, a, weights.A_log, weights.dt_bias, stream);
```

At line 214, rename `conv1d_` → `conv1d`:

```cpp
// Before:
... weights.conv1d_, ...
// After:
... weights.conv1d, ...
```

- [ ] **Step 3: Fix moe_ffn_layer.cc**

At lines 56 and 61, replace accessor calls with direct member access:

```cpp
// Before (line 56):
auto& weight = gate.weight();
// After:
auto& w = gate.weight;

// Before (line 57):
TM_CHECK_EQ(input.shape(1), weight.shape(0));
// After:
TM_CHECK_EQ(input.shape(1), w.shape(0));

// Before (line 58):
Tensor_<float> logits{{input.shape(0), weight.shape(1)}, kDEVICE};
// After:
Tensor_<float> logits{{input.shape(0), w.shape(1)}, kDEVICE};

// Before (line 61):
ApplyBias(logits, gate.bias(), core::Context::stream().handle());
// After:
ApplyBias(logits, gate.bias, core::Context::stream().handle());
```

At line 87, replace `score_correction_bias()` accessor with direct member access:

```cpp
// Before:
if (auto* scb = moe.score_correction_bias()) {
    correction_bias = scb->size() > 0 ? scb->data<float>() : nullptr;
}
// After:
if (moe.score_correction_bias) {
    correction_bias = moe.score_correction_bias.size() > 0 ? moe.score_correction_bias.data<float>() : nullptr;
}
```

- [ ] **Step 4: Fix LlamaFfnLayer.cc**

Search for any LinearWeight accessor usage (`.weight()`, `.bias()`, `.scales()`, `.zeros()`) and replace with direct member access. Key patterns:

```cpp
// Before: mlp.is_fused_silu() — this stays (it's a method, not a param accessor)
// Before: any .weight() → .weight, .bias() → .bias, etc.
```

- [ ] **Step 5: Search for remaining accessor calls**

Run: `grep -rn '\.weight()\|\.bias()\|\.scales()\|\.zeros()' src/turbomind/ --include='*.cc' --include='*.cu'`

Fix any remaining hits by replacing `.weight()` with `.weight`, `.bias()` with `.bias`, etc.

- [ ] **Step 6: Commit**

```bash
git add src/turbomind/models/llama/
git commit -m "refactor(execution layers): update param member names and remove accessor calls"
```

---

### Task 11: Python side cleanup

**Files:**
- Modify: `src/turbomind/python/bind.cpp`
- Modify: `lmdeploy/turbomind/deploy/load_context.py`

- [ ] **Step 1: Remove set_fused_silu pybind binding in bind.cpp**

Delete lines 643-649:

```cpp
// Delete this block:
        .def("set_fused_silu",
             [](ft::core::Module& m, bool val) {
                 if (auto* ffn = dynamic_cast<turbomind::FfnWeight*>(&m)) {
                     ffn->set_fused_silu(val);
                 }
             },
             "val"_a);
```

- [ ] **Step 2: Fix load_context.py — pass fuse_silu through config**

In `lmdeploy/turbomind/deploy/load_context.py`, function `commit_ffn` (around line 354):

The current code calls `ffn_mod.set_fused_silu(fused_silu)` on line 366. Since the FfnWeight constructor already computes `is_fused_silu_` from `cfg.fuse_silu`, this is redundant IF the config was set correctly at creation time. Remove the `set_fused_silu` call:

```python
# Before (line 366):
        ffn_mod.set_fused_silu(fused_silu)
# After:
        # fused_silu is already set via config at creation time
```

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/python/bind.cpp lmdeploy/turbomind/deploy/load_context.py
git commit -m "cleanup: remove set_fused_silu pybind binding, remove redundant call in load_context"
```

---

### Task 12: Build and test

- [ ] **Step 1: Build**

```bash
cd build && ninja
```

Expected: Clean build with no errors.

If there are compilation errors from missed `weight_`/`bias_`/`scales_`/`zeros_` references or accessor calls, fix them and rebuild.

- [ ] **Step 2: Run model test**

Use the turbomind-tester agent or run:

```bash
PYTHONPATH=${PWD}/lmdeploy:${PWD}/build/lib python scripts/test_turbomind_model.py
```

Test at least one model end-to-end with a prompt requesting 128+ tokens. Verify the output is coherent (not gibberish).

- [ ] **Step 3: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: resolve compilation/test issues from module overhaul"
```
