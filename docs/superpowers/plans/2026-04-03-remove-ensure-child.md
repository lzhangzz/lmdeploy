# Remove `ensure_child` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all `ensure_child` overrides from C++ composite modules, making the module tree fully explicit.

**Architecture:** Change `Module::get(segment)` from `child() || ensure_child()` to just `child()`. Add `TM_CHECK` to the typed `get<T>()` template for clear errors on missing children. Remove `ModuleList` factory-based lazy creation. Delete all `ensure_child` overrides from 6 composite modules.

**Tech Stack:** C++17, pybind11, CUDA

---

## File Structure

| File | Change |
|------|--------|
| Modify | `src/turbomind/core/module.h` | Remove `ensure_child` virtual, simplify ModuleList, add TM_CHECK to `get<T>()` |
| Modify | `src/turbomind/core/module.cc` | Remove `ensure_child` impl, simplify `get()`, remove ModuleList factory |
| Modify | `src/turbomind/models/attention_weight.h` | Remove `ensure_child` override declaration |
| Modify | `src/turbomind/models/attention_weight.cc` | Remove `ensure_child` definition |
| Modify | `src/turbomind/models/decoder_layer_weight.h` | Remove `ensure_child` override declaration |
| Modify | `src/turbomind/models/decoder_layer_weight.cc` | Remove `ensure_child` definition |
| Modify | `src/turbomind/models/ffn_weight.h` | Remove `ensure_child` override declaration |
| Modify | `src/turbomind/models/ffn_weight.cc` | Remove `ensure_child` definition |
| Modify | `src/turbomind/models/moe_weight.h` | Remove `ensure_child` override declaration |
| Modify | `src/turbomind/models/moe_weight.cc` | Remove `ensure_child` definition |
| Modify | `src/turbomind/models/delta_net_weight.h` | Remove `ensure_child` override declaration |
| Modify | `src/turbomind/models/delta_net_weight.cc` | Remove `ensure_child` definition |
| Modify | `src/turbomind/models/model_weight.h` | Remove `ensure_child` override declaration |
| Modify | `src/turbomind/models/model_weight.cc` | Remove `ensure_child` definition |

---

## Task 1: Core Module Changes (module.h + module.cc)

**Files:**
- Modify: `src/turbomind/core/module.h`
- Modify: `src/turbomind/core/module.cc`

- [ ] **Step 1: Edit module.h — remove `ensure_child` virtual and update `get<T>()`**

In `src/turbomind/core/module.h`, make these changes:

**1a.** Remove the entire `// ----- Lazy child creation -----` section (lines 99-104):
```cpp
    // ----- Lazy child creation -----

    /// Override in composite modules to create children on demand.
    /// Called by get() when a child doesn't exist yet.
    /// Returns pointer to the newly-created child, or nullptr if segment is invalid.
    virtual Module* ensure_child(const std::string& segment);
```

**1b.** Replace the `get<T>()` template (lines 91-94) with a TM_CHECK version:
```cpp
    /// Typed child accessor. Aborts if child not found.
    template<typename T>
    T* get(const std::string& name) const {
        auto* c = child(name);
        TM_CHECK(c != nullptr) << "child '" << name << "' not found in " << type();
        return static_cast<T*>(c);
    }
```

**1c.** Update the `get(segment)` doc comment (line 111-112) to remove lazy-creation language:
```cpp
    /// Find a child by single segment name.
    Module* get(const std::string& segment);
```

**1d.** Remove `ModuleList::Factory`, `explicit ModuleList(Factory factory)`, and `ensure_child` override. Add default constructor. The ModuleList class becomes:
```cpp
class ModuleList: public Module {
public:
    const char* type() const override
    {
        return "ModuleList";
    }

    ModuleList() = default;

    /// Override to also track the child in the indexed_ vector.
    Module* add_child(std::string name, std::unique_ptr<Module> child) override;

    /// Number of children created so far.
    int size() const;

private:
    std::vector<Module*> indexed_;
};
```

**1e.** Update the Module class doc comment (lines 24-31) to remove lazy-creation references. Change:
```cpp
/// Type-erased hierarchical module with virtual lifecycle and lazy child creation.
///
/// The module tree is built incrementally as weights arrive:
///   - ``get(segment)`` returns an existing child or lazily creates one via
///     the virtual ``ensure_child()`` hook.
///   - ``alloc(param_name, spec)`` allocates tensors on demand and returns
///     a handle for data copying.
///   - ``prepare()`` runs post-load processing (format conversion, fusion).
///   - ``verify()`` walks the tree and collects uninitialized params/modules.
```
to:
```cpp
/// Type-erased hierarchical module with virtual lifecycle.
///
/// The module tree is built explicitly via ``create_child()`` from the Python
/// loading pipeline. Children are looked up by name; no lazy creation.
///   - ``alloc(param_name, spec)`` allocates tensors on demand and returns
///     a handle for data copying.
///   - ``prepare()`` runs post-load processing (format conversion, fusion).
///   - ``verify()`` walks the tree and collects uninitialized params/modules.
```

**1f.** Update the ModuleList doc comment (lines 160-161) from:
```cpp
/// A systematic container for indexed module sequences (layers, experts).
/// Children are created lazily from a factory function.
```
to:
```cpp
/// A systematic container for indexed module sequences (layers, experts).
/// Children are added explicitly via ``add_child`` or ``create_child``.
```

- [ ] **Step 2: Edit module.cc — remove `ensure_child` impl, simplify `get()`, remove ModuleList factory**

**2a.** Remove `Module::ensure_child` implementation (lines 100-105):
```cpp
// ----- Lazy child creation -----

Module* Module::ensure_child(const std::string& /*segment*/)
{
    return nullptr;  // base Module cannot create children lazily
}
```

**2b.** Replace `Module::get(segment)` (lines 137-143):
```cpp
Module* Module::get(const std::string& segment)
{
    if (auto* c = child(segment)) {
        return c;
    }
    return ensure_child(segment);
}
```
with:
```cpp
Module* Module::get(const std::string& segment)
{
    auto* c = child(segment);
    TM_CHECK(c != nullptr) << "child '" << segment << "' not found in " << type();
    return c;
}
```

**2c.** Remove `ModuleList` constructor (line 208):
```cpp
ModuleList::ModuleList(Factory factory): factory_{std::move(factory)} {}
```

**2d.** Remove `ModuleList::ensure_child` (lines 231-264):
```cpp
Module* ModuleList::ensure_child(const std::string& segment)
{
    // Try to parse segment as an integer index.
    int index = 0;
    {
        std::istringstream iss(segment);
        if (!(iss >> index) || !iss.eof()) {
            return nullptr;
        }
    }

    // Negative indices are invalid.
    if (index < 0) {
        return nullptr;
    }

    // Grow the indexed vector if needed.
    if (index >= static_cast<int>(indexed_.size())) {
        indexed_.resize(index + 1, nullptr);
    }

    // Already created?
    if (indexed_[index]) {
        return indexed_[index];
    }

    // Create via factory.
    auto child = factory_(index);
    TM_CHECK(child != nullptr) << "ModuleList factory returned nullptr for index " << index;

    auto* raw = add_child(segment, std::move(child));
    indexed_[index] = raw;
    return raw;
}
```

**2e.** Simplify `ModuleListRegistrar` (lines 277-292) to remove the crashing factory:
```cpp
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
} // anonymous namespace
```

- [ ] **Step 3: Build**

```bash
cd /data/lmdeploy-modeling/build && ninja
```

Expected: Clean build. No callers of `ensure_child` remain (those are removed in Task 2).

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/core/module.h src/turbomind/core/module.cc
git commit -m "refactor(core): remove ensure_child, simplify Module::get() with TM_CHECK"
```

---

## Task 2: Remove ensure_child from Composite Modules

**Files:**
- Modify: `src/turbomind/models/attention_weight.h` (line 33)
- Modify: `src/turbomind/models/attention_weight.cc` (lines 39-117)
- Modify: `src/turbomind/models/decoder_layer_weight.h` (line 27)
- Modify: `src/turbomind/models/decoder_layer_weight.cc` (lines 25-100)
- Modify: `src/turbomind/models/ffn_weight.h` (line 20)
- Modify: `src/turbomind/models/ffn_weight.cc` (lines 25-49)
- Modify: `src/turbomind/models/moe_weight.h` (line 27)
- Modify: `src/turbomind/models/moe_weight.cc` (lines 35-70)
- Modify: `src/turbomind/models/delta_net_weight.h` (line 29)
- Modify: `src/turbomind/models/delta_net_weight.cc` (lines 33-76)
- Modify: `src/turbomind/models/model_weight.h` (line 28)
- Modify: `src/turbomind/models/model_weight.cc` (lines 32-58)

- [ ] **Step 1: Remove ensure_child from AttentionWeight**

In `attention_weight.h`, delete line 33:
```cpp
    core::Module* ensure_child(const std::string& segment) override;
```

Also remove the doc comment on line 18:
```cpp
    /// Construct with config for lazy child creation.
```
(replace with just the constructor declaration without the comment)

In `attention_weight.cc`, delete the entire `ensure_child` definition (lines 39-117):
```cpp
core::Module* AttentionWeight::ensure_child(const std::string& segment)
{
    // ... entire 80-line function ...
}
```

- [ ] **Step 2: Remove ensure_child from DecoderLayerWeight**

In `decoder_layer_weight.h`, delete line 27:
```cpp
    core::Module* ensure_child(const std::string& segment) override;
```

In `decoder_layer_weight.cc`, delete the `ensure_child` definition (lines 25-100):
```cpp
core::Module* DecoderLayerWeight::ensure_child(const std::string& segment)
{
    // ... entire function ...
}
```

- [ ] **Step 3: Remove ensure_child from FfnWeight**

In `ffn_weight.h`, delete line 20:
```cpp
    Module* ensure_child(const std::string& segment) override;
```

In `ffn_weight.cc`, delete the `ensure_child` definition (lines 25-49):
```cpp
core::Module* FfnWeight::ensure_child(const std::string& segment)
{
    // ... entire function ...
}
```

- [ ] **Step 4: Remove ensure_child from MoeWeight**

In `moe_weight.h`, delete line 27:
```cpp
    Module* ensure_child(const std::string& segment) override;
```

In `moe_weight.cc`, delete the `ensure_child` definition (lines 35-70):
```cpp
core::Module* MoeWeight::ensure_child(const std::string& segment)
{
    // ... entire function ...
}
```

- [ ] **Step 5: Remove ensure_child from DeltaNetWeight**

In `delta_net_weight.h`, delete line 29:
```cpp
    Module* ensure_child(const std::string& segment) override;
```

In `delta_net_weight.cc`, delete the `ensure_child` definition (lines 33-76):
```cpp
core::Module* DeltaNetWeight::ensure_child(const std::string& segment)
{
    // ... entire function ...
}
```

- [ ] **Step 6: Remove ensure_child from ModelWeight**

In `model_weight.h`, delete line 28:
```cpp
    core::Module* ensure_child(const std::string& segment) override;
```

In `model_weight.cc`, delete the `ensure_child` definition (lines 32-58):
```cpp
core::Module* ModelWeight::ensure_child(const std::string& segment)
{
    // ... entire function ...
}
```

- [ ] **Step 7: Build**

```bash
cd /data/lmdeploy-modeling/build && ninja
```

Expected: Clean build. All `ensure_child` references are gone.

- [ ] **Step 8: Commit**

```bash
git add src/turbomind/models/attention_weight.h src/turbomind/models/attention_weight.cc \
       src/turbomind/models/decoder_layer_weight.h src/turbomind/models/decoder_layer_weight.cc \
       src/turbomind/models/ffn_weight.h src/turbomind/models/ffn_weight.cc \
       src/turbomind/models/moe_weight.h src/turbomind/models/moe_weight.cc \
       src/turbomind/models/delta_net_weight.h src/turbomind/models/delta_net_weight.cc \
       src/turbomind/models/model_weight.h src/turbomind/models/model_weight.cc
git commit -m "refactor(models): remove ensure_child overrides from all composite modules"
```

---

## Task 3: Verify All Models

Test sequentially (one model at a time) to avoid GPU conflicts. Test each model with TP=1 and TP=2.

**Environment:** `PYTHONPATH=lmdeploy:build/lib`

Use model-server MCP tools to get cache paths. Use `build/test_model.py` as the test harness.

**Available models and their cache dirs:**

| Model ID | Cache Dir | Notes |
|----------|-----------|-------|
| Qwen/Qwen3-4B | /nvme4/.../hub | Dense |
| Qwen/Qwen3-4B-AWQ | /nvme4/.../hub | Quantized |
| Qwen/Qwen3-30B-A3B | /nvme4/.../hub | MoE |
| Qwen/Qwen3-30B-A3B-FP8 | /nvme4/.../hub | MoE FP8 |
| /data/model/Qwen3-30B-A3B-GPTQ-Int4 | /host_mnt/160_nvme4/.../hub | MoE GPTQ |
| openai/gpt-oss-20b | /host_mnt/160_nvme4/.../hub | MoE packed experts |
| unsloth/gpt-oss-20b-BF16 | /host_mnt/160_nvme4/.../hub | MoE BF16 |
| zai-org/GLM-4.7-Flash | /nvme2/.../hub | MLA MoE |
| QuantTrio/GLM-4.7-Flash-AWQ | /host_mnt/160_nvme4/.../hub | MLA MoE AWQ |
| Qwen/Qwen3.5-27B | /host_mnt/160_nvme4/.../hub | Linear attention |
| Qwen/Qwen3.5-35B-A3B | /host_mnt/160_nvme4/.../hub | Linear attn + MoE |
| Qwen/Qwen3.5-35B-A3B-FP8 | /host_mnt/160_nvme4/.../hub | Linear attn + MoE FP8 |
| QuantTrio/Qwen3.5-35B-A3B-AWQ | /host_mnt/160_nvme4/.../hub | Linear attn + MoE AWQ |

- [ ] **Step 1: Test each model with TP=1**

For each model, run:
```bash
PYTHONPATH=lmdeploy:build/lib CUDA_VISIBLE_DEVICES=0 python build/test_model.py "<model_id>" "<cache_dir>" 1 "0"
```

Expected: All 13 models show `PASS | tp=1`.

- [ ] **Step 2: Test each model with TP=2**

For each model, run:
```bash
PYTHONPATH=lmdeploy:build/lib CUDA_VISIBLE_DEVICES=0,1 python build/test_model.py "<model_id>" "<cache_dir>" 2 "0,1"
```

Expected: All 13 models show `PASS | tp=2`.

- [ ] **Step 3: Final commit if any fixes were needed**

If any test failures required fixes, commit them:
```bash
git add -u
git commit -m "fix: address issues found during ensure_child removal testing"
```
