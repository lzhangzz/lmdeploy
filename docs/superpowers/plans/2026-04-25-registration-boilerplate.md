# Registration Boilerplate Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace 9 identical ~12-line registration blocks with a template method + one-line macro.

**Architecture:** Add a templated `register_type<T, CfgT>()` overload to `ModuleRegistry` and a `TM_MODULE_REGISTER` macro in `registry.h`, then replace all 9 boilerplate blocks with one-liners.

**Tech Stack:** C++17, existing ModuleRegistry infrastructure.

---

### Task 1: Add template overload and macro to registry.h

**Files:**
- Modify: `src/turbomind/core/registry.h`

- [ ] **Step 1: Add template method to ModuleRegistry class and TM_MODULE_REGISTER macro**

In `registry.h`, add the template overload to the `ModuleRegistry` class (after the existing `register_type` declaration on line 24) and add the macro after the class definition:

```cpp
// After line 24 (existing register_type declaration):
template<typename T, typename CfgT = ModuleConfig>
void register_type(const std::string& name)
{
    register_type(name, [](const ModuleConfig& cfg) -> std::unique_ptr<Module> {
        return std::make_unique<T>(static_cast<const CfgT&>(cfg));
    });
}
```

After the closing `}  // namespace turbomind::core` on line 39, add:

```cpp
#define TM_MODULE_REGISTER(ModuleClass, ConfigType)                              \
    namespace {                                                                   \
    static const bool _tm_module_registered_ =                                   \
        ::turbomind::core::ModuleRegistry::instance()                             \
            .register_type<ModuleClass, ConfigType>(#ModuleClass);                \
    }
```

- [ ] **Step 2: Build to verify compilation**

Run: `cd build && ninja _turbomind`
Expected: Clean build with no errors.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/core/registry.h
git commit -m "feat: add template register_type overload and TM_MODULE_REGISTER macro"
```

---

### Task 2: Add DecoderLayerWeight config constructor

`DecoderLayerWeight` has only `DecoderLayerWeight() = default;` but the template always passes a config arg. Add a constructor that accepts (and ignores) the config, matching the `ModuleList` pattern.

**Files:**
- Modify: `src/turbomind/models/decoder_layer_weight.h:29`

- [ ] **Step 1: Add constructor**

On line 29, after `DecoderLayerWeight() = default;`, add:

```cpp
DecoderLayerWeight(const core::ModuleConfig&) {}
```

The class should look like:

```cpp
DecoderLayerWeight() = default;
DecoderLayerWeight(const core::ModuleConfig&) {}
```

- [ ] **Step 2: Build to verify compilation**

Run: `cd build && ninja _turbomind`
Expected: Clean build.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/models/decoder_layer_weight.h
git commit -m "refactor: add config-accepting constructor to DecoderLayerWeight"
```

---

### Task 3: Replace boilerplate in 8 model weight files

**Files:**
- Modify: `src/turbomind/models/model_weight.cc:85-97`
- Modify: `src/turbomind/models/attention_weight.cc:93-105`
- Modify: `src/turbomind/models/decoder_layer_weight.cc:34-45`
- Modify: `src/turbomind/models/ffn_weight.cc:46-58`
- Modify: `src/turbomind/models/linear_weight.cc:278-290`
- Modify: `src/turbomind/models/moe_weight.cc:166-178`
- Modify: `src/turbomind/models/norm_weight.cc:20-32`
- Modify: `src/turbomind/models/delta_net_weight.cc:33-45`

- [ ] **Step 1: Replace each registration block with TM_MODULE_REGISTER**

Each file has a block matching this pattern (varying only in class/config names):

```cpp
namespace {
struct XxxRegistrar {
    XxxRegistrar() {
        core::ModuleRegistry::instance().register_type(
            "Xxx",
            [](const core::ModuleConfig& base_cfg) -> std::unique_ptr<core::Module> {
                return std::make_unique<Xxx>(
                    static_cast<const core::XxxConfig&>(base_cfg));
            });
    }
};
static XxxRegistrar _xxx_reg;
}  // anonymous namespace
```

Replace each with the corresponding one-liner:

| File (lines) | Replacement |
|---|---|
| `model_weight.cc:85-97` | `TM_MODULE_REGISTER(ModelWeight, core::ModelWeightConfig);` |
| `attention_weight.cc:93-105` | `TM_MODULE_REGISTER(AttentionWeight, core::AttentionConfig);` |
| `decoder_layer_weight.cc:34-45` | `TM_MODULE_REGISTER(DecoderLayerWeight, core::ModuleConfig);` |
| `ffn_weight.cc:46-58` | `TM_MODULE_REGISTER(FfnWeight, core::FfnConfig);` |
| `linear_weight.cc:278-290` | `TM_MODULE_REGISTER(LinearWeight, core::LinearConfig);` |
| `moe_weight.cc:166-178` | `TM_MODULE_REGISTER(MoeWeight, core::MoeConfig);` |
| `norm_weight.cc:20-32` | `TM_MODULE_REGISTER(NormWeight, core::NormConfig);` |
| `delta_net_weight.cc:33-45` | `TM_MODULE_REGISTER(DeltaNetWeight, core::DeltaNetConfig);` |

- [ ] **Step 2: Build to verify compilation**

Run: `cd build && ninja _turbomind`
Expected: Clean build with no errors.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/models/model_weight.cc src/turbomind/models/attention_weight.cc src/turbomind/models/decoder_layer_weight.cc src/turbomind/models/ffn_weight.cc src/turbomind/models/linear_weight.cc src/turbomind/models/moe_weight.cc src/turbomind/models/norm_weight.cc src/turbomind/models/delta_net_weight.cc
git commit -m "refactor: replace registration boilerplate with TM_MODULE_REGISTER in model weights"
```

---

### Task 4: Replace boilerplate in module.cc

**Files:**
- Modify: `src/turbomind/core/module.cc:191-202`

- [ ] **Step 1: Replace the ModuleList registration block**

The existing block (lines 191-202):

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

Replace with:

```cpp
TM_MODULE_REGISTER(ModuleList, ModuleListConfig);
```

Note: `module.cc` is inside `namespace turbomind::core`, so `ModuleList` and `ModuleListConfig` resolve without the `core::` qualifier. `#ModuleList` stringifies to `"ModuleList"` matching the existing registered name.

- [ ] **Step 2: Build to verify compilation**

Run: `cd build && ninja _turbomind`
Expected: Clean build with no errors.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/core/module.cc
git commit -m "refactor: replace ModuleList registration boilerplate with TM_MODULE_REGISTER"
```

---

### Task 5: Verify with model test

**Files:** None (verification only)

- [ ] **Step 1: Check GPU is free**

Run: Use `get_gpu_usage` MCP tool.
Expected: GPU memory usage near zero.

- [ ] **Step 2: Run model test**

Run: `cd /data/lmdeploy-modeling && python scripts/test_turbomind_model.py <model_id> --tp 1`
Expected: Model loads, responds with coherent text to the test prompt, at least 128 tokens.

- [ ] **Step 3: Verify response quality**

Read the output and confirm the model responds with meaningful, relevant text. Gibberish = bug.
