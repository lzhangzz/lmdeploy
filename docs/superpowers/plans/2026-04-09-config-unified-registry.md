# Self-Registering Config Fields Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace plain POD config struct fields with self-registering `ConfigField<T>` wrappers that inherit from `FieldDescriptor`, enabling automatic field iteration and zero-boilerplate pybind11 binding.

**Architecture:** `ConfigField<T>` inherits `FieldDescriptor` (name + type_tag). Each field self-registers with its parent `ModuleConfig` on construction via `{*this, "name"}` NSDMI. `ModuleConfig` stores a `vector<FieldDescriptor*>` for iteration. Copy construction clears the vector, lets fields re-register via NSDMI, then copies values. pybind11 binding uses a generic `bind_config<T>()` template with `def_property` and name-based field lookup.

**Tech Stack:** C++17, pybind11, Python, ninja build

**Spec:** `docs/superpowers/specs/2026-04-09-self-registering-config-fields-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `src/turbomind/core/module_config.h` | FieldDescriptor, FieldType enum, FieldTypeTag, ConfigField<T>, TM_CONFIG_FIELD macro, updated ModuleConfig, all 8 config structs |
| `src/turbomind/python/bind.cpp` | Replace 100+ lines of manual def_readwrite with 8 bind_config<T>() calls |

---

### Task 1: Update module_config.h — infrastructure + all config structs

**Files:**
- Modify: `src/turbomind/core/module_config.h`

This task adds all the infrastructure (FieldDescriptor, FieldType, FieldTypeTag, ConfigField<T>, TM_CONFIG_FIELD macro, updated ModuleConfig) and converts all 8 config structs. Everything is in one header so it must be done atomically to compile.

- [ ] **Step 1: Replace the entire file with the new implementation**

```cpp
// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "src/turbomind/core/data_type.h"

namespace turbomind::core {

// ======================================================================
// Self-registering config field infrastructure
// ======================================================================

enum class FieldType : uint8_t { Int, Bool, Double, String, DataType };

struct FieldDescriptor {
    const char* name;
    FieldType   type_tag;
};

template<typename T> struct FieldTypeTag;
template<> struct FieldTypeTag<int>         { static constexpr FieldType v = FieldType::Int; };
template<> struct FieldTypeTag<bool>        { static constexpr FieldType v = FieldType::Bool; };
template<> struct FieldTypeTag<double>      { static constexpr FieldType v = FieldType::Double; };
template<> struct FieldTypeTag<std::string> { static constexpr FieldType v = FieldType::String; };
template<> struct FieldTypeTag<DataType>    { static constexpr FieldType v = FieldType::DataType; };

// Forward declaration for ConfigField
struct ModuleConfig;

template<typename T>
class ConfigField: public FieldDescriptor {
    T value_;

public:
    template<typename... Args>
    ConfigField(ModuleConfig& parent, const char* name, Args&&... args)
        : FieldDescriptor{name, FieldTypeTag<T>::v}
        , value_(std::forward<Args>(args)...)
    {
        parent.register_field(this);
    }

    // Implicit conversions — existing code works unchanged
    operator T&()             { return value_; }
    operator const T&() const { return value_; }
    ConfigField& operator=(const T& v) { value_ = v; return *this; }
};

#define TM_CONFIG_FIELD(Type, name, ...) \
    ConfigField<Type> name{*this, #name, ##__VA_ARGS__}

// ======================================================================
// ModuleConfig — base with field registry
// ======================================================================

struct ModuleConfig {
    std::string module_type;

    ModuleConfig() = default;

    // Copy: module_type is copied, fields_ starts empty.
    // ConfigField members self-register via NSDMI, then copy_values_from() copies values.
    ModuleConfig(const ModuleConfig& other)
        : module_type(other.module_type) {}

    ModuleConfig(ModuleConfig&&) = delete;
    ModuleConfig& operator=(ModuleConfig&&) = delete;

    // Copy assignment: fields are already registered, just copy values.
    ModuleConfig& operator=(const ModuleConfig& other) {
        if (this != &other) {
            module_type = other.module_type;
            copy_values_from(other);
        }
        return *this;
    }

    void register_field(FieldDescriptor* f) {
        fields_.push_back(f);
    }

    // Copy values from source by field index (fields registered in declaration order).
    void copy_values_from(const ModuleConfig& src) {
        for (size_t i = 0; i < fields_.size(); ++i) {
            auto* dst   = fields_[i];
            auto* src_f = src.fields_[i];
            switch (dst->type_tag) {
            case FieldType::Int:      static_cast<ConfigField<int>&>(*dst)         = static_cast<const ConfigField<int>&>(*src_f);         break;
            case FieldType::Bool:     static_cast<ConfigField<bool>&>(*dst)        = static_cast<const ConfigField<bool>&>(*src_f);        break;
            case FieldType::Double:   static_cast<ConfigField<double>&>(*dst)      = static_cast<const ConfigField<double>&>(*src_f);      break;
            case FieldType::String:   static_cast<ConfigField<std::string>&>(*dst) = static_cast<const ConfigField<std::string>&>(*src_f); break;
            case FieldType::DataType: static_cast<ConfigField<DataType>&>(*dst)    = static_cast<const ConfigField<DataType>&>(*src_f);    break;
            }
        }
    }

    // Find a field by name.
    FieldDescriptor* field(const char* name) const {
        for (auto* f : fields_)
            if (std::strcmp(f->name, name) == 0) return f;
        return nullptr;
    }

    const std::vector<FieldDescriptor*>& fields() const { return fields_; }

private:
    std::vector<FieldDescriptor*> fields_;
};

// ======================================================================
// Config structs — fields use TM_CONFIG_FIELD for self-registration
// ======================================================================

struct LinearConfig: ModuleConfig {
    LinearConfig(): ModuleConfig{"LinearWeight"} {}
    LinearConfig(const LinearConfig& other): ModuleConfig(other) { copy_values_from(other); }

    TM_CONFIG_FIELD(int,      input_dim);
    TM_CONFIG_FIELD(int,      output_dim);
    TM_CONFIG_FIELD(DataType, data_type);
    TM_CONFIG_FIELD(bool,     has_bias);
};

struct AttentionConfig: ModuleConfig {
    AttentionConfig(): ModuleConfig{"AttentionWeight"} {}
    AttentionConfig(const AttentionConfig& other): ModuleConfig(other) { copy_values_from(other); }

    TM_CONFIG_FIELD(int,      hidden_dim);
    TM_CONFIG_FIELD(int,      head_dim);
    TM_CONFIG_FIELD(int,      head_num);
    TM_CONFIG_FIELD(int,      kv_head_num);
    TM_CONFIG_FIELD(int,      kv_lora_rank);
    TM_CONFIG_FIELD(int,      q_lora_rank);
    TM_CONFIG_FIELD(int,      qk_rope_dim);
    TM_CONFIG_FIELD(int,      v_head_dim);
    TM_CONFIG_FIELD(bool,     has_bias);
    TM_CONFIG_FIELD(bool,     qk_norm);
    TM_CONFIG_FIELD(int,      tp_size);
    TM_CONFIG_FIELD(int,      tp_rank);
    TM_CONFIG_FIELD(DataType, data_type);
    TM_CONFIG_FIELD(int,      window_size, -1);
    TM_CONFIG_FIELD(bool,     attn_sink);
    TM_CONFIG_FIELD(bool,     attn_output_gate);
};

struct FfnConfig: ModuleConfig {
    FfnConfig(): ModuleConfig{"FfnWeight"} {}
    FfnConfig(const FfnConfig& other): ModuleConfig(other) { copy_values_from(other); }

    TM_CONFIG_FIELD(int,      hidden_dim);
    TM_CONFIG_FIELD(int,      inter_size);
    TM_CONFIG_FIELD(bool,     has_bias);
    TM_CONFIG_FIELD(int,      tp_size);
    TM_CONFIG_FIELD(int,      tp_rank);
    TM_CONFIG_FIELD(DataType, data_type);
    TM_CONFIG_FIELD(int,      act_type);
    TM_CONFIG_FIELD(bool,     fuse_silu);
    TM_CONFIG_FIELD(bool,     fused_moe);
};

struct MoeConfig: ModuleConfig {
    MoeConfig(): ModuleConfig{"MoeWeight"} {}
    MoeConfig(const MoeConfig& other): ModuleConfig(other) { copy_values_from(other); }

    TM_CONFIG_FIELD(int,         layer_id);
    TM_CONFIG_FIELD(int,         method);
    TM_CONFIG_FIELD(int,         experts_per_token);
    TM_CONFIG_FIELD(int,         inter_size);
    TM_CONFIG_FIELD(bool,        norm_topk_prob);
    TM_CONFIG_FIELD(bool,        shared_gate);
    TM_CONFIG_FIELD(double,      routed_scale);
    TM_CONFIG_FIELD(bool,        router_bias);
    TM_CONFIG_FIELD(int,         topk_group);
    TM_CONFIG_FIELD(std::string, topk_method);
    TM_CONFIG_FIELD(int,         n_group);
    TM_CONFIG_FIELD(std::string, scoring_func);
    TM_CONFIG_FIELD(int,         router_n_groups);
    TM_CONFIG_FIELD(int,         expert_num);
    TM_CONFIG_FIELD(int,         hidden_dim);
    TM_CONFIG_FIELD(bool,        mlp_bias);
    TM_CONFIG_FIELD(DataType,    data_type);
    TM_CONFIG_FIELD(int,         tp_size);
    TM_CONFIG_FIELD(int,         tp_rank);
    TM_CONFIG_FIELD(int,         act_type);
    TM_CONFIG_FIELD(bool,        fuse_silu);
};

struct DeltaNetConfig: ModuleConfig {
    DeltaNetConfig(): ModuleConfig{"DeltaNetWeight"} {}
    DeltaNetConfig(const DeltaNetConfig& other): ModuleConfig(other) { copy_values_from(other); }

    TM_CONFIG_FIELD(int,      hidden_dim);
    TM_CONFIG_FIELD(int,      num_k_heads);
    TM_CONFIG_FIELD(int,      num_v_heads);
    TM_CONFIG_FIELD(int,      key_head_dim);
    TM_CONFIG_FIELD(int,      value_head_dim);
    TM_CONFIG_FIELD(int,      d_conv, 4);
    TM_CONFIG_FIELD(bool,     has_bias);
    TM_CONFIG_FIELD(int,      tp_size);
    TM_CONFIG_FIELD(int,      tp_rank);
    TM_CONFIG_FIELD(DataType, data_type);
};

struct ModuleListConfig: ModuleConfig {
    ModuleListConfig(): ModuleConfig{"ModuleList"} {}
    ModuleListConfig(const ModuleListConfig& other): ModuleConfig(other) { copy_values_from(other); }
};

struct NormConfig: ModuleConfig {
    NormConfig(): ModuleConfig{"NormWeight"} {}
    NormConfig(const NormConfig& other): ModuleConfig(other) { copy_values_from(other); }

    TM_CONFIG_FIELD(int,      dim);
    TM_CONFIG_FIELD(DataType, data_type);
};

struct DecoderLayerConfig: ModuleConfig {
    DecoderLayerConfig(): ModuleConfig{"DecoderLayerWeight"} {}
    DecoderLayerConfig(const DecoderLayerConfig& other): ModuleConfig(other) { copy_values_from(other); }
};

}  // namespace turbomind::core
```

- [ ] **Step 2: Build to verify C++ compilation**

Run: `cd /data/lmdeploy-modeling/build && ninja 2>&1 | head -80`
Expected: Successful compilation. The implicit conversion operators on `ConfigField<T>` ensure all 31 consumer files compile without changes. Pay attention to any errors — they indicate code patterns that don't go through implicit conversion (e.g., template deduction, auto& bindings). Fix inline.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/core/module_config.h
git commit -m "feat(config): self-registering ConfigField<T> with FieldDescriptor base"
```

---

### Task 2: Update bind.cpp — generic bind_config

**Files:**
- Modify: `src/turbomind/python/bind.cpp`

Replace the 100+ lines of manual config bindings (lines 426–529) with generic `bind_config<T>()` calls plus the binding helpers.

- [ ] **Step 1: Add bind_config helpers and replace config binding section**

Add these helpers before the config binding section (before line 426, after the `MakeLinearWeightFormat` binding). Then replace the entire config binding section (lines 426–529) with the generic calls.

Add before the config section — the `bind_field` and `bind_config` templates:

```cpp
    // --- Generic config binding helpers ---

    template<typename Config, typename T>
    void bind_field(py::class_<Config, turbomind::core::ModuleConfig>& cls, const std::string& name)
    {
        using namespace turbomind::core;
        cls.def_property(name.c_str(),
            [name](const Config& c) -> T {
                return static_cast<const ConfigField<T>&>(*c.field(name.c_str()));
            },
            [name](Config& c, const T& v) {
                static_cast<ConfigField<T>&>(*c.field(name.c_str())) = v;
            });
    }

    template<typename Config>
    void bind_config(py::module_& m, const char* name)
    {
        using namespace turbomind::core;
        py::class_<Config, ModuleConfig> cls(m, name);
        cls.def(py::init<>());

        Config tmp;  // construct to discover registered fields
        for (auto* desc : tmp.fields()) {
            std::string fname(desc->name);
            switch (desc->type_tag) {
            case FieldType::Int:      bind_field<Config, int>(cls, fname);         break;
            case FieldType::Bool:     bind_field<Config, bool>(cls, fname);        break;
            case FieldType::Double:   bind_field<Config, double>(cls, fname);      break;
            case FieldType::String:   bind_field<Config, std::string>(cls, fname); break;
            case FieldType::DataType: bind_field<Config, DataType>(cls, fname);    break;
            }
        }

        cls.def("clone", [](const Config& c) { return Config(c); });
    }
```

Replace lines 426–529 (the entire `// --- Config struct bindings ---` block) with:

```cpp
    // --- Config struct bindings ---
    py::class_<turbomind::core::ModuleConfig>(m, "ModuleConfig")
        .def_readwrite("module_type", &turbomind::core::ModuleConfig::module_type);

    bind_config<turbomind::core::LinearConfig>(m, "LinearConfig");
    bind_config<turbomind::core::AttentionConfig>(m, "AttentionConfig");
    bind_config<turbomind::core::FfnConfig>(m, "FfnConfig");
    bind_config<turbomind::core::MoeConfig>(m, "MoeConfig");
    bind_config<turbomind::core::DeltaNetConfig>(m, "DeltaNetConfig");
    bind_config<turbomind::core::ModuleListConfig>(m, "ModuleListConfig");
    bind_config<turbomind::core::NormConfig>(m, "NormConfig");
    bind_config<turbomind::core::DecoderLayerConfig>(m, "DecoderLayerConfig");
```

Note: `ModuleListConfig` and `DecoderLayerConfig` have no fields, but `bind_config` handles that gracefully (empty loop).

- [ ] **Step 2: Build**

Run: `cd /data/lmdeploy-modeling/build && ninja 2>&1 | head -80`
Expected: Successful compilation of the Python extension.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/python/bind.cpp
git commit -m "refactor(bind): replace manual config bindings with generic bind_config<T>()"
```

---

### Task 3: Smoke test — verify Python bindings work

**Files:**
- None (testing only)

Verify that the Python bindings work end-to-end: field read/write, clone, introspection.

- [ ] **Step 1: Run a quick Python smoke test**

Run:
```bash
cd /data/lmdeploy-modeling && \
PYTHONPATH=$(pwd)/lmdeploy:$(pwd)/build/lib python3 -c "
from _turbomind import LinearConfig, AttentionConfig, NormConfig, MoeConfig

# Test field read/write
cfg = LinearConfig()
cfg.input_dim = 128
cfg.output_dim = 256
cfg.has_bias = True
assert cfg.input_dim == 128
assert cfg.output_dim == 256
assert cfg.has_bias == True
print('LinearConfig read/write: OK')

# Test clone
cfg2 = cfg.clone()
assert cfg2.input_dim == 128
assert cfg2.output_dim == 256
cfg2.input_dim = 512
assert cfg.input_dim == 128  # original unchanged
print('LinearConfig clone: OK')

# Test AttentionConfig with non-zero default
acfg = AttentionConfig()
assert acfg.window_size == -1
acfg.window_size = 4096
assert acfg.window_size == 4096
print('AttentionConfig default + read/write: OK')

# Test MoeConfig with string field
mcfg = MoeConfig()
mcfg.topk_method = 'group'
mcfg.routed_scale = 0.5
assert mcfg.topk_method == 'group'
assert mcfg.routed_scale == 0.5
print('MoeConfig string + double: OK')

# Test NormConfig
ncfg = NormConfig()
ncfg.dim = 64
assert ncfg.dim == 64
print('NormConfig: OK')

print('All smoke tests passed')
"
```
Expected: All assertions pass, "All smoke tests passed" printed.

- [ ] **Step 2: Run a model test to verify end-to-end correctness**

Use the turbomind-tester agent to run a quick model inference test (e.g., Qwen2.5-0.5B with 128+ token generation). Verify the response contains meaningful words, not gibberish.
