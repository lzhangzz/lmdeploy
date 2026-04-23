# X-Macro Config Fields Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ConfigField\<T\> self-registration with X-macro field lists and static `for_each` member functions, yielding plain C++ members, default copy/move, and zero per-instance overhead.

**Architecture:** Each config struct defines an X-macro field list inside its body, expanded twice: once for struct members (`TM_MEMBER`), once inside a `TM_FOR_EACH`-generated static `for_each` method. `for_each` passes each field's name and member pointer to a visitor, enabling generic `bind_config<T>()` with `def_readwrite` and generic introspection.

**Tech Stack:** C++17, pybind11, Python, ninja build

**Spec:** `docs/superpowers/specs/2026-04-09-xmacro-config-fields-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `src/turbomind/core/module_config.h` | TM_MEMBER/TM_PTR/TM_FOR_EACH macros, ModuleConfig base, all 8 config structs with X-macro field lists and for_each |
| `src/turbomind/python/bind.cpp` | Generic `bind_config<T>()` using `for_each` + `def_readwrite`, replaces old bind_field/bind_config helpers |
| `src/turbomind/models/ffn_weight.cc` | Revert `static_cast<const int&>` workaround back to direct `static_cast<ActivationType>(cfg.act_type)` |
| `src/turbomind/models/moe_weight.cc` | Revert `static_cast<const int&>` and `static_cast<const double&>` workarounds |

---

### Task 1: Rewrite module_config.h with X-macro approach

**Files:**
- Modify: `src/turbomind/core/module_config.h`

This replaces the entire file — remove ConfigField\<T\>, FieldDescriptor, FieldType enum, FieldTypeTag, TM_CONFIG_FIELD macro, and all custom copy constructors. Replace with TM_MEMBER/TM_PTR/TM_FOR_EACH macros, plain ModuleConfig base, and X-macro-based config structs.

- [ ] **Step 1: Replace module_config.h with the new implementation**

Write the complete new file:

```cpp
// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <string>
#include <utility>

#include "src/turbomind/core/data_type.h"

namespace turbomind::core {

// ======================================================================
// X-macro config field infrastructure
// ======================================================================

#define TM_MEMBER(Type, name, ...) Type name{__VA_ARGS__};
#define TM_PTR(Type, name, ...)    visitor(#name, &Config::name);
#define TM_FOR_EACH(ClassName, field_list) \
    template<typename Visitor> \
    static void for_each(Visitor&& visitor) { \
        using Config = ClassName; \
        field_list(TM_PTR) \
    }

// ======================================================================
// ModuleConfig — plain base
// ======================================================================

struct ModuleConfig {
    std::string_view module_type;
};

// ======================================================================
// Config structs — X-macro field lists + for_each
// ======================================================================

struct LinearConfig: ModuleConfig {
    LinearConfig(): ModuleConfig{"LinearWeight"} {}

    #define LINEAR_FIELDS(X) \
        X(int,      input_dim) \
        X(int,      output_dim) \
        X(DataType, data_type) \
        X(bool,     has_bias)

    LINEAR_FIELDS(TM_MEMBER)
    TM_FOR_EACH(LinearConfig, LINEAR_FIELDS)

    #undef LINEAR_FIELDS
};

struct AttentionConfig: ModuleConfig {
    AttentionConfig(): ModuleConfig{"AttentionWeight"} {}

    #define ATTENTION_FIELDS(X) \
        X(int,      hidden_dim) \
        X(int,      head_dim) \
        X(int,      head_num) \
        X(int,      kv_head_num) \
        X(int,      kv_lora_rank) \
        X(int,      q_lora_rank) \
        X(int,      qk_rope_dim) \
        X(int,      v_head_dim) \
        X(bool,     has_bias) \
        X(bool,     qk_norm) \
        X(int,      tp_size) \
        X(int,      tp_rank) \
        X(DataType, data_type) \
        X(int,      window_size, -1) \
        X(bool,     attn_sink) \
        X(bool,     attn_output_gate)

    ATTENTION_FIELDS(TM_MEMBER)
    TM_FOR_EACH(AttentionConfig, ATTENTION_FIELDS)

    #undef ATTENTION_FIELDS
};

struct FfnConfig: ModuleConfig {
    FfnConfig(): ModuleConfig{"FfnWeight"} {}

    #define FFN_FIELDS(X) \
        X(int,      hidden_dim) \
        X(int,      inter_size) \
        X(bool,     has_bias) \
        X(int,      tp_size) \
        X(int,      tp_rank) \
        X(DataType, data_type) \
        X(int,      act_type) \
        X(bool,     fuse_silu) \
        X(bool,     fused_moe)

    FFN_FIELDS(TM_MEMBER)
    TM_FOR_EACH(FfnConfig, FFN_FIELDS)

    #undef FFN_FIELDS
};

struct MoeConfig: ModuleConfig {
    MoeConfig(): ModuleConfig{"MoeWeight"} {}

    #define MOE_FIELDS(X) \
        X(int,         layer_id) \
        X(int,         method) \
        X(int,         experts_per_token) \
        X(int,         inter_size) \
        X(bool,        norm_topk_prob) \
        X(bool,        shared_gate) \
        X(double,      routed_scale) \
        X(bool,        router_bias) \
        X(int,         topk_group) \
        X(std::string, topk_method) \
        X(int,         n_group) \
        X(std::string, scoring_func) \
        X(int,         router_n_groups) \
        X(int,         expert_num) \
        X(int,         hidden_dim) \
        X(bool,        mlp_bias) \
        X(DataType,    data_type) \
        X(int,         tp_size) \
        X(int,         tp_rank) \
        X(int,         act_type) \
        X(bool,        fuse_silu)

    MOE_FIELDS(TM_MEMBER)
    TM_FOR_EACH(MoeConfig, MOE_FIELDS)

    #undef MOE_FIELDS
};

struct DeltaNetConfig: ModuleConfig {
    DeltaNetConfig(): ModuleConfig{"DeltaNetWeight"} {}

    #define DELTANET_FIELDS(X) \
        X(int,      hidden_dim) \
        X(int,      num_k_heads) \
        X(int,      num_v_heads) \
        X(int,      key_head_dim) \
        X(int,      value_head_dim) \
        X(int,      d_conv, 4) \
        X(bool,     has_bias) \
        X(int,      tp_size) \
        X(int,      tp_rank) \
        X(DataType, data_type)

    DELTANET_FIELDS(TM_MEMBER)
    TM_FOR_EACH(DeltaNetConfig, DELTANET_FIELDS)

    #undef DELTANET_FIELDS
};

struct ModuleListConfig: ModuleConfig {
    ModuleListConfig(): ModuleConfig{"ModuleList"} {}
    template<typename Visitor>
    static void for_each(Visitor&&) {}
};

struct NormConfig: ModuleConfig {
    NormConfig(): ModuleConfig{"NormWeight"} {}

    #define NORM_FIELDS(X) \
        X(int,      dim) \
        X(DataType, data_type)

    NORM_FIELDS(TM_MEMBER)
    TM_FOR_EACH(NormConfig, NORM_FIELDS)

    #undef NORM_FIELDS
};

struct DecoderLayerConfig: ModuleConfig {
    DecoderLayerConfig(): ModuleConfig{"DecoderLayerWeight"} {}
    template<typename Visitor>
    static void for_each(Visitor&&) {}
};

}  // namespace turbomind::core
```

- [ ] **Step 2: Revert static_cast workarounds in ffn_weight.cc**

In `src/turbomind/models/ffn_weight.cc`, revert the `static_cast<const int&>` wrappers. Fields are now plain `int`, so direct `static_cast<ActivationType>` works.

Line 17 — change:
```cpp
    , act_type_{static_cast<ActivationType>(static_cast<const int&>(cfg.act_type))}
```
to:
```cpp
    , act_type_{static_cast<ActivationType>(cfg.act_type)}
```

Line 18 — change:
```cpp
    , is_fused_silu_{cfg.fuse_silu && static_cast<ActivationType>(static_cast<const int&>(cfg.act_type)) == ActivationType::kSilu}
```
to:
```cpp
    , is_fused_silu_{cfg.fuse_silu && static_cast<ActivationType>(cfg.act_type) == ActivationType::kSilu}
```

- [ ] **Step 3: Revert static_cast workarounds in moe_weight.cc**

In `src/turbomind/models/moe_weight.cc`, revert the workarounds.

Line 14 — change:
```cpp
    moe_param_.method = static_cast<MoeParam::Method>(static_cast<const int&>(cfg.method));
```
to:
```cpp
    moe_param_.method = static_cast<MoeParam::Method>(cfg.method);
```

Line 19 — change:
```cpp
    moe_param_.routed_scale = static_cast<float>(static_cast<const double&>(cfg.routed_scale));
```
to:
```cpp
    moe_param_.routed_scale = static_cast<float>(cfg.routed_scale);
```

Line 32 — change:
```cpp
    act_type_ = static_cast<ActivationType>(static_cast<const int&>(cfg.act_type));
```
to:
```cpp
    act_type_ = static_cast<ActivationType>(cfg.act_type);
```

- [ ] **Step 4: Update bind.cpp — replace bind_field/bind_config helpers and config binding section**

Replace the old config binding helpers (lines 306–343) with the new generic `bind_config` using `for_each`:

Replace lines 306–343 (from `// --- Generic config binding helpers ---` through the closing `}` of the old `bind_config`) with:

```cpp
// --- Generic config binding helper ---

template<typename Config>
void bind_config(py::module_& m, const char* name) {
    py::class_<Config, turbomind::core::ModuleConfig> cls(m, name);
    cls.def(py::init<>());
    Config::for_each([&](const char* fname, auto member_ptr) {
        cls.def_readwrite(fname, member_ptr);
    });
    cls.def("clone", [](const Config& c) { return Config(c); });
}
```

Also update the ModuleConfig binding. The current binding uses `def_readwrite` with `std::string`, but ModuleConfig now uses `std::string_view`. Change:

```cpp
    py::class_<turbomind::core::ModuleConfig>(m, "ModuleConfig")
        .def_readwrite("module_type", &turbomind::core::ModuleConfig::module_type);
```

to:

```cpp
    py::class_<turbomind::core::ModuleConfig>(m, "ModuleConfig")
        .def_property("module_type",
            [](const turbomind::core::ModuleConfig& c) -> std::string { return std::string(c.module_type); },
            [](turbomind::core::ModuleConfig& c, const std::string& v) { c.module_type = v; });
```

`string_view` can't be directly exposed as a Python `str` via `def_readwrite`, so we use `def_property` with explicit string conversion.

The 8 `bind_config<...>(m, "...")` calls remain the same — no changes needed.

- [ ] **Step 5: Build to verify C++ compilation**

Run: `cd /data/lmdeploy-modeling/build && ninja 2>&1 | head -80`
Expected: Successful compilation. All 31+ consumer files should compile because fields are plain types (int, bool, etc.) — same names, same access patterns.

- [ ] **Step 6: Commit**

```bash
git add src/turbomind/core/module_config.h src/turbomind/python/bind.cpp src/turbomind/models/ffn_weight.cc src/turbomind/models/moe_weight.cc
git commit -m "refactor(config): replace ConfigField<T> with X-macro field lists and static for_each"
```

---

### Task 2: Smoke test — verify Python bindings work

**Files:**
- None (testing only)

- [ ] **Step 1: Run Python smoke test**

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

- [ ] **Step 2: Run model inference test**

Use the turbomind-tester agent to run a quick model inference test (e.g., Qwen2.5-0.5B with 128+ token generation). Verify the response contains meaningful words, not gibberish.
