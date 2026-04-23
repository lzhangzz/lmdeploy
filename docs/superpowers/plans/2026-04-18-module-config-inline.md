# Module Config Inline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Co-locate each config struct with its corresponding weight class by inlining configs into `_weight.h` files, then delete `core/module_config.h`.

**Architecture:** Move X-macro infrastructure + `ModuleConfig` base + `ModuleListConfig` into `core/module.h`. Forward-declare `ModuleConfig` in `core/registry.h` to break the circular dependency. Inline each derived config struct into its `_weight.h` above the weight class definition, keeping it in `core` namespace.

**Tech Stack:** C++ header reorganization, ninja build system.

---

### Task 1: Move infrastructure into `core/module.h`

**Files:**
- Modify: `src/turbomind/core/module.h:1-20` (add infrastructure before Module class)
- Modify: `src/turbomind/core/module_config.h:1-178` (strip to config structs only)

- [ ] **Step 1: Add X-macro infrastructure + ModuleConfig + ModuleListConfig to `module.h`**

Insert the following block into `src/turbomind/core/module.h` **after** the includes (after line 17) and **before** `namespace turbomind::core {`, i.e. between the includes block and the namespace opening. Remove the old `#include "src/turbomind/core/module_config.h"` (line 15).

The `#include <string>` is already present (line 7). Add `#include <string_view>` if not present.

Insert this block between the includes and the X-macro expansion comment block (before line 22):

```cpp
#include <string_view>

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
// ModuleConfig — plain base for typed config structs
// ======================================================================

struct ModuleConfig {
    std::string_view module_type;
};

struct ModuleListConfig: ModuleConfig {
    ModuleListConfig(): ModuleConfig{"ModuleList"} {}
    template<typename Visitor>
    static void for_each(Visitor&&) {}
};
```

Remove line 15: `#include "src/turbomind/core/module_config.h"`

- [ ] **Step 2: Strip `module_config.h` to config structs only**

Replace the entire content of `src/turbomind/core/module_config.h` with:

```cpp
// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

// Transitional header — config structs pending migration to their _weight.h files.
// Infrastructure (TM_MEMBER, TM_PTR, TM_FOR_EACH, ModuleConfig) lives in module.h.

#include "src/turbomind/core/module.h"

namespace turbomind::core {

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
        X(bool,     attn_output_gate) \
        X(int,      rope_dim) \
        X(int,      repeat_kv) \
        X(int,      qk_nope_dim)

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

Note: This file now includes `module.h` instead of defining the macros itself. No circular dependency since `module.h` no longer includes `module_config.h`.

- [ ] **Step 3: Build to verify no breakage**

Run: `cd /data/lmdeploy-modeling/build && ninja`
Expected: Clean build, no errors.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/core/module.h src/turbomind/core/module_config.h
git commit -m "refactor(config): move X-macro infrastructure and ModuleConfig base into module.h"
```

---

### Task 2: Forward-declare `ModuleConfig` in `core/registry.h`

**Files:**
- Modify: `src/turbomind/core/registry.h:9`

- [ ] **Step 1: Replace include with forward declaration**

In `src/turbomind/core/registry.h`, replace line 9:

```cpp
#include "src/turbomind/core/module_config.h"
```

with:

```cpp
namespace turbomind::core {
struct ModuleConfig;
}  // namespace turbomind::core
```

- [ ] **Step 2: Build to verify**

Run: `cd /data/lmdeploy-modeling/build && ninja`
Expected: Clean build. The factory `std::function<std::unique_ptr<Module>(const ModuleConfig&)>` only needs a declaration in the header.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/core/registry.h
git commit -m "refactor(registry): forward-declare ModuleConfig, remove module_config.h include"
```

---

### Task 3: Inline `LinearConfig` into `linear_weight.h`

**Files:**
- Modify: `src/turbomind/models/linear_weight.h`
- Modify: `src/turbomind/core/module_config.h`

- [ ] **Step 1: Add LinearConfig and remove module_config.h include**

In `src/turbomind/models/linear_weight.h`:

Remove line 7: `#include "src/turbomind/core/module_config.h"`

Insert the following block between line 8 (`#include "src/turbomind/kernels/gemm/types.h"`) and `namespace turbomind {` (line 10), inside a `namespace turbomind::core` block:

```cpp
namespace turbomind::core {

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

}  // namespace turbomind::core
```

- [ ] **Step 2: Remove LinearConfig from module_config.h**

Delete the `LinearConfig` struct definition (lines 35-48 in the original) from `src/turbomind/core/module_config.h`.

- [ ] **Step 3: Build to verify**

Run: `cd /data/lmdeploy-modeling/build && ninja`
Expected: Clean build.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/linear_weight.h src/turbomind/core/module_config.h
git commit -m "refactor(config): inline LinearConfig into linear_weight.h"
```

---

### Task 4: Inline `AttentionConfig` into `attention_weight.h`

**Files:**
- Modify: `src/turbomind/models/attention_weight.h`
- Modify: `src/turbomind/core/module_config.h`

- [ ] **Step 1: Add AttentionConfig and remove module_config.h include**

In `src/turbomind/models/attention_weight.h`:

Remove line 6: `#include "src/turbomind/core/module_config.h"`

Insert the following block between the includes and `namespace turbomind {`, inside a `namespace turbomind::core` block:

```cpp
namespace turbomind::core {

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
        X(bool,     attn_output_gate) \
        X(int,      rope_dim) \
        X(int,      repeat_kv) \
        X(int,      qk_nope_dim)

    ATTENTION_FIELDS(TM_MEMBER)
    TM_FOR_EACH(AttentionConfig, ATTENTION_FIELDS)

    #undef ATTENTION_FIELDS
};

}  // namespace turbomind::core
```

- [ ] **Step 2: Remove AttentionConfig from module_config.h**

Delete the `AttentionConfig` struct definition from `src/turbomind/core/module_config.h`.

- [ ] **Step 3: Build to verify**

Run: `cd /data/lmdeploy-modeling/build && ninja`
Expected: Clean build.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/attention_weight.h src/turbomind/core/module_config.h
git commit -m "refactor(config): inline AttentionConfig into attention_weight.h"
```

---

### Task 5: Inline `FfnConfig` into `ffn_weight.h`

**Files:**
- Modify: `src/turbomind/models/ffn_weight.h`
- Modify: `src/turbomind/core/module_config.h`

- [ ] **Step 1: Add FfnConfig and remove module_config.h include**

In `src/turbomind/models/ffn_weight.h`:

Remove line 6: `#include "src/turbomind/core/module_config.h"`

Insert the following block between the includes and `namespace turbomind {`, inside a `namespace turbomind::core` block:

```cpp
namespace turbomind::core {

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

}  // namespace turbomind::core
```

- [ ] **Step 2: Remove FfnConfig from module_config.h**

Delete the `FfnConfig` struct definition from `src/turbomind/core/module_config.h`.

- [ ] **Step 3: Build to verify**

Run: `cd /data/lmdeploy-modeling/build && ninja`
Expected: Clean build.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/ffn_weight.h src/turbomind/core/module_config.h
git commit -m "refactor(config): inline FfnConfig into ffn_weight.h"
```

---

### Task 6: Inline `MoeConfig` into `moe_weight.h`

**Files:**
- Modify: `src/turbomind/models/moe_weight.h`
- Modify: `src/turbomind/core/module_config.h`

- [ ] **Step 1: Add MoeConfig and remove module_config.h include**

In `src/turbomind/models/moe_weight.h`:

Remove line 6: `#include "src/turbomind/core/module_config.h"`

Insert the following block between the includes and `namespace turbomind {`, inside a `namespace turbomind::core` block:

```cpp
namespace turbomind::core {

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

}  // namespace turbomind::core
```

Note: `MoeConfig` uses `std::string`, so `moe_weight.h` needs `#include <string>` added (check if already transitive through other includes — `core.h` likely provides it, but add explicitly if build fails).

- [ ] **Step 2: Remove MoeConfig from module_config.h**

Delete the `MoeConfig` struct definition from `src/turbomind/core/module_config.h`.

- [ ] **Step 3: Build to verify**

Run: `cd /data/lmdeploy-modeling/build && ninja`
Expected: Clean build.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/moe_weight.h src/turbomind/core/module_config.h
git commit -m "refactor(config): inline MoeConfig into moe_weight.h"
```

---

### Task 7: Inline `DeltaNetConfig` into `delta_net_weight.h`

**Files:**
- Modify: `src/turbomind/models/delta_net_weight.h`
- Modify: `src/turbomind/core/module_config.h`

- [ ] **Step 1: Add DeltaNetConfig and remove module_config.h include**

In `src/turbomind/models/delta_net_weight.h`:

Remove line 6: `#include "src/turbomind/core/module_config.h"`

Insert the following block between the includes and `namespace turbomind {`, inside a `namespace turbomind::core` block:

```cpp
namespace turbomind::core {

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

}  // namespace turbomind::core
```

- [ ] **Step 2: Remove DeltaNetConfig from module_config.h**

Delete the `DeltaNetConfig` struct definition from `src/turbomind/core/module_config.h`.

- [ ] **Step 3: Build to verify**

Run: `cd /data/lmdeploy-modeling/build && ninja`
Expected: Clean build.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/delta_net_weight.h src/turbomind/core/module_config.h
git commit -m "refactor(config): inline DeltaNetConfig into delta_net_weight.h"
```

---

### Task 8: Inline `NormConfig` into `norm_weight.h`

**Files:**
- Modify: `src/turbomind/models/norm_weight.h`
- Modify: `src/turbomind/models/norm_weight.cc`
- Modify: `src/turbomind/core/module_config.h`

- [ ] **Step 1: Add NormConfig and remove module_config.h include in header**

In `src/turbomind/models/norm_weight.h`:

Remove line 6: `#include "src/turbomind/core/module_config.h"`

Insert the following block between the includes and `namespace turbomind {`, inside a `namespace turbomind::core` block:

```cpp
namespace turbomind::core {

struct NormConfig: ModuleConfig {
    NormConfig(): ModuleConfig{"NormWeight"} {}

    #define NORM_FIELDS(X) \
        X(int,      dim) \
        X(DataType, data_type)

    NORM_FIELDS(TM_MEMBER)
    TM_FOR_EACH(NormConfig, NORM_FIELDS)

    #undef NORM_FIELDS
};

}  // namespace turbomind::core
```

- [ ] **Step 2: Remove module_config.h include from norm_weight.cc**

In `src/turbomind/models/norm_weight.cc`, remove line 5:

```cpp
#include "src/turbomind/core/module_config.h"
```

This is no longer needed since `norm_weight.h` now provides `NormConfig` inline.

- [ ] **Step 3: Remove NormConfig from module_config.h**

Delete the `NormConfig` struct definition from `src/turbomind/core/module_config.h`.

- [ ] **Step 4: Build to verify**

Run: `cd /data/lmdeploy-modeling/build && ninja`
Expected: Clean build.

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/models/norm_weight.h src/turbomind/models/norm_weight.cc src/turbomind/core/module_config.h
git commit -m "refactor(config): inline NormConfig into norm_weight.h"
```

---

### Task 9: Inline `DecoderLayerConfig` into `decoder_layer_weight.h`

**Files:**
- Modify: `src/turbomind/models/decoder_layer_weight.h`
- Modify: `src/turbomind/core/module_config.h`

- [ ] **Step 1: Add DecoderLayerConfig and remove module_config.h include**

In `src/turbomind/models/decoder_layer_weight.h`:

Remove line 5: `#include "src/turbomind/core/module_config.h"`

Insert the following block between the includes and `namespace turbomind {`, inside a `namespace turbomind::core` block:

```cpp
namespace turbomind::core {

struct DecoderLayerConfig: ModuleConfig {
    DecoderLayerConfig(): ModuleConfig{"DecoderLayerWeight"} {}
    template<typename Visitor>
    static void for_each(Visitor&&) {}
};

}  // namespace turbomind::core
```

- [ ] **Step 2: Remove DecoderLayerConfig from module_config.h**

Delete the `DecoderLayerConfig` struct definition from `src/turbomind/core/module_config.h`.

At this point `module_config.h` should be empty except for the copyright, pragma once, and include of module.h — all config structs have been moved out.

- [ ] **Step 3: Build to verify**

Run: `cd /data/lmdeploy-modeling/build && ninja`
Expected: Clean build.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/decoder_layer_weight.h src/turbomind/core/module_config.h
git commit -m "refactor(config): inline DecoderLayerConfig into decoder_layer_weight.h"
```

---

### Task 10: Update `bind.cpp`, delete `module_config.h`, final verification

**Files:**
- Modify: `src/turbomind/python/bind.cpp:20`
- Delete: `src/turbomind/core/module_config.h`

- [ ] **Step 1: Update bind.cpp includes**

In `src/turbomind/python/bind.cpp`, replace line 20:

```cpp
#include "src/turbomind/core/module_config.h"
```

with:

```cpp
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/attention_weight.h"
#include "src/turbomind/models/ffn_weight.h"
#include "src/turbomind/models/moe_weight.h"
#include "src/turbomind/models/delta_net_weight.h"
#include "src/turbomind/models/norm_weight.h"
#include "src/turbomind/models/decoder_layer_weight.h"
```

`ModuleListConfig` is available through `module.h` (already included by bind.cpp).

- [ ] **Step 2: Delete module_config.h**

```bash
rm src/turbomind/core/module_config.h
```

- [ ] **Step 3: Build to verify**

Run: `cd /data/lmdeploy-modeling/build && ninja`
Expected: Clean build. No remaining references to module_config.h.

- [ ] **Step 4: Verify no stale includes**

Run: `grep -r 'module_config\.h' src/`
Expected: No results.

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/python/bind.cpp
git rm src/turbomind/core/module_config.h
git commit -m "refactor(config): update bind.cpp includes, delete module_config.h"
```
