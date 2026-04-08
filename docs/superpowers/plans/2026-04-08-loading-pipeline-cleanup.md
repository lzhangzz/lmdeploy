# Loading Pipeline Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up the TurboMind model loading pipeline with typed configs, unified Parameter storage, and a single loading API.

**Architecture:** Add per-module typed config structs (C++) and dataclasses (Python) to replace raw dicts and long param lists. Unify C++ leaf tensor storage on `Parameter`. Consolidate commit functions into `LoadContext`. Decompose the god method `_load_layer`.

**Tech Stack:** C++17, Python 3.10+, pybind11, PyTorch

**Spec:** `docs/superpowers/specs/2026-04-08-loading-pipeline-cleanup-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/turbomind/core/module_config.h` | Create | C++ config struct definitions |
| `lmdeploy/turbomind/deploy/configs.py` | Create | Python config dataclasses |
| `src/turbomind/core/module.h` | Modify | Add `persist()`, keep `release()`/`to_device()` until Phase 5 |
| `src/turbomind/core/module.cc` | Modify | Implement `persist()` |
| `src/turbomind/models/linear_weight.h` | Modify | `Parameter` for tensors, `LinearConfig` constructor |
| `src/turbomind/models/linear_weight.cc` | Modify | Implement `LinearConfig` constructor, `Parameter`-based alloc |
| `src/turbomind/models/attention_weight.h` | Modify | `AttentionConfig` constructor |
| `src/turbomind/models/attention_weight.cc` | Modify | Implement `AttentionConfig` constructor, update registrar |
| `src/turbomind/models/ffn_weight.h` | Modify | `FfnConfig` constructor |
| `src/turbomind/models/ffn_weight.cc` | Modify | Implement `FfnConfig` constructor, update registrar |
| `src/turbomind/models/moe_weight.h` | Modify | `MoeConfig` constructor |
| `src/turbomind/models/moe_weight.cc` | Modify | Implement `MoeConfig` constructor, `copy_metadata_to`, update registrar |
| `src/turbomind/models/delta_net_weight.h` | Modify | `DeltaNetConfig` constructor |
| `src/turbomind/models/delta_net_weight.cc` | Modify | Implement `DeltaNetConfig` constructor, update registrar |
| `src/turbomind/models/norm_weight.h` | Modify | `Parameter` for weight tensor |
| `src/turbomind/models/norm_weight.cc` | Modify | Remove `alloc()` override |
| `src/turbomind/python/bind.cpp` | Modify | Config struct bindings, `persist()`, config-based `create_child` |
| `lmdeploy/turbomind/deploy/text_model_loader.py` | Modify | Decompose `_load_layer`, use typed configs |
| `lmdeploy/turbomind/deploy/load_context.py` | Modify | Absorb commit functions, TP rules |
| `lmdeploy/turbomind/deploy/spec.py` | Modify | `SpecAttnConfig` in `configure` |
| `lmdeploy/turbomind/deploy/commit.py` | Modify | Thin facade → eventual deletion |
| `lmdeploy/turbomind/deploy/module.py` | Modify | Update re-exports |

---

## Phase 1: C++ Typed Config Structs

### Task 1: Create C++ config header

**Files:**
- Create: `src/turbomind/core/module_config.h`

- [ ] **Step 1: Create `module_config.h` with config structs**

Each struct mirrors the current constructor parameters of the corresponding weight class. Note: `AttentionConfig` in the `core` namespace is distinct from the existing `AttentionConfig` in `config.py` (which is for TurbomindModelConfig's attention settings).

```cpp
// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/data_type.h"

namespace turbomind::core {

struct LinearConfig {
    int      input_dim{};
    int      output_dim{};
    DataType data_type{};
    bool     has_bias{};
};

struct AttentionConfig {
    int      hidden_dim{};
    int      head_dim{};
    int      head_num{};
    int      kv_head_num{};
    int      kv_lora_rank{};
    int      q_lora_rank{};
    int      qk_rope_dim{};
    int      v_head_dim{};
    bool     has_bias{};
    bool     qk_norm{};
    int      tp_size{};
    int      tp_rank{};
    DataType data_type{};
    int      window_size{-1};
    bool     attn_sink{};
    bool     attn_output_gate{};
};

struct FfnConfig {
    int      hidden_dim{};
    int      inter_size{};
    bool     has_bias{};
    int      tp_size{};
    int      tp_rank{};
    DataType data_type{};
    int      act_type{};
    bool     fuse_silu{};
    bool     fused_moe{};
};

struct MoeConfig {
    int              layer_id{};
    int              method{};
    int              experts_per_token{};
    int              inter_size{};
    bool             norm_topk_prob{};
    bool             shared_gate{};
    double           routed_scale{};
    bool             router_bias{};
    int              topk_group{};
    int              topk_method{};
    int              n_group{};
    int              scoring_func{};
    int              router_n_groups{};
    int              expert_num{};
    int              hidden_dim{};
    bool             mlp_bias{};
    DataType         data_type{};
    int              tp_size{};
    int              tp_rank{};
    int              act_type{};
    bool             fuse_silu{};
};

struct DeltaNetConfig {
    int      hidden_dim{};
    int      num_k_heads{};
    int      num_v_heads{};
    int      key_head_dim{};
    int      value_head_dim{};
    int      d_conv{4};
    bool     has_bias{};
    int      tp_size{};
    int      tp_rank{};
    DataType data_type{};
};

}  // namespace turbomind::core
```

- [ ] **Step 2: Build to verify header compiles**

Run: `cd build && ninja -j$(nproc) 2>&1 | head -20`
Expected: Build succeeds (header is not yet included anywhere, but should parse cleanly)

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/core/module_config.h
git commit -m "feat(core): add typed config structs for module constructors"
```

---

### Task 2: Add config-based constructors to weight classes

This task adds new constructors that accept the typed config structs, alongside the existing constructors. The old constructors remain until all callers are migrated.

**Files:**
- Modify: `src/turbomind/models/linear_weight.h` (line 30)
- Modify: `src/turbomind/models/linear_weight.cc` (after line 72)
- Modify: `src/turbomind/models/attention_weight.h` (line 17-30)
- Modify: `src/turbomind/models/attention_weight.cc` (after line 37)
- Modify: `src/turbomind/models/ffn_weight.h` (line 17-18)
- Modify: `src/turbomind/models/ffn_weight.cc`
- Modify: `src/turbomind/models/moe_weight.h` (line 17-25)
- Modify: `src/turbomind/models/moe_weight.cc`
- Modify: `src/turbomind/models/delta_net_weight.h` (line 18-27)
- Modify: `src/turbomind/models/delta_net_weight.cc`

For each weight class, the pattern is:

1. Add `#include "src/turbomind/core/module_config.h"` to the `.h` file
2. Add a new constructor declaration: `AttentionWeight(const core::AttentionConfig& cfg);`
3. Implement the new constructor in the `.cc` file by extracting fields from the config struct
4. Keep the old constructor (it's still used by the dict-based registry path)

**Example for AttentionWeight:**

In `attention_weight.h`, add after line 30:
```cpp
    AttentionWeight(const core::AttentionConfig& cfg);
```

In `attention_weight.cc`, add after line 37:
```cpp
AttentionWeight::AttentionWeight(const core::AttentionConfig& cfg)
    : hidden_dim_(cfg.hidden_dim)
    , head_dim_(cfg.head_dim)
    , head_num_(cfg.head_num)
    , kv_head_num_(cfg.kv_head_num)
    , mla_(MLAParam{cfg.q_lora_rank, cfg.kv_lora_rank, cfg.qk_rope_dim, cfg.v_head_dim})
    , bias_(cfg.has_bias)
    , qk_norm_(cfg.qk_norm)
    , tp_size_(cfg.tp_size)
    , tp_rank_(cfg.tp_rank)
    , data_type_(cfg.data_type)
    , window_size_(cfg.window_size)
    , sink_(cfg.attn_sink)
    , attn_output_gate_(cfg.attn_output_gate)
{
}
```

**Example for LinearWeight:**

In `linear_weight.h`, add after line 30:
```cpp
    LinearWeight(const core::LinearConfig& cfg);
```

In `linear_weight.cc`, add:
```cpp
LinearWeight::LinearWeight(const core::LinearConfig& cfg)
{
    configure(cfg.input_dim, cfg.output_dim, cfg.data_type, cfg.has_bias);
}
```

**Example for FfnWeight:**

In `ffn_weight.h`, add after line 18:
```cpp
    FfnWeight(const core::FfnConfig& cfg);
```

In `ffn_weight.cc`:
```cpp
FfnWeight::FfnWeight(const core::FfnConfig& cfg)
    : FfnWeight(cfg.hidden_dim, cfg.inter_size, cfg.has_bias, cfg.tp_size, cfg.tp_rank,
                cfg.data_type, static_cast<ActivationType>(cfg.act_type), cfg.fuse_silu)
{
}
```

**Example for MoeWeight:**

In `moe_weight.h`, add after line 25:
```cpp
    MoeWeight(const core::MoeConfig& cfg);
```

In `moe_weight.cc`:
```cpp
MoeWeight::MoeWeight(const core::MoeConfig& cfg)
{
    layer_id_ = cfg.layer_id;
    moe_param_.method = static_cast<MoeParam::Method>(cfg.method);
    moe_param_.experts_per_token = cfg.experts_per_token;
    moe_param_.inter_size = cfg.inter_size;
    moe_param_.norm_topk_prob = cfg.norm_topk_prob;
    moe_param_.shared_gate = cfg.shared_gate;
    moe_param_.routed_scale = cfg.routed_scale;
    moe_param_.router_bias = cfg.router_bias;
    moe_param_.topk_group = cfg.topk_group;
    moe_param_.topk_method = cfg.topk_method;
    moe_param_.n_group = cfg.n_group;
    moe_param_.scoring_func = cfg.scoring_func;
    moe_param_.router_n_groups = cfg.router_n_groups;
    hidden_dim_ = cfg.hidden_dim;
    mlp_bias_ = cfg.mlp_bias;
    data_type_ = cfg.data_type;
    tp_size_ = cfg.tp_size;
    tp_rank_ = cfg.tp_rank;
    act_type_ = static_cast<ActivationType>(cfg.act_type);
    fuse_silu_act_ = cfg.fuse_silu;
    expert_num_ = cfg.expert_num;
}
```

**Example for DeltaNetWeight:**

In `delta_net_weight.h`, add after line 27:
```cpp
    DeltaNetWeight(const core::DeltaNetConfig& cfg);
```

In `delta_net_weight.cc`:
```cpp
DeltaNetWeight::DeltaNetWeight(const core::DeltaNetConfig& cfg)
    : DeltaNetWeight(cfg.hidden_dim, cfg.num_k_heads, cfg.num_v_heads,
                     cfg.key_head_dim, cfg.value_head_dim, cfg.d_conv,
                     cfg.has_bias, cfg.tp_size, cfg.tp_rank, cfg.data_type)
{
}
```

- [ ] **Step 1: Add LinearConfig constructor to LinearWeight**

- [ ] **Step 2: Add AttentionConfig constructor to AttentionWeight**

- [ ] **Step 3: Add FfnConfig constructor to FfnWeight**

- [ ] **Step 4: Add MoeConfig constructor to MoeWeight**

- [ ] **Step 5: Add DeltaNetConfig constructor to DeltaNetWeight**

- [ ] **Step 6: Build to verify all constructors compile**

Run: `cd build && ninja -j$(nproc) 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 7: Commit**

```bash
git add src/turbomind/models/
git commit -m "feat(models): add typed config constructors to weight classes"
```

---

### Task 3: Add config-based factory registrations alongside existing dict-based ones

Each weight class `.cc` file has an anonymous namespace registrar at the bottom. Add a second registrar that creates modules via the typed config path. The existing dict-based registrar stays until the migration is complete.

**Files:**
- Modify: `src/turbomind/models/attention_weight.cc` (after line 119)
- Modify: `src/turbomind/models/ffn_weight.cc`
- Modify: `src/turbomind/models/moe_weight.cc`
- Modify: `src/turbomind/models/delta_net_weight.cc`
- Modify: `src/turbomind/models/linear_weight.cc` (already has registrar at end of file)

The existing dict-based registrars convert `ModuleConfig` → long param lists → old constructor. They stay unchanged. No new registrars are needed yet — the config-based `create_child` will go through the binding layer, not the registry. The registry factories are only used by the dict-based path.

**For now, no changes to the registrars.** The binding layer will handle the config → C++ struct conversion directly. We'll revisit this in Task 5 (bind.cpp).

- [ ] **Step 1: Skip this task — no registrar changes needed yet**

---

### Task 4: Expose config structs and config-based create_child in pybind11

**Files:**
- Modify: `src/turbomind/python/bind.cpp` (add bindings around line 547-620)

This adds:
1. Python bindings for each config struct
2. A new `create_child` overload that accepts a config object instead of `(name, type_name, dict)`

- [ ] **Step 1: Add config struct bindings**

Add after the existing DataType/DataFormat bindings (around line 412):

```cpp
// --- Config struct bindings ---
py::class_<turbomind::core::LinearConfig>(m, "LinearConfig")
    .def(py::init<>())
    .def_readwrite("input_dim", &turbomind::core::LinearConfig::input_dim)
    .def_readwrite("output_dim", &turbomind::core::LinearConfig::output_dim)
    .def_readwrite("data_type", &turbomind::core::LinearConfig::data_type)
    .def_readwrite("has_bias", &turbomind::core::LinearConfig::has_bias)
    .def("clone", [](const turbomind::core::LinearConfig& c) {
        return turbomind::core::LinearConfig(c);
    });

py::class_<turbomind::core::AttentionConfig>(m, "AttentionConfig")
    .def(py::init<>())
    .def_readwrite("hidden_dim", &turbomind::core::AttentionConfig::hidden_dim)
    .def_readwrite("head_dim", &turbomind::core::AttentionConfig::head_dim)
    .def_readwrite("head_num", &turbomind::core::AttentionConfig::head_num)
    .def_readwrite("kv_head_num", &turbomind::core::AttentionConfig::kv_head_num)
    .def_readwrite("kv_lora_rank", &turbomind::core::AttentionConfig::kv_lora_rank)
    .def_readwrite("q_lora_rank", &turbomind::core::AttentionConfig::q_lora_rank)
    .def_readwrite("qk_rope_dim", &turbomind::core::AttentionConfig::qk_rope_dim)
    .def_readwrite("v_head_dim", &turbomind::core::AttentionConfig::v_head_dim)
    .def_readwrite("has_bias", &turbomind::core::AttentionConfig::has_bias)
    .def_readwrite("qk_norm", &turbomind::core::AttentionConfig::qk_norm)
    .def_readwrite("tp_size", &turbomind::core::AttentionConfig::tp_size)
    .def_readwrite("tp_rank", &turbomind::core::AttentionConfig::tp_rank)
    .def_readwrite("data_type", &turbomind::core::AttentionConfig::data_type)
    .def_readwrite("window_size", &turbomind::core::AttentionConfig::window_size)
    .def_readwrite("attn_sink", &turbomind::core::AttentionConfig::attn_sink)
    .def_readwrite("attn_output_gate", &turbomind::core::AttentionConfig::attn_output_gate)
    .def("clone", [](const turbomind::core::AttentionConfig& c) {
        return turbomind::core::AttentionConfig(c);
    });
```

Similarly for `FfnConfig`, `MoeConfig`, `DeltaNetConfig`.

- [ ] **Step 2: Add config-based create_child overload**

Add a new `.def("create_child", ...)` overload before the existing dict-based one. This overload accepts a C++ config struct object and uses the new constructors:

```cpp
// Config-based create_child overload
.def("create_child",
    [with_context](ft::core::Module& m, const std::string& name,
                   const py::object& config_obj) -> ft::core::Module* {
        return with_context(m, [&]() -> ft::core::Module* {
            // Try each config type
            try {
                auto cfg = config_obj.cast<turbomind::core::AttentionConfig>();
                auto child = std::make_unique<turbomind::AttentionWeight>(cfg);
                auto* raw = child.get();
                m.add_child(name, std::move(child));
                return raw;
            } catch (py::cast_error&) {}

            try {
                auto cfg = config_obj.cast<turbomind::core::FfnConfig>();
                auto child = std::make_unique<turbomind::FfnWeight>(cfg);
                auto* raw = child.get();
                m.add_child(name, std::move(child));
                return raw;
            } catch (py::cast_error&) {}

            try {
                auto cfg = config_obj.cast<turbomind::core::MoeConfig>();
                auto child = std::make_unique<turbomind::MoeWeight>(cfg);
                auto* raw = child.get();
                m.add_child(name, std::move(child));
                return raw;
            } catch (py::cast_error&) {}

            try {
                auto cfg = config_obj.cast<turbomind::core::DeltaNetConfig>();
                auto child = std::make_unique<turbomind::DeltaNetWeight>(cfg);
                auto* raw = child.get();
                m.add_child(name, std::move(child));
                return raw;
            } catch (py::cast_error&) {}

            try {
                auto cfg = config_obj.cast<turbomind::core::LinearConfig>();
                auto child = std::make_unique<turbomind::LinearWeight>(cfg);
                auto* raw = child.get();
                m.add_child(name, std::move(child));
                return raw;
            } catch (py::cast_error&) {}

            throw std::runtime_error("Unknown config type passed to create_child");
        });
    },
    py::return_value_policy::reference,
    "name"_a, "config"_a)
```

This approach uses `py::cast` try/catch for each config type. It's simple and avoids complex type dispatch. The existing dict-based overload stays unchanged.

**Note:** This requires including the weight class headers in `bind.cpp`:
```cpp
#include "src/turbomind/models/attention_weight.h"
#include "src/turbomind/models/ffn_weight.h"
#include "src/turbomind/models/moe_weight.h"
#include "src/turbomind/models/delta_net_weight.h"
#include "src/turbomind/models/linear_weight.h"
```

- [ ] **Step 3: Build and verify**

Run: `cd build && ninja -j$(nproc) 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 4: Quick smoke test — verify dict-based path still works**

Run: `cd /data/lmdeploy-modeling && PYTHONPATH=lmdeploy:build/lib python -c "import _turbomind as tm; print('Module bindings OK'); m = tm.Module(); print('Module creation OK')"`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/python/bind.cpp
git commit -m "feat(bind): expose config structs and config-based create_child"
```

---

## Phase 2: Python Typed Configs

### Task 5: Create Python config dataclasses

**Files:**
- Create: `lmdeploy/turbomind/deploy/configs.py`

- [ ] **Step 1: Create `configs.py`**

Note: This file is `configs.py` (plural), distinct from the existing `config.py` which has `ModelConfig`, `TurbomindModelConfig`, etc.

```python
# Copyright (c) OpenMMLab. All rights reserved.
"""Typed config dataclasses for C++ module creation.

Each config maps 1:1 to a C++ config struct in core/module_config.h.
The ``k_type_name`` class attribute maps to the C++ ModuleRegistry type name.
Configs are passed to ``Module.create_child(name, config)`` via the typed
binding path in bind.cpp.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import _turbomind as _tm


@dataclass
class LinearConfig:
    input_dim: int
    output_dim: int
    data_type: int = 0
    has_bias: bool = False

    k_type_name: str = 'LinearWeight'

    def to_cpp(self) -> _tm.LinearConfig:
        cfg = _tm.LinearConfig()
        cfg.input_dim = self.input_dim
        cfg.output_dim = self.output_dim
        cfg.data_type = self.data_type
        cfg.has_bias = self.has_bias
        return cfg


@dataclass
class AttentionConfig:
    hidden_dim: int = 0
    head_dim: int = 0
    head_num: int = 0
    kv_head_num: int = 0
    kv_lora_rank: int = 0
    q_lora_rank: int = 0
    qk_rope_dim: int = 0
    v_head_dim: int = 0
    has_bias: bool = False
    qk_norm: bool = False
    tp_size: int = 1
    tp_rank: int = 0
    data_type: int = 0
    window_size: int = -1
    attn_sink: bool = False
    attn_output_gate: bool = False

    k_type_name: str = 'AttentionWeight'

    @classmethod
    def from_model_config(cls, mc, *, tp_size, tp_rank, dtype, window_size):
        return cls(
            hidden_dim=mc.hidden_units,
            head_dim=mc.size_per_head,
            head_num=mc.head_num,
            kv_head_num=mc.kv_head_num,
            kv_lora_rank=mc.kv_lora_rank or 0,
            q_lora_rank=mc.q_lora_rank or 0,
            qk_rope_dim=mc.qk_rope_dim or 0,
            v_head_dim=mc.v_head_dim or 0,
            has_bias=mc.attn_bias,
            qk_norm=mc.qk_norm,
            tp_size=tp_size,
            tp_rank=tp_rank,
            data_type=dtype,
            window_size=window_size,
            attn_sink=mc.attn_sink,
            attn_output_gate=mc.attn_output_gate,
        )

    def for_rank(self, rank: int) -> AttentionConfig:
        return replace(self, tp_rank=rank)

    def to_cpp(self) -> _tm.AttentionConfig:
        cfg = _tm.AttentionConfig()
        cfg.hidden_dim = self.hidden_dim
        cfg.head_dim = self.head_dim
        cfg.head_num = self.head_num
        cfg.kv_head_num = self.kv_head_num
        cfg.kv_lora_rank = self.kv_lora_rank
        cfg.q_lora_rank = self.q_lora_rank
        cfg.qk_rope_dim = self.qk_rope_dim
        cfg.v_head_dim = self.v_head_dim
        cfg.has_bias = self.has_bias
        cfg.qk_norm = self.qk_norm
        cfg.tp_size = self.tp_size
        cfg.tp_rank = self.tp_rank
        cfg.data_type = self.data_type
        cfg.window_size = self.window_size
        cfg.attn_sink = self.attn_sink
        cfg.attn_output_gate = self.attn_output_gate
        return cfg


@dataclass
class FfnConfig:
    hidden_dim: int = 0
    inter_size: int = 0
    has_bias: bool = False
    tp_size: int = 1
    tp_rank: int = 0
    data_type: int = 0
    act_type: int = 0
    fuse_silu: bool = False
    fused_moe: bool = False

    k_type_name: str = 'FfnWeight'

    @classmethod
    def from_model_config(cls, mc, *, tp_size, tp_rank, dtype, act_type, fuse_silu, inter_size=None, fused_moe=False):
        return cls(
            hidden_dim=mc.hidden_units,
            inter_size=inter_size or mc.inter_size,
            has_bias=mc.mlp_bias,
            tp_size=tp_size,
            tp_rank=tp_rank,
            data_type=dtype,
            act_type=act_type,
            fuse_silu=fuse_silu,
            fused_moe=fused_moe,
        )

    def for_rank(self, rank: int) -> FfnConfig:
        return replace(self, tp_rank=rank)

    def to_cpp(self) -> _tm.FfnConfig:
        cfg = _tm.FfnConfig()
        cfg.hidden_dim = self.hidden_dim
        cfg.inter_size = self.inter_size
        cfg.has_bias = self.has_bias
        cfg.tp_size = self.tp_size
        cfg.tp_rank = self.tp_rank
        cfg.data_type = self.data_type
        cfg.act_type = self.act_type
        cfg.fuse_silu = self.fuse_silu
        cfg.fused_moe = self.fused_moe
        return cfg


@dataclass
class MoeConfig:
    layer_id: int = 0
    method: int = 1  # kFused
    experts_per_token: int = 0
    inter_size: int = 0
    norm_topk_prob: bool = False
    shared_gate: bool = False
    routed_scale: float = 1.0
    router_bias: bool = False
    topk_group: int = 0
    topk_method: str = ''
    n_group: int = 0
    scoring_func: str = ''
    router_n_groups: int = 0
    expert_num: int = 0
    hidden_dim: int = 0
    mlp_bias: bool = False
    data_type: int = 0
    tp_size: int = 1
    tp_rank: int = 0
    act_type: int = 0
    fuse_silu: bool = False

    k_type_name: str = 'MoeWeight'

    @classmethod
    def from_model_config(cls, mc, *, layer_id, tp_size, tp_rank, dtype, act_type, fuse_silu, expert_num):
        return cls(
            layer_id=layer_id,
            method=1,  # kFused
            experts_per_token=mc.experts_per_token,
            inter_size=mc.expert_inter_size or 0,
            norm_topk_prob=mc.norm_topk_prob,
            shared_gate=mc.moe_shared_gate,
            routed_scale=float(mc.routed_scale),
            router_bias=getattr(mc, 'expert_router_bias', False),
            topk_group=mc.topk_group,
            topk_method=mc.topk_method,
            n_group=mc.moe_group_num,
            scoring_func=mc.scoring_func,
            router_n_groups=max(0, getattr(mc, 'router_n_groups', -1)),
            expert_num=expert_num,
            hidden_dim=mc.hidden_units,
            mlp_bias=mc.mlp_bias,
            data_type=dtype,
            tp_size=tp_size,
            tp_rank=tp_rank,
            act_type=act_type,
            fuse_silu=fuse_silu,
        )

    def for_rank(self, rank: int) -> MoeConfig:
        return replace(self, tp_rank=rank)

    def to_cpp(self) -> _tm.MoeConfig:
        cfg = _tm.MoeConfig()
        cfg.layer_id = self.layer_id
        cfg.method = self.method
        cfg.experts_per_token = self.experts_per_token
        cfg.inter_size = self.inter_size
        cfg.norm_topk_prob = self.norm_topk_prob
        cfg.shared_gate = self.shared_gate
        cfg.routed_scale = self.routed_scale
        cfg.router_bias = self.router_bias
        cfg.topk_group = self.topk_group
        cfg.topk_method = self.topk_method
        cfg.n_group = self.n_group
        cfg.scoring_func = self.scoring_func
        cfg.router_n_groups = self.router_n_groups
        cfg.expert_num = self.expert_num
        cfg.hidden_dim = self.hidden_dim
        cfg.mlp_bias = self.mlp_bias
        cfg.data_type = self.data_type
        cfg.tp_size = self.tp_size
        cfg.tp_rank = self.tp_rank
        cfg.act_type = self.act_type
        cfg.fuse_silu = self.fuse_silu
        return cfg


@dataclass
class DeltaNetConfig:
    hidden_dim: int = 0
    num_k_heads: int = 0
    num_v_heads: int = 0
    key_head_dim: int = 0
    value_head_dim: int = 0
    d_conv: int = 4
    has_bias: bool = False
    tp_size: int = 1
    tp_rank: int = 0
    data_type: int = 0

    k_type_name: str = 'DeltaNetWeight'

    @classmethod
    def from_model_config(cls, mc, *, tp_size, tp_rank, dtype):
        return cls(
            hidden_dim=mc.hidden_units,
            num_k_heads=mc.linear_num_key_heads,
            num_v_heads=mc.linear_num_value_heads,
            key_head_dim=mc.linear_key_head_dim,
            value_head_dim=mc.linear_value_head_dim,
            d_conv=mc.linear_conv_kernel_dim or 4,
            has_bias=bool(mc.attn_bias),
            tp_size=tp_size,
            tp_rank=tp_rank,
            data_type=dtype,
        )

    def for_rank(self, rank: int) -> DeltaNetConfig:
        return replace(self, tp_rank=rank)

    def to_cpp(self) -> _tm.DeltaNetConfig:
        cfg = _tm.DeltaNetConfig()
        cfg.hidden_dim = self.hidden_dim
        cfg.num_k_heads = self.num_k_heads
        cfg.num_v_heads = self.num_v_heads
        cfg.key_head_dim = self.key_head_dim
        cfg.value_head_dim = self.value_head_dim
        cfg.d_conv = self.d_conv
        cfg.has_bias = self.has_bias
        cfg.tp_size = self.tp_size
        cfg.tp_rank = self.tp_rank
        cfg.data_type = self.data_type
        return cfg


@dataclass
class SpecAttnConfig:
    """Config for TextModelSpec.configure — carries spec-specific TP params."""
    tp: int = 1
    permute_qk: bool = True
    repeat_kv: int = 0
    head_dim: int = 0
    rope_dim: int = 0
    output_gate: bool = False
    kv_head_num: int = 0
```

- [ ] **Step 2: Verify configs.py imports correctly**

Run: `cd /data/lmdeploy-modeling && PYTHONPATH=lmdeploy:build/lib python -c "from lmdeploy.turbomind.deploy.configs import AttentionConfig; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/configs.py
git commit -m "feat(deploy): add Python config dataclasses for typed module creation"
```

---

### Task 6: Migrate text_model_loader.py to use typed configs

Replace the inline dict configs in `_load_layer` and `_load_global` with typed config objects. The `create_child` calls change from `create_child(name, type_name, dict)` to `create_child(name, cfg.to_cpp())`.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/text_model_loader.py`

The key change in `_load_layer`:

```python
# Before (line ~115):
attn_mod = handle.create_child('attention', 'AttentionWeight', {
    'hidden_dim': hidden,
    'head_dim': mc.size_per_head,
    ...16 more lines...
})

# After:
attn_cfg = self._attn_config.for_rank(attn_rank)
attn_mod = handle.create_child('attention', attn_cfg.to_cpp())
```

Update `__init__` to build configs once:

```python
def __init__(self, model: BaseOutputModel):
    self.model = model
    self.attn_tp = model.attn_tp_size
    self.mlp_tp = model.mlp_tp_size

    mc = model.model_config
    # Build configs lazily — some fields (window_size, inter_size) are per-layer
    self._base_attn_config = AttentionConfig.from_model_config(
        mc, tp_size=self.attn_tp, tp_rank=0,
        dtype=0, window_size=-1)  # rank/window filled per-layer
    self._base_ffn_config = FfnConfig.from_model_config(
        mc, tp_size=self.mlp_tp, tp_rank=0,
        dtype=0, act_type=0, fuse_silu=True)
```

Then in `_load_layer`, each `create_child` call uses:

```python
# Attention
attn_cfg = replace(self._base_attn_config, tp_rank=attn_rank,
                   data_type=dtype, window_size=window_size)
attn_mod = handle.create_child('attention', attn_cfg.to_cpp())

# FFN
ffn_cfg = replace(self._base_ffn_config, tp_rank=mlp_rank,
                  data_type=dtype, inter_size=inter_size,
                  act_type=_act_type_id(mc.activation_type))
ffn_mod = handle.create_child('feed_forward', ffn_cfg.to_cpp())

# MoE
moe_cfg = MoeConfig.from_model_config(mc, layer_id=layer, tp_size=self.mlp_tp,
                                       tp_rank=mlp_rank, dtype=dtype,
                                       act_type=_act_type_id(mc.activation_type),
                                       fuse_silu=True, expert_num=expert_num)
moe_mod = handle.create_child('moe_ffn', moe_cfg.to_cpp())

# DeltaNet
dn_cfg = DeltaNetConfig.from_model_config(mc, tp_size=self.attn_tp,
                                           tp_rank=attn_rank, dtype=dtype)
linear_attn_mod = handle.create_child('linear_attn', dn_cfg.to_cpp())
```

Similarly update `_load_global` for `tok_embeddings`, `norm`, and `output`.

- [ ] **Step 1: Update `__init__` to build base configs**

- [ ] **Step 2: Update `_load_layer` attention section to use `AttentionConfig`**

- [ ] **Step 3: Update `_load_layer` FFN section to use `FfnConfig`**

- [ ] **Step 4: Update `_load_layer` MoE section to use `MoeConfig`**

- [ ] **Step 5: Update `_load_layer` DeltaNet section to use `DeltaNetConfig`**

- [ ] **Step 6: Update `_load_global` to use `LinearConfig`**

- [ ] **Step 7: Build and test a model load**

Run a model test using the turbomind-tester agent (or `scripts/test_turbomind_model.py` directly) to verify a model loads and produces correct output with 128+ tokens. Test at least one dense model.

- [ ] **Step 8: Commit**

```bash
git add lmdeploy/turbomind/deploy/text_model_loader.py
git commit -m "refactor(loader): migrate create_child calls to typed configs"
```

---

## Phase 3: C++ Parameter Unification

### Task 7: Convert LinearWeight tensors to Parameter wrappers

This is the biggest C++ change. Convert the direct `Tensor` members (`weight`, `bias`, `scales`, `zeros`) to `Parameter` wrappers, and update all code that accesses them.

**Files:**
- Modify: `src/turbomind/models/linear_weight.h` (lines 50-53)
- Modify: `src/turbomind/models/linear_weight.cc` (lines 74-174, do_allocate and alloc)

**Key consideration:** `weight`, `bias`, `scales`, `zeros` are public members accessed directly by execution layers (LlamaLinear, etc.) and by `LinkLinearExperts`. We need to keep public accessors that return `Tensor&` or `Tensor*`.

**In `linear_weight.h`, replace lines 49-53:**

```cpp
// Before:
// Public data fields consumed by execution layers (LlamaLinear, etc.)
Tensor weight;
Tensor bias;
Tensor scales;
Tensor zeros;

// After:
// Public accessors for execution layers
Tensor& weight() { return *weight_; }
Tensor& bias() { return *bias_; }
Tensor& scales() { return *scales_; }
Tensor& zeros() { return *zeros_; }
const Tensor& weight() const { return *weight_; }
```

**Add private Parameter members:**

```cpp
private:
    mutable core::Parameter weight_{*this, "weight"};
    mutable core::Parameter bias_{*this, "bias"};
    mutable core::Parameter scales_{*this, "scales"};
    mutable core::Parameter zeros_{*this, "zeros"};
```

**Update `do_allocate()` in `linear_weight.cc`:**

Replace direct Tensor assignment with Parameter assignment:
```cpp
// Before:
weight = Tensor({input_dim, output_dim}, actual_weight_type, kDEVICE);
add_param("weight", weight);

// After:
*weight_ = Tensor({input_dim, output_dim}, actual_weight_type, kDEVICE);
// No add_param needed — Parameter auto-registers
```

Similarly for bias, scales, zeros.

**Update `alloc()` override:**

```cpp
// Before (returns direct member):
if (param_name == "weight" || param_name == "qweight") return weight;

// After (returns Parameter's tensor):
if (param_name == "weight" || param_name == "qweight") return *weight_;
```

**Update all callers** that access `weight`, `bias`, `scales`, `zeros` as fields:
- `moe_weight.cc` `LinkLinearExperts` — change `e0.weight` to `e0.weight()`, etc.
- `linear_weight.cc` `prepare()` — change `weight` to `weight()`, etc.
- Any execution layer code that accesses these directly

**IMPORTANT:** This is a high-risk change. Must test thoroughly after.

- [ ] **Step 1: Update linear_weight.h — replace Tensor members with Parameter + accessors**

- [ ] **Step 2: Update linear_weight.cc do_allocate() to assign through Parameter**

- [ ] **Step 3: Update linear_weight.cc alloc() to return through Parameter**

- [ ] **Step 4: Update linear_weight.cc prepare() to use accessors**

- [ ] **Step 5: Update moe_weight.cc LinkLinearExperts to use accessors**

- [ ] **Step 6: Update any other callers of LinearWeight::weight/bias/scales/zeros**

Search with: `grep -rn '\.weight\b' src/turbomind/ --include='*.cc' --include='*.h' | grep -i linear`

- [ ] **Step 7: Build**

Run: `cd build && ninja -j$(nproc) 2>&1 | tail -10`
Expected: Build succeeds

- [ ] **Step 8: Test a model load (dense BF16/FP16)**

Use turbomind-tester agent or test script. Verify 128+ token coherent output.

- [ ] **Step 9: Test a quantized model (AWQ or GPTQ)**

Verify quantized models still load correctly.

- [ ] **Step 10: Commit**

```bash
git add src/turbomind/models/
git commit -m "refactor(linear): convert tensor members to Parameter wrappers"
```

---

### Task 8: Convert remaining weight classes to use Parameter consistently

**Files:**
- Modify: `src/turbomind/models/norm_weight.h` (already uses internal `Tensor weight_`)
- Modify: `src/turbomind/models/norm_weight.cc`
- Modify: `src/turbomind/models/moe_weight.h` (line 56: `Tensor score_correction_bias_`)
- Modify: `src/turbomind/models/moe_weight.cc`

**NormWeight:** Currently stores `Tensor weight_` directly and registers it in `alloc()`. Convert to `Parameter`:

In `norm_weight.h`, add `#include "src/turbomind/core/module.h"` (already included via linear_weight.h chain), replace `Tensor weight_` with `mutable core::Parameter weight_{*this, "weight"};`.

Remove the `alloc()` override — the base class `Module::alloc()` will handle it since the Parameter is pre-registered.

**MoeWeight:** Convert `Tensor score_correction_bias_` to `Parameter score_correction_bias_{*this, "score_correction_bias"}`. Update `alloc()` and `score_correction_bias()` accessor.

- [ ] **Step 1: Convert NormWeight**

- [ ] **Step 2: Convert MoeWeight score_correction_bias_**

- [ ] **Step 3: Build and test**

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/
git commit -m "refactor(models): unify remaining weight classes on Parameter"
```

---

## Phase 4: Python Loading Flow

### Task 9: Move commit functions and TP rules into LoadContext

Absorb `commit_linear`, `commit_tensor`, `_commit_tensors`, and TP rules from `commit.py` into `LoadContext`. `commit.py` becomes a thin re-export facade.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/load_context.py`
- Modify: `lmdeploy/turbomind/deploy/commit.py`
- Modify: `lmdeploy/turbomind/deploy/module.py`

**In `load_context.py`:**
1. Move `_commit_tensors`, `_torch_dtype_to_cpp`, `_cast_shard_for_tm`, `_infer_cpp_linear_dtype`, `_infer_compute_dtype` from `commit.py` as private methods
2. Move `_ATTN_TP_RULES`, `_FFN_TP_RULES`, `_LINEAR_ATTN_TP_RULES`, `_SPLIT_SIDE_TO_DIM` as class constants
3. Update `load_linear` and `load_tensor` to use the internal methods instead of importing from `.module`
4. Add `commit_linear` and `commit_tensor` as wrapper methods that delegate to internal logic (for backward compat)

**In `commit.py`:**
Replace implementations with imports from `load_context.py`:
```python
from .load_context import LoadContext
# Re-export for backward compatibility
commit_linear = LoadContext.commit_linear
commit_tensor = LoadContext.commit_tensor
# etc.
```

**In `module.py`:**
Update imports to come from `load_context` instead of `commit`.

- [ ] **Step 1: Copy TP rules and helpers into LoadContext**

- [ ] **Step 2: Update load_linear and load_tensor to use internal helpers**

- [ ] **Step 3: Add commit_linear/commit_tensor as static methods for backward compat**

- [ ] **Step 4: Update commit.py to re-export from LoadContext**

- [ ] **Step 5: Update module.py facade imports**

- [ ] **Step 6: Build and test a model load**

- [ ] **Step 7: Commit**

```bash
git add lmdeploy/turbomind/deploy/
git commit -m "refactor(deploy): absorb commit functions into LoadContext"
```

---

### Task 10: Decompose _load_layer into per-component methods

**Files:**
- Modify: `lmdeploy/turbomind/deploy/text_model_loader.py`

Split the 274-line `_load_layer` into focused methods:

```python
def _load_layer(self, layer, spec):
    for gpu in range(self.model.gpu_count):
        self._load_layer_on_gpu(layer, spec, gpu)

def _load_layer_on_gpu(self, layer, spec, gpu):
    root = self.model.root(gpu)
    if root is None:
        return
    attn_rank, mlp_rank = self.model.tp_ranks(gpu)

    # Setup
    layers = root.child('layers') or root.create_child('layers', 'ModuleList', {})
    layer_name = str(layer)
    layer_mod = layers.child(layer_name) or layers.create_child(layer_name, 'DecoderLayerWeight', {})

    tp_config = self._make_tp_config(attn_rank)
    ctx = LoadContext(layer_mod, tp_config, self.model.model_config)
    spec.configure(SpecAttnConfig(
        tp=ctx.tp_size,
        permute_qk=ctx._tp_config.get('permute_qk', True),
        repeat_kv=ctx.repeat_kv,
        head_dim=ctx.head_dim,
        rope_dim=ctx.rope_dim,
        output_gate=ctx.attn_output_gate,
        kv_head_num=ctx.kv_head_num,
    ))

    self._load_norms(ctx, spec, layer)
    self._load_attention(ctx, spec, layer, attn_rank)
    self._load_ffn_or_moe(ctx, spec, layer, mlp_rank)
    self._load_linear_attn(ctx, spec, layer, attn_rank)
    self._load_raw_tensors(ctx, spec, layer, attn_rank)
```

Each sub-method handles one component. Extract the corresponding code block from the current `_load_layer`.

- [ ] **Step 1: Extract `_load_layer_on_gpu`**

- [ ] **Step 2: Extract `_load_norms`**

- [ ] **Step 3: Extract `_load_attention`**

- [ ] **Step 4: Extract `_load_ffn_or_moe`**

- [ ] **Step 5: Extract `_load_moe`**

- [ ] **Step 6: Extract `_load_linear_attn`**

- [ ] **Step 7: Extract `_load_raw_tensors`**

- [ ] **Step 8: Build and test a model load**

- [ ] **Step 9: Commit**

```bash
git add lmdeploy/turbomind/deploy/text_model_loader.py
git commit -m "refactor(loader): decompose _load_layer into per-component methods"
```

---

### Task 11: Simplify TextModelSpec.configure with SpecAttnConfig

**Files:**
- Modify: `lmdeploy/turbomind/deploy/spec.py` (lines 130-151)
- Modify: all source model specs that call `spec.configure`

Replace the 7-parameter `configure` with a single `SpecAttnConfig` object.

```python
# Before (spec.py line 130):
def configure(self, attn_tp=1, permute_qk=True, repeat_kv=0,
              head_dim=0, rope_dim=0, attn_output_gate=False, kv_head_num=0):
    self._attn_tp = attn_tp
    self._permute_qk = permute_qk
    ...

# After:
def configure(self, cfg: SpecAttnConfig):
    self._attn_tp = cfg.tp
    self._permute_qk = cfg.permute_qk
    self._repeat_kv = cfg.repeat_kv
    self._head_dim = cfg.head_dim
    self._rope_dim = cfg.rope_dim if cfg.rope_dim else cfg.head_dim
    self._attn_output_gate = cfg.output_gate
    self._kv_head_num = cfg.kv_head_num
```

Update callers — search for all places that call `spec.configure(...)`:

```bash
grep -rn '\.configure(' lmdeploy/turbomind/deploy/ --include='*.py'
```

Only `text_model_loader.py` calls it (already updated in Task 10 to use `SpecAttnConfig`).

- [ ] **Step 1: Update configure signature in spec.py**

- [ ] **Step 2: Verify text_model_loader.py already passes SpecAttnConfig**

- [ ] **Step 3: Test a model load**

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/spec.py
git commit -m "refactor(spec): simplify configure with SpecAttnConfig"
```

---

## Phase 5: C++ Lifecycle Cleanup

### Task 12: Add persist() method to Module

**Files:**
- Modify: `src/turbomind/core/module.h` (add after line 93)
- Modify: `src/turbomind/core/module.cc` (add after line 139)
- Modify: `src/turbomind/python/bind.cpp` (add binding)

Add `persist(PersistOp)` alongside the existing `release()` and `to_device()`. Both paths coexist until callers are migrated.

**In `module.h`, add:**

```cpp
enum class PersistOp { Sleep, WakeUp };

class Module {
    // ... existing methods ...

    /// Move tensors between CPU/GPU for power management.
    /// Sleep: move to CPU (preserve data, free GPU memory).
    /// WakeUp: move back to GPU.
    virtual void persist(PersistOp op);
};
```

**In `module.cc`, add:**

```cpp
void Module::persist(PersistOp op)
{
    for (auto& [name, child] : children_) {
        child->persist(op);
    }
    for (auto& [name, ptr] : params_) {
        if (!ptr || !*ptr) continue;

        if (op == PersistOp::Sleep) {
            if (ptr->device().type == kDEVICE) {
                Tensor cpu{ptr->shape(), ptr->dtype(), Device{kCPU, ptr->device().id}};
                Copy(*ptr, cpu);
                *ptr = std::move(cpu);
            }
        } else {  // WakeUp
            if (ptr->device().type == kCPU) {
                Tensor gpu{ptr->shape(), ptr->dtype(), Device{kDEVICE, ptr->device().id}};
                Copy(*ptr, gpu);
                *ptr = std::move(gpu);
            }
        }
    }
}
```

**In `bind.cpp`, add:**

```cpp
py::enum_<ft::core::PersistOp>(m, "PersistOp")
    .value("Sleep", ft::core::PersistOp::Sleep)
    .value("WakeUp", ft::core::PersistOp::WakeUp);

// Add to Module class binding:
.def("persist", &ft::core::Module::persist, "op"_a,
     py::call_guard<py::gil_scoped_release>())
```

- [ ] **Step 1: Add PersistOp enum and persist() declaration to module.h**

- [ ] **Step 2: Implement persist() in module.cc**

- [ ] **Step 3: Add persist binding in bind.cpp**

- [ ] **Step 4: Build**

- [ ] **Step 5: Verify existing release/to_device still work**

- [ ] **Step 6: Commit**

```bash
git add src/turbomind/core/ src/turbomind/python/bind.cpp
git commit -m "feat(core): add persist() for Sleep/WakeUp power management"
```

---

### Task 13: Reduce MoE boilerplate with copy_metadata_to

**Files:**
- Modify: `src/turbomind/models/linear_weight.h` (add method)
- Modify: `src/turbomind/models/linear_weight.cc` (implement)
- Modify: `src/turbomind/models/moe_weight.cc` (use in LinkLinearExperts)

**In `linear_weight.h`, add after line 45:**

```cpp
    /// Copy metadata fields to another LinearWeight (for MoE block view).
    void copy_metadata_to(LinearWeight& dst) const;
```

**In `linear_weight.cc`:**

```cpp
void LinearWeight::copy_metadata_to(LinearWeight& dst) const
{
    dst.input_dim    = input_dim;
    dst.output_dim   = output_dim;
    dst.group_size   = group_size;
    dst.data_type    = data_type;
    dst.weight_format = weight_format;
    dst.format_      = format_;
    dst.policy_      = policy_;
    dst.epilogue     = epilogue;
    dst.has_bias_    = has_bias_;
    dst.is_grouped_  = is_grouped_;
}
```

**In `moe_weight.cc` `LinkLinearExperts`, replace lines 49-58 (manual field copying) with:**

```cpp
e0.copy_metadata_to(d);
```

Then the rest of `LinkLinearExperts` (bias allocation, pointer assembly, weight/scale tensor creation) stays the same but uses `d.weight()` / `d.scales()` accessor syntax (already updated in Task 7).

- [ ] **Step 1: Add copy_metadata_to to LinearWeight**

- [ ] **Step 2: Update LinkLinearExperts to use copy_metadata_to**

- [ ] **Step 3: Build and test a MoE model load**

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/
git commit -m "refactor(moe): reduce boilerplate with copy_metadata_to"
```

---

## Phase 6: Integration Testing

### Task 14: Full integration test across model types

After all phases are complete, run comprehensive tests:

- [ ] **Step 1: Test dense BF16 model (TP=1)**

Use turbomind-tester agent. Verify 128+ token coherent output.

- [ ] **Step 2: Test dense FP16 model (TP=1)**

- [ ] **Step 3: Test AWQ quantized model (TP=1)**

- [ ] **Step 4: Test GPTQ quantized model (TP=1)**

- [ ] **Step 5: Test FP8 model (TP=1)**

- [ ] **Step 6: Test dense model (TP=2) if multi-GPU available**

- [ ] **Step 7: Test MoE model (if available)**

- [ ] **Step 8: Fix any regressions found**

- [ ] **Step 9: Final commit**

```bash
git add -A
git commit -m "test: verify loading pipeline cleanup across model types"
```

---

## Self-Review

**1. Spec coverage check:**
- Per-module typed configs (Python + C++) → Tasks 1-6 ✓
- Keep Submodule<T> → No changes ✓
- Unify on Parameter → Tasks 7-8 ✓
- Single loading API (LoadContext absorbs commit) → Task 9 ✓
- Decompose god method → Task 10 ✓
- Keep fusion tuple → No changes ✓
- Keep prepare() → No changes ✓
- persist() lifecycle → Task 12 ✓
- MoE boilerplate → Task 13 ✓
- SpecAttnConfig → Task 11 ✓

**2. Placeholder scan:** No TBDs, TODOs, or vague steps found.

**3. Type consistency:**
- `AttentionConfig.for_rank()` returns `AttentionConfig` (via `dataclasses.replace`)
- `to_cpp()` returns `_tm.AttentionConfig` (the pybind11-bound C++ struct)
- `create_child('attention', attn_cfg.to_cpp())` passes the C++ struct to the binding overload
- `SpecAttnConfig` is separate from `AttentionConfig` — consistent with spec
- `copy_metadata_to` accesses fields via accessors (after Parameter conversion) — consistent with Task 7
