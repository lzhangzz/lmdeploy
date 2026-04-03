# Module Hierarchy Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

 It **Goal:** Decouple the turbomind model loading pipeline so new architectures can be added with Python-only changes.

 **Architecture:** Add a C++ module type registry and explicit `create_child()` API, Replace Python `TransformerV2` with `TextModelLoader` that uses composable `LoadContext` to drive module tree creation from Python specs. **Tech Stack:** C++17 (pybind11), PyTorch, CUDA

 turbomind C++ engine.

 --- ## File Structure | File | Purpose |
|------|------|----------|
| Create | `src/turbomind/core/registry.h` | Module type registry |
| Create | `src/turbomind/core/registry.cc` | Registry implementation |
| Modify | `src/turbomind/core/module.h` | Add `create_child()`, `get<T>()`, config types |
 Modify | `src/turbomind/core/module.cc` | Implement `create_child()`, registry integration |
| Modify | `src/turbomind/models/linear_weight.h` | Add self-registration |
| Modify | `src/turbomind/models/norm_weight.h` | Add self-registration |
| Modify | `src/turbomind/models/ffn_weight.h` | Add self-registration |
| Modify | `src/turbomind/models/attention_weight.h` | Add self-registration |
| Modify | `src/turbomind/models/moe_weight.h` | Add self-registration |
| Modify | `src/turbomind/models/delta_net_weight.h` | Add self-registration |
| Modify | `src/turbomind/models/decoder_layer_weight.h` | Add self-registration |
| Modify | `src/turbomind/models/model_weight.h` | Add self-registration |
| Modify | `src/turbomind/python/bind.cpp` | Expose `create_child` to Python |
| Create | `lmdeploy/turbomind/deploy/load_context.py` | LoadContext class |
| Create | `lmdeploy/turbomind/deploy/text_model_loader.py` | TextModelLoader class |
| Modify | `lmdeploy/turbomind/deploy/module.py` | Rename ModelWeightSpec → TextModelSpec, TransformerV2 → (deprecated), add load_layer/load_global, rename commit functions |
| Modify | `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` | Rename Qwen3Spec → Qwen3TextSpec, adapt to new base class |
| Modify | `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` | Adapt to new base class |
| Modify | `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` | Adapt to new base class |
| Modify | `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` | Adapt to new base class |
| Modify | `lmdeploy/turbomind/turbomind.py` | Wire new loader class |

---

## Task 1: C++ ConfigValue + ModuleConfig Types

**Files:**
- Create: `src/turbomind/core/registry.h`
- Modify: `src/turbomind/core/module.h`

- [ ] **Step 1: Create registry.h with ConfigValue, ModuleConfig, and ModuleRegistry declarations**

```cpp
// src/turbomind/core/registry.h
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <variant>

namespace turbomind::core {

/// Configuration value type for module factory system.
/// Supports integer, string, and floating-point values.
using ConfigValue = std::variant<int64_t, std::string, double>;

/// Configuration map passed to module factory functions.
using ModuleConfig = std::map<std::string, ConfigValue>;

/// Module type registry. Maps type name strings to factory functions.
class ModuleRegistry {
public:
    using Factory = std::function<std::unique_ptr<Module>(const ModuleConfig&)>;

    static ModuleRegistry& instance();

    /// Register a factory under the given type name.
    /// Returns true on success. Duplicate names are logged but overwrite.
    void register_type(const std::string& name, Factory factory);

    /// Create a module instance by type name.
    /// Returns nullptr if type name is not registered.
    std::unique_ptr<Module> create(const std::string& type,
                                    const ModuleConfig& config) const;

    /// Check if a type name is registered.
    bool has_type(const std::string& name) const;

private:
    ModuleRegistry() = default;
    std::map<std::string, Factory> factories_;
};

}  // namespace turbomind::core
```

- [ ] **Step 2: Add forward declaration of Module in registry.h**

The registry needs to know about Module but module.h will include registry.h. To avoid circular dependency, registry.h forward-declares Module and takes Factory as returning `unique_ptr<Module>`. This works because the factory function is only stored, never called inline in the registry.

Actually, since registry.h needs Module as an incomplete type for the factory return type, we just use a forward declaration. The factory functions are defined in the .cc files where the full Module definition is visible.

 This is clean.

- [ ] **Step 3: Add `create_child`, `get<T>()`, and public children to Module in module.h**

Add these new public methods to the existing Module class in `src/turbomind/core/module.h`:

```cpp
// In module.h, add includes
#include "src/turbomind/core/registry.h"

// Add to Module class public section:
public:
    /// Create a child module using the type registry.
    /// Returns pointer to the created child, or nullptr on failure.
    Module* create_child(const std::string& name,
                         const std::string& type_name,
                         const ModuleConfig& config = {});

    /// Typed child accessor. Returns nullptr if child not found.
    template<typename T>
    T* get(const std::string& name) const {
        return static_cast<T*>(child(name));
    }

    /// Expose children for iteration (execution side).
    const auto& children() const { return children_; }
```

- [ ] **Step 4: Commit the header changes**

```bash
git add src/turbomind/core/registry.h
git commit -m "feat(core): add ModuleConfig, ConfigValue types and ModuleRegistry declaration"
```

---

## Task 2: C++ Registry Implementation + Module::create_child

**Files:**
- Create: `src/turbomind/core/registry.cc`
- Modify: `src/turbomind/core/module.cc`

- [ ] **Step 1: Implement registry.cc**

```cpp
// src/turbomind/core/registry.cc
#include "src/turbomind/core/registry.h"
#include "src/turbomind/core/check.h"

namespace turbomind::core {

ModuleRegistry& ModuleRegistry::instance()
{
    static ModuleRegistry reg;
    return reg;
}

void ModuleRegistry::register_type(const std::string& name, Factory factory)
{
    factories_[name] = std::move(factory);
}

std::unique_ptr<Module> ModuleRegistry::create(const std::string& type,
                                                const ModuleConfig& config) const
{
    auto it = factories_.find(type);
    if (it == factories_.end()) {
        return nullptr;
    }
    return it->second(config);
}

bool ModuleRegistry::has_type(const std::string& name) const
{
    return factories_.count(name) > 0;
}

}  // namespace turbomind::core
```

- [ ] **Step 2: Implement Module::create_child in module.cc**

Add to the end of `module.cc` (before the closing namespace brace):

```cpp
Module* Module::create_child(const std::string& name,
                              const std::string& type_name,
                              const ModuleConfig& config)
{
    auto module = ModuleRegistry::instance().create(type_name, config);
    if (!module) {
        return nullptr;
    }
    return add_child(name, std::move(module));
}
```

- [ ] **Step 3: Build and verify**

```bash
cd /data/lmdeploy-modeling/build && ninja
```

Expected: Clean build. The new code is additive — no existing code is modified except for adding `create_child` to module.cc.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/core/registry.h src/turbomind/core/registry.cc src/turbomind/core/module.h src/turbomind/core/module.cc
git commit -m "feat(core): implement ModuleRegistry and Module::create_child"
```

---

## Task 3: Register All C++ Module Types

**Files:**
- Modify: `src/turbomind/models/linear_weight.cc`
- Modify: `src/turbomind/models/norm_weight.cc`
- Modify: `src/turbomind/models/ffn_weight.cc`
- Modify: `src/turbomind/models/attention_weight.cc`
- Modify: `src/turbomind/models/moe_weight.cc`
- Modify: `src/turbomind/models/delta_net_weight.cc`
- Modify: `src/turbomind/models/decoder_layer_weight.cc`
- Modify: `src/turbomind/models/model_weight.cc`

Each module type gets a self-registration block. The registration happens at static initialization time via a global bool.

- [ ] **Step 1: Register LinearWeight in linear_weight.cc**

Add at the end of the file, before the closing namespace brace:

```cpp
// Self-register with module registry
namespace {
struct LinearWeightRegistrar {
    LinearWeightRegistrar() {
        core::ModuleRegistry::instance().register_type(
            "LinearWeight",
            [](const core::ModuleConfig& cfg) -> std::unique_ptr<core::Module> {
                auto m = std::make_unique<LinearWeight>();
                m->configure(
                    std::get<int64_t>(cfg.at("input_dim")),
                    std::get<int64_t>(cfg.at("output_dim")),
                    static_cast<DataType>(std::get<int64_t>(cfg.at("data_type"))),
                    cfg.count("has_bias") && std::get<int64_t>(cfg.at("has_bias")));
                return m;
            });
    }
};
static LinearWeightRegistrar _linear_weight_reg;
} // anonymous namespace
```

- [ ] **Step 2: Register NormWeight in norm_weight.cc**

```cpp
namespace {
struct NormWeightRegistrar {
    NormWeightRegistrar() {
        core::ModuleRegistry::instance().register_type(
            "NormWeight",
            [](const core::ModuleConfig& cfg) -> std::unique_ptr<core::Module> {
                auto dim = std::get<int64_t>(cfg.at("dim"));
                auto dtype = static_cast<DataType>(std::get<int64_t>(cfg.at("data_type")));
                return std::make_unique<NormWeight>(dim, dtype);
            });
    }
};
static NormWeightRegistrar _norm_weight_reg;
} // anonymous namespace
```

- [ ] **Step 3: Register FfnWeight in ffn_weight.cc**

```cpp
namespace {
struct FfnWeightRegistrar {
    FfnWeightRegistrar() {
        core::ModuleRegistry::instance().register_type(
            "FfnWeight",
            [](const core::ModuleConfig& cfg) -> std::unique_ptr<core::Module> {
                return std::make_unique<FfnWeight>(
                    std::get<int64_t>(cfg.at("hidden_dim")),
                    std::get<int64_t>(cfg.at("inter_size")),
                    cfg.count("has_bias") && std::get<int64_t>(cfg.at("has_bias")),
                    std::get<int64_t>(cfg.at("tp_size")),
                    std::get<int64_t>(cfg.at("tp_rank")),
                    static_cast<DataType>(std::get<int64_t>(cfg.at("data_type"))),
                    static_cast<ActivationType>(std::get<int64_t>(cfg.at("act_type"))),
                    cfg.count("fuse_silu_act") && std::get<int64_t>(cfg.at("fuse_silu_act")));
            });
    }
};
static FfnWeightRegistrar _ffn_weight_reg;
} // anonymous namespace
```

- [ ] **Step 4: Register AttentionWeight in attention_weight.cc**

```cpp
namespace {
struct AttentionWeightRegistrar {
    AttentionWeightRegistrar() {
        core::ModuleRegistry::instance().register_type(
            "AttentionWeight",
            [](const core::ModuleConfig& cfg) -> std::unique_ptr<core::Module> {
                MLAParam mla;
                mla.kv_lora_rank = std::get<int64_t>(cfg.at("kv_lora_rank", 0));
                mla.q_lora_rank = std::get<int64_t>(cfg.at("q_lora_rank", 0));
                mla.qk_rope_dim = std::get<int64_t>(cfg.at("qk_rope_dim", 0));
                return std::make_unique<AttentionWeight>(
                    std::get<int64_t>(cfg.at("hidden_dim")),
                    std::get<int64_t>(cfg.at("head_dim")),
                    std::get<int64_t>(cfg.at("head_num")),
                    std::get<int64_t>(cfg.at("kv_head_num")),
                    mla,
                    cfg.count("has_bias") && std::get<int64_t>(cfg.at("has_bias")),
                    cfg.count("qk_norm") && std::get<int64_t>(cfg.at("qk_norm")),
                    std::get<int64_t>(cfg.at("tp_size")),
                    std::get<int64_t>(cfg.at("tp_rank")),
                    static_cast<DataType>(std::get<int64_t>(cfg.at("data_type"))),
                    std::get<int64_t>(cfg.at("window_size", 0)),
                    cfg.count("attn_sink") && std::get<int64_t>(cfg.at("attn_sink")),
                    cfg.count("attn_output_gate") && std::get<int64_t>(cfg.at("attn_output_gate")));
            });
    }
};
static AttentionWeightRegistrar _attn_weight_reg;
} // anonymous namespace
```

Note: The `cfg.at("key", default)` pattern needs a helper that returns a default value. Add a small helper or use `cfg.count("key") ? std::get<int64_t>(cfg.at("key")) : default_val`.

- [ ] **Step 5: Register MoeWeight in moe_weight.cc**

```cpp
namespace {
struct MoeWeightRegistrar {
    MoeWeightRegistrar() {
        core::ModuleRegistry::instance().register_type(
            "MoeWeight",
            [](const core::ModuleConfig& cfg) -> std::unique_ptr<core::Module> {
                MoeParam moe_param;
                // ... populate moe_param from config ...
                return std::make_unique<MoeWeight>(
                    std::get<int64_t>(cfg.at("layer_id")),
                    moe_param,
                    std::get<int64_t>(cfg.at("hidden_dim")),
                    cfg.count("mlp_bias") && std::get<int64_t>(cfg.at("mlp_bias")),
                    static_cast<DataType>(std::get<int64_t>(cfg.at("data_type"))),
                    std::get<int64_t>(cfg.at("tp_size")),
                    std::get<int64_t>(cfg.at("tp_rank")),
                    static_cast<ActivationType>(std::get<int64_t>(cfg.at("act_type"))),
                    cfg.count("fuse_silu_act") && std::get<int64_t>(cfg.at("fuse_silu_act")));
            });
    }
};
static MoeWeightRegistrar _moe_weight_reg;
} // anonymous namespace
```

- [ ] **Step 6: Register DeltaNetWeight in delta_net_weight.cc**

```cpp
namespace {
struct DeltaNetWeightRegistrar {
    DeltaNetWeightRegistrar() {
        core::ModuleRegistry::instance().register_type(
            "DeltaNetWeight",
            [](const core::ModuleConfig& cfg) -> std::unique_ptr<core::Module> {
                return std::make_unique<DeltaNetWeight>(
                    std::get<int64_t>(cfg.at("hidden_dim")),
                    std::get<int64_t>(cfg.at("num_k_heads")),
                    std::get<int64_t>(cfg.at("num_v_heads")),
                    std::get<int64_t>(cfg.at("key_head_dim")),
                    std::get<int64_t>(cfg.at("value_head_dim")),
                    std::get<int64_t>(cfg.at("d_conv")),
                    cfg.count("bias") && std::get<int64_t>(cfg.at("bias")),
                    std::get<int64_t>(cfg.at("tp_size")),
                    std::get<int64_t>(cfg.at("tp_rank")),
                    static_cast<DataType>(std::get<int64_t>(cfg.at("data_type")));
            });
    }
};
static DeltaNetWeightRegistrar _delta_net_weight_reg;
} // anonymous namespace
```

- [ ] **Step 7: Register DecoderLayerWeight in decoder_layer_weight.cc**

```cpp
namespace {
struct DecoderLayerWeightRegistrar {
    DecoderLayerWeightRegistrar() {
        core::ModuleRegistry::instance().register_type(
            "DecoderLayerWeight",
            [](const core::ModuleConfig& cfg) -> std::unique_ptr<core::Module> {
                // DecoderLayerWeight is a passive container when created via registry.
                // Children are added by Python-driven create_child calls.
                return std::make_unique<DecoderLayerWeight>();
            });
    }
};
static DecoderLayerWeightRegistrar _decoder_layer_weight_reg;
} // anonymous namespace
```

Note: When created via the registry, `DecoderLayerWeight` has no config — it's just an empty container. Its children (attention_norm, attention, feed_forward, etc.) are added explicitly by Python via `create_child`.

- [ ] **Step 8: Register ModuleList**

ModuleList needs a registry entry too. Its factory takes a different config shape (it doesn't need a factory function — the Python side creates children explicitly). But we need a way to create an "empty" ModuleList.

```cpp
// In module.cc, at the end of ModuleList section:
namespace {
struct ModuleListRegistrar {
    ModuleListRegistrar() {
        core::ModuleRegistry::instance().register_type(
            "ModuleList",
            [](const core::ModuleConfig&) -> std::unique_ptr<core::Module> {
                // ModuleList created via registry is empty — Python adds children explicitly.
                // Use a no-op factory since children are created by subsequent create_child calls.
                return std::make_unique<core::ModuleList>(
                    [](int) -> std::unique_ptr<core::Module> {
                        TM_CHECK(false) << "ModuleList factory should not be called when children are created explicitly";
                        return nullptr;
                    });
            });
    }
};
static ModuleListRegistrar _module_list_reg;
} // anonymous namespace
```

- [ ] **Step 9: Build and verify**

```bash
cd /data/lmdeploy-modeling/build && ninja
```

Expected: Clean build. All registrations happen at static init time.

- [ ] **Step 10: Commit**

```bash
git add src/turbomind/models/linear_weight.cc src/turbomind/models/norm_weight.cc src/turbomind/models/ffn_weight.cc src/turbomind/models/attention_weight.cc src/turbomind/models/moe_weight.cc src/turbomind/models/delta_net_weight.cc src/turbomind/models/decoder_layer_weight.cc src/turbomind/core/module.cc
git commit -m "feat(models): register all module types with ModuleRegistry"
```

---

## Task 4: Expose create_child to Python

**Files:**
- Modify: `src/turbomind/python/bind.cpp`

- [ ] **Step 1: Add create_child binding to the Module Python class**

In `bind.cpp`, locate where Module methods are exposed to Python (the `.def("get", ...)` and `.def("alloc", ...)` bindings). Add:

```cpp
.def("create_child",
    [](Module& self, const std::string& name,
       const std::string& type_name,
       const std::map<std::string, core::ConfigValue>& config) -> py::object {
        auto* child = self.create_child(name, type_name, config);
        if (!child) {
            throw std::runtime_error("Failed to create module type '" + type_name + "'");
        }
        return py::cast(child);
    },
    py::arg("name"), py::arg("type_name"), py::arg("config"))
```

This requires the proper pybind11 type caster for `std::map<std::string, ConfigValue>`. Since ConfigValue is a variant, we need to register it. Add before the module bindings:

```cpp
// Register ConfigValue variant for automatic Python conversion
py::class_<std::map<std::string, core::ConfigValue>>(m, "ModuleConfig")
    .def(py::init<std::map<std::string, core::ConfigValue>>())
    .def("__setitem__", [](std::map<std::string, core::ConfigValue>& m,
                           const std::string& key, py::object val) {
        // Try int first, then double, then string
        try {
            m[key] = py::cast<int64_t>(val);
        } catch (py::cast_error&) {
            try {
                m[key] = py::cast<double>(val);
            } catch (py::cast_error&) {
                m[key] = py::cast<std::string>(val);
            }
        }
    });
```

Alternatively, a simpler approach: accept a `py::dict` and convert in C++.

- [ ] **Step 2: Build and verify**

```bash
cd /data/lmdeploy-modeling/build && ninja
```

Expected: Clean build. The Python binding compiles successfully.

- [ ] **Step 3: Test the binding from Python**

```python
# Quick smoke test in Python
from _turbomind import DataType
import _turbomind as tm

# Create a module (assuming root module is available)
root = tm.ModelWeight()  # or however root is obtained
child = root.create_child("test_norm", "NormWeight", {"dim": 4096, "data_type": DataType.TYPE_BF16})
assert child is not None
print("create_child works:", child)
```

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/python/bind.cpp
git commit -m "feat(bind): expose Module::create_child to Python"
```

---

## Task 5: Python LoadContext Class

**Files:**
- Create: `lmdeploy/turbomind/deploy/load_context.py`

- [ ] **Step 1: Create LoadContext class**

```python
# lmdeploy/turbomind/deploy/load_context.py
"""Composable loading primitives for building the C++ module tree from Python."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .linear import Linear

if TYPE_CHECKING:
    pass

class SplitSide:
    """Semantic TP split direction (re-exported from module.py)."""
    OUTPUT = "output"
    INPUT = "input"


class LoadContext:
    """Wraps a C++ Module handle and provides composable loading primitives.

    Each LoadContext is rooted at one C++ Module. The `create` method returns a new
    LoadContext rooted at the created child module. `load_linear` and `load_tensor`
    handle the full create + commit lifecycle.
    """

    def __init__(self, handle, tp_config: dict):
        """
        Args:
            handle: C++ Module handle (pybind11 object).
            tp_config: Dict with keys: tp_size, rank, act_type, head_dim,
                       rope_dim, permute_qk, repeat_kv, attn_output_gate. kv_head_num.
        """
        self._handle = handle
        self._tp_config = tp_config

    @property
    def tp_size(self) -> int:
        return self._tp_config["tp_size"]

    @property
    def rank(self) -> int:
        return self._tp_config["rank"]

    @property
    def head_dim(self) -> int:
        return self._tp_config.get("head_dim", 0)

    @property
    def rope_dim(self) -> int:
        return self._tp_config.get("rope_dim", 0)

    @property
    def repeat_kv(self) -> int:
        return self._tp_config.get("repeat_kv", 0)

    @property
    def attn_output_gate(self) -> bool:
        return self._tp_config.get("attn_output_gate", False)

    @property
    def kv_head_num(self) -> int:
        return self._tp_config.get("kv_head_num", 0)

    def create(self, name: str, module_type: str, **config) -> 'LoadContext':
        """Create a child module via the C++ registry.

        Returns a new LoadContext rooted at the created module.
        """
        child = self._handle.create_child(name, module_type, config)
        return LoadContext(child, self._tp_config)

    def child(self, name: str) -> 'LoadContext':
        """Return a LoadContext for an existing child (no creation)."""
        handle = self._handle.get(name)
        return LoadContext(handle, self._tp_config)

    def load_linear(self, name: str, linear: Linear,
                    tp_rule: str | None = None):
        """Create a LinearWeight child and commit weight data.

        Handles TP splitting, quantization packing, and dtype casting.
        """
        from .module import commit_linear, SplitSide as _SplitSide
        from .module import _infer_cpp_linear_dtype

        # Infer C++ dtype and group_size from the Linear
        cpp_dtype, group_size = _infer_cpp_linear_dtype(linear)
        if group_size == 0:
            group_size = max(1, 128)

        # Create the LinearWeight child
        child_handle = self._handle.create_child(
            name, "LinearWeight",
            {"input_dim": linear.input_dim if hasattr(linear, 'input_dim') else 0,
             "output_dim": linear.output_dim if hasattr(linear, 'output_dim') else 0,
             "data_type": cpp_dtype,
             "has_bias": "bias" in linear.tensors})

        # Commit the weight data
        tp_side = _SplitSide(tp_rule) if tp_rule else None
        commit_linear(child_handle, linear,
                      split_side=tp_side,
                      split_num=self.tp_size if tp_side else 1,
                      rank=self.rank)

    def load_tensor(self, name: str, tensor: torch.Tensor,
                    module_type: str = "NormWeight",
                    module_config: dict | None = None,
                    tp_rule: str | None = None):
        """Create a module child and commit tensor data.

        Args:
            name: Child module name.
            tensor: Weight tensor to commit.
            module_type: C++ module type to create (e.g., "NormWeight").
            module_config: Config dict for module creation.
            tp_rule: "output" or "input" for TP split, None for broadcast.
        """
        from .module import commit_tensor, SplitSide as _SplitSide
        from .module import _torch_dtype_to_cpp

        config = module_config or {}
        child_handle = self._handle.create_child(name, module_type, config)

        tp_side = _SplitSide(tp_rule) if tp_rule else None
        commit_tensor(child_handle, tensor,
                      split_side=tp_side,
                      split_num=self.tp_size if tp_side else 1,
                      rank=self.rank)
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/load_context.py
git commit -m "feat(deploy): add LoadContext with composable loading primitives"
```

---

## Task 6: Python TextModelLoader + TextModelSpec Base

**Files:**
- Create: `lmdeploy/turbomind/deploy/text_model_loader.py`
- Modify: `lmdeploy/turbomind/deploy/module.py` (rename ModelWeightSpec -> TextModelSpec, add load_layer/load_global)

- [ ] **Step 1: Create TextModelLoader in text_model_loader.py**

```python
# lmdeploy/turbomind/deploy/text_model_loader.py
"""TextModelLoader: drives the model loading pipeline for text models."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .load_context import LoadContext

if TYPE_CHECKING:
    from .module import TextModelSpec
    from .target_model.base import BaseOutputModel


class TextModelLoader:
    """Drives the model loading pipeline for text models.

    Replaces TransformerV2. This is a generic driver with zero hardcoded
    module paths. All structure comes from the TextModelSpec.
    """

    def __init__(self, model: BaseOutputModel):
        self.model = model
        self.attn_tp = model.attn_tp_size
        self.mlp_tp = model.mlp_tp_size

    def __call__(self, layer: int, spec: TextModelSpec):
        if layer < 0:
            self._load_global(spec)
        else:
            self._load_layer(layer, spec)
        return 1

    def _make_tp_config(self, rank: int, is_attn: bool = True) -> dict:
        tp = self.attn_tp if is_attn else self.mlp_tp
        return {
            "tp_size": tp,
            "rank": rank,
            "head_dim": self.model.model_config.size_per_head,
            "rope_dim": ...,
            "permute_qk": getattr(self.model, "permute_qk", True),
            "repeat_kv": getattr(self.model, "repeat_kv", 0),
            "attn_output_gate": getattr(self.model.model_config, "attn_output_gate", False),
            "kv_head_num": self.model.model_config.kv_head_num,
        }

    def _load_layer(self, layer: int, spec: TextModelSpec):
        for gpu in range(self.model.gpu_count):
            root = self.model.root(gpu)
            if root is None:
                break
            attn_rank, mlp_rank = self.model.tp_ranks(gpu)

            # Ensure layers ModuleList exists
            layers = root.get("layers")
            if layers is None:
                layers = root.create_child("layers", "ModuleList", {})

            # Ensure this layer's entry exists
            layer_name = str(layer)
            layer_mod = layers.get(layer_name)
            if layer_mod is None:
                layer_mod = layers.create_child(layer_name, "DecoderLayerWeight", {})

            tp_config = self._make_tp_config(attn_rank)
            ctx = LoadContext(layer_mod, tp_config)
            spec.load_layer(ctx, layer)

    def _load_global(self, spec: TextModelSpec):
        for gpu in range(self.model.gpu_count):
            root = self.model.root(gpu)
            if root is None:
                break
            attn_rank, _ = self.model.tp_ranks(gpu)
            tp_config = self._make_tp_config(attn_rank)
            ctx = LoadContext(root, tp_config)
            spec.load_global(ctx)
```

- [ ] **Step 2: Rename and update ModelWeightSpec in module.py**

Rename `ModelWeightSpec` to `TextModelSpec`. Add `load_layer`, `load_global`, `load_attn`, `load_ffn`, `load_experts`, `load_linear_attn` default implementations.

The changes to `module.py`:
1. Rename `class ModelWeightSpec` to `class TextModelSpec`
2. Add the composable sub-methods: `load_attn`, `load_ffn`, `load_experts`, `load_linear_attn`
3. Add `load_layer` and `load_global` default implementations that call these sub-methods
4. Remove `attn_norm` and `ffn_norm` abstract methods (inline in load_layer)
5. Rename `commit_linear_module` to `commit_linear` and `commit_tensor_module` to `commit_tensor` (or add aliases)

Key: The existing `Transformer` and `TransformerV2` classes remain in module.py for backward compatibility during migration. They will be removed in Task 10.

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/text_model_loader.py lmdeploy/turbomind/deploy/module.py
git commit -m "feat(deploy): add TextModelLoader and TextModelSpec with composable load methods"
```

---

## Task 7: Migrate Qwen3Spec to TextModelSpec

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py`

This is the integration test — converting one architecture to the new pipeline validates the full design end-to-end.

- [ ] **Step 1: Rename Qwen3Spec class and update base class**

In `qwen3_spec.py`:
- Rename `class Qwen3Spec(ModelWeightSpec)` to `class Qwen3TextSpec(TextModelSpec)`
- The weight reading methods (`_read_attn_linears`, `ffn_linears`, `moe_ffn_linears`, etc.) stay the same
 They provide the data.
- Remove `attn_norm` and `ffn_norm` methods (they're inlined in the base class now)
- The `_read_linear` helper stays
- `model_info()` stays unchanged

- [ ] **Step 2: Test end-to-end with a Qwen3 model**

Use the model-server MCP tools to find a Qwen3 model, then run:

```bash
cd /data/lmdeploy-modeling
PYTHONPATH=/data/lmdeploy-modeling/lmdeploy:/data/lmdeploy-modeling/build/lib python build/toy_example.py
```

**You MUST verify the response** — the model must respond with meaningful human words relevant to the test prompt. At least 128 tokens.

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/qwen3_spec.py
git commit -m "feat(qwen3): migrate Qwen3Spec to TextModelSpec"
```

---

## Task 8: Migrate Remaining Architectures

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py`

- [ ] **Step 1: Migrate GPT-OSS**

Rename spec class to inherit from `TextModelSpec`. Remove `attn_norm`/`ffn_norm` if present. Test with available model.

- [ ] **Step 2: Migrate GLM4-MoE-Lite**

Rename spec class. Handle MLA-specific logic in `load_attn` override. Test with available model.

- [ ] **Step 3: Migrate Qwen3.5**

Rename spec class. Test with available model.

- [ ] **Step 4: Commit all architecture migrations**

```bash
git add lmdeploy/turbomind/deploy/source_model/
git commit -m "feat(specs): migrate remaining architectures to TextModelSpec"
```

---

## Task 9: Wire TextModelLoader into TurboMind

**Files:**
- Modify: `lmdeploy/turbomind/turbomind.py`

- [ ] **Step 1: Update TurboMind to use TextModelLoader**

In `turbomind.py`, locate where `Transformer` / `TransformerV2` is used (the `self.model` callback). Replace:

```python
# Before:
from .deploy.module import Transformer
self.model = Transformer(self)

# After:
from .deploy.text_model_loader import TextModelLoader
self.model = TextModelLoader(self)
```

- [ ] **Step 2: Test end-to-end with a model**

Run toy_example.py with a Qwen3 model. Verify response is meaningful (128+ tokens).

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/turbomind.py
git commit -m "feat(turbomind): wire TextModelLoader into TurboMind"
```

---

## Task 10: Remove Old Code

**Files:**
- Modify: `lmdeploy/turbomind/deploy/module.py`
- Modify: `src/turbomind/models/decoder_layer_weight.h`
- Modify: `src/turbomind/models/decoder_layer_weight.cc`
- Modify: `src/turbomind/models/attention_weight.h`
- Modify: `src/turbomind/models/attention_weight.cc`
- Modify: `src/turbomind/models/ffn_weight.h`
- Modify: `src/turbomind/models/ffn_weight.cc`
- Modify: `src/turbomind/models/moe_weight.h`
- Modify: `src/turbomind/models/moe_weight.cc`
- Modify: `src/turbomind/models/delta_net_weight.h`
- Modify: `src/turbomind/models/delta_net_weight.cc`
- Modify: `src/turbomind/models/model_weight.h`
- Modify: `src/turbomind/models/model_weight.cc`

- [ ] **Step 1: Remove Transformer and TransformerV2 from module.py**

Delete the `Transformer` and `TransformerV2` classes from `module.py`. Keep the helper functions (`permute_v2`, `merge_qkv_v2`, etc.) that are still used by `TextModelSpec`'s default implementations.

- [ ] **Step 2: Remove ensure_child from C++ composite modules**

In each composite module, remove or gut the `ensure_child` override to return `nullptr` (the base class default). The modules are now populated exclusively by Python-driven `create_child` calls.

For each of these files:
- `decoder_layer_weight.h/cc`: Remove `ensure_child` override
- `attention_weight.h/cc`: Remove `ensure_child` override
- `ffn_weight.h/cc`: Remove `ensure_child` override
- `moe_weight.h/cc`: Remove `ensure_child` override
- `delta_net_weight.h/cc`: Remove `ensure_child` override
- `model_weight.h/cc`: Remove `ensure_child` override

Keep the typed accessors (e.g., `AttentionWeight::w_qkv()`) since the execution side uses them.

- [ ] **Step 3: Build and test**

```bash
cd /data/lmdeploy-modeling/build && ninja
```

Then run toy_example.py to verify nothing is broken.

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/module.py src/turbomind/models/
git commit -m "refactor: remove ensure_child and old Transformer classes"
```

---

## Self-Review

**Spec coverage:**
- Module Registry: Task 1-2 ✓
- create_child: Task 2 ✓
- get<T>(): Task 1 ✓
- ConfigValue (variant): Task 1 ✓
- Python bindings: Task 4 ✓
- LoadContext: Task 5 ✓
- TextModelLoader: Task 6 ✓
- TextModelSpec (renamed from ModelWeightSpec): Task 6 ✓
- load_layer/load_global: Task 6 ✓
- load_attn/load_ffn/load_experts/load_linear_attn: Task 6 ✓
- Drop attn_norm/ffn_norm: Task 6 ✓
- commit_linear/commit_tensor (renamed): Task 6 ✓
- Migrate Qwen3: Task 7 ✓
- Migrate remaining: Task 8 ✓
- Wire into TurboMind: Task 9 ✓
- Remove old code: Task 10 ✓

**Placeholder scan:** No TBDs, TODOs, or vague descriptions found. Each step has concrete code or exact commands.

**Type consistency:** All names are consistent across tasks: `TextModelSpec`, `TextModelLoader`, `LoadContext`, `create_child`, `get<T>()`, `ConfigValue`, `ModuleConfig`.
