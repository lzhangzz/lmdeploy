# Loading Pipeline Cleanup Design

Date: 2026-04-08
Scope: Python loading flow + C++ module system
Goal: Clean architecture — clear separation of concerns, typed configs, single loading API

## Problem Statement

The TurboMind model loading pipeline has accumulated complexity despite recent refactoring (April 3-7):

**Python loading flow:**
- `TextModelLoader._load_layer` is a 274-line god method with hardcoded module type strings and inline config dicts
- Module creation scattered across `TextModelLoader`, `LoadContext`, and `commit_linear`
- `commit_linear` takes 7 params and creates modules AND commits weights
- `TextModelSpec.configure` takes 7 params, some with "TODO: doesn't belong here" comments
- Config data flows as raw dicts with magic string keys from Python to C++

**C++ module system:**
- Inconsistent parameter storage: mix of `Parameter` wrapper, direct `Tensor` members, and lazy `alloc()` overrides
- Weight classes have long constructor parameter lists (13 params for `AttentionWeight`)
- `release()` and `to_device()` are separate virtual methods doing similar recursive tensor management
- MoE block view creation is 62 lines of manual field-copying boilerplate

## Design Decisions

1. **Per-module typed configs** — Replace raw dicts and long param lists with typed config dataclasses (Python) / structs (C++)
2. **Keep Submodule<T>** — The slot-wiring mechanism works; no need to replace it
3. **Unify on Parameter** — All leaf tensors use the `Parameter` wrapper for consistent storage and allocation
4. **Single loading API** — `LoadContext` absorbs `commit_linear` / `commit_tensor`, becoming the only way to commit weights
5. **Decompose the god method** — `_load_layer` splits into per-component methods
6. **Keep the fusion tuple** — No new result object for `fuse_ffn_linears`
7. **Keep prepare()** — Rename is not necessary

## Part 1: Per-Module Typed Configs

### Python Config Dataclasses

New file `lmdeploy/turbomind/deploy/configs.py`:

```python
from dataclasses import dataclass

@dataclass
class LinearConfig:
    input_dim: int
    output_dim: int
    dtype: int  # DataType enum value
    has_bias: bool = False

@dataclass
class AttentionConfig:
    hidden_dim: int
    head_dim: int
    head_num: int
    kv_head_num: int
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

    @classmethod
    def from_model_config(cls, mc, *, tp_size, tp_rank, dtype, window_size):
        return cls(
            hidden_dim=mc.hidden_size,
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

@dataclass
class FfnConfig:
    hidden_dim: int
    inter_size: int
    has_bias: bool = False
    tp_size: int = 1
    tp_rank: int = 0
    data_type: int = 0
    act_type: int = 0
    fuse_silu: bool = False

    @classmethod
    def from_model_config(cls, mc, *, tp_size, tp_rank, dtype, act_type, fuse_silu):
        return cls(
            hidden_dim=mc.hidden_size,
            inter_size=mc.inter_size,
            has_bias=mc.mlp_bias,
            tp_size=tp_size,
            tp_rank=tp_rank,
            data_type=dtype,
            act_type=act_type,
            fuse_silu=fuse_silu,
        )

@dataclass
class MoeConfig:
    layer_id: int
    num_experts: int
    top_k: int
    hidden_dim: int
    has_bias: bool = False
    tp_size: int = 1
    tp_rank: int = 0
    data_type: int = 0
    act_type: int = 0
    fuse_silu: bool = False
    # MoE routing method: 0=topk, 1=shared, 2=topk_group
    method: int = 0
    # Number of shared experts (0 if none)
    num_shared_experts: int = 0
    # Whether to use grouped GEMM for MoE
    grouped_gemm: bool = False

    @classmethod
    def from_model_config(cls, mc, *, layer_id, tp_size, tp_rank, dtype, act_type, fuse_silu):
        return cls(
            layer_id=layer_id,
            num_experts=mc.num_experts,
            top_k=mc.moe_top_k,
            hidden_dim=mc.hidden_size,
            has_bias=mc.mlp_bias,
            tp_size=tp_size,
            tp_rank=tp_rank,
            data_type=dtype,
            act_type=act_type,
            fuse_silu=fuse_silu,
            method=mc.moe_method,
            num_shared_experts=mc.num_shared_experts or 0,
            grouped_gemm=mc.moe_grouped_gemm,
        )

@dataclass
class DeltaNetConfig:
    hidden_dim: int
    num_k_heads: int
    num_v_heads: int
    key_head_dim: int
    value_head_dim: int
    d_conv: int = 4
    has_bias: bool = False
    tp_size: int = 1
    tp_rank: int = 0
    data_type: int = 0

    @classmethod
    def from_model_config(cls, mc, *, tp_size, tp_rank, dtype):
        return cls(
            hidden_dim=mc.hidden_size,
            num_k_heads=mc.num_k_heads,
            num_v_heads=mc.num_v_heads,
            key_head_dim=mc.key_head_dim,
            value_head_dim=mc.value_head_dim,
            d_conv=mc.d_conv or 4,
            has_bias=mc.attn_bias,
            tp_size=tp_size,
            tp_rank=tp_rank,
            data_type=dtype,
        )
```

### C++ Config Structs

New file `src/turbomind/core/module_config.h`:

```cpp
#pragma once
#include "src/turbomind/core/data_type.h"
#include <vector>
#include <string>

namespace turbomind::core {

struct LinearConfig {
    int input_dim{};
    int output_dim{};
    DataType dtype{};
    bool has_bias{};
};

struct AttentionConfig {
    int hidden_dim{};
    int head_dim{};
    int head_num{};
    int kv_head_num{};
    int kv_lora_rank{};
    int q_lora_rank{};
    int qk_rope_dim{};
    int v_head_dim{};
    bool has_bias{};
    bool qk_norm{};
    int tp_size{};
    int tp_rank{};
    DataType data_type{};
    int window_size{-1};
    bool attn_sink{};
    bool attn_output_gate{};
};

struct FfnConfig {
    int hidden_dim{};
    int inter_size{};
    bool has_bias{};
    int tp_size{};
    int tp_rank{};
    DataType data_type{};
    int act_type{};
    bool fuse_silu{};
};

struct MoeConfig {
    int layer_id{};
    int num_experts{};
    int top_k{};
    int hidden_dim{};
    bool has_bias{};
    int tp_size{};
    int tp_rank{};
    DataType data_type{};
    int act_type{};
    bool fuse_silu{};
};

struct DeltaNetConfig {
    int hidden_dim{};
    int num_k_heads{};
    int num_v_heads{};
    int key_head_dim{};
    int value_head_dim{};
    int d_conv{4};
    bool has_bias{};
    int tp_size{};
    int tp_rank{};
    DataType data_type{};
};

}  // namespace turbomind::core
```

### Module constructors use typed configs

```cpp
// Before:
AttentionWeight(int hidden_dim, int head_dim, int head_num, int kv_head_num,
                MLAParam mla, bool bias, bool qk_norm, int tp_size, int tp_rank,
                DataType data_type, int window_size, bool sink, bool attn_output_gate);

// After:
AttentionWeight(const AttentionConfig& cfg);
```

Same pattern for `FfnWeight(FfnConfig)`, `MoeWeight(MoeConfig)`, `DeltaNetWeight(DeltaNetConfig)`, `LinearWeight(LinearConfig)`.

### Python-C++ binding

The pybind11 layer converts Python dataclass fields to C++ struct fields:

```cpp
// In bind.cpp
py::class_<AttentionConfig>(m, "AttentionConfig")
    .def(py::init<>())
    .def_readwrite("hidden_dim", &AttentionConfig::hidden_dim)
    .def_readwrite("head_dim", &AttentionConfig::head_dim)
    // ... all fields
    ;
```

Python constructs the config dataclass, passes it to `create_child`:

```python
# Before:
attn_mod = handle.create_child('attention', 'AttentionWeight', {
    'hidden_dim': hidden,
    'head_dim': mc.size_per_head,
    # ... 15 more magic-string keys
})

# After:
attn_cfg = AttentionConfig.from_model_config(mc, tp_size=self.attn_tp, tp_rank=attn_rank, dtype=dtype, window_size=window_size)
attn_mod = handle.create_child('attention', attn_cfg)
```

`create_child` is overloaded in the Python binding to accept either a typed config object or the legacy `(name, type_name, config_dict)` triple:

```cpp
// In bind.cpp — new overload
.def("create_child",
    [](Module& m, const std::string& name, py::object config_obj) -> Module* {
        // Each config type is bound with a static kTypeName field
        // and a C++ factory that accepts the config struct
        std::string type_name = config_obj.attr("k_type_name").cast<std::string>();
        auto cfg = config_to_cpp(config_obj);  // per-type conversion
        return m.create_child(name, type_name, cfg);
    })

// Existing overload stays for backward compatibility during migration
.def("create_child",
    [](Module& m, const std::string& name, const std::string& type_name,
       const std::map<std::string, py::object>& config) -> Module* {
        // ... existing dict-based path
    })
```

Python config dataclasses have a `k_type_name` class attribute (e.g., `"AttentionWeight"`) that maps to the C++ registry type. During migration, both paths coexist; the legacy dict path is removed once all callers are migrated.

### TextModelSpec.configure simplification

```python
# Before:
def configure(self, attn_tp=1, permute_qk=True, repeat_kv=0, head_dim=0,
              rope_dim=0, attn_output_gate=False, kv_head_num=0):

# After:
@dataclass
class SpecAttnConfig:
    tp: int = 1
    permute_qk: bool = True
    repeat_kv: int = 0
    head_dim: int = 0
    rope_dim: int = 0
    output_gate: bool = False
    kv_head_num: int = 0

def configure(self, attn_cfg: SpecAttnConfig):
```

`SpecAttnConfig` is separate from the C++ `AttentionConfig` because it carries spec-specific concerns (permute_qk, repeat_kv) that don't apply to the C++ module constructor.

## Part 2: C++ Module System Cleanup

### Unify on Parameter for all leaf tensors

**Before (LinearWeight):**
```cpp
class LinearWeight: public Module {
    Tensor weight;     // direct member
    Tensor bias;       // direct member
    Tensor scales;     // direct member
    Tensor zeros;      // direct member
    // Custom alloc() override with 30+ lines
};
```

**After:**
```cpp
class LinearWeight: public Module {
    mutable Parameter weight_{*this, "weight"};
    mutable Parameter bias_{*this, "bias"};
    mutable Parameter scales_{*this, "scales"};
    mutable Parameter zeros_{*this, "zeros"};

    Tensor& weight() { return *weight_; }
    Tensor* bias() { return bias_.ptr(); }
    // No alloc() override needed — base class handles it
};
```

All weight classes follow this pattern. Remove `alloc()` overrides from:
- `LinearWeight` — weight, bias, scales, zeros → Parameter
- `NormWeight` — weight_ → Parameter (already partially there)
- `AttentionWeight` — sinks_ already Parameter, keep as-is
- `DeltaNetWeight` — conv1d_, A_log_, dt_bias_ already Parameter, keep as-is
- `MoeWeight` — score_correction_bias_ → Parameter

### Simplify alloc()

Base `Module::alloc(name, spec)` already returns the pre-registered Parameter tensor. No overrides needed in most classes. The only exception is `LinearWeight`, which needs to trigger full allocation (all 4 tensors at once) on first access to "weight" — this can be a single override:

```cpp
Tensor alloc(const std::string& param_name, const WeightSpec& spec) override {
    if (!weight_) {
        // Allocate all tensors at once
        *weight_ = Tensor({input_dim_, output_dim_}, spec.dtype, kDEVICE);
        if (has_bias_) *bias_ = Tensor({output_dim_}, spec.dtype, kDEVICE);
        if (is_quantized_) {
            *scales_ = Tensor(scale_shape, TYPE_FP32, kDEVICE);
            *zeros_ = Tensor(zero_shape, TYPE_FP32, kDEVICE);
        }
    }
    return Module::alloc(param_name, spec);  // Return the pre-registered param
}
```

### Merge release() and to_device() into persist()

```cpp
enum class PersistOp { Sleep, WakeUp };

class Module {
    virtual void persist(PersistOp op) {
        for (auto& [name, child] : children_) {
            child->persist(op);
        }
        if (op == PersistOp::Sleep) {
            // Move tensors to CPU (cheap), free GPU memory
            for (auto& [name, ptr] : params_) {
                if (ptr && ptr->where() == kDEVICE) {
                    Tensor cpu = Tensor{ptr->shape(), ptr->type(), kCPU};
                    cpu.copy_from(*ptr);
                    *ptr = std::move(cpu);
                }
            }
        } else {  // WakeUp
            // Move tensors back to GPU
            for (auto& [name, ptr] : params_) {
                if (ptr && ptr->where() == kCPU) {
                    Tensor gpu = Tensor{ptr->shape(), ptr->type(), kDEVICE};
                    gpu.copy_from(*ptr);
                    *ptr = std::move(gpu);
                }
            }
        }
    }
};
```

Remove `release()` and `to_device()` virtual methods. `persist(Sleep)` replaces `release()` (but preserves CPU copies for fast WakeUp). `persist(WakeUp)` replaces `to_device(kGPU)` (copies CPU→GPU).

### Reduce MoE boilerplate

The `LinkLinearExperts` function in `moe_weight.cc` manually copies all `LinearWeight` fields (weight, bias, scales, zeros, plus all metadata) for each expert into a batched block view. Replace with:

```cpp
// Add to LinearWeight
void copy_metadata_to(LinearWeight& dst) const {
    dst.input_dim_ = input_dim_;
    dst.output_dim_ = output_dim_;
    dst.group_size_ = group_size_;
    dst.data_type_ = data_type_;
    dst.weight_format_ = weight_format_;
    dst.format_ = format_;
    dst.policy_ = policy_;
    dst.epilogue_ = epilogue_;
    dst.has_bias_ = has_bias_;
    dst.is_grouped_ = is_grouped_;
}
```

Then `LinkLinearExperts` becomes a loop calling `expert[i].copy_to(block[i])` instead of 62 lines of per-field copying.

## Part 3: Python Loading Flow

### Decompose _load_layer

The 274-line `_load_layer` method splits into focused methods:

```python
class TextModelLoader:
    def __call__(self, layer, spec):
        """Entry point — called per layer"""
        self._load_layer(layer, spec)

    def _load_layer(self, layer, spec):
        ctx = self._make_ctx(layer, spec)
        self._load_attention(ctx, spec, layer)
        self._load_ffn_or_moe(ctx, spec, layer)
        self._load_linear_attn(ctx, spec, layer)
        self._load_raw_tensors(ctx, spec, layer)

    def _load_attention(self, ctx, spec, layer):
        """Load attention weights for one layer"""

    def _load_ffn_or_moe(self, ctx, spec, layer):
        """Load FFN or MoE weights for one layer"""

    def _load_ffn(self, ctx, spec, layer):
        """Load dense FFN weights"""

    def _load_moe(self, ctx, spec, layer):
        """Load MoE expert weights"""

    def _load_linear_attn(self, ctx, spec, layer):
        """Load linear attention (DeltaNet) weights"""

    def _load_raw_tensors(self, ctx, spec, layer):
        """Load raw tensors (norms, etc.)"""
```

Each method uses typed configs:

```python
def _load_attention(self, ctx, spec, layer):
    attn_cfg = self._attn_config.for_rank(ctx.rank)
    ctx.create_child('attention', attn_cfg)

    linears = spec.attn_linears(layer)
    # ... merge QKV, commit weights via ctx.load_linear(...)
```

### LoadContext absorbs commit functions

`commit_linear` and `commit_tensor` move from `commit.py` into `LoadContext` methods:

```python
class LoadContext:
    def load_linear(self, name, linear, split_side=None):
        """Create LinearWeight child + commit all tensors"""
        # Creates module via registry
        # Infers dtype, group_size
        # Calls _commit_tensors internally

    def load_tensor(self, name, tensor, split_side=None):
        """Create NormWeight child + commit tensor"""
```

The internal `_commit_tensors` becomes a private method of `LoadContext`. TP split rules (`_ATTN_TP_RULES`, `_FFN_TP_RULES`, `_LINEAR_ATTN_TP_RULES`) also move into `LoadContext`.

`commit.py` becomes a thin re-export facade for backward compatibility during migration, then gets deleted once all callers use `LoadContext` directly.

### Config construction

Configs are built once in `TextModelLoader.__init__` from the model config, not at every call site:

```python
class TextModelLoader:
    def __init__(self, model):
        mc = model.model_config
        self._attn_config = AttentionConfig.from_model_config(mc, ...)
        self._ffn_config = FfnConfig.from_model_config(mc, ...)
        # ... per-layer configs (MoE) built on demand

    def _load_attention(self, ctx, spec, layer):
        cfg = self._attn_config.for_rank(ctx.rank)  # copy with different rank
        ctx.create_child('attention', cfg)
        # ...
```

`for_rank()` returns a copy of the config with a different `tp_rank`, avoiding rebuilding the entire config for each rank.

## Migration Strategy

The refactoring proceeds in these phases:

### Phase 1: C++ typed configs
1. Add config structs in `module_config.h`
2. Add new constructors to weight classes (old constructors kept for compat)
3. Update pybind11 to expose config structs
4. Update `ModuleRegistry` to support typed config creation

### Phase 2: Python typed configs
1. Add `configs.py` with dataclass definitions
2. Add `from_model_config` factory methods
3. Update `TextModelLoader` to build configs from `ModelConfig`
4. Update `create_child` calls to use config objects

### Phase 3: C++ Parameter unification
1. Convert `LinearWeight` tensors to `Parameter` wrappers
2. Simplify `alloc()` overrides
3. Convert remaining weight classes

### Phase 4: Python loading flow
1. Move `commit_linear`/`commit_tensor` into `LoadContext`
2. Move TP split rules into `LoadContext`
3. Decompose `_load_layer` into per-component methods
4. Simplify `TextModelSpec.configure` with `SpecAttnConfig`
5. Delete `commit.py` facade once all callers migrated

### Phase 5: C++ lifecycle cleanup
1. Add `persist(PersistOp)` to Module
2. Migrate Sleep/WakeUp callers
3. Remove `release()` and `to_device()` virtual methods
4. Reduce MoE boilerplate with `copy_to` helper

## Files Changed

### New files
- `lmdeploy/turbomind/deploy/configs.py` — Python config dataclasses
- `src/turbomind/core/module_config.h` — C++ config structs

### Modified files (C++)
- `src/turbomind/core/module.h` — Add `persist()`, remove `release()`/`to_device()`
- `src/turbomind/core/module.cc` — Implement `persist()`
- `src/turbomind/models/linear_weight.h/cc` — `Parameter` for tensors, `LinearConfig` constructor
- `src/turbomind/models/attention_weight.h/cc` — `AttentionConfig` constructor
- `src/turbomind/models/ffn_weight.h/cc` — `FfnConfig` constructor
- `src/turbomind/models/moe_weight.h/cc` — `MoeConfig` constructor, reduce boilerplate
- `src/turbomind/models/delta_net_weight.h/cc` — `DeltaNetConfig` constructor
- `src/turbomind/models/norm_weight.h/cc` — `Parameter` for weight
- `src/turbomind/models/model_weight.h/cc` — Typed config constructor
- `src/turbomind/python/bind.cpp` — Config struct bindings, `persist()`, `create_child` with config

### Modified files (Python)
- `lmdeploy/turbomind/deploy/text_model_loader.py` — Decompose `_load_layer`, use typed configs
- `lmdeploy/turbomind/deploy/load_context.py` — Absorb commit functions, TP rules
- `lmdeploy/turbomind/deploy/spec.py` — `SpecAttnConfig` in `configure`
- `lmdeploy/turbomind/deploy/commit.py` — Thin facade → eventual deletion
- `lmdeploy/turbomind/deploy/module.py` — Update re-exports

### Unchanged files
- `lmdeploy/turbomind/deploy/transforms.py`
- `lmdeploy/turbomind/deploy/linear.py`
- `lmdeploy/turbomind/deploy/kind_map.py`
- `lmdeploy/turbomind/deploy/converter.py`
- All source model specs
- `src/turbomind/core/registry.h/cc` — Extended, not redesigned
- `src/turbomind/core/data_format.h/cc`
