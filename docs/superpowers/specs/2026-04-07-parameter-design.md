# Parameter Design — Replace Misused NormWeight with Self-Registering Tensor Members

## Problem

`NormWeight` (a Module wrapping a single tensor for normalization) is used as a generic parameter
container for non-norm parameters:

| Module | Field | Actually stores |
|---|---|---|
| AttentionWeight | sinks_mod | Attention sinks (per-head bias) |
| DeltaNetWeight | conv1d | 2D causal conv kernel |
| DeltaNetWeight | A_log | Log of state transition matrix |
| DeltaNetWeight | dt_bias | Delta-time bias |

Each creates an unnecessary child Module (NormWeight) just to hold one tensor.

`q_norm` and `k_norm` are legitimate per-head RMSNorm — they stay as `Submodule<NormWeight>`.

## Solution

Introduce `Parameter` — a self-registering tensor member. Like PyTorch's `nn.Parameter`:
the tensor lives directly on its parent module, no intermediate child module.

### Parameter class

```cpp
class Parameter {
    std::string name_;
    Module*     parent_;
    Tensor      tensor_;

public:
    Parameter(Module& parent, std::string name)
        : name_(std::move(name)), parent_(&parent)
    {
        parent.add_param(name_, tensor_);  // register in existing params_
    }

    explicit operator bool() const { return static_cast<bool>(tensor_); }

    Tensor&       operator*()        { return tensor_; }
    const Tensor& operator*()  const { return tensor_; }

    Tensor*       ptr()              { return &tensor_; }
    const Tensor* ptr()        const { return &tensor_; }
};
```

- `add_param(name_, tensor_)` registers the (empty) tensor in the parent's `params_` map
- `operator bool()` — check if allocated (same as Submodule)
- `operator*()` / `ptr()` — access the tensor (no cast needed, already typed)

### C++ usage

**Before** (AttentionWeight):
```cpp
core::Submodule<NormWeight> sinks_mod {*this, "sinks"};
// + convenience accessor:
Tensor* sinks() const { return sinks_mod ? &sinks_mod->weight() : nullptr; }
```

**After**:
```cpp
core::Parameter sinks_ {*this, "sinks"};
// + convenience accessor:
Tensor* sinks() const { return sinks_ ? sinks_.ptr() : nullptr; }
```

### Lazy allocation via alloc override

```cpp
Tensor AttentionWeight::alloc(const std::string& param_name, const core::WeightSpec& spec)
{
    if (param_name == "sinks" && !sinks_) {
        *sinks_ = Tensor{{head_num_ / tp_size_}, spec.dtype, kDEVICE};
    }
    if (param_name == "sinks") return *sinks_;
    return Module::alloc(param_name, spec);
}
```

The Python loader calls `module.alloc("sinks", dtype, 0)` which routes to this override.
The tensor is allocated lazily using `spec.dtype` from Python.

### Python loader change

The auto-creation loop detects direct parameters via `param()` check:

```python
for seg in parts[:-1]:
    child = mod.child(seg)
    if child is None:
        if mod.param(seg) is not None:   # ← direct Parameter, not a child module
            break
        child = mod.create_child(seg, 'NormWeight', ...)
    mod = child
```

- `Parameter` registers its tensor via `add_param` at construction
- `mod.param("sinks")` finds it in `params_` (empty tensor, but registered)
- The break short-circuits the loop — no NormWeight child created
- `commit_tensor(mod, tensor, "sinks")` → `alloc("sinks", ...)` → lazy allocation

### Python spec path change

Paths lose the `.weight` suffix (parameter name replaces child module name):

| Before | After |
|---|---|
| `("attention.sinks.weight", ...)` | `("attention.sinks", ...)` |
| `("linear_attn.conv1d.weight", ...)` | `("linear_attn.conv1d", ...)` |
| `("linear_attn.A_log.weight", ...)` | `("linear_attn.A_log", ...)` |
| `("linear_attn.dt_bias.weight", ...)` | `("linear_attn.dt_bias", ...)` |

### Caller changes

Callers that use convenience accessors stay the same — the accessor signature doesn't change:
```cpp
// Before: weights.sinks() → Tensor* (via sinks_mod->weight())
// After:  weights.sinks() → Tensor* (via sinks_.ptr())
```

Callers that access `weights.conv1d->weight()` directly need a new accessor:
```cpp
// DeltaNetWeight needs:
Tensor* conv1d() const { return conv1d_ ? conv1d_.ptr() : nullptr; }
Tensor* A_log() const { return A_log_ ? A_log_.ptr() : nullptr; }
Tensor* dt_bias() const { return dt_bias_ ? dt_bias_.ptr() : nullptr; }
```

## What gets deleted

- `Submodule<NormWeight> sinks_mod` in AttentionWeight
- `Submodule<NormWeight> conv1d / A_log / dt_bias` in DeltaNetWeight
- `#include "norm_weight.h"` from delta_net_weight.h (if no other NormWeight members remain)

## What gets added

- `Parameter` class in module.h
- `Parameter` members in AttentionWeight, DeltaNetWeight
- `alloc()` overrides in AttentionWeight, DeltaNetWeight
- Convenience accessors for conv1d, A_log, dt_bias on DeltaNetWeight
- One-line check in text_model_loader.py auto-creation loop
- Shorter paths in gpt_oss_spec.py, qwen3_5_spec.py

## Files

- `src/turbomind/core/module.h` — add Parameter class
- `src/turbomind/models/attention_weight.h` — sinks_ as Parameter
- `src/turbomind/models/attention_weight.cc` — alloc override, accessor
- `src/turbomind/models/delta_net_weight.h` — conv1d_, A_log_, dt_bias_ as Parameter
- `src/turbomind/models/delta_net_weight.cc` — alloc override, accessors
- `src/turbomind/models/llama/GatedDeltaNetLayer.cc` — accessor changes
- `lmdeploy/turbomind/deploy/text_model_loader.py` — param check in auto-creation
- `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` — shorter paths
- `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` — shorter paths
