# Parameter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace misused `Submodule<NormWeight>` for non-norm parameters (sinks, conv1d, A_log, dt_bias) with a self-registering `Parameter` class that holds tensors directly on the parent module.

**Architecture:** Introduce `Parameter` alongside `Submodule<T>` in module.h. Parameter auto-registers its tensor via `add_param` on the parent module. Spec paths change from 3-segment (`"attention.sinks.weight"`) to 2-segment (`"attention.sinks"`) — the shorter path naturally avoids the auto-creation loop creating a child module, since `parts[:-1]` only processes the parent module name.

**Tech Stack:** C++17, pybind11, Python

---

## File Structure

| File | Responsibility |
|---|---|
| `src/turbomind/core/module.h` | Add Parameter class |
| `src/turbomind/models/attention_weight.h` | Replace sinks_mod Submodule with Parameter |
| `src/turbomind/models/attention_weight.cc` | alloc override, update sinks() accessor |
| `src/turbomind/models/delta_net_weight.h` | Replace conv1d/A_log/dt_bias Submodules with Parameter |
| `src/turbomind/models/delta_net_weight.cc` | alloc override, add convenience accessors |
| `src/turbomind/models/llama/GatedDeltaNetLayer.cc` | Update accessor calls |
| `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` | Shorter paths for sinks |
| `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` | Shorter paths for conv1d/A_log/dt_bias |

---

### Task 1: Add Parameter class to module.h

**Files:**
- Modify: `src/turbomind/core/module.h`

- [ ] **Step 1: Add Parameter class after Submodule**

Add after the `Submodule<T>` template (after line 192, before ModuleList):

```cpp
// ======================================================================
// Parameter — self-registering tensor member (like PyTorch nn.Parameter)
// ======================================================================

class Parameter {
    std::string name_;
    Tensor      tensor_;

public:
    Parameter(Module& parent, std::string name)
        : name_(std::move(name))
    {
        parent.add_param(name_, tensor_);
    }

    explicit operator bool() const { return static_cast<bool>(tensor_); }

    Tensor&       operator*()        { return tensor_; }
    const Tensor& operator*()  const { return tensor_; }

    Tensor*       ptr()              { return &tensor_; }
    const Tensor* ptr()        const { return &tensor_; }
};
```

- [ ] **Step 2: Commit**

```bash
git add src/turbomind/core/module.h
git commit -m "feat(core): add Parameter class for self-registering tensor members"
```

---

### Task 2: Migrate AttentionWeight sinks to Parameter

**Files:**
- Modify: `src/turbomind/models/attention_weight.h:45`
- Modify: `src/turbomind/models/attention_weight.cc:63-66`

- [ ] **Step 1: Replace Submodule declaration in header**

Change:
```cpp
    core::Submodule<NormWeight>   sinks_mod         {*this, "sinks"};
```
To:
```cpp
    core::Parameter              sinks_             {*this, "sinks"};
```

- [ ] **Step 2: Update sinks() convenience accessor in .cc**

Change:
```cpp
Tensor* AttentionWeight::sinks() const
{
    return sinks_mod ? &sinks_mod->weight() : nullptr;
}
```
To:
```cpp
Tensor* AttentionWeight::sinks() const
{
    return sinks_ ? sinks_.ptr() : nullptr;
}
```

- [ ] **Step 3: Add alloc() override declaration in header**

Add to the public section of AttentionWeight:
```cpp
    Tensor alloc(const std::string& param_name, const core::WeightSpec& spec) override;
```

- [ ] **Step 4: Add alloc() implementation in .cc**

Add before the anonymous namespace block:
```cpp
Tensor AttentionWeight::alloc(const std::string& param_name, const core::WeightSpec& spec)
{
    if (param_name == "sinks" && !sinks_) {
        *sinks_ = Tensor{{head_num_ / tp_size_}, spec.dtype, kDEVICE};
    }
    if (param_name == "sinks") {
        return *sinks_;
    }
    return Module::alloc(param_name, spec);
}
```

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/models/attention_weight.h src/turbomind/models/attention_weight.cc
git commit -m "refactor(attention): replace sinks NormWeight with Parameter"
```

---

### Task 3: Migrate DeltaNetWeight to Parameter

**Files:**
- Modify: `src/turbomind/models/delta_net_weight.h:34-36`
- Modify: `src/turbomind/models/delta_net_weight.cc`

- [ ] **Step 1: Replace Submodule declarations in header**

Change:
```cpp
    core::Submodule<NormWeight>   conv1d{*this, "conv1d"};
    core::Submodule<NormWeight>   A_log{*this, "A_log"};
    core::Submodule<NormWeight>   dt_bias{*this, "dt_bias"};
```
To:
```cpp
    core::Parameter              conv1d_{*this, "conv1d"};
    core::Parameter              A_log_{*this, "A_log"};
    core::Parameter              dt_bias_{*this, "dt_bias"};
```

Note: `norm` stays as `Submodule<NormWeight>` (legitimate norm).

- [ ] **Step 2: Add convenience accessors and alloc declaration in header**

Add to the public section:
```cpp
    const Tensor* conv1d() const { return conv1d_ ? conv1d_.ptr() : nullptr; }
    const Tensor* A_log() const { return A_log_ ? A_log_.ptr() : nullptr; }
    const Tensor* dt_bias() const { return dt_bias_ ? dt_bias_.ptr() : nullptr; }

    Tensor alloc(const std::string& param_name, const core::WeightSpec& spec) override;
```

Callers use `const DeltaNetWeight&`, so accessors return `const Tensor*`. Parameter's `const ptr()` returns `const Tensor*`. Compatible.

- [ ] **Step 3: Add alloc() implementation in .cc**

Add before the anonymous namespace block:
```cpp
Tensor DeltaNetWeight::alloc(const std::string& param_name, const core::WeightSpec& spec)
{
    if (param_name == "A_log" && !A_log_) {
        *A_log_ = Tensor{{num_v_heads_ / tp_size_}, spec.dtype, kDEVICE};
    }
    if (param_name == "A_log") return *A_log_;

    if (param_name == "dt_bias" && !dt_bias_) {
        *dt_bias_ = Tensor{{num_v_heads_ / tp_size_}, spec.dtype, kDEVICE};
    }
    if (param_name == "dt_bias") return *dt_bias_;

    if (param_name == "conv1d" && !conv1d_) {
        int conv_dim = (num_k_heads_ * key_head_dim_ + num_v_heads_ * (value_head_dim_ + key_head_dim_)) / tp_size_;
        *conv1d_ = Tensor{{d_conv_, conv_dim}, spec.dtype, kDEVICE};
    }
    if (param_name == "conv1d") return *conv1d_;

    return Module::alloc(param_name, spec);
}
```

Shape reference (from qwen3_5_spec.py): A_log and dt_bias are `[num_v_heads]` per-shard. conv1d is `[d_conv, conv_dim]` per-shard where conv_dim is the QKV projection dimension.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/delta_net_weight.h src/turbomind/models/delta_net_weight.cc
git commit -m "refactor(deltanet): replace conv1d/A_log/dt_bias NormWeight with Parameter"
```

---

### Task 4: Update GatedDeltaNetLayer.cc callers

**Files:**
- Modify: `src/turbomind/models/llama/GatedDeltaNetLayer.cc:197,214`

- [ ] **Step 1: Update accessor calls**

Change line 197:
```cpp
        ComputeBetaG_v2(beta, g, b, a, weights.A_log->weight(), weights.dt_bias->weight(), stream);
```
To:
```cpp
        ComputeBetaG_v2(beta, g, b, a, *weights.A_log(), *weights.dt_bias(), stream);
```

Change line 214:
```cpp
                              weights.conv1d->weight(),
```
To:
```cpp
                              *weights.conv1d(),
```

- [ ] **Step 2: Commit**

```bash
git add src/turbomind/models/llama/GatedDeltaNetLayer.cc
git commit -m "refactor(deltanet): update callers to use Parameter accessors"
```

---

### Task 5: Update Python spec paths

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py:136`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py:243,262`

- [ ] **Step 1: Update gpt_oss_spec.py sinks path**

Change:
```python
            tensors.append(("attention.sinks.weight", sinks, SplitSide.OUTPUT))
```
To:
```python
            tensors.append(("attention.sinks", sinks, SplitSide.OUTPUT))
```

- [ ] **Step 2: Update qwen3_5_spec.py A_log/dt_bias paths**

Change:
```python
                    tensors.append((f"linear_attn.{key}.weight", t, SplitSide.OUTPUT))
```
To:
```python
                    tensors.append((f"linear_attn.{key}", t, SplitSide.OUTPUT))
```

- [ ] **Step 3: Update qwen3_5_spec.py conv1d path**

Change:
```python
                tensors.append(("linear_attn.conv1d.weight", conv1d, SplitSide.OUTPUT))
```
To:
```python
                tensors.append(("linear_attn.conv1d", conv1d, SplitSide.OUTPUT))
```

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py
git commit -m "refactor(specs): use shorter paths for direct Parameter tensors"
```

---

### Task 6: Verify with model tests

- [ ] **Step 1: Build**

```bash
cd /data/lmdeploy-modeling/build && ninja
```

- [ ] **Step 2: Test gpt-oss-20b-BF16 (sinks parameter)**

Run the turbomind-tester agent with gpt-oss-20b-BF16, TP=1. Verify meaningful output (at least 128 tokens).

- [ ] **Step 3: Test Qwen3.5-30B-A3B (conv1d, A_log, dt_bias parameters)**

Run the turbomind-tester agent with Qwen3.5-30B-A3B, TP=1. Verify meaningful output (at least 128 tokens).

- [ ] **Step 4: Test a Qwen3 model (q_norm, k_norm — should still work as NormWeight)**

Run the turbomind-tester agent with Qwen3-8B or similar. Verify meaningful output (at least 128 tokens).
