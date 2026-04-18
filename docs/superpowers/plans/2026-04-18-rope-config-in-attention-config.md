# RopeConfig in AttentionConfig Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the C++ `RopeParam` (union-based struct) with a flat `RopeConfig` X-macro struct nested inside `AttentionConfig`, flowing rope params from Python through the X-macro/pybind pipeline.

**Architecture:** Add a plain `RopeConfig` struct (no `ModuleConfig` base) with all rope fields flattened (no union). Nest it in `AttentionConfig` via the X-macro `X(RopeConfig, rope, {})`. Register it in pybind via a new `bind_struct` template. Python specs populate `attn_cfg.rope.*` instead of scattered scalars. C++ layers read `RopeConfig` instead of `RopeParam`.

**Tech Stack:** C++17, pybind11, Python 3, X-macro config system

---

## Task 1: Add RopeConfig, bind_struct, init_rope_kernel_param overload, and to_rope_config

All C++ structural changes in one task. Downstream consumers (UnifiedAttentionLayer etc.) will break until Task 2 fixes them.

**Files:**
- Modify: `src/turbomind/models/attention_weight.h`
- Modify: `src/turbomind/models/attention_weight.cc`
- Modify: `src/turbomind/python/bind.cpp`

- [ ] **Step 1: Add RopeConfig struct to attention_weight.h**

Add `#include <array>` to the includes. Then insert before `struct AttentionConfig` (before line 12), after `namespace turbomind::core {`:

```cpp
using MropeSection = std::array<int, 3>;

struct RopeConfig {
    #define ROPE_FIELDS(X) \
        X(int,            type, 0) \
        X(float,          base, 10000.f) \
        X(float,          factor, 1.f) \
        X(int,            max_position_embeddings, 0) \
        X(float,          yarn_attention_factor, 1.f) \
        X(float,          yarn_beta_fast, 32.f) \
        X(float,          yarn_beta_slow, 1.f) \
        X(float,          llama3_low_freq_factor, 1.f) \
        X(float,          llama3_high_freq_factor, 4.f) \
        X(int,            llama3_original_max_position_embeddings, 0) \
        X(MropeSection,   mrope_section, {})

    ROPE_FIELDS(TM_MEMBER)
    TM_FOR_EACH(RopeConfig, ROPE_FIELDS)

    #undef ROPE_FIELDS
};
```

- [ ] **Step 2: Add to_rope_config conversion function**

After `RopeConfig` and before `AttentionConfig`, add:

```cpp
inline RopeConfig to_rope_config(const ::turbomind::RopeParam& p) {
    using ::turbomind::RopeType;
    RopeConfig cfg;
    cfg.type                    = static_cast<int>(p.type);
    cfg.base                    = p.base;
    cfg.dim                     = p.dim;
    cfg.factor                  = p.factor;
    cfg.max_position_embeddings = p.max_position_embeddings;
    if (p.type == RopeType::kYarn) {
        cfg.yarn_attention_factor = p.yarn.attention_factor;
        cfg.yarn_beta_fast        = p.yarn.beta_fast;
        cfg.yarn_beta_slow        = p.yarn.beta_slow;
    }
    else if (p.type == RopeType::kLlama3) {
        cfg.llama3_low_freq_factor                  = p.llama3.low_freq_factor;
        cfg.llama3_high_freq_factor                 = p.llama3.high_freq_factor;
        cfg.llama3_original_max_position_embeddings = p.llama3.original_max_position_embeddings;
    }
    else if (p.type == RopeType::kMrope) {
        cfg.mrope_section = {p.mrope.section.x, p.mrope.section.y, p.mrope.section.z};
    }
    return cfg;
}
```

- [ ] **Step 3: Update ATTENTION_FIELDS — add rope, remove rope_dim and max_position_embeddings**

Replace the `ATTENTION_FIELDS` macro in `AttentionConfig` with:

```cpp
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
        X(RopeConfig, rope, {}) \
        X(int,      repeat_kv) \
        X(int,      qk_nope_dim) \
        X(float,    softmax_scale, 0.f) \
        X(bool,     use_logn_attn, false)
```

Changes: removed `X(int, rope_dim)` and `X(int, max_position_embeddings, 0)`, added `X(RopeConfig, rope, {})`.

- [ ] **Step 4: Replace max_position_embeddings_ with rope_ on AttentionWeight**

In the `AttentionWeight` class (lines 93-95), replace:

```cpp
    bool     use_logn_attn_{};
    int      max_position_embeddings_{};
```

with:

```cpp
    bool     use_logn_attn_{};
    core::RopeConfig rope_{};
```

- [ ] **Step 5: Update AttentionWeight constructor**

In `attention_weight.cc`, replace the last two initializer list entries:

```cpp
    , use_logn_attn_(cfg.use_logn_attn)
    , max_position_embeddings_(cfg.max_position_embeddings)
```

with:

```cpp
    , use_logn_attn_(cfg.use_logn_attn)
    , rope_(cfg.rope)
```

- [ ] **Step 6: Add init_rope_kernel_param overload for RopeConfig**

In `attention_weight.h`, the overload must be in `namespace turbomind` (not `core`) because `RopeKernelParam` lives in `namespace turbomind`. Place the forward declaration and function declaration after `} // namespace turbomind::core` and before `class AttentionWeight`:

```cpp
}  // namespace turbomind::core

namespace turbomind {

struct RopeKernelParam;
void init_rope_kernel_param(const core::RopeConfig& rope, RopeKernelParam& rope_kernel);

class AttentionWeight: public core::Module {
```

In `attention_weight.cc` (which is inside `namespace turbomind { }`), add the implementation at the end of the file, before `TM_MODULE_METHODS`:

```cpp
void init_rope_kernel_param(const core::RopeConfig& rope, RopeKernelParam& rope_kernel)
{
    auto rope_type = static_cast<RopeType>(rope.type);

    rope_kernel.type         = rope_type;
    rope_kernel.dim          = rope.dim;
    rope_kernel.scale_factor = -std::log2(rope.base) / rope.dim;
    if (rope_type == RopeType::kDynamic) {
        rope_kernel.inv_factor = 1.f;
    }
    else {
        rope_kernel.inv_factor = (rope.factor != 0.f) ? 1.0 / rope.factor : 1.f;
    }

    if (rope_type == RopeType::kYarn) {
        auto&        dst = rope_kernel.yarn;
        const double PI  = 3.14159265358979323846;

        auto find_correction_dim = [&](float num_rotations) {
            return (rope.dim * std::log(rope.max_position_embeddings / (num_rotations * 2 * PI)))
                   / (2 * std::log(rope.base));
        };

        auto find_correction_range = [&](float low_rot, float high_rot, float& low, float& high) {
            low  = std::floor(find_correction_dim(low_rot));
            high = std::ceil(find_correction_dim(high_rot));
            low  = std::max(low, 0.f);
            high = std::min(high, rope.dim - 1.f);
        };

        float low, high;
        find_correction_range(rope.yarn_beta_fast, rope.yarn_beta_slow, low, high);
        if (low == high) {
            high += 0.001f;
        }
        dst.ramp_inv_factor_div_2   = 1.0 / (high - low) / 2.0;
        dst.ramp_inv_factor_mul_min = 1.0 / (high - low) * low;
        dst.attention_factor        = rope.yarn_attention_factor;
    }
    else if (rope_type == RopeType::kLlama3) {
        auto& dst = rope_kernel.llama3;

        float inv_diff_freq_factor = 1.0 / (rope.llama3_high_freq_factor - rope.llama3_low_freq_factor);
        dst.alpha                  = rope.llama3_original_max_position_embeddings / (2 * 3.14159265358979323846) * inv_diff_freq_factor;
        dst.beta                   = rope.llama3_low_freq_factor * inv_diff_freq_factor;
    }
    else if (rope_type == RopeType::kMrope) {
        auto& dst     = rope_kernel.mrope;
        dst.section.x = rope.mrope_section[0] * 2;
        dst.section.y = rope.mrope_section[1] * 2 + dst.section.x;
        dst.section.z = rope.mrope_section[2] * 2 + dst.section.y;
    }
}
```

`llama_rope.h` is already transitively included through `llama_params.h`.

- [ ] **Step 7: Add bind_struct template and register RopeConfig in bind.cpp**

In `src/turbomind/python/bind.cpp`, after the `bind_config` template (after line 315), add:

```cpp
template<typename T>
void bind_struct(py::module_& m, const char* name) {
    py::class_<T> cls(m, name);
    cls.def(py::init<>());
    T::for_each([&](const char* fname, auto member_ptr) {
        cls.def_readwrite(fname, member_ptr);
    });
}
```

Then add the registration call immediately before `bind_config<turbomind::core::AttentionConfig>(m, "AttentionConfig");` (line 440):

```cpp
    bind_struct<turbomind::core::RopeConfig>(m, "RopeConfig");
```

This must come before `AttentionConfig` so the `RopeConfig` Python type exists when `def_readwrite("rope", ...)` is called.

- [ ] **Step 8: Commit**

```bash
git add src/turbomind/models/attention_weight.h \
        src/turbomind/models/attention_weight.cc \
        src/turbomind/python/bind.cpp
git commit -m "feat(config): add RopeConfig X-macro struct, bind_struct, overload, to_rope_config"
```

---

## Task 2: Update C++ consumers (UnifiedAttentionLayer, UnifiedDecoder)

**Files:**
- Modify: `src/turbomind/models/llama/unified_attention_layer.h`
- Modify: `src/turbomind/models/llama/unified_attention_layer.cc`
- Modify: `src/turbomind/models/llama/unified_decoder.cc`

- [ ] **Step 1: Update UnifiedAttentionLayer header**

In `src/turbomind/models/llama/unified_attention_layer.h`:

Change constructor signature (lines 56-64) from `const RopeParam& rope` to `const core::RopeConfig& rope`:

```cpp
    UnifiedAttentionLayer(int                     quant_policy,
                          const std::vector<int>& layer_types,
                          int                     layer_num,
                          const core::RopeConfig& rope,
                          int                     cache_block_seq_len,
                          const EngineParam&      engine,
                          const Context&          context,
                          int                     phases,
                          bool                    init);
```

Change member (line 85) from `const RopeParam rope_` to `const core::RopeConfig rope_`.

- [ ] **Step 2: Update UnifiedAttentionLayer constructor**

In `src/turbomind/models/llama/unified_attention_layer.cc`:

Change the constructor signature (line 97) from `const RopeParam& rope` to `const core::RopeConfig& rope`.

The initializer list entry `rope_{rope}` is unchanged (same name, different type).

The `init_rope_kernel_param(rope_, rope_param_)` call at line 118 now resolves to the `RopeConfig` overload added in Task 1 Step 6 — no change needed at this call site.

- [ ] **Step 3: Update init_dynamic_ntk signature**

Change line 189 from:

```cpp
static void init_dynamic_ntk(RequestCache& cache, const RopeParam& rope)
```

to:

```cpp
static void init_dynamic_ntk(RequestCache& cache, const core::RopeConfig& rope)
```

The body reads `.base`, `.factor`, `.dim`, `.max_position_embeddings` — all flat fields on `RopeConfig`. No other changes needed.

- [ ] **Step 4: Update max_position_embeddings consumer in Forward**

Change line 525 from:

```cpp
        params.max_position_embeddings = weights.max_position_embeddings_;
```

to:

```cpp
        params.max_position_embeddings = weights.rope_.max_position_embeddings;
```

- [ ] **Step 5: Update UnifiedDecoder constructor call**

In `src/turbomind/models/llama/unified_decoder.cc`, change lines 54-63 from:

```cpp
    attn_layer_ = std::make_unique<UnifiedAttentionLayer>(
        model.quant_policy,
        model.layer_types,
        model.layer_num,
        attn.rope,
        attn.cache_block_seq_len,
        engine,
        ctx,
        phases,
        (bool)moe_ffn_layer_);
```

to:

```cpp
    attn_layer_ = std::make_unique<UnifiedAttentionLayer>(
        model.quant_policy,
        model.layer_types,
        model.layer_num,
        core::to_rope_config(attn.rope),
        attn.cache_block_seq_len,
        engine,
        ctx,
        phases,
        (bool)moe_ffn_layer_);
```

`attn.rope` is still a `RopeParam` on `AttentionParam` (kept for the old YAML path). `to_rope_config()` converts it to `RopeConfig`.

Add include if not already present:

```cpp
#include "src/turbomind/models/attention_weight.h"
```

- [ ] **Step 6: Build to verify**

Run: `ninja -C build`
Expected: Clean C++ build. Python specs will fail at runtime because they reference `attn_cfg.rope_dim` — fixed in Task 3.

- [ ] **Step 7: Commit**

```bash
git add src/turbomind/models/llama/unified_attention_layer.h \
        src/turbomind/models/llama/unified_attention_layer.cc \
        src/turbomind/models/llama/unified_decoder.cc
git commit -m "refactor(cpp): replace RopeParam with RopeConfig in attention layer and decoder"
```

---

## Task 3: Update Python specs

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/utils.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py`

- [ ] **Step 1: Add rope_type_to_int helper to utils.py**

In `lmdeploy/turbomind/deploy/source_model/utils.py`, add before `parse_rope_param` (before line 31):

```python
_ROPE_TYPE_MAP = {
    'default': 1,
    'linear': 2,
    'dynamic': 3,
    'yarn': 4,
    'llama3': 5,
    'mrope': 6,
}


def rope_type_to_int(type_str: str) -> int:
    return _ROPE_TYPE_MAP[type_str]
```

Values match `RopeType` enum: `kNull=0, kDefault=1, kLinear=2, kDynamic=3, kYarn=4, kLlama3=5, kMrope=6`.

- [ ] **Step 2: Update qwen3_spec.py**

Add `rope_type_to_int` to the import from `.utils`:

```python
from .utils import _pad_inter_size, reorder_rotary_emb_linear, rope_type_to_int
```

Replace the attention config block (lines 54-59):

```python
        self._attn_cfg.rope_dim    = self._rope.dim
        self._attn_cfg.window_size = 0
        self._attn_cfg.tp_size     = engine_cfg.attn_tp_size
        self._attn_cfg.data_type   = dtype
        self._attn_cfg.softmax_scale          = self._softmax_scale
        self._attn_cfg.max_position_embeddings = self._max_position_embeddings
```

with:

```python
        self._attn_cfg.rope.type = rope_type_to_int(self._rope.type)
        self._attn_cfg.rope.base = self._rope.base
        self._attn_cfg.rope.dim  = self._rope.dim
        self._attn_cfg.rope.factor = self._rope.factor
        self._attn_cfg.rope.max_position_embeddings = self._max_position_embeddings
        self._attn_cfg.window_size = 0
        self._attn_cfg.tp_size     = engine_cfg.attn_tp_size
        self._attn_cfg.data_type   = dtype
        self._attn_cfg.softmax_scale = self._softmax_scale
```

- [ ] **Step 3: Update qwen3_5_spec.py**

Add `rope_type_to_int` to the import from `.utils`:

```python
from .utils import (_pad_inter_size, reorder_rotary_emb,
                    reorder_rotary_emb_linear, rope_type_to_int)
```

Replace lines 69-74:

```python
        self._attn_cfg.rope_dim         = self._rope.dim
        self._attn_cfg.window_size      = 0
        self._attn_cfg.tp_size          = engine_cfg.attn_tp_size
        self._attn_cfg.data_type        = dtype
        self._attn_cfg.softmax_scale          = self._softmax_scale
        self._attn_cfg.max_position_embeddings = self._max_position_embeddings
```

with:

```python
        self._attn_cfg.rope.type = rope_type_to_int(self._rope.type)
        self._attn_cfg.rope.base = self._rope.base
        self._attn_cfg.rope.dim  = self._rope.dim
        self._attn_cfg.rope.factor = self._rope.factor
        self._attn_cfg.rope.max_position_embeddings = self._max_position_embeddings
        self._attn_cfg.window_size      = 0
        self._attn_cfg.tp_size          = engine_cfg.attn_tp_size
        self._attn_cfg.data_type        = dtype
        self._attn_cfg.softmax_scale    = self._softmax_scale
```

- [ ] **Step 4: Update gpt_oss_spec.py**

Add `rope_type_to_int` to the import from `.utils`:

```python
from .utils import _pad_inter_size, reorder_rotary_emb_linear, rope_type_to_int
```

Replace lines 60-65:

```python
        self._attn_cfg.rope_dim    = self._rope.dim
        self._attn_cfg.window_size = 0
        self._attn_cfg.tp_size     = engine_cfg.attn_tp_size
        self._attn_cfg.data_type   = dtype
        self._attn_cfg.softmax_scale          = self._softmax_scale
        self._attn_cfg.max_position_embeddings = self._max_position_embeddings
```

with:

```python
        self._attn_cfg.rope.type = rope_type_to_int(self._rope.type)
        self._attn_cfg.rope.base = self._rope.base
        self._attn_cfg.rope.dim  = self._rope.dim
        self._attn_cfg.rope.factor = self._rope.factor
        self._attn_cfg.rope.max_position_embeddings = self._max_position_embeddings
        self._attn_cfg.window_size = 0
        self._attn_cfg.tp_size     = engine_cfg.attn_tp_size
        self._attn_cfg.data_type   = dtype
        self._attn_cfg.softmax_scale          = self._softmax_scale
```

- [ ] **Step 5: Update glm4_moe_lite_spec.py**

Add `rope_type_to_int` to the import from `.utils`:

```python
from .utils import _pad_inter_size, get_yarn_params, parse_rope_param, rope_type_to_int
```

Replace lines 88-93:

```python
        self._attn_cfg.rope_dim        = 0   # MLA handles rope separately
        self._attn_cfg.window_size     = 0
        self._attn_cfg.tp_size         = engine_cfg.attn_tp_size
        self._attn_cfg.data_type       = dtype
        self._attn_cfg.softmax_scale          = self._softmax_scale
        self._attn_cfg.max_position_embeddings = self._max_position_embeddings
```

with:

```python
        self._attn_cfg.rope.type = rope_type_to_int(self._rope.type)
        self._attn_cfg.rope.base = self._rope.base
        self._attn_cfg.rope.dim  = 0  # MLA handles rope separately
        self._attn_cfg.rope.factor = self._rope.factor
        self._attn_cfg.rope.max_position_embeddings = self._max_position_embeddings
        if self._rope.type == 'yarn':
            self._attn_cfg.rope.yarn_attention_factor = self._rope.attention_factor
            self._attn_cfg.rope.yarn_beta_fast = self._rope.beta_fast
            self._attn_cfg.rope.yarn_beta_slow = self._rope.beta_slow
        self._attn_cfg.window_size     = 0
        self._attn_cfg.tp_size         = engine_cfg.attn_tp_size
        self._attn_cfg.data_type       = dtype
        self._attn_cfg.softmax_scale          = self._softmax_scale
```

Note: The YaRN conditional block is needed because `glm4_moe_lite_spec` overrides `self._rope.attention_factor` and related YaRN params at lines 66-71 (before this code runs).

- [ ] **Step 6: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/utils.py \
        lmdeploy/turbomind/deploy/source_model/qwen3_spec.py \
        lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py \
        lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py \
        lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py
git commit -m "refactor(python): populate attn_cfg.rope.* from RopeParam in all specs"
```

---

## Task 4: Build and smoke test

**Files:** None (verification only)

- [ ] **Step 1: Full build**

Run: `ninja -C build`
Expected: Clean build, no errors.

- [ ] **Step 2: Check GPU availability**

Use the `get_gpu_usage` MCP tool to confirm an empty GPU.

- [ ] **Step 3: Test Qwen3-4B (non-MoE, uses qk_norm, default rope)**

Use `list_models` and `get_model_cache_path` MCP tools to find the Qwen3-4B model path.

Run: `python scripts/test_turbomind_model.py <model_path> --prompt "Hello, how are you?" --max_tokens 128`
Expected: Meaningful human-language response, 128+ tokens.

- [ ] **Step 4: Test Qwen3.5-35B-A3B-AWQ (MoE, uses attention_norm/ffn_norm)**

Use `list_models` and `get_model_cache_path` MCP tools to find the Qwen3.5 model path.

Run: `python scripts/test_turbomind_model.py <model_path> --prompt "Hello, how are you?" --max_tokens 128`
Expected: Meaningful human-language response, 128+ tokens.

- [ ] **Step 5: Commit fixes if needed**

```bash
git add -u
git commit -m "fix: address build/test issues from RopeConfig refactor"
```
