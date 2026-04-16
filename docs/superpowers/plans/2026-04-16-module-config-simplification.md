# Module Config Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate Python config dataclasses, factory functions construct C++ pybind structs directly.

**Architecture:** Each `from_model_config()` classmethod becomes a plain factory function returning a C++ struct. Each `to_cpp()` method disappears — the Builder already holds a C++ struct. The `for_rank()` / `replace()` pattern becomes `cfg.clone()` + mutate. `SpecAttnConfig` and `configure()` are deleted; specs read values from `self._mc` directly.

**Tech Stack:** Python 3, pybind11, C++ (X-macro config structs)

---

### Task 1: Add fields to C++ AttentionConfig

The C++ `AttentionConfig` struct is missing 3 fields that the Python dataclass carries for weight-processing logic. Add them so the C++ struct can carry all needed data.

**Files:**
- Modify: `src/turbomind/core/module_config.h`

- [ ] **Step 1: Add fields to ATTENTION_FIELDS X-macro**

In `src/turbomind/core/module_config.h`, add 3 fields to the `ATTENTION_FIELDS` list, before the closing of the macro:

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
        X(int,      rope_dim) \
        X(int,      repeat_kv) \
        X(int,      qk_nope_dim)
```

- [ ] **Step 2: Build**

Run: `cd build && ninja`
Expected: Clean build with no errors.

- [ ] **Step 3: Verify new fields are accessible from Python**

```bash
cd /data/lmdeploy-modeling && python -c "
import _turbomind as _tm
cfg = _tm.AttentionConfig()
print('rope_dim:', cfg.rope_dim)
print('repeat_kv:', cfg.repeat_kv)
print('qk_nope_dim:', cfg.qk_nope_dim)
cfg.rope_dim = 64
cfg.repeat_kv = 2
cfg.qk_nope_dim = 512
print('rope_dim:', cfg.rope_dim)
print('repeat_kv:', cfg.repeat_kv)
print('qk_nope_dim:', cfg.qk_nope_dim)
print('clone:', cfg.clone().rope_dim)
"
```

Expected: All fields read/write correctly, clone works.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/core/module_config.h
git commit -m "feat(config): add rope_dim, repeat_kv, qk_nope_dim to C++ AttentionConfig"
```

---

### Task 2: Rewrite module_configs.py

Replace all dataclass definitions with factory functions that construct C++ structs directly.

**Files:**
- Rewrite: `lmdeploy/turbomind/deploy/module_configs.py`

- [ ] **Step 1: Rewrite the file**

```python
# Copyright (c) OpenMMLab. All rights reserved.
"""Factory functions for C++ module config structs.

Each factory constructs a pybind-bound C++ config struct directly from
ModelConfig (or explicit parameters), eliminating the intermediate Python
dataclass layer.
"""
from __future__ import annotations

import _turbomind as _tm

# Re-export simple C++ configs that need no transformation
from _turbomind import DecoderLayerConfig, ModuleListConfig


# ---------------------------------------------------------------------------
# Simple configs (few fields, no ModelConfig dependency)
# ---------------------------------------------------------------------------


def make_linear_config(*, input_dim, output_dim, data_type, has_bias=False):
    cfg = _tm.LinearConfig()
    cfg.input_dim = input_dim
    cfg.output_dim = output_dim
    cfg.data_type = data_type
    cfg.has_bias = has_bias
    return cfg


def make_norm_config(*, dim, data_type):
    cfg = _tm.NormConfig()
    cfg.dim = dim
    cfg.data_type = data_type
    return cfg


# ---------------------------------------------------------------------------
# Attention configs
# ---------------------------------------------------------------------------


def make_attention_config(mc, *, tp_size, tp_rank, dtype, window_size,
                         rope_dim=0, repeat_kv=0):
    """Build C++ AttentionConfig from ModelConfig."""
    cfg = _tm.AttentionConfig()
    cfg.hidden_dim = mc.hidden_units
    cfg.head_dim = mc.size_per_head
    cfg.head_num = mc.head_num
    cfg.kv_head_num = mc.kv_head_num
    cfg.kv_lora_rank = mc.kv_lora_rank or 0
    cfg.q_lora_rank = mc.q_lora_rank or 0
    cfg.qk_rope_dim = mc.qk_rope_dim or 0
    cfg.v_head_dim = mc.v_head_dim or 0
    cfg.has_bias = mc.attn_bias
    cfg.qk_norm = mc.qk_norm
    cfg.tp_size = tp_size
    cfg.tp_rank = tp_rank
    cfg.data_type = dtype
    cfg.window_size = window_size
    cfg.attn_sink = mc.attn_sink
    cfg.attn_output_gate = mc.attn_output_gate
    cfg.rope_dim = rope_dim
    cfg.repeat_kv = repeat_kv
    return cfg


def make_mla_config(mc, *, tp_size, tp_rank, dtype, window_size,
                    qk_nope_dim=0):
    """Build C++ AttentionConfig for MLA from ModelConfig."""
    qk_rope_dim = mc.qk_rope_dim or 0
    kv_lora_rank = mc.kv_lora_rank or 0
    v_head_dim = mc.v_head_dim or 0
    size_per_head = qk_nope_dim + qk_rope_dim
    if kv_lora_rank and kv_lora_rank != qk_nope_dim:
        size_per_head = kv_lora_rank + qk_rope_dim
        v_head_dim = kv_lora_rank

    cfg = _tm.AttentionConfig()
    cfg.hidden_dim = mc.hidden_units
    cfg.head_dim = size_per_head
    cfg.head_num = mc.head_num
    cfg.kv_head_num = mc.kv_head_num
    cfg.kv_lora_rank = kv_lora_rank
    cfg.q_lora_rank = mc.q_lora_rank or 0
    cfg.qk_rope_dim = qk_rope_dim
    cfg.qk_nope_dim = qk_nope_dim
    cfg.v_head_dim = v_head_dim
    cfg.tp_size = tp_size
    cfg.tp_rank = tp_rank
    cfg.data_type = dtype
    cfg.window_size = window_size
    cfg.has_bias = False
    cfg.qk_norm = False
    cfg.attn_sink = False
    cfg.attn_output_gate = False
    cfg.rope_dim = 0
    cfg.repeat_kv = 0
    return cfg


# ---------------------------------------------------------------------------
# FFN / MoE configs
# ---------------------------------------------------------------------------


def make_ffn_config(mc, *, tp_size, tp_rank, dtype, act_type,
                    fuse_silu, inter_size=None, fused_moe=False):
    """Build C++ FfnConfig from ModelConfig."""
    cfg = _tm.FfnConfig()
    cfg.hidden_dim = mc.hidden_units
    cfg.inter_size = inter_size if inter_size is not None else mc.inter_size
    cfg.has_bias = mc.mlp_bias
    cfg.tp_size = tp_size
    cfg.tp_rank = tp_rank
    cfg.data_type = dtype
    cfg.act_type = act_type
    cfg.fuse_silu = fuse_silu
    cfg.fused_moe = fused_moe
    return cfg


def make_moe_config(mc, *, layer_id, tp_size, tp_rank, dtype,
                    act_type, fuse_silu, expert_num):
    """Build C++ MoeConfig from ModelConfig."""
    cfg = _tm.MoeConfig()
    cfg.layer_id = layer_id
    cfg.method = 1  # kFused
    cfg.experts_per_token = mc.experts_per_token
    cfg.inter_size = mc.expert_inter_size or 0
    cfg.norm_topk_prob = mc.norm_topk_prob
    cfg.shared_gate = mc.moe_shared_gate
    cfg.routed_scale = float(mc.routed_scale)
    cfg.router_bias = mc.expert_router_bias
    cfg.topk_group = mc.topk_group
    cfg.topk_method = mc.topk_method
    cfg.n_group = mc.moe_group_num
    cfg.scoring_func = mc.scoring_func
    cfg.router_n_groups = max(0, mc.router_n_groups)
    cfg.expert_num = expert_num
    cfg.hidden_dim = mc.hidden_units
    cfg.mlp_bias = mc.mlp_bias
    cfg.data_type = dtype
    cfg.tp_size = tp_size
    cfg.tp_rank = tp_rank
    cfg.act_type = act_type
    cfg.fuse_silu = fuse_silu
    return cfg


# ---------------------------------------------------------------------------
# DeltaNet config
# ---------------------------------------------------------------------------


def make_deltanet_config(mc, *, tp_size, tp_rank, dtype):
    """Build C++ DeltaNetConfig from ModelConfig."""
    cfg = _tm.DeltaNetConfig()
    cfg.hidden_dim = mc.hidden_units
    cfg.num_k_heads = mc.linear_num_key_heads
    cfg.num_v_heads = mc.linear_num_value_heads
    cfg.key_head_dim = mc.linear_key_head_dim
    cfg.value_head_dim = mc.linear_value_head_dim
    cfg.d_conv = mc.linear_conv_kernel_dim or 4
    cfg.has_bias = bool(mc.attn_bias)
    cfg.tp_size = tp_size
    cfg.tp_rank = tp_rank
    cfg.data_type = dtype
    return cfg
```

- [ ] **Step 2: Verify import works**

```bash
cd /data/lmdeploy-modeling && python -c "
from lmdeploy.turbomind.deploy.module_configs import (
    make_attention_config, make_ffn_config, make_moe_config,
    make_mla_config, make_deltanet_config,
    make_linear_config, make_norm_config,
    ModuleListConfig, DecoderLayerConfig,
)
print('All imports OK')
"
```

Expected: No import errors.

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/module_configs.py
git commit -m "refactor(config): replace dataclasses with C++ struct factory functions"
```

---

### Task 3: Update builder.py

Update `_ensure_handles` to use `clone()` instead of `for_rank().to_cpp()`, update `_add_norm_child` to use C++ struct directly, simplify `FfnBuilder.add_ffn` config mutation, update MLABuilder to use `head_dim` instead of `size_per_head`.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder.py`

- [ ] **Step 1: Remove `from dataclasses import replace` and update NormConfig import**

Replace the import section at the top of the file:

```python
# Remove this line:
from dataclasses import replace

# Change this import:
from .module_configs import NormConfig
# To:
from .module_configs import make_norm_config
```

- [ ] **Step 2: Update `_ensure_handles`**

Replace the `_ensure_handles` method body (lines 409-421):

```python
    def _ensure_handles(self):
        """Lazily create C++ module handles on first access."""
        if self._handles_created:
            return
        handles = []
        for i, ctx in enumerate(self._contexts):
            with ctx:
                if self._tp > 1:
                    cfg = self.config.clone()
                    cfg.tp_rank = self._ranks[i]
                else:
                    cfg = self.config
                handle = _tm.create_module(cfg)
                handles.append(handle)
        object.__setattr__(self, '_handles', handles)
        object.__setattr__(self, '_handles_created', True)
```

- [ ] **Step 3: Update `_add_norm_child`**

Replace line 590-594:

```python
        norm_cfg = make_norm_config(dim=tensor.shape[-1], data_type=data_type)

        for i, handle in enumerate(self._handles):
            with self._contexts[i]:
                child = handle.create_child(name, norm_cfg)
```

(Note: `norm_cfg` is already a C++ struct, no `.to_cpp()` needed.)

- [ ] **Step 4: Simplify `FfnBuilder.add_ffn` config mutation**

Replace line 1138:

```python
        # Before:
        self.config = replace(self.config, fuse_silu=fused_silu)
        # After:
        self.config.fuse_silu = fused_silu
```

- [ ] **Step 5: Update MLABuilder to use `head_dim` instead of `size_per_head`**

In `_fold_and_pad_hf` (around line 1325-1333), replace:

```python
        cfg = self.config
        head_num = cfg.head_num
        qk_rope_dim = cfg.qk_rope_dim
        qk_nope_dim = cfg.qk_nope_dim
        kv_lora_rank = cfg.kv_lora_rank
        v_head_dim = cfg.v_head_dim
        size_per_head = cfg.head_dim  # was: cfg.size_per_head
```

Also update the comment in `add_projections` (around line 1281) to note that `self.config.data_type` reads directly from the C++ struct.

- [ ] **Step 6: Verify builder.py parses**

```bash
cd /data/lmdeploy-modeling && python -c "
from lmdeploy.turbomind.deploy.builder import Builder, AttentionBuilder, FfnBuilder, MLABuilder
print('Builder imports OK')
"
```

Expected: No errors.

- [ ] **Step 7: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder.py
git commit -m "refactor(builder): use C++ structs directly, remove to_cpp/for_rank/replace"
```

---

### Task 4: Update spec.py and text_model_loader.py

Remove `configure()`, `SpecAttnConfig`, and redundant `_` attributes from `TextModelSpec`. Remove the `configure()` call from `TextModelLoader`.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/spec.py`
- Modify: `lmdeploy/turbomind/deploy/text_model_loader.py`

- [ ] **Step 1: Update spec.py imports**

Replace line 10:

```python
# Before:
from .module_configs import LinearConfig, NormConfig, SpecAttnConfig
# After:
from .module_configs import make_linear_config, make_norm_config
```

- [ ] **Step 2: Remove redundant class attributes and `configure()` from TextModelSpec**

Delete lines 22-31 (the `# Default values` block and `_linear_qkv_split`):

```python
    # DELETE these lines:
    # Default values for configure() fields; overwritten by configure().
    _attn_tp: int = 1
    _permute_qk: bool = True
    _repeat_kv: int = 0
    _head_dim: int = 0
    _rope_dim: int = 0
    _attn_output_gate: bool = False
    _kv_head_num: int = 0
    # TODO: there dont belong here
    _linear_qkv_split: tuple[int, int, int] | None = None
```

Add back only the ones still needed:

```python
    _attn_tp: int = 1
    _repeat_kv: int = 0
    _rope_dim: int = 0
    _linear_qkv_split: tuple[int, int, int] | None = None
```

Note: `_attn_tp`, `_repeat_kv`, and `_rope_dim` are still injected by `TextModelLoader`. `_rope_dim` is derived from `attention_config.rope_param.dim` and can differ from `mc.size_per_head` for models with partial rotary embeddings (e.g. Qwen3.5). `_linear_qkv_split` is set per-spec in `__init__`.

- [ ] **Step 3: Delete the `configure()` method**

Delete lines 45-53:

```python
    # DELETE:
    def configure(self, cfg: SpecAttnConfig):
        """Set TP and model parameters. Called by TextModelLoader."""
        ...
```

- [ ] **Step 4: Update `token_embeds` to use factory function**

Replace the `LinearConfig(...)` call (line 65-67):

```python
        cfg = make_linear_config(input_dim=padded_vocab,
                                 output_dim=mc.hidden_units // tp,
                                 data_type=dtype)
```

- [ ] **Step 5: Update `output_norm` to use factory function**

Replace line 74:

```python
        cfg = make_norm_config(dim=self._mc.hidden_units, data_type=self._cpp_dtype())
```

- [ ] **Step 6: Update `lm_head` to use factory function**

Replace lines 87-89:

```python
        cfg = make_linear_config(input_dim=mc.hidden_units,
                                 output_dim=padded_vocab // tp,
                                 data_type=dtype)
```

- [ ] **Step 7: Update `norm` to use factory function**

Replace line 96:

```python
        cfg = make_norm_config(dim=self._mc.hidden_units, data_type=self._cpp_dtype())
```

- [ ] **Step 8: Update text_model_loader.py**

Delete the `SpecAttnConfig` import (line 7):

```python
# DELETE:
from .module_configs import SpecAttnConfig
```

Delete the `spec.configure(SpecAttnConfig(...))` call (lines 51-60):

```python
# DELETE these lines:
        mc = self.model.model_config
        rope_param = self.model.attention_config.rope_param
        spec.configure(SpecAttnConfig(
            tp=self.attn_tp,
            repeat_kv=self.model.repeat_kv,
            head_dim=mc.size_per_head,
            rope_dim=rope_param.dim if rope_param else mc.size_per_head,
            output_gate=mc.attn_output_gate,
            kv_head_num=mc.kv_head_num,
        ))
```

The `__call__` method should now look like:

```python
    def __call__(self, layer: int, spec: 'TextModelSpec'):
        spec._contexts = self._contexts
        spec._root_handles = self._root_handles
        spec._mc = self.model.model_config
        spec._attn_tp = self.attn_tp
        spec._attn_cp = self.model.attn_cp_size
        spec._mlp_tp = self.mlp_tp
        spec._attn_ranks = self._attn_ranks
        spec._mlp_ranks = self._mlp_ranks
        spec._repeat_kv = self.model.repeat_kv
        rope_param = self.model.attention_config.rope_param
        spec._rope_dim = rope_param.dim if rope_param else self.model.model_config.size_per_head
        spec.model()
        return 1
```

- [ ] **Step 9: Verify imports**

```bash
cd /data/lmdeploy-modeling && python -c "
from lmdeploy.turbomind.deploy.spec import TextModelSpec
from lmdeploy.turbomind.deploy.text_model_loader import TextModelLoader
print('Imports OK')
"
```

- [ ] **Step 10: Commit**

```bash
git add lmdeploy/turbomind/deploy/spec.py lmdeploy/turbomind/deploy/text_model_loader.py
git commit -m "refactor(spec): remove SpecAttnConfig, configure(), use factory functions"
```

---

### Task 5: Update qwen3_spec.py

Replace config dataclass usage with factory functions. Replace `self._rope_dim`, `self._head_dim`, `self._permute_qk` with direct reads.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py`

- [ ] **Step 1: Update imports**

Replace lines 17-20:

```python
# Before:
from ..module_configs import (
    AttentionConfig, DecoderLayerConfig, FfnConfig,
    ModuleListConfig, MoeConfig,
)
# After:
from ..module_configs import (
    DecoderLayerConfig, ModuleListConfig,
    make_attention_config, make_ffn_config, make_moe_config,
)
```

- [ ] **Step 2: Update `attn()` method**

Replace lines 72-79:

```python
        attn_cfg = make_attention_config(
            mc, tp_size=tp, tp_rank=0, dtype=dtype,
            window_size=window_size,
            rope_dim=self._rope_dim,
            repeat_kv=self._repeat_kv)
        attn = AttentionBuilder(attn_cfg, self._contexts,
                                tp=tp, ranks=self._attn_ranks)
```

Note: `rope_dim` is set to `self._rope_dim` (injected by TextModelLoader from `attention_config.rope_param.dim`). `permute_qk` is always True — no longer passed.

Replace lines 87-89 (remove `_permute_qk` conditional):

```python
        # Before:
        if self._permute_qk:
            q_norm = reorder_rotary_emb(q_norm, self._head_dim, self._rope_dim)
            k_norm = reorder_rotary_emb(k_norm, self._head_dim, self._rope_dim)
        # After:
        q_norm = reorder_rotary_emb(q_norm, mc.size_per_head, self._rope_dim)
        k_norm = reorder_rotary_emb(k_norm, mc.size_per_head, self._rope_dim)
```

- [ ] **Step 3: Update `ffn()` method**

Replace lines 109-114:

```python
        ffn_cfg = make_ffn_config(
            mc, tp_size=tp, tp_rank=0, dtype=dtype,
            act_type=_act_type_id(mc.activation_type),
            fuse_silu=False, inter_size=inter_size,
            fused_moe=fused_moe)
```

- [ ] **Step 4: Update `moe()` method**

Replace lines 132-136:

```python
        moe_cfg = make_moe_config(
            mc, layer_id=layer, tp_size=tp, tp_rank=0, dtype=dtype,
            act_type=_act_type_id(mc.activation_type),
            fuse_silu=True, expert_num=expert_num)
```

- [ ] **Step 5: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/qwen3_spec.py
git commit -m "refactor(qwen3): use factory functions, read rope_dim from config"
```

---

### Task 6: Update qwen3_5_spec.py

Same pattern as Task 5, plus NormConfig usage in `output_norm` and `norm`.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py`

- [ ] **Step 1: Update imports**

Replace lines 24-27:

```python
# Before:
from ..module_configs import (
    AttentionConfig, DecoderLayerConfig, DeltaNetConfig, FfnConfig,
    ModuleListConfig, MoeConfig, NormConfig,
)
# After:
from ..module_configs import (
    DecoderLayerConfig, ModuleListConfig,
    make_attention_config, make_deltanet_config, make_ffn_config,
    make_moe_config, make_norm_config,
)
```

- [ ] **Step 2: Update `output_norm()` and `norm()`**

Replace `NormConfig(dim=..., data_type=...)` with `make_norm_config(dim=..., data_type=...)` in both methods (lines 144 and 151).

- [ ] **Step 3: Update `attn()` method**

Replace lines 172-179:

```python
        attn_cfg = make_attention_config(
            mc, tp_size=tp, tp_rank=0, dtype=dtype,
            window_size=window_size,
            rope_dim=self._rope_dim,
            repeat_kv=self._repeat_kv)
```

Replace lines 187-189 (remove `_permute_qk` conditional):

```python
        q_norm = reorder_rotary_emb(q_norm, mc.size_per_head, self._rope_dim)
        k_norm = reorder_rotary_emb(k_norm, mc.size_per_head, self._rope_dim)
```

- [ ] **Step 4: Update `linear_attn()` method**

Replace lines 200-201:

```python
        dn_cfg = make_deltanet_config(
            mc, tp_size=tp, tp_rank=0, dtype=dtype)
```

- [ ] **Step 5: Update `ffn()` method**

Replace lines 240-245:

```python
        ffn_cfg = make_ffn_config(
            mc, tp_size=tp, tp_rank=0, dtype=dtype,
            act_type=_act_type_id(mc.activation_type),
            fuse_silu=False, inter_size=inter_size,
            fused_moe=fused_moe)
```

- [ ] **Step 6: Update `moe()` method**

Replace lines 263-267:

```python
        moe_cfg = make_moe_config(
            mc, layer_id=layer, tp_size=tp, tp_rank=0, dtype=dtype,
            act_type=_act_type_id(mc.activation_type),
            fuse_silu=True, expert_num=expert_num)
```

- [ ] **Step 7: Update `_packed_moe_expert_indexed()`**

Replace lines 361-366:

```python
        ffn_cfg = make_ffn_config(
            mc, tp_size=tp, tp_rank=0, dtype=dtype,
            act_type=_act_type_id(mc.activation_type),
            fuse_silu=False, inter_size=inter_size,
            fused_moe=True)
```

- [ ] **Step 8: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py
git commit -m "refactor(qwen3.5): use factory functions, read rope_dim from config"
```

---

### Task 7: Update gpt_oss_spec.py

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py`

- [ ] **Step 1: Update imports**

Replace lines 28-31:

```python
# Before:
from ..module_configs import (
    AttentionConfig, DecoderLayerConfig, FfnConfig,
    ModuleListConfig, MoeConfig,
)
# After:
from ..module_configs import (
    DecoderLayerConfig, ModuleListConfig,
    make_attention_config, make_ffn_config, make_moe_config,
)
```

- [ ] **Step 2: Update `attn()` method**

Replace lines 90-97:

```python
        attn_cfg = make_attention_config(
            mc, tp_size=tp, tp_rank=0, dtype=dtype,
            window_size=window_size,
            rope_dim=self._rope_dim,
            repeat_kv=self._repeat_kv)
```

- [ ] **Step 3: Update `ffn()` method**

Replace lines 122-127:

```python
        ffn_cfg = make_ffn_config(
            mc, tp_size=tp, tp_rank=0, dtype=dtype,
            act_type=_act_type_id(mc.activation_type),
            fuse_silu=False, inter_size=inter_size,
            fused_moe=fused_moe)
```

- [ ] **Step 4: Update `moe()` method**

Replace lines 145-149:

```python
        moe_cfg = make_moe_config(
            mc, layer_id=layer, tp_size=tp, tp_rank=0, dtype=dtype,
            act_type=_act_type_id(mc.activation_type),
            fuse_silu=True, expert_num=expert_num)
```

- [ ] **Step 5: Update `_moe_expert_ffn()`**

Replace lines 244-249:

```python
        ffn_cfg = make_ffn_config(
            mc, tp_size=tp, tp_rank=0, dtype=dtype,
            act_type=_act_type_id(mc.activation_type),
            fuse_silu=False, inter_size=expert_inter,
            fused_moe=True)
```

- [ ] **Step 6: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py
git commit -m "refactor(gpt-oss): use factory functions, read rope_dim from config"
```

---

### Task 8: Update glm4_moe_lite_spec.py

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py`

- [ ] **Step 1: Update imports**

Replace lines 22-25:

```python
# Before:
from ..module_configs import (
    DecoderLayerConfig, FfnConfig,
    MLAConfig, ModuleListConfig, MoeConfig,
)
# After:
from ..module_configs import (
    DecoderLayerConfig, ModuleListConfig,
    make_ffn_config, make_mla_config, make_moe_config,
)
```

- [ ] **Step 2: Update `attn()` method**

Replace lines 72-76:

```python
        mla_cfg = make_mla_config(
            mc, tp_size=tp, tp_rank=0, dtype=dtype, window_size=0,
            qk_nope_dim=cfg['qk_nope_head_dim'])
```

- [ ] **Step 3: Update `ffn()` method**

Replace lines 108-113:

```python
        ffn_cfg = make_ffn_config(
            mc, tp_size=tp, tp_rank=0, dtype=dtype,
            act_type=_act_type_id(mc.activation_type),
            fuse_silu=False, inter_size=inter_size,
            fused_moe=fused_moe)
```

- [ ] **Step 4: Update `moe()` method**

Replace lines 131-135:

```python
        moe_cfg = make_moe_config(
            mc, layer_id=layer, tp_size=tp, tp_rank=0, dtype=dtype,
            act_type=_act_type_id(mc.activation_type),
            fuse_silu=True, expert_num=expert_num)
```

- [ ] **Step 5: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py
git commit -m "refactor(glm4): use factory functions"
```

---

### Task 9: Test all models

Run the test script for each supported model to verify nothing is broken.

**Files:**
- No code changes

- [ ] **Step 1: Check GPU availability**

Use the `get_gpu_usage` MCP tool. Ensure at least one GPU is free.

- [ ] **Step 2: Test Qwen3**

```bash
cd /data/lmdeploy-modeling && python scripts/test_turbomind_model.py <qwen3-model-path>
```

Verify the response contains meaningful human words (not gibberish).

- [ ] **Step 3: Test GLM-4 MoE Lite**

```bash
cd /data/lmdeploy-modeling && python scripts/test_turbomind_model.py <glm4-model-path>
```

- [ ] **Step 4: Test Qwen3.5 (if available)**

```bash
cd /data/lmdeploy-modeling && python scripts/test_turbomind_model.py <qwen3_5-model-path>
```

- [ ] **Step 5: Commit if all tests pass**

No code changes to commit for this task. All commits were made in previous tasks.
