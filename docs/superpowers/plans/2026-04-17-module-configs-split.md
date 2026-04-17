# module_configs Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Co-locate each `make_*_config` factory function with its builder class, delete `module_configs.py`.

**Architecture:** Move 6 factory functions and 2 C++ re-exports from `module_configs.py` into their respective builder files. Update all imports. Delete the source file.

**Tech Stack:** Python, pybind11 (`_turbomind` module)

---

### Task 1: Move make_linear_config into builder/linear.py

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/linear.py`

- [ ] **Step 1: Add `import _turbomind as _tm` and `make_linear_config` to builder/linear.py**

Add at the top of the file (after existing imports):

```python
import _turbomind as _tm
```

Add before the `LinearBuilder` class:

```python
def make_linear_config(*, input_dim, output_dim, data_type, has_bias=False):
    cfg = _tm.LinearConfig()
    cfg.input_dim = input_dim
    cfg.output_dim = output_dim
    cfg.data_type = data_type
    cfg.has_bias = has_bias
    return cfg
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/linear.py
git commit -m "refactor(builder): move make_linear_config into linear.py"
```

---

### Task 2: Move make_norm_config into builder/norm.py

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/norm.py`

- [ ] **Step 1: Add `import _turbomind as _tm` and `make_norm_config` to builder/norm.py**

Current file starts with:
```python
import torch
import _turbomind as _tm
```

It already imports `_tm`. Add before the `NormBuilder` class:

```python
def make_norm_config(*, dim, data_type):
    cfg = _tm.NormConfig()
    cfg.dim = dim
    cfg.data_type = data_type
    return cfg
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/norm.py
git commit -m "refactor(builder): move make_norm_config into norm.py"
```

---

### Task 3: Move make_attention_config into builder/attention.py

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/attention.py`

- [ ] **Step 1: Add `import _turbomind as _tm` and `make_attention_config` to builder/attention.py**

Add `import _turbomind as _tm` to the imports at the top.

Add before the `_ATTN_TP_RULES` dict:

```python
def make_attention_config(mc, *, tp_size, tp_rank=0, dtype, window_size=0,
                         rope_dim=0):
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
    return cfg
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/attention.py
git commit -m "refactor(builder): move make_attention_config into attention.py"
```

---

### Task 4: Move make_mla_config into builder/mla.py

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/mla.py`

- [ ] **Step 1: Add `import _turbomind as _tm` and `make_mla_config` to builder/mla.py**

Add `import _turbomind as _tm` to the imports at the top.

Add before the `fold_kv_b` function:

```python
def make_mla_config(mc, *, tp_size, tp_rank=0, dtype, window_size=0,
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
    return cfg
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/mla.py
git commit -m "refactor(builder): move make_mla_config into mla.py"
```

---

### Task 5: Move make_ffn_config into builder/ffn.py

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/ffn.py`

- [ ] **Step 1: Add `import _turbomind as _tm` and `make_ffn_config` to builder/ffn.py**

Add `import _turbomind as _tm` to the imports at the top.

Add before the `_should_fuse_silu` function:

```python
def make_ffn_config(mc, *, tp_size, tp_rank=0, dtype, act_type,
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
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/ffn.py
git commit -m "refactor(builder): move make_ffn_config into ffn.py"
```

---

### Task 6: Move make_moe_config into builder/moe.py

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/moe.py`

- [ ] **Step 1: Add `import _turbomind as _tm` and `make_moe_config` to builder/moe.py**

Add `import _turbomind as _tm` to the imports at the top.

Add before the `MoeBuilder` class:

```python
def make_moe_config(mc, *, layer_id, tp_size, tp_rank=0, dtype,
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
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/moe.py
git commit -m "refactor(builder): move make_moe_config into moe.py"
```

---

### Task 7: Move make_deltanet_config into builder/deltanet.py

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/deltanet.py`

- [ ] **Step 1: Add `import _turbomind as _tm` and `make_deltanet_config` to builder/deltanet.py**

Add `import _turbomind as _tm` to the imports at the top.

Add before the `tp_interleave_tensor` function:

```python
def make_deltanet_config(mc, *, tp_size, tp_rank=0, dtype):
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

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/deltanet.py
git commit -m "refactor(builder): move make_deltanet_config into deltanet.py"
```

---

### Task 8: Move C++ re-exports into trivial builder files

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/decoder_layer.py`
- Modify: `lmdeploy/turbomind/deploy/builder/module_list.py`

- [ ] **Step 1: Add `DecoderLayerConfig` re-export to decoder_layer.py**

Current content:
```python
from ._base import Builder


class DecoderLayerBuilder(Builder):
    """Pure container builder for decoder layers."""
    pass
```

Change to:
```python
import _turbomind as _tm

from ._base import Builder

DecoderLayerConfig = _tm.DecoderLayerConfig


class DecoderLayerBuilder(Builder):
    """Pure container builder for decoder layers."""
    pass
```

- [ ] **Step 2: Add `ModuleListConfig` re-export to module_list.py**

Current content:
```python
from ._base import Builder


class ModuleListBuilder(Builder):
    """Builder for ModuleList containers."""
    pass
```

Change to:
```python
import _turbomind as _tm

from ._base import Builder

ModuleListConfig = _tm.ModuleListConfig


class ModuleListBuilder(Builder):
    """Builder for ModuleList containers."""
    pass
```

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/decoder_layer.py lmdeploy/turbomind/deploy/builder/module_list.py
git commit -m "refactor(builder): move DecoderLayerConfig and ModuleListConfig into their builders"
```

---

### Task 9: Update builder/__init__.py re-exports

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/__init__.py`

- [ ] **Step 1: Update imports and __all__**

Replace the entire file with:

```python
# Copyright (c) OpenMMLab. All rights reserved.
"""Builder sub-package — spec-driven module loading for TurboMind.

Each builder wraps N GPU handles for a single logical module and distributes
module creation, child binding, and weight commits across all GPUs with bound
TP configuration.
"""
from __future__ import annotations

from ._base import (Builder, TextModelBuilder, SplitSide,
                    _cpp_dtype, _act_type_id, _torch_dtype_to_cpp)
from .attention import AttentionBuilder, make_attention_config
from .deltanet import DeltaNetBuilder, make_deltanet_config
from .decoder_layer import DecoderLayerBuilder, DecoderLayerConfig
from .ffn import FfnBuilder, fuse_ffn_linears, make_ffn_config
from .linear import LinearBuilder, make_linear_config
from .mla import MLABuilder, make_mla_config
from .moe import MoeBuilder, make_moe_config
from .module_list import ModuleListBuilder, ModuleListConfig
from .norm import NormBuilder, make_norm_config

__all__ = [
    # Base
    'Builder', 'TextModelBuilder', 'SplitSide',
    '_cpp_dtype', '_act_type_id', '_torch_dtype_to_cpp',
    # Builders
    'AttentionBuilder', 'FfnBuilder', 'MoeBuilder',
    'DeltaNetBuilder', 'MLABuilder',
    'DecoderLayerBuilder', 'ModuleListBuilder',
    'NormBuilder', 'LinearBuilder',
    # Config factories
    'make_linear_config', 'make_norm_config',
    'make_attention_config', 'make_mla_config',
    'make_ffn_config', 'make_moe_config',
    'make_deltanet_config',
    # C++ config re-exports
    'DecoderLayerConfig', 'ModuleListConfig',
    # Helper functions
    'fuse_ffn_linears',
]
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/__init__.py
git commit -m "refactor(builder): add config factory re-exports to __init__.py"
```

---

### Task 10: Update consumer imports and delete module_configs.py

**Files:**
- Modify: `lmdeploy/turbomind/deploy/spec.py`
- Modify: `lmdeploy/turbomind/deploy/builder/_base.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py`
- Delete: `lmdeploy/turbomind/deploy/module_configs.py`

- [ ] **Step 1: Update spec.py**

Change:
```python
from .module_configs import make_linear_config, make_norm_config
```
To:
```python
from .builder import make_linear_config, make_norm_config
```

- [ ] **Step 2: Update builder/_base.py**

Change:
```python
from ..module_configs import make_norm_config
```
To:
```python
from .norm import make_norm_config
```

- [ ] **Step 3: Update source_model/qwen3_spec.py**

Change:
```python
from ..module_configs import (
    DecoderLayerConfig, ModuleListConfig,
    make_attention_config, make_ffn_config, make_moe_config,
)
```
To:
```python
from ..builder import (
    DecoderLayerConfig, ModuleListConfig,
    make_attention_config, make_ffn_config, make_moe_config,
)
```

- [ ] **Step 4: Update source_model/qwen3_5_spec.py**

Change:
```python
from ..module_configs import (
    DecoderLayerConfig, ModuleListConfig,
    make_attention_config, make_deltanet_config, make_ffn_config,
    make_moe_config, make_norm_config,
)
```
To:
```python
from ..builder import (
    DecoderLayerConfig, ModuleListConfig,
    make_attention_config, make_deltanet_config, make_ffn_config,
    make_moe_config, make_norm_config,
)
```

- [ ] **Step 5: Update source_model/glm4_moe_lite_spec.py**

Change:
```python
from ..module_configs import (
    DecoderLayerConfig, ModuleListConfig,
    make_ffn_config, make_mla_config, make_moe_config,
)
```
To:
```python
from ..builder import (
    DecoderLayerConfig, ModuleListConfig,
    make_ffn_config, make_mla_config, make_moe_config,
)
```

- [ ] **Step 6: Update source_model/gpt_oss_spec.py**

Change:
```python
from ..module_configs import (
    DecoderLayerConfig, ModuleListConfig,
    make_attention_config, make_ffn_config, make_moe_config,
)
```
To:
```python
from ..builder import (
    DecoderLayerConfig, ModuleListConfig,
    make_attention_config, make_ffn_config, make_moe_config,
)
```

- [ ] **Step 7: Delete module_configs.py**

```bash
git rm lmdeploy/turbomind/deploy/module_configs.py
```

- [ ] **Step 8: Verify no remaining references to module_configs**

```bash
grep -r 'module_configs' lmdeploy/turbomind/deploy/
```

Expected: no output.

- [ ] **Step 9: Test with Qwen3.5-35B-A3B TP=1**

```bash
cd build && python scripts/test_turbomind_model.py Qwen3.5-35B-A3B --tp 1 --prompt "Hello" --max_new_tokens 128
```

Expected: meaningful response, no import errors.

- [ ] **Step 10: Commit**

```bash
git add -A lmdeploy/turbomind/deploy/
git commit -m "refactor(builder): switch all consumers to builder imports, delete module_configs.py"
```

---

### Task 11: Squash into single commit (optional)

If desired, squash all 10 commits into one:

```bash
git rebase -i HEAD~10
```

Squash tasks 1-10 into a single commit with message:

```
refactor(builder): co-locate config factories with builders, delete module_configs.py

Move all make_*_config functions from module_configs.py into their
respective builder files. Update all consumer imports. Delete module_configs.py.
```
