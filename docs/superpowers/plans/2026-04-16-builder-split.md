# Builder Module Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the monolithic `builder.py` (1393 lines) into a `builder/` sub-package with one file per builder class.

**Architecture:** Replace `builder.py` (file) with `builder/` (package). Each builder class gets its own file alongside its co-located helpers and TP split rules. `__init__.py` re-exports everything so no spec imports change.

**Tech Stack:** Python package, pure file reorganization (no logic changes).

---

## File Structure

```
lmdeploy/turbomind/deploy/
  builder/                        # new package, replaces builder.py
    __init__.py                   # re-exports public API
    _base.py                      # Builder, TextModelBuilder, SplitSide, dtype maps, commit methods
    decoder_layer.py              # DecoderLayerBuilder
    module_list.py                # ModuleListBuilder
    norm.py                       # NormBuilder
    linear.py                     # LinearBuilder
    attention.py                  # AttentionBuilder + QKV merge helpers + TP rules
    mla.py                        # MLABuilder + TP rules
    deltanet.py                   # DeltaNetBuilder + GDN fusion + TP rules
    ffn.py                        # FfnBuilder + fusion helpers
    moe.py                        # MoeBuilder
```

## Source line ranges in builder.py

| Target file | Lines | Contents |
|---|---|---|
| `_base.py` | 1-18 (imports), 25-33 (SplitSide), 40-62 (dtype maps), 70-134 (dtype helpers), 143-165 (TP rules — moved to specialized files), 173-241 (_commit_tensors), 331-605 (Builder class), 612-629 (TextModelBuilder) | Note: TP rules go to their respective builder files, not _base |
| `decoder_layer.py` | 636-641 | DecoderLayerBuilder |
| `module_list.py` | 649-654 | ModuleListBuilder |
| `norm.py` | 662-681 | NormBuilder |
| `linear.py` | 688-706 | LinearBuilder |
| `attention.py` | 143-149 (_ATTN_TP_RULES), 713-754 (_reorder_rotary_emb), 757-785 (_merge_qkv/_merge_qkvg), 788-803 (_dequant_linear/_ensure_compatible_formats), 808-940 (_block_ops_need_dequant/merge_qkv_linear), 1060-1110 (AttentionBuilder) | |
| `deltanet.py` | 151-158 (_LINEAR_ATTN_TP_RULES), 805-806 (_GDN_IN_PROJ_KEYS), 943-947 (_tp_interleave_tensor), 950-1052 (fuse_gdn_in_proj), 1182-1254 (DeltaNetBuilder) | |
| `ffn.py` | 249-323 (fuse helpers), 1117-1155 (FfnBuilder) | |
| `mla.py` | 160-165 (_MLA_TP_RULES), 1261-1393 (MLABuilder) | |
| `moe.py` | 1162-1175 (MoeBuilder) | |

---

### Task 1: Scaffold builder/ package

**Files:**
- Create: `lmdeploy/turbomind/deploy/builder/__init__.py`
- Rename: `lmdeploy/turbomind/deploy/builder.py` → `lmdeploy/turbomind/deploy/builder/_old.py`

This task creates the package and keeps all imports working through a temporary `_old.py` facade.

- [ ] **Step 1: Create builder/ directory and rename builder.py**

```bash
cd /data/lmdeploy-modeling/lmdeploy/turbomind/deploy
mkdir -p builder
git mv builder.py builder/_old.py
```

- [ ] **Step 2: Create `builder/__init__.py` that re-exports from `_old.py`**

Create `lmdeploy/turbomind/deploy/builder/__init__.py`:

```python
# Copyright (c) OpenMMLab. All rights reserved.
"""Temporary facade — re-exports everything from _old.py during migration."""
from ._old import *  # noqa: F401,F403
```

- [ ] **Step 3: Verify imports still work**

```bash
cd /data/lmdeploy-modeling
python -c "from lmdeploy.turbomind.deploy.builder import AttentionBuilder, FfnBuilder, MoeBuilder, DeltaNetBuilder, MLABuilder, _act_type_id; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/
git commit -m "refactor: scaffold builder/ package, keep _old.py facade"
```

---

### Task 2: Create `_base.py` with Builder base class and infrastructure

**Files:**
- Create: `lmdeploy/turbomind/deploy/builder/_base.py`

Extract from `builder/_old.py`:
- Lines 1-18: Copyright, imports (adapt relative paths: `.kind_map` → `..kind_map`, `.linear` → `..linear`, `.module_configs` → `..module_configs`)
- Lines 25-33: `SplitSide` enum
- Lines 40-62: `_STR_TO_DTYPE`, `_TORCH_TO_CPP`, `_FP8_DTYPES`, `_SPLIT_SIDE_TO_DIM`
- Lines 70-134: `_cpp_dtype`, `_act_type_id`, `_torch_dtype_to_cpp`, `_cast_sharm_for_tm`, `_infer_cpp_linear_dtype`, `_infer_compute_dtype`
- Lines 173-241: `_commit_tensors`
- Lines 331-423: `Builder` class (init, setattr, setitem, _ensure_handles)
- Lines 428-605: Builder commit methods (_commit_linear, _commit_tensor, _add_norm_child)
- Lines 612-629: `TextModelBuilder`

- [ ] **Step 1: Create `_base.py`**

Create `lmdeploy/turbomind/deploy/builder/_base.py` with the copyright header and module docstring, then copy the sections listed above from `_old.py`. Change the relative imports:

```python
from __future__ import annotations

import enum
import torch

import _turbomind as _tm

from ..linear import Linear
from ..module_configs import make_norm_config
```

All other code is copied verbatim from `_old.py`. The TP split rules (`_ATTN_TP_RULES`, `_LINEAR_ATTN_TP_RULES`, `_MLA_TP_RULES`) are NOT included in `_base.py` — they go to their respective builder files.

- [ ] **Step 2: Verify `_base.py` imports cleanly**

```bash
cd /data/lmdeploy-modeling
python -c "from lmdeploy.turbomind.deploy.builder._base import Builder, TextModelBuilder, SplitSide, _cpp_dtype, _act_type_id, _commit_tensors; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/_base.py
git commit -m "refactor(builder): extract _base.py with Builder base class and infrastructure"
```

---

### Task 3: Create trivial builder files

**Files:**
- Create: `lmdeploy/turbomind/deploy/builder/decoder_layer.py`
- Create: `lmdeploy/turbomind/deploy/builder/module_list.py`
- Create: `lmdeploy/turbomind/deploy/builder/norm.py`
- Create: `lmdeploy/turbomind/deploy/builder/linear.py`

Each is a small file importing `Builder` from `_base`.

- [ ] **Step 1: Create `decoder_layer.py`**

```python
# Copyright (c) OpenMMLab. All rights reserved.
from ._base import Builder


class DecoderLayerBuilder(Builder):
    """Pure container builder for decoder layers."""
    pass
```

- [ ] **Step 2: Create `module_list.py`**

```python
# Copyright (c) OpenMMLab. All rights reserved.
from ._base import Builder


class ModuleListBuilder(Builder):
    """Builder for ModuleList containers."""
    pass
```

- [ ] **Step 3: Create `norm.py`**

Copy from `_old.py` lines 662-681. The imports needed:

```python
# Copyright (c) OpenMMLab. All rights reserved.
import torch

import _turbomind as _tm

from ._base import Builder, _torch_dtype_to_cpp, _cast_shard_for_tm


class NormBuilder(Builder):
    """Builder for a single norm weight module."""

    def set_weight(self, tensor: torch.Tensor):
        """Commit the norm weight tensor to all GPU handles."""
        self._ensure_handles()
        if tensor is None:
            return
        for i, handle in enumerate(self._handles):
            with self._contexts[i]:
                shard = tensor
                if not shard.is_cuda:
                    shard = shard.cuda(0).contiguous()
                elif not shard.is_contiguous():
                    shard = shard.contiguous()
                cpp_dtype = _torch_dtype_to_cpp(shard.dtype)
                dst = handle.param('weight').alloc(list(shard.shape), cpp_dtype)
                shard = _cast_shard_for_tm(shard, dst)
                dst.copy_from(shard)
```

- [ ] **Step 4: Create `linear.py`**

Copy from `_old.py` lines 688-706:

```python
# Copyright (c) OpenMMLab. All rights reserved.
from ._base import Builder, SplitSide


class LinearBuilder(Builder):
    """Builder for standalone linear layers (embeddings, lm_head).

    Wraps a C++ LinearWeight module. Use ``set_weight()`` to commit
    the weight tensor.
    """

    def set_weight(self, tensor, split_side=None):
        """Commit the weight tensor to all GPU handles."""
        self._commit_tensor('weight', tensor, split_side)
```

- [ ] **Step 5: Verify all trivial builders import**

```bash
cd /data/lmdeploy-modeling
python -c "
from lmdeploy.turbomind.deploy.builder.decoder_layer import DecoderLayerBuilder
from lmdeploy.turbomind.deploy.builder.module_list import ModuleListBuilder
from lmdeploy.turbomind.deploy.builder.norm import NormBuilder
from lmdeploy.turbomind.deploy.builder.linear import LinearBuilder
print('OK')
"
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/decoder_layer.py \
       lmdeploy/turbomind/deploy/builder/module_list.py \
       lmdeploy/turbomind/deploy/builder/norm.py \
       lmdeploy/turbomind/deploy/builder/linear.py
git commit -m "refactor(builder): extract trivial builder files"
```

---

### Task 4: Create `attention.py`

**Files:**
- Create: `lmdeploy/turbomind/deploy/builder/attention.py`

Extract from `_old.py`:
- Lines 143-149: `_ATTN_TP_RULES`
- Lines 713-754: `_reorder_rotary_emb`
- Lines 757-785: `_merge_qkv`, `_merge_qkvg`
- Lines 788-803: `_dequant_linear`, `_ensure_compatible_formats`
- Lines 808-940: `_block_ops_need_dequant`, `merge_qkv_linear`
- Lines 1060-1110: `AttentionBuilder`

Imports needed:

```python
from __future__ import annotations

import torch

from ..kind_map import TRIVIAL_FORMAT
from ..linear import Linear
from ._base import Builder, SplitSide
```

- [ ] **Step 1: Create `attention.py`**

Copy all sections listed above from `_old.py` verbatim into the new file. Add the copyright header and the imports shown. Place `_ATTN_TP_RULES` at the top (after imports), then the helper functions, then `AttentionBuilder` at the bottom.

- [ ] **Step 2: Verify import**

```bash
cd /data/lmdeploy-modeling
python -c "from lmdeploy.turbomind.deploy.builder.attention import AttentionBuilder, merge_qkv_linear; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/attention.py
git commit -m "refactor(builder): extract AttentionBuilder and QKV merge helpers"
```

---

### Task 5: Create `deltanet.py`

**Files:**
- Create: `lmdeploy/turbomind/deploy/builder/deltanet.py`

Extract from `_old.py`:
- Lines 151-158: `_LINEAR_ATTN_TP_RULES`
- Lines 805-806: `_GDN_IN_PROJ_KEYS`
- Lines 943-947: `_tp_interleave_tensor`
- Lines 950-1052: `fuse_gdn_in_proj`
- Lines 1182-1254: `DeltaNetBuilder`

Imports needed:

```python
from __future__ import annotations

import torch

from ..linear import Linear
from ._base import Builder, SplitSide
```

- [ ] **Step 1: Create `deltanet.py`**

Copy all sections listed above from `_old.py` verbatim. Add copyright header and imports.

- [ ] **Step 2: Verify import**

```bash
cd /data/lmdeploy-modeling
python -c "from lmdeploy.turbomind.deploy.builder.deltanet import DeltaNetBuilder, fuse_gdn_in_proj; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/deltanet.py
git commit -m "refactor(builder): extract DeltaNetBuilder and GDN fusion helpers"
```

---

### Task 6: Create `ffn.py`

**Files:**
- Create: `lmdeploy/turbomind/deploy/builder/ffn.py`

Extract from `_old.py`:
- Lines 249-323: `_should_fuse_silu`, `_can_fuse_w1w3`, `fuse_ffn_linears`
- Lines 1117-1155: `FfnBuilder`

Imports needed:

```python
from __future__ import annotations

from ..linear import Linear, chunk_linears as _chunk_linears, interleave_linears as _interleave_linears
from ._base import Builder, SplitSide
```

Note: `_chunk_linears` and `_interleave_linears` are used by `fuse_ffn_linears`. The import `import torch` is not needed — `fuse_ffn_linears` doesn't call torch directly.

- [ ] **Step 1: Create `ffn.py`**

Copy all sections listed above from `_old.py` verbatim. Add copyright header and imports.

- [ ] **Step 2: Verify import**

```bash
cd /data/lmdeploy-modeling
python -c "from lmdeploy.turbomind.deploy.builder.ffn import FfnBuilder, fuse_ffn_linears; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/ffn.py
git commit -m "refactor(builder): extract FfnBuilder and FFN fusion helpers"
```

---

### Task 7: Create `mla.py` and `moe.py`

**Files:**
- Create: `lmdeploy/turbomind/deploy/builder/mla.py`
- Create: `lmdeploy/turbomind/deploy/builder/moe.py`

**`mla.py`** — extract from `_old.py`:
- Lines 160-165: `_MLA_TP_RULES`
- Lines 1261-1393: `MLABuilder`

Imports:
```python
from __future__ import annotations

import torch

from ..linear import Linear
from ._base import Builder, SplitSide
```

**`moe.py`** — extract from `_old.py`:
- Lines 1162-1175: `MoeBuilder`

Imports:
```python
from __future__ import annotations

from ._base import Builder, SplitSide
```

- [ ] **Step 1: Create `mla.py`**

Copy sections from `_old.py` verbatim. Add copyright header and imports.

- [ ] **Step 2: Create `moe.py`**

Copy `MoeBuilder` from `_old.py` verbatim. Add copyright header and imports.

- [ ] **Step 3: Verify imports**

```bash
cd /data/lmdeploy-modeling
python -c "
from lmdeploy.turbomind.deploy.builder.mla import MLABuilder
from lmdeploy.turbomind.deploy.builder.moe import MoeBuilder
print('OK')
"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/mla.py \
       lmdeploy/turbomind/deploy/builder/moe.py
git commit -m "refactor(builder): extract MLABuilder and MoeBuilder"
```

---

### Task 8: Switch `__init__.py` to re-export from split files, delete `_old.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/__init__.py`
- Delete: `lmdeploy/turbomind/deploy/builder/_old.py`

- [ ] **Step 1: Replace `__init__.py` with proper re-exports**

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
from .attention import AttentionBuilder, merge_qkv_linear
from .deltanet import DeltaNetBuilder, fuse_gdn_in_proj
from .decoder_layer import DecoderLayerBuilder
from .ffn import FfnBuilder, fuse_ffn_linears
from .linear import LinearBuilder
from .mla import MLABuilder
from .moe import MoeBuilder
from .module_list import ModuleListBuilder
from .norm import NormBuilder

__all__ = [
    # Base
    'Builder', 'TextModelBuilder', 'SplitSide',
    '_cpp_dtype', '_act_type_id', '_torch_dtype_to_cpp',
    # Builders
    'AttentionBuilder', 'FfnBuilder', 'MoeBuilder',
    'DeltaNetBuilder', 'MLABuilder',
    'DecoderLayerBuilder', 'ModuleListBuilder',
    'NormBuilder', 'LinearBuilder',
    # Helper functions
    'merge_qkv_linear', 'fuse_gdn_in_proj', 'fuse_ffn_linears',
]
```

- [ ] **Step 2: Delete `_old.py`**

```bash
git rm lmdeploy/turbomind/deploy/builder/_old.py
```

- [ ] **Step 3: Verify the package imports match the old module**

```bash
cd /data/lmdeploy-modeling
python -c "
from lmdeploy.turbomind.deploy.builder import (
    Builder, TextModelBuilder, SplitSide,
    _cpp_dtype, _act_type_id, _torch_dtype_to_cpp,
    AttentionBuilder, FfnBuilder, MoeBuilder,
    DeltaNetBuilder, MLABuilder,
    DecoderLayerBuilder, ModuleListBuilder,
    NormBuilder, LinearBuilder,
    merge_qkv_linear, fuse_gdn_in_proj, fuse_ffn_linears,
)
print('All imports OK')
"
```

Expected: `All imports OK`

- [ ] **Step 4: Verify spec imports still work**

```bash
python -c "
from lmdeploy.turbomind.deploy.spec import TextModelSpec
from lmdeploy.turbomind.deploy.source_model.qwen3_spec import Qwen3Spec
from lmdeploy.turbomind.deploy.source_model.qwen3_5_spec import Qwen3_5Spec
from lmdeploy.turbomind.deploy.source_model.gpt_oss_spec import GptOssSpec
from lmdeploy.turbomind.deploy.source_model.glm4_moe_lite_spec import Glm4MoeLiteSpec
print('All spec imports OK')
"
```

Expected: `All spec imports OK`

- [ ] **Step 5: Commit**

```bash
git add -A lmdeploy/turbomind/deploy/builder/
git commit -m "refactor(builder): switch __init__.py to split files, delete _old.py"
```

---

### Task 9: Test all model architectures

**Files:** None (verification only)

Test each model architecture to verify the split didn't break anything. Each model exercises different builder types:

| Model | GPU | Builders exercised |
|---|---|---|
| Qwen3-4B | check available | AttentionBuilder, FfnBuilder |
| GLM-4.7-Flash | check available | MLABuilder, FfnBuilder, MoeBuilder |
| Qwen3.5-35B-A3B-AWQ | check available | AttentionBuilder, FfnBuilder, MoeBuilder, DeltaNetBuilder |
| gpt-oss-20b | check available | AttentionBuilder, MoeBuilder |

- [ ] **Step 1: Check GPU availability**

```bash
python -c "
from lmdeploy.turbomind.deploy.builder import _act_type_id
" && echo "Module loads cleanly"
```

Use `get_gpu_usage` MCP tool to find 2 free GPUs.

- [ ] **Step 2: Test Qwen3-4B (dense attention)**

Run with at least 128 output tokens. Verify the response is meaningful English text.

- [ ] **Step 3: Test GLM-4.7-Flash (MLA + MoE)**

Run with at least 128 output tokens. Verify the response is meaningful.

- [ ] **Step 4: Test Qwen3.5-35B-A3B-AWQ (linear attention + MoE + DeltaNet)**

Run with at least 128 output tokens. Verify the response is meaningful.

- [ ] **Step 5: Test gpt-oss-20b (packed MoE)**

Run with at least 128 output tokens. Verify the response is meaningful.

- [ ] **Step 6: Commit (if any fixups were needed)**

Only if bugs were found and fixed during testing.
