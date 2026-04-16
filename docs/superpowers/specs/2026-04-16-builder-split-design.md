# Builder Module Split

Dismantle the monolithic `builder.py` (1393 lines) into a `builder/` sub-package with one file per builder class. No API changes for consumers — `__init__.py` re-exports everything.

## Problem

`builder.py` mixes 8 distinct concerns: dtype mappings, TP split rules, tensor commit infrastructure, FFN fusion helpers, the Builder base class, trivial container builders, QKV/RoPE/GDN merge functions, and 5 specialized builders (Attention, FFN, MoE, DeltaNet, MLA). At 1393 lines it's too large to hold in context and the concerns evolve independently.

## Solution

Replace `builder.py` (file) with `builder/` (package). Each builder class gets its own file alongside its co-located helpers and TP split rules.

### File layout

```
deploy/
  builder/                      # replaces builder.py
    __init__.py                 # re-exports public API
    _base.py                    # Builder, TextModelBuilder, SplitSide, dtype maps, commit methods, _commit_tensors
    decoder_layer.py            # DecoderLayerBuilder
    module_list.py              # ModuleListBuilder
    norm.py                     # NormBuilder
    linear.py                   # LinearBuilder
    attention.py                # AttentionBuilder, _ATTN_TP_RULES, merge_qkv_linear, _reorder_rotary_emb, _merge_qkv, _merge_qkvg, block-ops helpers
    mla.py                      # MLABuilder, _MLA_TP_RULES
    deltanet.py                 # DeltaNetBuilder, _LINEAR_ATTN_TP_RULES, fuse_gdn_in_proj, _tp_interleave_tensor
    ffn.py                      # FfnBuilder, fuse_ffn_linears, _should_fuse_silu, _can_fuse_w1w3
    moe.py                      # MoeBuilder
```

### `_base.py` contents

Shared infrastructure that all builders depend on:

- `SplitSide` enum
- Dtype maps: `_STR_TO_DTYPE`, `_TORCH_TO_CPP`, `_FP8_DTYPES`, `_SPLIT_SIDE_TO_DIM`
- Dtype helpers: `_cpp_dtype()`, `_act_type_id()`, `_torch_dtype_to_cpp()`, `_cast_shard_for_tm()`, `_infer_cpp_linear_dtype()`, `_infer_compute_dtype()`
- `_commit_tensors()` free function
- `Builder` base class: `__init__`, `__setattr__`, `__setitem__`, `_ensure_handles`, `_commit_linear`, `_commit_tensor`, `_add_norm_child`
- `TextModelBuilder`

### Specialized builder files

Each imports `Builder`, `SplitSide`, and commit helpers from `_base`:

**`attention.py`** (~280 lines):
- `AttentionBuilder` (add_qkv_proj, add_o_proj, add_linear, add_qk_norm, add_param)
- `_ATTN_TP_RULES`
- `merge_qkv_linear()`, `_reorder_rotary_emb()`, `_merge_qkv()`, `_merge_qkvg()`
- `_dequant_linear()`, `_ensure_compatible_formats()`, `_block_ops_need_dequant()`

**`mla.py`** (~130 lines):
- `MLABuilder` (add_projections, add_norms, _fold_and_pad, _fold_and_pad_hf)
- `_MLA_TP_RULES`

**`deltanet.py`** (~120 lines):
- `DeltaNetBuilder` (add_input_projections, add_scalar_params, add_conv1d, add_norm)
- `_LINEAR_ATTN_TP_RULES`
- `fuse_gdn_in_proj()`, `_tp_interleave_tensor()`, `_GDN_IN_PROJ_KEYS`

**`ffn.py`** (~80 lines):
- `FfnBuilder` (add_ffn)
- `fuse_ffn_linears()`, `_should_fuse_silu()`, `_can_fuse_w1w3()`

**`moe.py`** (~15 lines):
- `MoeBuilder` (add_gate, add_param)

### Trivial builder files

Each is 5-15 lines, importing only `Builder` from `_base`:

**`decoder_layer.py`**: `DecoderLayerBuilder(Builder)` — pass
**`module_list.py`**: `ModuleListBuilder(Builder)` — pass
**`norm.py`**: `NormBuilder(Builder)` — set_weight method
**`linear.py`**: `LinearBuilder(Builder)` — set_weight method

### `__init__.py`

Re-exports the full public API so existing import paths work unchanged:

```python
from ._base import (Builder, TextModelBuilder, SplitSide,
                    _cpp_dtype, _act_type_id, _torch_dtype_to_cpp)
from .attention import AttentionBuilder, merge_qkv_linear
from .ffn import FfnBuilder, fuse_ffn_linears
from .moe import MoeBuilder
from .deltanet import DeltaNetBuilder, fuse_gdn_in_proj
from .mla import MLABuilder
from .decoder_layer import DecoderLayerBuilder
from .module_list import ModuleListBuilder
from .norm import NormBuilder
from .linear import LinearBuilder
```

### Import paths

No spec files change. `from ..builder import AttentionBuilder` resolves through `__init__.py` the same way it resolved from the old `builder.py` file. Python treats packages and modules identically at the import site.

## What does NOT change

- No spec file changes (qwen3_spec.py, qwen3_5_spec.py, gpt_oss_spec.py, glm4_moe_lite_spec.py, spec.py)
- No code logic changes — pure file reorganization
- No C++ changes
- `load_context.py` has its own `_commit_tensors` copy — not affected
