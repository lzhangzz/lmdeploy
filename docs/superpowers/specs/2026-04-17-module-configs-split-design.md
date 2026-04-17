# module_configs Split Design

## Goal

Co-locate each `make_*_config` factory function with its builder class.
Delete `module_configs.py`. No behavioral changes.

## Current State

`module_configs.py` (167 lines) contains 6 factory functions and 2 C++ re-exports:

| Factory | Builder | Consumers |
|---------|---------|-----------|
| `make_linear_config` | `LinearBuilder` | `spec.py` |
| `make_norm_config` | `NormBuilder` | `spec.py`, `_base.py`, `qwen3_5_spec` |
| `make_attention_config` | `AttentionBuilder` | `qwen3_spec`, `qwen3_5_spec`, `gpt_oss_spec` |
| `make_mla_config` | `MLABuilder` | `glm4_moe_lite_spec` |
| `make_ffn_config` | `FfnBuilder` | 4 specs |
| `make_moe_config` | `MoeBuilder` | 3 specs |
| `make_deltanet_config` | `DeltaNetBuilder` | `qwen3_5_spec` |
| `DecoderLayerConfig` | `DecoderLayerBuilder` | 3 specs |
| `ModuleListConfig` | `ModuleListBuilder` | 3 specs |

All consumers import from `..module_configs` (specs) or `.module_configs` (builder internals).

## Changes

### 1. Move factories into builder files

Each `make_*_config` moves into the same file as its builder. Function bodies unchanged.

| Function | Target file |
|----------|-------------|
| `make_linear_config` | `builder/linear.py` |
| `make_norm_config` | `builder/norm.py` |
| `make_attention_config` | `builder/attention.py` |
| `make_mla_config` | `builder/mla.py` |
| `make_ffn_config` | `builder/ffn.py` |
| `make_moe_config` | `builder/moe.py` |
| `make_deltanet_config` | `builder/deltanet.py` |
| `DecoderLayerConfig` | `builder/decoder_layer.py` |
| `ModuleListConfig` | `builder/module_list.py` |

### 2. Update builder/__init__.py

Re-export all config factories and C++ configs from `builder/__init__.py`:

```python
from .attention import AttentionBuilder, make_attention_config
from .deltanet import DeltaNetBuilder, make_deltanet_config
from .decoder_layer import DecoderLayerBuilder, DecoderLayerConfig
from .ffn import FfnBuilder, fuse_ffn_linears, make_ffn_config
from .linear import LinearBuilder, make_linear_config
from .mla import MLABuilder, make_mla_config
from .moe import MoeBuilder, make_moe_config
from .module_list import ModuleListBuilder, ModuleListConfig
from .norm import NormBuilder, make_norm_config
```

Add to `__all__`: all `make_*_config` names, `DecoderLayerConfig`, `ModuleListConfig`.

### 3. Update consumer imports

| File | Before | After |
|------|--------|-------|
| `spec.py` | `from .module_configs import make_linear_config, make_norm_config` | `from .builder import make_linear_config, make_norm_config` |
| `_base.py` | `from ..module_configs import make_norm_config` | `from .norm import make_norm_config` |
| `qwen3_spec.py` | `from ..module_configs import DecoderLayerConfig, ModuleListConfig, make_attention_config, ...` | `from ..builder import DecoderLayerConfig, ModuleListConfig, make_attention_config, ...` |
| `qwen3_5_spec.py` | `from ..module_configs import ...` | `from ..builder import ...` |
| `glm4_moe_lite_spec.py` | `from ..module_configs import ...` | `from ..builder import ...` |
| `gpt_oss_spec.py` | `from ..module_configs import ...` | `from ..builder import ...` |

### 4. Delete module_configs.py

Remove `lmdeploy/turbomind/deploy/module_configs.py` entirely.

## Scope

Pure file reorganization. No function body changes. No behavioral changes.

Single commit touching:
- 7 builder files (add factory functions)
- 1 builder `__init__.py` (add re-exports)
- 6 consumer files (update imports)
- 1 file deleted (`module_configs.py`)
