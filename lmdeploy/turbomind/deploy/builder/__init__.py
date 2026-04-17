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
