# Copyright (c) OpenMMLab. All rights reserved.
"""Builder sub-package — spec-driven module loading for TurboMind.

Each builder wraps N GPU handles for a single logical module and distributes
module creation, child binding, and weight commits across all GPUs with bound
TP configuration.
"""
from __future__ import annotations

from ._base import (Builder, TextModelBuilder, SplitSide,
                    _cpp_dtype, _act_type_id, _torch_dtype_to_cpp)
from .attention import AttentionBuilder
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
    'fuse_gdn_in_proj', 'fuse_ffn_linears',
]
