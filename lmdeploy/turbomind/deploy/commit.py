# Copyright (c) OpenMMLab. All rights reserved.
"""Backward-compatible facade re-exporting commit functions from load_context.

All names previously defined here are still importable.
New code should import directly from load_context.
"""
from __future__ import annotations

from .load_context import (
    _ATTN_TP_RULES,
    _FFN_TP_RULES,
    _LINEAR_ATTN_TP_RULES,
    _SPLIT_SIDE_TO_DIM,
    _cast_shard_for_tm,
    _commit_tensors,
    _infer_compute_dtype,
    _infer_cpp_linear_dtype,
    _torch_dtype_to_cpp,
    commit_linear,
    commit_tensor,
)
