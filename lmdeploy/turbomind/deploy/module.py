# Copyright (c) OpenMMLab. All rights reserved.
"""Backward-compatible facade re-exporting from the layered split.

All names previously available from this module are still importable.
New code should import directly from the specific layer:
  - ``spec.py``      — Read & Assemble (TextModelSpec, QKV/GDN merge)
  - ``transforms.py`` — Transform (FFN fusion, TP shard extraction)
  - ``commit.py``     — Shard & Commit (alloc, copy, TP rules)
"""
from __future__ import annotations

# Suppress basedpyright "not accessed" warnings for intentional re-exports.
# fmt: off
__all__ = [
    # spec.py — Read & Assemble
    "SplitSide", "TextModelSpec",
    "_GDN_IN_PROJ_KEYS", "_block_ops_need_dequant", "_dequant_linear",
    "_ensure_compatible_formats", "_tp_interleave_tensor",
    "fuse_gdn_in_proj", "merge_qkvg_v2", "merge_qkv_linear",
    "merge_qkv_v2", "permute_v2", "permute_v2_partial",
    # transforms.py — Transform
    "_can_fuse_w1w3", "_should_fuse_silu",
    "fuse_ffn_linears",
    # commit.py — Shard & Commit
    "_ATTN_TP_RULES", "_FFN_TP_RULES", "_LINEAR_ATTN_TP_RULES",
    "_SPLIT_SIDE_TO_DIM", "_commit_tensors",
    "_cast_shard_for_tm", "_infer_compute_dtype", "_infer_cpp_linear_dtype",
    "_torch_dtype_to_cpp", "commit_linear", "commit_tensor",
]
# fmt: on

# Read & Assemble layer
from .spec import (
    SplitSide,
    TextModelSpec,
    _GDN_IN_PROJ_KEYS,
    _block_ops_need_dequant,
    _dequant_linear,
    _ensure_compatible_formats,
    _tp_interleave_tensor,
    fuse_gdn_in_proj,
    merge_qkvg_v2,
    merge_qkv_linear,
    merge_qkv_v2,
    permute_v2,
    permute_v2_partial,
)

# Transform layer
from .transforms import (
    _can_fuse_w1w3,
    _should_fuse_silu,
    fuse_ffn_linears,
)

# Shard & Commit layer
from .load_context import (
    _ATTN_TP_RULES,
    _FFN_TP_RULES,
    _LINEAR_ATTN_TP_RULES,
    _SPLIT_SIDE_TO_DIM,
    _commit_tensors,
    _cast_shard_for_tm,
    _infer_compute_dtype,
    _infer_cpp_linear_dtype,
    _torch_dtype_to_cpp,
    commit_linear,
    commit_tensor,
)
