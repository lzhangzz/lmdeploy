# Copyright (c) OpenMMLab. All rights reserved.
"""DeltaNet weight loading builder and GDN input-projection fusion helpers.

Provides ``DeltaNetBuilder`` for committing DeltaNet weights (GDN input
projections, scalar params, conv1d) and ``fuse_gdn_in_proj`` for merging
in_proj_qkv/z/b/a into a single ``in_proj_all`` with TP interleaving.
"""
from __future__ import annotations

import torch

from ..linear import Linear
from ._base import Builder, SplitSide

# ---------------------------------------------------------------------------
# TP rules for DeltaNet / linear-attention linear weights
# ---------------------------------------------------------------------------

_LINEAR_ATTN_TP_RULES: dict[str, dict] = {
    "in_proj_qkv": dict(split_side=SplitSide.OUTPUT),
    "in_proj_z":   dict(split_side=SplitSide.OUTPUT),
    "in_proj_b":   dict(split_side=SplitSide.OUTPUT),
    "in_proj_a":   dict(split_side=SplitSide.OUTPUT),
    "in_proj_all": dict(split_side=SplitSide.OUTPUT),
    "out_proj":    dict(split_side=SplitSide.INPUT),
}

# ---------------------------------------------------------------------------
# GDN input-projection keys
# ---------------------------------------------------------------------------

_GDN_IN_PROJ_KEYS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")


def _tp_interleave_tensor(t: torch.Tensor, tp: int, d: int) -> torch.Tensor:
    """Reshape last dim as [tp, per_tp] and flatten to interleave by TP rank."""
    shape = list(t.shape)
    new_shape = shape[:d] + [tp, shape[d] // tp] + shape[d + 1:]
    return t.reshape(new_shape)


def fuse_gdn_in_proj(la_linears: dict[str, Linear], tp: int,
                     qkv_split: tuple[int, int, int] | None = None) -> dict[str, Linear]:
    """Fuse GDN input projections into ``in_proj_all`` with TP interleaving.

    Pops ``in_proj_qkv``, ``in_proj_z``, ``in_proj_b``, ``in_proj_a`` from
    *la_linears* and inserts a single ``in_proj_all``.  Returns the updated
    dict (does not modify the input in place).

    For ``tp=1`` this reduces to a plain ``concat_out_dim``.

    When *qkv_split* ``(q_dim, k_dim, v_dim)`` is provided and ``tp > 1``,
    the in_proj_qkv weight is split into its Q, K, V sub-projections and
    each is TP-interleaved independently before concatenation.  This is
    necessary because Q, K, V may have different output dimensions, so a
    naive column split would mix data from different projections across
    TP ranks.
    """
    from .attention import _ensure_compatible_formats

    result = dict(la_linears)
    components: list[Linear] = []
    for key in _GDN_IN_PROJ_KEYS:
        lin = result.pop(key, None)
        if lin is not None:
            components.append(lin)
    if not components:
        return result

    group = {f"c{i}": c for i, c in enumerate(components)}
    group = _ensure_compatible_formats(group)
    components = [group[f"c{i}"] for i in range(len(group))]

    first = components[0]
    if tp <= 1:
        result["in_proj_all"] = Linear.concat_out_dim(components)
        return result

    # sub-projections Q, K, V with different output dims.  Split and
    # interleave each separately to respect head boundaries.
    if qkv_split is not None:
        q_dim, k_dim, v_dim = qkv_split
        qkv_lin = components[0]
        rest = components[1:]

        fused_tensors: dict[str, torch.Tensor] = {}
        for kind in first.tensors:
            qkv_t = qkv_lin.tensors.get(kind)
            if qkv_t is None:
                continue
            d = qkv_t.dim() - 1
            if qkv_t.dim() <= 1:
                # 1-D tensors (bias): simple split
                parts = [qkv_t[..., :q_dim], qkv_t[..., q_dim:q_dim + k_dim],
                         qkv_t[..., q_dim + k_dim:]]
                rest_ts = [lin.tensors.get(kind) for lin in rest
                           if lin.tensors.get(kind) is not None]
                fused_tensors[kind] = torch.cat(parts + rest_ts, dim=0)
                continue

            # Split QKV into Q, K, V along the output dim
            q_t = qkv_t[..., :q_dim]
            k_t = qkv_t[..., q_dim:q_dim + k_dim]
            v_t = qkv_t[..., q_dim + k_dim:]

            # TP-interleave each sub-projection independently
            interleaved = [
                _tp_interleave_tensor(q_t, tp, d),
                _tp_interleave_tensor(k_t, tp, d),
                _tp_interleave_tensor(v_t, tp, d),
            ]
            for lin in rest:
                t = lin.tensors.get(kind)
                if t is not None:
                    interleaved.append(_tp_interleave_tensor(t, tp, d))

            fused = torch.cat(interleaved, dim=d + 1)
            shape = list(fused.shape)
            final = shape[:d] + [shape[d] * shape[d + 1]] + shape[d + 2:]
            fused_tensors[kind] = fused.reshape(final)

        result["in_proj_all"] = Linear(tensors=fused_tensors, weight_format=first.weight_format,
                                       data_format=first.data_format)
        return result

    # Default path: all components have compatible output dims for naive split.
    fused_tensors: dict[str, torch.Tensor] = {}
    for kind in first.tensors:
        tensors = [lin.tensors[kind] for lin in components]
        t0 = tensors[0]
        d = t0.dim() - 1
        if t0.dim() <= 1:
            fused_tensors[kind] = torch.cat(tensors, dim=0)
            continue
        reshaped = []
        for t in tensors:
            reshaped.append(_tp_interleave_tensor(t, tp, d))
        fused = torch.cat(reshaped, dim=d + 1)
        shape = list(fused.shape)
        final = shape[:d] + [shape[d] * shape[d + 1]] + shape[d + 2:]
        fused_tensors[kind] = fused.reshape(final)

    # All components share the same format after _ensure_compatible_formats.
    result["in_proj_all"] = Linear(tensors=fused_tensors, weight_format=first.weight_format,
                                   data_format=first.data_format)
    return result


# ---------------------------------------------------------------------------
# DeltaNetBuilder -- Gated Delta Net input projections, scalar params, conv1d
# ---------------------------------------------------------------------------


class DeltaNetBuilder(Builder):
    """DeltaNet (Gated Delta Net) weight loading builder."""

    def add_input_projections(self, *, in_proj_qkv=None, in_proj_z=None,
                              in_proj_b=None, in_proj_a=None, out_proj=None,
                              qkv_split=None):
        """Fuse GDN input projections, commit all linears with TP rules.

        Internally calls ``fuse_gdn_in_proj`` to merge qkv/z/b/a into a
        single ``in_proj_all`` with TP interleaving.  Commits each resulting
        linear using ``_LINEAR_ATTN_TP_RULES`` for split-side lookup.
        """
        linears = {}
        if in_proj_qkv is not None:
            linears["in_proj_qkv"] = in_proj_qkv
        if in_proj_z is not None:
            linears["in_proj_z"] = in_proj_z
        if in_proj_b is not None:
            linears["in_proj_b"] = in_proj_b
        if in_proj_a is not None:
            linears["in_proj_a"] = in_proj_a
        if out_proj is not None:
            linears["out_proj"] = out_proj

        linears = fuse_gdn_in_proj(linears, self._tp, qkv_split)

        model_dtype = self.config.data_type
        for name, lin in linears.items():
            rule = _LINEAR_ATTN_TP_RULES.get(name, {})
            split_side = rule.get('split_side')
            self._commit_linear(name, lin, split_side=split_side,
                                model_dtype=model_dtype)

    def add_scalar_params(self, a_log=None, dt_bias=None):
        """Commit A_log and dt_bias as OUTPUT-split tensors."""
        if a_log is not None:
            self._commit_tensor("A_log", a_log, split_side=SplitSide.OUTPUT)
        if dt_bias is not None:
            self._commit_tensor("dt_bias", dt_bias, split_side=SplitSide.OUTPUT)

    def add_conv1d(self, conv1d, qkv_split=None):
        """Transpose HF layout to TM layout, TP-reshape if needed, commit.

        HF stores conv1d as [conv_dim, d_conv]; TM kernel expects
        [d_conv, conv_dim].  When tp > 1 and *qkv_split* is provided,
        the Q/K/V sub-dims are TP-interleaved.
        """
        if conv1d is None:
            return
        # Squeeze leading singleton dim if present
        if conv1d.ndim == 3 and conv1d.shape[1] == 1:
            conv1d = conv1d.squeeze(1)
        # Transpose: HF [conv_dim, d_conv] -> TM [d_conv, conv_dim]
        conv1d = conv1d.t().contiguous()
        # TP Q/K/V interleaving
        if self._tp > 1 and qkv_split is not None:
            q_dim, k_dim, v_dim = qkv_split
            d_conv = conv1d.shape[0]
            tp = self._tp
            q_part = conv1d[:, :q_dim]
            k_part = conv1d[:, q_dim:q_dim + k_dim]
            v_part = conv1d[:, q_dim + k_dim:]
            conv1d = torch.cat([
                q_part.reshape(d_conv, tp, q_dim // tp),
                k_part.reshape(d_conv, tp, k_dim // tp),
                v_part.reshape(d_conv, tp, v_dim // tp),
            ], dim=2).reshape(d_conv, -1).contiguous()
        self._commit_tensor("conv1d", conv1d, split_side=SplitSide.OUTPUT)

    def add_norm(self, norm_weight, data_type):
        """Add inline norm child."""
        self._add_norm_child("norm", norm_weight, data_type=data_type)
