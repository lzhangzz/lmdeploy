# Copyright (c) OpenMMLab. All rights reserved.
"""Attention weight loading builder and QKV fusion pipeline.

Provides ``AttentionBuilder`` for committing attention weights (QKV fusion,
O-proj, QK-norm, direct params) and pipeline functions (``dequant_mixed``,
``pad_for_tp``, ``split_output_gate``, ``fuse_qkv``) for fusing Q/K/V
Linear bundles into a single interleaved w_qkv with KV head padding and
output-gate splitting.
"""
from __future__ import annotations

import torch

from ..linear import Linear
from ._base import Builder, SplitSide, _dequant_linear, transform_tensors

# ---------------------------------------------------------------------------
# TP split rules (attention)
# ---------------------------------------------------------------------------
# Maps attention parameter names to their TP split side.  Parameters with
# SplitSide.OUTPUT are column-parallel (sharded along output dim, gathered
# via all-reduce).  Parameters with SplitSide.INPUT are row-parallel
# (sharded along input dim, gathered via all-reduce).  Keys absent from the
# table are broadcast to all TP ranks (no split needed).

_ATTN_TP_RULES: dict[str, dict] = {
    "w_qkv":     dict(split_side=SplitSide.OUTPUT),  # column-parallel
    "wo":        dict(split_side=SplitSide.INPUT),   # row-parallel
    "q_proj":    dict(split_side=SplitSide.OUTPUT),
    "q_b_proj":  dict(split_side=SplitSide.OUTPUT),
    "kv_b_proj": dict(split_side=SplitSide.OUTPUT),
}

# ---------------------------------------------------------------------------
# New pipeline functions (replacing merge_qkv_linear)
# ---------------------------------------------------------------------------


def dequant_mixed(q: Linear, k: Linear, v: Linear) -> tuple[Linear, Linear, Linear]:
    """Dequantize to trivial if formats are mixed.

    Two cases:
    1. q, k, v have different weight formats -> dequant all to trivial
    2. Some are already trivial (e.g. from reorder_rotary_emb_linear)
       -> dequant the rest so all match for fusion
    """
    names = {lin.weight_format.name for lin in (q, k, v) if lin.weight_format}
    if len(names) <= 1:
        # All same format (or all None) -- check if any are trivial while others aren't
        trivial = {lin.weight_format.name == 'trivial' for lin in (q, k, v)
                   if lin.weight_format}
        if len(trivial) <= 1:
            return q, k, v
    return _dequant_linear(q), _dequant_linear(k), _dequant_linear(v)


def _infer_heads(linear: Linear, head_dim: int) -> int:
    """Derive head count from the weight tensor's output dimension."""
    w = linear.tensors.get('weight')
    if w is None:
        return 0
    return w.size(-1) // head_dim


@transform_tensors
def _repeat_kv_heads_2d(tensor: torch.Tensor, *, n_repeat: int,
                        heads: int) -> torch.Tensor:
    per_head = tensor.size(-1) // heads
    t = tensor.view(-1, heads, per_head)
    target_heads = heads * n_repeat
    return t.repeat(1, n_repeat, 1).reshape(-1, target_heads * per_head)


def _repeat_kv_heads(linear: Linear, tp: int, head_dim: int) -> Linear:
    """Repeat KV heads to reach a TP-divisible count."""
    heads = _infer_heads(linear, head_dim)
    if heads % tp == 0:
        return linear
    target_heads = ((heads + tp - 1) // tp) * tp
    assert target_heads % heads == 0, (
        f"target_heads={target_heads} must be divisible by heads={heads}")
    return _repeat_kv_heads_2d(linear, n_repeat=target_heads // heads, heads=heads)


def pad_for_tp(q: Linear, k: Linear, v: Linear, *,
               tp: int, head_dim: int) -> tuple[Linear, Linear, Linear]:
    """Repeat KV heads to reach a TP-divisible count.

    Q is asserted to already be TP-divisible.
    Head counts are derived from actual tensor shapes, not config parameters.
    """
    assert _infer_heads(q, head_dim) % tp == 0, (
        f"Q heads={_infer_heads(q, head_dim)} must be divisible by tp={tp}")
    k = _repeat_kv_heads(k, tp, head_dim)
    v = _repeat_kv_heads(v, tp, head_dim)
    return q, k, v


def split_output_gate(q: Linear, *, head_dim: int) -> tuple[Linear, Linear]:
    """Split output gate from Q projection (Qwen3.5).

    Q's output dim is 2 * head_num * head_dim. Reshape to
    [batch, head_num, 2, head_dim], split into q_real and gate.
    """
    new_q_tensors = {}
    gate_tensors = {}

    for kind, tensor in q.tensors.items():
        head_num = tensor.size(-1) // (head_dim * 2)
        was_1d = tensor.dim() == 1
        if was_1d:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.view(tensor.size(0), head_num, 2, head_dim)
        q_real = tensor[:, :, 0, :].contiguous().reshape(-1, head_num * head_dim)
        gate = tensor[:, :, 1, :].contiguous().reshape(-1, head_num * head_dim)
        if was_1d:
            q_real = q_real.squeeze(0)
            gate = gate.squeeze(0)
        new_q_tensors[kind] = q_real
        gate_tensors[kind] = gate

    return (Linear(tensors=new_q_tensors, weight_format=q.weight_format,
                   data_format=q.data_format),
            Linear(tensors=gate_tensors, weight_format=q.weight_format,
                   data_format=q.data_format))


def fuse_qkv(q: Linear, k: Linear, v: Linear, *,
             tp: int, gate: Linear | None = None) -> Linear:
    """Fuse Q, K, V (and optionally gate) into a single w_qkv Linear.

    Concatenates output channels with TP interleaving.
    Layout per tp-shard: [Q | K | V] or [Q | K | V | Gate].
    """
    merged_tensors: dict[str, torch.Tensor] = {}

    for kind, qt in q.tensors.items():
        kt = k.tensors[kind]
        vt = v.tensors[kind]

        was_1d = qt.dim() == 1
        raw = [qt, kt, vt]
        if gate is not None:
            raw.append(gate.tensors[kind])
        if was_1d:
            raw = [t.unsqueeze(0) for t in raw]

        components = [t.view(t.size(0), tp, -1) for t in raw]
        merged = torch.cat(components, dim=-1)
        merged = merged.view(-1, merged.size(-1) * tp)
        if was_1d:
            merged = merged.squeeze(0)
        merged_tensors[kind] = merged

    return Linear(tensors=merged_tensors, weight_format=q.weight_format,
                  data_format=q.data_format)


# ---------------------------------------------------------------------------
# AttentionBuilder
# ---------------------------------------------------------------------------


class AttentionBuilder(Builder):
    """Attention weight loading builder."""

    _PARAM_TP_RULES: dict[str, SplitSide] = {
        'sinks': SplitSide.OUTPUT,
    }

    def add_qkv_proj(self, q, k, v):
        """Fuse Q/K/V into a single w_qkv with TP interleave, commit.

        Pipeline: dequant_mixed -> pad_for_tp -> [split_output_gate] -> fuse_qkv -> commit.
        RoPE permutation is done by the spec before calling this method.
        """
        q, k, v = dequant_mixed(q, k, v)
        q, k, v = pad_for_tp(q, k, v, tp=self._tp,
                              head_dim=self.config.head_dim)
        gate = None
        if self.config.attn_output_gate:
            q, gate = split_output_gate(q, head_dim=self.config.head_dim)
        merged = fuse_qkv(q, k, v, tp=self._tp, gate=gate)
        self._commit_linear('w_qkv', merged, SplitSide.OUTPUT,
                            model_dtype=self.config.data_type)

    def add_o_proj(self, o):
        """Shard along input dim, commit."""
        self._commit_linear('wo', o, SplitSide.INPUT,
                            model_dtype=self.config.data_type)

    def add_linear(self, name, linear):
        """Commit a named attention linear using TP rules.

        Looks up ``_ATTN_TP_RULES`` for the split side; absent keys are
        broadcast (no TP split).  Used for MLA projections (q_b_proj,
        kv_b_proj, o_proj) and other non-QKV attention linears.
        """
        rule = _ATTN_TP_RULES.get(name, {})
        split_side = rule.get('split_side')
        self._commit_linear(name, linear, split_side=split_side,
                            model_dtype=self.config.data_type)

    def add_qk_norm(self, q, k, *, norm_eps):
        """Create NormConfig children for q_norm, k_norm, commit tensors."""
        if q is not None:
            self._add_norm_child('q_norm', q, data_type=self.config.data_type, norm_eps=norm_eps)
        if k is not None:
            self._add_norm_child('k_norm', k, data_type=self.config.data_type, norm_eps=norm_eps)

    def add_param(self, name, tensor):
        """Commit a direct parameter. Builder determines split side."""
        split_side = self._PARAM_TP_RULES.get(name)
        self._commit_tensor(name, tensor, split_side)
