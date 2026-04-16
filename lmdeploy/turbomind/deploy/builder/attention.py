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

from ..linear import Linear, pad_out_dim
from ._base import Builder, SplitSide, _dequant_linear

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


def pad_for_tp(q: Linear, k: Linear, v: Linear, *,
               tp: int, head_dim: int) -> tuple[Linear, Linear, Linear]:
    """Make head counts tp-divisible.

    q: pad with zero heads to reach tp-divisible count.
    kv: repeat heads to reach tp-divisible count (preserves real data).
    Also handles quantization block alignment.

    Head counts are derived from actual tensor shapes, not config parameters,
    because config may have been mutated by finalize_config (e.g. kv_head_num
    already padded to attn_tp).
    """
    def _infer_heads(linear):
        """Derive head count from the weight tensor's output dimension."""
        w = linear.tensors.get('weight')
        if w is None:
            return 0
        return w.size(-1) // head_dim

    def _adjust_linear(linear, heads, is_kv: bool):
        """Adjust one linear's head count. Pad for q, repeat for kv."""
        wfmt = linear.weight_format
        block_out = (wfmt.block_out or 0) if wfmt is not None else 0

        if heads % tp == 0:
            return linear

        target_heads = ((heads + tp - 1) // tp) * tp
        if is_kv:
            assert target_heads % heads == 0, (
                f"target_heads={target_heads} must be divisible by heads={heads}")
        new_tensors = {}

        for kind, tensor in linear.tensors.items():
            is_block_kind = kind in ("scales", "zeros") and block_out > 0

            if is_block_kind:
                # Block-scale: pad or repeat at block granularity
                # Assumes block_out >= head_dim and block_out % head_dim == 0
                # (dequant_mixed or reorder_rotary_emb_linear handles misalignment)
                assert block_out % head_dim == 0, (
                    f"block_out={block_out} must be divisible by head_dim={head_dim}")
                blocks_per_head = block_out // head_dim
                head_blocks = tensor.size(-1) // blocks_per_head
                target_blocks = target_heads * blocks_per_head
                deficit = target_blocks - head_blocks
                if deficit > 0:
                    if is_kv:
                        # Repeat: each head's blocks get repeated
                        n_repeat = target_heads // heads
                    else:
                        # Pad with identity scale=1, zero=0
                        pad_val = 1.0 if kind == "scales" else 0.0
                        padding = torch.full(
                            [*tensor.shape[:-1], deficit],
                            pad_val, dtype=tensor.dtype, device=tensor.device)
                        new_tensors[kind] = torch.cat([tensor, padding], dim=-1)
                else:
                    new_tensors[kind] = tensor
            else:
                # Per-element tensor (weight, bias, qweight)
                out_dim = tensor.dim() - 1
                per_head = tensor.size(out_dim) // heads
                target_size = target_heads * per_head
                deficit = target_size - tensor.size(out_dim)

                if deficit > 0:
                    if is_kv:
                        # Repeat: reshape to [batch, heads, head_dim] then repeat
                        if tensor.dim() == 2:
                            reshaped = tensor.view(tensor.size(0), heads, per_head)
                            n_repeat = target_heads // heads
                            reshaped = reshaped.repeat(1, 1, n_repeat)
                            new_tensors[kind] = reshaped.reshape(
                                tensor.size(0), target_heads * per_head)
                        else:
                            reshaped = tensor.view(heads, per_head)
                            n_repeat = target_heads // heads
                            reshaped = reshaped.repeat(1, n_repeat)
                            new_tensors[kind] = reshaped.reshape(target_heads * per_head)
                    else:
                        # Pad with zeros
                        new_tensors[kind] = pad_out_dim(tensor, target_size, out_dim)
                else:
                    new_tensors[kind] = tensor

        return Linear(tensors=new_tensors, weight_format=linear.weight_format,
                      data_format=linear.data_format)

    q = _adjust_linear(q, _infer_heads(q), is_kv=False)
    k = _adjust_linear(k, _infer_heads(k), is_kv=True)
    v = _adjust_linear(v, _infer_heads(v), is_kv=True)
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
        orig_shape = list(tensor.shape)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.view(tensor.size(0), head_num, 2, head_dim)
        q_real = tensor[:, :, 0, :].contiguous().reshape(-1, head_num * head_dim)
        gate = tensor[:, :, 1, :].contiguous().reshape(-1, head_num * head_dim)
        if len(orig_shape) == 1:
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
    all_kinds = sorted(set(q.tensors) | set(k.tensors) | set(v.tensors))

    for kind in all_kinds:
        qt = q.tensors.get(kind)
        kt = k.tensors.get(kind)
        vt = v.tensors.get(kind)
        if qt is None or kt is None or vt is None:
            continue

        is_2d = qt.dim() == 2

        def reshape(x):
            return x.view(x.size(0), tp, -1) if is_2d else x.view(tp, -1)

        components = [reshape(qt), reshape(kt), reshape(vt)]
        if gate is not None:
            gt = gate.tensors.get(kind)
            if gt is not None:
                components.append(reshape(gt))

        merged = torch.cat(components, dim=-1)
        merged = merged.view(-1, merged.size(-1) * tp)
        if not is_2d:
            merged.squeeze_()
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

    def add_qk_norm(self, q, k):
        """Create NormConfig children for q_norm, k_norm, commit tensors."""
        if q is not None:
            self._add_norm_child('q_norm', q, data_type=self.config.data_type)
        if k is not None:
            self._add_norm_child('k_norm', k, data_type=self.config.data_type)

    def add_param(self, name, tensor):
        """Commit a direct parameter. Builder determines split side."""
        split_side = self._PARAM_TP_RULES.get(name)
        self._commit_tensor(name, tensor, split_side)
