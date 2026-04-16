# Copyright (c) OpenMMLab. All rights reserved.
"""Attention weight loading builder and QKV merge helpers.

Provides ``AttentionBuilder`` for committing attention weights (QKV fusion,
O-proj, QK-norm, direct params) and ``merge_qkv_linear`` for fusing Q/K/V
Linear bundles into a single interleaved w_qkv with RoPE permutation, KV
head repetition, and output-gate splitting.
"""
from __future__ import annotations

import torch

from ..linear import Linear, pad_out_dim
from ._base import Builder, SplitSide, _dequant_linear, _ensure_compatible_formats, _block_ops_need_dequant

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
               tp: int, head_dim: int,
               q_heads: int, kv_heads: int) -> tuple[Linear, Linear, Linear]:
    """Make head counts tp-divisible.

    q: pad with zero heads to reach tp-divisible count.
    kv: repeat heads to reach tp-divisible count (preserves real data).
    Also handles quantization block alignment.
    """
    def _adjust_linear(linear, heads, is_kv: bool):
        """Adjust one linear's head count. Pad for q, repeat for kv."""
        wfmt = linear.weight_format
        block_out = (wfmt.block_out or 0) if wfmt is not None else 0

        if heads % tp == 0:
            return linear

        target_heads = ((heads + tp - 1) // tp) * tp
        new_tensors = {}

        for kind, tensor in linear.tensors.items():
            is_block_kind = kind in ("scales", "zeros") and block_out > 0

            if is_block_kind:
                # Block-scale: pad or repeat at block granularity
                blocks_per_head = block_out // head_dim
                head_blocks = tensor.size(-1) // blocks_per_head
                target_blocks = target_heads * blocks_per_head
                deficit = target_blocks - head_blocks
                if deficit > 0:
                    if is_kv:
                        # Repeat: each head's blocks get repeated
                        n_repeat = target_heads // heads
                        new_tensors[kind] = tensor.repeat_interleave(n_repeat, dim=-1)
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

    q = _adjust_linear(q, q_heads, is_kv=False)
    k = _adjust_linear(k, kv_heads, is_kv=True)
    v = _adjust_linear(v, kv_heads, is_kv=True)
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
# QKV merge / RoPE permutation helpers
# ---------------------------------------------------------------------------


def _reorder_rotary_emb(x: torch.Tensor, head_dim: int, rope_dim: int):
    """Interleave rotary embedding layout for TurboMind's RoPE kernel.

    Combines the former ``permute_v2`` (full permutation when
    ``rope_dim == head_dim``) and ``permute_v2_partial`` (partial
    permutation when ``rope_dim < head_dim``) into a single function.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor whose last dimension is ``head_num * head_dim``.
    head_dim : int
        Full head dimension.
    rope_dim : int
        Rotary embedding dimension (``<= head_dim``).
    """
    assert x.size(-1) > 1
    assert rope_dim % 2 == 0, f'rope_dim must be even, got {rope_dim}'
    assert rope_dim <= head_dim, f'rope_dim ({rope_dim}) must be <= head_dim ({head_dim})'
    output_dims = x.size(-1)
    assert output_dims % head_dim == 0, (f'output_dims ({output_dims}) must be divisible by '
                                          f'head_dim ({head_dim})')
    head_num = output_dims // head_dim
    orig_shape = x.shape
    if x.dim() == 1:
        x = x.unsqueeze(0)

    x = x.view(x.size(0), head_num, head_dim)

    if rope_dim < head_dim:
        # Partial permutation: only interleave the rotary portion
        rotary = x[:, :, :rope_dim]
        passthrough = x[:, :, rope_dim:]
        rotary = rotary.view(x.size(0), head_num, 2, rope_dim // 2).transpose(2, 3).contiguous()
        rotary = rotary.view(x.size(0), head_num, rope_dim)
        x = torch.cat([rotary, passthrough], dim=-1)
    else:
        # Full permutation: interleave all elements
        x = x.view(x.size(0), head_num, 2, head_dim // 2).transpose(2, 3).contiguous()
        x = x.view(x.size(0), head_num, head_dim)

    return x.reshape(orig_shape)


def _merge_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tp: int):
    """Merge Q, K, V with TP interleaving.

    Contract: x.size(-1) is output dims.
    """
    def reshape(x):
        return x.view(x.size(0), tp, -1) if q.dim() == 2 else x.view(tp, -1)

    qkv = torch.cat(tuple(map(reshape, (q, k, v))), dim=-1)
    qkv = qkv.view(-1, qkv.size(-1) * tp)
    if q.dim() == 1:
        qkv.squeeze_()
    return qkv


def _merge_qkvg(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                gate: torch.Tensor, tp: int):
    """Merge Q, K, V, and Gate with gate appended after V.

    Layout per tp-shard: [Q | K | V | Gate].
    """
    def reshape(x):
        return x.view(x.size(0), tp, -1) if q.dim() == 2 else x.view(tp, -1)

    qkvg = torch.cat(tuple(map(reshape, (q, k, v, gate))), dim=-1)
    qkvg = qkvg.view(-1, qkvg.size(-1) * tp)
    if q.dim() == 1:
        qkvg.squeeze_()
    return qkvg


def merge_qkv_linear(
    q: Linear,
    k: Linear,
    v: Linear,
    tp: int,
    head_dim: int,
    rope_dim: int,
    permute_qk: bool = True,
    attn_output_gate: bool = False,
    repeat_kv: int = 0,
    kv_head_num: int = 0,
) -> Linear:
    """Merge Q/K/V ``Linear`` bundles into a single interleaved ``w_qkv`` Linear.

    Applies RoPE permutation, KV head repetition, and output-gate splitting
    as required, then interleaves components for TP.  Returns a ``Linear``
    in TM layout ``[in, out]``.

    Per-kind routing
    ----------------
    - ``weight`` / ``qweight`` / ``bias``: per-element structure along the
      output dim; all transforms applied when ``tensor.size(-1) % head_dim == 0``.
    - ``scales`` / ``zeros`` with ``block_out > 0``: block structure; KV
      repetition uses ``repeat_interleave`` at block granularity.  All other
      per-element transforms (permute, gate-split) are skipped for these kinds.

    Block-unsafe operations
    -----------------------
    If any planned transformation would cross block boundaries (detected via
    ``_block_ops_need_dequant``), the entire group is dequantised to trivial
    bf16/fp16 before merging, with a warning.
    """
    group = _ensure_compatible_formats({"q": q, "k": k, "v": v})
    q, k, v = group["q"], group["k"], group["v"]

    # Pre-dequantise if any output-dimension transformation is inter-block.
    if any(_block_ops_need_dequant(lin, head_dim, bool(repeat_kv),
                                   attn_output_gate, permute_qk)
           for lin in (q, k, v)):
        import warnings
        _wfmt = q.weight_format or k.weight_format or v.weight_format
        warnings.warn(
            f"QKV merge with format '{_wfmt.name if _wfmt else None}' "
            f"(block_out={_wfmt.block_out if _wfmt else None}) and "
            f"head_dim={head_dim}: inter-block transformation detected; "
            f"dequantising to trivial.")
        q, k, v = _dequant_linear(q), _dequant_linear(k), _dequant_linear(v)

    merged_tensors: dict[str, torch.Tensor] = {}
    all_kinds = sorted(set(q.tensors) | set(k.tensors) | set(v.tensors))
    for kind in all_kinds:
        qt = q.tensors.get(kind)
        kt = k.tensors.get(kind)
        vt = v.tensors.get(kind)
        if qt is None or kt is None or vt is None:
            continue

        # Block scales (scales/zeros with block_out > 0) carry one value per
        # block of output elements; they need block-granular operations.
        wfmt = q.weight_format
        block_out = (wfmt.block_out or 0) if wfmt is not None else 0
        is_block_kind = kind in ("scales", "zeros") and block_out > 0

        # Per-element flag: True when the output dim is head_dim-aligned and
        # the kind is not a block scale (so reshape/repeat work correctly).
        full_res = not is_block_kind and (qt.size(-1) % head_dim == 0)
        gate = None

        if repeat_kv:
            n = repeat_kv
            kv_heads = kv_head_num // n
            if is_block_kind:
                # Block-scale KV repetition: repeat each output-block entry n
                # times.  The pre-dequant check above guarantees alignment.
                kt = kt.repeat_interleave(n, dim=-1)
                vt = vt.repeat_interleave(n, dim=-1)
            elif full_res:
                kt = kt.reshape(-1, kv_heads, head_dim).repeat(1, 1, n).reshape(-1, kv_heads * n * head_dim)
                vt = vt.reshape(-1, kv_heads, head_dim).repeat(1, 1, n).reshape(-1, kv_heads * n * head_dim)

        if attn_output_gate and full_res:
            head_num = qt.size(-1) // (head_dim * 2)
            orig_shape = list(qt.shape)
            if qt.dim() == 1:
                qt = qt.unsqueeze(0)
            qt = qt.view(qt.size(0), head_num, 2, head_dim)
            q_real = qt[:, :, 0, :].contiguous().reshape(-1, head_num * head_dim)
            gate = qt[:, :, 1, :].contiguous().reshape(-1, head_num * head_dim)
            if len(orig_shape) == 1:
                q_real = q_real.squeeze(0)
                gate = gate.squeeze(0)
            qt = q_real

        if permute_qk and full_res:
            qt = _reorder_rotary_emb(qt, head_dim, rope_dim)
            kt = _reorder_rotary_emb(kt, head_dim, rope_dim)

        if gate is not None:
            merged_tensors[kind] = _merge_qkvg(qt, kt, vt, gate, tp)
        else:
            merged_tensors[kind] = _merge_qkv(qt, kt, vt, tp)

    # All three linears share the same format after _ensure_compatible_formats.
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
        """Fuse Q/K/V into a single w_qkv, apply RoPE + TP interleave, commit."""
        merged = merge_qkv_linear(
            q, k, v,
            tp=self._tp,
            head_dim=self.config.head_dim,
            rope_dim=self.config.rope_dim or self.config.head_dim,
            permute_qk=True,
            attn_output_gate=self.config.attn_output_gate,
            repeat_kv=self.config.repeat_kv,
            kv_head_num=self.config.kv_head_num,
        )
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
