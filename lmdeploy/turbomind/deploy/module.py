# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import TYPE_CHECKING

import torch

from .linear import Linear, pad_out_dim
from .linear import transpose as linear_transpose
from .parameter import get_params, pack_u4_row

if TYPE_CHECKING:
    from .source_model.base import BaseReader
    from .target_model.base import BaseOutputModel

def permute_v2(x: torch.Tensor, size_per_head: int = 128):
    """
        Contract: x.size(-1) is output dims
    """

    assert x.size(-1) > 1

    output_dims = x.size(-1)
    head_num = output_dims // size_per_head

    return x.view(-1, head_num, 2, size_per_head // 2).transpose(2, 3).reshape(x.shape)


def permute_v2_partial(x: torch.Tensor, size_per_head: int, rotary_dim: int):
    """Permute only the first rotary_dim elements of each head.

    Used when partial_rotary_factor < 1.0: only the rotary portion needs interleaving for TurboMind's RoPE kernel
    layout.
    """
    assert x.size(-1) > 1
    assert rotary_dim % 2 == 0, f'rotary_dim must be even, got {rotary_dim}'
    assert rotary_dim <= size_per_head, f'rotary_dim ({rotary_dim}) must be <= size_per_head ({size_per_head})'
    output_dims = x.size(-1)
    assert output_dims % size_per_head == 0, (f'output_dims ({output_dims}) must be divisible by '
                                              f'size_per_head ({size_per_head})')
    head_num = output_dims // size_per_head
    orig_shape = x.shape
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = x.view(x.size(0), head_num, size_per_head)
    rotary = x[:, :, :rotary_dim]
    passthrough = x[:, :, rotary_dim:]
    # Interleave rotary part: [2, rotary_dim//2] -> [rotary_dim//2, 2]
    rotary = rotary.view(x.size(0), head_num, 2, rotary_dim // 2).transpose(2, 3).contiguous()
    rotary = rotary.view(x.size(0), head_num, rotary_dim)
    x = torch.cat([rotary, passthrough], dim=-1)
    return x.reshape(orig_shape)


def merge_qkv_v2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tp: int):
    """
        Contract: x.size(-1) is output dims
    """

    def reshape(x):
        return x.view(x.size(0), tp, -1) if q.dim() == 2 else x.view(tp, -1)

    qkv = torch.cat(tuple(map(reshape, (q, k, v))), dim=-1)

    qkv = qkv.view(-1, qkv.size(-1) * tp)
    if q.dim() == 1:
        qkv.squeeze_()

    return qkv


def merge_qkvg_v2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gate: torch.Tensor, tp: int):
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


def transpose(x):
    return x.t() if x is not None else x


def pad_out_dims(x: torch.Tensor, dims: int):
    pad = dims - x.size(-1)
    assert pad >= 0
    return torch.nn.functional.pad(x, (0, pad), 'constant', 0)


def pad_in_dims(x: torch.Tensor, dims: int):
    if x.dim() == 1:  # 1-dim object does not have input dim (e.g. bias)
        return x
    pad = dims - x.size(0)
    assert x.dim() == 2
    assert pad >= 0
    return torch.nn.functional.pad(x, (0, 0, 0, pad), 'constant', 0)


# split out dims -> copy A, split-out-dims B (qkv, w1, w3)
# split  in dims -> split-in-dims A,  copy B (  o, w2)
def get_lora_flags(kind: str):
    return ('lora_a' in kind, 'lora_b' in kind)


class Module(ABC):

    def __init__(self, model: BaseOutputModel):
        self.model = model

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    @abstractmethod
    def apply(self, idx: int, r: BaseReader):
        pass


class LayerNorm(Module):

    def apply(self, i: int, r: BaseReader):
        attn_norm = r.attn_norm(i)
        ffn_norm = r.ffn_norm(i)
        self.model.save_split(attn_norm, f'layers.{i}.attention_norm.weight')
        self.model.save_split(ffn_norm, f'layers.{i}.ffn_norm.weight')


class Ffn(Module):
    """
    requires:
        r.ffn(i, kind)
    """

    _ffn = 'layers.{0}.feed_forward.{1}.{2}'

    def __init__(self, model: BaseOutputModel):
        self.model = model
        self.tp = model.mlp_tp_size
        # inter_sizes in config are padded and may be different from what's
        # in the weights
        self.inter_size = model.model_config.inter_size
        self.group_size = max(1, model.model_config.group_size)

    def _export(self, inter_size: int, fmt: str, idx: int, w123, kind: str, pack_fn, apply_gs=[], **kwargs):
        is_lora_a, is_lora_b = get_lora_flags(kind)
        w1, w2, w3 = map(transpose, w123)

        gs1 = self.group_size if 'w1' in apply_gs else 1
        w1 = pad_out_dims(w1, inter_size // gs1)

        gs3 = self.group_size if 'w3' in apply_gs else 1
        w3 = pad_out_dims(w3, inter_size // gs3)

        gs2 = self.group_size if 'w2' in apply_gs else 1
        w2 = pad_in_dims(w2, inter_size // gs2)

        w1, w2, w3 = map(pack_fn, (w1, w2, w3))
        self.model.save_split(w1, fmt.format(idx, 'w1', kind), split_dim=-1, split_num=self.tp, copy=is_lora_a)
        self.model.save_split(w3, fmt.format(idx, 'w3', kind), split_dim=-1, split_num=self.tp, copy=is_lora_a)
        self.model.save_split(w2, fmt.format(idx, 'w2', kind), split_dim=0, split_num=self.tp, copy=is_lora_b)

    def apply(self, i: int, r: BaseReader):
        if i >= len(self.inter_size) or not self.inter_size[i]:
            return
        keys = r.ffn(i, None)

        for e in get_params(keys):
            e(partial(self._export, self.inter_size[i], self._ffn), partial(r.ffn, i), i)


class MoeFfn(Ffn):
    """
    requires:
        r.moe_ffn_expert(e, i, kind)
        r.moe_ffn_gate(i)
        r.moe_ffn_shared_gate(i)
    """

    _moe_ffn_expert = 'layers.{0}.moe_ffn.experts.E.{1}.{2}'
    _moe_ffn_gate = 'layers.{0}.moe_ffn.gate.{1}'
    _moe_ffn_shared_gate = 'layers.{0}.moe_ffn.shared_gate.weight'

    def __init__(self, model: BaseOutputModel):
        super().__init__(model)
        self.expert_num = model.model_config.expert_num
        self.inter_size = model.model_config.expert_inter_size
        self.shared_gate = model.model_config.moe_shared_gate

    def apply(self, i: int, r: BaseReader):
        if i >= len(self.expert_num) or self.expert_num[i] == 0:
            return

        # Export expert weights with outer loop over experts (not params)
        # to ensure each expert's full weight set is grouped together
        for e in range(self.expert_num[i]):
            for p in get_params(r.moe_ffn_expert(), 1):
                fmt = self._moe_ffn_expert.replace('E', str(e))
                p(partial(self._export, self.inter_size, fmt), partial(r.moe_ffn_expert, e, i), i)

        # router
        gate = transpose(r.moe_ffn_gate(i, 'weight'))
        self.model.save_split(gate, self._moe_ffn_gate.format(i, 'weight'))
        bias = r.moe_ffn_gate(i, 'bias')
        if bias is not None:
            self.model.save_split(bias, self._moe_ffn_gate.format(i, 'bias'))

        # Export score_correction_bias for noaux_tc routing (GLM 4.7 Flash)
        correction_bias = getattr(r, 'moe_ffn_gate_correction_bias', None)
        if callable(correction_bias):
            correction = correction_bias(i)
            if correction is not None:
                self.model.save_split(correction, self._moe_ffn_gate.format(i, 'score_correction_bias'))

        if self.shared_gate:
            shared_gate = transpose(r.moe_ffn_shared_gate(i))
            self.model.save_split(shared_gate, self._moe_ffn_shared_gate.format(i))


class Attn(Module):
    """
    requires:
        r.attn(i, kind)
    """

    _attn = 'layers.{0}.attention.{1}.{2}'

    def __init__(self, model: BaseOutputModel):
        self.model = model
        self.tp = model.attn_tp_size
        self.head_dim = model.model_config.size_per_head
        self.attn_bias = model.model_config.attn_bias
        self.qk_norm = model.model_config.qk_norm
        self.attn_sink = model.model_config.attn_sink
        self.group_size = max(1, model.model_config.group_size)
        self.attn_output_gate = model.model_config.attn_output_gate
        rope_param = model.attention_config.rope_param
        self.rope_dim = rope_param.dim if rope_param else self.head_dim
        self.head_num = model.model_config.head_num

    def _split_q_gate(self, q):
        """Split interleaved Q+gate tensor into separate Q and gate.

        HF layout: [Q_head0, Gate_head0, Q_head1, Gate_head1, ...]
        Returns: (q_real, gate) each with shape [..., num_heads * head_dim]
        """
        output_dims = q.size(-1)
        head_num = output_dims // (self.head_dim * 2)
        orig_shape = list(q.shape)
        if q.dim() == 1:
            q = q.unsqueeze(0)
        q = q.view(q.size(0), head_num, 2, self.head_dim)
        q_real = q[:, :, 0, :].contiguous()
        gate = q[:, :, 1, :].contiguous()
        new_last_dim = head_num * self.head_dim
        q_real = q_real.reshape(-1, new_last_dim)
        gate = gate.reshape(-1, new_last_dim)
        if len(orig_shape) == 1:
            q_real = q_real.squeeze(0)
            gate = gate.squeeze(0)
        return q_real, gate

    def _reorder_and_merge(self, qkvo, gs: int):
        q, k, v, o = qkvo
        gate = None
        # When attn_output_gate, Q is interleaved [Q0, G0, Q1, G1, ...]
        # Split into separate Q and gate before permuting
        if self.attn_output_gate and q is not None:
            q, gate = self._split_q_gate(q)
        # reorder output dim for tm's rotary embedding layout
        if self.model.permute_qk:
            if gs == 1:
                if self.rope_dim < self.head_dim:
                    q = permute_v2_partial(q, self.head_dim, self.rope_dim)
                    k = permute_v2_partial(k, self.head_dim, self.rope_dim)
                else:
                    q = permute_v2(q, self.head_dim)
                    k = permute_v2(k, self.head_dim)
            else:
                assert gs % self.head_dim == 0
        # Merge QKV with gate appended at end if present
        if gate is not None:
            qkv = merge_qkvg_v2(q, k, v, gate, self.tp)
        else:
            qkv = merge_qkv_v2(q, k, v, self.tp)
        # zero bias for `wo` when `w_qkv` has bias but `wo` doesn't
        if o is None and q.dim() == 1:
            o = torch.zeros_like(q)
        return qkv, o

    def _repeat_kv(self, qkvo, gs: int, kind: str):
        """Replicate kv."""
        q, k, v, o = qkvo
        head_dim = self.model.model_config.size_per_head // gs
        kv_head_num = self.model.model_config.kv_head_num // self.model.repeat_kv
        hidden_dim = self.model.model_config.hidden_units

        def _repeat(x):
            n = self.model.repeat_kv

            x = x.reshape(-1, kv_head_num, head_dim)
            x = x.repeat(1, 1, n)
            x = x.reshape(-1, kv_head_num * n * head_dim)

            return x

        k, v = map(_repeat, (k, v))

        if kind == 'bias':
            if o is None:
                o = torch.zeros(hidden_dim, dtype=q.dtype, device=q.device)
            q, k, v, o = map(torch.squeeze, (q, k, v, o))

        return (q, k, v, o)

    def _export(self, idx: int, qkvo, kind: str, pack_fn, apply_gs=[], **kwargs):
        if all(x is None for x in qkvo):
            return
        is_lora_a, is_lora_b = get_lora_flags(kind)
        assert not (is_lora_a or is_lora_b)

        qkvo = tuple(map(transpose, qkvo))

        gs = self.group_size if ('w1' in apply_gs) else 1

        if self.model.repeat_kv:
            qkvo = self._repeat_kv(qkvo, gs, kind)

        qkv, o = self._reorder_and_merge(qkvo, gs)

        self.model.save_split(pack_fn(qkv),
                              self._attn.format(idx, 'w_qkv', kind),
                              split_dim=-1,
                              split_num=self.tp,
                              copy=is_lora_a)
        self.model.save_split(pack_fn(o),
                              self._attn.format(idx, 'wo', kind),
                              split_dim=0,
                              split_num=self.tp,
                              copy=is_lora_b)

    def apply(self, i: int, r: BaseReader):
        for e in get_params(r.attn(i, None), bias=self.attn_bias):
            e(self._export, partial(r.attn, i), i)
        if self.qk_norm:
            q, k = r.qk_norm(i)
            if q is not None and k is not None:
                if self.model.permute_qk:
                    if self.rope_dim < self.head_dim:
                        q = permute_v2_partial(q, self.head_dim, self.rope_dim)
                        k = permute_v2_partial(k, self.head_dim, self.rope_dim)
                    else:
                        q = permute_v2(q, self.head_dim)
                        k = permute_v2(k, self.head_dim)
                self.model.save_split(q, self._attn.format(i, 'q_norm', '')[:-1])
                self.model.save_split(k, self._attn.format(i, 'k_norm', '')[:-1])
        if self.attn_sink:
            sinks = r.attn_sinks(i)
            self.model.save_split(sinks, self._attn.format(i, 'sinks', '')[:-1], split_dim=-1, split_num=self.tp)


class MLA(Module):
    """
    requires:
        r.mla(i, kind)
        r.mla_norm(i)
    """

    _mla = 'layers.{0}.attention.{1}.{2}'

    def __init__(self, model: BaseOutputModel):
        self.model = model

    def _export(self, idx: int, xs, kind: str, pack_fn, **kwargs):
        if all(x is None for x in xs):
            return
        q_a, q_b, q, kv_a, kv_b, o = xs

        cfg = self.model.model_config
        head_num = cfg.head_num
        kv_lora_rank = cfg.kv_lora_rank
        qk_rope_dim = cfg.qk_rope_dim
        size_per_head = cfg.size_per_head
        v_head_dim = cfg.v_head_dim

        # ========== MLA Weight Folding for Dimension Mismatch ==========
        # When kv_lora_rank != qk_nope_dim (e.g., GLM 4.7 Flash: 512 != 512+64=576),
        # fold the kc/vc compression/decompression BMMs into q_b_proj/o_proj weights
        # at conversion time to avoid runtime overhead.
        if kind == 'weight' and kv_lora_rank and q is None and q_b is not None and kv_b is not None and o is not None:
            if not (torch.is_floating_point(q_b) and torch.is_floating_point(kv_b) and torch.is_floating_point(o)):
                raise ValueError('MLA weight folding requires floating-point attention weights.')

            orig_q_head_dim = q_b.size(0) // head_num
            orig_qk_nope_dim = orig_q_head_dim - qk_rope_dim
            orig_kv_dim_total = kv_b.size(0) // head_num
            orig_v_head_dim = o.size(1) // head_num
            actual_orig_qk_nope_dim = orig_kv_dim_total - orig_v_head_dim

            if abs(orig_qk_nope_dim - actual_orig_qk_nope_dim) > 1:
                raise ValueError(f'Dimension mismatch: inferred qk_nope from q_b ({orig_qk_nope_dim}) != '
                                 f'inferred from kv_b ({actual_orig_qk_nope_dim})')

            orig_qk_nope_dim = actual_orig_qk_nope_dim
            target_nope_dim = size_per_head - qk_rope_dim
            target_v_head_dim = v_head_dim

            if orig_qk_nope_dim != target_nope_dim or orig_v_head_dim != target_v_head_dim:
                if target_nope_dim != kv_lora_rank or target_v_head_dim != kv_lora_rank:
                    raise ValueError(f'MLA folding expects v_head_dim and nope_dim to equal kv_lora_rank, '
                                     f'got nope={target_nope_dim}, v_head={target_v_head_dim}, rank={kv_lora_rank}')

                if kv_b.size(1) != kv_lora_rank:
                    raise ValueError(f'kv_b_proj second dim must equal kv_lora_rank for MLA folding, '
                                     f'got {kv_b.size(1)} != {kv_lora_rank}')

                # Split kv_b into kc and vc
                kv_b_per_head = kv_b.reshape(head_num, orig_qk_nope_dim + orig_v_head_dim, kv_lora_rank)
                kc_w = kv_b_per_head[:, :orig_qk_nope_dim, :]
                vc_w = kv_b_per_head[:, orig_qk_nope_dim:, :]

                # Fold kc into q_b_proj
                q_b_per_head = q_b.reshape(head_num, orig_q_head_dim, q_b.size(1))
                q_nope_w = q_b_per_head[:, :orig_qk_nope_dim, :]
                q_rope_w = q_b_per_head[:, orig_qk_nope_dim:, :]
                q_nope_expanded = torch.bmm(kc_w.transpose(1, 2), q_nope_w)
                q_b_folded = torch.cat([q_nope_expanded, q_rope_w], dim=1)
                q_b = q_b_folded.reshape(head_num * size_per_head, q_b.size(1))

                # Fold vc into o_proj
                o_per_head = o.reshape(o.size(0), head_num, orig_v_head_dim)
                o_folded = torch.bmm(o_per_head.permute(1, 0, 2), vc_w)
                o = o_folded.permute(1, 0, 2).reshape(o.size(0), head_num * kv_lora_rank)

                # Set kv_b to identity (kc/vc are now absorbed)
                eye = torch.eye(kv_lora_rank, dtype=kv_b.dtype, device=kv_b.device)
                kv_b = torch.cat([eye, eye], dim=0).repeat(head_num, 1)
        # ========== End MLA Weight Folding ==========

        # Transpose after folding
        q_a, q_b, q, kv_a, kv_b, o = map(transpose, (q_a, q_b, q, kv_a, kv_b, o))

        if q is not None:
            q_b = q

        # Pad o_proj to size_per_head if present
        if o is not None:
            o = o.reshape(head_num, v_head_dim, -1)
            o = torch.nn.functional.pad(o, (0, 0, size_per_head - v_head_dim, 0, 0, 0))
            o = o.view(head_num * size_per_head, cfg.hidden_units)

        tp = self.model.attn_tp_size

        # Export MLA weights (handle None for folded-away tensors)
        if q_a is not None:
            self.model.save_split(pack_fn(q_a), self._mla.format(idx, 'q_a_proj', kind))
        q_b_name = 'q_proj' if q_a is None else 'q_b_proj'
        if q_b is not None:
            self.model.save_split(pack_fn(q_b), self._mla.format(idx, q_b_name, kind), split_dim=-1, split_num=tp)
        if kv_a is not None:
            self.model.save_split(pack_fn(kv_a), self._mla.format(idx, 'kv_a_proj', kind))
        # if kv_b is not None:
        #     self.model.save_split(pack_fn(kv_b), self._mla.format(idx, 'kv_b_proj', kind), split_dim=-1, split_num=tp)
        if o is not None:
            self.model.save_split(pack_fn(o), self._mla.format(idx, 'wo', kind), split_dim=0, split_num=tp)

    _layernorm = 'layers.{0}.attention.{1}_a_layernorm'

    def apply(self, i: int, r: BaseReader):

        for f in get_params(r.attn(i, None), bias=False):
            f(self._export, partial(r.mla, i), i)

        q, k = r.mla_norm(i)
        if q is not None:
            self.model.save_split(q, self._layernorm.format(i, 'q'))
        self.model.save_split(k, self._layernorm.format(i, 'kv'))


class LinearAttn(Module):
    _linear_attn = 'layers.{0}.linear_attn.{1}.{2}'

    def __init__(self, model: BaseOutputModel):
        self.model = model
        self.tp = model.attn_tp_size
        cfg = model.model_config
        self.key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
        self.value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim

    def _tp_interleave_qkv(self, tensor, dim):
        """Split a concatenated [Q, K, V] tensor into components, reshape each
        for TP interleaving, and re-concatenate.

        in_proj_qkv layout along ``dim``: Q(key_dim) | K(key_dim) | V(value_dim).
        A naive split doesn't respect component boundaries when key_dim and
        value_dim differ.  This method splits Q/K/V, reshapes each to
        ``(tp, -1)`` along ``dim``, concatenates per-TP-shard, then flattens
        so that a subsequent ``save_split(split_dim=dim)`` gives each rank the
        correct portion.
        """
        if dim < 0:
            dim = tensor.dim() + dim
        q, k, v = torch.split(tensor, [self.key_dim, self.key_dim, self.value_dim], dim=dim)

        def reshape(x):
            # Move TP axis to a new dimension right after ``dim``
            shape = list(x.shape)
            d = shape[dim]
            new_shape = shape[:dim] + [self.tp, d // self.tp] + shape[dim + 1:]
            return x.view(new_shape)

        parts = torch.cat([reshape(q), reshape(k), reshape(v)], dim=dim + 1)
        # Collapse tp and per-shard dims back
        shape = list(parts.shape)
        final_shape = shape[:dim] + [shape[dim] * shape[dim + 1]] + shape[dim + 2:]
        return parts.reshape(final_shape)

    def apply(self, i: int, r: BaseReader):
        layer_types = getattr(self.model.model_config, 'layer_types', [])
        if i >= len(layer_types) or layer_types[i] != 'linear_attention':
            return

        for kind in ['weight', 'bias']:
            weights = r.linear_attn(i, kind)
            if not weights:
                continue

            names = ['conv1d', 'in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a', 'out_proj', 'A_log', 'dt_bias']
            for name, tensor in zip(names, weights):
                if tensor is None:
                    continue
                if name == 'conv1d':
                    # conv1d shape: (conv_dim, 1, d_conv) where
                    # conv_dim = key_dim*2 + value_dim.  Interleave Q/K/V
                    # portions along dim 0 before splitting for TP.
                    tensor = self._tp_interleave_qkv(tensor, dim=0)
                    self.model.save_split(tensor,
                                          self._linear_attn.format(i, name, kind),
                                          split_dim=0,
                                          split_num=self.tp)
                elif name in ['A_log', 'dt_bias']:
                    # Split per-head params across TP ranks (use -1 to
                    # avoid the 1-D copy shortcut in save_split).
                    self.model.save_split(tensor,
                                          self._linear_attn.format(i, name, kind),
                                          split_dim=-1,
                                          split_num=self.tp)
                elif name == 'out_proj':
                    self.model.save_split(transpose(tensor),
                                          self._linear_attn.format(i, name, kind),
                                          split_dim=0,
                                          split_num=self.tp)
                elif name == 'in_proj_qkv':
                    # in_proj_qkv: (conv_dim, hidden) where conv_dim =
                    # key_dim*2 + value_dim.  After transpose the QKV
                    # components are along dim -1.  Interleave for TP so
                    # each shard gets the correct Q/K/V slice.
                    t = transpose(tensor)
                    t = self._tp_interleave_qkv(t, dim=-1)
                    self.model.save_split(t, self._linear_attn.format(i, name, kind), split_dim=-1, split_num=self.tp)
                else:
                    self.model.save_split(transpose(tensor),
                                          self._linear_attn.format(i, name, kind),
                                          split_dim=-1,
                                          split_num=self.tp)

        norm = r.linear_norm(i, 'weight')
        if norm is not None:
            self.model.export_weight(norm, f'layers.{i}.linear_attn.norm.weight')


class Misc(Module):
    """
    requires:
        r.tok_embeddings()
        r.norm_weight()
        r.output_weight()
    """

    def apply(self, i: int, r: BaseReader):
        """Export embedding, norm, output weight."""
        emb = r.tok_embeddings()
        norm_weight = r.norm_weight()
        output_weight = r.output_weight()

        def pad_weight(tensor: torch.Tensor, tp: int):
            pad_size = None
            vocab_size = self.model.model_config.vocab_size
            if vocab_size % tp != 0:
                pad_size = (vocab_size + tp - 1) // tp * tp - vocab_size
            if pad_size is None:
                return tensor
            return torch.nn.functional.pad(tensor, (0, 0, 0, pad_size), 'constant', 0)

        tp = self.model.attn_tp_size * self.model.attn_cp_size
        if emb is not None:
            emb = pad_weight(emb, tp=tp)
            self.model.save_split(emb, 'tok_embeddings.weight', split_dim=1, split_num=tp)
        if norm_weight is not None:
            self.model.export_weight(norm_weight, 'norm.weight')
        if output_weight is not None:
            output_weight = pad_weight(output_weight, tp=tp)
            # transpose
            self.model.save_split(output_weight.t(), 'output.weight', split_dim=1, split_num=tp)


class Transformer:

    def __init__(self, model: BaseOutputModel):
        self.model = model
        modules = [LayerNorm]
        if model.model_config.kv_lora_rank:
            modules.append(MLA)
        else:
            modules.append(Attn)
        if getattr(model.model_config, 'layer_types', []):
            modules.append(LinearAttn)
        if model.model_config.inter_size:
            modules.append(Ffn)
        if model.model_config.expert_num:
            modules.append(MoeFfn)
        self.modules = [c(model) for c in modules]
        self.misc = Misc(model)
        self._v2 = TransformerV2(model)

    def __call__(self, i: int, r):
        if isinstance(r, ModelWeightSpec):
            return self._v2(i, r)
        if i >= 0:
            for m in self.modules:
                m(i, r)
            return 1
        else:
            self.misc(i, r)


# ===================================================================
# New pipeline: ModelWeightSpec + composable-ops Transformer
# ===================================================================


class ModelWeightSpec(ABC):
    """Declarative weight mapping for a model architecture.

    Subclasses define how to read and transform weights for a specific model.
    Methods return ``Linear`` for linear layers and raw ``Tensor`` for norms,
    embeddings, scalars, etc.

    The ``TransformerV2`` consumes a spec: it iterates the returned dicts,
    applies TP split rules, and commits each weight to C++.
    """

    # -- Linear bundles (TP-split by the transformer) --

    def attn_linears(self, layer: int) -> dict[str, Linear]:
        """Return ``{tm_name: Linear}`` for attention weights.

        Standard attention should return ``{"w_qkv": ..., "wo": ...}``.
        MLA should return ``{"q_a_proj": ..., "q_b_proj": ..., ...}``.
        """
        return {}

    def ffn_linears(self, layer: int) -> dict[str, Linear]:
        """Return ``{tm_name: Linear}`` for dense FFN / shared-expert weights."""
        return {}

    def moe_ffn_linears(self, layer: int, expert: int) -> dict[str, Linear]:
        """Return ``{tm_name: Linear}`` for one MoE routed expert."""
        return {}

    def linear_attn_linears(self, layer: int) -> dict[str, Linear]:
        """Return ``{tm_name: Linear}`` for linear-attention (GDN) weights."""
        return {}

    # -- Raw tensors (broadcast or simple split) --

    def attn_norm(self, layer: int) -> torch.Tensor | None:
        return None

    def ffn_norm(self, layer: int) -> torch.Tensor | None:
        return None

    def tok_embeddings(self) -> torch.Tensor | None:
        return None

    def output_weight(self) -> torch.Tensor | None:
        return None

    def norm_weight(self) -> torch.Tensor | None:
        return None

    def moe_ffn_gate(self, layer: int) -> torch.Tensor | None:
        return None

    def moe_ffn_gate_bias(self, layer: int) -> torch.Tensor | None:
        return None

    def moe_ffn_gate_correction_bias(self, layer: int) -> torch.Tensor | None:
        return None

    def moe_ffn_shared_gate(self, layer: int) -> torch.Tensor | None:
        return None

    def qk_norm(self, layer: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return None, None

    def mla_norm(self, layer: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return None, None

    def linear_attn_norm(self, layer: int) -> torch.Tensor | None:
        return None

    def linear_attn_scalars(self, layer: int) -> dict[str, torch.Tensor]:
        return {}

    def attn_sinks(self, layer: int) -> torch.Tensor | None:
        return None

    # -- metadata --

    @abstractmethod
    def model_info(self) -> dict:
        """Return model metadata (num_layer, head_num, etc.)."""

    def num_experts(self, layer: int) -> int:
        return 0

    def has_shared_gate(self) -> bool:
        return False


# -----------------------------------------------------------------------
# Commit helpers
# -----------------------------------------------------------------------


def _infer_cpp_linear_dtype(linear: Linear):
    """Determine C++ DataType and group_size from a Linear bundle."""
    try:
        import _turbomind as _tm
    except ImportError:
        return None, 0

    if "qweight" in linear.tensors:
        return _tm.DataType.TYPE_UINT4, 0
    weight = linear.tensors.get("weight")
    if weight is not None:
        if weight.dtype == torch.float8_e4m3fn:
            return _tm.DataType.TYPE_FP8_E4M3, 128
        if weight.dtype == torch.uint8 and "scales" in linear.tensors:
            scales = linear.tensors["scales"]
            if scales.dtype == torch.uint8:
                return _tm.DataType.TYPE_FP4_E2M1, 32
            return _tm.DataType.TYPE_FP8_E4M3, 128
        if weight.dtype == torch.bfloat16:
            return _tm.DataType.TYPE_BF16, 0
        if weight.dtype == torch.float16:
            return _tm.DataType.TYPE_FP16, 0
    return None, 0


def commit_linear(model: BaseOutputModel, linear: Linear, name: str,
                  split_dim=None, split_num=1, copy=False):
    """Export every tensor in a ``Linear`` bundle via ``model.save_split``.

    uint8 tensors (unpacked 4-bit weights) are re-packed to int32 before
    export so the C++ side receives the format it expects.

    For deferred-emplace models, this also triggers C++ ``allocate()`` so
    the weight tensors are created before ``copy_from``.
    """
    cpp_dtype, group_size = None, 0
    if hasattr(model, 'model_comm') and model.model_comm is not None:
        cpp_dtype, group_size = _infer_cpp_linear_dtype(linear)
        if cpp_dtype is not None:
            if group_size == 0:
                group_size = max(1, model.model_config.group_size)
            is_qweight = "qweight" in linear.tensors
            weight_kind = "qweight" if is_qweight else "weight"
            if split_dim is not None or copy:
                for rank in range(split_num):
                    alloc_name = f"{name}.{rank}.{weight_kind}"
                    model.allocate_weight(alloc_name, cpp_dtype, group_size)
            else:
                alloc_name = f"{name}.{weight_kind}"
                model.allocate_weight(alloc_name, cpp_dtype, group_size)

    for kind, tensor in linear.tensors.items():
        if kind == "qweight" and tensor.dtype == torch.uint8:
            tensor = pack_u4_row(tensor)
        elif kind == "weight" and tensor.dtype == torch.uint8 and cpp_dtype is not None:
            try:
                import _turbomind as _tm
            except ImportError:
                _tm = None
            if _tm is not None and cpp_dtype == _tm.DataType.TYPE_FP4_E2M1:
                tensor = pack_u4_row(tensor)
        model.save_split(tensor, f"{name}.{kind}",
                         split_dim=split_dim, split_num=split_num, copy=copy)


def commit_tensor(model: BaseOutputModel, tensor: torch.Tensor | None,
                  name: str, split_dim=None, split_num=1, copy=False):
    """Export a single raw tensor."""
    if tensor is None:
        return
    if split_dim is not None or copy:
        model.save_split(tensor, name, split_dim=split_dim,
                         split_num=split_num, copy=copy)
    else:
        model.export_weight(tensor, name)


# -----------------------------------------------------------------------
# TP split rules
# -----------------------------------------------------------------------

_ATTN_TP_RULES: dict[str, dict] = {
    "w_qkv": dict(split_dim=-1),
    "wo": dict(split_dim=0),
    "q_proj": dict(split_dim=-1),
    "q_a_proj": dict(),
    "q_b_proj": dict(split_dim=-1),
    "kv_a_proj": dict(),
    "kv_b_proj": dict(split_dim=-1),
}

_FFN_TP_RULES: dict[str, dict] = {
    "w1": dict(split_dim=-1),
    "w3": dict(split_dim=-1),
    "w2": dict(split_dim=0),
}

_LINEAR_ATTN_TP_RULES: dict[str, dict] = {
    "conv1d": dict(split_dim=0),
    "in_proj_qkv": dict(split_dim=-1),
    "in_proj_z": dict(split_dim=-1),
    "in_proj_b": dict(split_dim=-1),
    "in_proj_a": dict(split_dim=-1),
    "out_proj": dict(split_dim=0),
}


# -----------------------------------------------------------------------
# TransformerV2
# -----------------------------------------------------------------------


class TransformerV2:
    """Composable-ops transformer that consumes ``ModelWeightSpec``.

    Unlike the legacy ``Transformer``, this class does not contain per-model
    logic.  All model-specific decisions (key mapping, QKV merge, MLA folding,
    RoPE permutation, zero-centered norms, etc.) live in the spec.
    """

    def __init__(self, model: BaseOutputModel):
        self.model = model
        cfg = model.model_config
        self.attn_tp = model.attn_tp_size
        self.mlp_tp = model.mlp_tp_size
        self.num_layer = cfg.num_layer
        self.inter_size = cfg.inter_size
        self.expert_inter_size = getattr(cfg, "expert_inter_size", 0)
        self.expert_num = getattr(cfg, "expert_num", None)
        self.has_moe_shared_gate = getattr(cfg, "moe_shared_gate", False)
        self.vocab_size = cfg.vocab_size
        self.head_dim = cfg.size_per_head
        self.head_num = cfg.head_num
        self.permute_qk = getattr(model, "permute_qk", True)
        self.repeat_kv = getattr(model, "repeat_kv", 0)
        self.attn_output_gate = getattr(cfg, "attn_output_gate", False)
        self.group_size = max(1, cfg.group_size)
        rope_param = model.attention_config.rope_param
        self.rope_dim = rope_param.dim if rope_param else self.head_dim

    def __call__(self, layer: int, spec: ModelWeightSpec):
        if layer >= 0:
            if layer >= self.num_layer:
                return 0
            self._process_layer(layer, spec)
            return 1
        else:
            self._process_misc(spec)

    # -- QKV merge helpers -------------------------------------------------

    def _permute_qk(self, q, k):
        """Apply RoPE layout permutation to Q and K weight tensors."""
        if self.rope_dim < self.head_dim:
            q = permute_v2_partial(q, self.head_dim, self.rope_dim)
            k = permute_v2_partial(k, self.head_dim, self.rope_dim)
        else:
            q = permute_v2(q, self.head_dim)
            k = permute_v2(k, self.head_dim)
        return q, k

    def _split_q_gate(self, q):
        """Split interleaved Q+gate tensor into (q, gate)."""
        output_dims = q.size(-1)
        head_num = output_dims // (self.head_dim * 2)
        orig_shape = list(q.shape)
        if q.dim() == 1:
            q = q.unsqueeze(0)
        q = q.view(q.size(0), head_num, 2, self.head_dim)
        q_real = q[:, :, 0, :].contiguous().reshape(-1, head_num * self.head_dim)
        gate = q[:, :, 1, :].contiguous().reshape(-1, head_num * self.head_dim)
        if len(orig_shape) == 1:
            q_real = q_real.squeeze(0)
            gate = gate.squeeze(0)
        return q_real, gate

    def _repeat_kv_tensor(self, t):
        """Replicate KV heads for TP when tp > kv_head_num."""
        n = self.repeat_kv
        kv_head_num = self.model.model_config.kv_head_num // n
        head_dim = self.head_dim
        t = t.reshape(-1, kv_head_num, head_dim)
        t = t.repeat(1, 1, n).reshape(-1, kv_head_num * n * head_dim)
        return t

    def _merge_qkv_kind(self, q, k, v, kind: str, layer: int):
        """Merge one tensor kind of Q/K/V into a merged w_qkv tensor.

        repeat_kv, split_q_gate, and permute_qk all operate along the output
        dimension (head layout) and are safe to apply regardless of whether the
        tensors are quantisation-grouped along the input dimension.

        Block-compressed tensors (e.g. FP8 per-block scales with shape
        [ceil(out/block), ceil(in/block)]) have a reduced output dimension
        that does not carry per-element head structure.  We detect this via
        ``q.size(-1) % head_dim != 0`` and skip per-element operations.
        """
        full_res = q.size(-1) % self.head_dim == 0
        gate = None
        if self.repeat_kv and full_res:
            k = self._repeat_kv_tensor(k)
            v = self._repeat_kv_tensor(v)
        if self.attn_output_gate and q is not None and full_res:
            q, gate = self._split_q_gate(q)
        if self.permute_qk and full_res:
            q, k = self._permute_qk(q, k)

        if gate is not None:
            merged = merge_qkvg_v2(q, k, v, gate, self.attn_tp)
        else:
            merged = merge_qkv_v2(q, k, v, self.attn_tp)
        return merged

    def _process_attn_qkv(self, layer: int, linears: dict[str, Linear], spec: ModelWeightSpec | None = None):
        """Handle ``w_qkv.{q,k,v}`` + ``wo`` pattern: merge Q/K/V with RoPE
        permutation and TP interleaving, then commit."""
        q_lin = linears.pop("w_qkv.q")
        k_lin = linears.pop("w_qkv.k")
        v_lin = linears.pop("w_qkv.v")
        o_lin = linears.pop("wo", None)

        # Allocate the merged w_qkv on C++ side (deferred-emplace)
        if hasattr(self.model, 'model_comm') and self.model.model_comm is not None:
            cpp_dtype, group_size = _infer_cpp_linear_dtype(q_lin)
            if cpp_dtype is not None:
                if group_size == 0:
                    group_size = max(1, self.group_size)
                is_qweight = "qweight" in q_lin.tensors
                weight_kind = "qweight" if is_qweight else "weight"
                for rank in range(self.attn_tp):
                    alloc_name = f"layers.{layer}.attention.w_qkv.{rank}.{weight_kind}"
                    self.model.allocate_weight(alloc_name, cpp_dtype, group_size)

        all_kinds = set(q_lin.tensors) | set(k_lin.tensors) | set(v_lin.tensors)
        for kind in sorted(all_kinds):
            q = q_lin.tensors.get(kind)
            k = k_lin.tensors.get(kind)
            v = v_lin.tensors.get(kind)
            if q is None or k is None or v is None:
                continue
            merged = self._merge_qkv_kind(q, k, v, kind, layer)
            if kind == "qweight" and merged.dtype == torch.uint8:
                merged = pack_u4_row(merged)
            self.model.save_split(merged,
                                  f"layers.{layer}.attention.w_qkv.{kind}",
                                  split_dim=-1, split_num=self.attn_tp)

        if o_lin is not None:
            has_qkv_bias = "bias" in q_lin.tensors
            has_o_bias = "bias" in o_lin.tensors
            if has_qkv_bias and not has_o_bias:
                q_bias = q_lin.tensors["bias"]
                o_lin.tensors["bias"] = torch.zeros_like(q_bias)
            commit_linear(self.model, o_lin,
                          f"layers.{layer}.attention.wo",
                          split_dim=0, split_num=self.attn_tp)

        for name, lin in linears.items():
            rule = _ATTN_TP_RULES.get(name, {})
            tp = self.attn_tp if rule.get("split_dim") is not None else 1
            commit_linear(self.model, lin,
                          f"layers.{layer}.attention.{name}",
                          split_num=tp, **rule)

    # -- per-layer ---------------------------------------------------------

    def _process_layer(self, layer: int, spec: ModelWeightSpec):
        # Layer norms (broadcast, no TP split)
        commit_tensor(self.model, spec.attn_norm(layer),
                      f"layers.{layer}.attention_norm.weight")
        commit_tensor(self.model, spec.ffn_norm(layer),
                      f"layers.{layer}.ffn_norm.weight")

        # Attention linears
        attn_linears = spec.attn_linears(layer)
        if "w_qkv.q" in attn_linears:
            self._process_attn_qkv(layer, attn_linears, spec)
        else:
            for name, lin in attn_linears.items():
                rule = _ATTN_TP_RULES.get(name, {})
                tp = self.attn_tp if rule.get("split_dim") is not None else 1
                commit_linear(self.model, lin,
                              f"layers.{layer}.attention.{name}",
                              split_num=tp, **rule)

        # QK norm (with RoPE permutation)
        q_norm, k_norm = spec.qk_norm(layer)
        if q_norm is not None and k_norm is not None and self.permute_qk:
            q_norm, k_norm = self._permute_qk(q_norm, k_norm)
        if q_norm is not None:
            commit_tensor(self.model, q_norm,
                          f"layers.{layer}.attention.q_norm")
        if k_norm is not None:
            commit_tensor(self.model, k_norm,
                          f"layers.{layer}.attention.k_norm")

        # MLA norm
        mla_q, mla_kv = spec.mla_norm(layer)
        if mla_q is not None:
            commit_tensor(self.model, mla_q,
                          f"layers.{layer}.attention.q_a_layernorm")
        if mla_kv is not None:
            commit_tensor(self.model, mla_kv,
                          f"layers.{layer}.attention.kv_a_layernorm")

        # Attention sinks
        sinks = spec.attn_sinks(layer)
        if sinks is not None:
            commit_tensor(self.model, sinks,
                          f"layers.{layer}.attention.sinks",
                          split_dim=-1, split_num=self.attn_tp)

        # Dense FFN linears
        for name, lin in spec.ffn_linears(layer).items():
            rule = _FFN_TP_RULES.get(name, {})
            tp = self.mlp_tp if rule.get("split_dim") is not None else 1
            commit_linear(self.model, lin,
                          f"layers.{layer}.feed_forward.{name}",
                          split_num=tp, **rule)

        # MoE experts
        n_experts = spec.num_experts(layer)
        for e in range(n_experts):
            for name, lin in spec.moe_ffn_linears(layer, e).items():
                rule = _FFN_TP_RULES.get(name, {})
                tp = self.mlp_tp if rule.get("split_dim") is not None else 1
                commit_linear(self.model, lin,
                              f"layers.{layer}.moe_ffn.experts.{e}.{name}",
                              split_num=tp, **rule)

        # MoE router
        if n_experts > 0:
            gate = spec.moe_ffn_gate(layer)
            if gate is not None:
                gate = linear_transpose(gate) if gate.dim() > 1 else gate
                commit_tensor(self.model, gate,
                              f"layers.{layer}.moe_ffn.gate.weight")
            gate_bias = spec.moe_ffn_gate_bias(layer)
            commit_tensor(self.model, gate_bias,
                          f"layers.{layer}.moe_ffn.gate.bias")
            correction = spec.moe_ffn_gate_correction_bias(layer)
            commit_tensor(self.model, correction,
                          f"layers.{layer}.moe_ffn.gate.score_correction_bias")

        # MoE shared gate
        if spec.has_shared_gate():
            sg = spec.moe_ffn_shared_gate(layer)
            if sg is not None:
                sg = linear_transpose(sg) if sg.dim() > 1 else sg
                commit_tensor(self.model, sg,
                              f"layers.{layer}.moe_ffn.shared_gate.weight")

        # Linear attention
        for name, lin in spec.linear_attn_linears(layer).items():
            rule = _LINEAR_ATTN_TP_RULES.get(name, {})
            tp = self.attn_tp if rule.get("split_dim") is not None else 1
            commit_linear(self.model, lin,
                          f"layers.{layer}.linear_attn.{name}",
                          split_num=tp, **rule)

        for name, tensor in spec.linear_attn_scalars(layer).items():
            commit_tensor(self.model, tensor,
                          f"layers.{layer}.linear_attn.{name}.weight",
                          split_dim=-1, split_num=self.attn_tp)

        la_norm = spec.linear_attn_norm(layer)
        commit_tensor(self.model, la_norm,
                      f"layers.{layer}.linear_attn.norm.weight")

    # -- misc (embeddings, output head, final norm) ------------------------

    def _process_misc(self, spec: ModelWeightSpec):
        tp = self.attn_tp * self.model.attn_cp_size
        padded_vocab = ((self.vocab_size + tp - 1) // tp) * tp

        emb = spec.tok_embeddings()
        if emb is not None:
            emb = pad_out_dim(emb, padded_vocab, dim=0)
            self.model.save_split(emb, "tok_embeddings.weight",
                                  split_dim=1, split_num=tp)

        norm = spec.norm_weight()
        commit_tensor(self.model, norm, "norm.weight")

        output = spec.output_weight()
        if output is not None:
            output = pad_out_dim(output, padded_vocab, dim=0)
            output = output.t()
            self.model.save_split(output, "output.weight",
                                  split_dim=1, split_num=tp)
