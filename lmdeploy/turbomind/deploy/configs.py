# Copyright (c) OpenMMLab. All rights reserved.
"""Typed config dataclasses for C++ module creation.

Each config maps 1:1 to a C++ config struct in core/module_config.h.
Use ``cfg.to_cpp()`` to convert to the pybind11-bound C++ struct,
then pass to ``Module.create_child(name, cfg.to_cpp())``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import _turbomind as _tm


@dataclass
class LinearConfig:
    input_dim: int
    output_dim: int
    data_type: int = 0
    has_bias: bool = False

    k_type_name: str = 'LinearWeight'

    def for_rank(self, rank: int) -> LinearConfig:
        return self

    def to_cpp(self) -> _tm.LinearConfig:
        cfg = _tm.LinearConfig()
        cfg.input_dim = self.input_dim
        cfg.output_dim = self.output_dim
        cfg.data_type = _tm.DataType(self.data_type) if self.data_type else _tm.DataType(0)
        cfg.has_bias = self.has_bias
        return cfg


@dataclass
class AttentionConfig:
    hidden_dim: int = 0
    head_dim: int = 0
    head_num: int = 0
    kv_head_num: int = 0
    kv_lora_rank: int = 0
    q_lora_rank: int = 0
    qk_rope_dim: int = 0
    v_head_dim: int = 0
    has_bias: bool = False
    qk_norm: bool = False
    tp_size: int = 1
    tp_rank: int = 0
    data_type: int = 0
    window_size: int = -1
    attn_sink: bool = False
    attn_output_gate: bool = False

    k_type_name: str = 'AttentionWeight'

    @classmethod
    def from_model_config(cls, mc, *, tp_size, tp_rank, dtype, window_size):
        """Build from ModelConfig. dtype is the C++ DataType value."""
        return cls(
            hidden_dim=mc.hidden_units,
            head_dim=mc.size_per_head,
            head_num=mc.head_num,
            kv_head_num=mc.kv_head_num,
            kv_lora_rank=mc.kv_lora_rank or 0,
            q_lora_rank=mc.q_lora_rank or 0,
            qk_rope_dim=mc.qk_rope_dim or 0,
            v_head_dim=mc.v_head_dim or 0,
            has_bias=mc.attn_bias,
            qk_norm=mc.qk_norm,
            tp_size=tp_size,
            tp_rank=tp_rank,
            data_type=dtype,
            window_size=window_size,
            attn_sink=mc.attn_sink,
            attn_output_gate=mc.attn_output_gate,
        )

    def for_rank(self, rank: int) -> AttentionConfig:
        """Return a copy with a different tp_rank."""
        return replace(self, tp_rank=rank)

    def to_cpp(self) -> _tm.AttentionConfig:
        cfg = _tm.AttentionConfig()
        cfg.hidden_dim = self.hidden_dim
        cfg.head_dim = self.head_dim
        cfg.head_num = self.head_num
        cfg.kv_head_num = self.kv_head_num
        cfg.kv_lora_rank = self.kv_lora_rank
        cfg.q_lora_rank = self.q_lora_rank
        cfg.qk_rope_dim = self.qk_rope_dim
        cfg.v_head_dim = self.v_head_dim
        cfg.has_bias = self.has_bias
        cfg.qk_norm = self.qk_norm
        cfg.tp_size = self.tp_size
        cfg.tp_rank = self.tp_rank
        cfg.data_type = _tm.DataType(self.data_type) if self.data_type else _tm.DataType(0)
        cfg.window_size = self.window_size
        cfg.attn_sink = self.attn_sink
        cfg.attn_output_gate = self.attn_output_gate
        return cfg


@dataclass
class FfnConfig:
    hidden_dim: int = 0
    inter_size: int = 0
    has_bias: bool = False
    tp_size: int = 1
    tp_rank: int = 0
    data_type: int = 0
    act_type: int = 0
    fuse_silu: bool = False
    fused_moe: bool = False

    k_type_name: str = 'FfnWeight'

    @classmethod
    def from_model_config(cls, mc, *, tp_size, tp_rank, dtype, act_type,
                          fuse_silu, inter_size=None, fused_moe=False):
        return cls(
            hidden_dim=mc.hidden_units,
            inter_size=inter_size if inter_size is not None else mc.inter_size,
            has_bias=mc.mlp_bias,
            tp_size=tp_size,
            tp_rank=tp_rank,
            data_type=dtype,
            act_type=act_type,
            fuse_silu=fuse_silu,
            fused_moe=fused_moe,
        )

    def for_rank(self, rank: int) -> FfnConfig:
        return replace(self, tp_rank=rank)

    def to_cpp(self) -> _tm.FfnConfig:
        cfg = _tm.FfnConfig()
        cfg.hidden_dim = self.hidden_dim
        cfg.inter_size = self.inter_size
        cfg.has_bias = self.has_bias
        cfg.tp_size = self.tp_size
        cfg.tp_rank = self.tp_rank
        cfg.data_type = _tm.DataType(self.data_type) if self.data_type else _tm.DataType(0)
        cfg.act_type = self.act_type
        cfg.fuse_silu = self.fuse_silu
        cfg.fused_moe = self.fused_moe
        return cfg


@dataclass
class MoeConfig:
    layer_id: int = 0
    method: int = 1  # kFused
    experts_per_token: int = 0
    inter_size: int = 0
    norm_topk_prob: bool = False
    shared_gate: bool = False
    routed_scale: float = 1.0
    router_bias: bool = False
    topk_group: int = 0
    topk_method: str = ''
    n_group: int = 0
    scoring_func: str = ''
    router_n_groups: int = 0
    expert_num: int = 0
    hidden_dim: int = 0
    mlp_bias: bool = False
    data_type: int = 0
    tp_size: int = 1
    tp_rank: int = 0
    act_type: int = 0
    fuse_silu: bool = False

    k_type_name: str = 'MoeWeight'

    @classmethod
    def from_model_config(cls, mc, *, layer_id, tp_size, tp_rank, dtype,
                          act_type, fuse_silu, expert_num):
        return cls(
            layer_id=layer_id,
            method=1,  # kFused
            experts_per_token=mc.experts_per_token,
            inter_size=mc.expert_inter_size or 0,
            norm_topk_prob=mc.norm_topk_prob,
            shared_gate=mc.moe_shared_gate,
            routed_scale=float(mc.routed_scale),
            router_bias=getattr(mc, 'expert_router_bias', False),
            topk_group=mc.topk_group,
            topk_method=mc.topk_method,
            n_group=mc.moe_group_num,
            scoring_func=mc.scoring_func,
            router_n_groups=max(0, getattr(mc, 'router_n_groups', -1)),
            expert_num=expert_num,
            hidden_dim=mc.hidden_units,
            mlp_bias=mc.mlp_bias,
            data_type=dtype,
            tp_size=tp_size,
            tp_rank=tp_rank,
            act_type=act_type,
            fuse_silu=fuse_silu,
        )

    def for_rank(self, rank: int) -> MoeConfig:
        return replace(self, tp_rank=rank)

    def to_cpp(self) -> _tm.MoeConfig:
        cfg = _tm.MoeConfig()
        cfg.layer_id = self.layer_id
        cfg.method = self.method
        cfg.experts_per_token = self.experts_per_token
        cfg.inter_size = self.inter_size
        cfg.norm_topk_prob = self.norm_topk_prob
        cfg.shared_gate = self.shared_gate
        cfg.routed_scale = self.routed_scale
        cfg.router_bias = self.router_bias
        cfg.topk_group = self.topk_group
        cfg.topk_method = self.topk_method
        cfg.n_group = self.n_group
        cfg.scoring_func = self.scoring_func
        cfg.router_n_groups = self.router_n_groups
        cfg.expert_num = self.expert_num
        cfg.hidden_dim = self.hidden_dim
        cfg.mlp_bias = self.mlp_bias
        cfg.data_type = _tm.DataType(self.data_type) if self.data_type else _tm.DataType(0)
        cfg.tp_size = self.tp_size
        cfg.tp_rank = self.tp_rank
        cfg.act_type = self.act_type
        cfg.fuse_silu = self.fuse_silu
        return cfg


@dataclass
class DeltaNetConfig:
    hidden_dim: int = 0
    num_k_heads: int = 0
    num_v_heads: int = 0
    key_head_dim: int = 0
    value_head_dim: int = 0
    d_conv: int = 4
    has_bias: bool = False
    tp_size: int = 1
    tp_rank: int = 0
    data_type: int = 0

    k_type_name: str = 'DeltaNetWeight'

    @classmethod
    def from_model_config(cls, mc, *, tp_size, tp_rank, dtype):
        return cls(
            hidden_dim=mc.hidden_units,
            num_k_heads=mc.linear_num_key_heads,
            num_v_heads=mc.linear_num_value_heads,
            key_head_dim=mc.linear_key_head_dim,
            value_head_dim=mc.linear_value_head_dim,
            d_conv=mc.linear_conv_kernel_dim or 4,
            has_bias=bool(mc.attn_bias),
            tp_size=tp_size,
            tp_rank=tp_rank,
            data_type=dtype,
        )

    def for_rank(self, rank: int) -> DeltaNetConfig:
        return replace(self, tp_rank=rank)

    def to_cpp(self) -> _tm.DeltaNetConfig:
        cfg = _tm.DeltaNetConfig()
        cfg.hidden_dim = self.hidden_dim
        cfg.num_k_heads = self.num_k_heads
        cfg.num_v_heads = self.num_v_heads
        cfg.key_head_dim = self.key_head_dim
        cfg.value_head_dim = self.value_head_dim
        cfg.d_conv = self.d_conv
        cfg.has_bias = self.has_bias
        cfg.tp_size = self.tp_size
        cfg.tp_rank = self.tp_rank
        cfg.data_type = _tm.DataType(self.data_type) if self.data_type else _tm.DataType(0)
        return cfg


@dataclass
class SpecAttnConfig:
    """Config for TextModelSpec.configure -- carries spec-specific TP params.

    Separate from AttentionConfig because it carries concerns (permute_qk,
    repeat_kv) that don't apply to the C++ module constructor.
    """
    tp: int = 1
    permute_qk: bool = True
    repeat_kv: int = 0
    head_dim: int = 0
    rope_dim: int = 0
    output_gate: bool = False
    kv_head_num: int = 0


@dataclass
class ModuleListConfig:
    """Config for ModuleList (pure container, no parameters)."""
    def for_rank(self, rank: int) -> ModuleListConfig:
        return self
    def to_cpp(self) -> _tm.ModuleListConfig:
        return _tm.ModuleListConfig()


@dataclass
class NormConfig:
    """Config for NormWeight."""
    dim: int = 0
    data_type: int = 0
    def for_rank(self, rank: int) -> NormConfig:
        return self
    def to_cpp(self) -> _tm.NormConfig:
        cfg = _tm.NormConfig()
        cfg.dim = self.dim
        cfg.data_type = _tm.DataType(self.data_type) if self.data_type else _tm.DataType(0)
        return cfg


@dataclass
class DecoderLayerConfig:
    """Config for DecoderLayerWeight (pure container)."""
    def for_rank(self, rank: int) -> DecoderLayerConfig:
        return self
    def to_cpp(self) -> _tm.DecoderLayerConfig:
        return _tm.DecoderLayerConfig()
