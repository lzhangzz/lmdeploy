# Copyright (c) OpenMMLab. All rights reserved.
"""Factory functions for C++ module config structs.

Each factory constructs a pybind-bound C++ config struct directly from
ModelConfig (or explicit parameters), eliminating the intermediate Python
dataclass layer.
"""
from __future__ import annotations

import _turbomind as _tm

# Re-export simple C++ configs that need no transformation
from _turbomind import DecoderLayerConfig, ModuleListConfig


# ---------------------------------------------------------------------------
# Simple configs (few fields, no ModelConfig dependency)
# ---------------------------------------------------------------------------


def make_linear_config(*, input_dim, output_dim, data_type, has_bias=False):
    cfg = _tm.LinearConfig()
    cfg.input_dim = input_dim
    cfg.output_dim = output_dim
    cfg.data_type = data_type
    cfg.has_bias = has_bias
    return cfg


def make_norm_config(*, dim, data_type):
    cfg = _tm.NormConfig()
    cfg.dim = dim
    cfg.data_type = data_type
    return cfg


# ---------------------------------------------------------------------------
# Attention configs
# ---------------------------------------------------------------------------


def make_attention_config(mc, *, tp_size, tp_rank=0, dtype, window_size=0,
                         rope_dim=0):
    """Build C++ AttentionConfig from ModelConfig."""
    cfg = _tm.AttentionConfig()
    cfg.hidden_dim = mc.hidden_units
    cfg.head_dim = mc.size_per_head
    cfg.head_num = mc.head_num
    cfg.kv_head_num = mc.kv_head_num
    cfg.kv_lora_rank = mc.kv_lora_rank or 0
    cfg.q_lora_rank = mc.q_lora_rank or 0
    cfg.qk_rope_dim = mc.qk_rope_dim or 0
    cfg.v_head_dim = mc.v_head_dim or 0
    cfg.has_bias = mc.attn_bias
    cfg.qk_norm = mc.qk_norm
    cfg.tp_size = tp_size
    cfg.tp_rank = tp_rank
    cfg.data_type = dtype
    cfg.window_size = window_size
    cfg.attn_sink = mc.attn_sink
    cfg.attn_output_gate = mc.attn_output_gate
    cfg.rope_dim = rope_dim
    return cfg


def make_mla_config(mc, *, tp_size, tp_rank=0, dtype, window_size=0,
                    qk_nope_dim=0):
    """Build C++ AttentionConfig for MLA from ModelConfig."""
    qk_rope_dim = mc.qk_rope_dim or 0
    kv_lora_rank = mc.kv_lora_rank or 0
    v_head_dim = mc.v_head_dim or 0
    size_per_head = qk_nope_dim + qk_rope_dim
    if kv_lora_rank and kv_lora_rank != qk_nope_dim:
        size_per_head = kv_lora_rank + qk_rope_dim
        v_head_dim = kv_lora_rank

    cfg = _tm.AttentionConfig()
    cfg.hidden_dim = mc.hidden_units
    cfg.head_dim = size_per_head
    cfg.head_num = mc.head_num
    cfg.kv_head_num = mc.kv_head_num
    cfg.kv_lora_rank = kv_lora_rank
    cfg.q_lora_rank = mc.q_lora_rank or 0
    cfg.qk_rope_dim = qk_rope_dim
    cfg.qk_nope_dim = qk_nope_dim
    cfg.v_head_dim = v_head_dim
    cfg.tp_size = tp_size
    cfg.tp_rank = tp_rank
    cfg.data_type = dtype
    cfg.window_size = window_size
    cfg.has_bias = False
    cfg.qk_norm = False
    cfg.attn_sink = False
    cfg.attn_output_gate = False
    cfg.rope_dim = 0
    return cfg


# ---------------------------------------------------------------------------
# FFN / MoE configs
# ---------------------------------------------------------------------------


def make_ffn_config(mc, *, tp_size, tp_rank, dtype, act_type,
                    fuse_silu, inter_size=None, fused_moe=False):
    """Build C++ FfnConfig from ModelConfig."""
    cfg = _tm.FfnConfig()
    cfg.hidden_dim = mc.hidden_units
    cfg.inter_size = inter_size if inter_size is not None else mc.inter_size
    cfg.has_bias = mc.mlp_bias
    cfg.tp_size = tp_size
    cfg.tp_rank = tp_rank
    cfg.data_type = dtype
    cfg.act_type = act_type
    cfg.fuse_silu = fuse_silu
    cfg.fused_moe = fused_moe
    return cfg


def make_moe_config(mc, *, layer_id, tp_size, tp_rank, dtype,
                    act_type, fuse_silu, expert_num):
    """Build C++ MoeConfig from ModelConfig."""
    cfg = _tm.MoeConfig()
    cfg.layer_id = layer_id
    cfg.method = 1  # kFused
    cfg.experts_per_token = mc.experts_per_token
    cfg.inter_size = mc.expert_inter_size or 0
    cfg.norm_topk_prob = mc.norm_topk_prob
    cfg.shared_gate = mc.moe_shared_gate
    cfg.routed_scale = float(mc.routed_scale)
    cfg.router_bias = mc.expert_router_bias
    cfg.topk_group = mc.topk_group
    cfg.topk_method = mc.topk_method
    cfg.n_group = mc.moe_group_num
    cfg.scoring_func = mc.scoring_func
    cfg.router_n_groups = max(0, mc.router_n_groups)
    cfg.expert_num = expert_num
    cfg.hidden_dim = mc.hidden_units
    cfg.mlp_bias = mc.mlp_bias
    cfg.data_type = dtype
    cfg.tp_size = tp_size
    cfg.tp_rank = tp_rank
    cfg.act_type = act_type
    cfg.fuse_silu = fuse_silu
    return cfg


# ---------------------------------------------------------------------------
# DeltaNet config
# ---------------------------------------------------------------------------


def make_deltanet_config(mc, *, tp_size, tp_rank, dtype):
    """Build C++ DeltaNetConfig from ModelConfig."""
    cfg = _tm.DeltaNetConfig()
    cfg.hidden_dim = mc.hidden_units
    cfg.num_k_heads = mc.linear_num_key_heads
    cfg.num_v_heads = mc.linear_num_value_heads
    cfg.key_head_dim = mc.linear_key_head_dim
    cfg.value_head_dim = mc.linear_value_head_dim
    cfg.d_conv = mc.linear_conv_kernel_dim or 4
    cfg.has_bias = bool(mc.attn_bias)
    cfg.tp_size = tp_size
    cfg.tp_rank = tp_rank
    cfg.data_type = dtype
    return cfg
