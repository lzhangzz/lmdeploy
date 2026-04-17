# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import _turbomind as _tm

from ._base import Builder, SplitSide


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------


def make_moe_config(mc, *, layer_id, tp_size, tp_rank=0, dtype,
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
# MoeBuilder -- gate, non-expert params
# ---------------------------------------------------------------------------


class MoeBuilder(Builder):
    """MoE weight loading builder."""

    def add_gate(self, name, linear, model_dtype=None):
        """Commit a gate linear (broadcast, no split)."""
        self._commit_linear(name, linear, split_side=None,
                            model_dtype=model_dtype)

    def add_param(self, name, tensor, split_side=None):
        """Commit a non-expert MoE parameter."""
        if split_side is not None and not isinstance(split_side, SplitSide):
            split_side = None  # specs may pass None for broadcast
        self._commit_tensor(name, tensor, split_side)
