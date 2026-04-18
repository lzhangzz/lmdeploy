# Copyright (c) OpenMMLab. All rights reserved.
"""Qwen3 TextModelSpec for the new pipeline.

Qwen3 is a standard Llama-like model with QK norm and optional MoE.
No shared expert in the MoE variant, no linear attention, no zero-centered
norm.
"""
from __future__ import annotations

import _turbomind as _tm

from ..builder import (AttentionBuilder, DecoderLayerBuilder, FfnBuilder,
                       MoeBuilder, ModuleListBuilder, TextModelBuilder,
                       _act_type_id)
from ..builder import DecoderLayerConfig, ModuleListConfig
from ..linear import Linear
from ..spec import TextModelSpec
from .base import INPUT_MODELS
from .utils import (_pad_inter_size, reorder_rotary_emb,
                    reorder_rotary_emb_linear)

_LAYER_PATTERN = r'model\.layers\.([0-9]+).'


@INPUT_MODELS.register_module(name='qwen3-moe')
@INPUT_MODELS.register_module(name='qwen3')
class Qwen3TextSpec(TextModelSpec):
    """Weight spec for Qwen3 (dense) and Qwen3-MoE."""

    _layer_pattern = _LAYER_PATTERN
    # Qwen3 always uses the plain `model.*` layout — pin to skip on-load
    # re-detection by TextModelSpec.set_params.
    _pin_layer_prefix = True

    def __init__(self, hf_cfg: dict, engine_cfg, *, group_size: int = 0):
        super().__init__(hf_cfg, engine_cfg, group_size=group_size)

        # Fixed layer prefix for Qwen3
        self._layer_prefix = 'model.layers'
        self._embed_key = 'model.embed_tokens.weight'
        self._norm_key = 'model.norm.weight'

        self._n_experts = hf_cfg.get('num_experts', 0)

        # ---- Attention template ----
        dtype = self._cpp_dtype()
        self._attn_cfg = _tm.AttentionConfig()
        self._attn_cfg.hidden_dim  = self._hidden_units
        self._attn_cfg.head_dim    = self._head_dim
        self._attn_cfg.head_num    = self._head_num
        self._attn_cfg.kv_head_num = self._kv_head_num_padded
        self._attn_cfg.has_bias    = hf_cfg.get('attention_bias', 0)
        self._attn_cfg.qk_norm     = True
        self._attn_cfg.rope_dim    = self._rope.dim
        self._attn_cfg.window_size = 0
        self._attn_cfg.tp_size     = engine_cfg.attn_tp_size
        self._attn_cfg.data_type   = dtype
        self._attn_cfg.softmax_scale          = self._softmax_scale
        self._attn_cfg.max_position_embeddings = self._max_position_embeddings

        # ---- FFN template ----
        self._ffn_cfg = _tm.FfnConfig()
        self._ffn_cfg.hidden_dim = self._hidden_units
        self._ffn_cfg.has_bias   = False
        self._ffn_cfg.tp_size    = engine_cfg.mlp_tp_size
        self._ffn_cfg.data_type  = dtype
        self._ffn_cfg.act_type   = _act_type_id('silu')
        # fuse_silu / fused_moe / inter_size set per-call in ffn()/moe()

        # ---- MoE template (only if MoE variant) ----
        if self._n_experts > 0:
            self._moe_cfg = _tm.MoeConfig()
            self._moe_cfg.method            = 1  # kFused
            self._moe_cfg.experts_per_token = hf_cfg.get('num_experts_per_tok', 8)
            self._moe_cfg.norm_topk_prob    = hf_cfg.get('norm_topk_prob', False)
            self._moe_cfg.shared_gate       = False
            self._moe_cfg.routed_scale      = 1.0
            self._moe_cfg.router_bias       = False
            self._moe_cfg.topk_group        = 1
            self._moe_cfg.topk_method       = 'greedy'
            self._moe_cfg.n_group           = 1
            self._moe_cfg.scoring_func      = 'softmax'
            self._moe_cfg.router_n_groups   = 0
            self._moe_cfg.hidden_dim        = self._hidden_units
            self._moe_cfg.mlp_bias          = False
            self._moe_cfg.data_type         = dtype
            self._moe_cfg.tp_size           = engine_cfg.mlp_tp_size
            self._moe_cfg.act_type          = _act_type_id('silu')
            self._moe_cfg.fuse_silu         = True

            self._expert_inter_size_padded = _pad_inter_size(
                hf_cfg.get('moe_intermediate_size', 768),
                self._group_size, engine_cfg.mlp_tp_size)
        else:
            self._expert_inter_size_padded = 0

        # ---- Per-layer inter_size (dense FFN) ----
        raw_inter = hf_cfg.get('intermediate_size', 0) if self._n_experts == 0 else 0
        self._inter_sizes_padded = [
            _pad_inter_size(raw_inter, self._group_size,
                            engine_cfg.mlp_tp_size)
            for _ in range(self._num_layer)
        ]
        self._expert_nums = (
            [self._n_experts] * self._num_layer if self._n_experts > 0 else []
        )

    # ------------------------------------------------------------------
    # model() — walks full hierarchy (same as existing code)
    # ------------------------------------------------------------------

    def model(self):
        root = TextModelBuilder(self._root_handles, self._contexts)
        root.tok_embeddings = self.token_embeds(self._embed_key)
        root.norm = self.output_norm(self._norm_key)
        lm_key = self._embed_key if self._tie_embeddings else 'lm_head.weight'
        root.output = self.lm_head(lm_key)
        root.layers = self.layers(self._layer_prefix)

    # ------------------------------------------------------------------
    # Norm variants (Qwen3 uses standard RMSNorm, no zero-centering)
    # ------------------------------------------------------------------

    def output_norm(self, key):
        from ..builder import NormBuilder, make_norm_config
        w = self._get(key)
        cfg = make_norm_config(dim=self._hidden_units, data_type=self._cpp_dtype())
        m = NormBuilder(cfg, self._contexts)
        m.set_weight(w)
        return m

    def norm(self, key):
        return self.output_norm(key)

    # ------------------------------------------------------------------
    # Attention / FFN / MoE factories
    # ------------------------------------------------------------------

    def attn(self, pfx, layer):
        q = self._linear(f'{pfx}.q_proj')
        k = self._linear(f'{pfx}.k_proj')
        v = self._linear(f'{pfx}.v_proj')
        o = self._linear(f'{pfx}.o_proj')

        q = reorder_rotary_emb_linear(q, self._head_dim, self._rope.dim)
        k = reorder_rotary_emb_linear(k, self._head_dim, self._rope.dim)

        cfg = self._attn_cfg.clone()
        # No per-layer attention fields for Qwen3 (no sliding window).
        attn = AttentionBuilder(cfg, self._contexts,
                                tp=self.engine_cfg.attn_tp_size,
                                ranks=self._attn_ranks)

        attn.add_qkv_proj(q, k, v)
        attn.add_o_proj(o)

        q_norm = self._get(f'{pfx}.q_norm.weight')
        k_norm = self._get(f'{pfx}.k_norm.weight')
        q_norm = reorder_rotary_emb(q_norm, self._head_dim, self._rope.dim)
        k_norm = reorder_rotary_emb(k_norm, self._head_dim, self._rope.dim)
        attn.add_qk_norm(q_norm, k_norm)

        return attn

    def ffn(self, pfx, layer, inter_size=None, fused_moe=False):
        w1 = self._linear(f'{pfx}.gate_proj')
        w3 = self._linear(f'{pfx}.up_proj')
        w2 = self._linear(f'{pfx}.down_proj')

        cfg = self._ffn_cfg.clone()
        cfg.inter_size = (inter_size if inter_size is not None
                          else self._inter_sizes_padded[layer])
        cfg.fuse_silu  = False
        cfg.fused_moe  = fused_moe

        m = FfnBuilder(cfg, self._contexts,
                       tp=self.engine_cfg.mlp_tp_size,
                       ranks=self._mlp_ranks)
        m.add_ffn(w1, w2, w3)
        return m

    def moe(self, pfx, layer):
        if self.num_experts(layer) <= 0:
            return None

        cfg = self._moe_cfg.clone()
        cfg.layer_id   = layer
        cfg.expert_num = self._expert_nums[layer]
        cfg.inter_size = self._expert_inter_size_padded

        m = MoeBuilder(cfg, self._contexts,
                       tp=self.engine_cfg.mlp_tp_size,
                       ranks=self._mlp_ranks)

        gate_w = self._get(f'{pfx}.gate.weight')
        gate_w = gate_w.t() if gate_w.dim() > 1 else gate_w
        m.add_gate('gate', Linear({'weight': gate_w}),
                   model_dtype=self._cpp_dtype())

        experts = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for e in range(self.num_experts(layer)):
            experts[str(e)] = self.ffn(
                f'{pfx}.experts.{e}', layer,
                inter_size=self._expert_inter_size_padded, fused_moe=True)
        m.experts = experts
        return m

    def layers(self, pfx):
        layers = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for i in range(self._num_layer):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._contexts)
            d.attention_norm = self.norm(
                f'{pfx}.{i}.input_layernorm.weight')
            d.attention = self.attn(f'{pfx}.{i}.self_attn', i)
            d.ffn_norm = self.norm(
                f'{pfx}.{i}.post_attention_layernorm.weight')
            if self.num_experts(i) > 0:
                d.moe_ffn = self.moe(f'{pfx}.{i}.mlp', i)
            else:
                d.feed_forward = self.ffn(f'{pfx}.{i}.mlp', i)
            layers[str(i)] = d
        return layers

    def num_experts(self, layer: int) -> int:
        return self._n_experts
