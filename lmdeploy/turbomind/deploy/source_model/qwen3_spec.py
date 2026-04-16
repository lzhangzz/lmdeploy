# Copyright (c) OpenMMLab. All rights reserved.
"""Qwen3 TextModelSpec for the new pipeline.

Qwen3 is a standard Llama-like model with QK norm and optional MoE.
No shared expert in the MoE variant, no linear attention, no zero-centered norm.
"""
from __future__ import annotations

import torch

from ..builder import (
    AttentionBuilder, DecoderLayerBuilder, FfnBuilder,
    MoeBuilder, ModuleListBuilder, TextModelBuilder,
    _act_type_id,
)
from ..linear import Linear
from ..module_configs import (
    DecoderLayerConfig, ModuleListConfig,
    make_attention_config, make_ffn_config, make_moe_config,
)
from ..spec import TextModelSpec
from .base import INPUT_MODELS, BaseInputModel
from .utils import parse_rope_param, reorder_rotary_emb, reorder_rotary_emb_linear

_LAYER_PATTERN = r'model\.layers\.([0-9]+).'


class Qwen3TextSpec(TextModelSpec):
    """Weight spec for Qwen3 (dense) and Qwen3-MoE."""

    _layer_prefix = "model.layers"

    def __init__(self, params: dict[str, torch.Tensor], model_cfg: dict):
        self.params = params
        self.cfg = model_cfg
        self._num_layer = model_cfg["num_hidden_layers"]
        self._n_experts = model_cfg.get("num_experts", 0)

    # ------------------------------------------------------------------
    # Spec-driven loading: build full model hierarchy
    # ------------------------------------------------------------------

    def model(self):
        root = TextModelBuilder(self._root_handles, self._contexts)
        root.tok_embeddings = self.token_embeds('model.embed_tokens.weight')
        root.norm = self.output_norm('model.norm.weight')
        tie = self.cfg.get("tie_word_embeddings", False)
        lm_key = "model.embed_tokens.weight" if tie else "lm_head.weight"
        root.output = self.lm_head(lm_key)
        root.layers = self.layers('model.layers')

    # ------------------------------------------------------------------
    # Factory methods: read weights, create builders, return them
    # ------------------------------------------------------------------

    def attn(self, pfx, layer):
        """Return AttentionBuilder for the given layer."""
        q = self._linear(f"{pfx}.q_proj")
        k = self._linear(f"{pfx}.k_proj")
        v = self._linear(f"{pfx}.v_proj")
        o = self._linear(f"{pfx}.o_proj")

        q = reorder_rotary_emb_linear(q, self._mc.size_per_head, self._rope_dim)
        k = reorder_rotary_emb_linear(k, self._mc.size_per_head, self._rope_dim)

        mc = self._mc
        tp = self._attn_tp
        dtype = self._cpp_dtype()

        attn_cfg = make_attention_config(
            mc, tp_size=tp, dtype=dtype,
            rope_dim=self._rope_dim)
        attn = AttentionBuilder(attn_cfg, self._contexts,
                                tp=tp, ranks=self._attn_ranks)

        attn.add_qkv_proj(q, k, v)
        attn.add_o_proj(o)

        # Inline qk norm
        q_norm = self._get(f"{pfx}.q_norm.weight")
        k_norm = self._get(f"{pfx}.k_norm.weight")
        q_norm = reorder_rotary_emb(q_norm, mc.size_per_head, self._rope_dim)
        k_norm = reorder_rotary_emb(k_norm, mc.size_per_head, self._rope_dim)
        attn.add_qk_norm(q_norm, k_norm)

        return attn

    def ffn(self, pfx, layer, inter_size=None, fused_moe=False):
        """Return FfnBuilder for the given layer."""
        w1 = self._linear(f"{pfx}.gate_proj")
        w3 = self._linear(f"{pfx}.up_proj")
        w2 = self._linear(f"{pfx}.down_proj")

        mc = self._mc
        tp = self._mlp_tp
        dtype = self._cpp_dtype()

        if inter_size is None:
            is_list = mc.inter_size
            inter_size = is_list[layer] if is_list and layer < len(
                is_list) else 0

        ffn_cfg = make_ffn_config(
            mc, tp_size=tp, dtype=dtype,
            act_type=_act_type_id(mc.activation_type),
            fuse_silu=False, inter_size=inter_size,
            fused_moe=fused_moe)
        m = FfnBuilder(ffn_cfg, self._contexts, tp=tp, ranks=self._mlp_ranks)
        m.add_ffn(w1, w2, w3)
        return m

    def moe(self, pfx, layer):
        """Build MoeBuilder for the given MoE layer."""
        if self.num_experts(layer) <= 0:
            return None

        mc = self._mc
        tp = self._mlp_tp
        dtype = self._cpp_dtype()

        expert_num = 0
        en_list = mc.expert_num
        if en_list and layer < len(en_list):
            expert_num = en_list[layer]

        moe_cfg = make_moe_config(
            mc, layer_id=layer, tp_size=tp, dtype=dtype,
            act_type=_act_type_id(mc.activation_type),
            fuse_silu=True, expert_num=expert_num)
        m = MoeBuilder(moe_cfg, self._contexts, tp=tp, ranks=self._mlp_ranks)

        # Inline gate read
        gate_w = self._get(f'{pfx}.gate.weight')
        gate_w = gate_w.t() if gate_w.dim() > 1 else gate_w
        m.add_gate('gate', Linear({"weight": gate_w}), model_dtype=dtype)

        expert_inter = mc.expert_inter_size or 0
        experts = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for e in range(self.num_experts(layer)):
            experts[str(e)] = self.ffn(
                f'{pfx}.experts.{e}', layer,
                inter_size=expert_inter, fused_moe=True)

        m.experts = experts
        return m

    def layers(self, pfx):
        """Return ModuleListBuilder with all decoder layers."""
        layers = ModuleListBuilder(ModuleListConfig(), self._contexts)

        for i in range(self._num_layer):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._contexts)
            d.attention_norm = self.norm(
                f'{pfx}.{i}.input_layernorm.weight')
            d.attention = self.attn(
                f'{pfx}.{i}.self_attn', i)
            d.ffn_norm = self.norm(
                f'{pfx}.{i}.post_attention_layernorm.weight')
            if self.num_experts(i) > 0:
                d.moe_ffn = self.moe(
                    f'{pfx}.{i}.mlp', i)
            else:
                d.feed_forward = self.ffn(
                    f'{pfx}.{i}.mlp', i)
            layers[str(i)] = d

        return layers

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def num_experts(self, layer: int) -> int:
        return self._n_experts


@INPUT_MODELS.register_module(name='qwen3-moe')
@INPUT_MODELS.register_module(name='qwen3')
class Qwen3InputModel(BaseInputModel):
    """Input model for Qwen3 (dense and MoE)."""

    _layer_pattern = _LAYER_PATTERN
    _spec_class = Qwen3TextSpec

    def model_info(self) -> dict:
        cfg = self.model_config
        attn_head_num = cfg['num_attention_heads']
        hidden_units = cfg['hidden_size']
        head_dim = cfg.get('head_dim', None) or hidden_units // attn_head_num
        rope_param, max_position_embeddings = parse_rope_param(cfg, head_dim)
        info = dict(
            num_layer=cfg['num_hidden_layers'],
            norm_eps=cfg['rms_norm_eps'],
            head_num=attn_head_num,
            kv_head_num=cfg.get('num_key_value_heads', attn_head_num),
            hidden_units=hidden_units,
            size_per_head=head_dim,
            inter_size=cfg.get('intermediate_size', 0),
            vocab_size=cfg['vocab_size'],
            max_position_embeddings=max_position_embeddings,
            rope_param=rope_param,
            qk_norm=True,
            attn_bias=cfg.get('attention_bias', 0),
        )
        n_experts = cfg.get('num_experts', 0)
        if n_experts:
            info.update(
                expert_num=n_experts,
                experts_per_token=cfg.get('num_experts_per_tok', 8),
                expert_inter_size=cfg.get('moe_intermediate_size', 768),
                inter_size=0,
                norm_topk_prob=cfg.get('norm_topk_prob', False),
            )
        return info
