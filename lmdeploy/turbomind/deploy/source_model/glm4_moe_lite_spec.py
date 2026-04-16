# Copyright (c) OpenMMLab. All rights reserved.
"""GLM-4 MoE Lite (GLM-4.7-Flash) TextModelSpec for the new pipeline.

Demonstrates the composable-ops pipeline with:
  - MLA (Multi-head Latent Attention)
  - MoE experts with dense first layer
  - noaux_tc routing with score correction bias
  - Raw tensors for norms, router, embeddings
"""
from __future__ import annotations

import os

import torch

from ..builder import (
    DecoderLayerBuilder, FfnBuilder,
    MLABuilder, MoeBuilder, ModuleListBuilder, TextModelBuilder,
    _act_type_id,
)
from ..linear import Linear
from ..module_configs import (
    DecoderLayerConfig, FfnConfig,
    MLAConfig, ModuleListConfig, MoeConfig,
)
from ..spec import TextModelSpec
from .base import INPUT_MODELS, BaseInputModel
from .utils import get_yarn_params, parse_rope_param

_LAYER_PATTERN = r'model\.layers\.([0-9]+).'


class Glm4MoeLiteSpec(TextModelSpec):
    """Weight spec for GLM-4 MoE Lite (e.g. GLM-4.7-Flash).

    Uses same key layout as DeepSeek2: ``model.layers.{i}.self_attn.*``,
    ``model.layers.{i}.mlp.*``.  First layer is dense FFN; remaining layers
    are MoE with ``n_routed_experts`` experts per layer.
    """

    _layer_prefix = "model.layers"

    def __init__(self, params: dict[str, torch.Tensor], model_cfg: dict):
        self.params = params
        self.cfg = model_cfg
        self._num_layer = model_cfg["num_hidden_layers"]
        self._n_experts = model_cfg.get("n_routed_experts", 0)
        self._dense_layers = model_cfg.get("first_k_dense_replace", 1)

    # ------------------------------------------------------------------
    # Builder-driven loading: build full model hierarchy
    # ------------------------------------------------------------------

    def model(self):
        root = TextModelBuilder(self._root_handles, self._contexts)
        root.tok_embeddings = self.token_embeds('model.embed_tokens.weight')
        root.norm = self.output_norm('model.norm.weight')
        root.output = self.lm_head('lm_head.weight')
        root.layers = self.layers('model.layers')

    # ------------------------------------------------------------------
    # Factory methods: read weights, create builders, return them
    # ------------------------------------------------------------------

    def attn(self, pfx, layer):
        """Return MLABuilder for MLA attention."""
        mc = self._mc
        tp = self._attn_tp
        dtype = self._cpp_dtype()
        cfg = self.cfg

        mla_cfg = MLAConfig.from_model_config(
            mc, tp_size=tp, tp_rank=0, dtype=dtype, window_size=-1,
            qk_nope_dim=cfg['qk_nope_head_dim'])
        builder = MLABuilder(mla_cfg, self._contexts,
                             tp=tp, ranks=self._attn_ranks)

        q_b = self._linear(f"{pfx}.q_b_proj") or self._linear(f"{pfx}.q_proj")
        builder.add_projections(
            q_a_proj=self._linear(f"{pfx}.q_a_proj"),
            q_b_proj=q_b,
            kv_a_proj=self._linear(f"{pfx}.kv_a_proj_with_mqa"),
            kv_b_proj=self._linear(f"{pfx}.kv_b_proj"),
            wo=self._linear(f"{pfx}.o_proj"),
        )
        builder.add_norms(
            q_a_norm=self._get(f"{pfx}.q_a_layernorm.weight"),
            kv_a_norm=self._get(f"{pfx}.kv_a_layernorm.weight"),
            data_type=dtype,
        )
        return builder

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

        ffn_cfg = FfnConfig.from_model_config(
            mc, tp_size=tp, tp_rank=0, dtype=dtype,
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

        moe_cfg = MoeConfig.from_model_config(
            mc, layer_id=layer, tp_size=tp, tp_rank=0, dtype=dtype,
            act_type=_act_type_id(mc.activation_type),
            fuse_silu=True, expert_num=expert_num)
        m = MoeBuilder(moe_cfg, self._contexts, tp=tp, ranks=self._mlp_ranks)

        # Inline gate read
        gate_w = self._get(f'{pfx}.gate.weight')
        gate_w = gate_w.t() if gate_w.dim() > 1 else gate_w
        tensors = {"weight": gate_w}
        gate_bias = self._get(f'{pfx}.gate.bias')
        if gate_bias is not None:
            tensors["bias"] = gate_bias
        m.add_gate('gate', Linear(tensors), model_dtype=dtype)

        # Inline score correction bias
        correction = self._get(
            f'{pfx}.gate.e_score_correction_bias')
        m.add_param("score_correction_bias", correction)

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
                f'{pfx}.{i}.self_attn', layer=i)
            d.ffn_norm = self.norm(
                f'{pfx}.{i}.post_attention_layernorm.weight')
            if i < self._dense_layers:
                # Dense FFN layer
                d.feed_forward = self.ffn(
                    f'{pfx}.{i}.mlp', layer=i)
            else:
                # MoE layer: shared expert as feed_forward, routed as moe_ffn
                d.feed_forward = self.ffn(
                    f'{pfx}.{i}.mlp.shared_experts', layer=i)
                d.moe_ffn = self.moe(
                    f'{pfx}.{i}.mlp', layer=i)
            layers[str(i)] = d

        return layers

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def num_experts(self, layer: int) -> int:
        if layer < self._dense_layers:
            return 0
        return self._n_experts


@INPUT_MODELS.register_module(name='glm4-moe-lite')
class Glm4MoeLiteInputModel(BaseInputModel):
    """Input model for GLM-4 MoE Lite (e.g. GLM-4.7-Flash)."""

    _layer_pattern = _LAYER_PATTERN
    _spec_class = Glm4MoeLiteSpec

    def model_info(self) -> dict:
        cfg = self.model_config

        # --- base transformer fields ---
        attn_head_num = cfg['num_attention_heads']
        hidden_units = cfg['hidden_size']

        # --- MLA head geometry ---
        qk_nope_dim = cfg['qk_nope_head_dim']
        qk_rope_dim = cfg['qk_rope_head_dim']
        kv_lora_rank = cfg['kv_lora_rank']
        q_head_dim = qk_nope_dim + qk_rope_dim
        size_per_head = q_head_dim
        v_head_dim = cfg['v_head_dim']
        softmax_scale = 0.0
        disable_mla_fold = os.getenv('LMDEPLOY_MLA_FOLD', '1').lower() in ('0', 'false', 'no')
        if kv_lora_rank and kv_lora_rank != qk_nope_dim and not disable_mla_fold:
            size_per_head = kv_lora_rank + qk_rope_dim
            v_head_dim = kv_lora_rank
            softmax_scale = q_head_dim**(-0.5)
        elif kv_lora_rank and kv_lora_rank != qk_nope_dim:
            softmax_scale = q_head_dim**(-0.5)

        # --- RoPE (dim = qk_rope_dim for MLA) ---
        rope_param, max_position_embeddings = parse_rope_param(cfg, qk_rope_dim)

        # --- MoE layout ---
        num_layer = cfg['num_hidden_layers']
        n_routed_experts = cfg.get('n_routed_experts', 0)
        n_shared_experts = cfg.get('n_shared_experts', 1)
        expert_inter_size = cfg['moe_intermediate_size']
        first_k_dense = cfg.get('first_k_dense_replace', 1)
        expert_num = [n_routed_experts] * num_layer
        for i in range(first_k_dense):
            expert_num[i] = 0
        inter_size = [n_shared_experts * expert_inter_size] * num_layer
        inter_size[0] = cfg.get('intermediate_size', n_shared_experts * expert_inter_size)

        # Ensure required routing fields exist (GLM may omit them)
        topk_method = cfg.get('topk_method', 'noaux_tc')
        topk_group = cfg.get('topk_group', 1)
        n_group = cfg.get('n_group', 1)
        scoring_func = cfg.get('scoring_func', 'sigmoid')

        info = dict(
            num_layer=num_layer,
            norm_eps=cfg['rms_norm_eps'],
            head_num=attn_head_num,
            kv_head_num=1,
            hidden_units=hidden_units,
            size_per_head=size_per_head,
            vocab_size=cfg['vocab_size'],
            max_position_embeddings=max_position_embeddings,
            rope_param=rope_param,
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=cfg.get('q_lora_rank') or 0,
            qk_rope_dim=qk_rope_dim,
            v_head_dim=v_head_dim,
            inter_size=inter_size,
            expert_num=expert_num,
            expert_inter_size=expert_inter_size,
            experts_per_token=cfg['num_experts_per_tok'],
            norm_topk_prob=cfg.get('norm_topk_prob', True),
            routed_scale=cfg.get('routed_scaling_factor', 1.0),
            topk_method=topk_method,
            topk_group=topk_group,
            moe_group_num=n_group,
            scoring_func=scoring_func,
            tune_layer_num=2,
        )
        if softmax_scale:
            info['softmax_scale'] = softmax_scale

        # YaRN RoPE for MLA (override attention_factor + softmax_scale)
        if 'rope_parameters' in cfg:
            rope_scaling = cfg['rope_parameters']
        else:
            rope_scaling = cfg.get('rope_scaling')
        if rope_scaling and rope_scaling.get('type') == 'yarn':
            attention_factor, yarn_scale = get_yarn_params(rope_scaling)
            yarn_scale *= q_head_dim**(-0.5)
            rope_param.max_position_embeddings = rope_scaling['original_max_position_embeddings']
            rope_param.attention_factor = attention_factor
            info.update(rope_param=rope_param, softmax_scale=yarn_scale)

        if 'router_n_groups' in cfg and cfg['router_n_groups'] > 0:
            info['router_n_groups'] = cfg['router_n_groups']

        # GLM-specific overrides
        info['topk_method'] = 'noaux_tc'
        info['scoring_func'] = 'sigmoid'

        return info
