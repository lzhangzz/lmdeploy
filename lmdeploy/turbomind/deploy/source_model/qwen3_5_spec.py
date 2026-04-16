# Copyright (c) OpenMMLab. All rights reserved.
"""Qwen3.5 TextModelSpec for the new pipeline.

Demonstrates the composable-ops pipeline with:
  - Mixed full-attention and linear-attention (Gated Delta Net) layers
  - Zero-centered RMSNorm (+1.0 transform)
  - MoE (optional) with shared expert and shared gate
  - Mixed AWQ: attention in fp16 while FFN/experts stay quantized
  - Linear attention scalar params (A_log, dt_bias)
"""
from __future__ import annotations

import re

import torch

from ..builder import (
    AttentionBuilder, DeltaNetBuilder, DecoderLayerBuilder, FfnBuilder,
    MoeBuilder, ModuleListBuilder, NormBuilder, TextModelBuilder,
    _act_type_id,
)
from ..kind_map import build_linear
from ..linear import Linear
from ..module_configs import (
    DecoderLayerConfig, ModuleListConfig,
    make_attention_config, make_deltanet_config, make_ffn_config,
    make_moe_config, make_norm_config,
)
from ..spec import TextModelSpec
from .base import INPUT_MODELS, BaseInputModel
from .utils import parse_rope_param, reorder_rotary_emb, reorder_rotary_emb_linear

_LAYER_PATTERN = r'(?:model\.language_model\.|model\.)layers\.([0-9]+)\.'


def map_packed_qwen35_experts(name: str) -> str:
    """Map packed expert names to weight names so that parameter.py can classify them.

    Only matches names ending without ``.weight``; a no-op for already-unpacked checkpoints.
    """
    return re.sub(r'(mlp\.experts\.(?:gate_up|down)_proj)$', r'\1.weight', name)


def _qwen35_model_info_base(cfg: dict) -> dict:
    """Build the common model_info dict for all Qwen3.5 variants."""
    attn_head_num = cfg['num_attention_heads']
    hidden_units = cfg['hidden_size']
    head_dim = cfg.get('head_dim', None) or hidden_units // attn_head_num
    rope_param, max_position_embeddings = parse_rope_param(cfg, head_dim)

    # partial_rotary_factor adjusts RoPE dim
    rope_params = cfg.get('rope_parameters', {})
    partial_rotary_factor = rope_params.get('partial_rotary_factor', cfg.get('partial_rotary_factor', 1.0))
    if partial_rotary_factor < 1.0:
        rope_param.dim = int(head_dim * partial_rotary_factor)

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

    layer_types = cfg.get('layer_types', [])
    if layer_types:
        info.update(
            layer_types=layer_types,
            linear_key_head_dim=cfg.get('linear_key_head_dim', 0),
            linear_value_head_dim=cfg.get('linear_value_head_dim', 0),
            linear_conv_kernel_dim=cfg.get('linear_conv_kernel_dim', 0),
            linear_num_key_heads=cfg.get('linear_num_key_heads', 0),
            linear_num_value_heads=cfg.get('linear_num_value_heads', 0),
            attn_output_gate=cfg.get('attn_output_gate', False),
        )

    return info


class Qwen3_5Spec(TextModelSpec):
    """Weight spec for Qwen3.5 (dense + linear attention + optional MoE)."""

    def __init__(self, params: dict[str, torch.Tensor], model_cfg: dict):
        self.params = params
        self.cfg = model_cfg
        self._layer_types = model_cfg.get("layer_types", [])
        self._num_layer = model_cfg["num_hidden_layers"]
        self._n_experts = model_cfg.get("num_experts", 0)

        # QKV dimensions for GDN layers: Q/K share key heads, V uses value heads
        ln_key_heads = model_cfg.get("linear_num_key_heads", 0)
        ln_val_heads = model_cfg.get("linear_num_value_heads", 0)
        ln_key_dim = model_cfg.get("linear_key_head_dim", 0)
        ln_val_dim = model_cfg.get("linear_value_head_dim", 0)
        if ln_key_heads and ln_val_heads:
            q_dim = ln_key_heads * ln_key_dim
            k_dim = ln_key_heads * ln_key_dim
            v_dim = ln_val_heads * ln_val_dim
            self._linear_qkv_split = (q_dim, k_dim, v_dim)
        else:
            self._linear_qkv_split = None

        if any(k.startswith("model.language_model.") for k in params):
            self._layer_prefix = "model.language_model.layers"
            self._embed_key = "model.language_model.embed_tokens.weight"
            self._norm_key = "model.language_model.norm.weight"
        else:
            self._layer_prefix = "model.layers"
            self._embed_key = "model.embed_tokens.weight"
            self._norm_key = "model.norm.weight"

    def _is_linear_attn(self, layer: int) -> bool:
        return (layer < len(self._layer_types)
                and self._layer_types[layer] == "linear_attention")

    def _is_moe_layer(self, layer: int) -> bool:
        return self._n_experts > 0

    # ------------------------------------------------------------------
    # Spec-driven loading: build full model hierarchy
    # ------------------------------------------------------------------

    def model(self):
        root = TextModelBuilder(self._root_handles, self._contexts)
        root.tok_embeddings = self.token_embeds(self._embed_key)
        root.norm = self.output_norm(self._norm_key)
        tie = self.cfg.get("tie_word_embeddings", False)
        lm_key = self._embed_key if tie else "lm_head.weight"
        root.output = self.lm_head(lm_key)
        root.layers = self.layers(self._layer_prefix)

    # ------------------------------------------------------------------
    # Factory methods: read weights, create builders, return them
    # ------------------------------------------------------------------

    def output_norm(self, key):
        w = self._zero_centered(self._get(key))
        cfg = make_norm_config(dim=self._mc.hidden_units, data_type=self._cpp_dtype())
        m = NormBuilder(cfg, self._contexts)
        m.set_weight(w)
        return m

    def norm(self, key):
        w = self._zero_centered(self._get(key))
        cfg = make_norm_config(dim=self._mc.hidden_units, data_type=self._cpp_dtype())
        m = NormBuilder(cfg, self._contexts)
        m.set_weight(w)
        return m

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

        window_size = 0
        ws_list = mc.window_size
        if ws_list and layer < len(ws_list):
            window_size = ws_list[layer]

        attn_cfg = make_attention_config(
            mc, tp_size=tp, dtype=dtype,
            rope_dim=self._rope_dim)
        attn = AttentionBuilder(attn_cfg, self._contexts,
                                tp=tp, ranks=self._attn_ranks)

        attn.add_qkv_proj(q, k, v)
        attn.add_o_proj(o)

        # Inline qk norm
        q_norm = self._zero_centered(self._get(f"{pfx}.q_norm.weight"))
        k_norm = self._zero_centered(self._get(f"{pfx}.k_norm.weight"))
        q_norm = reorder_rotary_emb(q_norm, mc.size_per_head, self._rope_dim)
        k_norm = reorder_rotary_emb(k_norm, mc.size_per_head, self._rope_dim)
        attn.add_qk_norm(q_norm, k_norm)

        return attn

    def linear_attn(self, pfx, layer):
        """Return DeltaNetBuilder for linear-attention (Gated Delta Net)."""
        mc = self._mc
        tp = self._attn_tp
        dtype = self._cpp_dtype()

        dn_cfg = make_deltanet_config(
            mc, tp_size=tp, dtype=dtype)
        builder = DeltaNetBuilder(dn_cfg, self._contexts,
                                  tp=tp, ranks=self._attn_ranks)

        builder.add_input_projections(
            in_proj_qkv=self._linear(f"{pfx}.in_proj_qkv"),
            in_proj_z=self._linear(f"{pfx}.in_proj_z"),
            in_proj_b=self._linear(f"{pfx}.in_proj_b"),
            in_proj_a=self._linear(f"{pfx}.in_proj_a"),
            out_proj=self._linear(f"{pfx}.out_proj"),
            qkv_split=self._linear_qkv_split)
        builder.add_scalar_params(
            a_log=self._get(f"{pfx}.A_log"),
            dt_bias=self._get(f"{pfx}.dt_bias"))
        builder.add_conv1d(
            self._get(f"{pfx}.conv1d.weight"),
            qkv_split=self._linear_qkv_split)
        builder.add_norm(
            self._get(f"{pfx}.norm.weight"), data_type=dtype)
        return builder

    def ffn(self, pfx, layer, inter_size=None, fused_moe=False):
        """Return FfnBuilder for the given layer, or None if no weights found."""
        w1 = self._linear(f"{pfx}.gate_proj")
        w3 = self._linear(f"{pfx}.up_proj")
        w2 = self._linear(f"{pfx}.down_proj")

        if w1 is None and w2 is None and w3 is None:
            return None

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
        # Shared expert gate
        sg = self._get(f'{pfx}.shared_expert_gate.weight')
        sg = sg.t() if sg.dim() > 1 else sg
        m.add_gate('shared_gate', Linear({"weight": sg}), model_dtype=dtype)

        expert_inter = mc.expert_inter_size or 0
        experts = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for e in range(self.num_experts(layer)):
            experts[str(e)] = self._moe_expert_ffn(pfx, layer, e, expert_inter)

        m.experts = experts
        return m

    def layers(self, pfx):
        """Return ModuleListBuilder with all decoder layers."""
        layers = ModuleListBuilder(ModuleListConfig(), self._contexts)

        for i in range(self._num_layer):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._contexts)
            d.attention_norm = self.norm(
                f'{pfx}.{i}.input_layernorm.weight')
            if self._is_linear_attn(i):
                d.linear_attn = self.linear_attn(
                    f'{pfx}.{i}.linear_attn', layer=i)
            else:
                d.attention = self.attn(
                    f'{pfx}.{i}.self_attn', layer=i)
            d.ffn_norm = self.norm(
                f'{pfx}.{i}.post_attention_layernorm.weight')
            if self.num_experts(i) > 0:
                d.feed_forward = self.ffn(
                    f'{pfx}.{i}.mlp.shared_expert', layer=i)
                d.moe_ffn = self.moe(
                    f'{pfx}.{i}.mlp', layer=i)
            else:
                d.feed_forward = self.ffn(
                    f'{pfx}.{i}.mlp', layer=i)
            layers[str(i)] = d

        return layers

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _zero_centered(self, w: torch.Tensor | None) -> torch.Tensor | None:
        """Zero-centered RMSNorm: add 1.0."""
        if w is not None:
            return w.float() + 1.0
        return None

    def _moe_expert_ffn(self, pfx, layer, expert_idx, inter_size):
        """Build FfnBuilder for one MoE expert, handling packed format."""
        expert_pfx = f'{pfx}.experts.{expert_idx}'
        # Try standard per-expert format first
        result = self.ffn(expert_pfx, layer,
                          inter_size=inter_size, fused_moe=True)
        if result is not None:
            return result
        # Fall back to packed format
        packed_pfx = f'{pfx}.experts'
        return self._packed_moe_expert_indexed(packed_pfx, expert_idx, inter_size)

    def _packed_moe_expert_indexed(self, pfx, expert_idx, inter_size):
        """Read a single expert from packed tensors by index."""
        gate_up_lin = build_linear(self.params, f"{pfx}.gate_up_proj", index=expert_idx)
        down_lin = build_linear(self.params, f"{pfx}.down_proj", index=expert_idx)
        if gate_up_lin is None or down_lin is None:
            return None

        # gate_up is in TM layout [in, 2*out]; split along output dim.
        gate_tensors: dict[str, torch.Tensor] = {}
        up_tensors: dict[str, torch.Tensor] = {}
        for kind, t in gate_up_lin.tensors.items():
            half = t.shape[-1] // 2
            gate_tensors[kind] = t[..., :half].contiguous()
            up_tensors[kind] = t[..., half:].contiguous()

        linears = {
            "w1": Linear(tensors=gate_tensors, weight_format=gate_up_lin.weight_format),
            "w2": down_lin,
            "w3": Linear(tensors=up_tensors, weight_format=gate_up_lin.weight_format),
        }

        mc = self._mc
        tp = self._mlp_tp
        dtype = self._cpp_dtype()

        ffn_cfg = make_ffn_config(
            mc, tp_size=tp, dtype=dtype,
            act_type=_act_type_id(mc.activation_type),
            fuse_silu=False, inter_size=inter_size,
            fused_moe=True)
        m = FfnBuilder(ffn_cfg, self._contexts, tp=tp, ranks=self._mlp_ranks)
        m.add_ffn(linears.get('w1'), linears.get('w2'), linears.get('w3'))
        return m

    def num_experts(self, layer: int) -> int:
        return self._n_experts


@INPUT_MODELS.register_module(name='qwen3_5')
class Qwen3_5InputModel(BaseInputModel):
    """Input model for Qwen3.5 (dense + optional linear attention)."""

    _layer_pattern = _LAYER_PATTERN
    _spec_class = Qwen3_5Spec

    def model_info(self) -> dict:
        cfg = self.model_config
        info = _qwen35_model_info_base(cfg)
        info.update(
            expert_num=cfg.get('num_experts', 0),
            expert_inter_size=cfg.get('moe_intermediate_size', 0),
            experts_per_token=cfg.get('num_experts_per_tok', 0),
            moe_shared_gate=True,
            scoring_func='softmax',
            norm_topk_prob=True,
        )
        shared_expert_size = cfg.get('shared_expert_intermediate_size')
        if shared_expert_size is not None:
            info['inter_size'] = shared_expert_size
        return info


@INPUT_MODELS.register_module(name='qwen3_5-moe')
class Qwen3_5MoeInputModel(BaseInputModel):
    """Input model for Qwen3.5-MoE."""

    _layer_pattern = _LAYER_PATTERN
    _spec_class = Qwen3_5Spec
    _loader_mappings = [map_packed_qwen35_experts]

    def model_info(self) -> dict:
        cfg = self.model_config
        info = _qwen35_model_info_base(cfg)
        info.update(
            expert_num=cfg.get('num_experts', 0),
            expert_inter_size=cfg.get('moe_intermediate_size', 0),
            experts_per_token=cfg.get('num_experts_per_tok', 0),
            inter_size=cfg.get('shared_expert_intermediate_size', 0),
            moe_shared_gate=True,
            scoring_func='softmax',
            norm_topk_prob=True,
        )
        return info
