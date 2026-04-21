# Copyright (c) OpenMMLab. All rights reserved.
"""Qwen3.5 TextModelSpec for the new pipeline."""
from __future__ import annotations

import re

import torch

import _turbomind as _tm

from ..builder import (AttentionBuilder, DecoderLayerBuilder, DeltaNetBuilder,
                       FfnBuilder, MoeBuilder, ModuleListBuilder,
                       TextModelBuilder, _act_type_id)
from ..builder import DecoderLayerConfig, ModuleListConfig
from ..builder.attention import split_output_gate
from ..kind_map import build_linear
from ..linear import Linear
from ..spec import TextModelSpec
from .base import INPUT_MODELS
from .utils import layer_progress, reorder_rotary_emb, reorder_rotary_emb_linear

_LAYER_PATTERN = r'(?:model\.language_model\.|model\.)layers\.([0-9]+)\.'


def map_packed_qwen35_experts(name: str) -> str:
    """Map packed expert names to weight names so parameter.py can classify."""
    return re.sub(r'(mlp\.experts\.(?:gate_up|down)_proj)$', r'\1.weight', name)


@INPUT_MODELS.register_module(name='qwen3_5-moe')
@INPUT_MODELS.register_module(name='qwen3_5')
class Qwen3_5Spec(TextModelSpec):
    """Weight spec for Qwen3.5 (dense + linear-attn + optional MoE).

    ``_pin_layer_prefix`` is intentionally left at its default False — Qwen3.5
    may be packaged as a multimodal root where the decoder lives under
    ``model.language_model.*``. The base-class ``set_params`` re-runs
    ``detect_layer_prefix`` when weights arrive to resolve the correct
    prefix.
    """

    _layer_pattern = _LAYER_PATTERN
    _loader_mappings = [map_packed_qwen35_experts]

    def __init__(self, hf_cfg: dict, engine_cfg, *, group_size: int = 0):
        super().__init__(hf_cfg, engine_cfg, group_size=group_size)

        # partial_rotary_factor adjusts rope_dim
        rope_params = hf_cfg.get('rope_parameters', {})
        partial_factor = rope_params.get(
            'partial_rotary_factor', hf_cfg.get('partial_rotary_factor', 1.0))
        if partial_factor < 1.0:
            self._rope.dim = int(self._head_dim * partial_factor)

        self._layer_types = hf_cfg.get('layer_types', [])
        self._n_experts = hf_cfg.get('num_experts', 0)
        dtype = self._cpp_dtype()

        # ---- Attention template ----
        self._attn_cfg = _tm.AttentionConfig()
        self._attn_cfg.hidden_dim       = self._hidden_units
        self._attn_cfg.head_dim         = self._head_dim
        self._attn_cfg.head_num         = self._head_num
        self._attn_cfg.kv_head_num      = self._kv_head_num
        self._attn_cfg.has_bias         = hf_cfg.get('attention_bias', 0)
        self._attn_cfg.qk_norm          = True
        self._attn_cfg.attn_output_gate = bool(self._layer_types) and \
                                          hf_cfg.get('attn_output_gate', False)
        self._apply_rope(self._attn_cfg.rope)
        self._attn_cfg.window_size      = 0
        self._attn_cfg.tp_size          = engine_cfg.attn_tp_size
        self._attn_cfg.data_type        = dtype
        self._attn_cfg.softmax_scale    = self._softmax_scale

        # ---- DeltaNet template (only if linear-attn layers present) ----
        if self._layer_types:
            ln_key_heads = hf_cfg['linear_num_key_heads']
            ln_val_heads = hf_cfg['linear_num_value_heads']
            ln_key_dim   = hf_cfg['linear_key_head_dim']
            ln_val_dim   = hf_cfg['linear_value_head_dim']

            self._dn_cfg = _tm.DeltaNetConfig()
            self._dn_cfg.hidden_dim      = self._hidden_units
            self._dn_cfg.num_k_heads     = ln_key_heads
            self._dn_cfg.num_v_heads     = ln_val_heads
            self._dn_cfg.key_head_dim    = ln_key_dim
            self._dn_cfg.value_head_dim  = ln_val_dim
            self._dn_cfg.d_conv          = hf_cfg.get('linear_conv_kernel_dim', 0) or 4
            self._dn_cfg.has_bias        = bool(self._attn_cfg.has_bias)
            self._dn_cfg.tp_size         = engine_cfg.attn_tp_size
            self._dn_cfg.data_type       = dtype

            q_dim = ln_key_heads * ln_key_dim
            v_dim = ln_val_heads * ln_val_dim
            self._linear_qkv_split = (q_dim, q_dim, v_dim)

        # ---- FFN template ----
        self._ffn_cfg = _tm.FfnConfig()
        self._ffn_cfg.hidden_dim = self._hidden_units
        self._ffn_cfg.has_bias   = False
        self._ffn_cfg.tp_size    = engine_cfg.mlp_tp_size
        self._ffn_cfg.data_type  = dtype
        self._ffn_cfg.act_type   = _act_type_id('silu')

        # ---- MoE template ----
        if self._n_experts > 0:
            self._moe_cfg = _tm.MoeConfig()
            self._moe_cfg.method            = 1
            self._moe_cfg.experts_per_token = hf_cfg['num_experts_per_tok']
            self._moe_cfg.norm_topk_prob    = True
            self._moe_cfg.shared_gate       = True
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

            self._expert_inter_size = hf_cfg['moe_intermediate_size']
            raw_shared = hf_cfg.get('shared_expert_intermediate_size', 0)
        else:
            self._expert_inter_size = 0
            raw_shared = hf_cfg.get('intermediate_size', 0)

        self._inter_sizes = [raw_shared] * self._num_layer
        self._expert_nums = (
            [self._n_experts] * self._num_layer if self._n_experts > 0 else []
        )

    def _is_linear_attn(self, layer: int) -> bool:
        return (layer < len(self._layer_types)
                and self._layer_types[layer] == 'linear_attention')

    def num_experts(self, layer: int) -> int:
        return self._n_experts

    # ------------------------------------------------------------------
    # model() — same topology as old code
    # ------------------------------------------------------------------

    def model(self):
        root = TextModelBuilder(self._root_handles, self._contexts)
        root.tok_embeddings = self.token_embeds(self._embed_key)
        root.norm = self.output_norm(self._norm_key)
        lm_key = self._embed_key if self._tie_embeddings else 'lm_head.weight'
        root.output = self.lm_head(lm_key)
        root.layers = self.layers(self._layer_prefix)

    # ------------------------------------------------------------------
    # Zero-centered norm
    # ------------------------------------------------------------------

    def output_norm(self, key):
        from ..builder import NormBuilder, make_norm_config
        w = self._zero_centered(self._get(key))
        cfg = make_norm_config(dim=self._hidden_units, data_type=self._cpp_dtype(), norm_eps=self._norm_eps)
        m = NormBuilder(cfg, self._contexts)
        m.set_weight(w)
        return m

    def norm(self, key):
        return self.output_norm(key)

    def _zero_centered(self, w):
        if w is not None:
            return w.float() + 1.0
        return None

    # ------------------------------------------------------------------
    # Attention / linear-attention factories
    # ------------------------------------------------------------------

    def attn(self, pfx, layer):
        q = self._linear(f'{pfx}.q_proj')
        k = self._linear(f'{pfx}.k_proj')
        v = self._linear(f'{pfx}.v_proj')
        o = self._linear(f'{pfx}.o_proj')

        q = reorder_rotary_emb_linear(q, self._head_dim, self._rope.dim)
        k = reorder_rotary_emb_linear(k, self._head_dim, self._rope.dim)

        q, gate = split_output_gate(q, head_dim=self._head_dim)

        cfg = self._attn_cfg.clone()
        attn = AttentionBuilder(cfg, self._contexts,
                                tp=self.engine_cfg.attn_tp_size,
                                ranks=self._attn_ranks)
        attn.add_qkv_proj(q, k, v, gate=gate)
        attn.add_o_proj(o)

        q_norm = self._zero_centered(self._get(f'{pfx}.q_norm.weight'))
        k_norm = self._zero_centered(self._get(f'{pfx}.k_norm.weight'))
        q_norm = reorder_rotary_emb(q_norm, self._head_dim, self._rope.dim)
        k_norm = reorder_rotary_emb(k_norm, self._head_dim, self._rope.dim)
        attn.add_qk_norm(q_norm, k_norm, norm_eps=self._norm_eps)
        return attn

    def linear_attn(self, pfx, layer):
        cfg = self._dn_cfg.clone()
        builder = DeltaNetBuilder(cfg, self._contexts,
                                  tp=self.engine_cfg.attn_tp_size,
                                  ranks=self._attn_ranks)

        builder.add_input_projections(
            in_proj_qkv=self._linear(f'{pfx}.in_proj_qkv'),
            in_proj_z=self._linear(f'{pfx}.in_proj_z'),
            in_proj_b=self._linear(f'{pfx}.in_proj_b'),
            in_proj_a=self._linear(f'{pfx}.in_proj_a'),
            out_proj=self._linear(f'{pfx}.out_proj'),
            qkv_split=self._linear_qkv_split)
        builder.add_scalar_params(
            a_log=self._get(f'{pfx}.A_log'),
            dt_bias=self._get(f'{pfx}.dt_bias'))
        builder.add_conv1d(
            self._get(f'{pfx}.conv1d.weight'),
            qkv_split=self._linear_qkv_split)
        builder.add_norm(
            self._get(f'{pfx}.norm.weight'), data_type=self._cpp_dtype(), norm_eps=self._norm_eps)
        return builder

    # ------------------------------------------------------------------
    # FFN / MoE factories
    # ------------------------------------------------------------------

    def ffn(self, pfx, layer, inter_size=None, fused_moe=False):
        w1 = self._linear(f'{pfx}.gate_proj')
        w3 = self._linear(f'{pfx}.up_proj')
        w2 = self._linear(f'{pfx}.down_proj')
        if w1 is None and w2 is None and w3 is None:
            return None

        cfg = self._ffn_cfg.clone()
        cfg.inter_size = (inter_size if inter_size is not None
                          else self._inter_sizes[layer])
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
        cfg.inter_size = self._expert_inter_size

        m = MoeBuilder(cfg, self._contexts,
                       tp=self.engine_cfg.mlp_tp_size,
                       ranks=self._mlp_ranks)

        dtype = self._cpp_dtype()
        gate_w = self._get(f'{pfx}.gate.weight')
        gate_w = gate_w.t() if gate_w.dim() > 1 else gate_w
        m.add_gate('gate', Linear({'weight': gate_w}), model_dtype=dtype)

        sg = self._get(f'{pfx}.shared_expert_gate.weight')
        sg = sg.t() if sg.dim() > 1 else sg
        m.add_gate('shared_gate', Linear({'weight': sg}), model_dtype=dtype)

        experts = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for e in range(self.num_experts(layer)):
            experts[str(e)] = self._moe_expert_ffn(
                pfx, layer, e, self._expert_inter_size)

        m.experts = experts
        return m

    def _moe_expert_ffn(self, pfx, layer, expert_idx, inter_size):
        expert_pfx = f'{pfx}.experts.{expert_idx}'
        result = self.ffn(expert_pfx, layer,
                          inter_size=inter_size, fused_moe=True)
        if result is not None:
            return result
        packed_pfx = f'{pfx}.experts'
        return self._packed_moe_expert_indexed(packed_pfx, expert_idx, inter_size)

    def _packed_moe_expert_indexed(self, pfx, expert_idx, inter_size):
        gate_up_lin = build_linear(self.params, f'{pfx}.gate_up_proj',
                                   index=expert_idx,
                                   block_in=self._group_size,
                                   block_out=self._group_size)
        down_lin = build_linear(self.params, f'{pfx}.down_proj',
                                index=expert_idx,
                                block_in=self._group_size,
                                block_out=self._group_size)
        if gate_up_lin is None or down_lin is None:
            return None

        gate_tensors: dict[str, torch.Tensor] = {}
        up_tensors: dict[str, torch.Tensor] = {}
        for kind, t in gate_up_lin.tensors.items():
            half = t.shape[-1] // 2
            gate_tensors[kind] = t[..., :half].contiguous()
            up_tensors[kind] = t[..., half:].contiguous()

        cfg = self._ffn_cfg.clone()
        cfg.inter_size = inter_size
        cfg.fuse_silu  = False
        cfg.fused_moe  = True

        m = FfnBuilder(cfg, self._contexts,
                       tp=self.engine_cfg.mlp_tp_size,
                       ranks=self._mlp_ranks)
        w1 = Linear(tensors=gate_tensors, weight_format=gate_up_lin.weight_format)
        w3 = Linear(tensors=up_tensors,   weight_format=gate_up_lin.weight_format)
        m.add_ffn(w1, down_lin, w3)
        return m

    # ------------------------------------------------------------------
    # layers() — dispatch by layer type
    # ------------------------------------------------------------------

    def layers(self, pfx):
        layers = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for i in layer_progress(self._num_layer):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._contexts)
            d.attention_norm = self.norm(f'{pfx}.{i}.input_layernorm.weight')
            if self._is_linear_attn(i):
                d.linear_attn = self.linear_attn(f'{pfx}.{i}.linear_attn', i)
            else:
                d.attention = self.attn(f'{pfx}.{i}.self_attn', i)
            d.ffn_norm = self.norm(f'{pfx}.{i}.post_attention_layernorm.weight')
            if self.num_experts(i) > 0:
                d.feed_forward = self.ffn(f'{pfx}.{i}.mlp.shared_expert', i)
                d.moe_ffn = self.moe(f'{pfx}.{i}.mlp', i)
            else:
                d.feed_forward = self.ffn(f'{pfx}.{i}.mlp', i)
            layers[str(i)] = d
        return layers
