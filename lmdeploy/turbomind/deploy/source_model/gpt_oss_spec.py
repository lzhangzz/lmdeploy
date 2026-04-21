# Copyright (c) OpenMMLab. All rights reserved.
"""gpt-oss TextModelSpec for the new pipeline."""
from __future__ import annotations

import re

import torch

import _turbomind as _tm

from ..builder import (AttentionBuilder, DecoderLayerBuilder, FfnBuilder,
                       MoeBuilder, ModuleListBuilder, TextModelBuilder,
                       _act_type_id)
from ..builder import DecoderLayerConfig, ModuleListConfig
from ..kind_map import build_linear
from ..linear import Linear
from ..spec import TextModelSpec
from .base import INPUT_MODELS
from .utils import layer_progress, reorder_rotary_emb_linear

_LAYER_PATTERN = r'model\.layers\.([0-9]+).'


def map_experts(s: str) -> str:
    s = re.sub(r'(experts.*proj)$', r'\1.weight', s)
    s = re.sub(r'(experts.*proj)_bias$', r'\1.bias', s)
    s = re.sub(r'(experts.*proj)_blocks$', r'\1.blocks', s)
    s = re.sub(r'(experts.*proj)_scales$', r'\1.scales', s)
    return s


@INPUT_MODELS.register_module(name='gpt-oss')
class GptOssSpec(TextModelSpec):
    """Weight spec for gpt-oss (MoE with packed experts)."""

    _layer_pattern = _LAYER_PATTERN
    _loader_mappings = [map_experts]
    # gpt-oss always uses the plain `model.*` layout — pin to skip on-load
    # re-detection by TextModelSpec.set_params.
    _pin_layer_prefix = True

    def __init__(self, hf_cfg: dict, engine_cfg, *, group_size: int = 0):
        super().__init__(hf_cfg, engine_cfg, group_size=group_size)

        self._layer_prefix = 'model.layers'
        self._embed_key = 'model.embed_tokens.weight'
        self._norm_key = 'model.norm.weight'

        self._n_experts = hf_cfg['num_local_experts']
        dtype = self._cpp_dtype()

        # ---- Attention template (sliding window set per layer) ----
        self._attn_cfg = _tm.AttentionConfig()
        self._attn_cfg.hidden_dim  = self._hidden_units
        self._attn_cfg.head_dim    = self._head_dim
        self._attn_cfg.head_num    = self._head_num
        self._attn_cfg.kv_head_num = self._kv_head_num
        self._attn_cfg.has_bias    = int(hf_cfg['attention_bias'])
        self._attn_cfg.attn_sink   = True
        self._apply_rope(self._attn_cfg.rope)
        self._attn_cfg.window_size = 0
        self._attn_cfg.tp_size     = engine_cfg.attn_tp_size
        self._attn_cfg.data_type   = dtype
        self._attn_cfg.softmax_scale          = self._softmax_scale

        # ---- FFN template ----
        self._ffn_cfg = _tm.FfnConfig()
        self._ffn_cfg.hidden_dim = self._hidden_units
        self._ffn_cfg.has_bias   = True
        self._ffn_cfg.tp_size    = engine_cfg.mlp_tp_size
        self._ffn_cfg.data_type  = dtype
        self._ffn_cfg.act_type   = _act_type_id('gpt-oss')

        # ---- MoE template ----
        self._moe_cfg = _tm.MoeConfig()
        self._moe_cfg.method            = 1
        self._moe_cfg.experts_per_token = hf_cfg['experts_per_token']
        self._moe_cfg.norm_topk_prob    = True
        self._moe_cfg.shared_gate       = False
        self._moe_cfg.routed_scale      = 1.0
        self._moe_cfg.router_bias       = True
        self._moe_cfg.topk_group        = 1
        self._moe_cfg.topk_method       = 'greedy'
        self._moe_cfg.n_group           = 1
        self._moe_cfg.scoring_func      = 'softmax'
        self._moe_cfg.router_n_groups   = 0
        self._moe_cfg.hidden_dim        = self._hidden_units
        self._moe_cfg.mlp_bias          = True
        self._moe_cfg.data_type         = dtype
        self._moe_cfg.tp_size           = engine_cfg.mlp_tp_size
        self._moe_cfg.act_type          = _act_type_id('gpt-oss')
        self._moe_cfg.fuse_silu         = True

        self._expert_inter_size = hf_cfg['intermediate_size']

        # Per-layer window sizes from layer_types
        types = hf_cfg['layer_types']
        sliding = hf_cfg['sliding_window']
        self._window_sizes = [
            sliding if t == 'sliding_attention' else 0 for t in types
        ]

        # Inter-size list (zero; gpt-oss has no dense FFN layers)
        self._inter_sizes = [0] * self._num_layer
        self._expert_nums = [self._n_experts] * self._num_layer

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

    # Standard RMSNorm
    def output_norm(self, key):
        from ..builder import NormBuilder, make_norm_config
        w = self._get(key)
        cfg = make_norm_config(dim=self._hidden_units, data_type=self._cpp_dtype(), norm_eps=self._norm_eps)
        m = NormBuilder(cfg, self._contexts)
        m.set_weight(w)
        return m

    def norm(self, key):
        return self.output_norm(key)

    # ------------------------------------------------------------------
    # Attention factory — sets per-layer window_size on the clone
    # ------------------------------------------------------------------

    def attn(self, pfx, layer):
        q = self._linear(f'{pfx}.q_proj')
        k = self._linear(f'{pfx}.k_proj')
        v = self._linear(f'{pfx}.v_proj')
        o = self._linear(f'{pfx}.o_proj')

        q = reorder_rotary_emb_linear(q, self._head_dim, self._rope.dim)
        k = reorder_rotary_emb_linear(k, self._head_dim, self._rope.dim)

        cfg = self._attn_cfg.clone()
        cfg.window_size = self._window_sizes[layer]

        attn = AttentionBuilder(cfg, self._contexts,
                                tp=self.engine_cfg.attn_tp_size,
                                ranks=self._attn_ranks)
        attn.add_qkv_proj(q, k, v)
        attn.add_o_proj(o)

        attn.add_param('sinks', self._get(f'{pfx}.sinks'))
        return attn

    # ------------------------------------------------------------------
    # FFN/MoE factories — packed-expert handling
    # ------------------------------------------------------------------

    def ffn(self, pfx, layer, inter_size=None, fused_moe=False):
        w1 = self._linear(f'{pfx}.gate_proj')
        w3 = self._linear(f'{pfx}.up_proj')
        w2 = self._linear(f'{pfx}.down_proj')

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
        gate_w = self._get(f'{pfx}.router.weight')
        gate_w = gate_w.t() if gate_w.dim() > 1 else gate_w
        tensors = {'weight': gate_w}
        gate_bias = self._get(f'{pfx}.router.bias')
        if gate_bias is not None:
            tensors['bias'] = gate_bias
        m.add_gate('gate', Linear(tensors), model_dtype=dtype)

        experts = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for e in range(self.num_experts(layer)):
            experts[str(e)] = self._packed_expert_ffn(
                f'{pfx}.experts.{e}', self._expert_inter_size)
        m.experts = experts
        return m

    def layers(self, pfx):
        layers = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for i in layer_progress(self._num_layer):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._contexts)
            d.attention_norm = self.norm(f'{pfx}.{i}.input_layernorm.weight')
            d.attention = self.attn(f'{pfx}.{i}.self_attn', i)
            d.ffn_norm = self.norm(f'{pfx}.{i}.post_attention_layernorm.weight')
            if self.num_experts(i) > 0:
                d.moe_ffn = self.moe(f'{pfx}.{i}.mlp', i)
            layers[str(i)] = d
        return layers

    # ------------------------------------------------------------------
    # Packed-expert decoding (gate_up interleaved, TM layout)
    # ------------------------------------------------------------------

    def _read_packed_expert(self, prefix: str, expert: int):
        lin = build_linear(self.params, prefix, index=expert,
                           block_in=self._group_size,
                           block_out=self._group_size)
        if lin is None:
            return None
        if lin.weight_format.name == 'trivial':
            w = lin.tensors.get('weight')
            if w is not None and w.dim() == 2:
                lin.tensors['weight'] = w.t().contiguous()
        return lin

    @staticmethod
    def _deinterleave(lin: Linear):
        gate_t: dict[str, torch.Tensor] = {}
        up_t: dict[str, torch.Tensor] = {}
        for kind, t in lin.tensors.items():
            gate_t[kind] = t[..., ::2].contiguous()
            up_t[kind]   = t[..., 1::2].contiguous()
        return (Linear(tensors=gate_t, weight_format=lin.weight_format),
                Linear(tensors=up_t,   weight_format=lin.weight_format))

    def _packed_expert_ffn(self, expert_pfx: str, expert_inter: int):
        base_pfx = expert_pfx.rsplit('.', 1)[0]
        expert_id = int(expert_pfx.rsplit('.', 1)[1])
        gate_up_lin = self._read_packed_expert(
            f'{base_pfx}.gate_up_proj', expert_id)
        down_lin = self._read_packed_expert(
            f'{base_pfx}.down_proj', expert_id)
        if gate_up_lin is None or down_lin is None:
            return None

        w1, w3 = self._deinterleave(gate_up_lin)

        cfg = self._ffn_cfg.clone()
        cfg.inter_size = expert_inter
        cfg.fuse_silu  = False
        cfg.fused_moe  = True

        m = FfnBuilder(cfg, self._contexts,
                       tp=self.engine_cfg.mlp_tp_size,
                       ranks=self._mlp_ranks)
        m.add_ffn(w1, down_lin, w3)
        return m
