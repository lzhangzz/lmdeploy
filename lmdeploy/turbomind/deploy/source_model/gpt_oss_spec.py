# Copyright (c) OpenMMLab. All rights reserved.
"""gpt-oss TextModelSpec for the V2 pipeline.

Key differences from standard Llama:
  - MoE-only (inter_size=0): no dense FFN, all experts
  - Packed expert weights: gate_up_proj / down_proj stored as
    [n_experts, ...] tensors with M-major (transposed) layout for BF16
  - gate_up interleaving: even rows = gate, odd rows = up (after transpose)
  - Router at mlp.router (not mlp.gate) with bias
  - Per-layer sliding window via layer_types
  - Attention sinks
  - Attention and MLP biases
  - activation_type='gpt-oss'
"""
from __future__ import annotations

import re

import torch

from ..builder import (
    AttentionBuilder, DecoderLayerBuilder, FfnBuilder,
    MoeBuilder, ModuleListBuilder, TextModelBuilder,
    _act_type_id,
)
from ..kind_map import build_linear
from ..linear import Linear
from ..builder import (
    DecoderLayerConfig, ModuleListConfig,
    make_attention_config, make_ffn_config, make_moe_config,
)
from ..spec import TextModelSpec
from .base import INPUT_MODELS, BaseInputModel
from .utils import parse_rope_param, reorder_rotary_emb_linear

_LAYER_PATTERN = r'model\.layers\.([0-9]+).'


def map_experts(s: str) -> str:
    s = re.sub(r'(experts.*proj)$', r'\1.weight', s)
    s = re.sub(r'(experts.*proj)_bias$', r'\1.bias', s)
    s = re.sub(r'(experts.*proj)_blocks$', r'\1.blocks', s)
    s = re.sub(r'(experts.*proj)_scales$', r'\1.scales', s)
    return s


class GptOssSpec(TextModelSpec):
    """Weight spec for gpt-oss (MoE with packed experts)."""

    _layer_prefix = "model.layers"

    def __init__(self, params: dict[str, torch.Tensor], model_cfg: dict):
        self.params = params
        self.cfg = model_cfg
        self._n_experts = model_cfg["num_local_experts"]

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

        window_size = 0
        ws_list = mc.window_size
        if ws_list and layer < len(ws_list):
            window_size = ws_list[layer]

        attn_cfg = make_attention_config(
            mc, tp_size=tp, dtype=dtype,
            window_size=window_size,
            rope_dim=self._rope_dim)
        attn = AttentionBuilder(attn_cfg, self._contexts,
                                tp=tp, ranks=self._attn_ranks)

        attn.add_qkv_proj(q, k, v)
        attn.add_o_proj(o)

        # Inline attn params -- attention sinks
        attn.add_param('sinks', self._get(f'{pfx}.sinks'))

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
        gate_w = self._get(f'{pfx}.router.weight')
        gate_w = gate_w.t() if gate_w.dim() > 1 else gate_w
        tensors = {"weight": gate_w}
        gate_bias = self._get(f'{pfx}.router.bias')
        if gate_bias is not None:
            tensors["bias"] = gate_bias
        m.add_gate('gate', Linear(tensors), model_dtype=dtype)

        expert_inter = mc.expert_inter_size or 0
        experts = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for e in range(self.num_experts(layer)):
            expert_pfx = f'{pfx}.experts.{e}'
            experts[str(e)] = self._moe_expert_ffn(expert_pfx, expert_inter)

        m.experts = experts
        return m

    def layers(self, pfx):
        """Return ModuleListBuilder with all decoder layers."""
        layers = ModuleListBuilder(ModuleListConfig(), self._contexts)

        for i in range(self._mc.num_layer):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._contexts)
            d.attention_norm = self.norm(
                f'{pfx}.{i}.input_layernorm.weight')
            d.attention = self.attn(
                f'{pfx}.{i}.self_attn', i)
            d.ffn_norm = self.norm(
                f'{pfx}.{i}.post_attention_layernorm.weight')
            if self.num_experts(i) > 0:
                d.moe_ffn = self.moe(f'{pfx}.{i}.mlp', i)
            layers[str(i)] = d

        return layers

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_packed_expert(self, prefix: str, expert: int) -> Linear | None:
        """Read one expert from packed ``[n_experts, ...]`` tensors.

        gpt-oss stores expert weights in M-major (TM) ``[K, N]`` layout.
        The trivial normalizer assumes HF ``[N, K]`` input, so the weight gets
        an extra ``.t()`` after normalisation to cancel the over-transpose.
        Quantized normalizers (AWQ, GPTQ, MXFP4, FP8) produce TM already.
        """
        lin = build_linear(self.params, prefix, index=expert)
        if lin is None:
            return None
        if lin.weight_format.name == "trivial":
            w = lin.tensors.get("weight")
            if w is not None and w.dim() == 2:
                lin.tensors["weight"] = w.t().contiguous()
        return lin

    @staticmethod
    def _deinterleave(lin: Linear) -> tuple[Linear, Linear]:
        """Split interleaved gate/up along the output dim: even -> gate, odd -> up.

        In TM layout ``[in, out]`` the interleaving is along the last axis.
        """
        gate_t: dict[str, torch.Tensor] = {}
        up_t: dict[str, torch.Tensor] = {}
        for kind, t in lin.tensors.items():
            gate_t[kind] = t[..., ::2].contiguous()
            up_t[kind] = t[..., 1::2].contiguous()
        return (
            Linear(tensors=gate_t, weight_format=lin.weight_format),
            Linear(tensors=up_t, weight_format=lin.weight_format),
        )

    def _moe_expert_ffn(self, pfx, expert_inter):
        """Build FfnBuilder for one MoE expert with packed weight decoding."""
        base_pfx = pfx.rsplit('.', 1)[0]  # strip .{expert_id}
        expert_id = int(pfx.rsplit('.', 1)[1])
        gate_up_lin = self._read_packed_expert(f"{base_pfx}.gate_up_proj",
                                                expert_id)
        down_lin = self._read_packed_expert(f"{base_pfx}.down_proj",
                                             expert_id)
        if gate_up_lin is None or down_lin is None:
            return None

        gate_lin, up_lin = self._deinterleave(gate_up_lin)
        w1 = gate_lin
        w3 = up_lin
        w2 = down_lin

        mc = self._mc
        tp = self._mlp_tp
        dtype = self._cpp_dtype()

        ffn_cfg = make_ffn_config(
            mc, tp_size=tp, dtype=dtype,
            act_type=_act_type_id(mc.activation_type),
            fuse_silu=False, inter_size=expert_inter,
            fused_moe=True)
        m = FfnBuilder(ffn_cfg, self._contexts, tp=tp, ranks=self._mlp_ranks)
        m.add_ffn(w1, w2, w3)
        return m

    def num_experts(self, layer: int) -> int:
        return self._n_experts


@INPUT_MODELS.register_module(name='gpt-oss')
class GptOssInputModel(BaseInputModel):
    """Input model for gpt-oss (MoE with packed experts)."""

    _layer_pattern = _LAYER_PATTERN
    _spec_class = GptOssSpec
    _loader_mappings = [map_experts]

    def model_info(self) -> dict:
        cfg = self.model_config
        attn_head_num = cfg['num_attention_heads']
        hidden_units = cfg['hidden_size']
        head_dim = cfg.get('head_dim', None) or hidden_units // attn_head_num
        rope_param, max_position_embeddings = parse_rope_param(cfg, head_dim)
        types = cfg['layer_types']
        sliding_window = cfg['sliding_window']
        info = dict(
            num_layer=cfg['num_hidden_layers'],
            norm_eps=cfg['rms_norm_eps'],
            head_num=attn_head_num,
            kv_head_num=cfg.get('num_key_value_heads', attn_head_num),
            hidden_units=hidden_units,
            size_per_head=head_dim,
            inter_size=0,
            vocab_size=cfg['vocab_size'],
            max_position_embeddings=max_position_embeddings,
            rope_param=rope_param,
        )
        info.update(
            attn_bias=int(cfg['attention_bias']),
            mlp_bias=True,
            expert_router_bias=True,
            expert_num=cfg['num_local_experts'],
            expert_inter_size=cfg['intermediate_size'],
            experts_per_token=cfg['experts_per_token'],
            norm_topk_prob=True,
            window_size=[sliding_window if x == 'sliding_attention' else 0 for x in types],
            attn_sink=True,
            activation_type='gpt-oss',
        )
        return info
