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
    AttentionBuilder, DecoderLayerBuilder, FfnBuilder, LinearBuilder,
    MoeBuilder, ModuleListBuilder, NormBuilder, SplitSide, TextModelBuilder,
    _act_type_id, _cpp_dtype as _cd,
)
from ..linear import Linear, pad_out_dim
from ..module_configs import (
    AttentionConfig, DecoderLayerConfig, FfnConfig, LinearConfig,
    ModuleListConfig, MoeConfig, NormConfig,
)
from ..kind_map import build_linear
from ..spec import TextModelSpec
from .base import INPUT_MODELS, BaseInputModel
from .utils import load_model_config, parse_rope_param

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
    # Builder-driven loading: build full model hierarchy
    # ------------------------------------------------------------------

    def _cpp_dtype(self):
        return _cd(self._mc.data_type)

    def model(self):
        root = TextModelBuilder(self._root_handles, self._contexts)
        root.tok_embeddings = self.token_embeds('model.embed_tokens')
        root.norm = self.root_norm('model.norm')
        root.output = self.lm_head('lm_head')
        root.layers = self.layers('model.layers')

    # ------------------------------------------------------------------
    # Factory methods: read weights, create builders, return them
    # ------------------------------------------------------------------

    def token_embeds(self, pfx):
        """Return LinearBuilder for tok_embeddings, or None."""
        emb = self._get(f'{pfx}.weight')
        if emb is None:
            return None

        mc = self._mc
        tp = self._attn_tp * self._attn_cp
        padded_vocab = ((mc.vocab_size + tp - 1) // tp) * tp
        emb_padded = pad_out_dim(emb, padded_vocab, dim=0)
        dtype = self._cpp_dtype()

        cfg = LinearConfig(input_dim=padded_vocab,
                           output_dim=mc.hidden_units // tp,
                           data_type=dtype)
        m = LinearBuilder(cfg, self._contexts, tp=tp, ranks=self._attn_ranks)
        m.set_weight(emb_padded, split_side=SplitSide.OUTPUT)
        return m

    def root_norm(self, pfx):
        """Return NormBuilder for the final norm, or None."""
        w = self._get(f'{pfx}.weight')
        if w is None:
            return None

        mc = self._mc
        dtype = self._cpp_dtype()
        cfg = NormConfig(dim=mc.hidden_units, data_type=dtype)
        m = NormBuilder(cfg, self._contexts)
        m.set_weight(w)
        return m

    def lm_head(self, pfx):
        """Return LinearBuilder for the output head, or None."""
        tie = self.cfg.get("tie_word_embeddings", False)
        key = "model.embed_tokens.weight" if tie else f"{pfx}.weight"
        output = self._get(key)
        if output is None:
            return None

        mc = self._mc
        tp = self._attn_tp * self._attn_cp
        padded_vocab = ((mc.vocab_size + tp - 1) // tp) * tp
        output_padded = pad_out_dim(output, padded_vocab, dim=0)
        output_t = output_padded.t()
        dtype = self._cpp_dtype()

        cfg = LinearConfig(input_dim=mc.hidden_units,
                           output_dim=padded_vocab // tp,
                           data_type=dtype)
        m = LinearBuilder(cfg, self._contexts, tp=tp, ranks=self._attn_ranks)
        m.set_weight(output_t, split_side=SplitSide.OUTPUT)
        return m

    def norm(self, pfx):
        """Return NormBuilder for the given prefix, or None."""
        w = self._get(f'{pfx}.weight')
        if w is None:
            return None

        dtype = self._cpp_dtype()
        cfg = NormConfig(dim=self._mc.hidden_units, data_type=dtype)
        m = NormBuilder(cfg, self._contexts)
        m.set_weight(w)
        return m

    def attn(self, pfx, layer):
        """Return AttentionBuilder for the given layer, or None."""
        q = self._linear(f"{pfx}.q_proj")
        k = self._linear(f"{pfx}.k_proj")
        v = self._linear(f"{pfx}.v_proj")
        o = self._linear(f"{pfx}.o_proj")

        if q is None and k is None and v is None and o is None:
            return None

        mc = self._mc
        tp = self._attn_tp
        dtype = self._cpp_dtype()

        window_size = 0
        ws_list = mc.window_size
        if ws_list and layer < len(ws_list):
            window_size = ws_list[layer]

        attn_cfg = AttentionConfig.from_model_config(
            mc, tp_size=tp, tp_rank=0, dtype=dtype,
            window_size=window_size,
            rope_dim=self._rope_dim,
            permute_qk=self._permute_qk,
            repeat_kv=self._repeat_kv)
        attn = AttentionBuilder(attn_cfg, self._contexts,
                                tp=tp, ranks=self._attn_ranks)

        if q is not None and k is not None and v is not None:
            attn.add_qkv_proj(q, k, v)
        if o is not None:
            attn.add_o_proj(o)

        # Inline attn params -- attention sinks
        sinks = self._get(f'{pfx}.sinks')
        if sinks is not None:
            attn.add_param('sinks', sinks)

        return attn

    def ffn(self, pfx, layer, inter_size=None, fused_moe=False):
        """Return FfnBuilder for the given layer, or None."""
        w1 = self._linear(f"{pfx}.gate_proj")
        w3 = self._linear(f"{pfx}.up_proj")
        w2 = self._linear(f"{pfx}.down_proj")
        linears = {}
        if w1 is not None: linears['w1'] = w1
        if w3 is not None: linears['w3'] = w3
        if w2 is not None: linears['w2'] = w2
        if not linears:
            return None

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
        """Return MoeBuilder for the given layer, or None."""
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
        gate_w = self._get(f'{pfx}.router.weight')
        if gate_w is not None:
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
            expert = self._moe_expert_ffn(expert_pfx, layer, expert_inter)
            if expert is not None:
                experts[str(e)] = expert

        m.experts = experts
        return m

    def layers(self, pfx):
        """Return ModuleListBuilder with all decoder layers."""
        layers = ModuleListBuilder(ModuleListConfig(), self._contexts)

        for i in range(self._mc.num_layer):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._contexts)
            d.attention_norm = self.norm(
                f'{pfx}.{i}.input_layernorm')
            d.attention = self.attn(
                f'{pfx}.{i}.self_attn', i)
            d.ffn_norm = self.norm(
                f'{pfx}.{i}.post_attention_layernorm')
            if self.num_experts(i) > 0:
                d.moe_ffn = self.moe(f'{pfx}.{i}.mlp', i)
            layers[str(i)] = d

        return layers

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _linear(self, prefix: str) -> Linear | None:
        """Read a Linear bundle from the checkpoint at *prefix*."""
        return build_linear(self.params, prefix)

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

    def _moe_expert_ffn(self, pfx, layer, expert_inter):
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

        ffn_cfg = FfnConfig.from_model_config(
            mc, tp_size=tp, tp_rank=0, dtype=dtype,
            act_type=_act_type_id(mc.activation_type),
            fuse_silu=False, inter_size=expert_inter,
            fused_moe=True)
        m = FfnBuilder(ffn_cfg, self._contexts, tp=tp, ranks=self._mlp_ranks)
        m.add_ffn(w1, w2, w3)
        return m

    def num_experts(self, layer: int) -> int:
        return self._n_experts

    # ---- metadata ----

    def model_info(self) -> dict:
        cfg = self.cfg
        hidden = cfg["hidden_size"]
        heads = cfg["num_attention_heads"]
        kv_heads = cfg.get("num_key_value_heads", heads)
        head_dim = cfg.get("head_dim", None) or hidden // heads

        types = cfg["layer_types"]
        sliding_window = cfg["sliding_window"]

        return dict(
            num_layer=cfg["num_hidden_layers"],
            hidden_units=hidden,
            head_num=heads,
            kv_head_num=kv_heads,
            size_per_head=head_dim,
            vocab_size=cfg["vocab_size"],
            norm_eps=cfg["rms_norm_eps"],
            attn_bias=int(cfg["attention_bias"]),
            mlp_bias=True,
            expert_router_bias=True,
            expert_num=self._n_experts,
            expert_inter_size=cfg["intermediate_size"],
            experts_per_token=cfg["experts_per_token"],
            norm_topk_prob=True,
            inter_size=0,
            window_size=[
                sliding_window if t == "sliding_attention" else 0
                for t in types
            ],
            attn_sink=True,
            activation_type="gpt-oss",
        )


@INPUT_MODELS.register_module(name='gpt-oss')
class GptOssInputModel(BaseInputModel):
    """Input model for gpt-oss (MoE with packed experts)."""

    _layer_pattern = _LAYER_PATTERN
    _spec_class = GptOssSpec
    _loader_mappings = [map_experts]

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        super().__init__(model_path, tokenizer_path)
        self.model_config = load_model_config(model_path)
        self.model_format = kwargs.get('model_format')
        self.fp8_quant = kwargs.get('fp8_quant', False)

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
