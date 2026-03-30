# Copyright (c) OpenMMLab. All rights reserved.
"""gpt-oss ModelWeightSpec for the V2 pipeline.

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

from ..linear import Linear
from ..module import ModelWeightSpec, SplitSide
from ..parameter import build_linear
from .base import INPUT_MODELS, BaseInputModel
from .utils import load_model_config, parse_rope_param

_LAYER_PATTERN = r'model\.layers\.([0-9]+).'


def map_experts(s: str) -> str:
    s = re.sub(r'(experts.*proj)$', r'\1.weight', s)
    s = re.sub(r'(experts.*proj)_bias$', r'\1.bias', s)
    s = re.sub(r'(experts.*proj)_blocks$', r'\1.blocks', s)
    s = re.sub(r'(experts.*proj)_scales$', r'\1.scales', s)
    return s


class GptOssSpec(ModelWeightSpec):
    """Weight spec for gpt-oss (MoE with packed experts)."""

    _layer_prefix = "model.layers"

    def __init__(self, params: dict[str, torch.Tensor], model_cfg: dict):
        self.params = params
        self.cfg = model_cfg
        self._n_experts = model_cfg["num_local_experts"]

    def _read_linear(self, prefix: str) -> Linear | None:
        return build_linear(self.params, prefix)

    def _read_packed_expert(self, prefix: str, expert: int) -> Linear | None:
        """Read one expert from packed ``[n_experts, ...]`` tensors.

        gpt-oss stores expert weights in M-major (TM) ``[K, N]`` layout.
        The dense normalizer assumes HF ``[N, K]`` input, so the weight gets
        an extra ``.t()`` after normalisation to cancel the over-transpose.
        Quantized normalizers (AWQ, GPTQ, MXFP4, FP8) produce TM already.
        """
        lin = build_linear(self.params, prefix, index=expert)
        if lin is None:
            return None
        if lin.weight_format.name == "dense":
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

    def _read_attn_linears(self, layer: int) -> dict[str, Linear]:
        pfx = f"{self._layer_prefix}.{layer}.self_attn"
        result: dict[str, Linear] = {}
        for tm_name, hf_key in [
            ("w_qkv.q", "q_proj"),
            ("w_qkv.k", "k_proj"),
            ("w_qkv.v", "v_proj"),
            ("wo", "o_proj"),
        ]:
            lin = self._read_linear(f"{pfx}.{hf_key}")
            if lin is not None:
                result[tm_name] = lin
        return result

    def ffn_linears(self, layer: int) -> dict[str, Linear]:
        return {}

    def moe_ffn_linears(self, layer: int, expert: int) -> dict[str, Linear]:
        pfx = f"{self._layer_prefix}.{layer}.mlp.experts"
        gate_up_lin = self._read_packed_expert(f"{pfx}.gate_up_proj", expert)
        down_lin = self._read_packed_expert(f"{pfx}.down_proj", expert)
        if gate_up_lin is None or down_lin is None:
            return {}
        gate_lin, up_lin = self._deinterleave(gate_up_lin)
        return {
            "w1": gate_lin,
            "w2": down_lin,
            "w3": up_lin,
        }

    def num_experts(self, layer: int) -> int:
        return self._n_experts

    def attn_norm(self, layer: int) -> torch.Tensor | None:
        return self._get(
            f"{self._layer_prefix}.{layer}.input_layernorm.weight")

    def ffn_norm(self, layer: int) -> torch.Tensor | None:
        return self._get(
            f"{self._layer_prefix}.{layer}.post_attention_layernorm.weight")

    def raw_layer_tensors(self, layer: int):
        tensors = []
        gate = self._get(f"{self._layer_prefix}.{layer}.mlp.router.weight")
        if gate is not None:
            gate = gate.t() if gate.dim() > 1 else gate
            tensors.append(("moe_ffn.gate.weight", gate, None))
        gate_bias = self._get(f"{self._layer_prefix}.{layer}.mlp.router.bias")
        if gate_bias is not None:
            tensors.append(("moe_ffn.gate.bias", gate_bias, None))
        sinks = self._get(f"{self._layer_prefix}.{layer}.self_attn.sinks")
        if sinks is not None:
            tensors.append(("attention.sinks", sinks, SplitSide.OUTPUT))
        return tensors

    def tok_embeddings(self) -> torch.Tensor | None:
        return self._get("model.embed_tokens.weight")

    def output_weight(self) -> torch.Tensor | None:
        tie = self.cfg.get("tie_word_embeddings", False)
        key = "model.embed_tokens.weight" if tie else "lm_head.weight"
        return self._get(key)

    def norm_weight(self) -> torch.Tensor | None:
        return self._get("model.norm.weight")

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
        self.policy = kwargs.get('input_policy')
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
