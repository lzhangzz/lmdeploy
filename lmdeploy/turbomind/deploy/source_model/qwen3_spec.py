# Copyright (c) OpenMMLab. All rights reserved.
"""Qwen3 TextModelSpec for the new pipeline.

Qwen3 is a standard Llama-like model with QK norm and optional MoE.
No shared expert in the MoE variant, no linear attention, no zero-centered norm.
"""
from __future__ import annotations

import torch

from ..linear import Linear
from ..spec import TextModelSpec
from .base import INPUT_MODELS, BaseInputModel
from .utils import load_model_config, parse_rope_param

_LAYER_PATTERN = r'model\.layers\.([0-9]+).'


class Qwen3TextSpec(TextModelSpec):
    """Weight spec for Qwen3 (dense) and Qwen3-MoE."""

    _layer_prefix = "model.layers"

    def __init__(self, params: dict[str, torch.Tensor], model_cfg: dict):
        self.params = params
        self.cfg = model_cfg
        self._num_layer = model_cfg["num_hidden_layers"]
        self._n_experts = model_cfg.get("num_experts", 0)

    # ---- Linear bundles: attention ----

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

    # ---- Linear bundles: FFN ----

    def ffn_linears(self, layer: int) -> dict[str, Linear]:
        if self._n_experts > 0:
            return {}
        pfx = f"{self._layer_prefix}.{layer}.mlp"
        return self._read_ffn_linears(pfx)

    # ---- Linear bundles: MoE experts ----

    def moe_ffn_linears(self, layer: int, expert: int) -> dict[str, Linear]:
        pfx = f"{self._layer_prefix}.{layer}.mlp.experts.{expert}"
        return self._read_ffn_linears(pfx)

    def num_experts(self, layer: int) -> int:
        return self._n_experts

    # ---- Raw tensors ----

    def attn_norm(self, layer: int) -> torch.Tensor | None:
        return self._get(f"{self._layer_prefix}.{layer}.input_layernorm.weight")

    def ffn_norm(self, layer: int) -> torch.Tensor | None:
        return self._get(f"{self._layer_prefix}.{layer}.post_attention_layernorm.weight")

    def attn_params(self, layer):
        params = {}
        q = self._get(f"{self._layer_prefix}.{layer}.self_attn.q_norm.weight")
        k = self._get(f"{self._layer_prefix}.{layer}.self_attn.k_norm.weight")
        if q is not None and k is not None:
            q, k = self._permute_qk_tensors(q, k)
        if q is not None:
            params["q_norm.weight"] = (q, None)
        if k is not None:
            params["k_norm.weight"] = (k, None)
        return params

    def moe_params(self, layer):
        params = {}
        if self._n_experts > 0:
            gate = self._get(
                f"{self._layer_prefix}.{layer}.mlp.gate.weight")
            if gate is not None:
                gate = gate.t() if gate.dim() > 1 else gate
                params["gate.weight"] = (gate, None)
        return params

    def tok_embeddings(self) -> torch.Tensor | None:
        return self._get("model.embed_tokens.weight")

    def output_weight(self) -> torch.Tensor | None:
        tie = self.cfg.get("tie_word_embeddings", False)
        key = "model.embed_tokens.weight" if tie else "lm_head.weight"
        return self._get(key)

    def norm_weight(self) -> torch.Tensor | None:
        return self._get("model.norm.weight")

    # ---- metadata ----

    def model_info(self) -> dict:
        cfg = self.cfg
        hidden = cfg["hidden_size"]
        heads = cfg["num_attention_heads"]
        kv_heads = cfg.get("num_key_value_heads", heads)
        head_dim = cfg.get("head_dim", hidden // heads)
        info = dict(
            num_layer=cfg["num_hidden_layers"],
            hidden_units=hidden,
            head_num=heads,
            kv_head_num=kv_heads,
            size_per_head=head_dim,
            vocab_size=cfg["vocab_size"],
            norm_eps=cfg["rms_norm_eps"],
            inter_size=cfg.get("intermediate_size", 0),
            qk_norm=True,
            attn_bias=cfg.get("attention_bias", 0),
        )
        if self._n_experts:
            info.update(
                expert_num=self._n_experts,
                expert_inter_size=cfg.get("moe_intermediate_size", 0),
                experts_per_token=cfg.get("num_experts_per_tok", 0),
                inter_size=0,
                norm_topk_prob=cfg.get("norm_topk_prob", False),
            )
        return info


@INPUT_MODELS.register_module(name='qwen3-moe')
@INPUT_MODELS.register_module(name='qwen3')
class Qwen3InputModel(BaseInputModel):
    """Input model for Qwen3 (dense and MoE)."""

    _layer_pattern = _LAYER_PATTERN
    _spec_class = Qwen3TextSpec

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
