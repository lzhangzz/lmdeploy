# Copyright (c) OpenMMLab. All rights reserved.
"""Qwen3 ModelWeightSpec for the new pipeline.

Qwen3 is a standard Llama-like model with QK norm and optional MoE.
No shared expert in the MoE variant, no linear attention, no zero-centered norm.
"""
from __future__ import annotations

import torch

from ..linear import Linear
from ..linear import transpose as linear_transpose
from ..module import ModelWeightSpec
from ..parameter import build_linear_from_format


class Qwen3Spec(ModelWeightSpec):
    """Weight spec for Qwen3 (dense) and Qwen3-MoE."""

    _prefix = "model.layers"

    def __init__(self, params: dict[str, torch.Tensor], model_cfg: dict,
                 model_format: str | None = None):
        self.params = params
        self.cfg = model_cfg
        self.model_format = model_format
        self._num_layer = model_cfg["num_hidden_layers"]
        self._n_experts = model_cfg.get("num_experts", 0)

    def _read_linear(self, prefix: str) -> Linear | None:
        return build_linear_from_format(
            self.params, prefix, self.model_format)

    def _get(self, key: str) -> torch.Tensor | None:
        return self.params.get(key)

    # ---- Linear bundles: attention ----

    def attn_linears(self, layer: int) -> dict[str, Linear]:
        pfx = f"{self._prefix}.{layer}.self_attn"
        result: dict[str, Linear] = {}
        for tm_name, hf_key in [
            ("w_qkv.q", "q_proj"),
            ("w_qkv.k", "k_proj"),
            ("w_qkv.v", "v_proj"),
            ("wo", "o_proj"),
        ]:
            lin = self._read_linear(f"{pfx}.{hf_key}")
            if lin is not None:
                result[tm_name] = linear_transpose(lin)
        return result

    # ---- Linear bundles: FFN ----

    def ffn_linears(self, layer: int) -> dict[str, Linear]:
        if self._n_experts > 0:
            return {}
        pfx = f"{self._prefix}.{layer}.mlp"
        return self._read_ffn_linears(pfx)

    def _read_ffn_linears(self, pfx: str) -> dict[str, Linear]:
        result: dict[str, Linear] = {}
        for tm_name, hf_key in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
            lin = self._read_linear(f"{pfx}.{hf_key}")
            if lin is not None:
                result[tm_name] = linear_transpose(lin)
        return result

    # ---- Linear bundles: MoE experts ----

    def moe_ffn_linears(self, layer: int, expert: int) -> dict[str, Linear]:
        pfx = f"{self._prefix}.{layer}.mlp.experts.{expert}"
        return self._read_ffn_linears(pfx)

    def num_experts(self, layer: int) -> int:
        return self._n_experts

    # ---- Raw tensors ----

    def attn_norm(self, layer: int) -> torch.Tensor | None:
        return self._get(f"{self._prefix}.{layer}.input_layernorm.weight")

    def ffn_norm(self, layer: int) -> torch.Tensor | None:
        return self._get(f"{self._prefix}.{layer}.post_attention_layernorm.weight")

    def qk_norm(self, layer: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        q = self._get(f"{self._prefix}.{layer}.self_attn.q_norm.weight")
        k = self._get(f"{self._prefix}.{layer}.self_attn.k_norm.weight")
        return q, k

    def moe_ffn_gate(self, layer: int) -> torch.Tensor | None:
        return self._get(f"{self._prefix}.{layer}.mlp.gate.weight")

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
