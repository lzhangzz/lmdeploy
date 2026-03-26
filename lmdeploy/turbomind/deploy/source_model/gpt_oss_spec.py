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

import torch

from ..kind_map import get_normalizer, get_suffix_map
from ..linear import Linear
from ..linear import transpose as linear_transpose
from ..module import ModelWeightSpec
from ..parameter import build_linear_from_format


class GptOssSpec(ModelWeightSpec):
    """Weight spec for gpt-oss (MoE with packed experts)."""

    _prefix = "model.layers"

    def __init__(self, params: dict[str, torch.Tensor], model_cfg: dict,
                 model_format: str | None = None):
        self.params = params
        self.cfg = model_cfg
        self.model_format = model_format
        self._n_experts = model_cfg["num_local_experts"]

    def _read_linear(self, prefix: str) -> Linear | None:
        lin = build_linear_from_format(
            self.params, prefix, self.model_format)
        if ".self_attn." in prefix:
            dense_lin = build_linear_from_format(
                self.params, prefix, None)
            if lin is None:
                lin = dense_lin
            elif dense_lin is not None:
                merged = dict(dense_lin.tensors)
                merged.update(lin.tensors)
                lin = Linear(tensors=merged, input_dim=lin.input_dim, output_dim=lin.output_dim)
        return lin

    def _get(self, key: str) -> torch.Tensor | None:
        return self.params.get(key)

    def _read_packed_expert(self, prefix: str, expert: int) -> Linear | None:
        """Read one expert from packed ``[n_experts, ...]`` tensors."""
        suffix_map = get_suffix_map(self.model_format)
        normalizer = get_normalizer(self.model_format)
        tensors: dict[str, torch.Tensor] = {}
        for suffix, kind in suffix_map.items():
            key = prefix + suffix
            packed = self.params.get(key)
            if packed is None:
                continue
            raw = packed[expert]
            t = normalizer(raw, kind)
            if kind == "weight" and t.dim() == 2 and self.model_format != "mxfp4":
                t = t.t()
            tensors[kind] = t
        if not tensors:
            return None
        return Linear(tensors=tensors, input_dim=0, output_dim=-1)

    @staticmethod
    def _deinterleave(lin: Linear) -> tuple[Linear, Linear]:
        """Split interleaved gate/up: even indices -> gate, odd -> up."""
        gate_t: dict[str, torch.Tensor] = {}
        up_t: dict[str, torch.Tensor] = {}
        for kind, t in lin.tensors.items():
            gate_t[kind] = t[::2]
            up_t[kind] = t[1::2]
        return (
            Linear(tensors=gate_t, input_dim=lin.input_dim,
                   output_dim=lin.output_dim),
            Linear(tensors=up_t, input_dim=lin.input_dim,
                   output_dim=lin.output_dim),
        )

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

    def ffn_linears(self, layer: int) -> dict[str, Linear]:
        return {}

    def moe_ffn_linears(self, layer: int, expert: int) -> dict[str, Linear]:
        pfx = f"{self._prefix}.{layer}.mlp.experts"
        gate_up_lin = self._read_packed_expert(
            f"{pfx}.gate_up_proj", expert)
        down_lin = self._read_packed_expert(f"{pfx}.down_proj", expert)
        if gate_up_lin is None or down_lin is None:
            return {}
        gate_lin, up_lin = self._deinterleave(gate_up_lin)
        return {
            "w1": linear_transpose(gate_lin),
            "w2": linear_transpose(down_lin),
            "w3": linear_transpose(up_lin),
        }

    def num_experts(self, layer: int) -> int:
        return self._n_experts

    def attn_norm(self, layer: int) -> torch.Tensor | None:
        return self._get(
            f"{self._prefix}.{layer}.input_layernorm.weight")

    def ffn_norm(self, layer: int) -> torch.Tensor | None:
        return self._get(
            f"{self._prefix}.{layer}.post_attention_layernorm.weight")

    def moe_ffn_gate(self, layer: int) -> torch.Tensor | None:
        return self._get(
            f"{self._prefix}.{layer}.mlp.router.weight")

    def moe_ffn_gate_bias(self, layer: int) -> torch.Tensor | None:
        return self._get(
            f"{self._prefix}.{layer}.mlp.router.bias")

    def attn_sinks(self, layer: int) -> torch.Tensor | None:
        return self._get(
            f"{self._prefix}.{layer}.self_attn.sinks")

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
