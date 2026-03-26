# Copyright (c) OpenMMLab. All rights reserved.
"""Qwen3.5 ModelWeightSpec for the new pipeline.

Demonstrates the composable-ops pipeline with:
  - Mixed full-attention and linear-attention (Gated Delta Net) layers
  - Zero-centered RMSNorm (+1.0 transform)
  - MoE (optional) with shared expert and shared gate
  - Mixed AWQ: attention in fp16 while FFN/experts stay quantized
  - Linear attention scalar params (A_log, dt_bias)
"""
from __future__ import annotations

import torch

from ..linear import Linear
from ..linear import transpose as linear_transpose
from ..module import ModelWeightSpec
from ..parameter import build_linear_from_format


class Qwen3_5Spec(ModelWeightSpec):
    """Weight spec for Qwen3.5 (dense + linear attention + optional MoE)."""

    def __init__(self, params: dict[str, torch.Tensor], model_cfg: dict,
                 model_format: str | None = None):
        self.params = params
        self.cfg = model_cfg
        self.model_format = model_format
        self._layer_types = model_cfg.get("layer_types", [])
        self._num_layer = model_cfg["num_hidden_layers"]
        self._n_experts = model_cfg.get("num_experts", 0)

        if any(k.startswith("model.language_model.") for k in params):
            self._prefix = "model.language_model.layers"
            self._embed_key = "model.language_model.embed_tokens.weight"
            self._norm_key = "model.language_model.norm.weight"
        else:
            self._prefix = "model.layers"
            self._embed_key = "model.embed_tokens.weight"
            self._norm_key = "model.norm.weight"

    def _read_linear(self, prefix: str) -> Linear | None:
        return build_linear_from_format(
            self.params, prefix, self.model_format)

    def _dequant_fp8_linear(self, lin: Linear) -> Linear:
        """Dequantize blocked-FP8 linear weights to bf16.

        Qwen3.5 linear attention mixes FP8 and dense input projections.
        The C++ GatedDeltaNet fusion path expects homogeneous input-projection
        dtypes, so we densify the FP8 members before export.
        """
        weight = lin.tensors.get("weight")
        scales = lin.tensors.get("scales")
        if weight is None or scales is None or weight.dtype != torch.uint8:
            return lin

        block_size = 128
        fp8_weight = weight.view(torch.float8_e4m3fn).float()
        scale = scales.float()
        scale = scale.repeat_interleave(block_size, dim=0)
        scale = scale.repeat_interleave(block_size, dim=1)
        scale = scale[:fp8_weight.shape[0], :fp8_weight.shape[1]]

        tensors = dict(lin.tensors)
        tensors.pop("scales", None)
        tensors["weight"] = (fp8_weight * scale).to(torch.bfloat16)
        return Linear(tensors=tensors, input_dim=lin.input_dim, output_dim=lin.output_dim)

    def _linear_attn_mixed_input_precision(self, layer: int) -> bool:
        if self.model_format != "fp8":
            return False
        pfx = f"{self._prefix}.{layer}.linear_attn"
        b = self._read_linear(f"{pfx}.in_proj_b")
        a = self._read_linear(f"{pfx}.in_proj_a")
        linears = [x for x in (b, a) if x is not None]
        return any("scales" not in lin.tensors for lin in linears)

    def _get(self, key: str) -> torch.Tensor | None:
        return self.params.get(key)

    def _is_linear_attn(self, layer: int) -> bool:
        return (layer < len(self._layer_types)
                and self._layer_types[layer] == "linear_attention")

    def _is_moe_layer(self, layer: int) -> bool:
        return self._n_experts > 0

    # ---- zero-centered RMSNorm: add 1.0 ----

    def _zero_centered(self, w: torch.Tensor | None) -> torch.Tensor | None:
        if w is not None:
            return w.float() + 1.0
        return None

    # ---- Linear bundles: standard attention ----

    def attn_linears(self, layer: int) -> dict[str, Linear]:
        if self._is_linear_attn(layer):
            return {}
        pfx = f"{self._prefix}.{layer}.self_attn"
        result: dict[str, Linear] = {}
        for tm_name, hf_key in [
            ("w_qkv.q", "q_proj"),
            ("w_qkv.k", "k_proj"),
            ("w_qkv.v", "v_proj"),
            ("wo", "o_proj"),
        ]:
            lin = self._read_linear(f"{pfx}.{hf_key}")
            if lin is None:
                continue
            lin = linear_transpose(lin)

            if tm_name == "wo" and self._is_mixed_attn_o(layer):
                lin = self._dequant_to_fp16(f"{pfx}.o_proj")
            result[tm_name] = lin
        return result

    def _is_mixed_attn_o(self, layer: int) -> bool:
        """Check if O projection is AWQ-quantized while QKV are fp16."""
        pfx = f"{self._prefix}.{layer}.self_attn"
        q_fp16 = f"{pfx}.q_proj.weight" in self.params
        o_awq = f"{pfx}.o_proj.qweight" in self.params
        return q_fp16 and o_awq

    def _dequant_to_fp16(self, prefix: str) -> Linear:
        """Dequantize an AWQ linear to fp16 and return as a dense Linear."""
        from lmdeploy.pytorch.backends.default.awq_modules import dequantize_gemm
        qweight = self.params[f"{prefix}.qweight"]
        scales = self.params[f"{prefix}.scales"]
        qzeros = self.params[f"{prefix}.qzeros"]
        group_size = qweight.shape[0] // scales.shape[0]
        w = dequantize_gemm(qweight, qzeros, scales, 4, group_size)
        return Linear(tensors={"weight": w.t()}, input_dim=0, output_dim=-1)

    # ---- Linear bundles: linear attention (Gated Delta Net) ----

    def linear_attn_linears(self, layer: int) -> dict[str, Linear]:
        if not self._is_linear_attn(layer):
            return {}
        pfx = f"{self._prefix}.{layer}.linear_attn"
        result: dict[str, Linear] = {}
        mixed_fp8 = self._linear_attn_mixed_input_precision(layer)
        for key in ["conv1d", "in_proj_qkv", "in_proj_z", "in_proj_b",
                     "in_proj_a", "out_proj"]:
            lin = self._read_linear(f"{pfx}.{key}")
            if lin is None:
                # Fall back to dequantizing AWQ if needed
                if f"{pfx}.{key}.qweight" in self.params:
                    lin = self._dequant_to_fp16(f"{pfx}.{key}")
            if lin is None:
                continue
            if mixed_fp8 and key in {"in_proj_qkv", "in_proj_z"}:
                lin = self._dequant_fp8_linear(lin)
            if key == "conv1d":
                w = lin.tensors["weight"]
                if w.ndim == 3 and w.shape[1] == 1:
                    lin = Linear(tensors={"weight": w.squeeze(1)},
                                 input_dim=lin.input_dim, output_dim=lin.output_dim)
            else:
                lin = linear_transpose(lin)
            result[key] = lin
        return result

    def linear_attn_scalars(self, layer: int) -> dict[str, torch.Tensor]:
        if not self._is_linear_attn(layer):
            return {}
        pfx = f"{self._prefix}.{layer}.linear_attn"
        result: dict[str, torch.Tensor] = {}
        for key in ["A_log", "dt_bias"]:
            t = self._get(f"{pfx}.{key}")
            if t is not None:
                result[key] = t
        return result

    def linear_attn_norm(self, layer: int) -> torch.Tensor | None:
        if not self._is_linear_attn(layer):
            return None
        return self._get(f"{self._prefix}.{layer}.linear_attn.norm.weight")

    # ---- Linear bundles: FFN / MoE ----

    def ffn_linears(self, layer: int) -> dict[str, Linear]:
        if self._n_experts > 0:
            return self._shared_expert_linears(layer)
        pfx = f"{self._prefix}.{layer}.mlp"
        return self._read_ffn_linears(pfx)

    def _shared_expert_linears(self, layer: int) -> dict[str, Linear]:
        pfx = f"{self._prefix}.{layer}.mlp.shared_expert"
        return self._read_ffn_linears(pfx)

    def _read_ffn_linears(self, pfx: str) -> dict[str, Linear]:
        result: dict[str, Linear] = {}
        for tm_name, hf_key in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
            lin = self._read_linear(f"{pfx}.{hf_key}")
            if lin is not None:
                result[tm_name] = linear_transpose(lin)
        return result

    def moe_ffn_linears(self, layer: int, expert: int) -> dict[str, Linear]:
        pfx = f"{self._prefix}.{layer}.mlp.experts.{expert}"
        result = self._read_ffn_linears(pfx)
        if result:
            return result
        return self._packed_moe_expert(layer, expert)

    def _packed_moe_expert(self, layer: int, expert: int) -> dict[str, Linear]:
        """Handle packed Qwen3.5 MoE checkpoints where all experts share one
        tensor with the expert index in the leading dimension."""
        pfx = f"{self._prefix}.{layer}.mlp.experts"
        gate_up = self._get(f"{pfx}.gate_up_proj.weight")
        down = self._get(f"{pfx}.down_proj.weight")
        if gate_up is None or down is None:
            return {}
        gate_up_e = gate_up[expert]
        down_e = down[expert]
        gate, up = gate_up_e.chunk(2, dim=0)
        return {
            "w1": linear_transpose(Linear(tensors={"weight": gate}, input_dim=0, output_dim=-1)),
            "w2": linear_transpose(Linear(tensors={"weight": down_e}, input_dim=0, output_dim=-1)),
            "w3": linear_transpose(Linear(tensors={"weight": up}, input_dim=0, output_dim=-1)),
        }

    def num_experts(self, layer: int) -> int:
        return self._n_experts

    def has_shared_gate(self) -> bool:
        return self._n_experts > 0

    def moe_ffn_gate(self, layer: int) -> torch.Tensor | None:
        return self._get(f"{self._prefix}.{layer}.mlp.gate.weight")

    def moe_ffn_shared_gate(self, layer: int) -> torch.Tensor | None:
        return self._get(f"{self._prefix}.{layer}.mlp.shared_expert_gate.weight")

    # ---- Raw tensors (zero-centered norm) ----

    def attn_norm(self, layer: int) -> torch.Tensor | None:
        return self._zero_centered(
            self._get(f"{self._prefix}.{layer}.input_layernorm.weight"))

    def ffn_norm(self, layer: int) -> torch.Tensor | None:
        return self._zero_centered(
            self._get(f"{self._prefix}.{layer}.post_attention_layernorm.weight"))

    def norm_weight(self) -> torch.Tensor | None:
        return self._zero_centered(self._get(self._norm_key))

    def qk_norm(self, layer: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        q = self._get(f"{self._prefix}.{layer}.self_attn.q_norm.weight")
        k = self._get(f"{self._prefix}.{layer}.self_attn.k_norm.weight")
        return self._zero_centered(q), self._zero_centered(k)

    def tok_embeddings(self) -> torch.Tensor | None:
        return self._get(self._embed_key)

    def output_weight(self) -> torch.Tensor | None:
        tie = self.cfg.get("tie_word_embeddings", False)
        key = self._embed_key if tie else "lm_head.weight"
        return self._get(key)

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
                inter_size=cfg.get("shared_expert_intermediate_size", 0),
                moe_shared_gate=True,
                scoring_func="softmax",
                norm_topk_prob=True,
            )
        if self._layer_types:
            info.update(
                layer_types=self._layer_types,
                linear_key_head_dim=cfg.get("linear_key_head_dim", 0),
                linear_value_head_dim=cfg.get("linear_value_head_dim", 0),
                linear_conv_kernel_dim=cfg.get("linear_conv_kernel_dim", 0),
                linear_num_key_heads=cfg.get("linear_num_key_heads", 0),
                linear_num_value_heads=cfg.get("linear_num_value_heads", 0),
                attn_output_gate=cfg.get("attn_output_gate", False),
            )
        rope_params = cfg.get("rope_parameters", {})
        partial_rot = rope_params.get("partial_rotary_factor",
                                      cfg.get("partial_rotary_factor", 1.0))
        if partial_rot < 1.0:
            info["rope_dim"] = int(head_dim * partial_rot)
        return info
