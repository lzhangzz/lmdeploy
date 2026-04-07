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

from ..linear import Linear
from ..module import TextModelSpec, SplitSide
from ..kind_map import build_linear
from .base import INPUT_MODELS, BaseInputModel
from .utils import load_model_config, parse_rope_param

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

    # ---- zero-centered RMSNorm: add 1.0 ----

    def _zero_centered(self, w: torch.Tensor | None) -> torch.Tensor | None:
        if w is not None:
            return w.float() + 1.0
        return None

    # ---- Linear bundles: standard attention ----

    def _read_attn_linears(self, layer: int) -> dict[str, Linear]:
        if self._is_linear_attn(layer):
            return {}
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

    # ---- Linear bundles: linear attention (Gated Delta Net) ----

    def _read_linear_attn_linears(self, layer: int) -> dict[str, Linear]:
        if not self._is_linear_attn(layer):
            return {}
        pfx = f"{self._layer_prefix}.{layer}.linear_attn"
        result: dict[str, Linear] = {}
        for key in ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]:
            lin = self._read_linear(f"{pfx}.{key}")
            if lin is not None:
                result[key] = lin
        return result

    # ---- Linear bundles: FFN / MoE ----

    def ffn_linears(self, layer: int) -> dict[str, Linear]:
        if self._n_experts > 0:
            return self._shared_expert_linears(layer)
        pfx = f"{self._layer_prefix}.{layer}.mlp"
        return self._read_ffn_linears(pfx)

    def _shared_expert_linears(self, layer: int) -> dict[str, Linear]:
        pfx = f"{self._layer_prefix}.{layer}.mlp.shared_expert"
        return self._read_ffn_linears(pfx)

    def moe_ffn_linears(self, layer: int, expert: int) -> dict[str, Linear]:
        pfx = f"{self._layer_prefix}.{layer}.mlp.experts.{expert}"
        result = self._read_ffn_linears(pfx)
        if result:
            return result
        return self._packed_moe_expert(layer, expert)

    def _packed_moe_expert(self, layer: int, expert: int) -> dict[str, Linear]:
        """Handle packed Qwen3.5 MoE checkpoints where all experts share one
        tensor with the expert index in the leading dimension.
        """
        pfx = f"{self._layer_prefix}.{layer}.mlp.experts"
        gate_up_lin = build_linear(self.params, f"{pfx}.gate_up_proj", index=expert)
        down_lin = build_linear(self.params, f"{pfx}.down_proj", index=expert)
        if gate_up_lin is None or down_lin is None:
            return {}

        # gate_up is in TM layout [in, 2*out]; split along output dim.
        gate_tensors: dict[str, torch.Tensor] = {}
        up_tensors: dict[str, torch.Tensor] = {}
        for kind, t in gate_up_lin.tensors.items():
            half = t.shape[-1] // 2
            gate_tensors[kind] = t[..., :half].contiguous()
            up_tensors[kind] = t[..., half:].contiguous()

        return {
            "w1": Linear(tensors=gate_tensors, weight_format=gate_up_lin.weight_format),
            "w2": down_lin,
            "w3": Linear(tensors=up_tensors, weight_format=gate_up_lin.weight_format),
        }

    def num_experts(self, layer: int) -> int:
        return self._n_experts

    # ---- Raw tensors (zero-centered norm) ----

    def attn_norm(self, layer: int) -> torch.Tensor | None:
        return self._zero_centered(
            self._get(f"{self._layer_prefix}.{layer}.input_layernorm.weight"))

    def ffn_norm(self, layer: int) -> torch.Tensor | None:
        return self._zero_centered(
            self._get(f"{self._layer_prefix}.{layer}.post_attention_layernorm.weight"))

    def norm_weight(self) -> torch.Tensor | None:
        return self._zero_centered(self._get(self._norm_key))

    def raw_layer_tensors(self, layer: int):
        tensors = []
        # QK norm (zero-centered, permuted for RoPE layout)
        if not self._is_linear_attn(layer):
            q = self._zero_centered(
                self._get(f"{self._layer_prefix}.{layer}.self_attn.q_norm.weight"))
            k = self._zero_centered(
                self._get(f"{self._layer_prefix}.{layer}.self_attn.k_norm.weight"))
            if q is not None and k is not None:
                q, k = self._permute_qk_tensors(q, k)
            if q is not None:
                tensors.append(("attention.q_norm.weight", q, None))
            if k is not None:
                tensors.append(("attention.k_norm.weight", k, None))
        # MoE gate and shared gate (transposed, broadcast)
        if self._n_experts > 0:
            gate = self._get(f"{self._layer_prefix}.{layer}.mlp.gate.weight")
            if gate is not None:
                gate = gate.t() if gate.dim() > 1 else gate
                tensors.append(("moe_ffn.gate.weight", gate, None))
            sg = self._get(f"{self._layer_prefix}.{layer}.mlp.shared_expert_gate.weight")
            if sg is not None:
                sg = sg.t() if sg.dim() > 1 else sg
                tensors.append(("moe_ffn.shared_gate.weight", sg, None))
        # GDN (linear attention) raw tensors
        if self._is_linear_attn(layer):
            pfx = f"{self._layer_prefix}.{layer}.linear_attn"
            for key in ["A_log", "dt_bias"]:
                t = self._get(f"{pfx}.{key}")
                if t is not None:
                    tensors.append((f"linear_attn.{key}", t, SplitSide.OUTPUT))
            conv1d = self._get(f"{pfx}.conv1d.weight")
            if conv1d is not None and conv1d.ndim == 3 and conv1d.shape[1] == 1:
                conv1d = conv1d.squeeze(1)
            # C++ kernel expects [d_conv, conv_dim]; HF stores [conv_dim, d_conv].
            if conv1d is not None:
                conv1d = conv1d.t().contiguous()
                if self._attn_tp > 1 and self._linear_qkv_split is not None:
                    q_dim, k_dim, v_dim = self._linear_qkv_split
                    d_conv = conv1d.shape[0]
                    tp = self._attn_tp
                    q_part = conv1d[:, :q_dim]
                    k_part = conv1d[:, q_dim:q_dim + k_dim]
                    v_part = conv1d[:, q_dim + k_dim:]
                    conv1d = torch.cat([
                        q_part.reshape(d_conv, tp, q_dim // tp),
                        k_part.reshape(d_conv, tp, k_dim // tp),
                        v_part.reshape(d_conv, tp, v_dim // tp),
                    ], dim=2).reshape(d_conv, -1).contiguous()
                tensors.append(("linear_attn.conv1d", conv1d, SplitSide.OUTPUT))
            norm = self._get(f"{pfx}.norm.weight")
            if norm is not None:
                tensors.append(("linear_attn.norm.weight", norm, None))
        return tensors

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


@INPUT_MODELS.register_module(name='qwen3_5')
class Qwen3_5InputModel(BaseInputModel):
    """Input model for Qwen3.5 (dense + optional linear attention)."""

    _layer_pattern = _LAYER_PATTERN
    _spec_class = Qwen3_5Spec

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        super().__init__(model_path, tokenizer_path)
        self.model_config = load_model_config(model_path)
        self.policy = kwargs.get('input_policy')
        self.model_format = kwargs.get('model_format')
        self.fp8_quant = kwargs.get('fp8_quant', False)

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

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        super().__init__(model_path, tokenizer_path)
        self.model_config = load_model_config(model_path)
        self.policy = kwargs.get('input_policy')
        self.model_format = kwargs.get('model_format')
        self.fp8_quant = kwargs.get('fp8_quant', False)

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
