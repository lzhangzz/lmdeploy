# Copyright (c) OpenMMLab. All rights reserved.
"""GLM-4 MoE Lite (GLM-4.7-Flash) ModelWeightSpec for the new pipeline.

Demonstrates the composable-ops pipeline with:
  - MLA (Multi-head Latent Attention) via ``read_linear``
  - MoE experts with dense first layer
  - noaux_tc routing with score correction bias
  - Raw tensors for norms, router, embeddings
"""
from __future__ import annotations

import torch

from ..linear import Linear
from ..linear import transpose as linear_transpose
from ..module import ModelWeightSpec
from ..parameter import build_linear_from_format


class Glm4MoeLiteSpec(ModelWeightSpec):
    """Weight spec for GLM-4 MoE Lite (e.g. GLM-4.7-Flash).

    Uses same key layout as DeepSeek2: ``model.layers.{i}.self_attn.*``,
    ``model.layers.{i}.mlp.*``.  First layer is dense FFN; remaining layers
    are MoE with ``n_routed_experts`` experts per layer.
    """

    _prefix = "model.layers"

    def __init__(self, params: dict[str, torch.Tensor], model_cfg: dict,
                 model_format: str | None = None):
        self.params = params
        self.cfg = model_cfg
        self.model_format = model_format
        self._num_layer = model_cfg["num_hidden_layers"]
        self._n_experts = model_cfg.get("n_routed_experts", 0)
        self._first_k_dense = model_cfg.get("first_k_dense_replace", 1)

    def _read_linear(self, prefix: str) -> Linear | None:
        lin = build_linear_from_format(
            self.params, prefix, self.model_format)
        if lin is None and self.model_format not in (None, "hf"):
            lin = build_linear_from_format(self.params, prefix, None)
        return lin

    def _get(self, key: str) -> torch.Tensor | None:
        return self.params.get(key)

    # ---- Linear bundles: MLA attention ----

    def attn_linears(self, layer: int) -> dict[str, Linear]:
        pfx = f"{self._prefix}.{layer}.self_attn"

        # Read all projections as raw (pre-transpose) Linear bundles
        raw: dict[str, Linear] = {}
        for tm_name, hf_key in [
            ("q_a_proj", "q_a_proj"),
            ("q_b_proj", "q_b_proj"),
            ("q_proj", "q_proj"),
            ("kv_a_proj", "kv_a_proj_with_mqa"),
            ("kv_b_proj", "kv_b_proj"),
            ("wo", "o_proj"),
        ]:
            lin = self._read_linear(f"{pfx}.{hf_key}")
            if lin is not None:
                raw[tm_name] = lin

        if "q_proj" in raw and "q_b_proj" not in raw:
            raw["q_b_proj"] = raw.pop("q_proj")

        self._mla_fold_and_pad(raw)

        # Transpose and collect (kv_b_proj is removed by folding)
        result: dict[str, Linear] = {}
        for name, lin in raw.items():
            result[name] = linear_transpose(lin)
        return result

    def _mla_fold_and_pad(self, linears: dict[str, Linear]):
        """Fold kv_b_proj into q_b_proj and wo, then pad wo.

        Mirrors the V1 ``MLA._export`` folding logic.  Operates on
        pre-transpose tensors (HF layout: ``[out_features, in_features]``).
        """
        cfg = self.cfg
        head_num = cfg["num_attention_heads"]
        qk_rope_dim = cfg["qk_rope_head_dim"]
        qk_nope_dim = cfg["qk_nope_head_dim"]
        kv_lora_rank = cfg["kv_lora_rank"]
        v_head_dim = cfg["v_head_dim"]
        size_per_head = qk_nope_dim + qk_rope_dim
        if kv_lora_rank and kv_lora_rank != qk_nope_dim:
            size_per_head = kv_lora_rank + qk_rope_dim
            v_head_dim = kv_lora_rank

        q_b_lin = linears.get("q_b_proj")
        kv_b_lin = linears.pop("kv_b_proj", None)
        o_lin = linears.get("wo")

        if q_b_lin is not None and kv_b_lin is not None and o_lin is not None:
            q_b = q_b_lin.tensors.get("weight")
            kv_b = kv_b_lin.tensors.get("weight")
            o = o_lin.tensors.get("weight")

            if q_b is not None and kv_b is not None and o is not None and torch.is_floating_point(q_b) and torch.is_floating_point(kv_b):
                orig_q_head_dim = q_b.size(0) // head_num
                orig_qk_nope_dim = orig_q_head_dim - qk_rope_dim
                orig_v_head_dim = o.size(1) // head_num
                target_nope_dim = size_per_head - qk_rope_dim

                if orig_qk_nope_dim != target_nope_dim or orig_v_head_dim != v_head_dim:
                    # Split kv_b into kc and vc
                    kv_b_per_head = kv_b.reshape(head_num, orig_qk_nope_dim + orig_v_head_dim, kv_lora_rank)
                    kc_w = kv_b_per_head[:, :orig_qk_nope_dim, :]
                    vc_w = kv_b_per_head[:, orig_qk_nope_dim:, :]

                    # Fold kc into q_b_proj
                    q_b_per_head = q_b.reshape(head_num, orig_q_head_dim, q_b.size(1))
                    q_nope_w = q_b_per_head[:, :orig_qk_nope_dim, :]
                    q_rope_w = q_b_per_head[:, orig_qk_nope_dim:, :]
                    q_nope_expanded = torch.bmm(kc_w.transpose(1, 2), q_nope_w)
                    q_b_folded = torch.cat([q_nope_expanded, q_rope_w], dim=1)
                    q_b_lin.tensors["weight"] = q_b_folded.reshape(
                        head_num * size_per_head, q_b.size(1))

                    # Fold vc into o_proj
                    o_per_head = o.reshape(o.size(0), head_num, orig_v_head_dim)
                    o_folded = torch.bmm(o_per_head.permute(1, 0, 2), vc_w)
                    o_lin.tensors["weight"] = o_folded.permute(1, 0, 2).reshape(
                        o.size(0), head_num * kv_lora_rank)

        # Pad wo from [hidden, head_num*v_head_dim] to [hidden, head_num*size_per_head]
        if o_lin is not None:
            o_w = o_lin.tensors["weight"]
            cur_v = o_w.size(1) // head_num
            if cur_v < size_per_head:
                o_w = o_w.reshape(o_w.size(0), head_num, cur_v)
                o_w = torch.nn.functional.pad(
                    o_w, (size_per_head - cur_v, 0, 0, 0, 0, 0))
                o_lin.tensors["weight"] = o_w.reshape(
                    o_w.size(0), head_num * size_per_head)

    # ---- Linear bundles: dense FFN (layer 0) ----

    def ffn_linears(self, layer: int) -> dict[str, Linear]:
        if layer >= self._first_k_dense:
            return self._shared_expert_linears(layer)
        pfx = f"{self._prefix}.{layer}.mlp"
        return self._read_ffn_linears(pfx)

    def _shared_expert_linears(self, layer: int) -> dict[str, Linear]:
        pfx = f"{self._prefix}.{layer}.mlp.shared_experts"
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
        if layer < self._first_k_dense:
            return 0
        return self._n_experts

    def has_shared_gate(self) -> bool:
        return False

    # ---- Raw tensors ----

    def attn_norm(self, layer: int) -> torch.Tensor | None:
        return self._get(f"{self._prefix}.{layer}.input_layernorm.weight")

    def ffn_norm(self, layer: int) -> torch.Tensor | None:
        return self._get(f"{self._prefix}.{layer}.post_attention_layernorm.weight")

    def mla_norm(self, layer: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        q = self._get(f"{self._prefix}.{layer}.self_attn.q_a_layernorm.weight")
        kv = self._get(f"{self._prefix}.{layer}.self_attn.kv_a_layernorm.weight")
        return q, kv

    def moe_ffn_gate(self, layer: int) -> torch.Tensor | None:
        return self._get(f"{self._prefix}.{layer}.mlp.gate.weight")

    def moe_ffn_gate_bias(self, layer: int) -> torch.Tensor | None:
        return self._get(f"{self._prefix}.{layer}.mlp.gate.bias")

    def moe_ffn_gate_correction_bias(self, layer: int) -> torch.Tensor | None:
        return self._get(f"{self._prefix}.{layer}.mlp.gate.e_score_correction_bias")

    def tok_embeddings(self) -> torch.Tensor | None:
        return self._get("model.embed_tokens.weight")

    def output_weight(self) -> torch.Tensor | None:
        return self._get("lm_head.weight")

    def norm_weight(self) -> torch.Tensor | None:
        return self._get("model.norm.weight")

    # ---- metadata ----

    def model_info(self) -> dict:
        cfg = self.cfg
        num_layer = cfg["num_hidden_layers"]
        n_experts = cfg.get("n_routed_experts", 0)
        first_k = cfg.get("first_k_dense_replace", 1)
        expert_num = [n_experts] * num_layer
        for i in range(first_k):
            expert_num[i] = 0

        qk_nope_dim = cfg["qk_nope_head_dim"]
        qk_rope_dim = cfg["qk_rope_head_dim"]
        kv_lora_rank = cfg["kv_lora_rank"]
        q_head_dim = qk_nope_dim + qk_rope_dim
        size_per_head = q_head_dim
        v_head_dim = cfg["v_head_dim"]
        softmax_scale = 0.0
        if kv_lora_rank and kv_lora_rank != qk_nope_dim:
            size_per_head = kv_lora_rank + qk_rope_dim
            v_head_dim = kv_lora_rank
            softmax_scale = q_head_dim ** (-0.5)
        return dict(
            num_layer=num_layer,
            hidden_units=cfg["hidden_size"],
            head_num=cfg["num_attention_heads"],
            kv_head_num=1,
            size_per_head=size_per_head,
            softmax_scale=softmax_scale,
            vocab_size=cfg["vocab_size"],
            norm_eps=cfg["rms_norm_eps"],
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=cfg.get("q_lora_rank", 0) or 0,
            qk_rope_dim=qk_rope_dim,
            v_head_dim=v_head_dim,
            inter_size=[cfg.get("n_shared_experts", 1) * cfg["moe_intermediate_size"]] * num_layer,
            expert_num=expert_num,
            expert_inter_size=cfg["moe_intermediate_size"],
            experts_per_token=cfg["num_experts_per_tok"],
            norm_topk_prob=cfg.get("norm_topk_prob", True),
            topk_method="noaux_tc",
            topk_group=cfg.get("topk_group", 1),
            moe_group_num=cfg.get("n_group", 1),
            scoring_func="sigmoid",
        )
