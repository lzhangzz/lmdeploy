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

from ..linear import Linear
from ..spec import TextModelSpec, SplitSide
from ..kind_map import build_linear
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

    def model(self):
        from ..builder import (TextModelBuilder, ModuleListBuilder,
                               DecoderLayerBuilder, NormBuilder, Builder,
                               SplitSide as BSplitSide)
        from ..module_configs import (ModuleListConfig, DecoderLayerConfig,
                                      NormConfig, LinearConfig)
        from ..commit import _cpp_dtype
        from ..linear import pad_out_dim

        mc = self._mc
        dtype = _cpp_dtype(mc.data_type)
        hidden = mc.hidden_units
        contexts = self._contexts
        attn_tp = self._attn_tp
        mlp_tp = self._mlp_tp
        attn_ranks = self._attn_ranks
        mlp_ranks = self._mlp_ranks
        attn_cp = self._attn_cp
        num_layer = mc.num_layer

        root = TextModelBuilder(self._root_handles, contexts)

        # --- tok_embeddings ---
        emb = self.tok_embeddings()
        if emb is not None:
            tp = attn_tp * attn_cp
            padded_vocab = ((mc.vocab_size + tp - 1) // tp) * tp
            emb_padded = pad_out_dim(emb, padded_vocab, dim=0)
            tok_cfg = LinearConfig(input_dim=padded_vocab,
                                   output_dim=hidden // tp,
                                   data_type=dtype)
            tok = Builder(tok_cfg, contexts, tp=tp, ranks=attn_ranks)
            tok._commit_tensor('weight', emb_padded,
                               split_side=BSplitSide.OUTPUT)
            root.tok_embeddings = tok

        # --- final norm ---
        norm_w = self.norm_weight()
        if norm_w is not None:
            norm_cfg = NormConfig(dim=hidden, data_type=dtype)
            norm_b = NormBuilder(norm_cfg, contexts)
            norm_b.set_weight(norm_w)
            root.norm = norm_b

        # --- output head ---
        output = self.output_weight()
        if output is not None:
            tp = attn_tp * attn_cp
            padded_vocab = ((mc.vocab_size + tp - 1) // tp) * tp
            output_padded = pad_out_dim(output, padded_vocab, dim=0)
            output_t = output_padded.t()
            out_cfg = LinearConfig(input_dim=hidden,
                                   output_dim=padded_vocab // tp,
                                   data_type=dtype)
            out = Builder(out_cfg, contexts, tp=tp, ranks=attn_ranks)
            out._commit_tensor('weight', output_t,
                               split_side=BSplitSide.OUTPUT)
            root.output = out

        # --- decoder layers ---
        layers = ModuleListBuilder(ModuleListConfig(), contexts)
        for i in range(num_layer):
            d = DecoderLayerBuilder(DecoderLayerConfig(), contexts)

            # attention_norm
            attn_norm_w = self.attn_norm(i)
            if attn_norm_w is not None:
                n_cfg = NormConfig(dim=hidden, data_type=dtype)
                n_b = NormBuilder(n_cfg, contexts)
                n_b.set_weight(attn_norm_w)
                d.attention_norm = n_b

            # attention
            self._build_attention(d, i, mc, dtype, attn_tp, attn_ranks,
                                  contexts)

            # ffn_norm
            ffn_norm_w = self.ffn_norm(i)
            if ffn_norm_w is not None:
                n_cfg = NormConfig(dim=hidden, data_type=dtype)
                n_b = NormBuilder(n_cfg, contexts)
                n_b.set_weight(ffn_norm_w)
                d.ffn_norm = n_b

            # MoE (all layers are MoE for gpt-oss)
            if self.num_experts(i) > 0:
                self._build_moe(d, i, mc, dtype, mlp_tp, mlp_ranks,
                                contexts)

            layers[str(i)] = d

        root.layers = layers

    def _build_attention(self, parent, layer, mc, dtype, tp, ranks,
                         contexts):
        """Build attention: spec reads projections, builder merges QKV."""
        from ..builder import AttentionBuilder
        from ..module_configs import AttentionConfig

        pfx = f"{self._layer_prefix}.{layer}.self_attn"

        # READ: spec reads projections directly from checkpoint
        q = self._read_linear(f"{pfx}.q_proj")
        k = self._read_linear(f"{pfx}.k_proj")
        v = self._read_linear(f"{pfx}.v_proj")
        o = self._read_linear(f"{pfx}.o_proj")

        if q is None and k is None and v is None and o is None:
            return

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
        attn = AttentionBuilder(attn_cfg, contexts, tp=tp, ranks=ranks)

        # TRANSFORM + COMMIT: builder handles QKV merge, RoPE perm, TP split
        if q is not None and k is not None and v is not None:
            attn.add_qkv_proj(q, k, v)
        if o is not None:
            attn.add_o_proj(o)

        # Direct params -- attention sinks
        for name, val in self.attn_params(layer).items():
            if isinstance(val, tuple):
                tensor, ss = val
            else:
                tensor, ss = val, None
            attn.add_param(name, tensor)

        parent.attention = attn

    def _build_moe(self, parent, layer, mc, dtype, tp, ranks, contexts):
        """Build MoE module: spec reads expert weights, builder handles fusion."""
        from ..builder import MoeBuilder, FfnBuilder, ModuleListBuilder
        from ..module_configs import MoeConfig, FfnConfig, ModuleListConfig
        from ..commit import _act_type_id

        if self.num_experts(layer) <= 0:
            return

        expert_num = 0
        en_list = mc.expert_num
        if en_list and layer < len(en_list):
            expert_num = en_list[layer]

        moe_cfg = MoeConfig.from_model_config(
            mc, layer_id=layer, tp_size=tp, tp_rank=0, dtype=dtype,
            act_type=_act_type_id(mc.activation_type),
            fuse_silu=True, expert_num=expert_num)
        moe = MoeBuilder(moe_cfg, contexts, tp=tp, ranks=ranks)

        # Gate linears
        for name, linear in self.moe_gate(layer).items():
            moe.add_gate(name, linear, model_dtype=dtype)

        # Non-expert MoE parameters
        for name, val in self.moe_params(layer).items():
            if isinstance(val, tuple):
                tensor, ss = val
            else:
                tensor, ss = val, None
            moe.add_param(name, tensor)

        # Experts: each expert is an FfnBuilder
        expert_inter = mc.expert_inter_size or 0
        experts = ModuleListBuilder(ModuleListConfig(), contexts)
        for e in range(self.num_experts(layer)):
            expert_linears = self.moe_ffn_linears(layer, e)
            w1 = expert_linears.get('w1')
            w3 = expert_linears.get('w3')
            w2 = expert_linears.get('w2')

            expert_cfg = FfnConfig.from_model_config(
                mc, tp_size=tp, tp_rank=0, dtype=dtype,
                act_type=_act_type_id(mc.activation_type),
                fuse_silu=False, inter_size=expert_inter,
                fused_moe=True)
            expert = FfnBuilder(expert_cfg, contexts, tp=tp, ranks=ranks)

            # TRANSFORM + COMMIT: builder handles w1+w3 fusion + TP split
            expert.add_ffn(w1, w2, w3)

            experts[str(e)] = expert

        moe.experts = experts
        parent.moe_ffn = moe

    # ------------------------------------------------------------------
    # Weight reading methods
    # ------------------------------------------------------------------

    def _read_linear(self, prefix: str) -> Linear | None:
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

    def attn_params(self, layer):
        params = {}
        sinks = self._get(
            f"{self._layer_prefix}.{layer}.self_attn.sinks")
        if sinks is not None:
            params["sinks"] = (sinks, SplitSide.OUTPUT)
        return params

    def moe_gate(self, layer):
        gates = {}
        gate = self._get(
            f"{self._layer_prefix}.{layer}.mlp.router.weight")
        if gate is not None:
            gate = gate.t() if gate.dim() > 1 else gate
            tensors = {"weight": gate}
            gate_bias = self._get(
                f"{self._layer_prefix}.{layer}.mlp.router.bias")
            if gate_bias is not None:
                tensors["bias"] = gate_bias
            gates["gate"] = Linear(tensors)
        return gates

    def moe_params(self, layer):
        return {}

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
