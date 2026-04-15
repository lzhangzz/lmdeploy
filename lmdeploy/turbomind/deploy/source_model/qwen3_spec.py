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

    # ------------------------------------------------------------------
    # Builder-driven loading: build full model hierarchy
    # ------------------------------------------------------------------

    def _cpp_dtype(self):
        from ..builder import _cpp_dtype as _cd
        return _cd(self._mc.data_type)

    def model(self):
        from ..builder import TextModelBuilder

        root = TextModelBuilder(self._root_handles, self._contexts)
        root.tok_embeddings = self.token_embeds()
        root.norm = self.root_norm()
        root.output = self.lm_head()
        root.layers = self.layers('model.layers')

    # ------------------------------------------------------------------
    # Factory methods: read weights, create builders, return them
    # ------------------------------------------------------------------

    def token_embeds(self):
        """Return LinearBuilder for tok_embeddings, or None."""
        from ..builder import LinearBuilder, SplitSide
        from ..module_configs import LinearConfig
        from ..linear import pad_out_dim

        emb = self.tok_embeddings()
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

    def root_norm(self):
        """Return NormBuilder for the final norm, or None."""
        from ..builder import NormBuilder
        from ..module_configs import NormConfig

        w = self.norm_weight()
        if w is None:
            return None

        mc = self._mc
        dtype = self._cpp_dtype()
        cfg = NormConfig(dim=mc.hidden_units, data_type=dtype)
        m = NormBuilder(cfg, self._contexts)
        m.set_weight(w)
        return m

    def lm_head(self):
        """Return LinearBuilder for the output head, or None."""
        from ..builder import LinearBuilder, SplitSide
        from ..module_configs import LinearConfig
        from ..linear import pad_out_dim

        output = self.output_weight()
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
        from ..builder import NormBuilder
        from ..module_configs import NormConfig

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
        from ..builder import AttentionBuilder
        from ..module_configs import AttentionConfig

        q = self._read_linear(f"{pfx}.q_proj")
        k = self._read_linear(f"{pfx}.k_proj")
        v = self._read_linear(f"{pfx}.v_proj")
        o = self._read_linear(f"{pfx}.o_proj")

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

        for name, tensor in self.attn_params(layer).items():
            attn.add_param(name, tensor)

        norm_children = self.attn_norm_children(layer)
        q_norm = norm_children.get('q_norm')
        k_norm = norm_children.get('k_norm')
        if q_norm is not None or k_norm is not None:
            attn.add_qk_norm(q_norm, k_norm)

        return attn

    def ffn(self, pfx, layer, linears=None, inter_size=None,
            fused_moe=False):
        """Return FfnBuilder for the given layer, or None."""
        from ..builder import FfnBuilder
        from ..module_configs import FfnConfig
        from ..builder import _act_type_id

        if linears is None:
            linears = self._read_ffn_linears(pfx)
        if not linears:
            return None

        w1 = linears.get('w1')
        w3 = linears.get('w3')
        w2 = linears.get('w2')

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
        from ..builder import MoeBuilder, ModuleListBuilder
        from ..module_configs import MoeConfig, ModuleListConfig
        from ..builder import _act_type_id

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

        for name, linear in self.moe_gate(layer).items():
            m.add_gate(name, linear, model_dtype=dtype)

        for name, tensor in self.moe_params(layer).items():
            m.add_param(name, tensor)

        expert_inter = mc.expert_inter_size or 0
        experts = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for e in range(self.num_experts(layer)):
            expert = self.ffn(
                f'{pfx}.experts.{e}', layer,
                linears=self.moe_ffn_linears(layer, e),
                inter_size=expert_inter, fused_moe=True)
            if expert is not None:
                experts[str(e)] = expert

        m.experts = experts
        return m

    def layers(self, pfx):
        """Return ModuleListBuilder with all decoder layers."""
        from ..builder import ModuleListBuilder, DecoderLayerBuilder
        from ..module_configs import ModuleListConfig, DecoderLayerConfig

        layers = ModuleListBuilder(ModuleListConfig(), self._contexts)

        for i in range(self._num_layer):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._contexts)
            d.attention_norm = self.norm(
                f'{pfx}.{i}.input_layernorm')
            d.attention = self.attn(
                f'{pfx}.{i}.self_attn', i)
            d.ffn_norm = self.norm(
                f'{pfx}.{i}.post_attention_layernorm')
            if self.num_experts(i) > 0:
                d.moe_ffn = self.moe(
                    f'{pfx}.{i}.mlp', i)
            else:
                d.feed_forward = self.ffn(
                    f'{pfx}.{i}.mlp', i)
            layers[str(i)] = d

        return layers

    # ------------------------------------------------------------------
    # Weight reading methods
    # ------------------------------------------------------------------

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
        return {}

    def attn_norm_children(self, layer):
        params = {}
        q = self._get(f"{self._layer_prefix}.{layer}.self_attn.q_norm.weight")
        k = self._get(f"{self._layer_prefix}.{layer}.self_attn.k_norm.weight")
        if q is not None and k is not None:
            q, k = self._permute_qk_tensors(q, k)
        if q is not None:
            params["q_norm"] = q
        if k is not None:
            params["k_norm"] = k
        return params

    def moe_gate(self, layer):
        gates = {}
        if self._n_experts > 0:
            gate = self._get(
                f"{self._layer_prefix}.{layer}.mlp.gate.weight")
            if gate is not None:
                gate = gate.t() if gate.dim() > 1 else gate
                gates["gate"] = Linear({"weight": gate})
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
