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

    def model(self):
        from ..builder import (TextModelBuilder, ModuleListBuilder,
                               DecoderLayerBuilder, NormBuilder, Builder,
                               SplitSide)
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

        root = TextModelBuilder(self._root_handles, contexts)

        # --- tok_embeddings (column-parallel raw tensor) ---
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
                               split_side=SplitSide.OUTPUT)
            root.tok_embeddings = tok

        # --- final norm (broadcast) ---
        norm_w = self.norm_weight()
        if norm_w is not None:
            norm_cfg = NormConfig(dim=hidden, data_type=dtype)
            norm_b = NormBuilder(norm_cfg, contexts)
            norm_b.set_weight(norm_w)
            root.norm = norm_b

        # --- output head (column-parallel, transposed) ---
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
                               split_side=SplitSide.OUTPUT)
            root.output = out

        # --- decoder layers ---
        layers = ModuleListBuilder(ModuleListConfig(), contexts)

        for i in range(self._num_layer):
            d = DecoderLayerBuilder(DecoderLayerConfig(), contexts)

            # attention_norm (broadcast)
            attn_norm_w = self.attn_norm(i)
            if attn_norm_w is not None:
                n_cfg = NormConfig(dim=hidden, data_type=dtype)
                n_b = NormBuilder(n_cfg, contexts)
                n_b.set_weight(attn_norm_w)
                d.attention_norm = n_b

            # attention (builder-driven: spec reads, builder transforms)
            self._build_attention(d, i, mc, dtype, attn_tp, attn_ranks,
                                  contexts)

            # ffn_norm (broadcast)
            ffn_norm_w = self.ffn_norm(i)
            if ffn_norm_w is not None:
                n_cfg = NormConfig(dim=hidden, data_type=dtype)
                n_b = NormBuilder(n_cfg, contexts)
                n_b.set_weight(ffn_norm_w)
                d.ffn_norm = n_b

            # feed_forward or moe_ffn
            if self.num_experts(i) > 0:
                self._build_moe(d, i, mc, dtype, mlp_tp, mlp_ranks,
                                contexts)
            else:
                self._build_ffn(d, i, mc, dtype, mlp_tp, mlp_ranks,
                                contexts, child_name='feed_forward')

            layers[str(i)] = d

        root.layers = layers

    def _build_attention(self, parent, layer, mc, dtype, tp, ranks,
                         contexts):
        """Build attention module: spec reads projections, builder merges QKV."""
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

        # Direct params (none for Qwen3, but hook exists for subclasses)
        for name, tensor in self.attn_params(layer).items():
            attn.add_param(name, tensor)

        # Norm children (q_norm, k_norm) — spec already permuted them
        norm_children = self.attn_norm_children(layer)
        q_norm = norm_children.get('q_norm')
        k_norm = norm_children.get('k_norm')
        if q_norm is not None or k_norm is not None:
            attn.add_qk_norm(q_norm, k_norm)

        parent.attention = attn

    def _build_ffn(self, parent, layer, mc, dtype, tp, ranks, contexts,
                   child_name='feed_forward', inter_size=None,
                   fused_moe=False):
        """Build dense FFN: spec reads w1/w2/w3, builder handles fusion."""
        from ..builder import FfnBuilder
        from ..module_configs import FfnConfig
        from ..commit import _act_type_id

        ffn_linears = self.ffn_linears(layer)
        if not ffn_linears:
            return

        w1 = ffn_linears.get('w1')
        w3 = ffn_linears.get('w3')
        w2 = ffn_linears.get('w2')

        if inter_size is None:
            is_list = mc.inter_size
            inter_size = is_list[layer] if is_list and layer < len(
                is_list) else 0

        # fuse_silu=False initially; builder updates it based on fusion result
        ffn_cfg = FfnConfig.from_model_config(
            mc, tp_size=tp, tp_rank=0, dtype=dtype,
            act_type=_act_type_id(mc.activation_type),
            fuse_silu=False, inter_size=inter_size,
            fused_moe=fused_moe)
        ffn = FfnBuilder(ffn_cfg, contexts, tp=tp, ranks=ranks)

        # TRANSFORM + COMMIT: builder handles w1+w3 fusion + TP split
        ffn.add_ffn(w1, w2, w3)

        setattr(parent, child_name, ffn)

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
        for name, tensor in self.moe_params(layer).items():
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
