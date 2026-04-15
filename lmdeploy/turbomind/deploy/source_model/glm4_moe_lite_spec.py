# Copyright (c) OpenMMLab. All rights reserved.
"""GLM-4 MoE Lite (GLM-4.7-Flash) TextModelSpec for the new pipeline.

Demonstrates the composable-ops pipeline with:
  - MLA (Multi-head Latent Attention) via ``read_linear``
  - MoE experts with dense first layer
  - noaux_tc routing with score correction bias
  - Raw tensors for norms, router, embeddings
"""
from __future__ import annotations

import os

import torch

from ..linear import Linear
from ..spec import TextModelSpec
from .base import INPUT_MODELS, BaseInputModel
from .utils import get_yarn_params, load_model_config, parse_rope_param

_LAYER_PATTERN = r'model\.layers\.([0-9]+).'


class Glm4MoeLiteSpec(TextModelSpec):
    """Weight spec for GLM-4 MoE Lite (e.g. GLM-4.7-Flash).

    Uses same key layout as DeepSeek2: ``model.layers.{i}.self_attn.*``,
    ``model.layers.{i}.mlp.*``.  First layer is dense FFN; remaining layers
    are MoE with ``n_routed_experts`` experts per layer.
    """

    _layer_prefix = "model.layers"

    def __init__(self, params: dict[str, torch.Tensor], model_cfg: dict):
        self.params = params
        self.cfg = model_cfg
        self._num_layer = model_cfg["num_hidden_layers"]
        self._n_experts = model_cfg.get("n_routed_experts", 0)
        self._first_k_dense = model_cfg.get("first_k_dense_replace", 1)

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
        num_layer = mc.num_layer

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

        for i in range(num_layer):
            d = DecoderLayerBuilder(DecoderLayerConfig(), contexts)

            # attention_norm (broadcast)
            attn_norm_w = self.attn_norm(i)
            if attn_norm_w is not None:
                n_cfg = NormConfig(dim=hidden, data_type=dtype)
                n_b = NormBuilder(n_cfg, contexts)
                n_b.set_weight(attn_norm_w)
                d.attention_norm = n_b

            # attention (MLA)
            self._build_attention(d, i, mc, dtype, attn_tp, attn_ranks,
                                  contexts)

            # ffn_norm (broadcast)
            ffn_norm_w = self.ffn_norm(i)
            if ffn_norm_w is not None:
                n_cfg = NormConfig(dim=hidden, data_type=dtype)
                n_b = NormBuilder(n_cfg, contexts)
                n_b.set_weight(ffn_norm_w)
                d.ffn_norm = n_b

            # feed_forward and/or moe_ffn
            if self.num_experts(i) > 0:
                # Shared expert as feed_forward
                self._build_ffn(d, i, mc, dtype, mlp_tp, mlp_ranks,
                                contexts, child_name='feed_forward')
                # MoE experts as moe_ffn
                self._build_moe(d, i, mc, dtype, mlp_tp, mlp_ranks,
                                contexts)
            else:
                # Dense FFN for first k layers
                self._build_ffn(d, i, mc, dtype, mlp_tp, mlp_ranks,
                                contexts, child_name='feed_forward')

            layers[str(i)] = d

        root.layers = layers

    def _build_attention(self, parent, layer, mc, dtype, tp, ranks,
                         contexts):
        """Build MLA attention: spec reads + folds projections, builder commits."""
        from ..builder import AttentionBuilder
        from ..module_configs import AttentionConfig

        pfx = f"{self._layer_prefix}.{layer}.self_attn"

        # READ: spec reads all MLA projections directly
        raw: dict = {}
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

        if not raw:
            return

        # Model-specific transform: MLA fold + pad (stays in spec)
        self._mla_fold_and_pad(raw)

        attn_cfg = AttentionConfig.from_model_config(
            mc, tp_size=tp, tp_rank=0, dtype=dtype,
            window_size=-1)
        attn = AttentionBuilder(attn_cfg, contexts, tp=tp, ranks=ranks)

        # COMMIT: use add_linear for each projection (MLA has no QKV merge)
        for name, lin in raw.items():
            attn.add_linear(name, lin)

        # Direct params (none for GLM-4)
        for name, val in self.attn_params(layer).items():
            if isinstance(val, tuple):
                tensor, ss = val
            else:
                tensor, ss = val, None
            attn.add_param(name, tensor)

        # Norm children (q_a_layernorm, kv_a_layernorm)
        for name, tensor in self.attn_norm_children(layer).items():
            attn._add_norm_child(name, tensor, data_type=dtype)

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

    # ---- MLA fold (model-specific weight transform) ----

    def _mla_fold_and_pad(self, linears: dict[str, Linear]):
        """Fold kv_b_proj into q_b_proj and wo, then pad wo.

        Mirrors the V1 ``MLA._export`` folding logic.  The fold arithmetic
        requires HF layout ``[out_features, in_features]``.  Weight tensors
        are temporarily transposed from TM to HF at the start and back to
        TM at the end.
        """
        # Temporarily convert weight tensors from TM [in, out] to HF [out, in].
        for lin in linears.values():
            for k in list(lin.tensors.keys()):
                t = lin.tensors[k]
                if t.dim() >= 2:
                    lin.tensors[k] = t.t().contiguous()
        try:
            self._mla_fold_and_pad_hf(linears)
        finally:
            # Convert weight tensors back from HF [out, in] to TM [in, out].
            for lin in linears.values():
                for k in list(lin.tensors.keys()):
                    t = lin.tensors[k]
                    if t.dim() >= 2:
                        lin.tensors[k] = t.t().contiguous()

    def _mla_fold_and_pad_hf(self, linears: dict[str, Linear]):
        """Inner fold logic; expects all weight tensors in HF layout [out, in]."""
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
        pfx = f"{self._layer_prefix}.{layer}.mlp"
        return self._read_ffn_linears(pfx)

    def _shared_expert_linears(self, layer: int) -> dict[str, Linear]:
        pfx = f"{self._layer_prefix}.{layer}.mlp.shared_experts"
        return self._read_ffn_linears(pfx)

    # ---- Linear bundles: MoE experts ----

    def moe_ffn_linears(self, layer: int, expert: int) -> dict[str, Linear]:
        pfx = f"{self._layer_prefix}.{layer}.mlp.experts.{expert}"
        return self._read_ffn_linears(pfx)

    def num_experts(self, layer: int) -> int:
        if layer < self._first_k_dense:
            return 0
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
        q_a = self._get(
            f"{self._layer_prefix}.{layer}.self_attn.q_a_layernorm.weight")
        kv_a = self._get(
            f"{self._layer_prefix}.{layer}.self_attn.kv_a_layernorm.weight")
        if q_a is not None:
            params["q_a_layernorm"] = q_a
        if kv_a is not None:
            params["kv_a_layernorm"] = kv_a
        return params

    def moe_gate(self, layer):
        gates = {}
        if self.num_experts(layer) > 0:
            gate = self._get(
                f"{self._layer_prefix}.{layer}.mlp.gate.weight")
            if gate is not None:
                gate = gate.t() if gate.dim() > 1 else gate
                tensors = {"weight": gate}
                gate_bias = self._get(
                    f"{self._layer_prefix}.{layer}.mlp.gate.bias")
                if gate_bias is not None:
                    tensors["bias"] = gate_bias
                gates["gate"] = Linear(tensors)
        return gates

    def moe_params(self, layer):
        params = {}
        if self.num_experts(layer) > 0:
            correction = self._get(
                f"{self._layer_prefix}.{layer}.mlp.gate.e_score_correction_bias")
            if correction is not None:
                params["score_correction_bias"] = (correction, None)
        return params

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


@INPUT_MODELS.register_module(name='glm4-moe-lite')
class Glm4MoeLiteInputModel(BaseInputModel):
    """Input model for GLM-4 MoE Lite (e.g. GLM-4.7-Flash)."""

    _layer_pattern = _LAYER_PATTERN
    _spec_class = Glm4MoeLiteSpec

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        super().__init__(model_path, tokenizer_path)
        self.model_config = load_model_config(model_path)
        self.model_format = kwargs.get('model_format')
        self.fp8_quant = kwargs.get('fp8_quant', False)

    def model_info(self) -> dict:
        cfg = self.model_config

        # --- base transformer fields ---
        attn_head_num = cfg['num_attention_heads']
        hidden_units = cfg['hidden_size']

        # --- MLA head geometry ---
        qk_nope_dim = cfg['qk_nope_head_dim']
        qk_rope_dim = cfg['qk_rope_head_dim']
        kv_lora_rank = cfg['kv_lora_rank']
        q_head_dim = qk_nope_dim + qk_rope_dim
        size_per_head = q_head_dim
        v_head_dim = cfg['v_head_dim']
        softmax_scale = 0.0
        disable_mla_fold = os.getenv('LMDEPLOY_MLA_FOLD', '1').lower() in ('0', 'false', 'no')
        if kv_lora_rank and kv_lora_rank != qk_nope_dim and not disable_mla_fold:
            size_per_head = kv_lora_rank + qk_rope_dim
            v_head_dim = kv_lora_rank
            softmax_scale = q_head_dim**(-0.5)
        elif kv_lora_rank and kv_lora_rank != qk_nope_dim:
            softmax_scale = q_head_dim**(-0.5)

        # --- RoPE (dim = qk_rope_dim for MLA) ---
        rope_param, max_position_embeddings = parse_rope_param(cfg, qk_rope_dim)

        # --- MoE layout ---
        num_layer = cfg['num_hidden_layers']
        n_routed_experts = cfg.get('n_routed_experts', 0)
        n_shared_experts = cfg.get('n_shared_experts', 1)
        expert_inter_size = cfg['moe_intermediate_size']
        first_k_dense = cfg.get('first_k_dense_replace', 1)
        expert_num = [n_routed_experts] * num_layer
        for i in range(first_k_dense):
            expert_num[i] = 0
        inter_size = [n_shared_experts * expert_inter_size] * num_layer
        inter_size[0] = cfg.get('intermediate_size', n_shared_experts * expert_inter_size)

        # Ensure required routing fields exist (GLM may omit them)
        topk_method = cfg.get('topk_method', 'noaux_tc')
        topk_group = cfg.get('topk_group', 1)
        n_group = cfg.get('n_group', 1)
        scoring_func = cfg.get('scoring_func', 'sigmoid')

        info = dict(
            num_layer=num_layer,
            norm_eps=cfg['rms_norm_eps'],
            head_num=attn_head_num,
            kv_head_num=1,
            hidden_units=hidden_units,
            size_per_head=size_per_head,
            vocab_size=cfg['vocab_size'],
            max_position_embeddings=max_position_embeddings,
            rope_param=rope_param,
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=cfg.get('q_lora_rank') or 0,
            qk_rope_dim=qk_rope_dim,
            v_head_dim=v_head_dim,
            inter_size=inter_size,
            expert_num=expert_num,
            expert_inter_size=expert_inter_size,
            experts_per_token=cfg['num_experts_per_tok'],
            norm_topk_prob=cfg.get('norm_topk_prob', True),
            routed_scale=cfg.get('routed_scaling_factor', 1.0),
            topk_method=topk_method,
            topk_group=topk_group,
            moe_group_num=n_group,
            scoring_func=scoring_func,
            tune_layer_num=2,
        )
        if softmax_scale:
            info['softmax_scale'] = softmax_scale

        # YaRN RoPE for MLA (override attention_factor + softmax_scale)
        if 'rope_parameters' in cfg:
            rope_scaling = cfg['rope_parameters']
        else:
            rope_scaling = cfg.get('rope_scaling')
        if rope_scaling and rope_scaling.get('type') == 'yarn':
            attention_factor, yarn_scale = get_yarn_params(rope_scaling)
            yarn_scale *= q_head_dim**(-0.5)
            rope_param.max_position_embeddings = rope_scaling['original_max_position_embeddings']
            rope_param.attention_factor = attention_factor
            info.update(rope_param=rope_param, softmax_scale=yarn_scale)

        if 'router_n_groups' in cfg and cfg['router_n_groups'] > 0:
            info['router_n_groups'] = cfg['router_n_groups']

        # GLM-specific overrides
        info['topk_method'] = 'noaux_tc'
        info['scoring_func'] = 'sigmoid'

        return info
