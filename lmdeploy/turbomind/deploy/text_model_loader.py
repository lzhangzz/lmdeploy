# Copyright (c) OpenMMLab. All rights reserved.
"""TextModelLoader: drives the model loading pipeline for text models."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .module_configs import (
    AttentionConfig, FfnConfig, MoeConfig, DeltaNetConfig, LinearConfig,
    SpecAttnConfig, ModuleListConfig, NormConfig, DecoderLayerConfig,
)
from .commit import (
    _cpp_dtype, _act_type_id,
    _ATTN_TP_RULES, _LINEAR_ATTN_TP_RULES,
)
from .builder import SplitSide
from .transforms import fuse_ffn_linears
from .distributor import Distributor

if TYPE_CHECKING:
    from .spec import TextModelSpec
    from .target_model.base import BaseOutputModel


class TextModelLoader:
    """Drives the model loading pipeline for text models.

    Replaces TransformerV2.  This is a generic driver with zero hardcoded
    module paths.  All structure comes from the TextModelSpec.
    """

    def __init__(self, model: BaseOutputModel):
        self.model = model
        self.attn_tp = model.attn_tp_size
        self.mlp_tp = model.mlp_tp_size

        # Eagerly initialize distributors
        self._attn_ranks = [model.tp_ranks(gpu)[0]
                            for gpu in range(model.gpu_count)]
        self._mlp_ranks = [model.tp_ranks(gpu)[1]
                           for gpu in range(model.gpu_count)]
        handles = []
        contexts = []
        for gpu in range(model.gpu_count):
            root = model.root(gpu)
            if root is None:
                break
            handles.append(root)
            contexts.append(model.context(gpu))
        self._contexts = contexts
        self._root = Distributor(handles, contexts)
        self._layers = self._root.create_child('layers', ModuleListConfig())

    def __call__(self, layer: int, spec: 'TextModelSpec'):
        # Try builder-driven path first
        try:
            spec._contexts = self._contexts
            spec._root_handles = [h for h in self._root._handles]
            # Inject model config and TP configuration
            spec._mc = self.model.model_config
            spec._attn_tp = self.attn_tp
            spec._attn_cp = self.model.attn_cp_size
            spec._mlp_tp = self.mlp_tp
            spec._attn_ranks = self._attn_ranks
            spec._mlp_ranks = self._mlp_ranks
            spec._repeat_kv = self.model.repeat_kv
            # Configure spec for QKV merge (needed by attn_linears via _read_attn_linears)
            mc = self.model.model_config
            rope_param = self.model.attention_config.rope_param
            spec.configure(SpecAttnConfig(
                tp=self.attn_tp,
                repeat_kv=self.model.repeat_kv,
                head_dim=mc.size_per_head,
                rope_dim=rope_param.dim if rope_param else mc.size_per_head,
                output_gate=mc.attn_output_gate,
                kv_head_num=mc.kv_head_num,
            ))
            spec.model()
            return 1
        except NotImplementedError:
            pass  # Fall through to legacy path

        # Legacy path (kept until all specs are migrated)
        if layer < 0:
            self._load_global(spec)
        elif layer >= self.model.model_config.num_layer:
            return 0
        else:
            self._load_layer(layer, spec)
        return 1

    # ------------------------------------------------------------------
    # Per-component processing methods (read -> transform -> commit)
    # ------------------------------------------------------------------

    def _process_norms(self, writer: Distributor, spec: 'TextModelSpec',
                       layer: int):
        """Read, transform, commit norm weights."""
        mc = self.model.model_config
        dtype = _cpp_dtype(mc.data_type)
        hidden = mc.hidden_units

        # --- READ ---
        attn_norm = spec.attn_norm(layer)
        ffn_norm = spec.ffn_norm(layer)

        # --- COMMIT ---
        norm_cfg = NormConfig(dim=hidden, data_type=dtype)
        attention_norm = writer.create_child('attention_norm', norm_cfg)
        ffn_norm_w = writer.create_child('ffn_norm', norm_cfg)
        attention_norm.commit_tensor('weight', attn_norm)
        ffn_norm_w.commit_tensor('weight', ffn_norm)

    def _process_attention(self, writer: Distributor, spec: 'TextModelSpec',
                           layer: int):
        """Read, transform, commit attention weights."""
        mc = self.model.model_config
        dtype = _cpp_dtype(mc.data_type)

        # --- READ ---
        attn_linears = spec.attn_linears(layer)
        if not attn_linears:
            return

        # --- COMMIT ---
        window_size = 0
        ws_list = mc.window_size
        if ws_list and layer < len(ws_list):
            window_size = ws_list[layer]

        attn_cfg = AttentionConfig.from_model_config(
            mc, tp_size=self.attn_tp, tp_rank=0,
            dtype=dtype, window_size=window_size)
        attn = writer.create_child('attention', attn_cfg,
                                   tp=self.attn_tp, ranks=self._attn_ranks)

        for name, lin in attn_linears.items():
            rule = _ATTN_TP_RULES.get(name, {})
            attn.commit_linear(name, lin, model_dtype=dtype, **rule)

        # --- Direct params (sinks, etc.) ---
        for name, (tensor, split_side) in spec.attn_params(layer).items():
            attn.commit_tensor(name, tensor, split_side=split_side)

        # --- Norm children (q_norm, k_norm, etc.) ---
        for name, tensor in spec.attn_norm_children(layer).items():
            child = attn.create_child(name, NormConfig(
                dim=tensor.shape[-1],
                data_type=dtype))
            child.commit_tensor('weight', tensor)

    def _process_ffn(self, writer: Distributor, spec: 'TextModelSpec',
                     layer: int):
        """Read, transform (fuse w1+w3), commit FFN weights."""
        mc = self.model.model_config
        dtype = _cpp_dtype(mc.data_type)

        # --- READ ---
        ffn_linears = spec.ffn_linears(layer)
        if not ffn_linears:
            return
        w1 = ffn_linears.get('w1')
        w3 = ffn_linears.get('w3')
        w2 = ffn_linears.get('w2')

        # --- TRANSFORM ---
        fused, fused_silu = (None, False)
        if w1 is not None and w3 is not None:
            fused, fused_silu = fuse_ffn_linears(
                w1, w3, self.mlp_tp, mc.activation_type, is_moe=False)

        # --- COMMIT ---
        inter_size = 0
        is_list = mc.inter_size
        if is_list and layer < len(is_list):
            inter_size = is_list[layer]

        ffn_cfg = FfnConfig.from_model_config(
            mc, tp_size=self.mlp_tp, tp_rank=0, dtype=dtype,
            act_type=_act_type_id(mc.activation_type),
            fuse_silu=fused_silu, inter_size=inter_size)
        ffn = writer.create_child('feed_forward', ffn_cfg,
                                  tp=self.mlp_tp, ranks=self._mlp_ranks)

        if fused is not None:
            ffn.commit_linear('w1w3', fused,
                              split_side=SplitSide.OUTPUT, model_dtype=dtype)
        else:
            if w1 is not None:
                ffn.commit_linear('w1', w1,
                                  split_side=SplitSide.OUTPUT, model_dtype=dtype)
            if w3 is not None:
                ffn.commit_linear('w3', w3,
                                  split_side=SplitSide.OUTPUT, model_dtype=dtype)

        if w2 is not None:
            ffn.commit_linear('w2', w2,
                              split_side=SplitSide.INPUT, model_dtype=dtype)

    def _process_moe(self, writer: Distributor, spec: 'TextModelSpec',
                     layer: int):
        """Read, transform, commit MoE weights (per-expert iteration)."""
        if spec.num_experts(layer) <= 0:
            return
        mc = self.model.model_config
        dtype = _cpp_dtype(mc.data_type)

        expert_num = 0
        en_list = mc.expert_num
        if en_list and layer < len(en_list):
            expert_num = en_list[layer]

        moe_cfg = MoeConfig.from_model_config(
            mc, layer_id=layer, tp_size=self.mlp_tp, tp_rank=0,
            dtype=dtype, act_type=_act_type_id(mc.activation_type),
            fuse_silu=True, expert_num=expert_num)
        moe = writer.create_child('moe_ffn', moe_cfg,
                                  tp=self.mlp_tp, ranks=self._mlp_ranks)

        # --- gate linears ---
        for name, linear in spec.moe_gate(layer).items():
            moe.commit_linear(name, linear, model_dtype=dtype)

        # --- non-expert MoE parameters (score_correction_bias, etc.) ---
        for name, (tensor, split_side) in spec.moe_params(layer).items():
            moe.commit_tensor(name, tensor, split_side=split_side)

        # --- experts: per-expert READ -> TRANSFORM -> CREATE -> COMMIT ---
        expert_inter = mc.expert_inter_size or 0
        experts = moe.create_child('experts', ModuleListConfig())
        for e in range(spec.num_experts(layer)):
            # READ
            expert_linears = spec.moe_ffn_linears(layer, e)
            w1 = expert_linears.get('w1')
            w3 = expert_linears.get('w3')
            w2 = expert_linears.get('w2')

            # TRANSFORM (must happen before CREATE so fused_silu is known)
            fused, fused_silu = (None, False)
            if w1 is not None and w3 is not None:
                fused, fused_silu = fuse_ffn_linears(
                    w1, w3, self.mlp_tp, mc.activation_type, is_moe=True)

            # CREATE with the correct fuse_silu flag
            expert_cfg = FfnConfig.from_model_config(
                mc, tp_size=self.mlp_tp, tp_rank=0, dtype=dtype,
                act_type=_act_type_id(mc.activation_type),
                fuse_silu=fused_silu, inter_size=expert_inter, fused_moe=True)
            expert = experts.create_child(str(e), expert_cfg)

            # COMMIT
            if fused is not None:
                expert.commit_linear('w1w3', fused,
                                     split_side=SplitSide.OUTPUT, model_dtype=dtype)
            else:
                if w1 is not None:
                    expert.commit_linear('w1', w1,
                                         split_side=SplitSide.OUTPUT, model_dtype=dtype)
                if w3 is not None:
                    expert.commit_linear('w3', w3,
                                         split_side=SplitSide.OUTPUT, model_dtype=dtype)

            if w2 is not None:
                expert.commit_linear('w2', w2,
                                     split_side=SplitSide.INPUT, model_dtype=dtype)

    def _process_linear_attn(self, writer: Distributor, spec: 'TextModelSpec',
                             layer: int):
        """Read, transform, commit linear-attention (DeltaNet) weights."""
        mc = self.model.model_config
        dtype = _cpp_dtype(mc.data_type)

        # --- READ ---
        la_linears = spec.linear_attn_linears(layer)
        if not la_linears:
            return

        # --- COMMIT ---
        dn_cfg = DeltaNetConfig.from_model_config(
            mc, tp_size=self.attn_tp, tp_rank=0, dtype=dtype)
        linear_attn = writer.create_child('linear_attn', dn_cfg,
                                          tp=self.attn_tp, ranks=self._attn_ranks)

        for name, lin in la_linears.items():
            rule = _LINEAR_ATTN_TP_RULES.get(name, {})
            linear_attn.commit_linear(name, lin, model_dtype=dtype, **rule)

        # --- Direct params (A_log, dt_bias, conv1d, etc.) ---
        for name, (tensor, split_side) in spec.linear_attn_params(layer).items():
            linear_attn.commit_tensor(name, tensor, split_side=split_side)

        # --- Norm children (norm, etc.) ---
        for name, tensor in spec.linear_attn_norm_children(layer).items():
            child = linear_attn.create_child(name, NormConfig(
                dim=tensor.shape[-1],
                data_type=dtype))
            child.commit_tensor('weight', tensor)

    # ------------------------------------------------------------------
    # Top-level orchestration
    # ------------------------------------------------------------------

    def _load_layer(self, layer: int, spec: 'TextModelSpec'):
        mc = self.model.model_config
        rope_param = self.model.attention_config.rope_param
        spec.configure(SpecAttnConfig(
            tp=self.attn_tp,
            repeat_kv=self.model.repeat_kv,
            head_dim=mc.size_per_head,
            rope_dim=rope_param.dim if rope_param else mc.size_per_head,
            output_gate=mc.attn_output_gate,
            kv_head_num=mc.kv_head_num,
        ))

        writer = self._layers.create_child(str(layer), DecoderLayerConfig())

        self._process_norms(writer, spec, layer)
        self._process_attention(writer, spec, layer)
        self._process_ffn(writer, spec, layer)
        self._process_moe(writer, spec, layer)
        self._process_linear_attn(writer, spec, layer)

    def _load_global(self, spec: 'TextModelSpec'):
        from .linear import pad_out_dim

        mc = self.model.model_config
        tp = self.attn_tp * self.model.attn_cp_size
        padded_vocab = ((mc.vocab_size + tp - 1) // tp) * tp
        dtype = _cpp_dtype(mc.data_type)
        hidden = mc.hidden_units

        root = self._root

        # Token embeddings (column-parallel)
        emb = spec.tok_embeddings()
        if emb is not None:
            emb_padded = pad_out_dim(emb, padded_vocab, dim=0)
            tok_cfg = LinearConfig(
                input_dim=padded_vocab,
                output_dim=hidden // tp,
                data_type=dtype)
            tok_emb = root.create_child('tok_embeddings', tok_cfg,
                                        tp=tp, ranks=self._attn_ranks)
            tok_emb.commit_tensor('weight', emb_padded,
                                  split_side=SplitSide.OUTPUT)

        # Final norm (broadcast)
        norm = spec.norm_weight()
        if norm is not None:
            norm_cfg = NormConfig(dim=hidden, data_type=dtype)
            norm_mod = root.create_child('norm', norm_cfg)
            norm_mod.commit_tensor('weight', norm)

        # Output head (column-parallel, transposed)
        output = spec.output_weight()
        if output is not None:
            output_padded = pad_out_dim(output, padded_vocab, dim=0)
            output_t = output_padded.t()
            out_cfg = LinearConfig(
                input_dim=hidden,
                output_dim=padded_vocab // tp,
                data_type=dtype)
            output_mod = root.create_child('output', out_cfg,
                                           tp=tp, ranks=self._attn_ranks)
            output_mod.commit_tensor('weight', output_t,
                                     split_side=SplitSide.OUTPUT)
