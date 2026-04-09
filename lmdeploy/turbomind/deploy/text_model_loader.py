# Copyright (c) OpenMMLab. All rights reserved.
"""TextModelLoader: drives the model loading pipeline for text models."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .configs import (
    AttentionConfig, FfnConfig, MoeConfig, DeltaNetConfig, LinearConfig,
    SpecAttnConfig, ModuleListConfig, NormConfig, DecoderLayerConfig,
)
from .load_context import (
    _cpp_dtype, _act_type_id,
    _ATTN_TP_RULES, _FFN_TP_RULES, _LINEAR_ATTN_TP_RULES,
)
from .spec import SplitSide
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
        self._attn_ranks = None
        self._mlp_ranks = None

    def _ensure_ranks(self):
        """Compute per-GPU rank lists lazily (gpu_count may be 0 at __init__ time)."""
        if self._attn_ranks is None:
            self._attn_ranks = [self.model.tp_ranks(gpu)[0]
                                for gpu in range(self.model.gpu_count)]
            self._mlp_ranks = [self.model.tp_ranks(gpu)[1]
                               for gpu in range(self.model.gpu_count)]

    def _layer_writer(self, layer: int) -> Distributor:
        """Create a Distributor for the given layer across all GPUs."""
        handles = []
        for gpu in range(self.model.gpu_count):
            root = self.model.root(gpu)
            if root is None:
                break
            layers = root.child('layers') or \
                root.create_child('layers', ModuleListConfig().to_cpp())
            layer_mod = layers.child(str(layer)) or \
                layers.create_child(str(layer), DecoderLayerConfig().to_cpp())
            handles.append(layer_mod)
        return Distributor(handles)

    def __call__(self, layer: int, spec: 'TextModelSpec'):
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

        # --- Parameters (q_norm, k_norm, sinks, etc.) ---
        for name, (tensor, split_side) in spec.attn_params(layer).items():
            parts = name.split('.')
            parent = attn
            for seg in parts[:-1]:
                parent = parent.create_child(seg, NormConfig(
                    dim=tensor.shape[-1] if tensor.dim() >= 1 else 0,
                    data_type=dtype))
            parent.commit_tensor(parts[-1], tensor, split_side=split_side)

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
        hidden = mc.hidden_units

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

        # --- gate + shared_gate modules (created first, params committed from spec) ---
        gate_cfg = LinearConfig(
            input_dim=hidden,
            output_dim=spec.num_experts(layer),
            data_type=dtype,
            has_bias=getattr(mc, 'expert_router_bias', False))
        moe.create_child('gate', gate_cfg)

        if mc.moe_shared_gate:
            shared_gate_cfg = LinearConfig(
                input_dim=hidden, output_dim=1, data_type=dtype, has_bias=False)
            moe.create_child('shared_gate', shared_gate_cfg)

        # --- Non-expert MoE parameters (gate/shared_gate weights, score_correction_bias) ---
        for name, (tensor, split_side) in spec.moe_params(layer).items():
            parts = name.split('.')
            parent = moe
            for seg in parts[:-1]:
                existing = parent._handles[0].child(seg) if parent._handles else None
                if existing is not None:
                    children = [h.child(seg) for h in parent._handles]
                    parent = Distributor(children)
                else:
                    parent = parent.create_child(seg, NormConfig(
                        dim=tensor.shape[-1] if tensor.dim() >= 1 else 0,
                        data_type=dtype))
            parent.commit_tensor(parts[-1], tensor, split_side=split_side)

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

        # --- Parameters (A_log, dt_bias, conv1d, norm.weight) ---
        for name, (tensor, split_side) in spec.linear_attn_params(layer).items():
            parts = name.split('.')
            parent = linear_attn
            for seg in parts[:-1]:
                parent = parent.create_child(seg, NormConfig(
                    dim=tensor.shape[-1] if tensor.dim() >= 1 else 0,
                    data_type=dtype))
            parent.commit_tensor(parts[-1], tensor, split_side=split_side)

    # ------------------------------------------------------------------
    # Top-level orchestration
    # ------------------------------------------------------------------

    def _load_layer(self, layer: int, spec: 'TextModelSpec'):
        self._ensure_ranks()
        mc = self.model.model_config
        rope_param = self.model.attention_config.rope_param
        spec.configure(SpecAttnConfig(
            tp=self.attn_tp,
            permute_qk=getattr(self.model, 'permute_qk', True),
            repeat_kv=getattr(self.model, 'repeat_kv', 0),
            head_dim=mc.size_per_head,
            rope_dim=rope_param.dim if rope_param else mc.size_per_head,
            output_gate=getattr(mc, 'attn_output_gate', False),
            kv_head_num=mc.kv_head_num,
        ))

        writer = self._layer_writer(layer)

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

        for gpu in range(self.model.gpu_count):
            root = self.model.root(gpu)
            if root is None:
                break
            attn_rank, _ = self.model.tp_ranks(gpu)
            dtype = _cpp_dtype(mc.data_type)
            hidden = mc.hidden_units

            # Token embeddings (column-parallel)
            emb = spec.tok_embeddings()
            if emb is not None:
                emb_padded = pad_out_dim(emb, padded_vocab, dim=0)
                tok_cfg = LinearConfig(
                    input_dim=padded_vocab,
                    output_dim=hidden // tp,
                    data_type=dtype)
                tok_emb = root.create_child('tok_embeddings', tok_cfg.to_cpp())
                commit_tensor(tok_emb, emb_padded,
                                     'weight',
                                     split_side=SplitSide.OUTPUT,
                                     split_num=tp, rank=attn_rank)

            # Final norm (broadcast)
            norm = spec.norm_weight()
            if norm is not None:
                import _turbomind as _tm
                norm_cfg = _tm.NormConfig()
                norm_cfg.dim = hidden
                norm_cfg.data_type = _tm.DataType(dtype) if isinstance(dtype, int) else dtype
                norm_mod = root.create_child('norm', norm_cfg)
                commit_tensor(norm_mod, norm, 'weight')

            # Output head (column-parallel, transposed)
            output = spec.output_weight()
            if output is not None:
                output_padded = pad_out_dim(output, padded_vocab, dim=0)
                output_t = output_padded.t()
                out_cfg = LinearConfig(
                    input_dim=hidden,
                    output_dim=padded_vocab // tp,
                    data_type=dtype)
                output_mod = root.create_child('output', out_cfg.to_cpp())
                commit_tensor(output_mod, output_t, 'weight',
                                     split_side=SplitSide.OUTPUT,
                                     split_num=tp, rank=attn_rank)
