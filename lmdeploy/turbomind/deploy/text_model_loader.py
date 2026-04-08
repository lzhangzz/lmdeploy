# Copyright (c) OpenMMLab. All rights reserved.
"""TextModelLoader: drives the model loading pipeline for text models."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .configs import (
    AttentionConfig, FfnConfig, MoeConfig, DeltaNetConfig, LinearConfig,
    SpecAttnConfig, ModuleListConfig, NormConfig, DecoderLayerConfig,
)
from .load_context import (
    LoadContext, _cpp_dtype, _act_type_id,
    commit_linear, commit_tensor,
    _ATTN_TP_RULES, _FFN_TP_RULES, _LINEAR_ATTN_TP_RULES,
    _fuse_and_commit_ffn,
)
from .spec import SplitSide
from .transforms import fuse_ffn_linears

if TYPE_CHECKING:
    from .spec import TextModelSpec
    from .target_model.base import BaseOutputModel


class LayerWriter:
    """Wraps all GPU handles for one logical layer.

    The GPU loop is internal.  Outside callers see single-layer semantics.
    """

    def __init__(self, handles, tp=1, ranks=None):
        self._handles = handles
        self._tp = tp
        self._ranks = ranks

    @property
    def tp_size(self):
        return self._tp

    def _rank_for(self, gpu_idx):
        if self._ranks and self._tp > 1:
            return self._ranks[gpu_idx]
        return 0

    def create_child(self, name, config, tp=None, ranks=None):
        """Create a typed module child on ALL GPUs.

        Calls ``config.for_rank(rank).to_cpp()`` per GPU.
        Returns a new LayerWriter scoped to the created children,
        with tp/ranks rebound if provided (otherwise inherited).
        """
        new_tp = tp if tp is not None else self._tp
        new_ranks = ranks if ranks is not None else self._ranks
        children = []
        for i, handle in enumerate(self._handles):
            rank = new_ranks[i] if new_ranks and new_tp > 1 else 0
            child = handle.create_child(name, config.for_rank(rank).to_cpp())
            children.append(child)
        return LayerWriter(children, tp=new_tp, ranks=new_ranks)

    def commit_linear(self, name, linear, split_side=None, model_dtype=None):
        """Commit a Linear bundle to all GPUs.

        If split_side is given, uses bound tp/ranks for sharding.
        If split_side is None, broadcasts (tp=1).
        """
        tp = self._tp if split_side else 1
        for i, handle in enumerate(self._handles):
            rank = self._rank_for(i) if tp > 1 else 0
            commit_linear(handle, linear, name,
                          split_side=split_side, split_num=tp,
                          rank=rank, model_dtype=model_dtype)

    def commit_tensor(self, name, tensor, split_side=None):
        """Commit a raw tensor to all GPUs."""
        tp = self._tp if split_side else 1
        for i, handle in enumerate(self._handles):
            rank = self._rank_for(i) if tp > 1 else 0
            commit_tensor(handle, tensor, name,
                          split_side=split_side, split_num=tp,
                          rank=rank)


class TextModelLoader:
    """Drives the model loading pipeline for text models.

    Replaces TransformerV2.  This is a generic driver with zero hardcoded
    module paths.  All structure comes from the TextModelSpec.
    """

    def __init__(self, model: BaseOutputModel):
        self.model = model
        self.attn_tp = model.attn_tp_size
        self.mlp_tp = model.mlp_tp_size
        self._attn_ranks = [model.tp_ranks(gpu)[0]
                            for gpu in range(model.gpu_count)]
        self._mlp_ranks = [model.tp_ranks(gpu)[1]
                           for gpu in range(model.gpu_count)]

    def _layer_writer(self, layer: int) -> LayerWriter:
        """Create a LayerWriter for the given layer across all GPUs."""
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
        return LayerWriter(handles)

    def __call__(self, layer: int, spec: 'TextModelSpec'):
        if layer < 0:
            self._load_global(spec)
        elif layer >= self.model.model_config.num_layer:
            return 0
        else:
            self._load_layer(layer, spec)
        return 1

    def _make_tp_config(self, rank: int, is_attn: bool = True) -> dict:
        tp = self.attn_tp if is_attn else self.mlp_tp
        cfg = self.model.model_config
        rope_param = self.model.attention_config.rope_param
        return {
            'tp_size': tp,
            'rank': rank,
            'head_dim': cfg.size_per_head,
            'rope_dim': rope_param.dim if rope_param else cfg.size_per_head,
            'permute_qk': getattr(self.model, 'permute_qk', True),
            'repeat_kv': getattr(self.model, 'repeat_kv', 0),
            'attn_output_gate': getattr(cfg, 'attn_output_gate', False),
            'kv_head_num': cfg.kv_head_num,
        }

    # ------------------------------------------------------------------
    # Per-component loading methods
    # ------------------------------------------------------------------

    def _load_norms(self, handle, spec: 'TextModelSpec', layer: int,
                    hidden: int, dtype):
        """Load attn_norm and ffn_norm weight tensors."""
        norm_cfg = {'dim': hidden, 'data_type': dtype}
        attention_norm = handle.create_child('attention_norm', 'NormWeight', norm_cfg)
        ffn_norm = handle.create_child('ffn_norm', 'NormWeight', norm_cfg)
        commit_tensor(attention_norm, spec.attn_norm(layer), 'weight')
        commit_tensor(ffn_norm, spec.ffn_norm(layer), 'weight')

    def _load_attention(self, handle, spec: 'TextModelSpec', layer: int,
                        mc, dtype, attn_rank: int):
        """Load attention weights (QKV, output projection, etc.)."""
        attn_linears = spec.attn_linears(layer)
        if not attn_linears:
            return

        window_size = 0
        ws_list = mc.window_size
        if ws_list and layer < len(ws_list):
            window_size = ws_list[layer]

        attn_cfg = AttentionConfig.from_model_config(
            mc, tp_size=self.attn_tp, tp_rank=attn_rank,
            dtype=dtype, window_size=window_size)
        attn_mod = handle.create_child('attention', attn_cfg.to_cpp())
        for name, lin in attn_linears.items():
            rule = _ATTN_TP_RULES.get(name, {})
            tp = self.attn_tp if 'split_side' in rule else 1
            commit_linear(attn_mod, lin, name,
                                split_num=tp, rank=attn_rank,
                                model_dtype=dtype, **rule)

    def _load_ffn(self, handle, spec: 'TextModelSpec', layer: int,
                  mc, dtype, mlp_rank: int):
        """Load dense FFN weights (gate, up, down projections)."""
        ffn_linears = spec.ffn_linears(layer)
        if not ffn_linears:
            return

        inter_size = 0
        is_list = mc.inter_size
        if is_list and layer < len(is_list):
            inter_size = is_list[layer]

        ffn_cfg = FfnConfig.from_model_config(
            mc, tp_size=self.mlp_tp, tp_rank=mlp_rank,
            dtype=dtype, act_type=_act_type_id(mc.activation_type),
            fuse_silu=True, inter_size=inter_size)
        ffn_mod = handle.create_child('feed_forward', ffn_cfg.to_cpp())
        w1 = ffn_linears.get('w1')
        w3 = ffn_linears.get('w3')
        w2 = ffn_linears.get('w2')
        if w1 is not None and w3 is not None:
            _fuse_and_commit_ffn(ffn_mod, w1, w3, w2,
                                 self.mlp_tp, mlp_rank,
                                 mc.activation_type, is_moe=False,
                                 model_dtype=dtype)
        else:
            for name, lin in ffn_linears.items():
                rule = _FFN_TP_RULES.get(name, {})
                tp = self.mlp_tp if 'split_side' in rule else 1
                commit_linear(ffn_mod, lin, name,
                                     split_num=tp, rank=mlp_rank,
                                     model_dtype=dtype, **rule)

    def _load_moe(self, handle, spec: 'TextModelSpec', layer: int,
                  mc, dtype, mlp_rank: int):
        """Load MoE weights (router, shared gate, expert FFNs)."""
        if spec.num_experts(layer) <= 0:
            return

        hidden = mc.hidden_units
        expert_num = 0
        en_list = mc.expert_num
        if en_list and layer < len(en_list):
            expert_num = en_list[layer]

        moe_cfg = MoeConfig.from_model_config(
            mc, layer_id=layer, tp_size=self.mlp_tp, tp_rank=mlp_rank,
            dtype=dtype, act_type=_act_type_id(mc.activation_type),
            fuse_silu=True, expert_num=expert_num)
        moe_mod = handle.create_child('moe_ffn', moe_cfg.to_cpp())

        # Create gate LinearWeight for router
        gate_linear = getattr(spec, 'moe_gate_linear', lambda l: None)(layer)
        if gate_linear is not None:
            commit_linear(moe_mod, gate_linear, 'gate',
                               model_dtype=dtype)
        else:
            # Spec handles gate via raw_layer_tensors; create the
            # LinearWeight child so the tensor is stored correctly.
            gate_cfg = LinearConfig(
                input_dim=hidden,
                output_dim=spec.num_experts(layer),
                data_type=dtype,
                has_bias=getattr(mc, 'expert_router_bias', False))
            moe_mod.create_child('gate', gate_cfg.to_cpp())

        # Create shared_gate if needed
        shared_gate_linear = getattr(spec, 'moe_shared_gate_linear', lambda l: None)(layer)
        if shared_gate_linear is not None:
            commit_linear(moe_mod, shared_gate_linear, 'shared_gate',
                               model_dtype=dtype)
        elif mc.moe_shared_gate:
            shared_gate_cfg = LinearConfig(
                input_dim=hidden,
                output_dim=1,
                data_type=dtype,
                has_bias=False)
            moe_mod.create_child('shared_gate', shared_gate_cfg.to_cpp())

        # Create experts ModuleList and expert children
        experts_list = moe_mod.create_child('experts', 'ModuleList', {})
        expert_inter = mc.expert_inter_size or 0
        for e in range(spec.num_experts(layer)):
            expert_name = str(e)
            expert_cfg = FfnConfig.from_model_config(
                mc, tp_size=self.mlp_tp, tp_rank=mlp_rank,
                dtype=dtype, act_type=_act_type_id(mc.activation_type),
                fuse_silu=True, inter_size=expert_inter, fused_moe=True)
            experts_list.create_child(expert_name, expert_cfg.to_cpp())

        for e in range(spec.num_experts(layer)):
            expert_linears = spec.moe_ffn_linears(layer, e)
            expert_mod = moe_mod.child('experts').child(str(e))
            w1 = expert_linears.get('w1')
            w3 = expert_linears.get('w3')
            w2 = expert_linears.get('w2')
            if w1 is not None and w3 is not None:
                _fuse_and_commit_ffn(expert_mod, w1, w3, w2,
                                     self.mlp_tp, mlp_rank,
                                     mc.activation_type, is_moe=True,
                                     model_dtype=dtype)
            else:
                for name, lin in expert_linears.items():
                    rule = _FFN_TP_RULES.get(name, {})
                    tp = self.mlp_tp if 'split_side' in rule else 1
                    commit_linear(expert_mod, lin, name,
                                         split_num=tp, rank=mlp_rank,
                                         model_dtype=dtype, **rule)

    def _load_linear_attn(self, handle, spec: 'TextModelSpec', layer: int,
                          mc, dtype, attn_rank: int):
        """Load linear-attention (DeltaNet / GDN) weights."""
        la_linears = spec.linear_attn_linears(layer)
        if not la_linears:
            return

        dn_cfg = DeltaNetConfig.from_model_config(
            mc, tp_size=self.attn_tp, tp_rank=attn_rank, dtype=dtype)
        linear_attn_mod = handle.create_child('linear_attn', dn_cfg.to_cpp())
        for name, lin in la_linears.items():
            rule = _LINEAR_ATTN_TP_RULES.get(name, {})
            tp = self.attn_tp if 'split_side' in rule else 1
            commit_linear(linear_attn_mod, lin, name,
                                split_num=tp, rank=attn_rank,
                                model_dtype=dtype, **rule)

    def _load_raw_tensors(self, handle, spec: 'TextModelSpec', layer: int,
                          dtype, attn_rank: int):
        """Load raw per-layer tensors (embeddings, biases, etc.)."""
        for tm_path, tensor, split_side in spec.raw_layer_tensors(layer):
            tp = self.attn_tp if split_side is not None else 1
            rank = attn_rank
            parts = tm_path.split('.')
            mod = handle
            for seg in parts[:-1]:
                child = mod.child(seg)
                if child is None:
                    # Auto-create missing intermediate modules as
                    # NormWeight (generic parameter holder).
                    # Adjust dimensions for TP split so allocation
                    # matches the per-shard tensor size.
                    if tensor.dim() > 1:
                        shape_list = list(tensor.shape)
                        if split_side is not None and tp > 1:
                            split_dim_idx = -1 if split_side == SplitSide.OUTPUT else 0
                            shape_list[split_dim_idx] //= tp
                        dims_str = ' '.join(str(s) for s in shape_list)
                        child = mod.create_child(
                            seg, 'NormWeight',
                            {'dims': dims_str, 'data_type': dtype})
                    else:
                        norm_dim = tensor.shape[-1] if tensor.dim() >= 1 else 0
                        if split_side is not None and tp > 1:
                            norm_dim //= tp
                        child = mod.create_child(
                            seg, 'NormWeight',
                            {'dim': norm_dim, 'data_type': dtype})
                mod = child
            commit_tensor(mod, tensor, parts[-1],
                                split_side=split_side,
                                split_num=tp, rank=rank)

    # ------------------------------------------------------------------
    # Top-level orchestration
    # ------------------------------------------------------------------

    def _load_layer(self, layer: int, spec: 'TextModelSpec'):
        mc = self.model.model_config

        for gpu in range(self.model.gpu_count):
            root = self.model.root(gpu)
            if root is None:
                break
            attn_rank, mlp_rank = self.model.tp_ranks(gpu)

            # Ensure layers ModuleList exists
            layers = root.child('layers') if root.child('layers') else \
                root.create_child('layers', 'ModuleList', {})

            # Ensure this layer's entry exists
            layer_name = str(layer)
            layer_mod = layers.child(layer_name)
            if layer_mod is None:
                layer_mod = layers.create_child(layer_name, 'DecoderLayerWeight', {})

            tp_config = self._make_tp_config(attn_rank)
            ctx = LoadContext(layer_mod, tp_config, mc)

            # Configure TP params for merge/fusion (idempotent)
            spec.configure(SpecAttnConfig(
                tp=ctx.tp_size,
                permute_qk=ctx._tp_config.get('permute_qk', True),
                repeat_kv=ctx.repeat_kv,
                head_dim=ctx.head_dim,
                rope_dim=ctx.rope_dim,
                output_gate=ctx.attn_output_gate,
                kv_head_num=ctx.kv_head_num,
            ))

            handle = layer_mod
            hidden = mc.hidden_units
            dtype = ctx.cpp_dtype

            self._load_norms(handle, spec, layer, hidden, dtype)
            self._load_attention(handle, spec, layer, mc, dtype, attn_rank)
            self._load_ffn(handle, spec, layer, mc, dtype, mlp_rank)
            self._load_moe(handle, spec, layer, mc, dtype, mlp_rank)
            self._load_linear_attn(handle, spec, layer, mc, dtype, attn_rank)
            self._load_raw_tensors(handle, spec, layer, dtype, attn_rank)

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
            tp_config = self._make_tp_config(attn_rank)
            ctx = LoadContext(root, tp_config, mc)
            dtype = ctx.cpp_dtype
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
                norm_mod = root.create_child('norm', 'NormWeight',
                                  {'dim': hidden, 'data_type': dtype})
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
