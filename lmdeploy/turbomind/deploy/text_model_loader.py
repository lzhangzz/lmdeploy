# Copyright (c) OpenMMLab. All rights reserved.
"""TextModelLoader: drives the model loading pipeline for text models."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .load_context import LoadContext, _act_type_id

if TYPE_CHECKING:
    from .module import ModelWeightSpec
    from .target_model.base import BaseOutputModel


class TextModelLoader:
    """Drives the model loading pipeline for text models.

    Replaces TransformerV2.  This is a generic driver with zero hardcoded
    module paths.  All structure comes from the ModelWeightSpec.
    """

    def __init__(self, model: BaseOutputModel):
        self.model = model
        self.attn_tp = model.attn_tp_size
        self.mlp_tp = model.mlp_tp_size

    def __call__(self, layer: int, spec: 'ModelWeightSpec'):
        if layer < 0:
            self._load_global(spec)
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

    def _load_layer(self, layer: int, spec: 'ModelWeightSpec'):
        from .module import (
            SplitSide,
            commit_linear_module,
            commit_tensor_module,
            _ATTN_TP_RULES,
            _FFN_TP_RULES,
            _LINEAR_ATTN_TP_RULES,
            _fuse_and_commit_ffn,
        )

        mc = self.model.model_config
        ec = self.model  # BaseOutputModel for engine-level config

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
            spec.configure(
                attn_tp=ctx.tp_size,
                permute_qk=ctx._tp_config.get('permute_qk', True),
                repeat_kv=ctx.repeat_kv,
                head_dim=ctx.head_dim,
                rope_dim=ctx.rope_dim,
                attn_output_gate=ctx.attn_output_gate,
                kv_head_num=ctx.kv_head_num,
            )

            handle = layer_mod
            hidden = mc.hidden_units
            dtype = ctx.cpp_dtype

            # --- Layer norms ---
            norm_cfg = {'dim': hidden, 'data_type': dtype}
            handle.create_child('attention_norm', 'NormWeight', norm_cfg)
            handle.create_child('ffn_norm', 'NormWeight', norm_cfg)
            commit_tensor_module(handle.get('attention_norm'),
                                spec.attn_norm(layer), 'weight')
            commit_tensor_module(handle.get('ffn_norm'),
                                spec.ffn_norm(layer), 'weight')

            # --- Attention ---
            attn_linears = spec.attn_linears(layer)
            if attn_linears:
                window_size = 0
                ws_list = mc.window_size
                if ws_list and layer < len(ws_list):
                    window_size = ws_list[layer]

                handle.create_child('attention', 'AttentionWeight', {
                    'hidden_dim': hidden,
                    'head_dim': mc.size_per_head,
                    'head_num': mc.head_num,
                    'kv_head_num': mc.kv_head_num,
                    'kv_lora_rank': mc.kv_lora_rank or 0,
                    'q_lora_rank': mc.q_lora_rank or 0,
                    'qk_rope_dim': mc.qk_rope_dim or 0,
                    'v_head_dim': mc.v_head_dim or 0,
                    'has_bias': mc.attn_bias,
                    'qk_norm': mc.qk_norm,
                    'tp_size': self.attn_tp,
                    'tp_rank': attn_rank,
                    'data_type': dtype,
                    'window_size': window_size,
                    'attn_sink': mc.attn_sink,
                    'attn_output_gate': mc.attn_output_gate,
                })
                attn_mod = handle.get('attention')
                for name, lin in attn_linears.items():
                    rule = _ATTN_TP_RULES.get(name, {})
                    tp = self.attn_tp if 'split_side' in rule else 1
                    commit_linear_module(attn_mod, lin, name,
                                         split_num=tp, rank=attn_rank, **rule)

            # --- Dense FFN ---
            ffn_linears = spec.ffn_linears(layer)
            if ffn_linears:
                inter_size = 0
                is_list = mc.inter_size
                if is_list and layer < len(is_list):
                    inter_size = is_list[layer]

                handle.create_child('feed_forward', 'FfnWeight', {
                    'hidden_dim': hidden,
                    'inter_size': inter_size,
                    'has_bias': mc.mlp_bias,
                    'tp_size': self.mlp_tp,
                    'tp_rank': mlp_rank,
                    'data_type': dtype,
                    'act_type': _act_type_id(mc.activation_type),
                    'fuse_silu_act': True,
                })
                ffn_mod = handle.get('feed_forward')
                w1 = ffn_linears.get('w1')
                w3 = ffn_linears.get('w3')
                w2 = ffn_linears.get('w2')
                if w1 is not None and w3 is not None:
                    _fuse_and_commit_ffn(ffn_mod, w1, w3, w2,
                                         self.mlp_tp, mlp_rank,
                                         mc.activation_type, is_moe=False)
                else:
                    for name, lin in ffn_linears.items():
                        rule = _FFN_TP_RULES.get(name, {})
                        tp = self.mlp_tp if 'split_side' in rule else 1
                        commit_linear_module(ffn_mod, lin, name,
                                             split_num=tp, rank=mlp_rank, **rule)

            # --- MoE ---
            if spec.num_experts(layer) > 0:
                expert_num = 0
                en_list = mc.expert_num
                if en_list and layer < len(en_list):
                    expert_num = en_list[layer]

                handle.create_child('moe_ffn', 'MoeWeight', {
                    'layer_id': layer,
                    'method': 1,  # kFused
                    'experts_per_token': mc.experts_per_token,
                    'inter_size': mc.expert_inter_size or 0,
                    'norm_topk_prob': mc.norm_topk_prob,
                    'shared_gate': mc.moe_shared_gate,
                    'routed_scale': float(mc.routed_scale),
                    'router_bias': getattr(mc, 'expert_router_bias', False),
                    'topk_group': mc.topk_group,
                    'topk_method': mc.topk_method,
                    'n_group': mc.moe_group_num,
                    'scoring_func': mc.scoring_func,
                    'router_n_groups': max(0, getattr(mc, 'router_n_groups', -1)),
                    'expert_num': expert_num,
                    'hidden_dim': hidden,
                    'mlp_bias': mc.mlp_bias,
                    'data_type': dtype,
                    'tp_size': self.mlp_tp,
                    'tp_rank': mlp_rank,
                    'act_type': _act_type_id(mc.activation_type),
                    'fuse_silu_act': True,
                })
                moe_mod = handle.get('moe_ffn')
                for e in range(spec.num_experts(layer)):
                    expert_linears = spec.moe_ffn_linears(layer, e)
                    expert_mod = moe_mod.get('experts').get(str(e))
                    w1 = expert_linears.get('w1')
                    w3 = expert_linears.get('w3')
                    w2 = expert_linears.get('w2')
                    if w1 is not None and w3 is not None:
                        _fuse_and_commit_ffn(expert_mod, w1, w3, w2,
                                             self.mlp_tp, mlp_rank,
                                             mc.activation_type, is_moe=True)
                    else:
                        for name, lin in expert_linears.items():
                            rule = _FFN_TP_RULES.get(name, {})
                            tp = self.mlp_tp if 'split_side' in rule else 1
                            commit_linear_module(expert_mod, lin, name,
                                                 split_num=tp, rank=mlp_rank, **rule)

            # --- Linear attention (GDN) ---
            for name, lin in spec.linear_attn_linears(layer).items():
                rule = _LINEAR_ATTN_TP_RULES.get(name, {})
                tp = self.attn_tp if 'split_side' in rule else 1
                linear_attn_mod = handle.get('linear_attn')
                commit_linear_module(linear_attn_mod, lin, name,
                                     split_num=tp, rank=attn_rank, **rule)

            # --- Raw per-layer tensors ---
            for tm_path, tensor, split_side in spec.raw_layer_tensors(layer):
                tp = self.attn_tp if split_side is not None else 1
                rank = attn_rank
                parts = tm_path.split('.')
                mod = handle
                for seg in parts[:-1]:
                    mod = mod.get(seg)
                commit_tensor_module(mod, tensor, parts[-1],
                                     split_side=split_side,
                                     split_num=tp, rank=rank)

    def _load_global(self, spec: 'ModelWeightSpec'):
        from .module import SplitSide, commit_tensor_module
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
                root.create_child('tok_embeddings', 'LinearWeight', {
                    'input_dim': padded_vocab,
                    'output_dim': hidden // tp,
                    'data_type': dtype,
                    'has_bias': False,
                })
                commit_tensor_module(root.get('tok_embeddings'), emb_padded,
                                     'weight',
                                     split_side=SplitSide.OUTPUT,
                                     split_num=tp, rank=attn_rank)

            # Final norm (broadcast)
            norm = spec.norm_weight()
            if norm is not None:
                root.create_child('norm', 'NormWeight',
                                  {'dim': hidden, 'data_type': dtype})
                commit_tensor_module(root.get('norm'), norm, 'weight')

            # Output head (column-parallel, transposed)
            output = spec.output_weight()
            if output is not None:
                output_padded = pad_out_dim(output, padded_vocab, dim=0)
                output_t = output_padded.t()
                root.create_child('output', 'LinearWeight', {
                    'input_dim': hidden,
                    'output_dim': padded_vocab // tp,
                    'data_type': dtype,
                    'has_bias': False,
                })
                commit_tensor_module(root.get('output'), output_t, 'weight',
                                     split_side=SplitSide.OUTPUT,
                                     split_num=tp, rank=attn_rank)
