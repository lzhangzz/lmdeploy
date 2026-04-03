# Copyright (c) OpenMMLab. All rights reserved.
"""TextModelLoader: drives the model loading pipeline for text models."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .load_context import LoadContext

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

    def __call__(self, layer: int, spec: ModelWeightSpec):
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

    def _load_layer(self, layer: int, spec: ModelWeightSpec):
        for gpu in range(self.model.gpu_count):
            root = self.model.root(gpu)
            if root is None:
                break
            attn_rank, mlp_rank = self.model.tp_ranks(gpu)

            # Ensure layers ModuleList exists
            layers = root.get('layers')
            if layers is None:
                layers = root.create_child('layers', 'ModuleList', {})

            # Ensure this layer's entry exists
            layer_name = str(layer)
            layer_mod = layers.get(layer_name)
            if layer_mod is None:
                layer_mod = layers.create_child(layer_name, 'DecoderLayerWeight', {})

            tp_config = self._make_tp_config(attn_rank)
            ctx = LoadContext(layer_mod, tp_config)
            spec.load_layer(ctx, layer)

    def _load_global(self, spec: ModelWeightSpec):
        for gpu in range(self.model.gpu_count):
            root = self.model.root(gpu)
            if root is None:
                break
            attn_rank, _ = self.model.tp_ranks(gpu)
            tp_config = self._make_tp_config(attn_rank)
            ctx = LoadContext(root, tp_config)
            spec.load_global(ctx)
