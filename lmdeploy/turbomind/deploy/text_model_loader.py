# Copyright (c) OpenMMLab. All rights reserved.
"""TextModelLoader: injects context into specs and calls model()."""
from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .spec import TextModelSpec
    from .target_model.base import BaseOutputModel


class TextModelLoader:
    """Drives the model loading pipeline for text models.

    All structure comes from the TextModelSpec. The loader only injects
    GPU handles, contexts, and TP configuration, then calls spec.model().
    """

    def __init__(self, model: BaseOutputModel):
        self.model = model
        self.attn_tp = model.attn_tp_size
        self.mlp_tp = model.mlp_tp_size

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
        self._root_handles = handles

    def __call__(self, layer: int, spec: 'TextModelSpec'):
        spec._contexts = self._contexts
        spec._root_handles = self._root_handles
        spec._mc = self.model.model_config
        spec._attn_tp = self.attn_tp
        spec._attn_cp = self.model.attn_cp_size
        spec._mlp_tp = self.mlp_tp
        spec._attn_ranks = self._attn_ranks
        spec._mlp_ranks = self._mlp_ranks
        rope_param = self.model.attention_config.rope_param
        spec._rope_dim = rope_param.dim if rope_param else self.model.model_config.size_per_head
        spec.model()
        return 1
