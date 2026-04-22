# Copyright (c) OpenMMLab. All rights reserved.
"""TextModelLoader: gathers runtime handles and binds them onto the spec."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .target_model.base import BaseOutputModel


class TextModelLoader:
    """Thin driver that binds GPU-topology handles from BaseOutputModel
    onto the spec. Future variants (VisionModelLoader, etc.) can coexist.
    """

    def __init__(self, model: 'BaseOutputModel'):
        self.model = model
        self._bind_runtime()

    def _bind_runtime(self):
        model = self.model
        attn_ranks = [model.tp_ranks(gpu)[0]
                      for gpu in range(model.gpu_count)]
        mlp_ranks = [model.tp_ranks(gpu)[1]
                     for gpu in range(model.gpu_count)]
        model_tp_ranks = [model.tp_ranks(gpu)[2]
                          for gpu in range(model.gpu_count)]
        handles = []
        contexts = []
        for gpu in range(model.gpu_count):
            root = model.root(gpu)
            if root is None:
                break
            handles.append(root)
            contexts.append(model.context(gpu))
        model.spec.bind_runtime(
            contexts=contexts,
            root_handles=handles,
            attn_ranks=attn_ranks,
            mlp_ranks=mlp_ranks,
            model_tp_ranks=model_tp_ranks,
        )
