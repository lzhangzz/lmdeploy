# Copyright (c) OpenMMLab. All rights reserved.
"""BaseOutputModel — drives the spec through TextModelLoader + export."""
from __future__ import annotations

from abc import ABC

from mmengine import Registry

OUTPUT_MODELS = Registry('target model',
                         locations=['lmdeploy.turbomind.deploy.target_model.base'])


class BaseOutputModel(ABC):
    """Base output model. Drives a TextModelSpec through loading + commit."""

    def __init__(self, spec, model_comm, gpu_count, model_path):
        from ..text_model_loader import TextModelLoader
        self.spec = spec
        self.model_comm = model_comm
        self.gpu_count = gpu_count
        # model_path is writable by update_params (Queue takes over).
        self.model_path = model_path

        # Bind runtime handles onto the spec. TextModelLoader pulls
        # contexts/root_handles/ranks from model_comm.
        self.model = TextModelLoader(self)

    # ------------------------------------------------------------------
    # GPU-topology helpers (used by TextModelLoader)
    # ------------------------------------------------------------------

    def root(self, index: int):
        return self.model_comm.root(index)

    def context(self, index: int):
        return self.model_comm.context(index)

    def tp_ranks(self, index: int):
        return (self.model_comm.attn_tp_rank(index),
                self.model_comm.mlp_tp_rank(index),
                self.model_comm.model_tp_rank(index))

    # ------------------------------------------------------------------
    # Export drivers
    # ------------------------------------------------------------------

    def export(self) -> None:
        import torch

        from ..loader import create_loader
        loader = create_loader(self.model_path, self.spec._layer_pattern,
                               self.spec._loader_mappings)
        self.spec.set_params(loader.all_items())
        self.spec.model()
        torch.cuda.empty_cache()

    def export_iter(self):
        import torch

        from ..loader import create_loader
        loader = create_loader(self.model_path, self.spec._layer_pattern,
                               self.spec._loader_mappings)
        self.spec.set_params(loader.all_items())
        self.spec.model()
        yield -1
        # Runs on StopIteration; preserves old readers() behavior.
        torch.cuda.empty_cache()
