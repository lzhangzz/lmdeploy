# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from mmengine import Registry

INPUT_MODELS = Registry('source model', locations=['lmdeploy.turbomind.deploy.source_model.base'])


class BaseInputModel(ABC):
    """Base class for input model."""

    # Subclasses set these to enable the default readers() implementation.
    _layer_pattern: str = ''
    _spec_class = None
    _loader_mappings: list = []

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        """Constructor for BaseInputModel.

        Args:
            model_path (str): the path of the model.
            tokenizer_path (str): the path of the tokenizer model.
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

    @abstractmethod
    def model_info(self) -> dict:
        """Read model info."""
        pass

    def readers(self) -> Iterator:
        """Yield a single ``(layer_id=-1, spec)`` pair with ALL weights."""
        import torch
        from ..loader import create_loader
        loader = create_loader(self.model_path, self._layer_pattern, self._loader_mappings)
        all_params = loader.all_items()
        yield -1, self._spec_class(all_params, self.model_config)
        torch.cuda.empty_cache()
