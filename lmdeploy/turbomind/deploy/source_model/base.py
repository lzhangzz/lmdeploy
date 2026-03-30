# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

import torch
from mmengine import Registry

from ..linear import Linear
from ..parameter import build_linear

INPUT_MODELS = Registry('source model', locations=['lmdeploy.turbomind.deploy.source_model.base'])


class BaseReader(ABC):
    """Mapping between TM modules and source modules."""

    params: dict[str, torch.Tensor]

    def __init__(self):
        pass

    def transform(self, x: torch.Tensor | None, kind: str) -> torch.Tensor | None:
        return None if x is None else self._transform(x, kind)

    @abstractmethod
    def _transform(self, x: torch.Tensor, kind: str):
        """Transform x."""
        pass

    # -- New API: build Linear bundles from checkpoint keys --

    def read_linear(self, prefix: str) -> Linear | None:
        """Build a ``Linear`` bundle for the checkpoint keys at *prefix*.

        Probes all known suffixes and auto-detects the format via
        ``WeightFormat.accepts``.
        """
        return build_linear(self.params, prefix)

    def get(self, key: str) -> torch.Tensor | None:
        """Retrieve a single raw tensor by its full checkpoint key."""
        t = self.params.get(key)
        if t is not None:
            t = self.transform(t, "weight")
        return t


class BaseInputModel(ABC):
    """Base class for input model."""

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

    @abstractmethod
    def readers(self) -> Iterator[BaseReader]:
        pass
