# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

import torch
from mmengine import Registry

from ..kind_map import get_normalizer, get_suffix_map
from ..linear import Linear
from ..parameter import build_linear

INPUT_MODELS = Registry('source model', locations=['lmdeploy.turbomind.deploy.source_model.base'])


class BaseReader(ABC):
    """Mapping between TM modules and source modules."""

    params: dict[str, torch.Tensor]
    model_format: str | None = None

    def __init__(self):
        pass

    def transform(self, x: torch.Tensor | None, kind: str) -> torch.Tensor | None:
        return None if x is None else self._transform(x, kind)

    @abstractmethod
    def _transform(self, x: torch.Tensor, kind: str):
        """Transform x."""
        pass

    # -- New API: build Linear bundles from checkpoint keys --

    def read_linear(self, prefix: str, input_dim: int = 0,
                    output_dim: int = -1) -> Linear | None:
        """Build a ``Linear`` bundle for the checkpoint keys at *prefix*.

        Uses the reader's ``model_format`` to select the suffix map and
        normalizer, then probes ``prefix + suffix`` in ``self.params``.
        """
        suffix_map = get_suffix_map(self.model_format)
        normalizer = get_normalizer(self.model_format)
        return build_linear(
            self.params, prefix,
            suffix_map=suffix_map,
            normalizer=normalizer,
            input_dim=input_dim,
            output_dim=output_dim,
        )

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
