# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from mmengine import Registry

INPUT_MODELS = Registry('source model', locations=['lmdeploy.turbomind.deploy.source_model.base'])


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
    def readers(self) -> Iterator:
        pass
