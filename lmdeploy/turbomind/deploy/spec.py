# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from abc import ABC, abstractmethod
import torch
from .linear import Linear
from .module_configs import SpecAttnConfig


class TextModelSpec(ABC):
    """Declarative weight mapping for a model architecture.

    Subclasses define how to read and transform weights for a specific model.
    The ``model()`` method is the entry point, called by TextModelLoader.
    """

    params: dict[str, torch.Tensor]

    # Default values for configure() fields; overwritten by configure().
    _attn_tp: int = 1
    _permute_qk: bool = True
    _repeat_kv: int = 0
    _head_dim: int = 0
    _rope_dim: int = 0
    _attn_output_gate: bool = False
    _kv_head_num: int = 0
    # TODO: there dont belong here
    _linear_qkv_split: tuple[int, int, int] | None = None

    # Injected by TextModelLoader
    _contexts: list = None
    _root_handles: list = None

    def model(self):
        """Build the full model hierarchy using builders. Override in subclasses."""

    def configure(self, cfg: SpecAttnConfig):
        """Set TP and model parameters. Called by TextModelLoader."""
        self._attn_tp = cfg.tp
        self._permute_qk = cfg.permute_qk
        self._repeat_kv = cfg.repeat_kv
        self._head_dim = cfg.head_dim
        self._rope_dim = cfg.rope_dim if cfg.rope_dim else cfg.head_dim
        self._attn_output_gate = cfg.output_gate
        self._kv_head_num = cfg.kv_head_num

    def _get(self, key: str) -> torch.Tensor | None:
        """Get a raw tensor from the checkpoint params."""
        return self.params.get(key)

    def _linear(self, pfx: str) -> Linear | None:
        """Read a Linear bundle from the checkpoint at *pfx*."""
        from .kind_map import build_linear
        return build_linear(self.params, pfx)

    @abstractmethod
    def model_info(self) -> dict:
        """Return model metadata (num_layer, head_num, etc.)."""

    def num_experts(self, layer: int) -> int:
        return 0
