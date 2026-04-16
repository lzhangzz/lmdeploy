# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from abc import ABC

import torch

from .builder import LinearBuilder, NormBuilder, SplitSide, _cpp_dtype as _cd
from .linear import Linear, pad_out_dim
from .module_configs import make_linear_config, make_norm_config


class TextModelSpec(ABC):
    """Declarative weight mapping for a model architecture.

    Subclasses define how to read and transform weights for a specific model.
    The ``model()`` method is the entry point, called by TextModelLoader.
    """

    params: dict[str, torch.Tensor]

    _attn_tp: int = 1
    _repeat_kv: int = 0
    _rope_dim: int = 0
    _linear_qkv_split: tuple[int, int, int] | None = None

    # Injected by TextModelLoader
    _contexts: list = None
    _root_handles: list = None
    _mc = None  # ModelConfig
    _attn_cp: int = 1
    _attn_ranks: list = None
    _mlp_tp: int = 1
    _mlp_ranks: list = None

    def model(self):
        """Build the full model hierarchy using builders. Override in subclasses."""

    def _cpp_dtype(self):
        return _cd(self._mc.data_type)

    def token_embeds(self, key):
        emb = self._get(key)
        mc = self._mc
        tp = self._attn_tp * self._attn_cp
        padded_vocab = ((mc.vocab_size + tp - 1) // tp) * tp
        emb_padded = pad_out_dim(emb, padded_vocab, dim=0)
        dtype = self._cpp_dtype()
        cfg = make_linear_config(input_dim=padded_vocab,
                                 output_dim=mc.hidden_units // tp,
                                 data_type=dtype)
        m = LinearBuilder(cfg, self._contexts, tp=tp, ranks=self._attn_ranks)
        m.set_weight(emb_padded, split_side=SplitSide.OUTPUT)
        return m

    def output_norm(self, key):
        w = self._get(key)
        cfg = make_norm_config(dim=self._mc.hidden_units, data_type=self._cpp_dtype())
        m = NormBuilder(cfg, self._contexts)
        m.set_weight(w)
        return m

    def lm_head(self, key):
        output = self._get(key)
        mc = self._mc
        tp = self._attn_tp * self._attn_cp
        padded_vocab = ((mc.vocab_size + tp - 1) // tp) * tp
        output_padded = pad_out_dim(output, padded_vocab, dim=0)
        output_t = output_padded.t()
        dtype = self._cpp_dtype()
        cfg = make_linear_config(input_dim=mc.hidden_units,
                                 output_dim=padded_vocab // tp,
                                 data_type=dtype)
        m = LinearBuilder(cfg, self._contexts, tp=tp, ranks=self._attn_ranks)
        m.set_weight(output_t, split_side=SplitSide.OUTPUT)
        return m

    def norm(self, key):
        w = self._get(key)
        cfg = make_norm_config(dim=self._mc.hidden_units, data_type=self._cpp_dtype())
        m = NormBuilder(cfg, self._contexts)
        m.set_weight(w)
        return m

    def _get(self, key: str) -> torch.Tensor | None:
        """Get a raw tensor from the checkpoint params."""
        return self.params.get(key)

    def _linear(self, pfx: str) -> Linear | None:
        """Read a Linear bundle from the checkpoint at *pfx*."""
        from .kind_map import build_linear
        return build_linear(self.params, pfx)

    def num_experts(self, layer: int) -> int:
        return 0
