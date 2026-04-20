# Copyright (c) OpenMMLab. All rights reserved.
"""TextModelSpec — per-architecture spec owning HF parsing and C++ configs."""
from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import torch

import _turbomind as _tm

from .builder import LinearBuilder, SplitSide, _cpp_dtype as _cd
from .builder import make_linear_config
from .config import AttentionConfig
from .linear import pad_out_dim
from .source_model.utils import (_pad_kv_head, detect_layer_prefix,
                                 parse_rope_param, rope_type_to_int)

if TYPE_CHECKING:
    from lmdeploy.messages import TurbomindEngineConfig


class TextModelSpec(ABC):
    """Text model spec: HF config -> C++ configs + weight commits.

    Subclass contract:
      - __init__ takes (hf_cfg, engine_cfg), calls super().__init__, then
        builds per-module C++ config templates as self._attn_cfg /
        self._ffn_cfg / self._moe_cfg / self._dn_cfg.
      - Factory method NAMES (attn/ffn/moe/linear_attn/mla/norm/...)
        are a convention for readability, NOT a protocol. Signatures may
        differ across subclasses; the base provides no stubs except the
        universal text-model primitives (token_embeds, lm_head).
    """

    # Class-level: checkpoint loader hints (HF key renaming + layer regex).
    _layer_pattern: str = ''
    _loader_mappings: list = []

    # If True, the subclass's __init__ has pinned _layer_prefix / _embed_key /
    # _norm_key to fixed values; set_params() will NOT re-detect from params.
    # Subclasses that need on-load detection (e.g. multimodal wrappers where
    # the decoder lives under model.language_model.*) leave this False and
    # let the base class re-detect when weights arrive.
    _pin_layer_prefix: bool = False

    # ------------------------------------------------------------------
    # Construction / parsing
    # ------------------------------------------------------------------

    def __init__(self, hf_cfg: dict, engine_cfg: 'TurbomindEngineConfig',
                 *, group_size: int = 0):
        """Parse HF config into orchestration scalars.

        ``group_size`` is the quantization group size the converter resolves
        from ``engine_cfg.model_format`` plus any user override. It lands on
        ``self._group_size`` so inter_size padding can use it during
        subclass ``__init__``. (It's not on ``TurbomindEngineConfig``
        today, so we take it as an explicit kwarg.)

        Subclasses override `_parse_base` (or extend in their own __init__)
        then construct C++ config templates and per-layer lists.
        """
        self.hf_cfg = hf_cfg
        self.engine_cfg = engine_cfg
        self._group_size = group_size
        self._parse_base(hf_cfg)

    def _parse_base(self, cfg: dict):
        """Fill canonical orchestration scalars from standard HF keys.

        Populated:
          _num_layer, _vocab_size, _norm_eps, _head_num, _kv_head_num,
          _kv_head_num_padded, _head_dim, _hidden_units, _rope,
          _max_position_embeddings, _tie_embeddings, _layer_prefix,
          _embed_key, _norm_key, _model_name, _tune_layer_num,
          _embedding_size.

        Subclass responsibilities (not set here):
          _softmax_scale (subclass default 0, MLA+YaRN overrides)
        """
        self._num_layer = cfg['num_hidden_layers']
        self._vocab_size = cfg['vocab_size']
        self._norm_eps = cfg['rms_norm_eps']
        self._tie_embeddings = cfg.get('tie_word_embeddings', False)
        self._model_name = cfg.get('model_type', '')
        self._tune_layer_num = 1
        self._embedding_size = self._vocab_size

        attn_head_num = cfg['num_attention_heads']
        hidden = cfg['hidden_size']
        head_dim = cfg.get('head_dim') or (hidden // attn_head_num)
        kv_head_num = cfg.get('num_key_value_heads', attn_head_num)
        self._hidden_units = hidden
        self._head_dim = head_dim
        self._head_num = attn_head_num
        self._kv_head_num = kv_head_num
        self._kv_head_num_padded = _pad_kv_head(
            kv_head_num, self.engine_cfg.attn_tp_size)

        self._rope, self._max_position_embeddings = parse_rope_param(
            cfg, head_dim)

        # Layer-prefix detection deferred until weights loaded; default now.
        # Subclasses that know their prefix unconditionally can override.
        self._layer_prefix, self._embed_key, self._norm_key = \
            detect_layer_prefix(None, cfg)

        # Default subclass can override (e.g. MLA+YaRN)
        self._softmax_scale = 0.0

    # ------------------------------------------------------------------
    # Runtime binding (called by TextModelLoader after model_comm exists)
    # ------------------------------------------------------------------

    def bind_runtime(self, *, contexts, root_handles, attn_ranks, mlp_ranks):
        self._contexts = contexts
        self._root_handles = root_handles
        self._attn_ranks = attn_ranks
        self._mlp_ranks = mlp_ranks

    def set_params(self, params: dict):
        self.params = params
        # Re-detect layer prefix from the actual checkpoint keys, unless the
        # subclass has pinned its prefix (class-level _pin_layer_prefix = True).
        if not self._pin_layer_prefix:
            self._layer_prefix, self._embed_key, self._norm_key = \
                detect_layer_prefix(params, self.hf_cfg)

    # ------------------------------------------------------------------
    # YAML export — produce AttentionConfig for C++ consumption
    # ------------------------------------------------------------------

    def to_attention_config(self) -> AttentionConfig:
        """Produce the AttentionConfig for YAML serialization."""
        return self._build_attention_config()

    def _build_attention_config(self) -> AttentionConfig:
        return AttentionConfig(
            rope_param=self._rope,
            max_position_embeddings=self._max_position_embeddings,
            softmax_scale=self._softmax_scale,
        )

    # ------------------------------------------------------------------
    # Checkpoint access helpers
    # ------------------------------------------------------------------

    def _get(self, key: str) -> torch.Tensor | None:
        return self.params.get(key)

    def _linear(self, pfx: str):
        from .kind_map import build_linear
        return build_linear(self.params, pfx)

    def _cpp_dtype(self):
        return _cd(self.engine_cfg.dtype)

    def _apply_rope(self, rope_cfg):
        """Copy self._rope fields into a C++ rope config object."""
        rope_cfg.type = rope_type_to_int(self._rope.type)
        rope_cfg.base = self._rope.base
        rope_cfg.dim  = self._rope.dim
        rope_cfg.factor = self._rope.factor
        rope_cfg.max_position_embeddings = self._max_position_embeddings
        if self._rope.type == 'yarn':
            rope_cfg.yarn_attention_factor = self._rope.attention_factor
            rope_cfg.yarn_beta_fast = self._rope.beta_fast
            rope_cfg.yarn_beta_slow = self._rope.beta_slow
        elif self._rope.type == 'llama3':
            rope_cfg.llama3_low_freq_factor = self._rope.low_freq_factor
            rope_cfg.llama3_high_freq_factor = self._rope.high_freq_factor
            rope_cfg.llama3_original_max_position_embeddings = self._rope.original_max_position_embeddings
        elif self._rope.type == 'mrope':
            rope_cfg.mrope_section = self._rope.mrope_section

    # ------------------------------------------------------------------
    # Text-model universals (default factory methods)
    # ------------------------------------------------------------------

    def token_embeds(self, key):
        emb = self._get(key)
        tp = self.engine_cfg.attn_tp_size * self.engine_cfg.attn_cp_size
        padded_vocab = ((self._vocab_size + tp - 1) // tp) * tp
        emb_padded = pad_out_dim(emb, padded_vocab, dim=0)
        dtype = self._cpp_dtype()
        cfg = make_linear_config(input_dim=padded_vocab,
                                 output_dim=self._hidden_units // tp,
                                 data_type=dtype)
        m = LinearBuilder(cfg, self._contexts, tp=tp, ranks=self._attn_ranks)
        m.set_weight(emb_padded, split_side=SplitSide.OUTPUT)
        return m

    def lm_head(self, key):
        output = self._get(key)
        tp = self.engine_cfg.attn_tp_size * self.engine_cfg.attn_cp_size
        padded_vocab = ((self._vocab_size + tp - 1) // tp) * tp
        output_padded = pad_out_dim(output, padded_vocab, dim=0)
        output_t = output_padded.t()
        dtype = self._cpp_dtype()
        cfg = make_linear_config(input_dim=self._hidden_units,
                                 output_dim=padded_vocab // tp,
                                 data_type=dtype)
        m = LinearBuilder(cfg, self._contexts, tp=tp, ranks=self._attn_ranks)
        m.set_weight(output_t, split_side=SplitSide.OUTPUT)
        return m
