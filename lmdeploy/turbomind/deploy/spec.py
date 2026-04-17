# Copyright (c) OpenMMLab. All rights reserved.
"""TextModelSpec — per-architecture spec owning HF parsing and C++ configs."""
from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import torch

import _turbomind as _tm

from .builder import LinearBuilder, NormBuilder, SplitSide, _cpp_dtype as _cd
from .builder import make_linear_config, make_norm_config
from .config import (AttentionConfig, LoraConfig, ModelConfig,
                     TurbomindModelConfig)
from .linear import Linear, pad_out_dim
from .source_model.utils import (_pad_inter_size, _pad_kv_head,
                                 detect_layer_prefix, parse_rope_param)

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
    # YAML export — mechanical copy from C++ configs + scalars
    # ------------------------------------------------------------------

    def to_legacy_config(self) -> TurbomindModelConfig:
        """Produce the narrowed TurbomindModelConfig for YAML serialization."""
        mc = ModelConfig()
        self._copy_template_fields(mc)
        self._copy_orchestration_fields(mc)
        self._copy_perlayer_fields(mc)
        ac = self._build_attention_config()
        return TurbomindModelConfig(model_config=mc, attention_config=ac,
                                    lora_config=LoraConfig())

    def _copy_template_fields(self, mc: ModelConfig):
        """Copy fields from C++ config templates onto ModelConfig.

        Handles standard _attn_cfg (+ MLA fields), _ffn_cfg, and optional
        _dn_cfg. Subclasses may extend to copy extra templates.
        """
        a = self._attn_cfg
        mc.hidden_units     = a.hidden_dim
        mc.head_num         = a.head_num
        mc.kv_head_num      = a.kv_head_num
        mc.size_per_head    = a.head_dim
        mc.q_lora_rank      = a.q_lora_rank
        mc.kv_lora_rank     = a.kv_lora_rank
        mc.qk_rope_dim      = a.qk_rope_dim
        mc.v_head_dim       = a.v_head_dim
        mc.attn_bias        = int(a.has_bias)
        mc.qk_norm          = a.qk_norm
        mc.attn_sink        = a.attn_sink
        mc.attn_output_gate = a.attn_output_gate

        f = self._ffn_cfg
        mc.mlp_bias        = f.has_bias
        mc.activation_type = _act_type_str(f.act_type)

        if hasattr(self, '_dn_cfg'):
            dn = self._dn_cfg
            mc.linear_num_key_heads    = dn.num_k_heads
            mc.linear_num_value_heads  = dn.num_v_heads
            mc.linear_key_head_dim     = dn.key_head_dim
            mc.linear_value_head_dim   = dn.value_head_dim
            mc.linear_conv_kernel_dim  = dn.d_conv

    def _copy_orchestration_fields(self, mc: ModelConfig):
        """Copy orchestration scalars (not in any C++ config) onto ModelConfig."""
        mc.num_layer      = self._num_layer
        mc.vocab_size     = self._vocab_size
        mc.embedding_size = self._embedding_size
        mc.norm_eps       = self._norm_eps
        mc.tune_layer_num = self._tune_layer_num
        mc.model_name     = self._model_name
        mc.data_type      = self.engine_cfg.dtype
        mc.session_len    = self.engine_cfg.session_len
        mc.group_size     = self._group_size
        mc.attn_tp_size   = self.engine_cfg.attn_tp_size
        mc.attn_cp_size   = self.engine_cfg.attn_cp_size
        mc.mlp_tp_size    = self.engine_cfg.mlp_tp_size
        mc.model_format   = self.engine_cfg.model_format

    def _copy_perlayer_fields(self, mc: ModelConfig):
        """Copy per-layer lists. Default covers only inter_size.

        Subclasses override to emit window_size / layer_types / etc.
        Every subclass MUST set self._inter_sizes_padded during __init__;
        we use direct attribute access so a missing assignment fails fast
        with AttributeError at to_legacy_config() time rather than silently
        emitting inter_size=[].
        """
        mc.inter_size = self._inter_sizes_padded

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


# ----------------------------------------------------------------------
# Helpers used by _copy_template_fields
# ----------------------------------------------------------------------

_ACT_ID_TO_STR = {0: 'silu', 1: 'gpt-oss'}


def _act_type_str(act_type: int) -> str:
    return _ACT_ID_TO_STR.get(act_type, 'silu')
