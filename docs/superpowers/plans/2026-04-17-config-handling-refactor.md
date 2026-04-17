# TurboMind Deploy Config Handling Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse `BaseInputModel` + `TextModelSpec` into one class per arch; store C++ module configs directly on the spec as the canonical parsed state; delete `make_*_config(mc, ...)` adapter factories; narrow `ModelConfig` to the YAML wire format.

**Architecture:** Spec parses HF config once in `__init__`, building `_tm.AttentionConfig`/`_tm.FfnConfig`/`_tm.MoeConfig`/`_tm.DeltaNetConfig` templates as `self._*_cfg` attributes. Per-layer factory methods `clone()` a template and set ≤3 per-layer fields, then pass to the matching Builder. `to_legacy_config()` mechanically copies fields from the C++ configs into the narrowed `ModelConfig` for YAML serialization to `turbomind.cc`.

**Tech Stack:** Python 3.10+, `_turbomind` pybind11 module, `pydantic.dataclasses`, `mmengine.Registry`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-17-config-handling-refactor-design.md`

---

## Execution strategy

This plan is a **single atomic refactor** on a feature branch. Intermediate commits may leave the branch in a non-runnable state; the final task (Task 14) runs the full test matrix as the merge gate. Commit at each task boundary so code review can follow the logical progression.

Phase ordering:

- **Phase 1** (Tasks 1–2): Additive infrastructure — shared helpers, narrowed `ModelConfig`. Does not break the old flow.
- **Phase 2** (Task 3): New `TextModelSpec` base. Breaking — existing subclasses stop working until migrated.
- **Phase 3** (Tasks 4–7): Migrate all four specs one at a time.
- **Phase 4** (Tasks 8–12): Remove old machinery — `BaseInputModel`, `make_*_config` factories, update orchestration.
- **Phase 5** (Tasks 13–14): Run the validation matrix.

---

## Phase 1 — Additive infrastructure

### Task 1: Add shared parsing/padding helpers to `source_model/utils.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/utils.py`

- [ ] **Step 1: Append three new helpers to the end of `utils.py`.**

```python
# --- TP padding helpers (moved from target_model/base.py) -----------------

def _pad_inter_size(inter_size: int, group_size: int, tp: int) -> int:
    """Pad inter_size so it is divisible by group_size * tp.

    Moved from target_model/base.py where it lived as a module-level helper
    inside finalize_config. Same formula.
    """
    group_size = max(1, group_size)
    group_num = (inter_size + group_size - 1) // group_size
    groups_per_rank = (group_num + tp - 1) // tp
    inter_size_padded = groups_per_rank * group_size * tp
    return inter_size_padded


def _pad_kv_head(kv_head_num: int, attn_tp: int) -> int:
    """Pad kv_head_num up to attn_tp when attn_tp is a multiple of kv_head_num.

    Matches the rule in finalize_config:
      if attn_tp > kv_head_num and attn_tp % kv_head_num == 0:
          kv_head_num = attn_tp
    """
    if attn_tp > kv_head_num and attn_tp % kv_head_num == 0:
        return attn_tp
    return kv_head_num


# --- Layer-prefix detection ------------------------------------------------

def detect_layer_prefix(params: dict | None, cfg: dict) -> tuple[str, str, str]:
    """Return (layer_prefix, embed_key, norm_key) for a HF checkpoint.

    Models that wrap the decoder in a ``language_model`` submodule (Molmo,
    some multimodal variants, Qwen3.5 when packaged as a multimodal root)
    store weights under ``model.language_model.*``. Plain decoder models
    use ``model.*``.

    If ``params`` is None (spec hasn't loaded weights yet), fall back to the
    standard ``model.*`` layout. Specs that need early disambiguation can
    override this during their own parsing.
    """
    if params is not None and any(
            k.startswith('model.language_model.') for k in params):
        return ('model.language_model.layers',
                'model.language_model.embed_tokens.weight',
                'model.language_model.norm.weight')
    return ('model.layers',
            'model.embed_tokens.weight',
            'model.norm.weight')
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/utils.py
git commit -m "deploy: add pad/prefix helpers to source_model.utils"
```

---

### Task 2: Narrow `ModelConfig` — remove 12 dead MoE fields

**Files:**
- Modify: `lmdeploy/turbomind/deploy/config.py`

- [ ] **Step 1: Delete the 12 MoE fields from `ModelConfig`.**

In `lmdeploy/turbomind/deploy/config.py`, inside the `@dataclass class ModelConfig:` block, remove these lines:

```python
expert_num: list[int] = field(default_factory=list)
expert_router_bias: bool = False
expert_inter_size: int = 0
experts_per_token: int = 0
moe_shared_gate: bool = False
norm_topk_prob: bool = False
routed_scale: float = 1.0
topk_group: int = 1
topk_method: str = 'greedy'
moe_group_num: int = 1
scoring_func: str = 'softmax'
router_n_groups: int = -1
```

Keep everything else intact (including MLA fields, linear-attn fields, window_size, etc.).

- [ ] **Step 2: Verify the narrowed dataclass parses.**

Run:

```bash
python -c "from lmdeploy.turbomind.deploy.config import ModelConfig, TurbomindModelConfig; print(TurbomindModelConfig.from_dict())"
```

Expected: prints a TurbomindModelConfig with MoE fields absent from the model_config section. No exception.

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/config.py
git commit -m "deploy: narrow ModelConfig by removing 12 dead MoE fields

These fields are never read by turbomind.cc from the YAML. They exist only
because make_moe_config(mc, ...) reads them to populate _tm.MoeConfig. After
the spec refactor, MoeConfig is built directly from parsed HF state.
"
```

---

## Phase 2 — New `TextModelSpec` base

### Task 3: Replace `spec.py` with the new `TextModelSpec` base class

**Files:**
- Modify: `lmdeploy/turbomind/deploy/spec.py`

After this task, existing spec subclasses will be broken until they're migrated in Phase 3. That's expected; this is a feature-branch refactor.

- [ ] **Step 1: Replace the entire body of `spec.py` with the new base class.**

```python
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
        # Redetect layer prefix now that we have weights; spec may override.
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
        """
        mc.inter_size = getattr(self, '_inter_sizes_padded', [])

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
```

- [ ] **Step 2: Sanity-import to catch typos.**

```bash
python -c "from lmdeploy.turbomind.deploy.spec import TextModelSpec; print(TextModelSpec)"
```

Expected: prints the class. No exception. (Subclasses are broken at this point — that's fine.)

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/spec.py
git commit -m "deploy: rewrite TextModelSpec as HF->C++config owning class

Subclasses will be broken until migrated in follow-up commits."
```

---

## Phase 3 — Spec migrations

Each spec migration task produces a self-contained rewritten spec file. The pattern is the same across all four; the differences are the arch-specific details.

Common pattern reminders (see spec doc §"C++ configs as canonical parsed state"):

1. `__init__(self, hf_cfg, engine_cfg)` calls `super().__init__(hf_cfg, engine_cfg)`, then builds `self._attn_cfg`, `self._ffn_cfg`, optional `self._moe_cfg`, `self._dn_cfg`, plus per-layer lists `self._inter_sizes_padded`, `self._window_sizes`, `self._expert_nums`, etc.
2. Register with `@INPUT_MODELS.register_module(name='...')` directly on the spec class.
3. Factory methods `attn`, `ffn`, `moe`, `linear_attn` clone the template and set per-layer fields.
4. `model()` walks the hierarchy exactly as in the existing file (same structure, no topology changes in this refactor).

**Do NOT change `model()` topology, builder pipelines, or weight transformations.** This refactor is config-plumbing only. Weight pipelines (`dequant_mixed`, `pad_for_tp`, `fuse_qkv`, etc.) and `add_qkv_proj`, `add_o_proj`, etc. calls stay exactly as they are today.

### Task 4: Migrate `qwen3_spec.py` (pilot)

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py`

This is the pilot — validate the pattern before touching others.

- [ ] **Step 1: Replace the file contents** with the version below. It preserves every behavior of today's `Qwen3TextSpec` + `Qwen3InputModel`, just restructured onto the new base.

```python
# Copyright (c) OpenMMLab. All rights reserved.
"""Qwen3 TextModelSpec for the new pipeline.

Qwen3 is a standard Llama-like model with QK norm and optional MoE.
No shared expert in the MoE variant, no linear attention, no zero-centered
norm.
"""
from __future__ import annotations

import _turbomind as _tm

from ..builder import (AttentionBuilder, DecoderLayerBuilder, FfnBuilder,
                       MoeBuilder, ModuleListBuilder, TextModelBuilder,
                       _act_type_id)
from ..builder import DecoderLayerConfig, ModuleListConfig
from ..linear import Linear
from ..spec import TextModelSpec
from .base import INPUT_MODELS
from .utils import (_pad_inter_size, reorder_rotary_emb,
                    reorder_rotary_emb_linear)

_LAYER_PATTERN = r'model\.layers\.([0-9]+).'


@INPUT_MODELS.register_module(name='qwen3-moe')
@INPUT_MODELS.register_module(name='qwen3')
class Qwen3TextSpec(TextModelSpec):
    """Weight spec for Qwen3 (dense) and Qwen3-MoE."""

    _layer_pattern = _LAYER_PATTERN
    # Qwen3 always uses the plain `model.*` layout — pin to skip on-load
    # re-detection by TextModelSpec.set_params.
    _pin_layer_prefix = True

    def __init__(self, hf_cfg: dict, engine_cfg, *, group_size: int = 0):
        super().__init__(hf_cfg, engine_cfg, group_size=group_size)

        # Fixed layer prefix for Qwen3
        self._layer_prefix = 'model.layers'
        self._embed_key = 'model.embed_tokens.weight'
        self._norm_key = 'model.norm.weight'

        self._n_experts = hf_cfg.get('num_experts', 0)

        # ---- Attention template ----
        dtype = self._cpp_dtype()
        self._attn_cfg = _tm.AttentionConfig()
        self._attn_cfg.hidden_dim  = self._hidden_units
        self._attn_cfg.head_dim    = self._head_dim
        self._attn_cfg.head_num    = self._head_num
        self._attn_cfg.kv_head_num = self._kv_head_num_padded
        self._attn_cfg.has_bias    = hf_cfg.get('attention_bias', 0)
        self._attn_cfg.qk_norm     = True
        self._attn_cfg.rope_dim    = self._rope.dim
        self._attn_cfg.window_size = 0
        self._attn_cfg.tp_size     = engine_cfg.attn_tp_size
        self._attn_cfg.data_type   = dtype

        # ---- FFN template ----
        self._ffn_cfg = _tm.FfnConfig()
        self._ffn_cfg.hidden_dim = self._hidden_units
        self._ffn_cfg.has_bias   = False
        self._ffn_cfg.tp_size    = engine_cfg.mlp_tp_size
        self._ffn_cfg.data_type  = dtype
        self._ffn_cfg.act_type   = _act_type_id('silu')
        # fuse_silu / fused_moe / inter_size set per-call in ffn()/moe()

        # ---- MoE template (only if MoE variant) ----
        if self._n_experts > 0:
            self._moe_cfg = _tm.MoeConfig()
            self._moe_cfg.method            = 1  # kFused
            self._moe_cfg.experts_per_token = hf_cfg.get('num_experts_per_tok', 8)
            self._moe_cfg.norm_topk_prob    = hf_cfg.get('norm_topk_prob', False)
            self._moe_cfg.shared_gate       = False
            self._moe_cfg.routed_scale      = 1.0
            self._moe_cfg.router_bias       = False
            self._moe_cfg.topk_group        = 1
            self._moe_cfg.topk_method       = 'greedy'
            self._moe_cfg.n_group           = 1
            self._moe_cfg.scoring_func      = 'softmax'
            self._moe_cfg.router_n_groups   = 0
            self._moe_cfg.hidden_dim        = self._hidden_units
            self._moe_cfg.mlp_bias          = False
            self._moe_cfg.data_type         = dtype
            self._moe_cfg.tp_size           = engine_cfg.mlp_tp_size
            self._moe_cfg.act_type          = _act_type_id('silu')
            self._moe_cfg.fuse_silu         = True

            self._expert_inter_size_padded = _pad_inter_size(
                hf_cfg.get('moe_intermediate_size', 768),
                self._group_size, engine_cfg.mlp_tp_size)
        else:
            self._expert_inter_size_padded = 0

        # ---- Per-layer inter_size (dense FFN) ----
        raw_inter = hf_cfg.get('intermediate_size', 0) if self._n_experts == 0 else 0
        self._inter_sizes_padded = [
            _pad_inter_size(raw_inter, self._group_size,
                            engine_cfg.mlp_tp_size)
            for _ in range(self._num_layer)
        ]
        self._expert_nums = (
            [self._n_experts] * self._num_layer if self._n_experts > 0 else []
        )

    # ------------------------------------------------------------------
    # model() — walks full hierarchy (same as existing code)
    # ------------------------------------------------------------------

    def model(self):
        root = TextModelBuilder(self._root_handles, self._contexts)
        root.tok_embeddings = self.token_embeds(self._embed_key)
        root.norm = self.output_norm(self._norm_key)
        lm_key = self._embed_key if self._tie_embeddings else 'lm_head.weight'
        root.output = self.lm_head(lm_key)
        root.layers = self.layers(self._layer_prefix)

    # ------------------------------------------------------------------
    # Norm variants (Qwen3 uses standard RMSNorm, no zero-centering)
    # ------------------------------------------------------------------

    def output_norm(self, key):
        from ..builder import NormBuilder, make_norm_config
        w = self._get(key)
        cfg = make_norm_config(dim=self._hidden_units, data_type=self._cpp_dtype())
        m = NormBuilder(cfg, self._contexts)
        m.set_weight(w)
        return m

    def norm(self, key):
        return self.output_norm(key)

    # ------------------------------------------------------------------
    # Attention / FFN / MoE factories
    # ------------------------------------------------------------------

    def attn(self, pfx, layer):
        q = self._linear(f'{pfx}.q_proj')
        k = self._linear(f'{pfx}.k_proj')
        v = self._linear(f'{pfx}.v_proj')
        o = self._linear(f'{pfx}.o_proj')

        q = reorder_rotary_emb_linear(q, self._head_dim, self._rope.dim)
        k = reorder_rotary_emb_linear(k, self._head_dim, self._rope.dim)

        cfg = self._attn_cfg.clone()
        # No per-layer attention fields for Qwen3 (no sliding window).
        attn = AttentionBuilder(cfg, self._contexts,
                                tp=self.engine_cfg.attn_tp_size,
                                ranks=self._attn_ranks)

        attn.add_qkv_proj(q, k, v)
        attn.add_o_proj(o)

        q_norm = self._get(f'{pfx}.q_norm.weight')
        k_norm = self._get(f'{pfx}.k_norm.weight')
        q_norm = reorder_rotary_emb(q_norm, self._head_dim, self._rope.dim)
        k_norm = reorder_rotary_emb(k_norm, self._head_dim, self._rope.dim)
        attn.add_qk_norm(q_norm, k_norm)

        return attn

    def ffn(self, pfx, layer, inter_size=None, fused_moe=False):
        w1 = self._linear(f'{pfx}.gate_proj')
        w3 = self._linear(f'{pfx}.up_proj')
        w2 = self._linear(f'{pfx}.down_proj')

        cfg = self._ffn_cfg.clone()
        cfg.inter_size = (inter_size if inter_size is not None
                          else self._inter_sizes_padded[layer])
        cfg.fuse_silu  = False
        cfg.fused_moe  = fused_moe

        m = FfnBuilder(cfg, self._contexts,
                       tp=self.engine_cfg.mlp_tp_size,
                       ranks=self._mlp_ranks)
        m.add_ffn(w1, w2, w3)
        return m

    def moe(self, pfx, layer):
        if self.num_experts(layer) <= 0:
            return None

        cfg = self._moe_cfg.clone()
        cfg.layer_id   = layer
        cfg.expert_num = self._expert_nums[layer]
        cfg.inter_size = self._expert_inter_size_padded

        m = MoeBuilder(cfg, self._contexts,
                       tp=self.engine_cfg.mlp_tp_size,
                       ranks=self._mlp_ranks)

        gate_w = self._get(f'{pfx}.gate.weight')
        gate_w = gate_w.t() if gate_w.dim() > 1 else gate_w
        m.add_gate('gate', Linear({'weight': gate_w}),
                   model_dtype=self._cpp_dtype())

        experts = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for e in range(self.num_experts(layer)):
            experts[str(e)] = self.ffn(
                f'{pfx}.experts.{e}', layer,
                inter_size=self._expert_inter_size_padded, fused_moe=True)
        m.experts = experts
        return m

    def layers(self, pfx):
        layers = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for i in range(self._num_layer):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._contexts)
            d.attention_norm = self.norm(
                f'{pfx}.{i}.input_layernorm.weight')
            d.attention = self.attn(f'{pfx}.{i}.self_attn', i)
            d.ffn_norm = self.norm(
                f'{pfx}.{i}.post_attention_layernorm.weight')
            if self.num_experts(i) > 0:
                d.moe_ffn = self.moe(f'{pfx}.{i}.mlp', i)
            else:
                d.feed_forward = self.ffn(f'{pfx}.{i}.mlp', i)
            layers[str(i)] = d
        return layers

    def num_experts(self, layer: int) -> int:
        return self._n_experts
```

- [ ] **Step 2: Verify the spec class imports.**

```bash
python -c "from lmdeploy.turbomind.deploy.source_model.qwen3_spec import Qwen3TextSpec; print(Qwen3TextSpec._layer_pattern)"
```

Expected: prints `'model\\.layers\\.([0-9]+).'`. No exception.

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/qwen3_spec.py
git commit -m "deploy: migrate qwen3_spec to new TextModelSpec base (pilot)"
```

---

### Task 5: Migrate `qwen3_5_spec.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py`

Qwen3.5 exercises the full complexity: dense + MoE variants, linear attention via DeltaNet, zero-centered RMSNorm, partial rotary factor, optional shared expert in MoE.

- [ ] **Step 1: Replace the file contents.**

```python
# Copyright (c) OpenMMLab. All rights reserved.
"""Qwen3.5 TextModelSpec for the new pipeline."""
from __future__ import annotations

import re

import torch

import _turbomind as _tm

from ..builder import (AttentionBuilder, DecoderLayerBuilder, DeltaNetBuilder,
                       FfnBuilder, MoeBuilder, ModuleListBuilder,
                       TextModelBuilder, _act_type_id)
from ..builder import DecoderLayerConfig, ModuleListConfig
from ..kind_map import build_linear
from ..linear import Linear
from ..spec import TextModelSpec
from .base import INPUT_MODELS
from .utils import (_pad_inter_size, detect_layer_prefix,
                    reorder_rotary_emb, reorder_rotary_emb_linear)

_LAYER_PATTERN = r'(?:model\.language_model\.|model\.)layers\.([0-9]+)\.'


def map_packed_qwen35_experts(name: str) -> str:
    """Map packed expert names to weight names so parameter.py can classify."""
    return re.sub(r'(mlp\.experts\.(?:gate_up|down)_proj)$', r'\1.weight', name)


@INPUT_MODELS.register_module(name='qwen3_5-moe')
@INPUT_MODELS.register_module(name='qwen3_5')
class Qwen3_5Spec(TextModelSpec):
    """Weight spec for Qwen3.5 (dense + linear-attn + optional MoE)."""

    _layer_pattern = _LAYER_PATTERN
    _loader_mappings = [map_packed_qwen35_experts]

    def __init__(self, hf_cfg: dict, engine_cfg, *, group_size: int = 0):
        super().__init__(hf_cfg, engine_cfg, group_size=group_size)

        # partial_rotary_factor adjusts rope_dim
        rope_params = hf_cfg.get('rope_parameters', {})
        partial_factor = rope_params.get(
            'partial_rotary_factor', hf_cfg.get('partial_rotary_factor', 1.0))
        if partial_factor < 1.0:
            self._rope.dim = int(self._head_dim * partial_factor)

        self._layer_types = hf_cfg.get('layer_types', [])
        self._n_experts = hf_cfg.get('num_experts', 0)
        dtype = self._cpp_dtype()

        # ---- Attention template ----
        self._attn_cfg = _tm.AttentionConfig()
        self._attn_cfg.hidden_dim       = self._hidden_units
        self._attn_cfg.head_dim         = self._head_dim
        self._attn_cfg.head_num         = self._head_num
        self._attn_cfg.kv_head_num      = self._kv_head_num_padded
        self._attn_cfg.has_bias         = hf_cfg.get('attention_bias', 0)
        self._attn_cfg.qk_norm          = True
        self._attn_cfg.attn_output_gate = bool(self._layer_types) and \
                                          hf_cfg.get('attn_output_gate', False)
        self._attn_cfg.rope_dim         = self._rope.dim
        self._attn_cfg.window_size      = 0
        self._attn_cfg.tp_size          = engine_cfg.attn_tp_size
        self._attn_cfg.data_type        = dtype

        # ---- DeltaNet template (only if linear-attn layers present) ----
        if self._layer_types:
            self._dn_cfg = _tm.DeltaNetConfig()
            self._dn_cfg.hidden_dim      = self._hidden_units
            self._dn_cfg.num_k_heads     = hf_cfg['linear_num_key_heads']
            self._dn_cfg.num_v_heads     = hf_cfg['linear_num_value_heads']
            self._dn_cfg.key_head_dim    = hf_cfg['linear_key_head_dim']
            self._dn_cfg.value_head_dim  = hf_cfg['linear_value_head_dim']
            self._dn_cfg.d_conv          = hf_cfg.get('linear_conv_kernel_dim', 0) or 4
            self._dn_cfg.has_bias        = bool(self._attn_cfg.has_bias)
            self._dn_cfg.tp_size         = engine_cfg.attn_tp_size
            self._dn_cfg.data_type       = dtype

            ln_key_heads = hf_cfg['linear_num_key_heads']
            ln_val_heads = hf_cfg['linear_num_value_heads']
            ln_key_dim   = hf_cfg['linear_key_head_dim']
            ln_val_dim   = hf_cfg['linear_value_head_dim']
            q_dim = ln_key_heads * ln_key_dim
            k_dim = ln_key_heads * ln_key_dim
            v_dim = ln_val_heads * ln_val_dim
            self._linear_qkv_split = (q_dim, k_dim, v_dim)

        # ---- FFN template ----
        self._ffn_cfg = _tm.FfnConfig()
        self._ffn_cfg.hidden_dim = self._hidden_units
        self._ffn_cfg.has_bias   = False
        self._ffn_cfg.tp_size    = engine_cfg.mlp_tp_size
        self._ffn_cfg.data_type  = dtype
        self._ffn_cfg.act_type   = _act_type_id('silu')

        # ---- MoE template ----
        if self._n_experts > 0:
            self._moe_cfg = _tm.MoeConfig()
            self._moe_cfg.method            = 1
            self._moe_cfg.experts_per_token = hf_cfg['num_experts_per_tok']
            self._moe_cfg.norm_topk_prob    = True
            self._moe_cfg.shared_gate       = True
            self._moe_cfg.routed_scale      = 1.0
            self._moe_cfg.router_bias       = False
            self._moe_cfg.topk_group        = 1
            self._moe_cfg.topk_method       = 'greedy'
            self._moe_cfg.n_group           = 1
            self._moe_cfg.scoring_func      = 'softmax'
            self._moe_cfg.router_n_groups   = 0
            self._moe_cfg.hidden_dim        = self._hidden_units
            self._moe_cfg.mlp_bias          = False
            self._moe_cfg.data_type         = dtype
            self._moe_cfg.tp_size           = engine_cfg.mlp_tp_size
            self._moe_cfg.act_type          = _act_type_id('silu')
            self._moe_cfg.fuse_silu         = True

            self._expert_inter_size_padded = _pad_inter_size(
                hf_cfg['moe_intermediate_size'], self._group_size,
                engine_cfg.mlp_tp_size)
            # Shared-expert inter_size = intermediate_size in MoE layers
            raw_shared = hf_cfg.get('shared_expert_intermediate_size', 0)
        else:
            self._expert_inter_size_padded = 0
            raw_shared = hf_cfg.get('intermediate_size', 0)

        self._inter_sizes_padded = [
            _pad_inter_size(raw_shared, self._group_size,
                            engine_cfg.mlp_tp_size)
            for _ in range(self._num_layer)
        ]
        self._expert_nums = (
            [self._n_experts] * self._num_layer if self._n_experts > 0 else []
        )

    def _is_linear_attn(self, layer: int) -> bool:
        return (layer < len(self._layer_types)
                and self._layer_types[layer] == 'linear_attention')

    def num_experts(self, layer: int) -> int:
        return self._n_experts

    # ------------------------------------------------------------------
    # Per-layer fields override (add layer_types)
    # ------------------------------------------------------------------

    def _copy_perlayer_fields(self, mc):
        super()._copy_perlayer_fields(mc)
        mc.layer_types = self._layer_types

    # ------------------------------------------------------------------
    # After params loaded: re-detect layer prefix for language_model wrapping
    # ------------------------------------------------------------------

    def set_params(self, params: dict):
        super().set_params(params)
        # detect_layer_prefix handles the model.language_model.* wrapping

    # ------------------------------------------------------------------
    # model() — same topology as old code
    # ------------------------------------------------------------------

    def model(self):
        root = TextModelBuilder(self._root_handles, self._contexts)
        root.tok_embeddings = self.token_embeds(self._embed_key)
        root.norm = self.output_norm(self._norm_key)
        lm_key = self._embed_key if self._tie_embeddings else 'lm_head.weight'
        root.output = self.lm_head(lm_key)
        root.layers = self.layers(self._layer_prefix)

    # ------------------------------------------------------------------
    # Zero-centered norm
    # ------------------------------------------------------------------

    def output_norm(self, key):
        from ..builder import NormBuilder, make_norm_config
        w = self._zero_centered(self._get(key))
        cfg = make_norm_config(dim=self._hidden_units, data_type=self._cpp_dtype())
        m = NormBuilder(cfg, self._contexts)
        m.set_weight(w)
        return m

    def norm(self, key):
        return self.output_norm(key)

    def _zero_centered(self, w):
        if w is not None:
            return w.float() + 1.0
        return None

    # ------------------------------------------------------------------
    # Attention / linear-attention factories
    # ------------------------------------------------------------------

    def attn(self, pfx, layer):
        q = self._linear(f'{pfx}.q_proj')
        k = self._linear(f'{pfx}.k_proj')
        v = self._linear(f'{pfx}.v_proj')
        o = self._linear(f'{pfx}.o_proj')

        q = reorder_rotary_emb_linear(q, self._head_dim, self._rope.dim)
        k = reorder_rotary_emb_linear(k, self._head_dim, self._rope.dim)

        cfg = self._attn_cfg.clone()
        attn = AttentionBuilder(cfg, self._contexts,
                                tp=self.engine_cfg.attn_tp_size,
                                ranks=self._attn_ranks)
        attn.add_qkv_proj(q, k, v)
        attn.add_o_proj(o)

        q_norm = self._zero_centered(self._get(f'{pfx}.q_norm.weight'))
        k_norm = self._zero_centered(self._get(f'{pfx}.k_norm.weight'))
        q_norm = reorder_rotary_emb(q_norm, self._head_dim, self._rope.dim)
        k_norm = reorder_rotary_emb(k_norm, self._head_dim, self._rope.dim)
        attn.add_qk_norm(q_norm, k_norm)
        return attn

    def linear_attn(self, pfx, layer):
        cfg = self._dn_cfg.clone()
        builder = DeltaNetBuilder(cfg, self._contexts,
                                  tp=self.engine_cfg.attn_tp_size,
                                  ranks=self._attn_ranks)

        builder.add_input_projections(
            in_proj_qkv=self._linear(f'{pfx}.in_proj_qkv'),
            in_proj_z=self._linear(f'{pfx}.in_proj_z'),
            in_proj_b=self._linear(f'{pfx}.in_proj_b'),
            in_proj_a=self._linear(f'{pfx}.in_proj_a'),
            out_proj=self._linear(f'{pfx}.out_proj'),
            qkv_split=self._linear_qkv_split)
        builder.add_scalar_params(
            a_log=self._get(f'{pfx}.A_log'),
            dt_bias=self._get(f'{pfx}.dt_bias'))
        builder.add_conv1d(
            self._get(f'{pfx}.conv1d.weight'),
            qkv_split=self._linear_qkv_split)
        builder.add_norm(
            self._get(f'{pfx}.norm.weight'), data_type=self._cpp_dtype())
        return builder

    # ------------------------------------------------------------------
    # FFN / MoE factories
    # ------------------------------------------------------------------

    def ffn(self, pfx, layer, inter_size=None, fused_moe=False):
        w1 = self._linear(f'{pfx}.gate_proj')
        w3 = self._linear(f'{pfx}.up_proj')
        w2 = self._linear(f'{pfx}.down_proj')
        if w1 is None and w2 is None and w3 is None:
            return None

        cfg = self._ffn_cfg.clone()
        cfg.inter_size = (inter_size if inter_size is not None
                          else self._inter_sizes_padded[layer])
        cfg.fuse_silu  = False
        cfg.fused_moe  = fused_moe

        m = FfnBuilder(cfg, self._contexts,
                       tp=self.engine_cfg.mlp_tp_size,
                       ranks=self._mlp_ranks)
        m.add_ffn(w1, w2, w3)
        return m

    def moe(self, pfx, layer):
        if self.num_experts(layer) <= 0:
            return None

        cfg = self._moe_cfg.clone()
        cfg.layer_id   = layer
        cfg.expert_num = self._expert_nums[layer]
        cfg.inter_size = self._expert_inter_size_padded

        m = MoeBuilder(cfg, self._contexts,
                       tp=self.engine_cfg.mlp_tp_size,
                       ranks=self._mlp_ranks)

        dtype = self._cpp_dtype()
        gate_w = self._get(f'{pfx}.gate.weight')
        gate_w = gate_w.t() if gate_w.dim() > 1 else gate_w
        m.add_gate('gate', Linear({'weight': gate_w}), model_dtype=dtype)

        sg = self._get(f'{pfx}.shared_expert_gate.weight')
        sg = sg.t() if sg.dim() > 1 else sg
        m.add_gate('shared_gate', Linear({'weight': sg}), model_dtype=dtype)

        experts = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for e in range(self.num_experts(layer)):
            experts[str(e)] = self._moe_expert_ffn(
                pfx, layer, e, self._expert_inter_size_padded)

        m.experts = experts
        return m

    def _moe_expert_ffn(self, pfx, layer, expert_idx, inter_size):
        expert_pfx = f'{pfx}.experts.{expert_idx}'
        result = self.ffn(expert_pfx, layer,
                          inter_size=inter_size, fused_moe=True)
        if result is not None:
            return result
        packed_pfx = f'{pfx}.experts'
        return self._packed_moe_expert_indexed(packed_pfx, expert_idx, inter_size)

    def _packed_moe_expert_indexed(self, pfx, expert_idx, inter_size):
        gate_up_lin = build_linear(self.params, f'{pfx}.gate_up_proj',
                                   index=expert_idx)
        down_lin = build_linear(self.params, f'{pfx}.down_proj',
                                index=expert_idx)
        if gate_up_lin is None or down_lin is None:
            return None

        gate_tensors: dict[str, torch.Tensor] = {}
        up_tensors: dict[str, torch.Tensor] = {}
        for kind, t in gate_up_lin.tensors.items():
            half = t.shape[-1] // 2
            gate_tensors[kind] = t[..., :half].contiguous()
            up_tensors[kind] = t[..., half:].contiguous()

        cfg = self._ffn_cfg.clone()
        cfg.inter_size = inter_size
        cfg.fuse_silu  = False
        cfg.fused_moe  = True

        m = FfnBuilder(cfg, self._contexts,
                       tp=self.engine_cfg.mlp_tp_size,
                       ranks=self._mlp_ranks)
        w1 = Linear(tensors=gate_tensors, weight_format=gate_up_lin.weight_format)
        w3 = Linear(tensors=up_tensors,   weight_format=gate_up_lin.weight_format)
        m.add_ffn(w1, down_lin, w3)
        return m

    # ------------------------------------------------------------------
    # layers() — dispatch by layer type
    # ------------------------------------------------------------------

    def layers(self, pfx):
        layers = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for i in range(self._num_layer):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._contexts)
            d.attention_norm = self.norm(f'{pfx}.{i}.input_layernorm.weight')
            if self._is_linear_attn(i):
                d.linear_attn = self.linear_attn(f'{pfx}.{i}.linear_attn', i)
            else:
                d.attention = self.attn(f'{pfx}.{i}.self_attn', i)
            d.ffn_norm = self.norm(f'{pfx}.{i}.post_attention_layernorm.weight')
            if self.num_experts(i) > 0:
                d.feed_forward = self.ffn(f'{pfx}.{i}.mlp.shared_expert', i)
                d.moe_ffn = self.moe(f'{pfx}.{i}.mlp', i)
            else:
                d.feed_forward = self.ffn(f'{pfx}.{i}.mlp', i)
            layers[str(i)] = d
        return layers
```

- [ ] **Step 2: Sanity import.**

```bash
python -c "from lmdeploy.turbomind.deploy.source_model.qwen3_5_spec import Qwen3_5Spec; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py
git commit -m "deploy: migrate qwen3_5_spec to new TextModelSpec base"
```

---

### Task 6: Migrate `gpt_oss_spec.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py`

Gpt-oss exercises sliding-window attention, attention sinks, packed MoE experts (interleaved gate_up), and MXFP4.

- [ ] **Step 1: Replace the file contents.**

```python
# Copyright (c) OpenMMLab. All rights reserved.
"""gpt-oss TextModelSpec for the new pipeline."""
from __future__ import annotations

import re

import torch

import _turbomind as _tm

from ..builder import (AttentionBuilder, DecoderLayerBuilder, FfnBuilder,
                       MoeBuilder, ModuleListBuilder, TextModelBuilder,
                       _act_type_id)
from ..builder import DecoderLayerConfig, ModuleListConfig
from ..kind_map import build_linear
from ..linear import Linear
from ..spec import TextModelSpec
from .base import INPUT_MODELS
from .utils import _pad_inter_size, reorder_rotary_emb_linear

_LAYER_PATTERN = r'model\.layers\.([0-9]+).'


def map_experts(s: str) -> str:
    s = re.sub(r'(experts.*proj)$', r'\1.weight', s)
    s = re.sub(r'(experts.*proj)_bias$', r'\1.bias', s)
    s = re.sub(r'(experts.*proj)_blocks$', r'\1.blocks', s)
    s = re.sub(r'(experts.*proj)_scales$', r'\1.scales', s)
    return s


@INPUT_MODELS.register_module(name='gpt-oss')
class GptOssSpec(TextModelSpec):
    """Weight spec for gpt-oss (MoE with packed experts)."""

    _layer_pattern = _LAYER_PATTERN
    _loader_mappings = [map_experts]
    # gpt-oss always uses the plain `model.*` layout — pin to skip on-load
    # re-detection by TextModelSpec.set_params.
    _pin_layer_prefix = True

    def __init__(self, hf_cfg: dict, engine_cfg, *, group_size: int = 0):
        super().__init__(hf_cfg, engine_cfg, group_size=group_size)

        self._layer_prefix = 'model.layers'
        self._embed_key = 'model.embed_tokens.weight'
        self._norm_key = 'model.norm.weight'

        self._n_experts = hf_cfg['num_local_experts']
        dtype = self._cpp_dtype()

        # ---- Attention template (sliding window set per layer) ----
        self._attn_cfg = _tm.AttentionConfig()
        self._attn_cfg.hidden_dim  = self._hidden_units
        self._attn_cfg.head_dim    = self._head_dim
        self._attn_cfg.head_num    = self._head_num
        self._attn_cfg.kv_head_num = self._kv_head_num_padded
        self._attn_cfg.has_bias    = int(hf_cfg['attention_bias'])
        self._attn_cfg.attn_sink   = True
        self._attn_cfg.rope_dim    = self._rope.dim
        self._attn_cfg.window_size = 0
        self._attn_cfg.tp_size     = engine_cfg.attn_tp_size
        self._attn_cfg.data_type   = dtype

        # ---- FFN template ----
        self._ffn_cfg = _tm.FfnConfig()
        self._ffn_cfg.hidden_dim = self._hidden_units
        self._ffn_cfg.has_bias   = True
        self._ffn_cfg.tp_size    = engine_cfg.mlp_tp_size
        self._ffn_cfg.data_type  = dtype
        self._ffn_cfg.act_type   = _act_type_id('gpt-oss')

        # ---- MoE template ----
        self._moe_cfg = _tm.MoeConfig()
        self._moe_cfg.method            = 1
        self._moe_cfg.experts_per_token = hf_cfg['experts_per_token']
        self._moe_cfg.norm_topk_prob    = True
        self._moe_cfg.shared_gate       = False
        self._moe_cfg.routed_scale      = 1.0
        self._moe_cfg.router_bias       = True
        self._moe_cfg.topk_group        = 1
        self._moe_cfg.topk_method       = 'greedy'
        self._moe_cfg.n_group           = 1
        self._moe_cfg.scoring_func      = 'softmax'
        self._moe_cfg.router_n_groups   = 0
        self._moe_cfg.hidden_dim        = self._hidden_units
        self._moe_cfg.mlp_bias          = True
        self._moe_cfg.data_type         = dtype
        self._moe_cfg.tp_size           = engine_cfg.mlp_tp_size
        self._moe_cfg.act_type          = _act_type_id('gpt-oss')
        self._moe_cfg.fuse_silu         = True

        self._expert_inter_size_padded = _pad_inter_size(
            hf_cfg['intermediate_size'], self._group_size,
            engine_cfg.mlp_tp_size)

        # Per-layer window sizes from layer_types
        types = hf_cfg['layer_types']
        sliding = hf_cfg['sliding_window']
        self._window_sizes = [
            sliding if t == 'sliding_attention' else 0 for t in types
        ]

        # Inter-size list (zero; gpt-oss has no dense FFN layers)
        self._inter_sizes_padded = [0] * self._num_layer
        self._expert_nums = [self._n_experts] * self._num_layer

    def num_experts(self, layer: int) -> int:
        return self._n_experts

    # Per-layer: add window_size list
    def _copy_perlayer_fields(self, mc):
        super()._copy_perlayer_fields(mc)
        mc.window_size = self._window_sizes

    # ------------------------------------------------------------------
    # model() — same topology as old code
    # ------------------------------------------------------------------

    def model(self):
        root = TextModelBuilder(self._root_handles, self._contexts)
        root.tok_embeddings = self.token_embeds(self._embed_key)
        root.norm = self.output_norm(self._norm_key)
        lm_key = self._embed_key if self._tie_embeddings else 'lm_head.weight'
        root.output = self.lm_head(lm_key)
        root.layers = self.layers(self._layer_prefix)

    # Standard RMSNorm
    def output_norm(self, key):
        from ..builder import NormBuilder, make_norm_config
        w = self._get(key)
        cfg = make_norm_config(dim=self._hidden_units, data_type=self._cpp_dtype())
        m = NormBuilder(cfg, self._contexts)
        m.set_weight(w)
        return m

    def norm(self, key):
        return self.output_norm(key)

    # ------------------------------------------------------------------
    # Attention factory — sets per-layer window_size on the clone
    # ------------------------------------------------------------------

    def attn(self, pfx, layer):
        q = self._linear(f'{pfx}.q_proj')
        k = self._linear(f'{pfx}.k_proj')
        v = self._linear(f'{pfx}.v_proj')
        o = self._linear(f'{pfx}.o_proj')

        q = reorder_rotary_emb_linear(q, self._head_dim, self._rope.dim)
        k = reorder_rotary_emb_linear(k, self._head_dim, self._rope.dim)

        cfg = self._attn_cfg.clone()
        cfg.window_size = self._window_sizes[layer]

        attn = AttentionBuilder(cfg, self._contexts,
                                tp=self.engine_cfg.attn_tp_size,
                                ranks=self._attn_ranks)
        attn.add_qkv_proj(q, k, v)
        attn.add_o_proj(o)

        attn.add_param('sinks', self._get(f'{pfx}.sinks'))
        return attn

    # ------------------------------------------------------------------
    # FFN/MoE factories — packed-expert handling
    # ------------------------------------------------------------------

    def ffn(self, pfx, layer, inter_size=None, fused_moe=False):
        w1 = self._linear(f'{pfx}.gate_proj')
        w3 = self._linear(f'{pfx}.up_proj')
        w2 = self._linear(f'{pfx}.down_proj')

        cfg = self._ffn_cfg.clone()
        cfg.inter_size = (inter_size if inter_size is not None
                          else self._inter_sizes_padded[layer])
        cfg.fuse_silu  = False
        cfg.fused_moe  = fused_moe

        m = FfnBuilder(cfg, self._contexts,
                       tp=self.engine_cfg.mlp_tp_size,
                       ranks=self._mlp_ranks)
        m.add_ffn(w1, w2, w3)
        return m

    def moe(self, pfx, layer):
        if self.num_experts(layer) <= 0:
            return None

        cfg = self._moe_cfg.clone()
        cfg.layer_id   = layer
        cfg.expert_num = self._expert_nums[layer]
        cfg.inter_size = self._expert_inter_size_padded

        m = MoeBuilder(cfg, self._contexts,
                       tp=self.engine_cfg.mlp_tp_size,
                       ranks=self._mlp_ranks)

        dtype = self._cpp_dtype()
        gate_w = self._get(f'{pfx}.router.weight')
        gate_w = gate_w.t() if gate_w.dim() > 1 else gate_w
        tensors = {'weight': gate_w}
        gate_bias = self._get(f'{pfx}.router.bias')
        if gate_bias is not None:
            tensors['bias'] = gate_bias
        m.add_gate('gate', Linear(tensors), model_dtype=dtype)

        experts = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for e in range(self.num_experts(layer)):
            experts[str(e)] = self._packed_expert_ffn(
                f'{pfx}.experts.{e}', self._expert_inter_size_padded)
        m.experts = experts
        return m

    def layers(self, pfx):
        layers = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for i in range(self._num_layer):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._contexts)
            d.attention_norm = self.norm(f'{pfx}.{i}.input_layernorm.weight')
            d.attention = self.attn(f'{pfx}.{i}.self_attn', i)
            d.ffn_norm = self.norm(f'{pfx}.{i}.post_attention_layernorm.weight')
            if self.num_experts(i) > 0:
                d.moe_ffn = self.moe(f'{pfx}.{i}.mlp', i)
            layers[str(i)] = d
        return layers

    # ------------------------------------------------------------------
    # Packed-expert decoding (gate_up interleaved, TM layout)
    # ------------------------------------------------------------------

    def _read_packed_expert(self, prefix: str, expert: int):
        lin = build_linear(self.params, prefix, index=expert)
        if lin is None:
            return None
        if lin.weight_format.name == 'trivial':
            w = lin.tensors.get('weight')
            if w is not None and w.dim() == 2:
                lin.tensors['weight'] = w.t().contiguous()
        return lin

    @staticmethod
    def _deinterleave(lin: Linear):
        gate_t: dict[str, torch.Tensor] = {}
        up_t: dict[str, torch.Tensor] = {}
        for kind, t in lin.tensors.items():
            gate_t[kind] = t[..., ::2].contiguous()
            up_t[kind]   = t[..., 1::2].contiguous()
        return (Linear(tensors=gate_t, weight_format=lin.weight_format),
                Linear(tensors=up_t,   weight_format=lin.weight_format))

    def _packed_expert_ffn(self, expert_pfx: str, expert_inter: int):
        base_pfx = expert_pfx.rsplit('.', 1)[0]
        expert_id = int(expert_pfx.rsplit('.', 1)[1])
        gate_up_lin = self._read_packed_expert(
            f'{base_pfx}.gate_up_proj', expert_id)
        down_lin = self._read_packed_expert(
            f'{base_pfx}.down_proj', expert_id)
        if gate_up_lin is None or down_lin is None:
            return None

        w1, w3 = self._deinterleave(gate_up_lin)

        cfg = self._ffn_cfg.clone()
        cfg.inter_size = expert_inter
        cfg.fuse_silu  = False
        cfg.fused_moe  = True

        m = FfnBuilder(cfg, self._contexts,
                       tp=self.engine_cfg.mlp_tp_size,
                       ranks=self._mlp_ranks)
        m.add_ffn(w1, down_lin, w3)
        return m
```

- [ ] **Step 2: Sanity import.**

```bash
python -c "from lmdeploy.turbomind.deploy.source_model.gpt_oss_spec import GptOssSpec; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py
git commit -m "deploy: migrate gpt_oss_spec to new TextModelSpec base"
```

---

### Task 7: Migrate `glm4_moe_lite_spec.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py`

GLM-4 MoE Lite exercises MLA + MoE + YaRN + dense first-k layers.

- [ ] **Step 1: Replace the file contents.**

```python
# Copyright (c) OpenMMLab. All rights reserved.
"""GLM-4 MoE Lite (GLM-4.7-Flash) TextModelSpec for the new pipeline."""
from __future__ import annotations

import torch

import _turbomind as _tm

from ..builder import (DecoderLayerBuilder, FfnBuilder, MLABuilder,
                       MoeBuilder, ModuleListBuilder, TextModelBuilder,
                       _act_type_id)
from ..builder import DecoderLayerConfig, ModuleListConfig
from ..linear import Linear
from ..spec import TextModelSpec
from .base import INPUT_MODELS
from .utils import _pad_inter_size, get_yarn_params, parse_rope_param

_LAYER_PATTERN = r'model\.layers\.([0-9]+).'


@INPUT_MODELS.register_module(name='glm4-moe-lite')
class Glm4MoeLiteSpec(TextModelSpec):
    """Weight spec for GLM-4 MoE Lite (e.g. GLM-4.7-Flash)."""

    _layer_pattern = _LAYER_PATTERN
    # GLM-4 always uses the plain `model.*` layout — pin to skip on-load
    # re-detection by TextModelSpec.set_params.
    _pin_layer_prefix = True

    def __init__(self, hf_cfg: dict, engine_cfg, *, group_size: int = 0):
        super().__init__(hf_cfg, engine_cfg, group_size=group_size)

        self._layer_prefix = 'model.layers'
        self._embed_key = 'model.embed_tokens.weight'
        self._norm_key = 'model.norm.weight'

        self._n_experts = hf_cfg.get('n_routed_experts', 0)
        self._dense_layers = hf_cfg.get('first_k_dense_replace', 1)
        dtype = self._cpp_dtype()

        # ---- MLA head geometry (recomputed; differs from _parse_base default) ----
        qk_nope_dim = hf_cfg['qk_nope_head_dim']
        qk_rope_dim = hf_cfg['qk_rope_head_dim']
        kv_lora_rank = hf_cfg['kv_lora_rank']
        q_head_dim = qk_nope_dim + qk_rope_dim
        size_per_head = q_head_dim
        v_head_dim = hf_cfg['v_head_dim']
        softmax_scale = 0.0
        if kv_lora_rank and kv_lora_rank != qk_nope_dim:
            size_per_head = kv_lora_rank + qk_rope_dim
            v_head_dim = kv_lora_rank
            softmax_scale = q_head_dim ** (-0.5)

        # Override _parse_base defaults for MLA geometry
        self._head_dim = size_per_head
        self._kv_head_num = 1
        self._kv_head_num_padded = 1   # MLA never padded to tp
        # RoPE dim = qk_rope_dim for MLA (not head_dim)
        self._rope, self._max_position_embeddings = parse_rope_param(
            hf_cfg, qk_rope_dim)
        self._softmax_scale = softmax_scale
        self._qk_nope_dim = qk_nope_dim

        # YaRN for MLA (override attention_factor + softmax_scale)
        rope_scaling = (hf_cfg.get('rope_parameters') or
                        hf_cfg.get('rope_scaling'))
        if rope_scaling and rope_scaling.get('type') == 'yarn':
            attention_factor, yarn_scale = get_yarn_params(rope_scaling)
            yarn_scale *= q_head_dim ** (-0.5)
            self._rope.max_position_embeddings = \
                rope_scaling['original_max_position_embeddings']
            self._rope.attention_factor = attention_factor
            self._softmax_scale = yarn_scale

        # ---- Attention template (for MLA; uses _tm.AttentionConfig) ----
        self._attn_cfg = _tm.AttentionConfig()
        self._attn_cfg.hidden_dim      = self._hidden_units
        self._attn_cfg.head_dim        = size_per_head
        self._attn_cfg.head_num        = self._head_num
        self._attn_cfg.kv_head_num     = 1
        self._attn_cfg.kv_lora_rank    = kv_lora_rank
        self._attn_cfg.q_lora_rank     = hf_cfg.get('q_lora_rank') or 0
        self._attn_cfg.qk_rope_dim     = qk_rope_dim
        self._attn_cfg.qk_nope_dim     = qk_nope_dim
        self._attn_cfg.v_head_dim      = v_head_dim
        self._attn_cfg.has_bias        = False
        self._attn_cfg.qk_norm         = False
        self._attn_cfg.attn_sink       = False
        self._attn_cfg.attn_output_gate = False
        self._attn_cfg.rope_dim        = 0   # MLA handles rope separately
        self._attn_cfg.window_size     = 0
        self._attn_cfg.tp_size         = engine_cfg.attn_tp_size
        self._attn_cfg.data_type       = dtype

        # ---- FFN template ----
        self._ffn_cfg = _tm.FfnConfig()
        self._ffn_cfg.hidden_dim = self._hidden_units
        self._ffn_cfg.has_bias   = False
        self._ffn_cfg.tp_size    = engine_cfg.mlp_tp_size
        self._ffn_cfg.data_type  = dtype
        self._ffn_cfg.act_type   = _act_type_id('silu')

        # ---- MoE template (GLM-specific: noaux_tc + sigmoid) ----
        if self._n_experts > 0:
            self._moe_cfg = _tm.MoeConfig()
            self._moe_cfg.method            = 1
            self._moe_cfg.experts_per_token = hf_cfg['num_experts_per_tok']
            self._moe_cfg.norm_topk_prob    = hf_cfg.get('norm_topk_prob', True)
            self._moe_cfg.shared_gate       = False
            self._moe_cfg.routed_scale      = hf_cfg.get('routed_scaling_factor', 1.0)
            self._moe_cfg.router_bias       = False
            self._moe_cfg.topk_group        = hf_cfg.get('topk_group', 1)
            self._moe_cfg.topk_method       = 'noaux_tc'  # GLM-specific
            self._moe_cfg.n_group           = hf_cfg.get('n_group', 1)
            self._moe_cfg.scoring_func      = 'sigmoid'   # GLM-specific
            self._moe_cfg.router_n_groups   = hf_cfg.get('router_n_groups', 0)
            self._moe_cfg.hidden_dim        = self._hidden_units
            self._moe_cfg.mlp_bias          = False
            self._moe_cfg.data_type         = dtype
            self._moe_cfg.tp_size           = engine_cfg.mlp_tp_size
            self._moe_cfg.act_type          = _act_type_id('silu')
            self._moe_cfg.fuse_silu         = True

            self._expert_inter_size_padded = _pad_inter_size(
                hf_cfg['moe_intermediate_size'], self._group_size,
                engine_cfg.mlp_tp_size)
        else:
            self._expert_inter_size_padded = 0

        # Per-layer inter_size:
        #   - dense layers use intermediate_size
        #   - MoE layers use n_shared_experts * moe_intermediate_size
        n_shared_experts = hf_cfg.get('n_shared_experts', 1)
        expert_inter = hf_cfg['moe_intermediate_size']
        raw_inter = [n_shared_experts * expert_inter] * self._num_layer
        raw_inter[0] = hf_cfg.get('intermediate_size',
                                  n_shared_experts * expert_inter)
        self._inter_sizes_padded = [
            _pad_inter_size(v, self._group_size, engine_cfg.mlp_tp_size)
            for v in raw_inter
        ]
        # Per-layer expert count (0 for dense layers)
        self._expert_nums = [
            self._n_experts if i >= self._dense_layers else 0
            for i in range(self._num_layer)
        ]

        self._tune_layer_num = 2  # GLM-MoE recommends tuning 2 layers

    def num_experts(self, layer: int) -> int:
        if layer < self._dense_layers:
            return 0
        return self._n_experts

    # ------------------------------------------------------------------
    # model() — same as old code
    # ------------------------------------------------------------------

    def model(self):
        root = TextModelBuilder(self._root_handles, self._contexts)
        root.tok_embeddings = self.token_embeds(self._embed_key)
        root.norm = self.output_norm(self._norm_key)
        root.output = self.lm_head('lm_head.weight')  # GLM: never tied
        root.layers = self.layers(self._layer_prefix)

    # Standard RMSNorm
    def output_norm(self, key):
        from ..builder import NormBuilder, make_norm_config
        w = self._get(key)
        cfg = make_norm_config(dim=self._hidden_units, data_type=self._cpp_dtype())
        m = NormBuilder(cfg, self._contexts)
        m.set_weight(w)
        return m

    def norm(self, key):
        return self.output_norm(key)

    # ------------------------------------------------------------------
    # MLA attention (uses MLABuilder + self._attn_cfg clone)
    # ------------------------------------------------------------------

    def attn(self, pfx, layer):
        cfg = self._attn_cfg.clone()
        builder = MLABuilder(cfg, self._contexts,
                             tp=self.engine_cfg.attn_tp_size,
                             ranks=self._attn_ranks)

        q_b = (self._linear(f'{pfx}.q_b_proj') or
               self._linear(f'{pfx}.q_proj'))
        builder.add_projections(
            q_a_proj=self._linear(f'{pfx}.q_a_proj'),
            q_b_proj=q_b,
            kv_a_proj=self._linear(f'{pfx}.kv_a_proj_with_mqa'),
            kv_b_proj=self._linear(f'{pfx}.kv_b_proj'),
            wo=self._linear(f'{pfx}.o_proj'),
        )
        builder.add_norms(
            q_a_norm=self._get(f'{pfx}.q_a_layernorm.weight'),
            kv_a_norm=self._get(f'{pfx}.kv_a_layernorm.weight'),
            data_type=self._cpp_dtype(),
        )
        return builder

    # ------------------------------------------------------------------
    # FFN / MoE factories
    # ------------------------------------------------------------------

    def ffn(self, pfx, layer, inter_size=None, fused_moe=False):
        w1 = self._linear(f'{pfx}.gate_proj')
        w3 = self._linear(f'{pfx}.up_proj')
        w2 = self._linear(f'{pfx}.down_proj')

        cfg = self._ffn_cfg.clone()
        cfg.inter_size = (inter_size if inter_size is not None
                          else self._inter_sizes_padded[layer])
        cfg.fuse_silu  = False
        cfg.fused_moe  = fused_moe

        m = FfnBuilder(cfg, self._contexts,
                       tp=self.engine_cfg.mlp_tp_size,
                       ranks=self._mlp_ranks)
        m.add_ffn(w1, w2, w3)
        return m

    def moe(self, pfx, layer):
        if self.num_experts(layer) <= 0:
            return None

        cfg = self._moe_cfg.clone()
        cfg.layer_id   = layer
        cfg.expert_num = self._n_experts
        cfg.inter_size = self._expert_inter_size_padded

        m = MoeBuilder(cfg, self._contexts,
                       tp=self.engine_cfg.mlp_tp_size,
                       ranks=self._mlp_ranks)

        dtype = self._cpp_dtype()
        gate_w = self._get(f'{pfx}.gate.weight')
        gate_w = gate_w.t() if gate_w.dim() > 1 else gate_w
        tensors = {'weight': gate_w}
        gate_bias = self._get(f'{pfx}.gate.bias')
        if gate_bias is not None:
            tensors['bias'] = gate_bias
        m.add_gate('gate', Linear(tensors), model_dtype=dtype)

        correction = self._get(f'{pfx}.gate.e_score_correction_bias')
        m.add_param('score_correction_bias', correction)

        experts = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for e in range(self._n_experts):
            experts[str(e)] = self.ffn(
                f'{pfx}.experts.{e}', layer,
                inter_size=self._expert_inter_size_padded, fused_moe=True)
        m.experts = experts
        return m

    def layers(self, pfx):
        layers = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for i in range(self._num_layer):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._contexts)
            d.attention_norm = self.norm(f'{pfx}.{i}.input_layernorm.weight')
            d.attention = self.attn(f'{pfx}.{i}.self_attn', i)
            d.ffn_norm = self.norm(f'{pfx}.{i}.post_attention_layernorm.weight')
            if i < self._dense_layers:
                d.feed_forward = self.ffn(f'{pfx}.{i}.mlp', i)
            else:
                d.feed_forward = self.ffn(f'{pfx}.{i}.mlp.shared_experts', i)
                d.moe_ffn = self.moe(f'{pfx}.{i}.mlp', i)
            layers[str(i)] = d
        return layers
```

- [ ] **Step 2: Sanity import.**

```bash
python -c "from lmdeploy.turbomind.deploy.source_model.glm4_moe_lite_spec import Glm4MoeLiteSpec; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py
git commit -m "deploy: migrate glm4_moe_lite_spec to new TextModelSpec base"
```

---

## Phase 4 — Remove old machinery

### Task 8: Delete `BaseInputModel` and update `source_model/__init__.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/base.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/__init__.py`

- [ ] **Step 1: Rewrite `source_model/base.py` to keep only the registry.**

```python
# Copyright (c) OpenMMLab. All rights reserved.
"""Source-model registry.

The INPUT_MODELS registry maps an architecture name to its TextModelSpec
subclass. Specs register themselves via ``@INPUT_MODELS.register_module(name=...)``.
"""
from __future__ import annotations

from mmengine import Registry

INPUT_MODELS = Registry('source model',
                        locations=['lmdeploy.turbomind.deploy.source_model.base'])
```

- [ ] **Step 2: Update `source_model/__init__.py` to import the spec classes instead of InputModel classes.**

```python
# Copyright (c) OpenMMLab. All rights reserved.
from .glm4_moe_lite_spec import Glm4MoeLiteSpec  # noqa: F401
from .gpt_oss_spec import GptOssSpec  # noqa: F401
from .qwen3_5_spec import Qwen3_5Spec  # noqa: F401
from .qwen3_spec import Qwen3TextSpec  # noqa: F401
```

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/base.py \
         lmdeploy/turbomind/deploy/source_model/__init__.py
git commit -m "deploy: drop BaseInputModel; specs register themselves directly"
```

---

### Task 9: Rewrite `BaseOutputModel` to take a spec

**Files:**
- Modify: `lmdeploy/turbomind/deploy/target_model/base.py`

- [ ] **Step 1: Replace the body of `target_model/base.py`.**

```python
# Copyright (c) OpenMMLab. All rights reserved.
"""BaseOutputModel — drives the spec through TextModelLoader + export."""
from __future__ import annotations

from abc import ABC

from mmengine import Registry

from ..config import (AttentionConfig, LoraConfig, ModelConfig,
                      TurbomindModelConfig)

OUTPUT_MODELS = Registry('target model',
                         locations=['lmdeploy.turbomind.deploy.target_model.base'])


class BaseOutputModel(ABC):
    """Base output model. Drives a TextModelSpec through loading + commit."""

    @classmethod
    def finalize_config(cls, spec, cfg: TurbomindModelConfig):
        """Assemble the YAML wire-format config from spec + pre-seeded fields.

        The spec has already been constructed with a resolved engine_config
        (dtype, model_format, session_len, tp sizes) plus group_size. The
        only fields that the converter set directly onto ``cfg`` without
        a corresponding engine_config field are ``model_arch``,
        ``chat_template``, and ``model_name`` (pure metadata).

        We generate ``produced`` from the spec, copy those three metadata
        fields from ``cfg`` onto it, then install ``produced`` back onto
        ``cfg``.
        """
        produced = spec.to_legacy_config()
        preserved = ('model_arch', 'chat_template', 'model_name')
        for name in preserved:
            val = getattr(cfg.model_config, name, None)
            if val not in (None, '', 0):
                setattr(produced.model_config, name, val)
        produced.model_config.verify()
        cfg.model_config     = produced.model_config
        cfg.attention_config = produced.attention_config
        cfg.lora_config      = produced.lora_config

    def __init__(self, spec, cfg, model_comm, gpu_count, model_path):
        from ..text_model_loader import TextModelLoader
        self.spec = spec
        self.tm_config = cfg
        self.model_config = cfg.model_config
        self.attention_config = cfg.attention_config
        self.lora_config = cfg.lora_config
        self.attn_tp_size = cfg.model_config.attn_tp_size
        self.attn_cp_size = cfg.model_config.attn_cp_size
        self.mlp_tp_size = cfg.model_config.mlp_tp_size
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
                self.model_comm.mlp_tp_rank(index))

    # ------------------------------------------------------------------
    # Export drivers
    # ------------------------------------------------------------------

    def export(self) -> None:
        from tqdm import tqdm
        import torch
        from ..loader import create_loader
        pbar = tqdm(total=1, desc='Convert to turbomind format', leave=False)
        loader = create_loader(self.model_path, self.spec._layer_pattern,
                               self.spec._loader_mappings)
        self.spec.set_params(loader.all_items())
        self.spec.model()
        torch.cuda.empty_cache()
        pbar.update(1)
        pbar.close()

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
```

- [ ] **Step 2: Verify the class imports.**

```bash
python -c "from lmdeploy.turbomind.deploy.target_model.base import BaseOutputModel; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/target_model/base.py
git commit -m "deploy: BaseOutputModel takes spec; finalize_config shrinks to shim"
```

---

### Task 10: Shrink `TextModelLoader`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/text_model_loader.py`

- [ ] **Step 1: Replace the body of `text_model_loader.py`.**

```python
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
        )
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/text_model_loader.py
git commit -m "deploy: shrink TextModelLoader to runtime-binder"
```

---

### Task 11: Update `converter.py` to create specs directly

**Files:**
- Modify: `lmdeploy/turbomind/deploy/converter.py`

- [ ] **Step 1: In `converter.py`, replace `get_tm_config()` with a version that creates the spec directly and returns `(spec, tm_cfg, model_path)`.**

Find the existing `def get_tm_config(model_path, ...)` (starts around line 120) and replace its body with:

```python
def get_tm_config(model_path,
                  model_name,
                  chat_template_name,
                  engine_config: TurbomindEngineConfig,
                  group_size: int = None):
    """Compute finalized TurbomindModelConfig and the TextModelSpec.

    Returns:
        tuple: (spec, tm_cfg, model_path)
    """
    _, cfg = get_model_arch(model_path)
    quant_config = search_nested_config(cfg.to_dict(), 'quantization_config')
    if quant_config:
        quant_method = quant_config.get('quant_method')
        _group_size = int(quant_config.get('group_size', 0))
        version = quant_config.get('version')
        assert engine_config.model_format is None or engine_config.model_format == quant_method, (
            f'mismatched quant method: user input "{engine_config.model_format}" '
            f'vs model quant_config "{quant_method}"')
        assert not group_size or group_size == _group_size, (
            f'mismatched quant group size: user input "{group_size}" '
            f'vs model quant_config "{_group_size}"')

        if quant_method == 'awq':
            assert version == 'gemm', f'unsupported quant config: {quant_config}'
        elif quant_method == 'gptq':
            assert not quant_config.get('desc_act', False) and quant_config.get(
                'sym', True), f'unsupported quant config: {quant_config}'
        elif quant_method == 'fp8':
            pass
        elif quant_method == 'mxfp4':
            _group_size = 32
        elif quant_method == 'compressed-tensors':
            _format = quant_config['config_groups']['group_0']['format']
            assert _format == 'pack-quantized', (
                'compressed-tennsors only supports pack-quantized format, '
                f'but got {_format}')
            _weights = quant_config['config_groups']['group_0']['weights']
            _group_size = _weights['group_size']
            _num_bits = _weights['num_bits']
            _type = _weights['type']
            assert _num_bits == 4 and _type == 'int', (
                'pack-quantized requires 4-bit int, '
                f'but got {_num_bits}-bit {_type}')
        else:
            assert 0, f'unsupported quant_config: {quant_config}'

        engine_config.model_format = quant_method
        group_size = _group_size

    group_size = _validate_quant_group_size(engine_config.model_format, group_size)

    input_model_name = get_input_model_registered_name(
        model_path, engine_config.model_format)

    # Build the converter-preseeded tm_cfg (dtype, format, model_arch, etc.)
    output_model_name, tm_cfg = get_output_model_registered_name_and_config(
        model_path=model_path,
        model_format=engine_config.model_format,
        dtype=engine_config.dtype,
        group_size=group_size)

    # Propagate resolved dtype + format + session_len back onto engine_config
    # so the spec sees the resolved values during its own parsing.
    engine_config.dtype = tm_cfg.model_config.data_type
    engine_config.model_format = tm_cfg.model_config.model_format
    if engine_config.session_len is None:
        engine_config.session_len = tm_cfg.model_config.session_len
    if engine_config.attn_tp_size is None:
        engine_config.attn_tp_size = 1
    if engine_config.attn_cp_size is None:
        engine_config.attn_cp_size = 1
    if engine_config.mlp_tp_size is None:
        engine_config.mlp_tp_size = 1

    tm_cfg.model_config.chat_template = chat_template_name
    tm_cfg.model_config.model_name = model_name

    if engine_config.attn_tp_size is not None:
        tm_cfg.model_config.attn_tp_size = engine_config.attn_tp_size
    if engine_config.attn_cp_size is not None:
        tm_cfg.model_config.attn_cp_size = engine_config.attn_cp_size
    if engine_config.mlp_tp_size is not None:
        tm_cfg.model_config.mlp_tp_size = engine_config.mlp_tp_size

    # Build the spec from hf_cfg + engine_config + resolved group_size
    hf_cfg = load_model_config(model_path)
    spec_cls = INPUT_MODELS.get(input_model_name)
    spec = spec_cls(hf_cfg, engine_config, group_size=group_size or 0)

    BaseOutputModel.finalize_config(spec, tm_cfg)

    return spec, tm_cfg, model_path
```

At the top of `converter.py`, ensure these imports exist (add `load_model_config` if not already there; `INPUT_MODELS` and `BaseOutputModel` are already imported):

```python
from .source_model.utils import load_model_config
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/converter.py
git commit -m "deploy: get_tm_config creates spec directly; returns (spec, tm_cfg, path)"
```

---

### Task 12: Update `turbomind.py._from_hf` and `update_params`

**Files:**
- Modify: `lmdeploy/turbomind/turbomind.py`

- [ ] **Step 1: Find `_from_hf` (around line 232) and update the part that unpacks `get_tm_config` and constructs the OUTPUT_MODELS instance.**

Replace the current code (rough shape: `input_model, tm_cfg = get_tm_config(...)` … `OUTPUT_MODELS.get('tm')(input_model=..., cfg=..., model_cls=..., model_comm=..., gpu_count=...)`) with:

```python
from .deploy.converter import get_tm_config
from .deploy.target_model.base import OUTPUT_MODELS

spec, tm_cfg, model_path = get_tm_config(
    model_path, self.model_name, self.chat_template_name, engine_config)

self._postprocess_config(tm_cfg, engine_config)

model_comm = _tm.TurboMind.create(model_dir='',
                                  config=yaml.safe_dump(self.config_dict))
self._create_weight(model_comm)

self._tm_model = OUTPUT_MODELS.get('tm')(
    spec=spec,
    cfg=tm_cfg,
    model_comm=model_comm,
    gpu_count=self.gpu_count,
    model_path=model_path)
```

The `TextModelLoader` import is no longer needed here (BaseOutputModel constructs it internally).

- [ ] **Step 2: Fix the `update_params` path to address `model_path` on the OutputModel directly.**

Find the line in `update_params` (around line 290):

```python
tm_model.input_model.model_path = que
```

Change it to:

```python
# update_params replaces the on-disk checkpoint source with a Queue; the
# OutputModel now owns model_path directly (input_model was removed).
tm_model.model_path = que
```

- [ ] **Step 3: Run a no-op sanity check — import turbomind without creating an engine.**

```bash
python -c "import lmdeploy.turbomind.turbomind; print('ok')"
```

Expected: prints `ok`. No import-time exception.

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/turbomind.py
git commit -m "deploy: _from_hf passes spec to OUTPUT_MODELS; fix update_params path"
```

---

### Task 13: Remove `make_*_config(mc, ...)` factories from builder files

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/attention.py`
- Modify: `lmdeploy/turbomind/deploy/builder/mla.py`
- Modify: `lmdeploy/turbomind/deploy/builder/ffn.py`
- Modify: `lmdeploy/turbomind/deploy/builder/moe.py`
- Modify: `lmdeploy/turbomind/deploy/builder/deltanet.py`
- Modify: `lmdeploy/turbomind/deploy/builder/__init__.py`

- [ ] **Step 1: Delete the `make_attention_config` function from `builder/attention.py`.**

Remove the function defined at `builder/attention.py:24-45`:

```python
def make_attention_config(mc, *, tp_size, tp_rank=0, dtype, window_size=0,
                         rope_dim=0):
    """Build C++ AttentionConfig from ModelConfig."""
    cfg = _tm.AttentionConfig()
    cfg.hidden_dim = mc.hidden_units
    # ... (all 18 lines)
    return cfg
```

Also remove the `# Config factory` section divider comment above it and any unused imports of `_tm` that resulted.

- [ ] **Step 2: Delete `make_mla_config` from `builder/mla.py`.**

Remove the function at `builder/mla.py:16-46`.

- [ ] **Step 3: Delete `make_ffn_config` from `builder/ffn.py`.**

Remove the function at `builder/ffn.py:28-41`.

- [ ] **Step 4: Delete `make_moe_config` from `builder/moe.py`.**

Remove the function at `builder/moe.py:14-39`.

- [ ] **Step 5: Delete `make_deltanet_config` from `builder/deltanet.py`.**

Remove the function at `builder/deltanet.py:23-36`.

- [ ] **Step 6: Update `builder/__init__.py` to drop the removed factories from imports and `__all__`.**

Replace the file contents:

```python
# Copyright (c) OpenMMLab. All rights reserved.
"""Builder sub-package — spec-driven module loading for TurboMind."""
from __future__ import annotations

from ._base import (Builder, TextModelBuilder, SplitSide,
                    _cpp_dtype, _act_type_id, _torch_dtype_to_cpp)
from .attention import AttentionBuilder
from .deltanet import DeltaNetBuilder
from .decoder_layer import DecoderLayerBuilder, DecoderLayerConfig
from .ffn import FfnBuilder, fuse_ffn_linears
from .linear import LinearBuilder, make_linear_config
from .mla import MLABuilder
from .moe import MoeBuilder
from .module_list import ModuleListBuilder, ModuleListConfig
from .norm import NormBuilder, make_norm_config

__all__ = [
    # Base
    'Builder', 'TextModelBuilder', 'SplitSide',
    '_cpp_dtype', '_act_type_id', '_torch_dtype_to_cpp',
    # Builders
    'AttentionBuilder', 'FfnBuilder', 'MoeBuilder',
    'DeltaNetBuilder', 'MLABuilder',
    'DecoderLayerBuilder', 'ModuleListBuilder',
    'NormBuilder', 'LinearBuilder',
    # Primitive config wrappers (still used by default token_embeds/lm_head)
    'make_linear_config', 'make_norm_config',
    # C++ config re-exports
    'DecoderLayerConfig', 'ModuleListConfig',
    # Helper functions
    'fuse_ffn_linears',
]
```

- [ ] **Step 7: Verify nothing left imports the removed factories.**

```bash
rg 'make_attention_config|make_mla_config|make_ffn_config|make_moe_config|make_deltanet_config' lmdeploy/
```

Expected: no output (all usages removed).

- [ ] **Step 8: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/
git commit -m "deploy: remove make_*_config(mc,...) factories from builders

Spec writes _tm config structs directly; adapter factories obsolete."
```

---

## Phase 5 — Validation

### Task 14: Run the turbomind-tester validation matrix

**Files:** none modified.

This task validates that the refactor preserves end-to-end model correctness for every supported architecture + quant combination. It uses the `turbomind-tester` subagent.

- [ ] **Step 1: Dispatch the turbomind-tester agent on the dense-baseline matrix.**

Run one tester invocation asking it to validate these model/config combinations:

```
- qwen3 dense,         tp=1, format=hf
- qwen3 dense,         tp=2, format=hf
- qwen3-moe,           tp=1, format=hf
- qwen3-moe,           tp=2, format=hf
```

Pass criterion: every row returns exit=0 AND the response contains meaningful words for a "what is 2+2?" style prompt with at least 128 new tokens.

If any fail, collect the full log + stack trace (Python or gdb for C++) and stop. Fix and repeat before moving on.

- [ ] **Step 2: Dispatch the turbomind-tester agent on the Qwen3.5 matrix.**

```
- qwen3_5 dense,       tp=1, format=hf
- qwen3_5 dense,       tp=2, format=hf
- qwen3_5-moe,         tp=1, format=hf
- qwen3_5-moe,         tp=2, format=hf
```

If an AWQ or FP8 variant is available in the local model cache, add those rows.

Pass criterion same as Step 1.

- [ ] **Step 3: Dispatch the turbomind-tester agent on gpt-oss.**

```
- gpt-oss,             tp=1, format=mxfp4
- gpt-oss,             tp=2, format=mxfp4
```

Pass criterion same as Step 1.

- [ ] **Step 4: Dispatch the turbomind-tester agent on glm4-moe-lite.**

```
- glm4-moe-lite,       tp=1, format=hf
- glm4-moe-lite,       tp=2, format=hf
```

Pass criterion same as Step 1.

- [ ] **Step 5: Dump YAML diff — confirm narrowing effect.**

For each model tested, verify that `tm_cfg.to_dict()['model_config']` no longer contains the 12 removed MoE keys:

```bash
python -c "
from lmdeploy.turbomind.deploy.converter import get_tm_config
from lmdeploy.messages import TurbomindEngineConfig
import json
ec = TurbomindEngineConfig(dtype='bfloat16', tp=1)
spec, cfg, _ = get_tm_config('<local-qwen3-moe-path>', 'qwen3', '', ec)
keys = set(cfg.to_dict()['model_config'].keys())
removed = {'expert_num','expert_router_bias','expert_inter_size','experts_per_token','moe_shared_gate','norm_topk_prob','routed_scale','topk_group','topk_method','moe_group_num','scoring_func','router_n_groups'}
assert not (keys & removed), f'unexpected MoE keys remaining: {keys & removed}'
print('narrowing ok; model_config has', len(keys), 'fields')
"
```

Expected: prints `narrowing ok; model_config has <N> fields` where N is the post-refactor count. No assertion failure.

- [ ] **Step 6: Commit the validation record.**

Write a short note to `docs/superpowers/plans/2026-04-17-config-handling-refactor.md` at the bottom under a `## Validation record` header, listing which models / TP configs / quant formats were validated and the date. Then commit:

```bash
git add docs/superpowers/plans/2026-04-17-config-handling-refactor.md
git commit -m "deploy: record validation matrix for config-handling refactor"
```

---

## Post-refactor follow-ups (not in this plan)

These are explicitly out of scope and should be separate plans:

- Extract Python-only fields (`model_arch`, `chat_template`, `attn_tp_size`, etc.) off `ModelConfig` into a `RuntimeConfig`.
- Eliminate `ModelConfig` entirely — requires `turbomind.cc` to stop reading the YAML wire format.
- Simplify `pad_for_tp` in `attention.py:96-105` now that `ModelConfig.kv_head_num` is no longer mutated under it.
