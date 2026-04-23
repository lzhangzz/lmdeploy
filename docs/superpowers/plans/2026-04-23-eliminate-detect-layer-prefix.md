# Eliminate detect_layer_prefix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `detect_layer_prefix` runtime detection and make each spec explicitly declare its checkpoint key prefix at construction time.

**Architecture:** Each spec sets `_layer_prefix`, `_embed_key`, `_norm_key` in its own `__init__`. The base class no longer provides defaults or re-detects from checkpoint keys. `_pin_layer_prefix` and `detect_layer_prefix` are deleted entirely.

**Tech Stack:** Python, lmdeploy/turbomind/deploy module

---

### Task 1: Remove detect_layer_prefix from utils.py

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/utils.py:246-267`

- [ ] **Step 1: Delete the function**

Delete lines 246–267 (the comment block and `detect_layer_prefix` function body):

```python
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

Remove the section comment and the blank line above it. The `layer_progress` function (line 270+) remains untouched.

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/utils.py
git commit -m "deploy: delete detect_layer_prefix utility"
```

---

### Task 2: Clean up TextModelSpec base class

**Files:**
- Modify: `lmdeploy/turbomind/deploy/spec.py`

- [ ] **Step 1: Remove detect_layer_prefix import**

Change line 13–14 from:
```python
from .source_model.utils import (detect_layer_prefix,
                                 parse_rope_param, rope_type_to_int)
```
to:
```python
from .source_model.utils import (parse_rope_param, rope_type_to_int)
```

- [ ] **Step 2: Delete _pin_layer_prefix class variable and comment**

Delete lines 41–46:
```python
    # If True, the subclass's __init__ has pinned _layer_prefix / _embed_key /
    # _norm_key to fixed values; set_params() will NOT re-detect from params.
    # Subclasses that need on-load detection (e.g. multimodal wrappers where
    # the decoder lives under model.language_model.*) leave this False and
    # let the base class re-detect when weights arrive.
    _pin_layer_prefix: bool = False
```

- [ ] **Step 3: Update _parse_base docstring**

Update the docstring's "Populated" list to remove `_layer_prefix`, `_embed_key`, `_norm_key`. Change from:
```
          _num_layer, _vocab_size, _norm_eps, _head_num, _kv_head_num,
          _head_dim, _hidden_units, _rope,
          _max_position_embeddings, _tie_embeddings, _layer_prefix,
          _embed_key, _norm_key, _model_name, _tune_layer_num,
          _embedding_size.
```
to:
```
          _num_layer, _vocab_size, _norm_eps, _head_num, _kv_head_num,
          _head_dim, _hidden_units, _rope,
          _max_position_embeddings, _tie_embeddings,
          _model_name, _tune_layer_num, _embedding_size.
```

- [ ] **Step 4: Delete prefix defaults from _parse_base**

Delete lines 113–116:
```python
        # Layer-prefix detection deferred until weights loaded; default now.
        # Subclasses that know their prefix unconditionally can override.
        self._layer_prefix, self._embed_key, self._norm_key = \
            detect_layer_prefix(None, cfg)
```

- [ ] **Step 5: Simplify set_params**

Change `set_params` from:
```python
    def set_params(self, params: dict):
        self.params = params
        # Re-detect layer prefix from the actual checkpoint keys, unless the
        # subclass has pinned its prefix (class-level _pin_layer_prefix = True).
        if not self._pin_layer_prefix:
            self._layer_prefix, self._embed_key, self._norm_key = \
                detect_layer_prefix(params, self.hf_cfg)
```
to:
```python
    def set_params(self, params: dict):
        self.params = params
```

- [ ] **Step 6: Commit**

```bash
git add lmdeploy/turbomind/deploy/spec.py
git commit -m "deploy: remove detect_layer_prefix from TextModelSpec base"
```

---

### Task 3: Remove _pin_layer_prefix from gpt_oss, qwen3, glm4_moe_lite specs

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py:38-40`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py:31-33`
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py:25-27`

These three specs already set `_layer_prefix`, `_embed_key`, `_norm_key` in `__init__`. They just need their `_pin_layer_prefix = True` and the preceding comment removed.

- [ ] **Step 1: gpt_oss_spec.py**

Delete lines 38–40:
```python
    # gpt-oss always uses the plain `model.*` layout — pin to skip on-load
    # re-detection by TextModelSpec.set_params.
    _pin_layer_prefix = True
```

- [ ] **Step 2: qwen3_spec.py**

Delete lines 31–33:
```python
    # Qwen3 always uses the plain `model.*` layout — pin to skip on-load
    # re-detection by TextModelSpec.set_params.
    _pin_layer_prefix = True
```

- [ ] **Step 3: glm4_moe_lite_spec.py**

Delete lines 25–27:
```python
    # GLM-4 always uses the plain `model.*` layout — pin to skip on-load
    # re-detection by TextModelSpec.set_params.
    _pin_layer_prefix = True
```

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py \
        lmdeploy/turbomind/deploy/source_model/qwen3_spec.py \
        lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py
git commit -m "deploy: remove _pin_layer_prefix from gpt-oss, qwen3, glm4-moe-lite"
```

---

### Task 4: Hardcode prefix in Qwen3.5 spec

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py`

- [ ] **Step 1: Update class docstring**

Change the docstring from:
```python
    """Weight spec for Qwen3.5 (dense + linear-attn + optional MoE).

    ``_pin_layer_prefix`` is intentionally left at its default False — Qwen3.5
    may be packaged as a multimodal root where the decoder lives under
    ``model.language_model.*``. The base-class ``set_params`` re-runs
    ``detect_layer_prefix`` when weights arrive to resolve the correct
    prefix.
    """
```
to:
```python
    """Weight spec for Qwen3.5 (dense + linear-attn + optional MoE)."""
```

- [ ] **Step 2: Add explicit prefix in __init__**

In `__init__`, after the `super().__init__()` call and before the `partial_rotary_factor` block (line 48), add:
```python
        self._layer_prefix = 'model.language_model.layers'
        self._embed_key = 'model.language_model.embed_tokens.weight'
        self._norm_key = 'model.language_model.norm.weight'
```

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py
git commit -m "deploy: hardcode language_model prefix in Qwen3.5 spec"
```

---

### Task 5: Verify with model tests

**Files:** None (verification only)

- [ ] **Step 1: Check GPU is free**

Run: `nvidia-smi` or use `get_gpu_usage` MCP tool.

- [ ] **Step 2: Test a model.lines model (e.g. qwen3 or gpt-oss)**

```bash
cd /data/lmdeploy-modeling/build && ninja && cd /data/lmdeploy-modeling
python scripts/test_turbomind_model.py Qwen/Qwen3-4B --tp 1
```

Verify the response contains meaningful human words and is at least 128 tokens.

- [ ] **Step 3: Test a model.language_model model (Qwen3.5)**

```bash
python scripts/test_turbomind_model.py Qwen/Qwen3.5-27B --tp 1
```

Verify the response contains meaningful human words and is at least 128 tokens.
