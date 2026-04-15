# Spec Deduplication Design

**Date:** 2026-04-15
**Scope:** Remove copy-paste duplication across the 4 model spec files.

---

## Problem

Four spec files (`qwen3_spec.py`, `gpt_oss_spec.py`, `qwen3_5_spec.py`, `glm4_moe_lite_spec.py`)
contain ~60-70% identical boilerplate: the same factory methods, the same None guards, dead
`model_info()` methods, and the same free function (`reorder_rotary_emb`) copy-pasted 3 times.

Additionally, all 5 `InputModel` subclasses duplicate an identical 5-line `__init__()`.

## Design

### 1. Base class gains shared factory methods

`TextModelSpec` (in `spec.py`) gains concrete methods that take checkpoint keys as parameters.
No key is hardcoded in the base — all passed in by each spec's `model()`.

**`token_embeds(key)`** — reads `self._get(key)`, pads vocab, returns `LinearBuilder`.

**`lm_head(key)`** — reads `self._get(key)`, pads vocab, transposes, returns `LinearBuilder`.
Each spec resolves tie-word-embeddings before calling (passes the correct key).

**`output_norm(key)`** — reads `self._get(key)`, returns `NormBuilder`.
Renamed from `root_norm`.

**`norm(key)`** — reads `self._get(key)`, returns `NormBuilder`.
Default used by 3/4 specs. qwen3_5 writes its own `norm` and `output_norm` from scratch
(including the `_zero_centered` transform) — no override/super pattern.

**`_cpp_dtype()`** — returns `_cd(self._mc.data_type)`. Moved from all 4 specs.

**`_linear(pfx)`** — already on base. Delete identical overrides from all 4 specs.

Example `model()` after change:

```python
def model(self):
    root = TextModelBuilder(self._root_handles, self._contexts)
    root.tok_embeddings = self.token_embeds('model.embed_tokens.weight')
    root.norm = self.output_norm('model.norm.weight')
    root.output = self.lm_head('lm_head.weight')
    root.layers = self.layers('model.layers')
```

### 2. `reorder_rotary_emb` → single free function

Move to `source_model/utils.py`. All 3 specs that need it (qwen3, gpt_oss, qwen3_5) import
from there. Delete the 3 duplicate copies.

### 3. Kill dead code

- **`Spec.model_info()`**: never called by anyone. Only `InputModel.model_info()` is consumed
  (by `finalize_config` in `target_model/base.py:37`). Delete the abstract declaration from
  `TextModelSpec` and all 4 concrete implementations.

- **`_linear()` overrides**: identical 1-liner in all 4 specs (`return build_linear(self.params,
  prefix)`). Base already has this. Delete all overrides.

### 4. Kill None guards

Remove all `if x is None: return None` from factory methods. If a required weight is missing
from the checkpoint, the model is broken — crash immediately with a clear error rather than
deferring to an obscure downstream `AttributeError`.

### 5. Absorb InputModel `__init__()` into `BaseInputModel`

Move the identical constructor body to `BaseInputModel.__init__`:

```python
def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
    self.model_path = model_path
    self.tokenizer_path = tokenizer_path
    self.model_config = load_model_config(model_path)
    self.model_format = kwargs.get('model_format')
    self.fp8_quant = kwargs.get('fp8_quant', False)
```

Delete `__init__` from all 5 InputModel subclasses (Qwen3InputModel, GptOssInputModel,
Qwen3_5InputModel, Qwen3_5MoeInputModel, Glm4MoeLiteInputModel).

---

## File changes

| File | Change |
|---|---|
| `deploy/spec.py` | Add `token_embeds(key)`, `lm_head(key)`, `output_norm(key)`, `norm(key)`, `_cpp_dtype()`. Remove abstract `model_info`. Remove None guards from new methods. |
| `deploy/source_model/utils.py` | Add `reorder_rotary_emb` free function. |
| `deploy/source_model/base.py` | Absorb `__init__` body from subclasses. |
| `deploy/source_model/qwen3_spec.py` | Delete `model_info`, `_linear`, `_cpp_dtype`, `token_embeds`, `root_norm`, `lm_head`, `norm`, `reorder_rotary_emb`, all None guards. Import `reorder_rotary_emb` from utils. |
| `deploy/source_model/gpt_oss_spec.py` | Same as qwen3. |
| `deploy/source_model/qwen3_5_spec.py` | Same as qwen3 + override `norm`/`output_norm` with `_zero_centered` transform. Import `reorder_rotary_emb` from utils. |
| `deploy/source_model/glm4_moe_lite_spec.py` | Delete `model_info`, `_linear`, `_cpp_dtype`, `token_embeds`, `root_norm`, `lm_head`, `norm`, None guards. No `reorder_rotary_emb` (MLA, no RoPE). |

## Not in scope (held)

- inter_size resolution helper
- layers() skeleton abstraction
