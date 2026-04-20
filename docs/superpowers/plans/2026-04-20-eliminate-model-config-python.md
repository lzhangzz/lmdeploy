# Eliminate Python-side ModelConfig Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the Python `ModelConfig` dataclass and all supporting infrastructure (to_legacy_config, _copy_*_fields), migrating the few remaining consumers to read from `TurbomindEngineConfig` or saved spec values.

**Architecture:** Four incremental PRs: (1) stop serializing dead `model_config` to YAML, (2) migrate Python consumers to alternative sources, (3) remove `to_legacy_config()` and `_copy_*_fields()` bridge methods, (4) delete the `ModelConfig` dataclass itself.

**Tech Stack:** Python, pydantic dataclasses, YAML serialization, pybind11

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `lmdeploy/turbomind/deploy/config.py` | Config dataclasses (ModelConfig, AttentionConfig, LoraConfig, TurbomindModelConfig) | Delete ModelConfig; simplify TurbomindModelConfig |
| `lmdeploy/turbomind/turbomind.py` | Main entry point: TurboMind + TurboMindInstance | Stop serializing model_config; migrate consumers |
| `lmdeploy/turbomind/deploy/spec.py` | TextModelSpec base class | Remove to_legacy_config, _copy_*_fields |
| `lmdeploy/turbomind/deploy/target_model/base.py` | BaseOutputModel — drives spec through loading | Simplify finalize_config; migrate consumers |
| `lmdeploy/turbomind/deploy/converter.py` | Orchestrates config construction | Stop populating ModelConfig fields |
| `lmdeploy/turbomind/deploy/load_context.py` | LoadContext for C++ module tree loading | Accept dtype directly |

---

## Task 1: Stop serializing `model_config` to YAML

**Files:**
- Modify: `lmdeploy/turbomind/deploy/config.py:188-192` (`TurbomindModelConfig.to_dict`)
- Modify: `lmdeploy/turbomind/turbomind.py:209-226` (`_postprocess_config`)

- [ ] **Step 1: Modify `TurbomindModelConfig.to_dict()` to exclude `model_config`**

In `lmdeploy/turbomind/deploy/config.py`, change the `to_dict()` method (lines 188-192):

```python
def to_dict(self):
    """Export to a dict."""
    return dict(attention_config=config_to_dict(self.attention_config),
                lora_config=config_to_dict(self.lora_config))
```

Remove `model_config=config_to_dict(self.model_config),` from the returned dict.

- [ ] **Step 2: Verify `_postprocess_config` still works**

In `lmdeploy/turbomind/turbomind.py`, the `_postprocess_config()` method (lines 209-226) calls `self.config.to_dict()` at line 223. After the change, `self.config_dict` will have keys `attention_config`, `lora_config`, and `engine_config` (added at line 224) — but NOT `model_config`. The `model_config` Python object is still on `self.config.model_config` for internal reads. No changes needed here.

- [ ] **Step 3: Test with a model**

Run the test script against any model (e.g., Qwen3-8B). Verify the model loads, responds to a prompt with meaningful text, and the YAML log no longer shows a `model_config` section.

Run: `python scripts/test_turbomind_model.py Qwen/Qwen3-8B --prompt "Hello, how are you?" --max-new-tokens 128`

Expected: Model responds normally. The logged config JSON no longer has `model_config` as a top-level key.

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/config.py
git commit -m "refactor(turbomind): stop serializing model_config to YAML

C++ ignores model_config. Stop including it in the YAML passed to
_tm.TurboMind.create(). The Python ModelConfig object is still
constructed for internal reads (to be removed in subsequent PRs)."
```

---

## Task 2: Migrate `session_len` consumer

**Files:**
- Modify: `lmdeploy/turbomind/turbomind.py:175`

- [ ] **Step 1: Change `self.session_len` to read from `engine_config`**

In `lmdeploy/turbomind/turbomind.py`, line 175, change:

```python
self.session_len = self.config.session_len
```

to:

```python
self.session_len = _engine_config.session_len
```

`_engine_config` is the local variable holding the `TurbomindEngineConfig` (set at line 139-141). This is the same value — `config.session_len` was a property that read `self.model_config.session_len`, which was populated from `engine_config.session_len` via `update_from_engine_config()`.

- [ ] **Step 2: Test with a model**

Run: `python scripts/test_turbomind_model.py Qwen/Qwen3-8B --prompt "Hello" --max-new-tokens 128`

Expected: Model responds normally. `session_len` is set correctly.

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/turbomind.py
git commit -m "refactor(turbomind): read session_len from engine_config directly"
```

---

## Task 3: Migrate `data_type` consumer in `prepare_embeddings`

**Files:**
- Modify: `lmdeploy/turbomind/turbomind.py:557-558`

- [ ] **Step 1: Change `prepare_embeddings` to read dtype from `engine_config`**

In `lmdeploy/turbomind/turbomind.py`, lines 557-558, change:

```python
_MAP = dict(bfloat16=torch.bfloat16, float16=torch.float16)
dtype = _MAP[self.tm_model.config.model_config.data_type]
```

to:

```python
_MAP = dict(bfloat16=torch.bfloat16, float16=torch.float16)
dtype = _MAP[self.tm_model.engine_config.dtype]
```

- [ ] **Step 2: Test with a model using input embeddings**

Testing embedding inputs requires a specific setup. At minimum, verify the model still starts normally and basic text inference works (the embedding path is only hit when `input_embeddings` is not None).

Run: `python scripts/test_turbomind_model.py Qwen/Qwen3-8B --prompt "Hello" --max-new-tokens 128`

Expected: Model responds normally.

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/turbomind.py
git commit -m "refactor(turbomind): read dtype from engine_config in prepare_embeddings"
```

---

## Task 4: Migrate `vocab_size` consumer in `async_stream_infer`

**Files:**
- Modify: `lmdeploy/turbomind/turbomind.py:228-251` (`_from_hf`)
- Modify: `lmdeploy/turbomind/turbomind.py:660`

- [ ] **Step 1: Save `vocab_size` from the spec during `_from_hf()`**

In `lmdeploy/turbomind/turbomind.py`, inside `_from_hf()` (around line 236-243), after `get_tm_config()` returns the spec, save `vocab_size`. Add this line after line 237 (`spec, tm_cfg, model_path = get_tm_config(...)`):

```python
spec, tm_cfg, model_path = get_tm_config(
    model_path, self.model_name, self.chat_template_name, engine_config)
self._vocab_size = spec._vocab_size
```

The `spec._vocab_size` is set in `TextModelSpec._parse_base()` at `spec.py:84` from `cfg['vocab_size']`.

- [ ] **Step 2: Change `async_stream_infer` to read from `self._vocab_size`**

In `lmdeploy/turbomind/turbomind.py`, line 660, change:

```python
vocab_size = self.tm_model.config.model_config.vocab_size
```

to:

```python
vocab_size = self.tm_model._vocab_size
```

- [ ] **Step 3: Test with a model**

Run: `python scripts/test_turbomind_model.py Qwen/Qwen3-8B --prompt "Hello" --max-new-tokens 128`

Expected: Model responds normally.

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/turbomind.py
git commit -m "refactor(turbomind): save vocab_size from spec instead of ModelConfig"
```

---

## Task 5: Migrate `LoadContext.cpp_dtype` away from `model_config`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/load_context.py:374-399`

- [ ] **Step 1: Change `LoadContext` to accept `dtype` directly**

In `lmdeploy/turbomind/deploy/load_context.py`, change `__init__` (lines 374-389) and `cpp_dtype` property (lines 396-399).

Change the constructor signature and body:

```python
def __init__(self, handle, tp_config: dict,
             dtype: str | None = None,
             context=None):
    self._handle = handle
    self._tp_config = tp_config
    self._dtype = dtype
    self._context = context
```

Change the `model_config` property (lines 391-394) to a `dtype` property:

```python
@property
def dtype(self) -> str:
    assert self._dtype is not None, 'dtype not set'
    return self._dtype
```

Change the `cpp_dtype` property (lines 396-399):

```python
@property
def cpp_dtype(self):
    """C++ DataType enum for the model's compute dtype."""
    return _cpp_dtype(self.dtype)
```

Update `create()` (line 431) and `child()` (line 436) to pass `self._dtype`:

```python
return LoadContext(child, self._tp_config, self._dtype)
```

```python
return LoadContext(handle, self._tp_config, self._dtype)
```

Also remove the `from .config import ModelConfig` import inside the `TYPE_CHECKING` block (lines 14-15).

- [ ] **Step 2: Verify no external callers pass `model_config`**

`LoadContext` is only instantiated from within its own `create()` and `child()` methods (no external callers currently). When it gets wired up in the future, callers will pass `dtype=engine_config.dtype` instead of `model_config=cfg.model_config`.

- [ ] **Step 3: Test with a model**

Run: `python scripts/test_turbomind_model.py Qwen/Qwen3-8B --prompt "Hello" --max-new-tokens 128`

Expected: Model responds normally (LoadContext is not in the active path yet, so this is a compile-time verification).

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/load_context.py
git commit -m "refactor(deploy): LoadContext accepts dtype string instead of ModelConfig"
```

---

## Task 6: Migrate `BaseOutputModel` consumers away from `model_config`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/target_model/base.py:44-61`

- [ ] **Step 1: Change `BaseOutputModel.__init__` to read tp sizes from engine_config**

In `lmdeploy/turbomind/deploy/target_model/base.py`, the `__init__` (lines 44-61) currently stores references from `cfg.model_config`. But at this point, we don't have direct access to `engine_config` in `BaseOutputModel.__init__`. The `engine_config` is set on `TurboMind` (the caller), not passed through.

Looking at the call chain: `turbomind.py:245-250` creates `BaseOutputModel` with `cfg=tm_cfg`. The `tm_cfg` is a `TurbomindModelConfig`. The tp sizes were set on `tm_cfg.model_config` in `converter.py:210-212`.

The cleanest approach: keep reading from `cfg.model_config` for now (it's still alive), but note these reads are build-time only. They'll be cleaned up in Task 8 when we simplify the construction pipeline.

**No changes in this task** — the `BaseOutputModel` reads from `cfg.model_config` which still exists. This gets cleaned up naturally when `ModelConfig` is removed in Task 8.

- [ ] **Step 2: Commit**

No commit needed — deferring to Task 8.

---

## Task 7: Remove `to_legacy_config()` and `_copy_*_fields()` methods

**Files:**
- Modify: `lmdeploy/turbomind/deploy/spec.py:135-224`
- Modify: `lmdeploy/turbomind/deploy/target_model/base.py:20-42`
- Modify: `lmdeploy/turbomind/deploy/converter.py:80-223`

- [ ] **Step 1: Remove `_copy_template_fields`, `_copy_orchestration_fields`, `_copy_perlayer_fields` from `spec.py`**

In `lmdeploy/turbomind/deploy/spec.py`, delete these methods:

- `_copy_template_fields()` (lines 145-190)
- `_copy_orchestration_fields()` (lines 192-206)
- `_copy_perlayer_fields()` (lines 208-217)

Keep `_build_attention_config()` (lines 219-224) — it's still needed for the `attention_config` YAML section.

- [ ] **Step 2: Replace `to_legacy_config()` with `to_attention_config()`**

In `lmdeploy/turbomind/deploy/spec.py`, replace `to_legacy_config()` (lines 135-143):

```python
def to_attention_config(self) -> AttentionConfig:
    """Produce the AttentionConfig for YAML serialization."""
    return self._build_attention_config()
```

This is a simpler version that only produces the `AttentionConfig` (the only YAML section C++ still reads).

- [ ] **Step 3: Update `BaseOutputModel.finalize_config()`**

In `lmdeploy/turbomind/deploy/target_model/base.py`, replace the `finalize_config()` method (lines 20-42):

```python
@classmethod
def finalize_config(cls, spec, cfg: TurbomindModelConfig):
    """Install attention_config from spec onto cfg."""
    cfg.attention_config = spec.to_attention_config()
```

This removes the `to_legacy_config()` call, the metadata copy (`model_arch`, `chat_template`, `model_name`), and the `verify()` call — all of which were dead since C++ ignores `model_config`.

- [ ] **Step 4: Stop populating `model_config` fields in `converter.py`**

In `lmdeploy/turbomind/deploy/converter.py`, the function `get_output_model_registered_name_and_config()` (lines 80-131) creates an empty `TurbomindModelConfig` and sets fields on `config.model_config`. These fields are now only needed for Python-internal reads (not YAML). Simplify by removing the `model_config` field assignments that were only for YAML serialization.

Change lines 125-129 from:

```python
config.model_config.model_arch = model_arch
config.model_config.data_type = dtype
config.model_config.model_format = model_format
config.model_config.group_size = group_size
config.model_config.session_len = session_len
```

to:

```python
config._model_arch = model_arch
config._data_type = dtype
config._model_format = model_format
config._group_size = group_size
config._session_len = session_len
```

These become private attributes on `TurbomindModelConfig` for internal use only.

Update `TurbomindModelConfig` in `config.py` to add these as optional fields:

```python
@dataclass
class TurbomindModelConfig:
    attention_config: AttentionConfig = None
    lora_config: LoraConfig = None
    # Internal fields (not serialized to YAML)
    _model_arch: str = ''
    _data_type: str = ''
    _model_format: str = 'hf'
    _group_size: int = 0
    _session_len: int = 0
    _model_name: str = ''
    _chat_template: str = ''
    _attn_tp_size: int = 1
    _attn_cp_size: int = 1
    _mlp_tp_size: int = 1
```

Wait — pydantic dataclasses don't support underscore-prefixed fields well. Use regular names instead:

Actually, looking at this more carefully, `converter.py` still needs to pass these values through to `get_tm_config()` and then to `BaseOutputModel`. The simplest approach: keep setting them on `config.model_config` for now (ModelConfig still exists), and let Task 8 clean them up when ModelConfig is deleted.

**Revised Step 4:** Keep the `model_config` field assignments in `converter.py` unchanged. They'll be cleaned up in Task 8.

- [ ] **Step 5: Update `converter.py` lines 197-212**

In `lmdeploy/turbomind/deploy/converter.py`, lines 197-212 currently sync between `tm_cfg.model_config` and `engine_config`. These reads from `tm_cfg.model_config` are still needed until ModelConfig is removed. Keep them for now.

- [ ] **Step 6: Test with a model**

Run: `python scripts/test_turbomind_model.py Qwen/Qwen3-8B --prompt "Hello, how are you today?" --max-new-tokens 128`

Expected: Model responds normally. The YAML log shows `attention_config`, `lora_config`, and `engine_config` — no `model_config`.

- [ ] **Step 7: Commit**

```bash
git add lmdeploy/turbomind/deploy/spec.py lmdeploy/turbomind/deploy/target_model/base.py
git commit -m "refactor(deploy): remove to_legacy_config and _copy_*_fields methods

Replace to_legacy_config() with to_attention_config() that only
produces the AttentionConfig C++ needs. Simplify finalize_config()
to only install attention_config."
```

---

## Task 8: Remove `ModelConfig` dataclass

**Files:**
- Modify: `lmdeploy/turbomind/deploy/config.py`
- Modify: `lmdeploy/turbomind/deploy/converter.py`
- Modify: `lmdeploy/turbomind/deploy/target_model/base.py`
- Modify: `lmdeploy/turbomind/turbomind.py`

- [ ] **Step 1: Add internal fields to `TurbomindModelConfig` in `config.py`**

In `lmdeploy/turbomind/deploy/config.py`, modify `TurbomindModelConfig` to absorb the fields that were on `ModelConfig`. Remove `model_config: ModelConfig = None` and add the needed fields directly:

```python
@dataclass
class TurbomindModelConfig:
    """Config for turbomind model."""
    attention_config: AttentionConfig = None
    lora_config: LoraConfig = None
    model_arch: str = ''
    chat_template: str = ''
    model_name: str = ''
    data_type: str = ''
    model_format: str = 'hf'
    session_len: int = 0
    group_size: int = 0
    attn_tp_size: int = 1
    attn_cp_size: int = 1
    mlp_tp_size: int = 1
```

Update `update_from_engine_config()` to set these fields directly instead of `self.model_config`:

```python
def update_from_engine_config(self, config: TurbomindEngineConfig):
    if config is None:
        return
    for key, value in asdict(config).items():
        if value is None:
            continue
        if hasattr(self, key):
            setattr(self, key, value)
        if hasattr(self.attention_config, key):
            setattr(self.attention_config, key, value)

    if config.rope_scaling_factor:
        rope_param = self.attention_config.rope_param or RopeParam(type='', base=0, dim=0)
        rope_param.type = 'dynamic'
        rope_param.factor = config.rope_scaling_factor
        rope_param.max_position_embeddings = self.attention_config.max_position_embeddings
        self.attention_config.rope_param = rope_param
        logger.warning(
            '`--rope-scaling-factor` will be removed in a future release. Please instead use `--hf-overrides`.')
```

Remove `from_dict()` — no longer used (was only for creating empty configs with `model_config`).

Update `to_dict()` (already updated in Task 1):

```python
def to_dict(self):
    return dict(attention_config=config_to_dict(self.attention_config),
                lora_config=config_to_dict(self.lora_config))
```

Remove the convenience properties `session_len`, `group_size`, `vocab_size` (consumers migrated in Tasks 2-4).

- [ ] **Step 2: Delete `ModelConfig` class**

In `lmdeploy/turbomind/deploy/config.py`, delete the entire `ModelConfig` class (lines 38-104). Also delete `ModelConfig.verify()`.

Update `config_to_dict()` to remove `ModelConfig` from the assertion:

```python
def config_to_dict(config):
    assert isinstance(config, (AttentionConfig, LoraConfig)), \
        f'A dataclass is expected, but got {type(config)}'
    return asdict(config)
```

- [ ] **Step 3: Update `converter.py` to use new `TurbomindModelConfig` fields**

In `lmdeploy/turbomind/deploy/converter.py`:

Change `get_output_model_registered_name_and_config()` lines 114-131. Replace `config.model_config.xxx` with `config.xxx`:

```python
config = TurbomindModelConfig()

session_len = _get_and_verify_max_len(model_config, None)

group_size = _validate_quant_group_size(model_format, group_size)

if model_format in ['awq', 'gptq', 'compressed-tensors']:
    dtype = 'float16'
    if model_format == 'compressed-tensors':
        model_format = 'awq'

config.model_arch = model_arch
config.data_type = dtype
config.model_format = model_format
config.group_size = group_size
config.session_len = session_len

return register_name, config
```

Change `get_tm_config()` lines 197-212. Replace `tm_cfg.model_config.xxx` with `tm_cfg.xxx`:

```python
engine_config.dtype = tm_cfg.data_type
engine_config.model_format = tm_cfg.model_format
if engine_config.session_len is None:
    engine_config.session_len = tm_cfg.session_len
if engine_config.attn_tp_size is None:
    engine_config.attn_tp_size = 1
if engine_config.attn_cp_size is None:
    engine_config.attn_cp_size = 1
if engine_config.mlp_tp_size is None:
    engine_config.mlp_tp_size = 1

tm_cfg.chat_template = chat_template_name
tm_cfg.model_name = model_name
tm_cfg.attn_tp_size = engine_config.attn_tp_size
tm_cfg.attn_cp_size = engine_config.attn_cp_size
tm_cfg.mlp_tp_size = engine_config.mlp_tp_size
```

- [ ] **Step 4: Update `BaseOutputModel` in `base.py`**

In `lmdeploy/turbomind/deploy/target_model/base.py`:

Update imports — remove `ModelConfig`:

```python
from ..config import (AttentionConfig, LoraConfig,
                      TurbomindModelConfig)
```

Update `finalize_config()` (already simplified in Task 7, no further changes needed).

Update `__init__()` lines 44-61. Replace `cfg.model_config.xxx` with `cfg.xxx`:

```python
def __init__(self, spec, cfg, model_comm, gpu_count, model_path):
    from ..text_model_loader import TextModelLoader
    self.spec = spec
    self.tm_config = cfg
    self.attn_tp_size = cfg.attn_tp_size
    self.attn_cp_size = cfg.attn_cp_size
    self.mlp_tp_size = cfg.mlp_tp_size
    self.model_comm = model_comm
    self.gpu_count = gpu_count
    self.model_path = model_path
    self.model = TextModelLoader(self)
```

Remove `self.model_config`, `self.attention_config`, `self.lora_config` stores — they were only used for reference passing that no longer needs them.

- [ ] **Step 5: Update `spec.py` imports**

In `lmdeploy/turbomind/deploy/spec.py`, update the import (line 14):

```python
from .config import (AttentionConfig, LoraConfig,
                     TurbomindModelConfig)
```

Remove `ModelConfig` from the import.

- [ ] **Step 6: Test with multiple models**

Run the test script against at least 2 different models to verify nothing broke:

```bash
python scripts/test_turbomind_model.py Qwen/Qwen3-8B --prompt "Hello, how are you?" --max-new-tokens 128
python scripts/test_turbomind_model.py Qwen/Qwen2.5-7B-Instruct --prompt "What is the capital of France?" --max-new-tokens 128
```

Expected: Both models respond normally with meaningful text.

- [ ] **Step 7: Commit**

```bash
git add lmdeploy/turbomind/deploy/config.py lmdeploy/turbomind/deploy/converter.py \
        lmdeploy/turbomind/deploy/target_model/base.py \
        lmdeploy/turbomind/deploy/spec.py \
        lmdeploy/turbomind/turbomind.py
git commit -m "refactor(turbomind): remove ModelConfig dataclass

Absorb the few needed fields (data_type, session_len, model_format,
tp sizes) directly into TurbomindModelConfig. All ModelConfig
consumers have been migrated to read from engine_config or
TurbomindModelConfig directly."
```

---

## Self-Review

### Spec Coverage

| Spec Requirement | Task |
|---|---|
| PR 1: Stop serializing model_config to YAML | Task 1 |
| PR 2: Migrate session_len consumer | Task 2 |
| PR 2: Migrate data_type consumer | Task 3 |
| PR 2: Migrate vocab_size consumer | Task 4 |
| PR 2: Migrate LoadContext.cpp_dtype | Task 5 |
| PR 2: Migrate BaseOutputModel consumers | Task 6 (deferred to Task 8) |
| PR 3: Remove to_legacy_config and _copy_*_fields | Task 7 |
| PR 4: Remove ModelConfig dataclass | Task 8 |

### Placeholder Scan

No TBDs, TODOs, or incomplete steps. Every step has exact file paths, line numbers, and code.

### Type Consistency

- `TurbomindModelConfig.to_attention_config()` returns `AttentionConfig` — matches `finalize_config()` expectation
- `LoadContext` accepts `dtype: str | None` — matches `_cpp_dtype()` input type
- `self._vocab_size` is `int` from `spec._vocab_size` — matches `TokenizerInfo.from_huggingface(vocab_size=...)` expectation
- `engine_config.dtype` is `str` — matches the `_MAP` dict keys in `prepare_embeddings`
