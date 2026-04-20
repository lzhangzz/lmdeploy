# Eliminate `TurbomindModelConfig` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete `TurbomindModelConfig` (plus the dead `LoraConfig` and `config_to_dict`) from `lmdeploy/turbomind/deploy/config.py`, migrating its live behavior into `spec.to_attention_config()` and inlining YAML building directly in `turbomind._from_hf`.

**Architecture:** Six incremental tasks. Task 1 moves the only non-mechanical behavior (AttentionConfig engine-config patching) onto `TextModelSpec`. Tasks 2–3 dismantle the surrounding scaffolding (`_postprocess_config`, `TurboMindInstance.config`, `BaseOutputModel.finalize_config`, the `tm_config`/tp-size attributes). Tasks 4–6 clean up now-dead tests, fold `get_output_model_registered_name_and_config` into `get_tm_config`, and finally delete the `TurbomindModelConfig`/`LoraConfig` classes.

**Tech Stack:** Python, pydantic dataclasses, YAML serialization, pybind11.

---

## File Structure

| File | Responsibility after change |
|---|---|
| `lmdeploy/turbomind/deploy/config.py` | Holds only `RopeParam` and `AttentionConfig` (the two live dataclasses) |
| `lmdeploy/turbomind/deploy/spec.py` | `TextModelSpec.to_attention_config()` owns all `AttentionConfig` construction including engine-config patching |
| `lmdeploy/turbomind/deploy/converter.py` | `get_tm_config(model_path, engine_config, group_size=None) → (spec, model_path)`, mutates `engine_config` in place |
| `lmdeploy/turbomind/deploy/target_model/base.py` | `BaseOutputModel.__init__(spec, engine_config, model_comm, gpu_count, model_path)` — no `finalize_config`, no tm_config/tp-size attributes |
| `lmdeploy/turbomind/turbomind.py` | `_from_hf` inlines YAML build from `spec.to_attention_config()` + `asdict(engine_config)`; no `_postprocess_config`; no `self.config` or `self.config_dict`; `TurboMindInstance.__init__(tm_model, cuda_stream_id)` |
| `tests/test_lmdeploy/test_turbomind/test_converter.py` | Keeps only `test_ffn_reader_kind_none` |
| `tests/test_lmdeploy/test_turbomind/test_compressed_tensors.py` | Keeps the three tests that don't use `converter.get_output_model_registered_name_and_config` |

---

## Task 1: Move `AttentionConfig` patching into `spec.to_attention_config()`

**Rationale:** `TextModelSpec` already holds `self.engine_cfg` and owns every other input to `AttentionConfig`. Move the three engine-config-sourced fields (`cache_block_seq_len`, `use_logn_attn`, `rope_scaling_factor → rope_param`) here; delete `TurbomindModelConfig.update_from_engine_config` and its sole call site.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/spec.py:129-141` (`to_attention_config`, `_build_attention_config`)
- Modify: `lmdeploy/turbomind/deploy/spec.py` (imports) — add `RopeParam`
- Modify: `lmdeploy/turbomind/deploy/config.py:73-99` (delete `update_from_engine_config`)
- Modify: `lmdeploy/turbomind/turbomind.py:216` (delete the call)

### Steps

- [ ] **Step 1: Rewrite `spec.to_attention_config()` to include patching**

In `lmdeploy/turbomind/deploy/spec.py`, replace the two methods `to_attention_config` and `_build_attention_config` (roughly lines 129-141):

```python
def to_attention_config(self) -> AttentionConfig:
    """Produce the AttentionConfig for YAML serialization.

    Reads from self.engine_cfg for runtime-patched fields
    (cache_block_seq_len, use_logn_attn, rope_scaling_factor).
    """
    cfg = AttentionConfig(
        rope_param=self._rope,
        max_position_embeddings=self._max_position_embeddings,
        softmax_scale=self._softmax_scale,
    )
    ec = self.engine_cfg
    if ec.cache_block_seq_len:
        cfg.cache_block_seq_len = ec.cache_block_seq_len
    if ec.use_logn_attn:
        cfg.use_logn_attn = int(ec.use_logn_attn)
    if ec.rope_scaling_factor:
        rope = cfg.rope_param or RopeParam(type='', base=0, dim=0)
        rope.type = 'dynamic'
        rope.factor = ec.rope_scaling_factor
        rope.max_position_embeddings = cfg.max_position_embeddings
        cfg.rope_param = rope
        logger.warning(
            '`--rope-scaling-factor` will be removed in a future release. '
            'Please instead use `--hf-overrides`.')
    return cfg
```

Remove the now-unused private helper `_build_attention_config`.

- [ ] **Step 2: Update imports in `spec.py`**

The new code references `RopeParam` and `logger`. Update the imports near the top of `spec.py`.

Change:

```python
from .config import AttentionConfig
```

to:

```python
from .config import AttentionConfig, RopeParam
```

Check whether `logger` is already defined in `spec.py` via `get_logger`. If not, add:

```python
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')
```

(If the file already has a module-level `logger`, skip this; the warning will use it.)

Run `rg -n "^logger\s*=" lmdeploy/turbomind/deploy/spec.py` to confirm.

- [ ] **Step 3: Delete the call site in `turbomind.py`**

In `lmdeploy/turbomind/turbomind.py`, `_postprocess_config` currently has:

```python
        self.config.update_from_engine_config(engine_config)
```

at line 216. Delete this single line. The comment on line 213-215 becomes stale; delete the 3-line comment block too.

The surrounding code at 209-226 should read:

```python
    def _postprocess_config(self, tm_config: TurbomindModelConfig, engine_config: TurbomindEngineConfig):
        """Postprocess turbomind config by."""
        import copy
        self.config = copy.deepcopy(tm_config)

        self.engine_config = engine_config

        # pack `self.config` and `self.engine_config` into a dict
        self.config_dict = self.config.to_dict()
        self.config_dict.update(dict(engine_config=asdict(self.engine_config)))
        logger.info(f'turbomind model config:\n\n'
                    f'{json.dumps(self.config_dict, indent=2)}')
```

(`_postprocess_config` still exists; it's fully dismantled in Task 2.)

- [ ] **Step 4: Delete `update_from_engine_config` method from `config.py`**

In `lmdeploy/turbomind/deploy/config.py`, delete the `update_from_engine_config` method (lines 73-99). Drop the now-unused imports at the top:

```python
from dataclasses import asdict   # delete if no other user
from lmdeploy.messages import TurbomindEngineConfig   # delete if no other user
from lmdeploy.utils import get_logger   # delete if no other user
logger = get_logger('lmdeploy')   # delete if no other user
```

Run `rg -n "asdict|TurbomindEngineConfig|logger" lmdeploy/turbomind/deploy/config.py` to confirm none of them remain in use.

The `TurbomindModelConfig` class shrinks to:

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

    def to_dict(self):
        """Export to a dict."""
        result = {}
        if self.attention_config is not None:
            result['attention_config'] = config_to_dict(self.attention_config)
        if self.lora_config is not None:
            result['lora_config'] = config_to_dict(self.lora_config)
        return result

    def __str__(self):
        return json.dumps(self.to_dict(), indent=2)
```

- [ ] **Step 5: Smoke test**

Run:

```bash
python scripts/test_turbomind_model.py Qwen/Qwen3-8B <cache_dir> 1 0
```

Expected: model loads, produces a coherent paragraph about reading books, exit 0.

- [ ] **Step 6: Commit**

```bash
git add lmdeploy/turbomind/deploy/spec.py lmdeploy/turbomind/deploy/config.py lmdeploy/turbomind/turbomind.py
git commit -m "$(cat <<'EOF'
refactor(turbomind): move AttentionConfig patching into spec

spec.to_attention_config() now reads self.engine_cfg and applies the
three runtime patches (cache_block_seq_len, use_logn_attn,
rope_scaling_factor) directly, inlining the former
_build_attention_config helper. TurbomindModelConfig.update_from_engine_config
is deleted along with its call site in _postprocess_config.
EOF
)"
```

---

## Task 2: Eliminate `_postprocess_config`; drop `TurboMindInstance.config`

**Rationale:** `_postprocess_config` is now three mechanical lines (engine_config store + YAML dict build + log). Inline it into `_from_hf`. The dead indirection `self.tm_model.config_dict['engine_config'].get('empty_init', False)` becomes a direct field read. `TurboMindInstance.config` is a stored-and-never-read attribute — drop the param.

**Files:**
- Modify: `lmdeploy/turbomind/turbomind.py:209-252` (`_postprocess_config` + `_from_hf`)
- Modify: `lmdeploy/turbomind/turbomind.py:352-360` (`create_instance`)
- Modify: `lmdeploy/turbomind/turbomind.py:484-501` (`TurboMindInstance.__init__`)

### Steps

- [ ] **Step 1: Rewrite `_from_hf` to inline YAML construction**

In `lmdeploy/turbomind/turbomind.py`, replace the body of `_from_hf` (lines 228-253) with:

```python
    def _from_hf(self, model_path: str, engine_config: TurbomindEngineConfig):
        """Load model which is in hf format."""
        assert is_supported(model_path), (
            f'turbomind does not support {model_path}. '
            'Plz try pytorch engine instead.')

        from .deploy.converter import get_tm_config
        from .deploy.target_model.base import OUTPUT_MODELS

        spec, tm_cfg, model_path = get_tm_config(
            model_path, self.model_name, self.chat_template_name, engine_config)

        self._vocab_size = spec._vocab_size
        self.engine_config = engine_config

        config_dict = {
            'attention_config': asdict(spec.to_attention_config()),
            'engine_config': asdict(engine_config),
        }
        logger.info(f'turbomind model config:\n\n'
                    f'{json.dumps(config_dict, indent=2)}')

        model_comm = _tm.TurboMind.create(
            model_dir='', config=yaml.safe_dump(config_dict))
        self._create_weight(model_comm)

        self._tm_model = OUTPUT_MODELS.get('tm')(
            spec=spec,
            cfg=tm_cfg,
            model_comm=model_comm,
            gpu_count=self.gpu_count,
            model_path=model_path)
        return model_comm
```

Note: `cfg=tm_cfg` is kept as-is in this task. `BaseOutputModel` still accepts it. Task 3 changes that call site to `engine_config=engine_config`.

- [ ] **Step 2: Delete `_postprocess_config` method**

Delete the entire `_postprocess_config` method (lines 209-226 of the pre-change file, now orphaned). Also delete the `self.config` and `self.config_dict` references — they exist only in the deleted method.

Run `rg -n "self\.config\b|self\.config_dict\b" lmdeploy/turbomind/turbomind.py` to confirm only the `TurboMindInstance.config` assignment at line 500 remains. That's addressed in Step 4.

- [ ] **Step 3: Update `session_len` read**

Line 175 currently reads `self.session_len = _engine_config.session_len`. Check that the local variable `_engine_config` is still in scope (it's set at lines 139-141). If not, change to `self.session_len = engine_config.session_len` referencing the argument.

Run `rg -n "self\.session_len" lmdeploy/turbomind/turbomind.py` to verify there's exactly one assignment.

- [ ] **Step 4: Fix `TurboMindInstance.__init__` and drop `config` param**

In `lmdeploy/turbomind/turbomind.py`, the existing `TurboMindInstance.__init__` (lines 484-517) has:

```python
    def __init__(self, tm_model: TurboMind, config: TurbomindModelConfig, cuda_stream_id: int = 0):
        self.tm_model = tm_model
        self.cuda_stream_id = cuda_stream_id

        # create model instances
        lazy_init = self.tm_model.config_dict['engine_config'].get('empty_init', False)
        self._model_inst = None if lazy_init else self._create_model_instance()

        self.config = config
        self.lock = None
        # ... errcode_map ...
```

Replace with:

```python
    def __init__(self, tm_model: 'TurboMind', cuda_stream_id: int = 0):
        self.tm_model = tm_model
        self.cuda_stream_id = cuda_stream_id

        # create model instances
        lazy_init = self.tm_model.engine_config.empty_init
        self._model_inst = None if lazy_init else self._create_model_instance()

        self.lock = None
        # ... errcode_map unchanged ...
```

Drop the `config: TurbomindModelConfig` parameter; drop the `self.config = config` line.

- [ ] **Step 5: Update `create_instance` call site**

In `lmdeploy/turbomind/turbomind.py:360`, change:

```python
        return TurboMindInstance(self, self.config, cuda_stream_id)
```

to:

```python
        return TurboMindInstance(self, cuda_stream_id)
```

- [ ] **Step 6: Smoke test**

Run:

```bash
python scripts/test_turbomind_model.py Qwen/Qwen3-8B <cache_dir> 1 0
```

Expected: model loads, produces a coherent paragraph, exit 0. The log should still show `turbomind model config` with `attention_config` + `engine_config` keys (and the still-present `lora_config`; that goes in Task 6).

- [ ] **Step 7: Commit**

```bash
git add lmdeploy/turbomind/turbomind.py
git commit -m "$(cat <<'EOF'
refactor(turbomind): inline YAML build; drop TurboMindInstance.config

_postprocess_config is deleted; its three surviving lines (engine_config
store, YAML dict build, log) inline into _from_hf. self.config and
self.config_dict attributes are removed; the one external reader
(TurboMindInstance lazy_init) now reads engine_config.empty_init
directly. TurboMindInstance.__init__ drops the dead config parameter.
EOF
)"
```

---

## Task 3: `BaseOutputModel` reads `engine_config`; delete `finalize_config`

**Rationale:** `BaseOutputModel.__init__` today reads three tp-size fields off `cfg` (the `TurbomindModelConfig`) but never uses them elsewhere. It also stores `self.tm_config` which is never read. `finalize_config(cls, spec, cfg)` installs `cfg.attention_config = spec.to_attention_config()` — but with YAML building moved inline (Task 2), no one needs `cfg.attention_config` to be populated anymore.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/target_model/base.py` (entire file)
- Modify: `lmdeploy/turbomind/deploy/converter.py:219` (delete `finalize_config` call)
- Modify: `lmdeploy/turbomind/turbomind.py` (change `cfg=tm_cfg` → `engine_config=engine_config` in `_from_hf`)

### Steps

- [ ] **Step 1: Rewrite `BaseOutputModel`**

Replace the entire contents of `lmdeploy/turbomind/deploy/target_model/base.py` with:

```python
# Copyright (c) OpenMMLab. All rights reserved.
"""BaseOutputModel — drives the spec through TextModelLoader + export."""
from __future__ import annotations

from abc import ABC

from mmengine import Registry

from lmdeploy.messages import TurbomindEngineConfig

OUTPUT_MODELS = Registry('target model',
                         locations=['lmdeploy.turbomind.deploy.target_model.base'])


class BaseOutputModel(ABC):
    """Base output model. Drives a TextModelSpec through loading + commit."""

    def __init__(self, spec, engine_config: TurbomindEngineConfig,
                 model_comm, gpu_count, model_path):
        from ..text_model_loader import TextModelLoader
        self.spec = spec
        self.engine_config = engine_config
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
        import torch
        from ..loader import create_loader
        loader = create_loader(self.model_path, self.spec._layer_pattern,
                               self.spec._loader_mappings)
        self.spec.set_params(loader.all_items())
        self.spec.model()
        torch.cuda.empty_cache()

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

Changes vs. the current file:
- `finalize_config` classmethod is gone.
- `__init__` takes `engine_config` instead of `cfg`. Stores `self.engine_config` only; no `self.tm_config`, `self.attn_tp_size`, `self.attn_cp_size`, `self.mlp_tp_size`.
- Import of `TurbomindModelConfig` replaced with `TurbomindEngineConfig`.

- [ ] **Step 2: Update `converter.py` to drop `finalize_config` call**

In `lmdeploy/turbomind/deploy/converter.py`, delete line 219 (`BaseOutputModel.finalize_config(spec, tm_cfg)`) and its blank-line neighbor. The trailing return stays:

```python
    ...
    spec = spec_cls(hf_cfg, engine_config, group_size=group_size or 0)

    return spec, tm_cfg, model_path
```

Also drop the `from .target_model.base import BaseOutputModel` import at line 13 if it was only used for `finalize_config`. Run `rg -n "BaseOutputModel" lmdeploy/turbomind/deploy/converter.py` to confirm.

- [ ] **Step 3: Update `_from_hf` caller in `turbomind.py`**

In `lmdeploy/turbomind/turbomind.py`, inside `_from_hf`, change:

```python
        self._tm_model = OUTPUT_MODELS.get('tm')(
            spec=spec,
            cfg=tm_cfg,
            model_comm=model_comm,
            gpu_count=self.gpu_count,
            model_path=model_path)
```

to:

```python
        self._tm_model = OUTPUT_MODELS.get('tm')(
            spec=spec,
            engine_config=engine_config,
            model_comm=model_comm,
            gpu_count=self.gpu_count,
            model_path=model_path)
```

- [ ] **Step 4: Smoke test**

Run:

```bash
python scripts/test_turbomind_model.py Qwen/Qwen3-8B <cache_dir> 1 0
```

Expected: model loads, produces a coherent paragraph, exit 0.

- [ ] **Step 5: Commit**

```bash
git add lmdeploy/turbomind/deploy/target_model/base.py lmdeploy/turbomind/deploy/converter.py lmdeploy/turbomind/turbomind.py
git commit -m "$(cat <<'EOF'
refactor(turbomind): BaseOutputModel takes engine_config directly

BaseOutputModel.__init__ now accepts engine_config instead of tm_cfg.
The tm_config/attn_tp_size/attn_cp_size/mlp_tp_size attributes (set but
never read) are gone. finalize_config is deleted — its sole effect,
installing cfg.attention_config, is no longer needed now that YAML
building is inline in _from_hf.
EOF
)"
```

---

## Task 4: Remove blocking tests

**Rationale:** Four tests in `test_converter.py` and one in `test_compressed_tensors.py` will break later tasks — either because they call `config.update_from_engine_config` (deleted in Task 1) or because they import / call `get_output_model_registered_name_and_config` (deleted in Task 5). A file-level import failure on `test_converter.py` would stop pytest from collecting the remaining surviving test. Surgically delete only the blocking tests (and the now-unused helpers / imports they used).

**Files:**
- Modify: `tests/test_lmdeploy/test_turbomind/test_converter.py` (delete 4 test functions + unused module-level imports)
- Modify: `tests/test_lmdeploy/test_turbomind/test_compressed_tensors.py` (delete 1 test function + its `_FakeModelConfig` helper + unused `pytest` import)

### Steps

- [ ] **Step 1: Delete blocking tests from `test_converter.py`**

In `tests/test_lmdeploy/test_turbomind/test_converter.py`, delete the following **functions** — leave `test_ffn_reader_kind_none` intact:

- `test_torch_dtype_fallback` (the `def test_torch_dtype_fallback():` block through the final `assert config.model_config.data_type in ('float16', 'bfloat16')`)
- `test_registered_models` (the full function body through `assert config.model_config.model_arch is not None`)
- `test_update_from_engine_config` (the full function body through `assert (config.attention_config.use_logn_attn == engine_config.use_logn_attn)`)
- `test_dtype` (the full function body through `assert _config.model_config.data_type == 'float16'`)

Then delete the now-unused module-level imports at the top of the file:

```python
from lmdeploy import TurbomindEngineConfig
from lmdeploy.turbomind import update_parallel_config
from lmdeploy.turbomind.deploy.converter import (
    get_input_model_registered_name,
    get_output_model_registered_name_and_config,
)
from lmdeploy.turbomind.deploy.source_model.base import INPUT_MODELS
```

All four of those import lines can go — `test_ffn_reader_kind_none` does its own `import re` and reader imports inside the function body.

Verify:

```bash
rg -n "get_output_model_registered_name_and_config|update_from_engine_config" \
    tests/test_lmdeploy/test_turbomind/test_converter.py
```

Expected: no matches.

- [ ] **Step 2: Delete blocking test from `test_compressed_tensors.py`**

In `tests/test_lmdeploy/test_turbomind/test_compressed_tensors.py`, delete:

- The `_FakeModelConfig` class (roughly lines 11-17) — only consumer is the deleted test.
- The `test_compressed_tensors_support_matrix` function (roughly lines 44-70).
- The `from lmdeploy.turbomind.deploy import converter` import at the top if no surviving test uses `converter`. Check with:

  ```bash
  rg -n "converter\." tests/test_lmdeploy/test_turbomind/test_compressed_tensors.py
  ```

- The `import pytest` line if no surviving test uses `pytest.raises`. Check with:

  ```bash
  rg -n "pytest\." tests/test_lmdeploy/test_turbomind/test_compressed_tensors.py
  ```

Keep everything else: `_reference_compressed_tensors_dequant`, `_DummyQwen35Reader`, and the three surviving `test_*` functions.

Verify:

```bash
rg -n "get_output_model_registered_name_and_config" \
    tests/test_lmdeploy/test_turbomind/test_compressed_tensors.py
```

Expected: no matches.

- [ ] **Step 3: Run the surviving tests**

Run:

```bash
pytest tests/test_lmdeploy/test_turbomind/test_converter.py \
       tests/test_lmdeploy/test_turbomind/test_compressed_tensors.py -v
```

Expected: all listed tests pass. No import errors.

- [ ] **Step 4: Commit**

```bash
git add tests/test_lmdeploy/test_turbomind/test_converter.py \
        tests/test_lmdeploy/test_turbomind/test_compressed_tensors.py
git commit -m "$(cat <<'EOF'
test(turbomind): remove tests that block the config elimination

Delete test_torch_dtype_fallback, test_registered_models,
test_update_from_engine_config, test_dtype (all reference
config.model_config.* and/or get_output_model_registered_name_and_config)
and test_compressed_tensors_support_matrix (calls the same deleted
converter API). Preserve the surviving non-blocking tests.
EOF
)"
```

---

## Task 5: Fold `get_output_model_registered_name_and_config` into `get_tm_config`

**Rationale:** With `TurbomindModelConfig` about to die, `get_output_model_registered_name_and_config` has no reason to exist as a separate function. Fold its body in, dedupe the duplicate `get_model_arch(model_path)` call, and extract a private `_resolve_dtype` helper. `get_tm_config` returns `(spec, model_path)` and mutates `engine_config` in place. Unused `model_name`/`chat_template_name` parameters are dropped.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/converter.py` (entire file, essentially)
- Modify: `lmdeploy/turbomind/turbomind.py` (single call site)

### Steps

- [ ] **Step 1: Rewrite `converter.py`**

Replace the contents of `lmdeploy/turbomind/deploy/converter.py` from the `get_output_model_registered_name_and_config` definition through the end of the file with:

```python
def _resolve_dtype(requested: str, hf_model_cfg) -> str:
    """Resolve 'auto' dtype against the HF config and the current device.

    Prefers `dtype` over the deprecated `torch_dtype` key. Falls back to
    float16 on hardware that does not support bfloat16.
    """
    has_bf16 = is_bf16_supported()
    dtype = requested
    if dtype == 'auto':
        dtype = 'bfloat16' if has_bf16 else 'float16'
        torch_dtype = getattr(hf_model_cfg, 'dtype', None)
        if torch_dtype is None:
            torch_dtype = getattr(hf_model_cfg, 'torch_dtype', None)
        TORCH_DTYPE_MAP = {torch.bfloat16: 'bfloat16', torch.float16: 'float16'}
        dtype = TORCH_DTYPE_MAP.get(torch_dtype, dtype)

    if dtype == 'bfloat16' and not has_bf16:
        logger.warning('data type fallback to float16 since '
                       'torch.cuda.is_bf16_supported is False')
        dtype = 'float16'
    return dtype


def get_tm_config(model_path,
                  engine_config: TurbomindEngineConfig,
                  group_size: int = None):
    """Resolve dtype/model_format/group_size/session_len, mutate engine_config
    in place, build the spec.

    Returns:
        tuple: (spec, model_path)
    """
    # 1. Load HF config once; reused for quant_config, dtype, and session_len.
    _, hf_model_cfg = get_model_arch(model_path)

    # 2. Reconcile quant_config (unchanged logic, today's 131-166).
    quant_config = search_nested_config(
        hf_model_cfg.to_dict(), 'quantization_config')
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
                'compressed-tensors only supports pack-quantized format, '
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
    if engine_config.model_format is None:
        engine_config.model_format = 'hf'

    # 3. Resolve dtype and format overrides.
    dtype = _resolve_dtype(engine_config.dtype, hf_model_cfg)
    if engine_config.model_format in ('awq', 'gptq', 'compressed-tensors'):
        dtype = 'float16'
        if engine_config.model_format == 'compressed-tensors':
            engine_config.model_format = 'awq'

    # 4. Resolve session_len default.
    session_len_default = _get_and_verify_max_len(hf_model_cfg, None)

    # 5. Mutate engine_config with resolved values.
    engine_config.dtype = dtype
    if engine_config.session_len is None:
        engine_config.session_len = session_len_default
    engine_config.attn_tp_size = engine_config.attn_tp_size or 1
    engine_config.attn_cp_size = engine_config.attn_cp_size or 1
    engine_config.mlp_tp_size = engine_config.mlp_tp_size or 1

    # 6. Build spec (hf_overrides handling unchanged).
    hf_cfg = load_model_config(model_path)
    if engine_config.hf_overrides:
        logger.warning(f'Overriding HF config with {engine_config.hf_overrides}')
        _deep_merge(hf_cfg, engine_config.hf_overrides)
    spec_name = get_spec_registered_name(model_path, engine_config.model_format)
    spec_cls = INPUT_MODELS.get(spec_name)
    spec = spec_cls(hf_cfg, engine_config, group_size=group_size or 0)

    return spec, model_path
```

Keep the top-of-file helpers and registrations untouched: `_DEFAULT_GROUP_SIZES`, `_SUPPORTED_GROUP_SIZES`, `_validate_quant_group_size`, `get_spec_registered_name`, and the `_deep_merge` helper used for hf_overrides.

Delete `get_output_model_registered_name_and_config` entirely. Drop the `TurbomindModelConfig` import from the top of `converter.py`.

Run `rg -n "TurbomindModelConfig|get_output_model_registered_name_and_config" lmdeploy/turbomind/deploy/converter.py` to confirm neither remains.

- [ ] **Step 2: Update the caller in `turbomind.py`**

In `lmdeploy/turbomind/turbomind.py`, `_from_hf` currently has:

```python
        spec, tm_cfg, model_path = get_tm_config(
            model_path, self.model_name, self.chat_template_name, engine_config)
```

Replace with:

```python
        spec, model_path = get_tm_config(model_path, engine_config)
```

The `tm_cfg` variable is gone. The `OUTPUT_MODELS.get('tm')(...)` call was updated in Task 3 to take `engine_config=engine_config`, so no further change here.

Run `rg -n "tm_cfg" lmdeploy/turbomind/turbomind.py` — should be empty.

- [ ] **Step 3: Smoke test both paths**

Run:

```bash
# Unquantized path
python scripts/test_turbomind_model.py Qwen/Qwen3-8B <cache_dir> 1 0

# Quantized path (group_size validation + dtype='float16' override)
python scripts/test_turbomind_model.py Qwen/Qwen3-8B-AWQ <cache_dir> 1 0
```

Expected: both runs produce coherent paragraphs, exit 0.

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/converter.py lmdeploy/turbomind/turbomind.py
git commit -m "$(cat <<'EOF'
refactor(turbomind): fold get_output_model_registered_name_and_config

get_tm_config(model_path, engine_config, group_size=None) now owns the
full converter pipeline: it resolves dtype (via new _resolve_dtype
helper), model_format, group_size, and session_len, mutates
engine_config in place, and returns (spec, model_path). The dead
model_name and chat_template_name parameters are dropped.
get_output_model_registered_name_and_config is deleted.
EOF
)"
```

---

## Task 6: Delete `TurbomindModelConfig`, `LoraConfig`, and the `lora_config` YAML key

**Rationale:** After Tasks 1–5, nothing imports `TurbomindModelConfig`, nothing constructs `LoraConfig`, and nothing reads `config_to_dict`. Delete them. The `lora_config` YAML key is gone because `_from_hf` (Task 2) already emits only `{attention_config, engine_config}`.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/config.py` (delete 3 things)
- Modify: `lmdeploy/turbomind/turbomind.py` (drop import)

### Steps

- [ ] **Step 1: Verify no remaining users**

Run:

```bash
rg -n "TurbomindModelConfig|LoraConfig|config_to_dict" lmdeploy/ tests/ scripts/ 2>/dev/null
```

Expected output: only hits inside `lmdeploy/turbomind/deploy/config.py` itself. If any hit lands in another file, fix it (it's a leftover from an earlier task).

- [ ] **Step 2: Rewrite `config.py`**

Replace the entire contents of `lmdeploy/turbomind/deploy/config.py` with:

```python
# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import asdict

# use pydantic.dataclasses.dataclass to check data type
from pydantic.dataclasses import dataclass


@dataclass
class RopeParam:
    type: str
    base: float
    dim: int
    factor: float = 1.0
    max_position_embeddings: int = None
    attention_factor: float = 1.0
    beta_fast: float = 32
    beta_slow: float = 1
    low_freq_factor: float = None
    high_freq_factor: float = None
    original_max_position_embeddings: int = None
    mrope_section: list[int] = None


@dataclass
class AttentionConfig:
    softmax_scale: float = 0
    cache_block_seq_len: int = 64
    use_logn_attn: int = 0
    max_position_embeddings: int = 0
    rope_param: RopeParam = None
```

Note: `asdict` is kept in the import list only if other callers inside the turbomind deploy package reimport it from here. Run `rg -n "from \.config import" lmdeploy/turbomind/deploy/` — if no one imports `asdict` from `config`, remove the `from dataclasses import asdict` line. (Most likely: only `RopeParam` and `AttentionConfig` are imported externally, so `asdict` can be dropped.)

- [ ] **Step 3: Drop `TurbomindModelConfig` import from `turbomind.py`**

In `lmdeploy/turbomind/turbomind.py`, delete line 28:

```python
from .deploy.config import TurbomindModelConfig
```

Run `rg -n "TurbomindModelConfig" lmdeploy/turbomind/turbomind.py` — should be empty.

- [ ] **Step 4: Smoke test**

Run:

```bash
python scripts/test_turbomind_model.py Qwen/Qwen3-8B <cache_dir> 1 0
```

Expected: model loads, produces a coherent paragraph, exit 0. The log now shows exactly two top-level keys: `attention_config` and `engine_config`.

Also run the pytest survivors to confirm nothing regressed:

```bash
pytest tests/test_lmdeploy/test_turbomind/test_converter.py tests/test_lmdeploy/test_turbomind/test_compressed_tensors.py -v
```

Expected: all pass.

- [ ] **Step 5: Final sanity sweep**

Run:

```bash
rg -n "TurbomindModelConfig|LoraConfig|config_to_dict|_postprocess_config|update_from_engine_config|get_output_model_registered_name_and_config|finalize_config" lmdeploy/ tests/ 2>/dev/null
```

Expected: no hits. If any remain, they're dead references — delete them.

- [ ] **Step 6: Commit**

```bash
git add lmdeploy/turbomind/deploy/config.py lmdeploy/turbomind/turbomind.py
git commit -m "$(cat <<'EOF'
refactor(turbomind): delete TurbomindModelConfig and LoraConfig

config.py shrinks to just RopeParam and AttentionConfig — the only
live dataclasses. TurbomindModelConfig, LoraConfig, and the
config_to_dict helper are deleted; the lora_config YAML key is gone
from the payload handed to C++ (which never read it).
EOF
)"
```

---

## Self-Review

### Spec Coverage

| Spec section | Task |
|---|---|
| Decision 1: Patching onto spec | Task 1 |
| Decision 2: Drop lora_config + delete LoraConfig | Task 6 (YAML drop actually happens in Task 2 when `_from_hf` emits the new dict) |
| Decision 3: Fold + delete tests | Task 4 (tests) + Task 5 (fold) |
| Design §1: `spec.to_attention_config()` with patching | Task 1 |
| Design §2: YAML inline in `_from_hf`; `_postprocess_config` deleted | Task 2 |
| Design §3: `get_tm_config` returns `(spec, model_path)`; mutates engine_config; drops unused params | Task 5 |
| Design §4: `BaseOutputModel` reads engine_config; `finalize_config` deleted | Task 3 |
| Design §5: `TurboMindInstance` drops `config` param; `lazy_init` direct read | Task 2 |
| Design §6: `config.py` shrinks to RopeParam + AttentionConfig | Task 6 |
| Files Changed table rows | All 7 files touched across Tasks 1–6 |
| Verification: unquantized smoke run | Task 5 step 3 (both paths); Task 1/2/3/6 each repeat the unquantized run |
| Verification: quantized smoke run | Task 5 step 3 |

### Placeholder Scan

- No "TBD", "TODO", "implement later", or "fill in details".
- No "add appropriate error handling" or "handle edge cases" without concrete code.
- Every code step includes complete code, not descriptions.
- Every shell step includes the exact command and expected result.

### Type Consistency

- `TurbomindEngineConfig` is consistently the type of the `engine_config` parameter everywhere it appears (spec.py already stores `self.engine_cfg` of this type; `BaseOutputModel.__init__` annotates it; `get_tm_config` annotates it).
- `spec.to_attention_config() → AttentionConfig` — matches the single call site in `turbomind._from_hf` where it's passed through `asdict()` into the YAML dict.
- `get_tm_config(model_path, engine_config, group_size=None) → (spec, model_path)` — matches the single call site after Task 5 (`spec, model_path = get_tm_config(model_path, engine_config)`).
- `TurboMindInstance(tm_model, cuda_stream_id=0)` — matches the single call site in `create_instance` after Task 2.
- `BaseOutputModel(spec, engine_config, model_comm, gpu_count, model_path)` — matches the single call site in `_from_hf` after Task 3 (positional-compatible via keyword args).

No inconsistencies found.
