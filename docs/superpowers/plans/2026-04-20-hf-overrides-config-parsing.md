# hf_overrides Config Parsing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply `hf_overrides` as a deep dict merge on the raw HF config before the spec parses it, giving full parity with the pytorch path.

**Architecture:** A private `_deep_merge` helper in `converter.py` merges `engine_config.hf_overrides` into the raw HF config dict between `load_model_config()` and spec construction. The post-hoc `hf_overrides` block in `update_from_engine_config` is removed.

**Tech Stack:** Python, pytest

---

### Task 1: Add `_deep_merge` helper and test

**Files:**
- Modify: `lmdeploy/turbomind/deploy/converter.py:1-16` (add import + helper)
- Create: `tests/test_lmdeploy/test_converter.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_lmdeploy/test_converter.py`:

```python
from lmdeploy.turbomind.deploy.converter import _deep_merge


class TestDeepMerge:

    def test_flat_override(self):
        base = {'a': 1, 'b': 2}
        _deep_merge(base, {'b': 99})
        assert base == {'a': 1, 'b': 99}

    def test_nested_override(self):
        base = {'rope_scaling': {'rope_type': 'default', 'factor': 1.0}}
        _deep_merge(base, {'rope_scaling': {'factor': 4.0}})
        assert base == {'rope_scaling': {'rope_type': 'default', 'factor': 4.0}}

    def test_new_key_warns(self, caplog):
        base = {'a': 1}
        _deep_merge(base, {'nonexistent_key': 'val'})
        assert base['nonexistent_key'] == 'val'
        assert 'nonexistent_key' in caplog.text

    def test_nested_new_key_warns(self, caplog):
        base = {'rope_scaling': {'factor': 1.0}}
        _deep_merge(base, {'rope_scaling': {'brand_new': 'yes'}})
        assert base['rope_scaling']['brand_new'] == 'yes'
        assert 'brand_new' in caplog.text

    def test_empty_override_is_noop(self):
        base = {'a': 1}
        _deep_merge(base, {})
        assert base == {'a': 1}

    def test_scalar_overrides_dict(self):
        base = {'a': {'nested': 1}}
        _deep_merge(base, {'a': 'flat'})
        assert base == {'a': 'flat'}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /data/lmdeploy-modeling && python -m pytest tests/test_lmdeploy/test_converter.py -v`
Expected: FAIL with `ImportError: cannot import name '_deep_merge'`

- [ ] **Step 3: Write minimal implementation**

In `lmdeploy/turbomind/deploy/converter.py`, add the helper after the imports (after line 16). Also add `logging` import for `caplog` compatibility — the existing `logger` from `get_logger('lmdeploy')` already uses the `lmdeploy` namespace, which pytest's `caplog` captures by default with `caplog.set_level` or `caplog.at_level`.

Add after line 16 (`logger = get_logger('lmdeploy')`):

```python
def _deep_merge(base: dict, override: dict, path: str = '') -> dict:
    """Recursively merge override into base, mutating base in-place."""
    for k, v in override.items():
        key_path = f'{path}.{k}' if path else k
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v, key_path)
        else:
            if k not in base:
                logger.warning(f'hf_overrides key "{key_path}" not found in config, applying anyway')
            base[k] = v
    return base
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /data/lmdeploy-modeling && python -m pytest tests/test_lmdeploy/test_converter.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_lmdeploy/test_converter.py lmdeploy/turbomind/deploy/converter.py
git commit -m "feat(turbomind): add _deep_merge helper for hf_overrides"
```

---

### Task 2: Apply `hf_overrides` in `get_tm_config` before spec creation

**Files:**
- Modify: `lmdeploy/turbomind/deploy/converter.py:201-203`

- [ ] **Step 1: Apply the merge in `get_tm_config`**

Replace lines 201-203 in `converter.py`:

```python
    hf_cfg = load_model_config(model_path)
    spec_cls = INPUT_MODELS.get(spec_name)
    spec = spec_cls(hf_cfg, engine_config, group_size=group_size or 0)
```

With:

```python
    hf_cfg = load_model_config(model_path)
    if engine_config.hf_overrides:
        logger.warning(f'Overriding HF config with {engine_config.hf_overrides}')
        _deep_merge(hf_cfg, engine_config.hf_overrides)
    spec_cls = INPUT_MODELS.get(spec_name)
    spec = spec_cls(hf_cfg, engine_config, group_size=group_size or 0)
```

- [ ] **Step 2: Verify no syntax errors**

Run: `cd /data/lmdeploy-modeling && python -c "from lmdeploy.turbomind.deploy.converter import get_tm_config; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/converter.py
git commit -m "feat(turbomind): apply hf_overrides before spec creation"
```

---

### Task 3: Remove post-hoc `hf_overrides` block from `update_from_engine_config`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/config.py:167-185`

- [ ] **Step 1: Delete the `hf_overrides` block**

In `config.py`, delete lines 167-185 (from `# update from hf_overrides` through `logger.warning(f'Overriding HF config with {hf_overrides}')`). The result should go straight from the field-copying loop to the legacy `rope_scaling_factor` block:

```python
            if hasattr(self.attention_config, key):
                setattr(self.attention_config, key, value)

        # use dynamic ntk
        if config.rope_scaling_factor:
```

- [ ] **Step 2: Verify no syntax errors**

Run: `cd /data/lmdeploy-modeling && python -c "from lmdeploy.turbomind.deploy.config import TurbomindModelConfig; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/config.py
git commit -m "refactor(turbomind): remove post-hoc hf_overrides block from update_from_engine_config"
```

---

### Task 4: Integration test with a real model

**Files:**
- None (manual verification)

- [ ] **Step 1: Check GPU availability**

Run: `cd /data/lmdeploy-modeling && python -c "from lmdeploy.turbomind.deploy.converter import get_tm_config; print('import OK')"`

Also check GPU is free using the `get_gpu_usage` MCP tool.

- [ ] **Step 2: Test with `hf_overrides` containing `rope_scaling`**

Pick a locally cached model (use `list_models` MCP tool) and run:

```bash
cd /data/lmdeploy-modeling && python scripts/test_turbomind_model.py <model_id> --hf-overrides '{"rope_scaling": {"type": "dynamic", "factor": 2.0}}'
```

Verify the model produces meaningful output (not gibberish) with at least 128 tokens.

- [ ] **Step 3: Test with `hf_overrides` containing a generic key**

```bash
cd /data/lmdeploy-modeling && python scripts/test_turbomind_model.py <model_id> --hf-overrides '{"max_position_embeddings": 8192}'
```

Verify the model starts and produces meaningful output. Check the log for the "Overriding HF config" warning.

- [ ] **Step 4: Test without `hf_overrides` (regression)**

```bash
cd /data/lmdeploy-modeling && python scripts/test_turbomind_model.py <model_id>
```

Verify the model works exactly as before — no regressions.
