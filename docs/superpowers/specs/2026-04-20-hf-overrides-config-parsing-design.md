# hf_overrides Support in Turbomind Config Parsing

**Date**: 2026-04-20
**Status**: Draft

## Problem

The turbomind deploy path only handles `rope_scaling` from `hf_overrides`, and applies it
post-hoc after the spec has already parsed the raw HF config. All other override keys are
silently dropped. In contrast, the pytorch path recursively merges all override keys into
the HF config before the model uses it.

## Design

Apply `hf_overrides` as a deep dict merge on the raw HF config dict **before** the spec
parses it, so all override keys work generically. Remove the post-hoc `hf_overrides`
handling from `update_from_engine_config`.

### Change 1: Deep merge helper in `converter.py`

A private recursive dict-merge function that mutates `base` in-place:

```python
def _deep_merge(base: dict, override: dict, path: str = '') -> dict:
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

Warns when an override key is not in the base config (catches typos) but applies it
regardless, since some overrides legitimately add fields absent from a particular model's
config.

### Change 2: Apply in `get_tm_config` before spec creation

In `converter.py:get_tm_config()`, after `hf_cfg = load_model_config(model_path)` and
before `spec = spec_cls(hf_cfg, ...)`:

```python
hf_cfg = load_model_config(model_path)

if engine_config.hf_overrides:
    logger.warning(f'Overriding HF config with {engine_config.hf_overrides}')
    _deep_merge(hf_cfg, engine_config.hf_overrides)

spec_cls = INPUT_MODELS.get(spec_name)
spec = spec_cls(hf_cfg, engine_config, group_size=group_size or 0)
```

### Change 3: Remove post-hoc `hf_overrides` block from `update_from_engine_config`

Delete lines 167-185 in `config.py` (the `hf_overrides` rope_scaling block). The
`rope_scaling_factor` legacy path remains untouched — it is a separate deprecated
mechanism.

## Scope

Three files touched:
- `lmdeploy/turbomind/deploy/converter.py` — add `_deep_merge`, apply in `get_tm_config`
- `lmdeploy/turbomind/deploy/config.py` — remove `hf_overrides` block from
  `update_from_engine_config`

No new public API. No new modules.

## Testing

Verify with `scripts/test_turbomind_model.py` on a model that uses `hf_overrides` with
`rope_scaling` (e.g., yarn or dynamic ntk configs). Confirm the RoPE parameters in the
output YAML match the overridden values.
