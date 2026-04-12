# Pipeline Core Cleanup

Date: 2026-04-12

## Summary

Targeted architectural cleanup of the turbomind model loading pipeline core: remove a dead re-export facade, eliminate dead navigation code, and fix a naming collision.

## Scope

Pipeline core files only: `module.py`, `spec.py`, `transforms.py`, `commit.py`, `distributor.py`, `text_model_loader.py`, `configs.py`. Source model specs updated only for import changes.

## Changes

### 1. Delete `module.py`

`module.py` is a pure re-export facade importing everything from `spec.py`, `transforms.py`, and `commit.py`. This was a migration artifact that is no longer needed.

**Action:** Delete the file. Update all import sites to point directly at source modules:

| Current | New |
|---|---|
| `from ..module import TextModelSpec` | `from ..spec import TextModelSpec` |
| `from ..module import SplitSide` | `from ..spec import SplitSide` |
| `from .module import commit_linear, commit_tensor, SplitSide, ...` | `from .spec import SplitSide` / `from .commit import commit_linear, commit_tensor` / `from .transforms import fuse_ffn_linears, ...` |

### 2. Simplify `_process_moe` in `text_model_loader.py`

The current code uses a dot-navigation loop with a dead branch that checks for pre-existing children:

```python
for name, (tensor, split_side) in spec.moe_params(layer).items():
    parts = name.split('.')
    parent = moe
    for seg in parts[:-1]:
        existing = parent._handles[0].child(seg) if parent._handles else None  # always finds child
        if existing is not None:
            children = [h.child(seg) for h in parent._handles]
            parent = Distributor(children, moe._contexts)
        else:
            parent = parent.create_child(seg, NormConfig(...))  # dead code
    parent.commit_tensor(parts[-1], tensor, split_side=split_side)
```

The `existing` check always succeeds because `gate` and `shared_gate` children are created unconditionally before the loop. The `else` branch is dead code.

Furthermore, the dot-navigation is unnecessary. C++ `Module::param(name)` resolves dotted names through the child tree, so `commit_tensor(handle, tensor, "gate.weight")` works directly.

**Action:** Replace the entire loop body with:

```python
for name, (tensor, split_side) in spec.moe_params(layer).items():
    moe.commit_tensor(name, tensor, split_side=split_side)
```

### 3. Rename `configs.py` → `module_configs.py`

`config.py` (top-level `ModelConfig`/`TurbomindModelConfig`) and `configs.py` (per-module C++ config dataclasses) have nearly identical names but serve different purposes.

**Action:** Rename `configs.py` to `module_configs.py`. Update all import sites from `.configs` / `..configs` to `.module_configs` / `..module_configs`.

## Files affected

- `lmdeploy/turbomind/deploy/module.py` — deleted
- `lmdeploy/turbomind/deploy/configs.py` — renamed to `module_configs.py`
- `lmdeploy/turbomind/deploy/text_model_loader.py` — simplify `_process_moe`, update imports
- `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` — import update
- `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` — import update
- `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` — import update
- `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` — import update
