# Pipeline Core Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three targeted cleanups in the turbomind deploy pipeline: delete `module.py` re-export facade, simplify `_process_moe` dead code, rename `configs.py` to `module_configs.py`.

**Architecture:** Pure refactor — no behavioral changes. Each task is independent and produces a working codebase.

**Tech Stack:** Python, git

---

### Task 1: Rename `configs.py` to `module_configs.py`

**Files:**
- Rename: `lmdeploy/turbomind/deploy/configs.py` → `lmdeploy/turbomind/deploy/module_configs.py`
- Modify: `lmdeploy/turbomind/deploy/text_model_loader.py:7`
- Modify: `lmdeploy/turbomind/deploy/spec.py:8`

- [ ] **Step 1: Rename the file**

```bash
git mv lmdeploy/turbomind/deploy/configs.py lmdeploy/turbomind/deploy/module_configs.py
```

- [ ] **Step 2: Update import in `text_model_loader.py`**

Line 7, change:
```python
from .configs import (
```
to:
```python
from .module_configs import (
```

- [ ] **Step 3: Update import in `spec.py`**

Line 8, change:
```python
from .configs import SpecAttnConfig
```
to:
```python
from .module_configs import SpecAttnConfig
```

- [ ] **Step 4: Verify nothing else imports configs**

Run:
```bash
grep -rn 'from.*\.configs import\|from.*configs import' lmdeploy/
```
Expected: no remaining references to `.configs` (except possibly in unrelated files like `target_model/` — check each).

- [ ] **Step 5: Build and verify imports resolve**

```bash
cd /data/lmdeploy-modeling/build && ninja
```

Then test import:
```bash
PYTHONPATH=/data/lmdeploy-modeling/lmdeploy:/data/lmdeploy-modeling/build/lib python -c "from lmdeploy.turbomind.deploy.text_model_loader import TextModelLoader; print('OK')"
```

- [ ] **Step 6: Commit**

```bash
git add lmdeploy/turbomind/deploy/module_configs.py lmdeploy/turbomind/deploy/text_model_loader.py lmdeploy/turbomind/deploy/spec.py
git commit -m "refactor(deploy): rename configs.py to module_configs.py

Disambiguates from config.py (top-level ModelConfig)."
```

---

### Task 2: Simplify `_process_moe` in `text_model_loader.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/text_model_loader.py:202-215`

- [ ] **Step 1: Replace the dead dot-navigation loop**

In `text_model_loader.py`, replace lines 202-215 (the comment and the loop):

```python
        # --- Non-expert MoE parameters (gate/shared_gate weights, score_correction_bias) ---
        for name, (tensor, split_side) in spec.moe_params(layer).items():
            parts = name.split('.')
            parent = moe
            for seg in parts[:-1]:
                existing = parent._handles[0].child(seg) if parent._handles else None
                if existing is not None:
                    children = [h.child(seg) for h in parent._handles]
                    parent = Distributor(children, moe._contexts)
                else:
                    parent = parent.create_child(seg, NormConfig(
                        dim=tensor.shape[-1] if tensor.dim() >= 1 else 0,
                        data_type=dtype))
            parent.commit_tensor(parts[-1], tensor, split_side=split_side)
```

with:

```python
        # --- Non-expert MoE parameters (gate/shared_gate weights, score_correction_bias) ---
        for name, (tensor, split_side) in spec.moe_params(layer).items():
            moe.commit_tensor(name, tensor, split_side=split_side)
```

This works because C++ `Module::param(name)` resolves dotted names (e.g. `"gate.weight"`) through the child tree. The `gate` and `shared_gate` children are already created above (lines 195, 200). Flat names like `"score_correction_bias"` resolve directly on `moe`.

- [ ] **Step 2: Check if `NormConfig` import is still needed**

After the edit, check if `NormConfig` (from the configs import at line 7-10) is still used elsewhere in `text_model_loader.py`. If not, remove it from the import. Search for other uses:

```bash
grep -n 'NormConfig' lmdeploy/turbomind/deploy/text_model_loader.py
```

If no other uses, remove `NormConfig` from the import on line 9.

- [ ] **Step 3: Verify import resolves**

```bash
PYTHONPATH=/data/lmdeploy-modeling/lmdeploy:/data/lmdeploy-modeling/build/lib python -c "from lmdeploy.turbomind.deploy.text_model_loader import TextModelLoader; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/text_model_loader.py
git commit -m "refactor(deploy): simplify _process_moe param commit loop

The dot-navigation loop was dead code — gate and shared_gate children
are always pre-created, and C++ Module::param resolves dotted names."
```

---

### Task 3: Delete `module.py` and update all import sites

**Files:**
- Delete: `lmdeploy/turbomind/deploy/module.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py:12`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py:18`
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py:17`
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py:22`

- [ ] **Step 1: Update `qwen3_spec.py`**

Line 12, change:
```python
from ..module import TextModelSpec
```
to:
```python
from ..spec import TextModelSpec
```

- [ ] **Step 2: Update `qwen3_5_spec.py`**

Line 18, change:
```python
from ..module import TextModelSpec, SplitSide
```
to:
```python
from ..spec import TextModelSpec, SplitSide
```

- [ ] **Step 3: Update `glm4_moe_lite_spec.py`**

Line 17, change:
```python
from ..module import TextModelSpec
```
to:
```python
from ..spec import TextModelSpec
```

- [ ] **Step 4: Update `gpt_oss_spec.py`**

Line 22, change:
```python
from ..module import TextModelSpec, SplitSide
```
to:
```python
from ..spec import TextModelSpec, SplitSide
```

- [ ] **Step 5: Verify no remaining imports from `module`**

```bash
grep -rn 'from.*\.module import\|from.*module import' lmdeploy/turbomind/deploy/
```

Expected: no results. (Other `module` references like `create_child` or C++ Module are unrelated.)

- [ ] **Step 6: Delete `module.py`**

```bash
git rm lmdeploy/turbomind/deploy/module.py
```

- [ ] **Step 7: Verify imports resolve**

```bash
PYTHONPATH=/data/lmdeploy-modeling/lmdeploy:/data/lmdeploy-modeling/build/lib python -c "
from lmdeploy.turbomind.deploy.source_model.qwen3_spec import Qwen3TextSpec
from lmdeploy.turbomind.deploy.source_model.qwen3_5_spec import Qwen3_5Spec
from lmdeploy.turbomind.deploy.source_model.glm4_moe_lite_spec import Glm4MoeLiteSpec
from lmdeploy.turbomind.deploy.source_model.gpt_oss_spec import GptOssSpec
print('OK')
"
```

- [ ] **Step 8: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/qwen3_spec.py lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py
git commit -m "refactor(deploy): delete module.py re-export facade

Update all source model specs to import directly from spec.py.
module.py was a migration artifact that re-exported from spec.py,
transforms.py, and commit.py for backward compatibility."
```
