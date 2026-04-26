# ModelLoader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `BaseOutputModel` + `TextModelLoader` + `TurbomindModel` with a single `ModelLoader` class.

**Architecture:** A new `ModelLoader` class in `deploy/model_loader.py` absorbs the handle extraction from `TextModelLoader` and the export logic from `BaseOutputModel`. `TurboMind._from_hf()` creates `ModelLoader` directly instead of going through the `OUTPUT_MODELS` registry. Three files and the `target_model/` directory are deleted.

**Tech Stack:** Python, existing TurboMind C++ bindings.

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `deploy/model_loader.py` | Single class: holds spec + model_comm, binds runtime, provides export/export_iter |
| Modify | `turbomind.py:174,213-271,306-313` | Use ModelLoader instead of OUTPUT_MODELS/TurbomindModel |
| Modify | `deploy/spec.py:111` | Update comment (TextModelLoader → ModelLoader) |
| Delete | `deploy/text_model_loader.py` | Absorbed by ModelLoader._bind_runtime() |
| Delete | `deploy/target_model/base.py` | Absorbed by ModelLoader |
| Delete | `deploy/target_model/fp.py` | Vacuous subclass, no longer needed |
| Delete | `deploy/target_model/__init__.py` | Directory removed |
| Delete | `deploy/target_model/` | Empty directory |

---

### Task 1: Create ModelLoader

**Files:**
- Create: `lmdeploy/turbomind/deploy/model_loader.py`

- [ ] **Step 1: Create `model_loader.py`**

```python
# Copyright (c) OpenMMLab. All rights reserved.
"""ModelLoader: coordinates loading a spec's weights into the TurboMind runtime."""
import torch

from .loader import create_loader


class ModelLoader:
    """Coordinates loading a spec's weights into the TurboMind runtime.

    Holds the spec, model_comm handle, and model_path. Extracts GPU topology
    handles from model_comm and binds them onto the spec at construction time.
    Provides export() and export_iter() to load checkpoint weights and commit
    them to the C++ runtime.
    """

    def __init__(self, spec, model_comm, gpu_count, model_path):
        self.spec = spec
        self.model_comm = model_comm
        self.gpu_count = gpu_count
        self.model_path = model_path
        self._bind_runtime()

    def _bind_runtime(self):
        mc = self.model_comm
        attn_ranks = [mc.attn_tp_rank(g) for g in range(self.gpu_count)]
        mlp_ranks = [mc.mlp_tp_rank(g) for g in range(self.gpu_count)]
        model_tp = [mc.model_tp_rank(g) for g in range(self.gpu_count)]
        contexts = [mc.context(g) for g in range(self.gpu_count)]
        handles = [mc.root(g) for g in range(self.gpu_count)]
        self.spec.bind_runtime(
            contexts=contexts,
            root_handles=handles,
            attn_ranks=attn_ranks,
            mlp_ranks=mlp_ranks,
            model_tp_ranks=model_tp,
        )

    def export(self):
        loader = create_loader(self.model_path, self.spec._layer_pattern,
                               self.spec._loader_mappings)
        self.spec.set_params(loader.all_items())
        self.spec.model()
        torch.cuda.empty_cache()

    def export_iter(self):
        loader = create_loader(self.model_path, self.spec._layer_pattern,
                               self.spec._loader_mappings)
        self.spec.set_params(loader.all_items())
        self.spec.model()
        yield -1
        torch.cuda.empty_cache()
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/model_loader.py
git commit -m "feat: add ModelLoader to replace BaseOutputModel + TextModelLoader"
```

---

### Task 2: Update TurboMind to use ModelLoader

**Files:**
- Modify: `lmdeploy/turbomind/turbomind.py:174,213-271,306-313`

- [ ] **Step 1: Update `_from_hf()` — change import and construction**

Replace the import on line 220 and the construction on lines 266-270.

Old (lines 219-220):
```python
        from .deploy.converter import get_tm_config
        from .deploy.target_model.base import OUTPUT_MODELS
```

New:
```python
        from .deploy.converter import get_tm_config
        from .deploy.model_loader import ModelLoader
```

Old (lines 266-270):
```python
        self._tm_model = OUTPUT_MODELS.get('tm')(
            spec=spec,
            model_comm=model_comm,
            gpu_count=self.gpu_count,
            model_path=model_path)
```

New:
```python
        self._model_loader = ModelLoader(
            spec=spec,
            model_comm=model_comm,
            gpu_count=self.gpu_count,
            model_path=model_path)
```

- [ ] **Step 2: Update `_load_weights()` — change method call**

Old (line 177):
```python
            self._tm_model.export()
```

New:
```python
            self._model_loader.export()
```

- [ ] **Step 3: Update `update_params()` — change attribute name**

Old (lines 307-313):
```python
            que = Queue()
            tm_model = self._tm_model
            # update_params replaces the on-disk checkpoint source with a Queue; the
            # OutputModel now owns model_path directly (input_model was removed).
            tm_model.model_path = que
            self._update_params_que = que
            self._export_iter = tm_model.export_iter()
```

New:
```python
            que = Queue()
            ml = self._model_loader
            ml.model_path = que
            self._update_params_que = que
            self._export_iter = ml.export_iter()
```

- [ ] **Step 4: Build to verify no import/syntax errors**

Run: `cd /data/lmdeploy-modeling/build && ninja`

Expected: clean build (no new errors).

- [ ] **Step 5: Test with a model**

Run: `python scripts/test_turbomind_model.py <model_id> --prompt "Hello, how are you today?" --max_new_tokens 128`

Expected: meaningful response, no errors.

- [ ] **Step 6: Commit**

```bash
git add lmdeploy/turbomind/turbomind.py
git commit -m "refactor: use ModelLoader instead of OUTPUT_MODELS registry"
```

---

### Task 3: Update spec.py comment and delete old files

**Files:**
- Modify: `lmdeploy/turbomind/deploy/spec.py:111`
- Delete: `lmdeploy/turbomind/deploy/text_model_loader.py`
- Delete: `lmdeploy/turbomind/deploy/target_model/__init__.py`
- Delete: `lmdeploy/turbomind/deploy/target_model/fp.py`
- Delete: `lmdeploy/turbomind/deploy/target_model/base.py`
- Delete: `lmdeploy/turbomind/deploy/target_model/` (directory)

- [ ] **Step 1: Update comment in spec.py**

Old (line 111):
```python
    # Runtime binding (called by TextModelLoader after model_comm exists)
```

New:
```python
    # Runtime binding (called by ModelLoader after model_comm exists)
```

- [ ] **Step 2: Delete old files**

```bash
rm lmdeploy/turbomind/deploy/text_model_loader.py
rm lmdeploy/turbomind/deploy/target_model/__init__.py
rm lmdeploy/turbomind/deploy/target_model/fp.py
rm lmdeploy/turbomind/deploy/target_model/base.py
rmdir lmdeploy/turbomind/deploy/target_model/
```

- [ ] **Step 3: Verify nothing else references the deleted files**

Run: `grep -rn "TextModelLoader\|BaseOutputModel\|OUTPUT_MODELS\|target_model\." --include="*.py" lmdeploy/turbomind/ | grep -v __pycache__`

Expected: no results (or only the updated spec.py comment).

- [ ] **Step 4: Test with a model**

Run: `python scripts/test_turbomind_model.py <model_id> --prompt "Hello, how are you today?" --max_new_tokens 128`

Expected: meaningful response, no errors.

- [ ] **Step 5: Commit**

```bash
git add -A lmdeploy/turbomind/deploy/
git commit -m "refactor: remove TextModelLoader, BaseOutputModel, TurbomindModel"
```
