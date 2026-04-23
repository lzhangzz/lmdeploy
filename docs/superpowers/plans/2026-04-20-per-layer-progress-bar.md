# Per-Layer Progress Bar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore a meaningful tqdm progress bar during Turbomind weight conversion by ticking once per decoder layer inside each spec's `layers()` loop, driven by a shared `layer_progress()` helper.

**Architecture:** Add a free function `layer_progress(num_layers)` in `source_model/utils.py` that returns `tqdm(range(num_layers), desc='Loading', leave=False)`. Each of the 4 existing `TextModelSpec` subclasses wraps its `range(self._num_layer)` loop with this helper. `BaseOutputModel.export()` drops its dead `total=1` pbar since per-layer reporting now lives in the spec.

**Tech Stack:** Python 3, tqdm, existing `lmdeploy/turbomind/deploy/` module tree.

**Spec:** `docs/superpowers/specs/2026-04-20-per-layer-progress-bar-design.md`

**Note on tests:** Per the spec, this change is cosmetic terminal output — no new automated tests. Verification is manual: run a conversion and observe the bar. The final task handles this.

---

## Task 1: Add `layer_progress` helper to `source_model/utils.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/utils.py` (append at end, after line 267)

- [ ] **Step 1: Append the helper function**

Open `lmdeploy/turbomind/deploy/source_model/utils.py`. The file currently ends at line 267 with the closing of `detect_layer_prefix`:

```python
    return ('model.layers',
            'model.embed_tokens.weight',
            'model.norm.weight')
```

Append a blank line and the new helper after it:

```python


def layer_progress(num_layers: int):
    """tqdm iterable for spec.layers() per-layer conversion loops.

    Yields the layer indices 0..num_layers-1, displaying a single-line
    progress bar on stderr. ``leave=False`` clears the bar when the loop
    completes. Lazy-imports tqdm so importing utils.py stays cheap.
    """
    from tqdm import tqdm
    return tqdm(range(num_layers), desc='Loading', leave=False)
```

- [ ] **Step 2: Smoke-test the import**

Run: `python -c "from lmdeploy.turbomind.deploy.source_model.utils import layer_progress; list(layer_progress(3))"`
Expected: runs without error; a brief `Loading: 3/3` bar may flash on stderr.

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/utils.py
git commit -m "feat(deploy): add layer_progress helper for per-layer tqdm bar

Free function in source_model/utils.py that returns a tqdm-wrapped
range(num_layers) with desc='Loading' and leave=False. Specs will use this
in their layers() loop to report per-decoder-layer progress during
turbomind conversion."
```

---

## Task 2: Remove dead pbar from `BaseOutputModel.export()`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/target_model/base.py:57-68`

- [ ] **Step 1: Replace `export()` body**

Current (lines 57-68):

```python
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
```

Replace with:

```python
    def export(self) -> None:
        import torch
        from ..loader import create_loader
        loader = create_loader(self.model_path, self.spec._layer_pattern,
                               self.spec._loader_mappings)
        self.spec.set_params(loader.all_items())
        self.spec.model()
        torch.cuda.empty_cache()
```

The dropped lines are:
- `from tqdm import tqdm`
- `pbar = tqdm(total=1, desc='Convert to turbomind format', leave=False)`
- `pbar.update(1)`
- `pbar.close()`

`export_iter()` (lines 70-79) is unchanged.

- [ ] **Step 2: Verify the file parses**

Run: `python -c "from lmdeploy.turbomind.deploy.target_model.base import BaseOutputModel"`
Expected: import succeeds without error.

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/target_model/base.py
git commit -m "refactor(turbomind): drop dead total=1 pbar from export()

The pbar became a no-op after the batched-params refactor: total=1 with
a single update(1) at the end shows 0/1 then 1/1. Per-layer progress now
lives in each spec's layers() loop via layer_progress()."
```

---

## Task 3: Wire `layer_progress` into all 4 specs

Four mechanical edits in one task — same pattern, identical diff shape.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` (import line 19-20, loop at line 209)
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` (import line 19-20, loop at line 329)
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` (import line 19, loop at line 213)
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` (import line 14, loop at line 260)

### Step 1: Edit `qwen3_spec.py`

- [ ] Replace the import on lines 19-20:

Current:

```python
from .utils import (_pad_inter_size, reorder_rotary_emb,
                    reorder_rotary_emb_linear)
```

Replace with:

```python
from .utils import (_pad_inter_size, layer_progress, reorder_rotary_emb,
                    reorder_rotary_emb_linear)
```

- [ ] Replace the loop header at line 209:

Current:

```python
        for i in range(self._num_layer):
```

Replace with:

```python
        for i in layer_progress(self._num_layer):
```

### Step 2: Edit `qwen3_5_spec.py`

- [ ] Replace the import on lines 19-20:

Current:

```python
from .utils import (_pad_inter_size, reorder_rotary_emb,
                    reorder_rotary_emb_linear, rope_type_to_int)
```

Replace with:

```python
from .utils import (_pad_inter_size, layer_progress, reorder_rotary_emb,
                    reorder_rotary_emb_linear, rope_type_to_int)
```

- [ ] Replace the loop header at line 329:

Current:

```python
        for i in range(self._num_layer):
```

Replace with:

```python
        for i in layer_progress(self._num_layer):
```

### Step 3: Edit `gpt_oss_spec.py`

- [ ] Replace the import on line 19:

Current:

```python
from .utils import _pad_inter_size, reorder_rotary_emb_linear
```

Replace with:

```python
from .utils import _pad_inter_size, layer_progress, reorder_rotary_emb_linear
```

- [ ] Replace the loop header at line 213:

Current:

```python
        for i in range(self._num_layer):
```

Replace with:

```python
        for i in layer_progress(self._num_layer):
```

### Step 4: Edit `glm4_moe_lite_spec.py`

- [ ] Replace the import on line 14:

Current:

```python
from .utils import _pad_inter_size, _pad_kv_head, get_yarn_params, parse_rope_param, rope_type_to_int
```

Replace with:

```python
from .utils import _pad_inter_size, _pad_kv_head, get_yarn_params, layer_progress, parse_rope_param, rope_type_to_int
```

- [ ] Replace the loop header at line 260:

Current:

```python
        for i in range(self._num_layer):
```

Replace with:

```python
        for i in layer_progress(self._num_layer):
```

### Step 5: Verify all 4 specs parse

- [ ] Run:

```bash
python -c "
import lmdeploy.turbomind.deploy.source_model.qwen3_spec
import lmdeploy.turbomind.deploy.source_model.qwen3_5_spec
import lmdeploy.turbomind.deploy.source_model.gpt_oss_spec
import lmdeploy.turbomind.deploy.source_model.glm4_moe_lite_spec
print('ok')
"
```

Expected: prints `ok` (all modules import cleanly, `layer_progress` resolves in each).

### Step 6: Sanity-check — no stale `range(self._num_layer)` left in spec `layers()` methods

- [ ] Run:

```bash
rg 'for i in range\(self\._num_layer\)' lmdeploy/turbomind/deploy/source_model/
```

Expected: no matches (all 4 occurrences replaced). If any match remains, fix it before committing.

### Step 7: Commit

- [ ] Run:

```bash
git add lmdeploy/turbomind/deploy/source_model/qwen3_spec.py \
        lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py \
        lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py \
        lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py
git commit -m "feat(deploy): per-layer tqdm progress in spec.layers()

Wrap the range(self._num_layer) loop in each of the 4 text-model specs
(qwen3, qwen3_5, gpt_oss, glm4_moe_lite) with the shared layer_progress()
helper. Users now see 'Loading: k/N' updating per decoder layer during
turbomind conversion, instead of the broken 0/1->1/1 stub."
```

---

## Task 4: Manual verification on a real conversion

**Files:** none modified.

- [ ] **Step 1: Pick a small supported model**

Any locally-cached HF checkpoint that one of the 4 specs handles will do — e.g. a Qwen3 dense model (smallest available). If none is cached, skip to step 4 (code-only confirmation).

- [ ] **Step 2: Run the conversion**

Run the standard conversion entry point the project uses (e.g. `lmdeploy convert <model>` or the `TurboMind._from_hf()` path exercised by existing scripts). The exact command depends on how the repo is normally driven — consult the project README or `my_generate.sh` if present.

Expected terminal output during the run:
- A single-line `Loading:` bar increments from `0/N` to `N/N`, where `N = num_hidden_layers` of the chosen model.
- Bar clears when the loop finishes (`leave=False`).
- No `Convert to turbomind format: 0/1` stub anywhere.

- [ ] **Step 3: Confirm no regression**

The conversion should complete successfully and the converted model should load and run exactly as before (behavioral parity — we only touched progress reporting).

- [ ] **Step 4: If step 2 cannot be run locally**

Verify statically:

```bash
rg 'layer_progress|range\(self\._num_layer\)|pbar' \
   lmdeploy/turbomind/deploy/source_model/ \
   lmdeploy/turbomind/deploy/target_model/base.py
```

Expected:
- `layer_progress` appears in `utils.py` (definition) and once per spec (call site): 5 total.
- `range(self._num_layer)` appears 0 times in spec files.
- `pbar` appears 0 times in `target_model/base.py`.

No commit in this task — verification only.

---

## Summary of Commits

| # | Subject | Files |
|---|---|---|
| 1 | `feat(deploy): add layer_progress helper for per-layer tqdm bar` | `source_model/utils.py` |
| 2 | `refactor(turbomind): drop dead total=1 pbar from export()` | `target_model/base.py` |
| 3 | `feat(deploy): per-layer tqdm progress in spec.layers()` | 4 spec files |

Three focused commits; each leaves the tree in a working state.
