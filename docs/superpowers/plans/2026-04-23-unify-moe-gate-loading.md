# Unify MoE Gate Loading via `self._linear` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the ~9-line manual `Linear(...)` construction used for MoE gate/router loading across all 4 MoE specs with the existing `self._linear(pfx)` helper on `TextModelSpec`. No behavior change.

**Architecture:** `build_linear` already documents the exact case ("fallback to `TRIVIAL_FORMAT` when the expected quantized tensors are absent, e.g. a gate/router linear in a quantized model"). `_normalize_trivial` does the `.t()` for 2D tensors. The manual block is redundant; every call site collapses to one `m.add_gate(...)` line.

**Tech Stack:** Python, PyTorch, TurboMind deploy pipeline

---

### Task 1: Refactor MoE gate loading in all 4 spec files

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py:17,193-198`
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py:12,236-247`
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py:16,198-209`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py:17,269-282`

- [ ] **Step 1: Update `qwen3_spec.py`**

Imports — drop `TRIVIAL_FORMAT` and `Linear` (no longer referenced in this file after the gate block is removed):

Line 16 before:
```python
from ..kind_map import TRIVIAL_FORMAT
```

Line 17 before:
```python
from ..linear import Linear
```

After: delete both import lines entirely.

Body — replace the gate-wrapping block in `moe()`. Lines 193-198 before:
```python
        gate_w = self._get(f'{pfx}.gate.weight')
        gate_w = gate_w.t() if gate_w.dim() > 1 else gate_w
        m.add_gate('gate', Linear({'weight': gate_w},
                                  weight_format=TRIVIAL_FORMAT,
                                  data_format=TRIVIAL_FORMAT.make_data_format(self._cpp_dtype())),
                   model_dtype=self._cpp_dtype())
```

After:
```python
        m.add_gate('gate', self._linear(f'{pfx}.gate'),
                   model_dtype=self._cpp_dtype())
```

- [ ] **Step 2: Update `glm4_moe_lite_spec.py`**

Imports — drop `TRIVIAL_FORMAT` and `Linear`:

Line 11 before:
```python
from ..kind_map import TRIVIAL_FORMAT
```

Line 12 before:
```python
from ..linear import Linear
```

After: delete both import lines entirely.

Body — replace the gate block in `moe()`. Lines 236-247 before:
```python
        dtype = self._cpp_dtype()
        gate_w = self._get(f'{pfx}.gate.weight')
        gate_w = gate_w.t() if gate_w.dim() > 1 else gate_w
        tensors = {'weight': gate_w}
        gate_bias = self._get(f'{pfx}.gate.bias')
        if gate_bias is not None:
            tensors['bias'] = gate_bias
        m.add_gate('gate', Linear(
            tensors,
            weight_format=TRIVIAL_FORMAT,
            data_format=TRIVIAL_FORMAT.make_data_format(self._cpp_dtype()),
        ), model_dtype=dtype)
```

After:
```python
        m.add_gate('gate', self._linear(f'{pfx}.gate'),
                   model_dtype=self._cpp_dtype())
```

The subsequent `m.add_param('score_correction_bias', correction)` line (originally ~249-250) stays unchanged — it is a separate MoE parameter, not a gate input.

- [ ] **Step 3: Update `gpt_oss_spec.py`**

Imports — drop `TRIVIAL_FORMAT` (keep `Linear` — still used by `_deinterleave`). Line 15 before:
```python
from ..kind_map import TRIVIAL_FORMAT, build_linear
```

After:
```python
from ..kind_map import build_linear
```

Body — replace the gate block in `moe()`. Lines 198-209 before:
```python
        dtype = self._cpp_dtype()
        gate_w = self._get(f'{pfx}.router.weight')
        gate_w = gate_w.t() if gate_w.dim() > 1 else gate_w
        tensors = {'weight': gate_w}
        gate_bias = self._get(f'{pfx}.router.bias')
        if gate_bias is not None:
            tensors['bias'] = gate_bias
        m.add_gate('gate', Linear(
            tensors,
            weight_format=TRIVIAL_FORMAT,
            data_format=TRIVIAL_FORMAT.make_data_format(self._cpp_dtype()),
        ), model_dtype=dtype)
```

After:
```python
        m.add_gate('gate', self._linear(f'{pfx}.router'),
                   model_dtype=self._cpp_dtype())
```

Note the prefix is `{pfx}.router` (not `.gate`) — gpt-oss uses HF's `router.weight` / `router.bias` naming. `self._linear` will pick up both via the trivial suffix map.

- [ ] **Step 4: Update `qwen3_5_spec.py`**

Imports — drop `TRIVIAL_FORMAT` (keep `Linear` — still used by `_packed_moe_expert_indexed`). Line 16 before:
```python
from ..kind_map import TRIVIAL_FORMAT, build_linear
```

After:
```python
from ..kind_map import build_linear
```

Body — replace the two gate blocks in `moe()`. Lines 269-282 before:
```python
        dtype = self._cpp_dtype()
        gate_w = self._get(f'{pfx}.gate.weight')
        gate_w = gate_w.t() if gate_w.dim() > 1 else gate_w
        m.add_gate('gate', Linear({'weight': gate_w},
                                  weight_format=TRIVIAL_FORMAT,
                                  data_format=TRIVIAL_FORMAT.make_data_format(self._cpp_dtype())),
                   model_dtype=dtype)

        sg = self._get(f'{pfx}.shared_expert_gate.weight')
        sg = sg.t() if sg.dim() > 1 else sg
        m.add_gate('shared_gate', Linear({'weight': sg},
                                         weight_format=TRIVIAL_FORMAT,
                                         data_format=TRIVIAL_FORMAT.make_data_format(self._cpp_dtype())),
                   model_dtype=dtype)
```

After:
```python
        m.add_gate('gate', self._linear(f'{pfx}.gate'),
                   model_dtype=self._cpp_dtype())

        m.add_gate('shared_gate', self._linear(f'{pfx}.shared_expert_gate'),
                   model_dtype=self._cpp_dtype())
```

- [ ] **Step 5: Verify no stale references remain**

Run:
```bash
rg -n 'TRIVIAL_FORMAT' lmdeploy/turbomind/deploy/source_model/
```

Expected: zero matches across the 4 spec files (`TRIVIAL_FORMAT` may still appear in other `source_model/` files — that's fine; this task only touches the 4 MoE specs).

Run:
```bash
rg -n '^from \.\.linear import Linear' lmdeploy/turbomind/deploy/source_model/qwen3_spec.py lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py
```

Expected: zero matches (import was removed from both files).

Run:
```bash
rg -n 'add_gate\(' lmdeploy/turbomind/deploy/source_model/
```

Expected: 5 matches — one in each of `qwen3_spec.py`, `glm4_moe_lite_spec.py`, `gpt_oss_spec.py`, and two in `qwen3_5_spec.py`.

- [ ] **Step 6: Sanity-check the 4 files import correctly**

```bash
python -c "
from lmdeploy.turbomind.deploy.source_model import qwen3_spec, qwen3_5_spec, glm4_moe_lite_spec, gpt_oss_spec
print('OK')
"
```

Expected output: `OK`. A `NameError` here (e.g. `Linear` or `TRIVIAL_FORMAT` still referenced) means an edit was incomplete — go back and fix.

- [ ] **Step 7: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/qwen3_spec.py \
        lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py \
        lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py \
        lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py
git commit -m "deploy: load MoE gate via self._linear instead of manual Linear wrapping"
```

---

### Task 2: Smoke-test every affected spec

**Files:** None (verification only)

There are no unit tests for individual specs in this codebase; verification is end-to-end model inference through `scripts/test_turbomind_model.py`. Every MoE spec must be exercised because each one has a distinct `moe()` path (prefix, bias presence, number of gates).

- [ ] **Step 1: Check GPU availability**

Use the `get_gpu_usage` MCP tool to find an empty GPU. If none are available, wait or skip to a quieter time.

- [ ] **Step 2: Pick one model per spec from the model registry**

Use the `list_models` MCP tool to find one cached model per spec:

| Spec file | Matching `model_type` (in `config.json`) |
|---|---|
| `qwen3_spec.py` | `qwen3` or `qwen3_moe` |
| `qwen3_5_spec.py` | `qwen3_next_text` / `qwen3_5_moe` (whichever is registered under `qwen3_5`) |
| `glm4_moe_lite_spec.py` | `glm4_moe` (e.g. GLM-4.7-Flash) |
| `gpt_oss_spec.py` | `gpt_oss` |

If multiple options exist per spec, prefer the smallest variant that fits on the available GPU. **At least one of the selected models must be a quantized checkpoint** (AWQ / GPTQ / FP8 / MXFP4) to cover the `TRIVIAL_FORMAT`-fallback path for gates in a quantized model.

- [ ] **Step 3: Run one model test per spec**

For each selected model, run:
```bash
python scripts/test_turbomind_model.py <model_id> --tp 1
```

Verify for each run:
- The conversion completes without errors.
- The model responds with meaningful text (not gibberish) to a test prompt.
- Response length is at least 128 tokens.

Any gibberish, truncation, or crash on any of the 4 models means a regression — return to Task 1 and debug before continuing.

- [ ] **Step 4: Commit (only if a fix was needed)**

If no regressions: no commit needed for this task.

If a fix was required during testing, commit it now:
```bash
git add <files-that-were-fixed>
git commit -m "deploy: fix <brief description of issue found in Task 2>"
```
