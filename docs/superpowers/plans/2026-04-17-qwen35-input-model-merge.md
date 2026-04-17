# Merge Qwen3.5 Input Model Classes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse `Qwen3_5InputModel` and `Qwen3_5MoeInputModel` into a single class with stacked registrations, mirroring the `qwen3`/`qwen3-moe` pattern.

**Architecture:** Single-file, single-edit refactor in `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py`. Replaces lines 366-411 (two near-duplicate classes) with one class that registers both names, always applies `_loader_mappings`, and gates MoE fields on `num_experts > 0` inside `model_info()`.

**Tech Stack:** Python, mmengine Registry, TurboMind deploy source-model framework.

**Spec:** `docs/superpowers/specs/2026-04-17-qwen35-input-model-merge-design.md`

---

## Task 1: Merge the two input-model classes

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py:366-411`

- [ ] **Step 1: Read the current file to confirm the target lines**

Run: open `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` and confirm lines 366-411 contain exactly the two classes `Qwen3_5InputModel` (line 367) and `Qwen3_5MoeInputModel` (line 391). If the line range has shifted, adjust the edit target — the replacement content below is unchanged.

- [ ] **Step 2: Replace lines 366-411 with the merged class**

Old content (lines 366-411):

```python
@INPUT_MODELS.register_module(name='qwen3_5')
class Qwen3_5InputModel(BaseInputModel):
    """Input model for Qwen3.5 (dense + optional linear attention)."""

    _layer_pattern = _LAYER_PATTERN
    _spec_class = Qwen3_5Spec

    def model_info(self) -> dict:
        cfg = self.model_config
        info = _qwen35_model_info_base(cfg)
        info.update(
            expert_num=cfg.get('num_experts', 0),
            expert_inter_size=cfg.get('moe_intermediate_size', 0),
            experts_per_token=cfg.get('num_experts_per_tok', 0),
            moe_shared_gate=True,
            scoring_func='softmax',
            norm_topk_prob=True,
        )
        shared_expert_size = cfg.get('shared_expert_intermediate_size')
        if shared_expert_size is not None:
            info['inter_size'] = shared_expert_size
        return info


@INPUT_MODELS.register_module(name='qwen3_5-moe')
class Qwen3_5MoeInputModel(BaseInputModel):
    """Input model for Qwen3.5-MoE."""

    _layer_pattern = _LAYER_PATTERN
    _spec_class = Qwen3_5Spec
    _loader_mappings = [map_packed_qwen35_experts]

    def model_info(self) -> dict:
        cfg = self.model_config
        info = _qwen35_model_info_base(cfg)
        info.update(
            expert_num=cfg.get('num_experts', 0),
            expert_inter_size=cfg.get('moe_intermediate_size', 0),
            experts_per_token=cfg.get('num_experts_per_tok', 0),
            inter_size=cfg.get('shared_expert_intermediate_size', 0),
            moe_shared_gate=True,
            scoring_func='softmax',
            norm_topk_prob=True,
        )
        return info
```

New content:

```python
@INPUT_MODELS.register_module(name='qwen3_5-moe')
@INPUT_MODELS.register_module(name='qwen3_5')
class Qwen3_5InputModel(BaseInputModel):
    """Input model for Qwen3.5 (dense and MoE)."""

    _layer_pattern = _LAYER_PATTERN
    _spec_class = Qwen3_5Spec
    _loader_mappings = [map_packed_qwen35_experts]

    def model_info(self) -> dict:
        cfg = self.model_config
        info = _qwen35_model_info_base(cfg)
        n_experts = cfg.get('num_experts', 0)
        if n_experts:
            info.update(
                expert_num=n_experts,
                expert_inter_size=cfg['moe_intermediate_size'],
                experts_per_token=cfg['num_experts_per_tok'],
                inter_size=cfg.get('shared_expert_intermediate_size', 0),
                moe_shared_gate=True,
                scoring_func='softmax',
                norm_topk_prob=True,
            )
        return info
```

- [ ] **Step 3: Verify the module still imports cleanly**

Run: `python -c "from lmdeploy.turbomind.deploy.source_model.qwen3_5_spec import Qwen3_5InputModel"`
Expected: exits 0 with no output. If it raises `ImportError` or `NameError`, the edit broke something — re-check imports and class body.

- [ ] **Step 4: Verify both registration names resolve to the merged class**

Run:

```bash
python -c "
from lmdeploy.turbomind.deploy.source_model.base import INPUT_MODELS
import lmdeploy.turbomind.deploy.source_model.qwen3_5_spec  # registers
dense = INPUT_MODELS.get('qwen3_5')
moe = INPUT_MODELS.get('qwen3_5-moe')
assert dense is moe, f'expected same class, got {dense} vs {moe}'
assert dense.__name__ == 'Qwen3_5InputModel'
assert dense._loader_mappings, 'loader mappings missing'
print('ok')
"
```

Expected: prints `ok`. If the two registrations resolve to different classes or `_loader_mappings` is empty, the edit is wrong.

- [ ] **Step 5: Lint check**

Run: `python -m pyflakes lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py`
Expected: no output (no warnings). A lingering reference to the removed `Qwen3_5MoeInputModel` name anywhere else would show up here or in the earlier import check.

- [ ] **Step 6: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py
git commit -m "$(cat <<'EOF'
refactor(source_model): merge Qwen3.5 dense and MoE input models

Collapse Qwen3_5InputModel and Qwen3_5MoeInputModel into a single class
with stacked registrations, matching the qwen3/qwen3-moe pattern. MoE
fields in model_info() are now gated on num_experts > 0;
map_packed_qwen35_experts applies to both registrations (no-op for
dense checkpoints).
EOF
)"
```

---

## Task 2: Verify dense Qwen3.5 deploy pipeline

**Files:** none modified.

- [ ] **Step 1: Dispatch the turbomind-tester agent**

Delegate to the `turbomind-tester` subagent (see `.cursor/agents/turbomind-tester.md`). Request: run `scripts/test_turbomind_model.py` against a dense Qwen3.5 checkpoint (non-MoE, e.g. a 4B-class model). Report pass/fail and any diagnostic output.

- [ ] **Step 2: Confirm pass**

Expected: agent reports the test passes. If it fails, inspect the diagnostic: the likely suspect is downstream code that reads `expert_num`, `moe_shared_gate`, `scoring_func`, or `norm_topk_prob` unconditionally for `qwen3_5`. Fix forward by adding the missing guard in the downstream consumer, not by reverting this change.

---

## Task 3: Verify Qwen3.5-MoE deploy pipeline

**Files:** none modified.

- [ ] **Step 1: Dispatch the turbomind-tester agent**

Delegate to the `turbomind-tester` subagent. Request: run `scripts/test_turbomind_model.py` against a Qwen3.5-MoE checkpoint (one that exercises the packed-experts path so `map_packed_qwen35_experts` is meaningfully applied). Report pass/fail and any diagnostic output.

- [ ] **Step 2: Confirm pass**

Expected: agent reports the test passes and `model_info()` output matches pre-change behavior except for the `moe_intermediate_size` / `num_experts_per_tok` lookups now being direct (which should not affect output for a valid Qwen3.5-MoE config).

---

## Self-Review

Spec coverage check:

- Design "Class shape" → Task 1 Step 2 contains the exact replacement code.
- Design decision 1 (`_loader_mappings` unconditional) → Task 1 Step 4 asserts `_loader_mappings` is present on the merged class.
- Design decision 2 (MoE fields gated on `num_experts > 0`) → present in Task 1 Step 2 replacement.
- Design decision 3 (direct lookups for `moe_intermediate_size` / `num_experts_per_tok`) → present in Task 1 Step 2 replacement.
- Design decision 4 (drop dense `shared_expert_intermediate_size` branch) → absent from Task 1 Step 2 replacement (as intended).
- Design decision 5 (decorator order matches `qwen3_spec.py`) → Task 1 Step 2 replacement has `-moe` outer, base inner.
- Spec "Testing" section → Tasks 2 and 3 cover dense and MoE integration runs.

Placeholder scan: no TBD/TODO, every code step contains actual code, every command has an expected result.

Type consistency: the single merged class references `Qwen3_5InputModel` consistently throughout; the registration names `qwen3_5` and `qwen3_5-moe` match the spec and the existing sibling `qwen3_spec.py` convention.

No issues found.
