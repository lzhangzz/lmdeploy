# add_projections Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose `_fold_and_pad` / `_fold_and_pad_hf` (90-line combined monolith) into a pipeline of named standalone functions operating in TM layout, removing all defensive guards.

**Architecture:** Rewrite fold math in TM layout `[in, out]` directly (eliminating the transpose-to-HF-and-back dance). Create `fold_kv_b` (matmul-based absorption of kv_b into q_b and wo) and `pad_wo_input` (shape manipulation). Delete `_MLA_TP_RULES`, `_fold_and_pad`, `_fold_and_pad_hf`, and the `LMDEPLOY_MLA_FOLD` env var.

**Tech Stack:** Python, PyTorch.

---

### Task 1: Remove `LMDEPLOY_MLA_FOLD` env var from spec

The env var `LMDEPLOY_MLA_FOLD` in `glm4_moe_lite_spec.py` can disable the MLA fold at runtime. This is dead code — the fold always happens for GLM-4. Remove the env var, the `os` import, and simplify `model_info()`.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py`

- [ ] **Step 1: Remove `import os`**

Delete line 12:
```python
import os
```

- [ ] **Step 2: Simplify `model_info()` — remove env var and conditional branches**

Change lines 216–225 from:
```python
        size_per_head = q_head_dim
        v_head_dim = cfg['v_head_dim']
        softmax_scale = 0.0
        disable_mla_fold = os.getenv('LMDEPLOY_MLA_FOLD', '1').lower() in ('0', 'false', 'no')
        if kv_lora_rank and kv_lora_rank != qk_nope_dim and not disable_mla_fold:
            size_per_head = kv_lora_rank + qk_rope_dim
            v_head_dim = kv_lora_rank
            softmax_scale = q_head_dim**(-0.5)
        elif kv_lora_rank and kv_lora_rank != qk_nope_dim:
            softmax_scale = q_head_dim**(-0.5)
```
to:
```python
        size_per_head = q_head_dim
        v_head_dim = cfg['v_head_dim']
        softmax_scale = 0.0
        if kv_lora_rank and kv_lora_rank != qk_nope_dim:
            size_per_head = kv_lora_rank + qk_rope_dim
            v_head_dim = kv_lora_rank
            softmax_scale = q_head_dim**(-0.5)
```

- [ ] **Step 3: Verify no remaining references**

Run:
```bash
cd /data/lmdeploy-modeling && grep -rn "LMDEPLOY_MLA_FOLD\|disable_mla_fold" lmdeploy/
```
Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py
git commit -m "refactor(spec): remove LMDEPLOY_MLA_FOLD env var from GLM-4 spec"
```

---

### Task 2: Rewrite `add_projections` pipeline

Create `fold_kv_b` and `pad_wo_input` standalone functions. Rewrite `add_projections` to use the new pipeline. Rewrite `add_norms` to remove None guards. Delete `_MLA_TP_RULES`, `_fold_and_pad`, `_fold_and_pad_hf`.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/mla.py`

- [ ] **Step 1: Add `fold_qkv` function (before the `MLABuilder` class)**

Add this function after the existing imports and before the `MLABuilder` class. Replace the `_MLA_TP_RULES` dict and comment block (lines 9–18) with:

```python
# ---------------------------------------------------------------------------
# MLA fold+pad pipeline (standalone functions)
# ---------------------------------------------------------------------------


def fold_kv_b(q_b: Linear, kv_b: Linear, wo: Linear, *,
              cfg) -> tuple[Linear, Linear]:
    """Fold kv_b into q_b and wo. Returns (q_b_folded, wo_folded).

    Splits kv_b into key-compressed (kc) and value-compressed (vc) parts.
    Folds kc into q_b via matmul (q_nope @ kc^T per head).
    Folds vc into wo via matmul (vc @ wo per head).
    All arithmetic in TM layout [in, out].
    """
    head_num = cfg.head_num
    qk_rope_dim = cfg.qk_rope_dim
    size_per_head = cfg.head_dim

    q_b_w = q_b.tensors["weight"]
    kv_b_w = kv_b.tensors["weight"]
    o_w = wo.tensors["weight"]

    # Derive original dimensions from tensor shapes
    orig_q_head_dim = q_b_w.shape[-1] // head_num
    orig_qk_nope_dim = orig_q_head_dim - qk_rope_dim
    orig_v_head_dim = o_w.shape[0] // head_num

    # Split kv_b into kc and vc: [kv_lora_rank, head_num, dim]
    kv_b_h = kv_b_w.reshape(kv_b_w.shape[0], head_num, -1)
    kc = kv_b_h[:, :, :orig_qk_nope_dim]
    vc = kv_b_h[:, :, orig_qk_nope_dim:]

    # Fold kc into q_b: q_nope @ kc^T per head
    q_b_h = q_b_w.reshape(q_b_w.shape[0], head_num, orig_q_head_dim)
    q_nope = q_b_h[:, :, :orig_qk_nope_dim].permute(1, 0, 2)   # [H, R, P]
    q_rope = q_b_h[:, :, orig_qk_nope_dim:].permute(1, 0, 2)   # [H, R, S]
    kc_t = kc.permute(1, 2, 0)                                  # [H, P, R]
    q_expanded = torch.bmm(q_nope, kc_t)                        # [H, R, R]
    q_folded = torch.cat([q_expanded, q_rope], dim=-1)          # [H, R, sp]
    q_folded = q_folded.permute(1, 0, 2).reshape(
        q_b_w.shape[0], head_num * size_per_head)

    # Fold vc into wo: vc @ wo per head
    vc_b = vc.permute(1, 0, 2)                                  # [H, R, V]
    o_h = o_w.reshape(head_num, orig_v_head_dim, -1)            # [H, V, N]
    o_folded = torch.bmm(vc_b, o_h)                             # [H, R, N]
    o_folded = o_folded.reshape(head_num * o_folded.shape[1], -1)

    return (Linear(tensors={"weight": q_folded.contiguous()},
                   weight_format=q_b.weight_format,
                   data_format=q_b.data_format),
            Linear(tensors={"weight": o_folded.contiguous()},
                   weight_format=wo.weight_format,
                   data_format=wo.data_format))
```

- [ ] **Step 2: Add `pad_wo_input` function (after `fold_kv_b`, before `MLABuilder`)**

```python
def pad_wo_input(wo: Linear, *, cfg) -> Linear:
    """Pad wo input dim from head_num * cur_dim to head_num * size_per_head."""
    head_num = cfg.head_num
    size_per_head = cfg.head_dim
    w = wo.tensors["weight"]
    cur_dim = w.shape[0] // head_num
    w = w.reshape(head_num, cur_dim, -1)
    w = torch.nn.functional.pad(w, (0, 0, size_per_head - cur_dim, 0))
    w = w.reshape(head_num * size_per_head, -1)
    return Linear(tensors={"weight": w.contiguous()},
                  weight_format=wo.weight_format,
                  data_format=wo.data_format)
```

- [ ] **Step 3: Delete `_MLA_TP_RULES` dict (now inlined)**

Delete the `_MLA_TP_RULES` dict (lines 13–18) and its comment block (lines 9–11).

- [ ] **Step 4: Rewrite `add_projections` method**

Replace the `add_projections` method with:

```python
    def add_projections(self, *, q_a_proj, q_b_proj, kv_a_proj, kv_b_proj,
                        wo):
        """Apply MLA fold+pad, then commit each projection."""
        q_b_proj, wo = fold_kv_b(q_b_proj, kv_b_proj, wo, cfg=self.config)
        wo = pad_wo_input(wo, cfg=self.config)

        model_dtype = self.config.data_type
        for name, lin, side in [
            ("q_a_proj", q_a_proj, SplitSide.OUTPUT),
            ("q_b_proj", q_b_proj, SplitSide.OUTPUT),
            ("kv_a_proj", kv_a_proj, SplitSide.OUTPUT),
            ("wo", wo, SplitSide.INPUT),
        ]:
            self._commit_linear(name, lin, split_side=side,
                                model_dtype=model_dtype)
```

- [ ] **Step 5: Rewrite `add_norms` method**

Replace the `add_norms` method with:

```python
    def add_norms(self, *, q_a_norm, kv_a_norm, data_type):
        """Create norm children for q_a_layernorm and kv_a_layernorm."""
        self._add_norm_child('q_a_layernorm', q_a_norm,
                             data_type=data_type)
        self._add_norm_child('kv_a_layernorm', kv_a_norm,
                             data_type=data_type)
```

- [ ] **Step 6: Delete `_fold_and_pad` and `_fold_and_pad_hf` methods**

Delete the `_fold_and_pad` method (lines 69–89) and the `_fold_and_pad_hf` method (lines 91–158). Also delete the comment block above `_fold_and_pad` (lines 65–67).

- [ ] **Step 7: Verify no remaining references to deleted code**

Run:
```bash
cd /data/lmdeploy-modeling && grep -rn "_fold_and_pad\|_MLA_TP_RULES\|fold_and_pad_hf" lmdeploy/
```
Expected: no output.

- [ ] **Step 8: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/mla.py
git commit -m "refactor(builder): decompose MLA fold+pad into fold_kv_b + pad_wo_input pipeline"
```

---

### Task 3: Verify with model test

- [ ] **Step 1: Check GPU availability**

Check `get_gpu_usage` MCP tool for empty GPUs.

- [ ] **Step 2: Test GLM-4.7-Flash TP=1**

Run:
```bash
cd /data/lmdeploy-modeling && python scripts/test_turbomind_model.py THUDM/GLM-4.7-Flash --tp 1
```
Expected: Model responds with meaningful text (at least 128 tokens).

- [ ] **Step 3: Test GLM-4.7-Flash TP=2**

Run:
```bash
cd /data/lmdeploy-modeling && python scripts/test_turbomind_model.py THUDM/GLM-4.7-Flash --tp 2
```
Expected: Model responds with meaningful text (at least 128 tokens).
