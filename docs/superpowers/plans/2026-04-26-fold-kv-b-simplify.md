# fold_kv_b Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify `fold_kv_b` by replacing shape-derived dimensions with config fields and permute+bmm with einsum.

**Architecture:** Single-function rewrite in `mla.py`. Config fields (`qk_nope_dim`, `v_head_dim`, `kv_lora_rank`) replace shape arithmetic. Einsum subscripts (`"ihp,jhp->ihj"` and `"rhv,hvn->hrn"`) eliminate all permutes by keeping the natural `[R, H, ...]` tensor layout. `torch.split` replaces manual slicing.

**Tech Stack:** PyTorch (einsum, split, reshape)

---

### Task 1: Rewrite fold_kv_b

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/mla.py:14-62`

- [ ] **Step 1: Replace the function body**

Replace lines 14-62 of `mla.py` (the entire `fold_kv_b` function) with:

```python
def fold_kv_b(q_b: Linear, kv_b: Linear, wo: Linear, *,
              cfg) -> tuple[Linear, Linear]:
    """Fold kv_b into q_b and wo. Returns (q_b_folded, wo_folded).

    Splits kv_b into key-compressed (kc) and value-compressed (vc) parts.
    Folds kc into q_b via matmul (q_nope @ kc^T per head).
    Folds vc into wo via matmul (vc @ wo per head).
    All arithmetic in TM layout [in, out].
    """
    H = cfg.head_num
    P = cfg.qk_nope_dim
    S = cfg.qk_rope_dim
    V = cfg.v_head_dim
    R = cfg.kv_lora_rank

    q_b_h = q_b.tensors["weight"].reshape(R, H, P + S)
    kc, vc = kv_b.tensors["weight"].reshape(R, H, P + V).split([P, V], dim=-1)
    q_nope, q_rope = q_b_h.split([P, S], dim=-1)

    # q_nope @ kc^T per head, staying in [R, H, ...] layout
    q_folded = torch.cat([
        torch.einsum("ihp,jhp->ihj", q_nope, kc),  # [R, H, R]
        q_rope,                                      # [R, H, S]
    ], dim=-1).reshape(R, H * (R + S))

    # vc @ wo per head
    o_folded = torch.einsum("rhv,hvn->hrn", vc,
                            wo.tensors["weight"].reshape(H, V, -1)
                            ).reshape(H * R, -1)

    return (Linear(tensors={"weight": q_folded.contiguous()},
                   weight_format=q_b.weight_format,
                   data_format=q_b.data_format),
            Linear(tensors={"weight": o_folded.contiguous()},
                   weight_format=wo.weight_format,
                   data_format=wo.data_format))
```

- [ ] **Step 2: Verify the file has no syntax errors**

```bash
python -c "import ast; ast.parse(open('lmdeploy/turbomind/deploy/builder/mla.py').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/mla.py
git commit -m "refactor: simplify fold_kv_b with config fields and einsum"
```

---

### Task 2: Verify correctness with an MLA model

**Files:** None (test only)

- [ ] **Step 1: Build TurboMind**

```bash
cd build && ninja
```

- [ ] **Step 2: Run model test**

```bash
python scripts/test_turbomind_model.py
```

Verify the model responds with meaningful human words relevant to the test prompt, at least 128 tokens.

- [ ] **Step 3: If test passes, no further action. If it fails, debug and fix before moving on.**
