# Move split_output_gate from Builder to Spec — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move Qwen3.5-specific output-gate splitting from the generic `AttentionBuilder` into `Qwen3_5Spec`, making the builder model-agnostic.

**Architecture:** `split_output_gate` is called by `Qwen3_5Spec.attn()` before passing `gate` to `AttentionBuilder.add_qkv_proj(q, k, v, gate=gate)`. `dequant_mixed` becomes variadic to handle the extra gate Linear. `pad_for_tp` is renamed to `repeat_kv_for_tp` and takes only k, v.

**Tech Stack:** Python, PyTorch, TurboMind deploy pipeline

---

## File Structure

| File | Change |
|------|--------|
| `lmdeploy/turbomind/deploy/builder/attention.py` | Modify `dequant_mixed`, rename `pad_for_tp`→`repeat_kv_for_tp`, simplify `split_output_gate`, update `add_qkv_proj` |
| `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` | Add import, call `split_output_gate` in `attn()`, pass `gate=` to `add_qkv_proj` |

---

### Task 1: Simplify `split_output_gate`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/attention.py:95-107`

- [ ] **Step 1: Replace the `split_output_gate` implementation**

```python
@transform_tensors
def split_output_gate(tensor: torch.Tensor, *, head_dim: int
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """Split output gate from Q projection (Qwen3.5).

    Q's output dim is 2 * head_num * head_dim. Reshape to
    [batch, head_num, 2, head_dim], split into q_real and gate.
    """
    head_num = tensor.size(-1) // (head_dim * 2)
    q, gate = tensor.view(-1, head_num, 2, head_dim).unbind(2)
    return q.reshape(-1, head_num * head_dim), gate.reshape(-1, head_num * head_dim)
```

- [ ] **Step 2: Test with Qwen3.5-27B**

```bash
cd /data/lmdeploy-modeling && python scripts/test_turbomind_model.py Qwen/Qwen3.5-27B --gpu 1 --max-new-tokens 128
```

Expected: Coherent English response. `split_output_gate` is exercised by Qwen3.5's attention layers.

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/attention.py
git commit -m "refactor: simplify split_output_gate with unbind"
```

---

### Task 2: Make `dequant_mixed` variadic

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/attention.py:39-54`

- [ ] **Step 1: Replace `dequant_mixed`**

```python
def dequant_mixed(*linears: Linear) -> tuple[Linear, ...]:
    """Dequantize to trivial if any arg is in trivial format.

    When any Linear has trivial weight format (e.g. from RoPE reordering),
    dequantize all non-trivial args so formats match for fusion.
    None args pass through unchanged.
    """
    has_trivial = any(
        l is not None
        and l.weight_format is not None
        and l.weight_format.name == 'trivial'
        for l in linears
    )
    if not has_trivial:
        return linears
    return tuple(_dequant_linear(l) if l is not None else l
                 for l in linears)
```

Note: `_dequant_linear` is imported from `._base` (already imported at line 15). It is a no-op on trivial/None formats.

- [ ] **Step 2: Update the call site in `add_qkv_proj`**

The call `q, k, v = dequant_mixed(q, k, v)` at line 143 already works with the variadic signature — `*linears` accepts three positional args and returns a 3-tuple. But we need to prepare for the gate parameter (added in Task 4). No change needed yet — the existing call `dequant_mixed(q, k, v)` returns `(q, k, v)` which unpacks correctly.

- [ ] **Step 3: Test with multiple models**

Test one quantized model (exercises dequant path) and one FP16 model:

```bash
cd /data/lmdeploy-modeling && python scripts/test_turbomind_model.py Qwen/Qwen3.5-27B --gpu 1 --max-new-tokens 128
```

```bash
cd /data/lmdeploy-modeling && python scripts/test_turbomind_model.py QuantTrio/Qwen3.5-35B-A3B-AWQ --gpu 1 --max-new-tokens 128
```

Expected: Both produce coherent responses.

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/attention.py
git commit -m "refactor: make dequant_mixed variadic"
```

---

### Task 3: Rename `pad_for_tp` to `repeat_kv_for_tp`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/attention.py:80-92, 6, 140, 144`

- [ ] **Step 1: Replace the function**

```python
def repeat_kv_for_tp(k: Linear, v: Linear, *,
                     tp: int, head_dim: int) -> tuple[Linear, Linear]:
    """Repeat KV heads to reach a TP-divisible count."""
    k = _repeat_kv_heads(k, tp=tp, heads=_infer_heads(k, head_dim))
    v = _repeat_kv_heads(v, tp=tp, heads=_infer_heads(v, head_dim))
    return k, v
```

- [ ] **Step 2: Update the module docstring (line 6)**

Change `pad_for_tp` to `repeat_kv_for_tp` in the module docstring:

```python
"""Attention weight loading builder and QKV fusion pipeline.

Provides ``AttentionBuilder`` for committing attention weights (QKV fusion,
O-proj, QK-norm, direct params) and pipeline functions (``dequant_mixed``,
``repeat_kv_for_tp``, ``split_output_gate``, ``fuse_qkv``) for fusing Q/K/V
Linear bundles into a single interleaved w_qkv with KV head padding and
output-gate splitting.
"""
```

- [ ] **Step 3: Update `add_qkv_proj` call site (line 144) and docstring (line 140)**

Replace:
```python
        q, k, v = pad_for_tp(q, k, v, tp=self._tp,
                              head_dim=self.config.head_dim)
```
With:
```python
        k, v = repeat_kv_for_tp(k, v, tp=self._tp,
                                head_dim=self.config.head_dim)
```

Update the docstring to:
```python
        """Fuse Q/K/V into a single w_qkv with TP interleave, commit.

        Pipeline: dequant_mixed -> repeat_kv_for_tp -> fuse_qkv -> commit.
        """
```

- [ ] **Step 4: Test**

```bash
cd /data/lmdeploy-modeling && python scripts/test_turbomind_model.py Qwen/Qwen3-4B --gpu 1 --max-new-tokens 128
```

Expected: Coherent response. Tests the non-gate path through `repeat_kv_for_tp`.

- [ ] **Step 5: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/attention.py
git commit -m "refactor: rename pad_for_tp to repeat_kv_for_tp, drop q param"
```

---

### Task 4: Move gate split to `Qwen3_5Spec`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py:11, 187-208`
- Modify: `lmdeploy/turbomind/deploy/builder/attention.py:137-151`

- [ ] **Step 1: Add import in `qwen3_5_spec.py`**

Add after line 14 (the second `from ..builder import` line):

```python
from ..builder.attention import split_output_gate
```

- [ ] **Step 2: Update `Qwen3_5Spec.attn()`**

Replace the `attn` method (lines 187-208) with:

```python
    def attn(self, pfx, layer):
        q = self._linear(f'{pfx}.q_proj')
        k = self._linear(f'{pfx}.k_proj')
        v = self._linear(f'{pfx}.v_proj')
        o = self._linear(f'{pfx}.o_proj')

        q = reorder_rotary_emb_linear(q, self._head_dim, self._rope.dim)
        k = reorder_rotary_emb_linear(k, self._head_dim, self._rope.dim)

        q, gate = split_output_gate(q, head_dim=self._head_dim)

        cfg = self._attn_cfg.clone()
        attn = AttentionBuilder(cfg, self._contexts,
                                tp=self.engine_cfg.attn_tp_size,
                                ranks=self._attn_ranks)
        attn.add_qkv_proj(q, k, v, gate=gate)
        attn.add_o_proj(o)

        q_norm = self._zero_centered(self._get(f'{pfx}.q_norm.weight'))
        k_norm = self._zero_centered(self._get(f'{pfx}.k_norm.weight'))
        q_norm = reorder_rotary_emb(q_norm, self._head_dim, self._rope.dim)
        k_norm = reorder_rotary_emb(k_norm, self._head_dim, self._rope.dim)
        attn.add_qk_norm(q_norm, k_norm, norm_eps=self._norm_eps)
        return attn
```

- [ ] **Step 3: Update `add_qkv_proj` in `attention.py`**

Replace the method (lines 137-151) with:

```python
    def add_qkv_proj(self, q, k, v, *, gate=None):
        """Fuse Q/K/V into a single w_qkv with TP interleave, commit.

        Pipeline: dequant_mixed -> repeat_kv_for_tp -> fuse_qkv -> commit.
        """
        q, k, v, gate = dequant_mixed(q, k, v, gate)
        k, v = repeat_kv_for_tp(k, v, tp=self._tp, head_dim=self.config.head_dim)
        merged = fuse_qkv(q, k, v, tp=self._tp, gate=gate)
        self._commit_linear('w_qkv', merged, SplitSide.OUTPUT,
                            model_dtype=self.config.data_type)
```

- [ ] **Step 4: Test with Qwen3.5**

```bash
cd /data/lmdeploy-modeling && python scripts/test_turbomind_model.py Qwen/Qwen3.5-27B --gpu 1 --max-new-tokens 128
```

Expected: Coherent English response. Gate is now split in the spec, not the builder.

- [ ] **Step 5: Test with non-gate model (regression check)**

```bash
cd /data/lmdeploy-modeling && python scripts/test_turbomind_model.py Qwen/Qwen3-4B --gpu 1 --max-new-tokens 128
```

Expected: Coherent response. Non-gate path works with `gate=None`.

- [ ] **Step 6: Commit**

```bash
git add lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py lmdeploy/turbomind/deploy/builder/attention.py
git commit -m "refactor: move split_output_gate from builder to Qwen3_5Spec"
```

---

### Task 5: Test AWQ quantized Qwen3.5

**Files:** None (verification only)

- [ ] **Step 1: Test AWQ model with gate**

```bash
cd /data/lmdeploy-modeling && python scripts/test_turbomind_model.py QuantTrio/Qwen3.5-35B-A3B-AWQ --gpu 1 --max-new-tokens 128
```

Expected: Coherent response. Exercises the `dequant_mixed` path where gate and q are dequantized together.

- [ ] **Step 2: Test GPT-OSS (non-gate regression)**

```bash
cd /data/lmdeploy-modeling && python scripts/test_turbomind_model.py openai/gpt-oss-20b --gpu 1 --max-new-tokens 128
```

Expected: Coherent response. Verifies `add_qkv_proj` with `gate=None` works for GptOssSpec.
