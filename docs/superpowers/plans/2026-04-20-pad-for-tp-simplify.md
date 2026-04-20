# Simplify `pad_for_tp` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 4-way branched `pad_for_tp` with a simpler KV-only repeat at head granularity.

**Architecture:** Assert Q is TP-divisible (always true in practice). For KV, compute `per_head = tensor.size(-1) // heads` uniformly for all tensor kinds (weight, bias, scales, zeros) and use `repeat_interleave` along the head axis. No block-vs-element distinction needed.

**Tech Stack:** Python, PyTorch

---

### Task 1: Replace `pad_for_tp` implementation

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/attention.py:57-151`

- [ ] **Step 1: Replace the old `pad_for_tp` and its helpers with the new implementation**

Replace lines 57–151 (the entire `pad_for_tp` function including `_infer_heads` and `_adjust_linear`) with:

```python
def _infer_heads(linear: Linear, head_dim: int) -> int:
    """Derive head count from the weight tensor's output dimension."""
    w = linear.tensors.get('weight')
    if w is None:
        return 0
    return w.size(-1) // head_dim


def _repeat_kv_heads(linear: Linear, tp: int, head_dim: int) -> Linear:
    """Repeat KV heads to reach a TP-divisible count."""
    heads = _infer_heads(linear, head_dim)
    if heads % tp == 0:
        return linear
    target_heads = ((heads + tp - 1) // tp) * tp
    assert target_heads % heads == 0, (
        f"target_heads={target_heads} must be divisible by heads={heads}")
    n_repeat = target_heads // heads
    new_tensors = {}
    for kind, tensor in linear.tensors.items():
        per_head = tensor.size(-1) // heads
        if tensor.dim() == 2:
            t = tensor.view(tensor.size(0), heads, per_head)
            t = t.repeat(1, n_repeat, 1)
            new_tensors[kind] = t.reshape(tensor.size(0), target_heads * per_head)
        else:
            t = tensor.view(heads, per_head)
            t = t.repeat(n_repeat, 1)
            new_tensors[kind] = t.reshape(target_heads * per_head)
    return Linear(tensors=new_tensors, weight_format=linear.weight_format,
                  data_format=linear.data_format)


def pad_for_tp(q: Linear, k: Linear, v: Linear, *,
               tp: int, head_dim: int) -> tuple[Linear, Linear, Linear]:
    """Repeat KV heads to reach a TP-divisible count.

    Q is asserted to already be TP-divisible.
    Head counts are derived from actual tensor shapes, not config parameters.
    """
    assert _infer_heads(q, head_dim) % tp == 0, (
        f"Q heads={_infer_heads(q, head_dim)} must be divisible by tp={tp}")
    k = _repeat_kv_heads(k, tp, head_dim)
    v = _repeat_kv_heads(v, tp, head_dim)
    return q, k, v
```

- [ ] **Step 2: Remove `pad_out_dim` import**

Change line 14 from:
```python
from ..linear import Linear, pad_out_dim
```
to:
```python
from ..linear import Linear
```

- [ ] **Step 3: Inline smoke test**

Run a quick Python snippet to verify behavior:

```bash
cd /data/lmdeploy-modeling && python -c "
import torch
from lmdeploy.turbomind.deploy.linear import Linear
from lmdeploy.turbomind.deploy.builder.attention import pad_for_tp

def mk(h, hd, bias=False):
    t = {'weight': torch.randn(64, h * hd)}
    if bias: t['bias'] = torch.randn(h * hd)
    return Linear(tensors=t)

tp, hd = 2, 8

# No-op: all TP-divisible
q, k, v = mk(8, hd), mk(2, hd), mk(2, hd)
q2, k2, v2 = pad_for_tp(q, k, v, tp=tp, head_dim=hd)
assert q2 is q and k2 is k and v2 is v, 'no-op failed'

# KV repeat: 1 head -> 2
q, k, v = mk(8, hd), mk(1, hd), mk(1, hd)
q2, k2, v2 = pad_for_tp(q, k, v, tp=tp, head_dim=hd)
assert k2.tensors['weight'].shape == (64, 2*hd)
assert torch.equal(k2.tensors['weight'][:,:hd], k.tensors['weight'])
assert torch.equal(k2.tensors['weight'][:,hd:], k.tensors['weight'])

# KV repeat with bias
q, k, v = mk(8, hd, True), mk(1, hd, True), mk(1, hd, True)
q2, k2, v2 = pad_for_tp(q, k, v, tp=tp, head_dim=hd)
assert k2.tensors['bias'].shape == (2*hd,)
assert torch.equal(k2.tensors['bias'][:hd], k.tensors['bias'])

# Q assert
try:
    pad_for_tp(mk(3, hd), mk(2, hd), mk(2, hd), tp=tp, head_dim=hd)
    assert False, 'should have raised'
except AssertionError:
    pass

print('All inline tests passed')
"
```
Expected: `All inline tests passed`

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/attention.py
git commit -m "refactor: simplify pad_for_tp to KV-only repeat at head granularity"
```

---

### Task 2: Verify with real model

**Files:** None (runtime verification)

- [ ] **Step 1: Run test_turbomind_model.py on a model with GQA**

Pick a model with `kv_head_num < q_head_num` (e.g. one with group-query attention). Use `list_models` to find one, then run:

```bash
cd /data/lmdeploy-modeling && python scripts/test_turbomind_model.py <model_id> --tp 2
```

Verify the response is meaningful (not gibberish) and at least 128 tokens.

- [ ] **Step 2: Commit (if any fixups were needed)**

If the real-model test revealed issues, commit the fixes:
```bash
git commit -m "fix: address pad_for_tp issue found during model testing"
```
