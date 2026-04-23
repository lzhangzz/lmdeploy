# Cleanup `fuse_qkv` and `split_output_gate` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify the 1D/2D tensor handling in `fuse_qkv` and `split_output_gate` to match the `was_1d` / squeeze/unsqueeze pattern established in `_repeat_kv_heads`.

**Architecture:** Replace the `is_2d` flag + nested `reshape` closure in `fuse_qkv` with a collect-unsqueeze-work-squeeze flow. Replace `orig_shape` tracking in `split_output_gate` with a `was_1d` flag. Remove meaningless None guards since q/k/v/gate always share the same tensor keys.

**Tech Stack:** Python, PyTorch

---

### Task 1: Replace `fuse_qkv` and `split_output_gate` implementations

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/attention.py:104-168`

- [ ] **Step 1: Replace `split_output_gate` (lines 104–130) with the new implementation**

Replace lines 104–130 with:

```python
def split_output_gate(q: Linear, *, head_dim: int) -> tuple[Linear, Linear]:
    """Split output gate from Q projection (Qwen3.5).

    Q's output dim is 2 * head_num * head_dim. Reshape to
    [batch, head_num, 2, head_dim], split into q_real and gate.
    """
    new_q_tensors = {}
    gate_tensors = {}

    for kind, tensor in q.tensors.items():
        head_num = tensor.size(-1) // (head_dim * 2)
        was_1d = tensor.dim() == 1
        if was_1d:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.view(tensor.size(0), head_num, 2, head_dim)
        q_real = tensor[:, :, 0, :].contiguous().reshape(-1, head_num * head_dim)
        gate = tensor[:, :, 1, :].contiguous().reshape(-1, head_num * head_dim)
        if was_1d:
            q_real = q_real.squeeze(0)
            gate = gate.squeeze(0)
        new_q_tensors[kind] = q_real
        gate_tensors[kind] = gate

    return (Linear(tensors=new_q_tensors, weight_format=q.weight_format,
                   data_format=q.data_format),
            Linear(tensors=gate_tensors, weight_format=q.weight_format,
                   data_format=q.data_format))
```

- [ ] **Step 2: Replace `fuse_qkv` (lines 133–168) with the new implementation**

Replace lines 133–168 with:

```python
def fuse_qkv(q: Linear, k: Linear, v: Linear, *,
             tp: int, gate: Linear | None = None) -> Linear:
    """Fuse Q, K, V (and optionally gate) into a single w_qkv Linear.

    Concatenates output channels with TP interleaving.
    Layout per tp-shard: [Q | K | V] or [Q | K | V | Gate].
    """
    merged_tensors: dict[str, torch.Tensor] = {}

    for kind, qt in q.tensors.items():
        kt = k.tensors[kind]
        vt = v.tensors[kind]

        was_1d = qt.dim() == 1
        raw = [qt, kt, vt]
        if gate is not None:
            raw.append(gate.tensors[kind])
        if was_1d:
            raw = [t.unsqueeze(0) for t in raw]

        components = [t.view(t.size(0), tp, -1) for t in raw]
        merged = torch.cat(components, dim=-1)
        merged = merged.view(-1, merged.size(-1) * tp)
        if was_1d:
            merged = merged.squeeze(0)
        merged_tensors[kind] = merged

    return Linear(tensors=merged_tensors, weight_format=q.weight_format,
                  data_format=q.data_format)
```

- [ ] **Step 3: Inline smoke test**

Run a quick Python snippet to verify both functions:

```bash
cd /data/lmdeploy-modeling && python -c "
import torch
from lmdeploy.turbomind.deploy.linear import Linear
from lmdeploy.turbomind.deploy.builder.attention import fuse_qkv, split_output_gate

tp = 2

# --- fuse_qkv tests ---

# 2D tensors, no gate
q = Linear(tensors={'weight': torch.randn(64, 8*8)})
k = Linear(tensors={'weight': torch.randn(64, 2*8)})
v = Linear(tensors={'weight': torch.randn(64, 2*8)})
merged = fuse_qkv(q, k, v, tp=tp)
assert merged.tensors['weight'].shape == (64, tp * (8 + 2 + 2) * 8 // tp)
# Per-shard: Q=8*8/tp=32, K=2*8/tp=8, V=2*8/tp=8 -> 48 per shard * 2 = 96
assert merged.tensors['weight'].shape == (64, 96)

# 2D + 1D (bias)
q = Linear(tensors={'weight': torch.randn(64, 16), 'bias': torch.randn(16)})
k = Linear(tensors={'weight': torch.randn(64, 4), 'bias': torch.randn(4)})
v = Linear(tensors={'weight': torch.randn(64, 4), 'bias': torch.randn(4)})
merged = fuse_qkv(q, k, v, tp=tp)
assert merged.tensors['weight'].shape == (64, 48)
assert merged.tensors['bias'].shape == (48,)

# With gate
q = Linear(tensors={'weight': torch.randn(64, 16)})
k = Linear(tensors={'weight': torch.randn(64, 4)})
v = Linear(tensors={'weight': torch.randn(64, 4)})
g = Linear(tensors={'weight': torch.randn(64, 16)})
merged = fuse_qkv(q, k, v, tp=tp, gate=g)
# Per shard: Q=8, K=2, V=2, G=8 -> 20 * 2 = 40
assert merged.tensors['weight'].shape == (64, 80)

print('fuse_qkv tests passed')

# --- split_output_gate tests ---

hd = 8
# 2D: 4 heads, output = 4*2*8 = 64
q = Linear(tensors={'weight': torch.randn(32, 64)})
q_real, gate = split_output_gate(q, head_dim=hd)
assert q_real.tensors['weight'].shape == (32, 32)
assert gate.tensors['weight'].shape == (32, 32)

# 1D (bias): output = 64
q = Linear(tensors={'bias': torch.randn(64)})
q_real, gate = split_output_gate(q, head_dim=hd)
assert q_real.tensors['bias'].shape == (32,)
assert gate.tensors['bias'].shape == (32,)

# 2D + 1D mixed
q = Linear(tensors={'weight': torch.randn(32, 64), 'bias': torch.randn(64)})
q_real, gate = split_output_gate(q, head_dim=hd)
assert q_real.tensors['weight'].shape == (32, 32)
assert gate.tensors['weight'].shape == (32, 32)
assert q_real.tensors['bias'].shape == (32,)
assert gate.tensors['bias'].shape == (32,)

print('split_output_gate tests passed')
print('All inline tests passed')
"
```

Expected: `All inline tests passed`

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/attention.py
git commit -m "refactor: unify 1D/2D handling in fuse_qkv and split_output_gate"
```

---

### Task 2: Verify with real model

**Files:** None (runtime verification)

- [ ] **Step 1: Run test_turbomind_model.py**

Pick a model with TP=2 and verify the response is meaningful and at least 128 tokens:

```bash
cd /data/lmdeploy-modeling && python scripts/test_turbomind_model.py Qwen/Qwen3-4B /nvme4/huggingface_hub/hub 2 0,2
```

Verify the response is meaningful (not gibberish) and at least 128 tokens.
