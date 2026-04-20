# Simplify `pad_for_tp` — Design

## Problem

`pad_for_tp` in `attention.py` has 4-way branching (block-scale vs per-element x pad vs repeat) that is hard to follow. In practice, block_out is always a multiple of head_dim and Q heads are always already TP-divisible by the time this function runs. The Q-padding path is dead code.

## Design

### Contract change

- Assert Q heads are TP-divisible (remove the zero-pad-Q path)
- Only repeat KV heads to reach TP-divisibility

### Head-granularity unification

Since `block_out % head_dim == 0` always holds, every tensor in a Linear (weight, bias, scales, zeros) has an output dimension that is a multiple of the head count. This means we can compute `per_head = tensor.size(-1) // heads` for any tensor kind and work at head granularity uniformly — no block-vs-element distinction needed.

### New structure

```python
def pad_for_tp(q, k, v, *, tp, head_dim):
    assert _infer_heads(q, head_dim) % tp == 0
    k = _repeat_kv_heads(k, tp, head_dim)
    v = _repeat_kv_heads(v, tp, head_dim)
    return q, k, v
```

`_repeat_kv_heads(linear, tp, head_dim)`:
1. Infer heads from weight shape
2. If already TP-divisible, return linear unchanged
3. For each tensor: reshape last dim to `(heads, per_head)`, `repeat_interleave` along head axis, reshape back
4. Return new Linear with repeated tensors

`_infer_heads(linear, head_dim)`: extract head count from weight output dim (unchanged logic).

### Tensor shapes

All tensors in a Linear are 1D or 2D. The last dim is always `heads * per_head`. For 1D tensors, reshape to `(heads, per_head)`; for 2D, reshape to `(in_dim, heads, per_head)`. After repeat_interleave, reshape back to the original rank.

### What is removed

- `_adjust_linear` helper (4-way branching)
- Block-scale vs per-element distinction
- Q zero-padding path
- `pad_out_dim` import (no longer used here)

## Scope

Single file: `lmdeploy/turbomind/deploy/builder/attention.py`. No callers change — the function signature and return type are identical.
