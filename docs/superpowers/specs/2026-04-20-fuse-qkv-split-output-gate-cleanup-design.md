# Cleanup `fuse_qkv` and `split_output_gate` — Design

## Problem

`fuse_qkv` uses an `is_2d` flag and nested `reshape` closure to handle 1D/2D tensors. `split_output_gate` tracks `orig_shape = list(tensor.shape)` / `len(orig_shape) == 1` for the same purpose. Both are inconsistent with the `was_1d` / squeeze/unsqueeze pattern established in `_repeat_kv_heads`.

## Design

### `fuse_qkv`

Replace `is_2d` flag and nested `reshape` closure with unsqueeze/squeeze.
Collect raw tensors into a list, use list comprehension for reshape, and
drop the redundant `gt is not None` guard (gate is derived from Q and has
the same tensor kinds):

```python
def fuse_qkv(q: Linear, k: Linear, v: Linear, *,
             tp: int, gate: Linear | None = None) -> Linear:
    """Fuse Q, K, V (and optionally gate) into a single w_qkv Linear.

    Concatenates output channels with TP interleaving.
    Layout per tp-shard: [Q | K | V] or [Q | K | V | Gate].
    """
    merged_tensors: dict[str, torch.Tensor] = {}
    all_kinds = sorted(set(q.tensors) | set(k.tensors) | set(v.tensors))

    for kind in all_kinds:
        qt = q.tensors.get(kind)
        kt = k.tensors.get(kind)
        vt = v.tensors.get(kind)
        if qt is None or kt is None or vt is None:
            continue

        was_1d = qt.dim() == 1

        parts = [qt, kt, vt]
        if gate is not None:
            parts.append(gate.tensors[kind])

        if was_1d:
            parts = [p.unsqueeze(0) for p in parts]

        components = [p.view(p.size(0), tp, -1) for p in parts]
        merged = torch.cat(components, dim=-1)
        merged = merged.view(-1, merged.size(-1) * tp)
        if was_1d:
            merged = merged.squeeze(0)
        merged_tensors[kind] = merged

    return Linear(tensors=merged_tensors, weight_format=q.weight_format,
                  data_format=q.data_format)
```

### `split_output_gate`

Replace `orig_shape` tracking with `was_1d` flag. Use `unbind` with list
comprehension to avoid the duplicated index-then-reshape pattern:

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
        q_real, gate = (t.contiguous().reshape(-1, head_num * head_dim)
                        for t in tensor.unbind(dim=2))
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

## What is removed

- `is_2d` flag and nested `reshape` closure in `fuse_qkv`
- Redundant `if gt is not None:` guard in `fuse_qkv` (gate is derived from Q)
- `orig_shape = list(tensor.shape)` / `len(orig_shape) == 1` in `split_output_gate`
- Duplicated index-then-reshape lines in `split_output_gate` (replaced by `unbind`)

## Scope

Single file: `lmdeploy/turbomind/deploy/builder/attention.py`. No callers change — function signatures and return types are identical.
