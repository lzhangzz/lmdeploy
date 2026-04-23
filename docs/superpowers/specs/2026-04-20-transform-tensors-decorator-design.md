# `@transform_tensors` Decorator

## Problem

Three functions in `builder/attention.py` share an identical boilerplate pattern:

1. Loop over `linear.tensors.items()` (kind, tensor)
2. Check `was_1d = tensor.dim() == 1`
3. Unsqueeze 1D tensors to 2D
4. Do the actual transformation on 2D tensors
5. Squeeze back to 1D if was_1d
6. Collect into a new dict
7. Return `Linear(tensors=new_tensors, ...)`

This pattern appears in `_repeat_kv_heads` (1-in/1-out), `split_output_gate` (1-in/2-out), and `fuse_qkv` (3-in/1-out).

## Design

A single `@transform_tensors` decorator that eliminates the boilerplate. The decorated function works on raw 2D `torch.Tensor` objects; the decorator handles the 1D/2D roundtrip and dict iteration.

### Signature inspection

The decorator inspects annotations at decoration time:

- **Parameters typed `torch.Tensor`** map to `Linear` positional args at call site. The decorator iterates over `kind`s, extracting the corresponding tensor from each Linear per kind.
- **Parameters typed `torch.Tensor | None`** are optional Linear args. When `None` is passed, `None` is forwarded to the inner function without the 1D/2D dance.
- **Other parameters** (keyword-only) pass through unchanged.
- **Return type `torch.Tensor`** produces a single `Linear` output.
- **Return type `tuple[torch.Tensor, ...]`** produces a tuple of `Linear` outputs (one per tensor in the tuple).

### 1D/2D handling

For each `kind`, the decorator unsqueezes 1D tensors before calling the inner function and squeezes them back after. The `was_1d` state is derived from the **first** `torch.Tensor`-typed parameter's tensor for that kind. All tensor-typed params are assumed to share the same dimensionality for a given kind (this is already the invariant in existing code).

### Format propagation

The output `Linear`(s) inherit `weight_format` and `data_format` from the first input `Linear`. All inputs are assumed to share the same format (already guaranteed by upstream `dequant_mixed`).

### Arity summary

| Inner signature | Call signature | Behavior |
|---|---|---|
| `(t: Tensor) -> Tensor` | `(linear: Linear) -> Linear` | 1-in, 1-out |
| `(t: Tensor) -> tuple[Tensor, Tensor]` | `(linear: Linear) -> tuple[Linear, Linear]` | 1-in, 2-out |
| `(q: Tensor, k: Tensor, v: Tensor) -> Tensor` | `(q: Linear, k: Linear, v: Linear) -> Linear` | 3-in, 1-out |
| `(..., gate: Tensor \| None) -> ...` | `(..., gate: Linear \| None) -> ...` | Optional tensor passthrough |

### Location

`lmdeploy/turbomind/deploy/builder/_base.py`, alongside existing builder primitives.

## Examples

### 1-in, 1-out (`_repeat_kv_heads`)

```python
@transform_tensors
def _repeat_kv_heads(tensor: torch.Tensor, *, tp: int, head_dim: int) -> torch.Tensor:
    heads = tensor.size(-1) // head_dim
    target_heads = ((heads + tp - 1) // tp) * tp
    n_repeat = target_heads // heads
    per_head = tensor.size(-1) // heads
    t = tensor.view(-1, heads, per_head)
    return t.repeat(1, n_repeat, 1).reshape(-1, target_heads * per_head)
```

Called as: `_repeat_kv_heads(linear, tp=tp, head_dim=head_dim) -> Linear`

Note: the pre-condition check (`heads % tp == 0` for Q, `target_heads % heads == 0`) stays in the caller (`pad_for_tp`).

### 1-in, 2-out (`split_output_gate`)

```python
@transform_tensors
def split_output_gate(tensor: torch.Tensor, *, head_dim: int
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    head_num = tensor.size(-1) // (head_dim * 2)
    t = tensor.view(-1, head_num, 2, head_dim)
    q_real = t[:, :, 0, :].contiguous().reshape(-1, head_num * head_dim)
    gate = t[:, :, 1, :].contiguous().reshape(-1, head_num * head_dim)
    return q_real, gate
```

Called as: `split_output_gate(linear, head_dim=head_dim) -> tuple[Linear, Linear]`

### 3-in, 1-out with optional (`fuse_qkv`)

```python
@transform_tensors
def fuse_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
             *, tp: int, gate: torch.Tensor | None = None) -> torch.Tensor:
    parts = [t.view(t.size(0), tp, -1) for t in (q, k, v)]
    if gate is not None:
        parts.append(gate.view(gate.size(0), tp, -1))
    merged = torch.cat(parts, dim=-1)
    return merged.view(-1, merged.size(-1) * tp)
```

Called as: `fuse_qkv(q, k, v, tp=tp, gate=gate_or_none) -> Linear`

## Scope

Only the three functions in `attention.py` are refactored. No other files change. The public API of these functions stays the same (same names, same call signatures at the Linear level).
