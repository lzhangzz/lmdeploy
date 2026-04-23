# Move `split_output_gate` from Builder to Spec — Design

## Problem

`AttentionBuilder.add_qkv_proj` contains model-specific branching logic for
Qwen3.5's output gate. The builder checks `self.config.attn_output_gate` and
conditionally calls `split_output_gate`. Only `Qwen3_5Spec` ever sets this flag.

## Solution

Move the `split_output_gate` call from `AttentionBuilder.add_qkv_proj` into
`Qwen3_5Spec.attn()`. The builder receives a pre-split `gate` as an optional
keyword argument. `dequant_mixed` becomes variadic to handle any number of
Linears including gate. `pad_for_tp` is renamed to `repeat_kv_for_tp` (no gate
needed — only KV heads are repeated).

The `attn_output_gate` flag stays in `AttentionConfig` for C++ runtime use.

## Changes

### `dequant_mixed` (attention.py)

Make variadic (`*linears`). If any arg has trivial weight format, dequantize
all other non-None args. `_dequant_linear` is a no-op on trivial/None formats,
so calling it on trivial args is safe:

```python
def dequant_mixed(*linears: Linear) -> tuple[Linear, ...]:
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

### `pad_for_tp` -> `repeat_kv_for_tp` (attention.py)

Rename. Only takes k and v — only KV heads need repeating for TP divisibility:

```python
def repeat_kv_for_tp(k: Linear, v: Linear, *,
                     tp: int, head_dim: int) -> tuple[Linear, Linear]:
    k = _repeat_kv_heads(k, tp=tp, heads=_infer_heads(k, head_dim))
    v = _repeat_kv_heads(v, tp=tp, heads=_infer_heads(v, head_dim))
    return k, v
```

### `AttentionBuilder.add_qkv_proj` (attention.py)

Remove the `if self.config.attn_output_gate` branch. Accept optional `gate`.
Pass gate through `dequant_mixed` but not `repeat_kv_for_tp`. Update docstring
to remove the pipeline mention (split_output_gate is no longer in the pipeline):

```python
def add_qkv_proj(self, q, k, v, *, gate=None):
    q, k, v, gate = dequant_mixed(q, k, v, gate)
    k, v = repeat_kv_for_tp(k, v, tp=self._tp, head_dim=self.config.head_dim)
    merged = fuse_qkv(q, k, v, tp=self._tp, gate=gate)
    self._commit_linear('w_qkv', merged, SplitSide.OUTPUT,
                        model_dtype=self.config.data_type)
```

### `Qwen3_5Spec.attn()` (qwen3_5_spec.py)

Add import `from ..builder.attention import split_output_gate`. Split Q before
calling the builder. No conditional needed — `attn()` is only called for
standard attention layers, which are always gated in Qwen3.5:

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
    ...
```

### Docstring updates (attention.py)

- Module docstring (line 6): `pad_for_tp` -> `repeat_kv_for_tp`
- `add_qkv_proj` docstring (line 140): remove `split_output_gate` from
  pipeline description

### `split_output_gate` simplification (attention.py)

Remove redundant `.contiguous()` calls (`.reshape()` handles non-contiguous).
Use `unbind` instead of manual indexing:

```python
@transform_tensors
def split_output_gate(tensor: torch.Tensor, *, head_dim: int
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    head_num = tensor.size(-1) // (head_dim * 2)
    q, gate = tensor.view(-1, head_num, 2, head_dim).unbind(2)
    return q.reshape(-1, head_num * head_dim), gate.reshape(-1, head_num * head_dim)
```

### Unchanged

- `attn_output_gate` stays in `AttentionConfig` (C++ runtime needs it)
- `split_output_gate` function stays in `attention.py`
- Other specs pass `gate=None` implicitly (default) — no changes needed
