# add_conv1d Refactor Design

## Goal

Extract TP interleaving and Q/K/V fusion from `add_conv1d` into separate standalone functions, remove all defensive guards.

## Current State

`DeltaNetBuilder.add_conv1d` (lines 123–150 in `deltanet.py`) does everything inline:
1. None guard: `if conv1d is None: return`
2. Squeeze leading singleton dim (HF artifact)
3. Transpose HF `[conv_dim, d_conv]` → TM `[d_conv, conv_dim]`
4. TP Q/K/V interleaving (guarded by `if self._tp > 1 and qkv_split is not None`)
5. Commit tensor

One call site: `qwen3_5_spec.py:207` — always passes a tensor and `qkv_split`.

## Changes

### 1. Make `_tp_interleave_tensor` public

Rename `_tp_interleave_tensor` → `tp_interleave_tensor`. No logic change.

### 2. New `fuse_qkv_conv1d` function

```python
def fuse_qkv_conv1d(t: Tensor, qkv_split: tuple[int, int, int],
                     tp: int) -> Tensor:
    """Split conv1d into Q/K/V parts, TP-interleave each, concatenate back."""
```

Splits output dim into Q/K/V parts by `qkv_split`, calls `tp_interleave_tensor` on each, concatenates back. For tp=1 the reshapes are identity.

### 3. Revised `add_conv1d`

```python
def add_conv1d(self, conv1d, qkv_split):
    if conv1d.ndim == 3 and conv1d.shape[1] == 1:
        conv1d = conv1d.squeeze(1)
    conv1d = conv1d.t().contiguous()
    conv1d = fuse_qkv_conv1d(conv1d, qkv_split, self._tp)
    self._commit_tensor("conv1d", conv1d, split_side=SplitSide.OUTPUT)
```

No guards. `qkv_split` is required (no default). `tp=1` handled naturally by the reshape math. Squeeze + transpose stays inline (trivial).

## Code to Delete

| What | Why |
|------|-----|
| `if conv1d is None: return` | Dead guard |
| `if self._tp > 1 and qkv_split is not None:` | Absorbed into `fuse_qkv_conv1d` |
| Inline TP interleaving | Extracted to `fuse_qkv_conv1d` |

## Scope

Single file (`deltanet.py`). No spec changes. No behavioral changes.
