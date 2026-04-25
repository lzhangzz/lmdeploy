# Design: `pack()` returns alloc metadata

## Problem

In `_commit_linear` (`_base.py:490-503`), after packing a tensor, the builder
reconstructs what the format already knows — whether the weight is quantized
and what the logical shape/dtype should be — via a 3-way branch:

```python
if kind == 'weight' and is_quantized:
    alloc_shape, alloc_dtype = ([in_dim, out_dim], weight_cpp_dtype)
elif kind == 'weight' and model_dtype is not None:
    alloc_shape, alloc_dtype = None, model_dtype
else:
    alloc_shape, alloc_dtype = None, None
```

This leaks format internals (`is_quantized`, `weight_cpp_dtype`) into the
builder and creates a coupling point that every new quantized format must
work around.

## Design

Make `WeightFormat.pack()` return a `PackedTensor` named tuple bundling the
packed tensor with optional alloc shape/dtype overrides. The format knows
whether packing changed the representation; it captures that metadata at the
source instead of requiring the builder to reconstruct it.

### New type

```python
class PackedTensor(NamedTuple):
    tensor:      torch.Tensor
    alloc_shape: list[int] | None       # None = inherit from tensor
    alloc_dtype: "_tm.DataType | None"  # None = inherit from tensor
```

### Base implementation

```python
class WeightFormat(ABC):
    def pack(self, tensor, kind) -> PackedTensor:
        return PackedTensor(tensor, None, None)
```

### Quantized overrides

All four quantized formats (Int4Format, AWQInt4Format,
CompressedTensorFormat, FP8Format) follow the same pattern:

```python
def pack(self, tensor, kind) -> PackedTensor:
    if kind == "weight" and tensor.dtype == torch.uint8:
        alloc_shape = list(tensor.shape)  # [in_dim, out_dim] pre-packing
        return PackedTensor(pack_u4_row(tensor), alloc_shape, self.weight_dtype)
    return PackedTensor(tensor, None, None)
```

`alloc_shape` is captured from the tensor *before* `pack_u4_row()` transforms
it, so `pack()` doesn't need `in_dim`/`out_dim` passed in.

### Builder usage

```python
for kind, tensor in tensors.items():
    packed = fmt.pack(t, k)
    shard = _shard(packed.tensor, kind_split_dims[kind], tp, rank)
    alloc_shape = packed.alloc_shape
    alloc_dtype = packed.alloc_dtype
    if alloc_dtype is None and kind == 'weight' and model_dtype is not None:
        alloc_dtype = model_dtype
    _copy_shard_to_param(mod, k, shard,
                         alloc_shape=alloc_shape, alloc_dtype=alloc_dtype)
```

The `model_dtype` override is the only remaining branch — it's a
deployment-level concern (preferred compute dtype), not a format concern.

### Removed

- `is_quantized` local (line 465) — no longer needed
- `weight_cpp_dtype` local (line 443) — no longer needed in the loop

### Decision matrix

| Case | packed.alloc_shape | packed.alloc_dtype | model_dtype applied? |
|------|-------------------|-------------------|---------------------|
| Quantized weight (UINT4/FP8) | `[in_dim, out_dim]` | `TYPE_UINT4` / `TYPE_FP8_E4M3` | no (already set) |
| Trivial weight + model_dtype | `None` | `None` | yes |
| Trivial weight, no model_dtype | `None` | `None` | no |
| Bias / scales / zeros | `None` | `None` | no (not weight) |

## Files changed

- `weight_format.py`: add `PackedTensor`, update base `pack()` and 4 quantized overrides
- `_base.py`: simplify `_commit_linear` loop, remove `is_quantized` and `weight_cpp_dtype`

## Invariants preserved

- `dst.byte_size == shard.nbytes` — alloc relabel only changes the logical view, never byte count
- All tensors go through `_copy_shard_to_param` with the same signature
- `model_dtype` still controls the C++ Linear compute dtype via `lin_cfg.data_type`
