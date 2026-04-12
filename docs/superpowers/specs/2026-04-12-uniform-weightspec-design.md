# Uniform WeightSpec for Dense Linear Weights

Date: 2026-04-12

## Summary

Fix `_infer_cpp_linear_dtype` to handle all dense dtypes (including FP32) so `set_weight_spec` is always called, removing the None-guard workaround.

## Problem

`_infer_cpp_linear_dtype` hard-codes BF16/FP16 dtype checks for dense weights:

```python
if weight.dtype == torch.bfloat16:
    return _tm.DataType.TYPE_BF16, 0
if weight.dtype == torch.float16:
    return _tm.DataType.TYPE_FP16, 0
return None, 0  # FP32 and everything else -> None
```

FP32 dense weights (MoE gate weights in quantized models) produce `cpp_dtype=None`, forcing a guard before `set_weight_spec`:

```python
if cpp_dtype is not None:
    linear_mod.set_weight_spec(cpp_dtype, group_size)
```

Without the guard, `TYPE_INVALID` reaches `MakeLinearWeightFormat` and crashes.

## Fix

Replace the hard-coded dtype checks with `_TORCH_TO_CPP.get(weight.dtype)`, which already maps all torch dtypes to C++ DataType values (FP32 → `TYPE_FP32`, BF16 → `TYPE_BF16`, etc.).

### Changes in `lmdeploy/turbomind/deploy/commit.py`

1. **Simplify `_infer_cpp_linear_dtype`** — replace the hard-coded BF16/FP16 checks (lines 73-80) with:

```python
    weight = linear.tensors.get("weight")
    if weight is not None:
        return _TORCH_TO_CPP.get(weight.dtype), 0
    return None, 0
```

2. **Remove the None guard** — make `set_weight_spec` unconditional:

```python
    linear_mod.set_weight_spec(cpp_dtype, group_size)
```

### What stays the same

- Quantized path: `cpp_dtype_name` → `getattr(_tm.DataType, ...)` still runs first, unchanged.
- `_commit_tensors` allocation logic: no changes needed.
- C++ side: no changes needed — `set_weight_spec` already handles dense-float coercion (FP32 weight in BF16 model gets coerced to BF16).

## Files affected

- `lmdeploy/turbomind/deploy/commit.py` — simplify `_infer_cpp_linear_dtype`, remove None guard
