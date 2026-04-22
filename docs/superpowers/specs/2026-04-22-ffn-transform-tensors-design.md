# Simplify FFN Transformations with `transform_output_dim` / `transform_input_dim`

## Motivation

`FfnBuilder` and `_pad_ffn_for_tp` manually iterate over `Linear.tensors` and
reconstruct `Linear` objects -- boilerplate that `transform_tensors` (used by
`AttentionBuilder`) eliminates. Meanwhile the sole `transform_tensors` decorator
doesn't distinguish between output-dim and input-dim semantics: it unsqueezes
1-D tensors, which is correct for output-dim ops but wrong for input-dim ops
where bias tensors must pass through unchanged.

## Changes

### 1. Rename `transform_tensors` to `transform_output_dim` (`_base.py`)

Pure rename. Update all references (3 in `attention.py`, 1 import in test file).

### 2. Add `transform_input_dim` decorator (`_base.py`)

Same structure as `transform_output_dim` except: when a tensor kind is 1-D
(has no input dimension), it is **passed through unchanged** — the inner
function is never called for that kind. For multi-output functions (returning
a tuple), 1-D tensors are duplicated into every output bucket (matching
`Linear.split_in_dim` behavior). The inner function only ever sees 2-D
tensors.

### 3. Move `interleave_linears` and `chunk_linears` to `ffn.py`

Rewrite as `@transform_output_dim`-decorated pure-tensor functions:

- `_interleave_w1w3(w1, w3)` -- stacks and reshapes
- `_chunk_w1w3(w1, w3, *, tp)` -- concatenates with TP interleaving

Remove from `linear.py`. Note: `_has_input_dim` stays in `linear.py` (used by
`Linear.split_in_dim` and `Linear.concat_in_dim`). `pad_out_dim` and
`pad_in_dim` also stay in `linear.py` (the former is used by `spec.py`).

### 4. Simplify `_pad_ffn_for_tp` (`ffn.py`)

Extract two decorated helpers:

```python
@transform_output_dim
def _pad_out(tensor, *, target):
    return pad_out_dim(tensor, target, dim=tensor.dim() - 1)

@transform_input_dim
def _pad_in(tensor, *, target):
    return pad_in_dim(tensor, target, dim=0)
```

`_pad_ffn_for_tp` calls these directly instead of defining nested functions.

### 5. Rename `fuse_ffn_linears` → `fuse_w1w3` (`ffn.py`)

Update imports: remove `chunk_linears`/`interleave_linears`, add
`transform_output_dim`/`transform_input_dim` from `._base`. Call the new
local `_interleave_w1w3` / `_chunk_w1w3` instead of the old imports.
Update all callers and the re-export in `builder/__init__.py`.

### 6. Delete `preprocess_linear` from `linear.py`

Dead code — no live callers.

Call `_interleave_w1w3` / `_chunk_w1w3` instead of the old `linear.py`
functions.

### 7. Update tests

Rename `transform_tensors` references to `transform_output_dim`. Add tests for
`transform_input_dim` (1-D passthrough, 2-D transformation).

## Files changed

| File | Change |
|---|---|
| File | Change |
|---|---|
| `lmdeploy/turbomind/deploy/builder/_base.py` | Rename + add `transform_input_dim` |
| `lmdeploy/turbomind/deploy/builder/attention.py` | Update decorator name |
| `lmdeploy/turbomind/deploy/builder/__init__.py` | Rename `fuse_ffn_linears` → `fuse_w1w3` |
| `lmdeploy/turbomind/deploy/builder/ffn.py` | Move functions here, simplify padding, rename `fuse_ffn_linears` |
| `lmdeploy/turbomind/deploy/linear.py` | Remove `interleave_linears`, `chunk_linears`, `preprocess_linear` |
| `tests/test_lmdeploy/test_turbomind/test_transform_tensors.py` | Update references, add tests |
