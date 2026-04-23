# Format Handling Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate all format handling into `kind_map.py`, delete dead legacy files (`policy.py`, `parameter.py`), and deduplicate commit logic between `module.py` and `load_context.py`.

**Architecture:** Move `build_linear()` and `pack_u4_row()` into `kind_map.py` (the existing format authority). Delete `policy.py` and `parameter.py` entirely. Extract the tensor-commit loop from `commit_linear()` into a shared `_commit_tensors()` function that both `module.py` and `load_context.py` call.

**Tech Stack:** Python only (no C++ changes). Files in `lmdeploy/turbomind/deploy/`.

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `lmdeploy/turbomind/deploy/kind_map.py` | Add `build_linear()`, `pack_u4_row()` |
| Delete | `lmdeploy/turbomind/deploy/policy.py` | Dead normalizer functions |
| Delete | `lmdeploy/turbomind/deploy/parameter.py` | Dead Parameter classes |
| Modify | `lmdeploy/turbomind/deploy/module.py` | Update import, extract `_commit_tensors()` |
| Modify | `lmdeploy/turbomind/deploy/load_context.py` | Remove `_commit_linear_to_handle`, use `_commit_tensors` |
| Modify | `lmdeploy/turbomind/deploy/converter.py` | Remove `input_policy` plumbing |
| Modify | `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` | Remove `self.policy` |
| Modify | `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` | Remove `self.policy`, update import |
| Modify | `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` | Remove `self.policy`, update import |
| Modify | `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` | Remove `self.policy` |

---

### Task 1: Move `build_linear()` and `pack_u4_row()` into `kind_map.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/kind_map.py` (append functions)
- Modify: `lmdeploy/turbomind/deploy/parameter.py:16-56,83-89` (source of moved code)

`build_linear()` imports `ALL_SUFFIXES`, `FORMAT_PRIORITY` from `kind_map.py` and `Linear` from `linear.py`. After moving it into `kind_map.py`, the self-imports become local references. The `Linear` import stays.

`pack_u4_row()` has no dependencies beyond `torch`.

- [ ] **Step 1: Add `build_linear()` and `pack_u4_row()` to `kind_map.py`**

Append at the end of `kind_map.py` (after the `ALL_SUFFIXES` definition on line 537):

```python
# ---------------------------------------------------------------------------
# build_linear — auto-detect format and build Linear from checkpoint tensors
# ---------------------------------------------------------------------------


def build_linear(
    params: dict[str, torch.Tensor],
    prefix: str,
    index: int | None = None,
) -> 'Linear | None':
    """Build a ``Linear`` bundle from checkpoint tensors at *prefix*.

    Probes every known checkpoint suffix (union of all format suffix maps),
    classifies the format by running each ``WeightFormat.accepts`` predicate
    in ``FORMAT_PRIORITY`` order, then normalises the collected tensors with
    the winning format's normalizer.

    When *index* is given, each collected tensor is sliced by ``[index]``
    before classification and normalisation (used for packed expert tensors
    where the expert dimension is the leading axis).

    The returned ``Linear`` is in TM layout ``[in, out]`` and carries the
    detected ``WeightFormat`` for downstream use in ``commit_linear``.
    Returns ``None`` if no tensors are found at *prefix*.
    """
    from .linear import Linear

    available: dict[str, torch.Tensor] = {
        s: params[prefix + s] for s in ALL_SUFFIXES if (prefix + s) in params
    }
    if index is not None:
        available = {s: t[index] for s, t in available.items()}

    fmt = next((f for f in FORMAT_PRIORITY if f.accepts(available)), None)
    if fmt is None:
        return None

    tensors: dict[str, torch.Tensor] = {
        kind: fmt.normalizer(available[s], kind)
        for s, kind in fmt.suffix_map.items()
        if s in available
    }
    if not tensors:
        return None

    fmt.complete_tensors(tensors)
    data_format = fmt.to_data_format(0, group_size=0)
    return Linear(tensors=tensors, weight_format=fmt, data_format=data_format)


# ---------------------------------------------------------------------------
# pack_u4_row — uint8 4-bit packing utility
# ---------------------------------------------------------------------------


def pack_u4_row(x: torch.Tensor) -> torch.Tensor:
    """Pack uint8 4-bit values into int32 rows.

    Each group of 8 consecutive uint8 values is packed into one int32 where
    the first element occupies the least significant nibble.
    """
    assert x.dtype == torch.uint8, f'x.dtype: {x.dtype}'
    xs = x.view(*x.shape[:-1], -1, 8).split(1, dim=-1)
    a = torch.zeros(xs[0].shape, dtype=torch.int32, device=x.device)
    for t in reversed(xs):
        a = (a << 4) | t
    return a.squeeze(dim=-1)
```

- [ ] **Step 2: Remove lazy imports from packer functions in `kind_map.py`**

The packer functions `_pack_u4_qweight` (line 301) and `_pack_mxfp4_weight` (line 309) currently do `from .parameter import pack_u4_row` inside the function body. Since `pack_u4_row` is now in the same file, change them to direct calls.

In `kind_map.py`, replace:

```python
def _pack_u4_qweight(tensor: Tensor, kind: str) -> Tensor:
    """Pack uint8 4-bit values into int32 rows; applied to ``qweight``."""
    if kind == "qweight" and tensor.dtype == torch.uint8:
        from .parameter import pack_u4_row
        return pack_u4_row(tensor)
    return tensor
```

with:

```python
def _pack_u4_qweight(tensor: Tensor, kind: str) -> Tensor:
    """Pack uint8 4-bit values into int32 rows; applied to ``qweight``."""
    if kind == "qweight" and tensor.dtype == torch.uint8:
        return pack_u4_row(tensor)
    return tensor
```

Similarly replace:

```python
def _pack_mxfp4_weight(tensor: Tensor, kind: str) -> Tensor:
    """Pack uint8 4-bit values into int32 rows; applied to mxfp4 ``weight``."""
    if kind == "weight" and tensor.dtype == torch.uint8:
        from .parameter import pack_u4_row
        return pack_u4_row(tensor)
    return tensor
```

with:

```python
def _pack_mxfp4_weight(tensor: Tensor, kind: str) -> Tensor:
    """Pack uint8 4-bit values into int32 rows; applied to mxfp4 ``weight``."""
    if kind == "weight" and tensor.dtype == torch.uint8:
        return pack_u4_row(tensor)
    return tensor
```

- [ ] **Step 3: Update `module.py` import**

In `lmdeploy/turbomind/deploy/module.py` line 171, change:

```python
            from .parameter import build_linear
```

to:

```python
            from .kind_map import build_linear
```

- [ ] **Step 4: Update `qwen3_5_spec.py` import**

In `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` line 19, change:

```python
from ..parameter import build_linear
```

to:

```python
from ..kind_map import build_linear
```

- [ ] **Step 5: Update `gpt_oss_spec.py` import**

In `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` line 23, change:

```python
from ..parameter import build_linear
```

to:

```python
from ..kind_map import build_linear
```

- [ ] **Step 6: Commit**

```bash
git add lmdeploy/turbomind/deploy/kind_map.py
git add lmdeploy/turbomind/deploy/module.py
git add lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py
git add lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py
git commit -m "refactor(format): move build_linear and pack_u4_row to kind_map.py"
```

---

### Task 2: Delete `parameter.py`

**Files:**
- Delete: `lmdeploy/turbomind/deploy/parameter.py`

After Task 1, the only symbols left in `parameter.py` are the dead old API: `Parameter` base class, `QuantWeightOnly`, `WeightScaleInv`, `Mxfp4Weight`, `Weight`, `Bias`, `PLora`, `get_params()`, plus the dtype helpers (`identity`, `to_half`, `to_float`, `to_fp8`, `generate_zero_point`). None of these have callers outside `parameter.py`.

- [ ] **Step 1: Verify no remaining importers**

```bash
grep -r "from .parameter import\|from ..parameter import\|from . import parameter\|import parameter" lmdeploy/turbomind/deploy/ --include="*.py"
```

Expected: no results (all imports updated in Task 1).

- [ ] **Step 2: Delete the file**

```bash
git rm lmdeploy/turbomind/deploy/parameter.py
```

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor(format): delete dead parameter.py (legacy Parameter classes)"
```

---

### Task 3: Delete `policy.py` and remove `input_policy` plumbing

**Files:**
- Delete: `lmdeploy/turbomind/deploy/policy.py`
- Modify: `lmdeploy/turbomind/deploy/converter.py:12,185,189`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py:141`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py:334,366`
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py:196`
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py:272`

`policy.py` exports `get_input_policy()` which is called in `converter.py` and passed to source model constructors. Source models store it as `self.policy` but never call it. This is all dead plumbing.

- [ ] **Step 1: Remove import and usage from `converter.py`**

In `lmdeploy/turbomind/deploy/converter.py`:

Remove line 12:
```python
from .policy import get_input_policy
```

Replace lines 184-189:
```python
    fp8_quant = (engine_config.model_format == 'fp8' and not quant_config)
    input_policy = get_input_policy(engine_config.model_format)
    _model_cls = INPUT_MODELS.get(input_model_name)
    input_model = _model_cls(model_path=model_path,
                             tokenizer_path=model_path,
                             input_policy=input_policy,
                             fp8_quant=fp8_quant,
                             model_format=engine_config.model_format)
```

with:
```python
    fp8_quant = (engine_config.model_format == 'fp8' and not quant_config)
    _model_cls = INPUT_MODELS.get(input_model_name)
    input_model = _model_cls(model_path=model_path,
                             tokenizer_path=model_path,
                             fp8_quant=fp8_quant,
                             model_format=engine_config.model_format)
```

- [ ] **Step 2: Remove `self.policy` from source model specs**

In each file, remove the line `self.policy = kwargs.get('input_policy')`:

- `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` line 141: remove `self.policy = kwargs.get('input_policy')`
- `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` line 334: remove `self.policy = kwargs.get('input_policy')`
- `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` line 366: remove `self.policy = kwargs.get('input_policy')`
- `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` line 196: remove `self.policy = kwargs.get('input_policy')`
- `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` line 272: remove `self.policy = kwargs.get('input_policy')`

- [ ] **Step 3: Verify no remaining references to policy.py**

```bash
grep -r "from .policy import\|import policy\|get_input_policy\|self\.policy" lmdeploy/turbomind/deploy/ --include="*.py"
```

Expected: no results.

- [ ] **Step 4: Delete `policy.py`**

```bash
git rm lmdeploy/turbomind/deploy/policy.py
```

- [ ] **Step 5: Commit**

```bash
git add lmdeploy/turbomind/deploy/converter.py
git add lmdeploy/turbomind/deploy/source_model/qwen3_spec.py
git add lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py
git add lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py
git add lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py
git commit -m "refactor(format): delete dead policy.py and input_policy plumbing"
```

---

### Task 4: Extract `_commit_tensors()` and deduplicate commit logic

**Files:**
- Modify: `lmdeploy/turbomind/deploy/module.py:939-1008` (extract tensor loop)
- Modify: `lmdeploy/turbomind/deploy/load_context.py:107-195` (use shared function)

The tensor-commit loop in `commit_linear()` (module.py lines 939-1008) and `_commit_linear_to_handle()` (load_context.py lines 144-195) are nearly identical. Extract into a shared `_commit_tensors()` function in `module.py`.

- [ ] **Step 1: Add `_commit_tensors()` to `module.py`**

Insert this function right before `commit_linear()` (before line 864):

```python
def _commit_tensors(handle, linear: Linear, cpp_dtype, group_size: int,
                    split_side: SplitSide | None, split_num: int, rank: int):
    """Commit tensor data from a ``Linear`` to a pre-created C++ LinearWeight handle.

    Handles packing, TP sharding, allocation, dtype casting, and padding.
    This is the shared tensor-commit loop used by both ``commit_linear`` and
    ``LoadContext.load_linear``.
    """
    split_dim = _SPLIT_SIDE_TO_DIM.get(split_side) if split_side else None

    packer = linear.weight_format.packer if linear.weight_format else None

    # Process "weight"/"qweight" first so that LinearWeight::alloc() triggers
    # do_allocate() before we encounter "scales"/"zeros"/"bias".  Without this,
    # lazy allocation returns an empty tensor for scales if it is iterated first.
    def _kind_order(item):
        k, _ = item
        if k in ("weight", "qweight"):
            return (0, k)
        return (1, k)

    for kind, tensor in sorted(linear.tensors.items(), key=_kind_order):
        if packer is not None:
            tensor = packer(tensor, kind)

        # Bias is NOT split for row-parallel (INPUT-split) linears — it is
        # replicated across all TP ranks and added after the all-reduce.
        tensor_split_dim = split_dim
        if kind == "bias" and split_side == SplitSide.INPUT:
            tensor_split_dim = None

        # Extract the shard for this rank
        if tensor_split_dim is not None and split_num > 1:
            split_size = tensor.shape[tensor_split_dim] // split_num
            shard = tensor.split(split_size, dim=tensor_split_dim)[rank]
        else:
            shard = tensor

        shard = shard.cuda().contiguous()

        # Allocate (first call triggers full allocation) and copy
        dst = handle.alloc(kind, cpp_dtype, group_size)
        if dst:
            shard = _cast_shard_for_tm(shard, dst)
            # Pad shard with zeros when C++ allocation is larger (e.g. due to
            # _pad_inter_size ensuring group_size alignment for TP splitting).
            if dst.byte_size != shard.nbytes and dst.byte_size > shard.nbytes:
                pad_dim = tensor_split_dim if tensor_split_dim is not None else -1
                if pad_dim < 0:
                    pad_dim = shard.dim() + pad_dim
                outer = shard.numel() // shard.shape[pad_dim]
                extra = (dst.byte_size - shard.nbytes) // (outer * shard.element_size())
                new_shape = list(shard.shape)
                new_shape[pad_dim] += extra
                padded = torch.zeros(new_shape, dtype=shard.dtype, device=shard.device)
                idx = [slice(None)] * shard.dim()
                idx[pad_dim] = slice(0, shard.shape[pad_dim])
                padded[tuple(idx)].copy_(shard)
                shard = padded
            dst.copy_from(shard)
```

- [ ] **Step 2: Refactor `commit_linear()` to call `_commit_tensors()`**

In `module.py`, replace `commit_linear()`. Keep the preamble (dtype inference, DataFormat creation, child creation, block-scale validation) but replace the tensor loop with a call to `_commit_tensors()`. The new `commit_linear()` is:

```python
def commit_linear(module, linear: Linear, name: str,
                         split_side: SplitSide | None = None,
                         split_num: int = 1, rank: int = 0,
                         copy: bool = False, model_dtype=None):
    """Commit a ``Linear`` bundle to a C++ ``Module`` handle for a specific TP rank.

    Parameters
    ----------
    module : C++ Module handle
        Parent module (e.g. an ``AttentionWeight``).
    linear : Linear
        The linear bundle to commit.
    name : str
        Child module name within *module* (e.g. ``"w_qkv"``).
    split_side : SplitSide | None
        TP split semantics.
    split_num : int
        Number of TP shards.
    rank : int
        Which shard to extract and copy.
    copy : bool
        If ``True``, copy the tensor as-is (no split).
    model_dtype : int | None
        The model's configured compute dtype (C++ DataType value).  When set,
        dense (non-quantized) weights use this dtype instead of the weight
        tensor's dtype.
    """
    cpp_dtype, group_size = _infer_cpp_linear_dtype(linear)
    if group_size == 0:
        group_size = max(1, 128)

    # Ensure the Linear has a DataFormat attached (deferred creation for formats
    # like AWQ/GPTQ where group_size is not known at build_linear time).
    if linear.data_format is None and linear.weight_format is not None:
        linear = Linear(tensors=linear.tensors,
                        weight_format=linear.weight_format,
                        data_format=linear.weight_format.to_data_format(
                            cpp_dtype.value if cpp_dtype else 0,
                            group_size))

    # Ensure the LinearWeight child exists
    linear_mod = module.child(name)
    if linear_mod is None:
        w = linear.tensors.get('weight')
        if w is None:
            w = linear.tensors.get('qweight')
        in_dim = w.shape[0]
        out_dim = w.shape[-1]
        if split_side == SplitSide.OUTPUT:
            out_dim = out_dim // split_num
        elif split_side == SplitSide.INPUT:
            in_dim = in_dim // split_num
        compute_dtype = _infer_compute_dtype(linear)
        if model_dtype is not None and compute_dtype is not None:
            fmt = linear.weight_format
            if fmt is None or fmt.name == 'dense':
                import _turbomind as _tm
                model_dt = _tm.DataType(model_dtype) if isinstance(model_dtype, int) else model_dtype
                compute_dtype = model_dt
        linear_mod = module.create_child(name, 'LinearWeight', {
            'input_dim': in_dim,
            'output_dim': out_dim,
            'data_type': compute_dtype.value if compute_dtype else 0,
            'has_bias': 1 if 'bias' in linear.tensors else 0,
        })

    # Block-scale TP split validation
    if split_side == SplitSide.OUTPUT and split_num > 1:
        wfmt = linear.weight_format
        if wfmt is not None and wfmt.block_out:
            for kind, tensor in linear.tensors.items():
                if kind in ("scales", "zeros"):
                    n_blocks = tensor.size(-1)
                    assert n_blocks % split_num == 0, (
                        f"TP split: {name}.{kind} has {n_blocks} output-dimension "
                        f"scale blocks (block_out={wfmt.block_out}), not "
                        f"divisible by split_num={split_num}.")

    _commit_tensors(linear_mod, linear, cpp_dtype, group_size,
                    split_side, split_num, rank)
```

- [ ] **Step 3: Refactor `load_context.py` to use `_commit_tensors`**

In `lmdeploy/turbomind/deploy/load_context.py`:

Replace the `load_linear` method (lines 107-142) and the entire `_commit_linear_to_handle` static method (lines 144-195) with:

```python
    def load_linear(self, name: str, linear: Linear,
                    tp_rule: str | None = None):
        """Create a LinearWeight child and commit weight data.

        Handles TP splitting, quantization packing, and dtype casting.
        The child is created via create_child, then weights are committed
        using the shared _commit_tensors function.
        """
        from .module import SplitSide as _SplitSide
        from .module import _commit_tensors, _infer_cpp_linear_dtype

        # Infer C++ dtype and group_size from the Linear
        cpp_dtype, group_size = _infer_cpp_linear_dtype(linear)
        if group_size == 0:
            group_size = max(1, 128)

        # Extract dimensions from weight tensor shape
        weight = linear.tensors.get('weight') or linear.tensors.get('qweight')
        input_dim = weight.shape[0] if weight is not None else 0
        output_dim = weight.shape[-1] if weight is not None else 0

        # Create the LinearWeight child
        child_handle = self._handle.create_child(
            name, 'LinearWeight', {
                'input_dim': input_dim,
                'output_dim': output_dim,
                'data_type': cpp_dtype,
                'has_bias': 'bias' in linear.tensors,
            })

        # Commit the weight data
        tp_side = _SplitSide[tp_rule] if tp_rule else None
        split_num = self.tp_size if tp_side else 1
        _commit_tensors(child_handle, linear, cpp_dtype, group_size,
                        tp_side, split_num, self.rank)
```

Note: `_commit_linear_to_handle` is fully removed.

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/module.py
git add lmdeploy/turbomind/deploy/load_context.py
git commit -m "refactor(commit): extract _commit_tensors, deduplicate commit logic"
```

---

### Task 5: Verify with model tests (TP=1 and TP=2)

**Files:** None (verification only)

Run each model format at TP=1 and TP=2 using the turbomind-tester agent. All models must produce meaningful human-readable output.

- [ ] **Step 1: Test dense model TP=1**

Use turbomind-tester agent to test a dense BF16/FP16 model (e.g. Qwen3-4B) with TP=1. Verify output is meaningful English.

- [ ] **Step 2: Test dense model TP=2**

Same model with TP=2. Verify output is meaningful.

- [ ] **Step 3: Test AWQ model TP=1**

Test an AWQ-quantized model with TP=1.

- [ ] **Step 4: Test AWQ model TP=2**

Same AWQ model with TP=2.

- [ ] **Step 5: Test GPTQ model TP=1**

Test a GPTQ-quantized model with TP=1.

- [ ] **Step 6: Test GPTQ model TP=2**

Same GPTQ model with TP=2.

- [ ] **Step 7: Test FP8 model TP=1**

Test an FP8 model with TP=1.

- [ ] **Step 8: Test FP8 model TP=2**

Same FP8 model with TP=2.

- [ ] **Step 9: Test MXFP4 model TP=1**

Test an MXFP4 model with TP=1.

- [ ] **Step 10: Test MXFP4 model TP=2**

Same MXFP4 model with TP=2.

- [ ] **Step 11: Commit final state if all tests pass**

If all tests pass, the refactoring is complete. No additional commit needed unless there were fixes.
