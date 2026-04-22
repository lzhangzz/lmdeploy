# FFN transform_tensors Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace manual tensor-kind iteration in FFN with `@transform_output_dim` / `@transform_input_dim` decorators, rename `transform_tensors`, move fusion functions to ffn.py, and clean up dead code.

**Architecture:** Two decorators in `_base.py` — `transform_output_dim` (renamed from `transform_tensors`, unsqueezes 1-D) and `transform_input_dim` (passes 1-D through unchanged). FFN fusion and padding helpers become thin `@transform_output_dim` / `@transform_input_dim` functions in `ffn.py`. `interleave_linears`, `chunk_linears`, and `preprocess_linear` are removed from `linear.py`.

**Tech Stack:** Python 3, PyTorch, pytest

---

### Task 1: Rename `transform_tensors` to `transform_output_dim` in `_base.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/_base.py:164`

- [ ] **Step 1: Rename the function definition**

In `_base.py`, change line 164:

```python
def transform_output_dim(fn):
```

Update the docstring to clarify the naming:

```python
def transform_output_dim(fn):
    """Decorator that lifts a tensor-level transform to Linear-level.

    For output-dim operations: 1-D tensors (bias) are unsqueezed to 2-D
    before calling *fn*, then squeezed back.  Convention: args that are
    ``Linear`` instances are treated as tensor inputs; all other args pass
    through unchanged.  Return type is detected at runtime:
    ``Tensor`` -> single ``Linear``, ``tuple`` -> tuple of ``Linear`` objects.
    """
```

The rest of the body stays identical.

- [ ] **Step 2: Update the section comment**

Change the section comment on line 160:

```python
# ---------------------------------------------------------------------------
# @transform_output_dim / @transform_input_dim decorators
# ---------------------------------------------------------------------------
```

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/_base.py
git commit -m "refactor: rename transform_tensors to transform_output_dim"
```

---

### Task 2: Update `attention.py` references

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/attention.py:15,49,72,85`

- [ ] **Step 1: Update import**

Line 15, change:

```python
from ._base import Builder, SplitSide, _dequant_linear, transform_output_dim
```

- [ ] **Step 2: Update three decorator uses**

Lines 49, 72, 85 — replace `@transform_tensors` with `@transform_output_dim`:

```python
@transform_output_dim
def _repeat_kv_heads(...):

@transform_output_dim
def split_output_gate(...):

@transform_output_dim
def fuse_qkv(...):
```

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/attention.py
git commit -m "refactor: update attention.py to use transform_output_dim"
```

---

### Task 3: Add `transform_input_dim` decorator to `_base.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/_base.py`
- Test: `tests/test_lmdeploy/test_turbomind/test_transform_tensors.py`

- [ ] **Step 1: Write failing tests for `transform_input_dim`**

Add these tests to `tests/test_lmdeploy/test_turbomind/test_transform_tensors.py`. First update the import on line 97:

```python
transform_output_dim = _base_mod.transform_output_dim
transform_input_dim = _base_mod.transform_input_dim
```

Then add a new test class at the end of the file:

```python
class TestTransformInputDim:

    def test_2d_transformed(self):
        """2-D tensors are passed through the inner function."""

        @transform_input_dim
        def pad_first_dim(tensor: torch.Tensor,
                          *, target: int) -> torch.Tensor:
            return torch.nn.functional.pad(
                tensor, [0, 0, 0, target - tensor.size(0)])

        lin = _make_linear(out_dim=4, in_dim=2)
        result = pad_first_dim(lin, target=6)
        assert isinstance(result, Linear)
        assert result.tensors['weight'].shape == (6, 4)

    def test_1d_passthrough(self):
        """1-D tensors (bias) pass through unchanged."""

        @transform_input_dim
        def pad_first_dim(tensor: torch.Tensor,
                          *, target: int) -> torch.Tensor:
            return torch.nn.functional.pad(
                tensor, [0, 0, 0, target - tensor.size(0)])

        lin = _make_linear(out_dim=4)  # 1-D weight
        result = pad_first_dim(lin, target=6)
        assert isinstance(result, Linear)
        assert result.tensors['weight'].shape == (4,)  # unchanged

    def test_mixed_dims_2d_transformed_1d_passthrough(self):
        """2-D weight is transformed; 1-D bias passes through."""

        @transform_input_dim
        def double_input_dim(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.repeat(2, 1)

        lin = _make_linear(out_dim=4, in_dim=3, has_bias=True)
        result = double_input_dim(lin)
        assert isinstance(result, Linear)
        assert set(result.tensors) == {'weight', 'bias'}
        assert result.tensors['weight'].shape == (6, 4)  # doubled
        assert result.tensors['bias'].shape == (4,)  # unchanged

    def test_1in_2out_distributes_1d(self):
        """Multi-output: 1-D tensors duplicated into all output buckets."""

        @transform_input_dim
        def split_input(tensor: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor]:
            mid = tensor.size(0) // 2
            return tensor[:mid], tensor[mid:]

        lin = _make_linear(out_dim=4, in_dim=6, has_bias=True)
        a, b = split_input(lin)
        assert isinstance(a, Linear)
        assert isinstance(b, Linear)
        assert a.tensors['weight'].shape == (3, 4)
        assert b.tensors['weight'].shape == (3, 4)
        assert a.tensors['bias'].shape == (4,)  # duplicated
        assert b.tensors['bias'].shape == (4,)  # duplicated
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_lmdeploy/test_turbomind/test_transform_tensors.py -v`
Expected: FAIL — `transform_input_dim` does not exist yet.

- [ ] **Step 3: Implement `transform_input_dim`**

Add this function in `_base.py`, right after `transform_output_dim`:

```python
def transform_input_dim(fn):
    """Decorator that lifts a tensor-level transform to Linear-level.

    For input-dim operations: 1-D tensors (bias) have no input dimension
    and are **passed through unchanged**.  The inner function only ever
    sees 2-D tensors for each kind.  For multi-output functions, 1-D
    tensors are duplicated into every output bucket.
    """
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        first = next(v for v in bound.arguments.values()
                     if isinstance(v, Linear))
        out_buckets = None
        deferred_1d: list[str] = []

        for kind in first.tensors:
            fn_kwargs = {}
            is_1d = False

            for name, val in bound.arguments.items():
                if isinstance(val, Linear):
                    t = val.tensors[kind]
                    if t.dim() < 2:
                        is_1d = True
                        break
                    fn_kwargs[name] = t
                else:
                    fn_kwargs[name] = val

            if is_1d:
                deferred_1d.append(kind)
                continue

            result = fn(**fn_kwargs)
            if not isinstance(result, tuple):
                result = (result,)
            if out_buckets is None:
                out_buckets = [{} for _ in result]
            for i, item in enumerate(result):
                out_buckets[i][kind] = item

        if out_buckets is None:
            out_buckets = [{}]
        for kind in deferred_1d:
            for bucket in out_buckets:
                bucket[kind] = first.tensors[kind]

        outputs = tuple(
            Linear(ts, weight_format=first.weight_format,
                   data_format=first.data_format)
            for ts in out_buckets)
        return outputs if len(outputs) > 1 else outputs[0]

    return wrapper
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lmdeploy/test_turbomind/test_transform_tensors.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/_base.py tests/test_lmdeploy/test_turbomind/test_transform_tensors.py
git commit -m "feat: add transform_input_dim decorator with 1-D passthrough"
```

---

### Task 4: Update test file for `transform_output_dim` rename

**Files:**
- Modify: `tests/test_lmdeploy/test_turbomind/test_transform_tensors.py`

> **Note:** The import on line 97 was already updated in Task 3 Step 1.

- [ ] **Step 1: Update all decorator uses in existing tests**

Throughout the `TestTransformTensors` class, replace all `@transform_tensors` with `@transform_output_dim` (lines 142, 157, 171, 193, 210, 225, 243, 259, 276, 293).

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_lmdeploy/test_turbomind/test_transform_tensors.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_lmdeploy/test_turbomind/test_transform_tensors.py
git commit -m "test: update tests for transform_output_dim rename"
```

---

### Task 5: Rewrite `ffn.py` — move fusion functions and simplify padding

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/ffn.py`

This is the main task. The final `ffn.py` should look like this:

- [ ] **Step 1: Update imports**

Replace the current import block (lines 15-17):

```python
from ..linear import (Linear, chunk_linears as _chunk_linears,
                       interleave_linears as _interleave_linears,
                       pad_in_dim, pad_out_dim)
from ._base import Builder, SplitSide
```

With:

```python
from ..linear import Linear, pad_in_dim, pad_out_dim
from ._base import Builder, SplitSide, transform_input_dim, transform_output_dim
```

- [ ] **Step 2: Update `__all__`**

Change line 22 from `'fuse_ffn_linears'` to `'fuse_w1w3'`:

```python
__all__ = [
    'FfnBuilder',
    'fuse_w1w3',
]
```

- [ ] **Step 3: Add `@transform_output_dim` fusion helpers**

Add these after the imports, before the existing helpers (`_should_fuse_silu`, etc.):

```python
# ---------------------------------------------------------------------------
# @transform_output_dim / @transform_input_dim helpers
# ---------------------------------------------------------------------------


@transform_output_dim
def _interleave_w1w3(w1: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    """Interleave w1 and w3 along output dim for fused SiLU epilogue."""
    return torch.stack([w1, w3], dim=-1).reshape(w1.shape[:-1] + (-1,)).contiguous()


@transform_output_dim
def _chunk_w1w3(w1: torch.Tensor, w3: torch.Tensor, *,
                tp: int) -> torch.Tensor:
    """Concatenate w1 and w3 along output dim with TP interleaving."""
    if tp <= 1:
        return torch.cat([w1, w3], dim=-1).contiguous()
    d = w1.dim() - 1
    r1 = w1.reshape(w1.shape[:d] + [tp, w1.shape[d] // tp])
    r3 = w3.reshape(w3.shape[:d] + [tp, w3.shape[d] // tp])
    combined = torch.cat([r1, r3], dim=d + 1)
    return combined.reshape(w1.shape[:d] + [-1]).contiguous()


@transform_output_dim
def _pad_out(tensor: torch.Tensor, *, target: int) -> torch.Tensor:
    """Pad output dimension to target size."""
    return pad_out_dim(tensor, target, dim=tensor.dim() - 1)


@transform_input_dim
def _pad_in(tensor: torch.Tensor, *, target: int) -> torch.Tensor:
    """Pad input dimension to target size (1-D tensors pass through)."""
    return pad_in_dim(tensor, target, dim=0)
```

Add `import torch` to the imports if not already present.

- [ ] **Step 4: Rewrite `_pad_ffn_for_tp` to use decorators**

Replace the entire `_pad_ffn_for_tp` function (lines 112-160) with:

```python
def _pad_ffn_for_tp(w1: Linear, w2: Linear, w3: Linear,
                     tp: int) -> tuple[Linear, Linear, Linear]:
    """Pad w1/w3 output dim and w2 input dim for TP sharding."""
    raw_inter = w1.tensors['weight'].size(-1)

    if tp <= 1:
        return w1, w2, w3
    fmt = w1.weight_format
    block_out = (fmt.block_out or 1) if fmt else 1
    block_in = (fmt.block_in or 1) if fmt else 1
    effective_block = math.lcm(block_in, block_out) if block_in != block_out else block_out

    groups = (raw_inter + effective_block - 1) // effective_block
    groups_per_rank = (groups + tp - 1) // tp
    padded_inter = groups_per_rank * effective_block * tp
    if padded_inter == raw_inter:
        return w1, w2, w3

    w1 = _pad_out(w1, target=padded_inter)
    w3 = _pad_out(w3, target=padded_inter)
    w2 = _pad_in(w2, target=padded_inter)
    return w1, w2, w3
```

- [ ] **Step 5: Rename `fuse_ffn_linears` to `fuse_w1w3` and use new helpers**

Replace the `fuse_ffn_linears` function (lines 77-104) with:

```python
def fuse_w1w3(
    w1: Linear,
    w3: Linear,
    tp: int,
    act_type: str,
    is_moe: bool = False,
) -> tuple[Linear | None, bool]:
    """Optionally fuse w1/w3 on full (unsharded) tensors for FFN.

    Returns (fused_w1w3_or_none, fused_silu).
    When fusion is possible, fused_w1w3 is set.
    When block-scale boundaries prevent fusion, returns (None, fused_silu).

    TP sharding is NOT done here — the caller's commit path handles it
    via split_side=SplitSide.OUTPUT.  ``tp`` is only used for the
    block-scale alignment check in ``_can_fuse_w1w3``.
    """
    fused_silu = _should_fuse_silu(w1, act_type, is_moe)
    can_fuse = _can_fuse_w1w3(w1, tp)

    if can_fuse:
        if fused_silu:
            w1w3 = _interleave_w1w3(w1, w3)
        else:
            w1w3 = _chunk_w1w3(w1, w3, tp=tp)
        return (w1w3, fused_silu)
    else:
        return (None, fused_silu)
```

- [ ] **Step 6: Update `add_ffn` call site**

In `FfnBuilder.add_ffn` (line 184), change:

```python
        fused, fused_silu = fuse_w1w3(
```

- [ ] **Step 7: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/ffn.py
git commit -m "refactor: rewrite ffn.py with transform_output_dim/input_dim"
```

---

### Task 6: Update `builder/__init__.py` for rename

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/__init__.py:10,31`

- [ ] **Step 1: Update import and __all__**

Line 10 — change:

```python
from .ffn import FfnBuilder, fuse_w1w3
```

Line 31 — change:

```python
    'fuse_w1w3',
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/__init__.py
git commit -m "refactor: rename fuse_ffn_linears to fuse_w1w3 in __init__"
```

---

### Task 7: Remove `interleave_linears`, `chunk_linears`, and `preprocess_linear` from `linear.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/linear.py:199-266`

- [ ] **Step 1: Delete the three functions**

Delete `preprocess_linear` (lines 199-213), `interleave_linears` (lines 216-234), and `chunk_linears` (lines 237-266).

Also remove the now-empty section comment:

```python
# ---------------------------------------------------------------------------
# Fusion helpers (w1 + w3 → w1w3)
# ---------------------------------------------------------------------------
```

- [ ] **Step 2: Verify nothing else in the codebase is broken**

Run: `pytest tests/test_lmdeploy/test_turbomind/test_transform_tensors.py -v`
Expected: ALL PASS

Also try importing the module:
```bash
python -c "from lmdeploy.turbomind.deploy.linear import Linear; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/linear.py
git commit -m "refactor: remove interleave_linears, chunk_linears, preprocess_linear from linear.py"
```

---

### Task 8: Run a model test to verify end-to-end correctness

**Files:**
- None (verification only)

- [ ] **Step 1: Check GPU availability**

Run: use `get_gpu_usage` MCP tool to find an empty GPU.

- [ ] **Step 2: Run a model test**

Pick any model from `list_models` (e.g., a small Llama model). Run:

```bash
cd /data/lmdeploy-modeling/build && python scripts/test_turbomind_model.py <model_id> --prompt "Hello, how are you?" --tokens 128
```

Verify the response contains meaningful human words. Gibberish = bug.

- [ ] **Step 3: Run a second model if available (different architecture)**

This catches edge cases in different FFN configurations (e.g., different quantization, different act types).
