# `@transform_tensors` Decorator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the repeated 1D/2D tensor boilerplate in three `attention.py` functions with a single `@transform_tensors` decorator.

**Architecture:** The decorator inspects the inner function's parameter and return type annotations at decoration time. At call time it unwraps `Linear` objects into per-kind tensors, handles the 1D→2D roundtrip, calls the inner function on 2D tensors, and wraps results back into `Linear` objects. Lives in `_base.py`.

**Tech Stack:** Python 3.12+, `torch`, `typing.get_type_hints`.

---

### Task 1: Write unit tests for `@transform_tensors`

**Files:**
- Create: `tests/test_lmdeploy/test_turbomind/test_transform_tensors.py`

- [ ] **Step 1: Write the test file**

```python
"""Tests for the @transform_tensors decorator."""
import torch
import pytest

from lmdeploy.turbomind.deploy.linear import Linear


def _make_linear(out_dim, in_dim=None, has_bias=False):
    """Create a trivial Linear with given dimensions."""
    tensors = {}
    if in_dim is not None:
        tensors['weight'] = torch.randn(in_dim, out_dim)
    else:
        tensors['weight'] = torch.randn(out_dim)
    if has_bias:
        tensors['bias'] = torch.randn(out_dim)
    return Linear(tensors=tensors)


# -- 1-in, 1-out --

def test_1in_1out_2d():
    """Single Linear in, single Linear out, 2D tensors."""
    from lmdeploy.turbomind.deploy.builder._base import transform_tensors

    @transform_tensors
    def double_last(tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat([tensor, tensor], dim=-1)

    lin = _make_linear(4, in_dim=3)
    result = double_last(lin)
    assert isinstance(result, Linear)
    assert result.tensors['weight'].shape == (3, 8)


def test_1in_1out_1d():
    """Single Linear in (1D bias), single Linear out."""
    from lmdeploy.turbomind.deploy.builder._base import transform_tensors

    @transform_tensors
    def double_last(tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat([tensor, tensor], dim=-1)

    lin = _make_linear(4, has_bias=True)
    bias = lin.tensors['bias']
    assert bias.dim() == 1  # bias is 1D

    result = double_last(lin)
    assert result.tensors['bias'].dim() == 1
    assert result.tensors['bias'].shape == (8,)


def test_1in_1out_mixed_dims():
    """Weight is 2D, bias is 1D -- both should be handled correctly."""
    from lmdeploy.turbomind.deploy.builder._base import transform_tensors

    @transform_tensors
    def double_last(tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat([tensor, tensor], dim=-1)

    lin = _make_linear(4, in_dim=3, has_bias=True)
    result = double_last(lin)
    assert result.tensors['weight'].shape == (3, 8)
    assert result.tensors['bias'].shape == (8,)
    assert result.tensors['bias'].dim() == 1


# -- 1-in, 2-out --

def test_1in_2out():
    """Single Linear in, tuple of two Linears out."""
    from lmdeploy.turbomind.deploy.builder._base import transform_tensors

    @transform_tensors
    def split_half(tensor: torch.Tensor
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        mid = tensor.size(-1) // 2
        return tensor[..., :mid], tensor[..., mid:]

    lin = _make_linear(8, in_dim=4)
    a, b = split_half(lin)
    assert isinstance(a, Linear)
    assert isinstance(b, Linear)
    assert a.tensors['weight'].shape == (4, 4)
    assert b.tensors['weight'].shape == (4, 4)


def test_1in_2out_1d():
    """Split on a 1D tensor produces 1D outputs."""
    from lmdeploy.turbomind.deploy.builder._base import transform_tensors

    @transform_tensors
    def split_half(tensor: torch.Tensor
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        mid = tensor.size(-1) // 2
        return tensor[..., :mid], tensor[..., mid:]

    lin = _make_linear(8)
    a, b = split_half(lin)
    assert a.tensors['weight'].dim() == 1
    assert b.tensors['weight'].dim() == 1


# -- Multi-in, 1-out --

def test_multi_in_1out():
    """Multiple Linears in, single Linear out."""
    from lmdeploy.turbomind.deploy.builder._base import transform_tensors

    @transform_tensors
    def concat_three(a: torch.Tensor, b: torch.Tensor,
                     c: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, b, c], dim=-1)

    lin_a = _make_linear(4, in_dim=3)
    lin_b = _make_linear(4, in_dim=3)
    lin_c = _make_linear(4, in_dim=3)
    result = concat_three(lin_a, lin_b, lin_c)
    assert result.tensors['weight'].shape == (3, 12)


# -- Multi-in, 1-out with optional --

def test_multi_in_optional_none():
    """Optional tensor arg passed as None."""
    from lmdeploy.turbomind.deploy.builder._base import transform_tensors

    @transform_tensors
    def concat_optional(a: torch.Tensor, b: torch.Tensor,
                        *, extra: torch.Tensor | None = None
                        ) -> torch.Tensor:
        parts = [a, b]
        if extra is not None:
            parts.append(extra)
        return torch.cat(parts, dim=-1)

    lin_a = _make_linear(4, in_dim=3)
    lin_b = _make_linear(4, in_dim=3)
    result = concat_optional(lin_a, lin_b)
    assert result.tensors['weight'].shape == (3, 8)


def test_multi_in_optional_provided():
    """Optional tensor arg provided."""
    from lmdeploy.turbomind.deploy.builder._base import transform_tensors

    @transform_tensors
    def concat_optional(a: torch.Tensor, b: torch.Tensor,
                        *, extra: torch.Tensor | None = None
                        ) -> torch.Tensor:
        parts = [a, b]
        if extra is not None:
            parts.append(extra)
        return torch.cat(parts, dim=-1)

    lin_a = _make_linear(4, in_dim=3)
    lin_b = _make_linear(4, in_dim=3)
    lin_c = _make_linear(2, in_dim=3)
    result = concat_optional(lin_a, lin_b, extra=lin_c)
    assert result.tensors['weight'].shape == (3, 10)


# -- Format propagation --

def test_format_propagation():
    """Output Linear inherits weight_format and data_format from first input."""
    from lmdeploy.turbomind.deploy.builder._base import transform_tensors

    @transform_tensors
    def identity(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    lin = _make_linear(4, in_dim=3)
    # Manually set formats
    object.__setattr__(lin, 'weight_format', 'mock_fmt')
    object.__setattr__(lin, 'data_format', 'mock_df')
    result = identity(lin)
    assert result.weight_format == 'mock_fmt'
    assert result.data_format == 'mock_df'


# -- Kwargs passthrough --

def test_kwargs_passthrough():
    """Non-tensor kwargs pass through unchanged."""
    from lmdeploy.turbomind.deploy.builder._base import transform_tensors

    @transform_tensors
    def scale(tensor: torch.Tensor, *, factor: float) -> torch.Tensor:
        return tensor * factor

    lin = _make_linear(4, in_dim=3)
    result = scale(lin, factor=2.0)
    assert torch.allclose(result.tensors['weight'], lin.tensors['weight'] * 2.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/lmdeploy-modeling && python -m pytest tests/test_lmdeploy/test_turbomind/test_transform_tensors.py -v`
Expected: FAIL — `ImportError: cannot import name 'transform_tensors'`

- [ ] **Step 3: Commit the test file**

```bash
git add tests/test_lmdeploy/test_turbomind/test_transform_tensors.py
git commit -m "test: add unit tests for @transform_tensors decorator"
```

---

### Task 2: Implement `@transform_tensors`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/_base.py:1-10` (add imports and decorator)

- [ ] **Step 1: Add the decorator to `_base.py`**

Add these imports at the top of `_base.py` (after the existing imports):

```python
import functools
import typing
```

Add the decorator after the `_ensure_compatible_formats` function (after line 148) and before the "Core tensor commit" section:

```python
# ---------------------------------------------------------------------------
# @transform_tensors decorator
# ---------------------------------------------------------------------------


def transform_tensors(fn):
    """Decorator that lifts a tensor-level transform to Linear-level.

    The decorated function operates on 2D ``torch.Tensor`` objects.  The
    wrapper handles the 1D/2D roundtrip (unsqueeze/squeeze), iterates over
    all tensor kinds in the ``Linear.tensors`` dict, and constructs new
    ``Linear`` objects from the results.

    Signature rules (determined by annotations):

    - Parameters typed ``torch.Tensor`` map to ``Linear`` positional args.
    - Parameters typed ``torch.Tensor | None`` are optional ``Linear`` args;
      ``None`` passes through without the 1D/2D dance.
    - Other parameters pass through unchanged.
    - Return ``torch.Tensor`` produces a single ``Linear``.
    - Return ``tuple[torch.Tensor, ...]`` produces a tuple of ``Linear`` objects.
    """
    hints = typing.get_type_hints(fn)
    ret_hint = hints.get('return', torch.Tensor)

    # Identify tensor-typed params and optional-tensor params
    sig = inspect.signature(fn)
    tensor_params = []      # param names that are torch.Tensor (required)
    optional_params = []    # param names that are torch.Tensor | None

    for pname, param in sig.parameters.items():
        hint = hints.get(pname)
        if hint is torch.Tensor:
            tensor_params.append(pname)
        elif _is_optional_tensor(hint):
            optional_params.append(pname)

    is_tuple_return = _is_tuple_of_tensors(ret_hint)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Bind args/kwargs to parameter names
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # Separate Linear args from passthrough kwargs
        linears = {}
        for pname in tensor_params:
            linears[pname] = bound.arguments[pname]
        for pname in optional_params:
            linears[pname] = bound.arguments[pname]

        # Passthrough kwargs: everything that's not a tensor param
        pass_kwargs = {k: v for k, v in bound.arguments.items()
                       if k not in tensor_params and k not in optional_params}

        # Use first non-None Linear for kind iteration and format
        first_linear = next(l for l in linears.values() if l is not None)

        # Collect output buckets
        if is_tuple_return:
            out_buckets = None  # will be list of dicts after first call
        else:
            out_tensors = {}

        for kind, _ in first_linear.tensors.items():
            was_1d = False
            fn_kwargs = dict(pass_kwargs)

            # Extract tensors for this kind, handle 1D/2D
            for pname in tensor_params:
                t = linears[pname].tensors[kind]
                if t.dim() == 1:
                    was_1d = True
                    t = t.unsqueeze(0)
                fn_kwargs[pname] = t

            for pname in optional_params:
                lin = linears[pname]
                if lin is None:
                    fn_kwargs[pname] = None
                else:
                    t = lin.tensors[kind]
                    if t.dim() == 1:
                        was_1d = True
                        t = t.unsqueeze(0)
                    fn_kwargs[pname] = t

            result = fn(**fn_kwargs)

            if is_tuple_return:
                items = result
                if out_buckets is None:
                    out_buckets = [{} for _ in items]
                for i, item in enumerate(items):
                    if was_1d:
                        item = item.squeeze(0)
                    out_buckets[i][kind] = item
            else:
                if was_1d:
                    result = result.squeeze(0)
                out_tensors[kind] = result

        if is_tuple_return:
            return tuple(
                Linear(tensors=b, weight_format=first_linear.weight_format,
                       data_format=first_linear.data_format)
                for b in out_buckets
            )
        return Linear(tensors=out_tensors,
                      weight_format=first_linear.weight_format,
                      data_format=first_linear.data_format)

    return wrapper


def _is_optional_tensor(hint) -> bool:
    """Check if hint is ``torch.Tensor | None``."""
    origin = getattr(hint, '__origin__', None)
    if origin is typing.Union:
        args = hint.__args__
        return torch.Tensor in args and type(None) in args
    return False


def _is_tuple_of_tensors(hint) -> bool:
    """Check if hint is ``tuple[torch.Tensor, ...]``."""
    origin = getattr(hint, '__origin__', None)
    if origin is tuple:
        args = getattr(hint, '__args__', ())
        return all(a is torch.Tensor for a in args)
    return False
```

Also add `import inspect` to the imports at the top of `_base.py`.

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /data/lmdeploy-modeling && python -m pytest tests/test_lmdeploy/test_turbomind/test_transform_tensors.py -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/_base.py
git commit -m "feat: add @transform_tensors decorator"
```

---

### Task 3: Refactor `_repeat_kv_heads` to use `@transform_tensors`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/attention.py:65-87`

- [ ] **Step 1: Replace `_repeat_kv_heads`**

The new `_repeat_kv_heads` keeps the early-return check and precondition assertion, then delegates the tensor work to a decorated helper. Replace lines 65-87 with:

```python
@transform_tensors
def _repeat_kv_heads_2d(tensor: torch.Tensor, *, n_repeat: int,
                        heads: int) -> torch.Tensor:
    per_head = tensor.size(-1) // heads
    t = tensor.view(-1, heads, per_head)
    target_heads = heads * n_repeat
    return t.repeat(1, n_repeat, 1).reshape(-1, target_heads * per_head)


def _repeat_kv_heads(linear: Linear, tp: int, head_dim: int) -> Linear:
    heads = _infer_heads(linear, head_dim)
    if heads % tp == 0:
        return linear
    target_heads = ((heads + tp - 1) // tp) * tp
    assert target_heads % heads == 0, (
        f"target_heads={target_heads} must be divisible by heads={heads}")
    return _repeat_kv_heads_2d(linear, n_repeat=target_heads // heads, heads=heads)
```

Also add `transform_tensors` to the imports from `._base`:

```python
from ._base import Builder, SplitSide, _dequant_linear, transform_tensors
```

- [ ] **Step 2: Run tests**

Run: `cd /data/lmdeploy-modeling && python -m pytest tests/test_lmdeploy/test_turbomind/test_transform_tensors.py -v`
Expected: All tests still PASS.

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/attention.py
git commit -m "refactor: use @transform_tensors in _repeat_kv_heads"
```

---

### Task 4: Refactor `split_output_gate` to use `@transform_tensors`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/attention.py:104-130`

- [ ] **Step 1: Replace `split_output_gate`**

Replace lines 104-130 with:

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

- [ ] **Step 2: Run tests**

Run: `cd /data/lmdeploy-modeling && python -m pytest tests/test_lmdeploy/test_turbomind/test_transform_tensors.py -v`
Expected: All tests still PASS.

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/attention.py
git commit -m "refactor: use @transform_tensors in split_output_gate"
```

---

### Task 5: Refactor `fuse_qkv` to use `@transform_tensors`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/attention.py:133-161`

- [ ] **Step 1: Replace `fuse_qkv`**

Replace lines 133-161 with:

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

Note: the public signature changes from `(q: Linear, k: Linear, v: Linear, *, tp: int, gate: Linear | None)` to the same thing — `Linear` in, `Linear` out — because the decorator handles the mapping transparently.

- [ ] **Step 2: Run tests**

Run: `cd /data/lmdeploy-modeling && python -m pytest tests/test_lmdeploy/test_turbomind/test_transform_tensors.py -v`
Expected: All tests still PASS.

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/builder/attention.py
git commit -m "refactor: use @transform_tensors in fuse_qkv"
```

---

### Task 6: Integration test — verify a model still runs

**Files:** None (uses existing `scripts/test_turbomind_model.py`)

- [ ] **Step 1: Check GPU availability**

Use the `get_gpu_usage` MCP tool to find an empty GPU.

- [ ] **Step 2: Pick a model and run integration test**

Run (adjust model, cache_dir, GPU based on availability):
```bash
cd /data/lmdeploy-modeling && python scripts/test_turbomind_model.py <model> <cache_dir> 1 0
```

Use a model that exercises the attention path (e.g. Qwen2.5-7B-Instruct or similar). Verify the response contains meaningful text and is at least 128 tokens.

- [ ] **Step 3: Commit all changes if test passes**

No new files to commit — this step validates the refactoring is correct.
