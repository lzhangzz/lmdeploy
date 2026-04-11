# Import & Dtype Map Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace 10 scattered `import _turbomind` calls (5 with misleading try/except guards) and 2 duplicated dtype maps with a single module-level import and consolidated constants per file.

**Architecture:** Add `import _turbomind as _tm` at module level in each affected file. Consolidate the two `torch.dtype → DataType` dicts in `load_context.py` into one module-level `_TORCH_TO_CPP` map. Remove all per-function imports and try/except guards. The `_noop` class stays — it's used as a context manager fallback for child LoadContexts, not as an import guard.

**Tech Stack:** Python, pybind11

**Spec:** `docs/superpowers/specs/2026-04-11-import-dtype-consolidation-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `lmdeploy/turbomind/deploy/load_context.py` | **Modify** | Module-level import, consolidated dtype maps, remove 7 per-function imports |
| `lmdeploy/turbomind/deploy/linear.py` | **Modify** | Replace guarded import with module-level import |
| `lmdeploy/turbomind/deploy/kind_map.py` | **Modify** | Add module-level import, remove 2 guards in `to_data_format()` |

---

### Task 1: Consolidate `load_context.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/load_context.py`

- [ ] **Step 1: Add module-level import and dtype constants**

Replace lines 1–14 (the top-of-file block) with:

```python
# Copyright (c) OpenMMLab. All rights reserved.
"""Composable loading primitives for building the C++ module tree from Python."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import _turbomind as _tm

from .linear import Linear
from .spec import SplitSide

if TYPE_CHECKING:
    from .config import ModelConfig

# Canonical dtype mappings
_STR_TO_DTYPE: dict[str, _tm.DataType] = {
    'float32':  _tm.DataType.TYPE_FP32,
    'float16':  _tm.DataType.TYPE_FP16,
    'bfloat16': _tm.DataType.TYPE_BF16,
}

_TORCH_TO_CPP: dict[torch.dtype, _tm.DataType] = {
    torch.float32:  _tm.DataType.TYPE_FP32,
    torch.float16:  _tm.DataType.TYPE_FP16,
    torch.bfloat16: _tm.DataType.TYPE_BF16,
    torch.int32:    _tm.DataType.TYPE_INT32,
    torch.int64:    _tm.DataType.TYPE_INT64,
    torch.int8:     _tm.DataType.TYPE_INT8,
    torch.uint8:    _tm.DataType.TYPE_UINT8,
}
```

- [ ] **Step 2: Simplify `_cpp_dtype`**

Replace lines 16–23 with:

```python
def _cpp_dtype(dtype_str: str):
    """Convert a model-config data_type string to C++ DataType enum."""
    return _STR_TO_DTYPE[dtype_str]
```

- [ ] **Step 3: Simplify `_torch_dtype_to_cpp`**

Replace lines 38–53 with:

```python
def _torch_dtype_to_cpp(dtype: torch.dtype):
    """Convert a torch dtype to the C++ ``DataType`` enum, or ``None``."""
    return _TORCH_TO_CPP.get(dtype)
```

- [ ] **Step 4: Simplify `_cast_shard_for_tm`**

Replace lines 56–69 with:

```python
def _cast_shard_for_tm(shard: torch.Tensor, tm_tensor) -> torch.Tensor:
    """Cast *shard* dtype to match *tm_tensor*'s C++ dtype when needed."""
    if tm_tensor.type == _tm.DataType.TYPE_FP32 and shard.dtype in (torch.float16, torch.bfloat16):
        return shard.float()
    if tm_tensor.type == _tm.DataType.TYPE_FP16 and shard.dtype != torch.float16:
        return shard.half()
    if tm_tensor.type == _tm.DataType.TYPE_BF16 and shard.dtype != torch.bfloat16:
        return shard.to(torch.bfloat16)
    return shard
```

- [ ] **Step 5: Simplify `_infer_cpp_linear_dtype`**

Replace lines 72–92 with:

```python
def _infer_cpp_linear_dtype(linear: Linear):
    """Determine C++ DataType and group_size from ``Linear.weight_format``."""
    fmt = linear.weight_format
    if fmt is not None and fmt.cpp_dtype_name is not None:
        cpp_dtype = getattr(_tm.DataType, fmt.cpp_dtype_name, None)
        if cpp_dtype is not None:
            return cpp_dtype, fmt.block_in or 0

    # Dense (or missing format): dtype from weight tensor
    weight = linear.tensors.get("weight")
    if weight is not None:
        if weight.dtype == torch.bfloat16:
            return _tm.DataType.TYPE_BF16, 0
        if weight.dtype == torch.float16:
            return _tm.DataType.TYPE_FP16, 0
    return None, 0
```

- [ ] **Step 6: Simplify `_infer_compute_dtype`**

Replace lines 95–131 with:

```python
def _infer_compute_dtype(linear: Linear):
    """Get the model's compute dtype from a Linear's tensors.

    For dense formats the weight itself carries the compute dtype.
    For quantized formats we infer from scales or bias.
    """
    w = linear.tensors.get('weight')
    if w is not None:
        d = _TORCH_TO_CPP.get(w.dtype)
        if d is not None:
            return d
        # FP8 weights: compute dtype is BF16 (or FP16 depending on model),
        # not FP32.  Fall through to scales/bias only for non-FP8 dtypes.
        _fp8_dtypes = {torch.uint8}
        for _attr in ('float8_e4m3fn', 'float8_e5m2fn'):
            _dt = getattr(torch, _attr, None)
            if _dt is not None:
                _fp8_dtypes.add(_dt)
        if w.dtype in _fp8_dtypes:
            # FP8 stored as uint8 after normalization; prefer BF16.
            return _tm.DataType.TYPE_BF16
    for key in ('scales', 'bias'):
        t = linear.tensors.get(key)
        if t is not None:
            d = _TORCH_TO_CPP.get(t.dtype)
            if d is not None:
                return d
    return None
```

- [ ] **Step 7: Remove per-function import in `commit_linear`**

Delete line 253 (`import _turbomind as _tm`). The rest of the function body stays unchanged — all `_tm.*` references resolve to the module-level import.

- [ ] **Step 8: Remove per-function import in `LoadContext.load_linear`**

Delete line 513 (`import _turbomind as _tm`) inside the `with self._context or _noop():` block. Keep the `with` block itself and all the code inside it.

- [ ] **Step 9: Verify import works**

Run: `cd /data/lmdeploy-modeling && PYTHONPATH=lmdeploy:build/lib python -c "from lmdeploy.turbomind.deploy.load_context import LoadContext; print('OK')"`
Expected: `OK`

- [ ] **Step 10: Commit**

```bash
git add lmdeploy/turbomind/deploy/load_context.py
git commit -m "refactor(load_context): module-level import and consolidated dtype maps"
```

---

### Task 2: Fix `linear.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/linear.py`

- [ ] **Step 1: Replace guarded import with module-level import**

Replace lines 30–33:

```python
try:
    from _turbomind import DataFormat
except ImportError:
    DataFormat = None
```

With:

```python
import _turbomind as _tm
```

Also update the `Linear` dataclass field (line 141) from:

```python
    data_format: DataFormat | None = field(default=None, compare=False, repr=False)
```

To:

```python
    data_format: _tm.DataFormat | None = field(default=None, compare=False, repr=False)
```

- [ ] **Step 2: Verify import works**

Run: `cd /data/lmdeploy-modeling && PYTHONPATH=lmdeploy:build/lib python -c "from lmdeploy.turbomind.deploy.linear import Linear; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add lmdeploy/turbomind/deploy/linear.py
git commit -m "refactor(linear): remove ImportError guard for _turbomind"
```

---

### Task 3: Fix `kind_map.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/kind_map.py`

- [ ] **Step 1: Add module-level import**

After line 21 (`from torch import Tensor`), add:

```python
import _turbomind as _tm
```

- [ ] **Step 2: Remove guards in `to_data_format`**

Replace lines 87–111 with:

```python
    def to_data_format(self, cpp_dtype: int, group_size: int = 0):
        """Construct a C++ DataFormat from this WeightFormat.

        Returns None when group_size is needed but not yet known (block_in==0
        and group_size==0), or when the format is dense (block_in is None).
        """
        if self.block_in is None:
            return None
        gs = group_size if self.block_in == 0 else self.block_in
        # Formats with block_in==0 need a real group_size; defer to commit time.
        if gs == 0:
            return None
        if self.cpp_dtype_name is not None:
            dt = getattr(_tm.DataType, self.cpp_dtype_name, None)
            if dt is not None:
                return _tm.MakeLinearWeightFormat(dt, dt, gs)
        return None
```

- [ ] **Step 3: Verify import works**

Run: `cd /data/lmdeploy-modeling && PYTHONPATH=lmdeploy:build/lib python -c "from lmdeploy.turbomind.deploy.kind_map import WeightFormat; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/kind_map.py
git commit -m "refactor(kind_map): remove ImportError guards for _turbomind"
```

---

### Task 4: Verify with model test

**Files:** None (verification only)

- [ ] **Step 1: Run a model test to verify no regression**

Use the turbomind-tester agent to verify at least one model loads and produces correct output with TP=1. The loading pipeline exercises all the changed code paths: dtype mapping, weight format inference, tensor commit.

---

## Self-Review

**1. Spec coverage:**
- "Module-level import in each file" → Tasks 1 (Step 1), 2 (Step 1), 3 (Step 1) ✓
- "Consolidate dtype maps" → Task 1 (Steps 1, 3, 6) ✓
- "Remove all per-function imports and guards" → Task 1 (Steps 2–8), Task 2, Task 3 ✓
- "Remove _noop" → Intentionally kept (it's a context manager fallback, not an import guard). Spec correction noted.

**2. Placeholder scan:** No TBDs, TODOs, or "implement later" patterns. All code shown.

**3. Type consistency:**
- `_TORCH_TO_CPP` dict defined in Task 1 Step 1, used in Task 1 Steps 3 and 6 — same name ✓
- `_STR_TO_DTYPE` dict defined in Task 1 Step 1, used in Task 1 Step 2 — same name ✓
- `_tm.DataFormat` used in Task 2 Step 1 matches the module-level `import _turbomind as _tm` ✓
- `_tm.MakeLinearWeightFormat` used in Task 3 Step 2 matches the module-level `import _turbomind as _tm` ✓
