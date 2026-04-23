# Weight Format Resolver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Centralize quant-format + compute-dtype threading behind a single `WeightFormatResolver` owned by the spec; replace the callable-field `WeightFormat` dataclass with an ABC + one concrete subclass per format; make format-resolution failures loud and distinct from legitimate module absence.

**Architecture:** New file `lmdeploy/turbomind/deploy/weight_format.py` holds a `WeightFormat` ABC with six concrete subclasses (`TrivialFormat`, `AWQFormat`, `GPTQFormat`, `CompressedTensorFormat`, `FP8Format`, `MXFP4Format`) and a `WeightFormatResolver` that carries the model compute dtype plus an ordered list of candidate formats. The converter builds the resolver (quant format first, trivial last). Specs accept `resolver=` instead of `weight_format=`; their `_linear()` delegates to `resolver.resolve()` and gains an `optional=` kwarg for legitimate-absence sites. Two utils (`read_packed_moe_expert`, `reorder_rotary_emb`'s Linear path) swap `data_type=, weight_format=` for `resolver=`. Builder's `_dequant_linear` drops the `fmt.dequant is None` guard and `_commit_linear` collapses its packer hoist to a single `fmt.pack(...)` call. The legacy `kind_map.py` is deleted.

**Tech Stack:** Python 3.10+ (match syntax), PyTorch, TurboMind deploy pipeline, pybind11 C++ bindings via `_turbomind`.

---

### Task 1: Create `weight_format.py` with the new class hierarchy and resolver

**Files:**
- Create: `lmdeploy/turbomind/deploy/weight_format.py`

This task adds a standalone, self-contained module. Nothing else in the tree imports from it yet; the subsequent tasks migrate consumers. After this task the tree still compiles and tests still pass exactly as before.

- [ ] **Step 1: Create the file with module docstring, imports, u4 helpers, and `pack_u4_row`**

Write the following to `lmdeploy/turbomind/deploy/weight_format.py`:

```python
# Copyright (c) OpenMMLab. All rights reserved.
"""Weight format resolution for TurboMind checkpoint loading.

Exports:

- ``WeightFormat`` (ABC) and six concrete subclasses: ``TrivialFormat``,
  ``AWQFormat``, ``GPTQFormat``, ``CompressedTensorFormat``, ``FP8Format``,
  ``MXFP4Format``. Each subclass declares its ``name``, ``suffix_map``,
  ``weight_dtype`` (``_tm.DataType`` or ``None``), ``has_zero_point`` flag,
  and overrides ``accepts`` + ``normalize``. Optional overrides: ``pack``
  (identity default), ``synthesize_zeros`` (raises by default), ``dequant``
  (raises by default; ``TrivialFormat.dequant`` is identity).

- ``WeightFormatResolver``: holds the model compute dtype plus an ordered
  list of candidate formats. ``resolve(params, prefix, *, index=None,
  optional=False)`` returns a ``Linear`` bundle in TM layout or raises
  (``KeyError`` on missing tensors without ``optional``, ``ValueError`` when
  tensors exist but no candidate matches).

- ``pack_u4_row``: uint8 → int32 row packer used by quantized ``pack``
  overrides and by downstream callers that pack packed-expert weights
  after slicing.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import torch
from torch import Tensor

import _turbomind as _tm

from .linear import Linear


# ---------------------------------------------------------------------------
# Low-level u4 packing / unpacking helpers (reused across normalize / pack)
# ---------------------------------------------------------------------------


def _get_u4_slices(x: Tensor, dtype: torch.dtype) -> list[Tensor]:
    MAP = {torch.int32: 8, torch.uint8: 2}
    xs = []
    for _ in range(MAP[x.dtype]):
        xs.append((x & 15).to(dtype))
        x = x >> 4
    return xs


def _unpack_awq_gemm(x: Tensor) -> Tensor:
    xs = _get_u4_slices(x, torch.uint8)
    order = [0, 4, 1, 5, 2, 6, 3, 7]
    ys = [xs[i] for i in order]
    return torch.stack(ys, dim=-1).view(*x.shape[:-1], -1)


def pack_u4_row(x: torch.Tensor) -> torch.Tensor:
    """Pack uint8 4-bit values into int32 rows along the last dim.

    Used by every int4 format's ``pack`` override and by callers that
    re-pack tensors after slicing (e.g. packed-MoE expert split).
    """
    assert x.dtype == torch.uint8, f'x.dtype: {x.dtype}'
    xs = x.view(*x.shape[:-1], -1, 8).split(1, dim=-1)
    a = torch.zeros(xs[0].shape, dtype=torch.int32, device=x.device)
    for t in reversed(xs):
        a = (a << 4) | t
    return a.squeeze(dim=-1)


def _zeros_int4_symmetric(scales: Tensor) -> Tensor:
    """Synthesize symmetric int4 zero-points (value = 8) matching *scales* shape."""
    return torch.full(scales.shape, 8, dtype=torch.uint8, device=scales.device)
```

- [ ] **Step 2: Append the `WeightFormat` ABC**

Append the following to `lmdeploy/turbomind/deploy/weight_format.py`:

```python
# ---------------------------------------------------------------------------
# WeightFormat ABC
# ---------------------------------------------------------------------------


class WeightFormat(ABC):
    """Abstract per-format policy object.

    Class attributes (override in subclasses):

    - ``name``: canonical format name used for string comparisons.
    - ``suffix_map``: ``{checkpoint_suffix: tm_kind}``. Drives which
      checkpoint tensors each format ingests at a given prefix.
    - ``weight_dtype``: ``_tm.DataType`` for the weight storage dtype;
      ``None`` for trivial (weight dtype equals compute dtype).
    - ``has_zero_point``: ``True`` when the format uses a zero-point
      tensor; gates the resolver's ``synthesize_zeros`` call.

    Instance attributes (set by subclass ``__init__``):

    - ``block_in``, ``block_out``: quantization block sizes. ``None`` for
      dimensions without blocking.

    Methods:

    - ``accepts`` (abstract): classify a checkpoint suffix dict.
    - ``normalize`` (abstract): raw-checkpoint tensor → TM layout.
    - ``pack``: optional commit-time packer. Identity default.
    - ``synthesize_zeros``: fabricate a zeros tensor when the checkpoint
      omits it. Raises ``NotImplementedError`` by default.
    - ``dequant``: produce a trivial ``{weight, bias?}`` dict from TM
      tensors for mixed-format fusion. Raises ``NotImplementedError`` by
      default. ``TrivialFormat.dequant`` is identity.
    - ``make_data_format``: build the ``_tm.DataFormat`` descriptor.

    Equality / hashing: two WeightFormats are equal iff they share class
    and block sizes. This matters for the set-based uniformity checks in
    ``Linear.concat_out_dim`` / ``Linear.concat_in_dim``.
    """

    name:           ClassVar[str]
    suffix_map:     ClassVar[dict[str, str]]
    weight_dtype:   ClassVar["_tm.DataType | None"]
    has_zero_point: ClassVar[bool]

    block_in:  int | None
    block_out: int | None

    def __init__(self, *, block_in: int | None = None,
                 block_out: int | None = None):
        self.block_in  = block_in
        self.block_out = block_out

    @abstractmethod
    def accepts(self, available: dict[str, Tensor]) -> bool: ...

    @abstractmethod
    def normalize(self, tensor: Tensor, kind: str) -> Tensor: ...

    def pack(self, tensor: Tensor, kind: str) -> Tensor:
        return tensor

    def synthesize_zeros(self, scales: Tensor) -> Tensor:
        raise NotImplementedError(
            f"{type(self).__name__}.synthesize_zeros not implemented")

    def dequant(self, tensors: dict[str, Tensor],
                data_type) -> dict[str, Tensor]:
        raise NotImplementedError(
            f"{type(self).__name__}.dequant not implemented")

    def make_data_format(self, data_type) -> "_tm.DataFormat":
        if self.weight_dtype is None:
            return _tm.ResolveLinearWeightFormat(data_type, data_type, 1, 1)
        return _tm.ResolveLinearWeightFormat(
            data_type, self.weight_dtype,
            self.block_in  or 1, self.block_out or 1)

    def __eq__(self, other) -> bool:
        if not isinstance(other, WeightFormat):
            return NotImplemented
        return (type(self) is type(other)
                and self.block_in  == other.block_in
                and self.block_out == other.block_out)

    def __hash__(self) -> int:
        return hash((type(self), self.block_in, self.block_out))
```

- [ ] **Step 3: Append `TrivialFormat`**

Append the following to `lmdeploy/turbomind/deploy/weight_format.py`:

```python
# ---------------------------------------------------------------------------
# Concrete subclasses
# ---------------------------------------------------------------------------


class TrivialFormat(WeightFormat):
    name           = "trivial"
    suffix_map     = {".weight": "weight", ".bias": "bias"}
    weight_dtype   = None
    has_zero_point = False

    def accepts(self, available: dict[str, Tensor]) -> bool:
        if not (available.keys() <= {".weight", ".bias"}):
            return False
        w = available.get(".weight")
        return w is None or w.dtype.is_floating_point

    def normalize(self, x: Tensor, kind: str) -> Tensor:
        x = x.cuda()
        if x.dim() >= 2:
            x = x.t()
        return x

    def dequant(self, tensors, data_type):
        # Already trivial — nothing to undo. Identity override for mixed
        # fusion groups.
        return tensors
```

- [ ] **Step 4: Append `AWQFormat`**

Append the following to `lmdeploy/turbomind/deploy/weight_format.py`:

```python
class AWQFormat(WeightFormat):
    name           = "awq"
    suffix_map     = {".qweight": "weight", ".scales": "scales",
                      ".qzeros": "zeros",   ".bias": "bias"}
    weight_dtype   = _tm.DataType.TYPE_UINT4
    has_zero_point = True

    def __init__(self, *, block_in: int):
        super().__init__(block_in=block_in, block_out=None)

    def accepts(self, available: dict[str, Tensor]) -> bool:
        qw = available.get(".qweight")
        if qw is None or qw.dtype != torch.int32:
            return False
        scales = available.get(".scales")
        if scales is not None and qw.ndim >= 2 and scales.ndim >= 2:
            return qw.shape[-1] * 8 == scales.shape[-1]
        return True

    def normalize(self, x: Tensor, kind: str) -> Tensor:
        # AWQ checkpoints store weights in TM-native layout:
        #   qweight: [K, N//8] int32 → unpack → [K, N] (TM, no .t())
        #   scales:  [K//g, N] float16 → already TM
        #   zeros:   [K//g, N//8] int32 → unpack → [K//g, N]
        x = x.cuda()
        if x.dtype == torch.int32:
            x = _unpack_awq_gemm(x)
        if kind == "zeros":
            x = x.to(torch.float16)
        return x

    def pack(self, tensor: Tensor, kind: str) -> Tensor:
        if kind == "weight" and tensor.dtype == torch.uint8:
            return pack_u4_row(tensor)
        return tensor

    def dequant(self, tensors, data_type):
        from lmdeploy.pytorch.backends.default.awq_modules import dequantize_gemm

        qweight = tensors["weight"]
        scales  = tensors["scales"]
        qzeros  = tensors["zeros"]
        group_size = qweight.shape[0] // scales.shape[0]
        w = dequantize_gemm(qweight, qzeros, scales, 4, group_size)
        result: dict[str, Tensor] = {"weight": w}
        if "bias" in tensors:
            result["bias"] = tensors["bias"]
        return result
```

- [ ] **Step 5: Append `GPTQFormat`**

Append the following to `lmdeploy/turbomind/deploy/weight_format.py`:

```python
class GPTQFormat(WeightFormat):
    name           = "gptq"
    suffix_map     = {".qweight": "weight", ".scales": "scales",
                      ".qzeros": "zeros",   ".bias": "bias"}
    weight_dtype   = _tm.DataType.TYPE_UINT4
    has_zero_point = True

    def __init__(self, *, block_in: int):
        super().__init__(block_in=block_in, block_out=None)

    def accepts(self, available: dict[str, Tensor]) -> bool:
        qw = available.get(".qweight")
        if qw is None or qw.dtype != torch.int32:
            return False
        scales = available.get(".scales")
        if scales is not None and qw.ndim >= 2 and scales.ndim >= 2:
            return qw.shape[-1] == scales.shape[-1]
        return True

    def normalize(self, x: Tensor, kind: str) -> Tensor:
        # GPTQ checkpoint stores weights in TM-native layout:
        #   qweight: [K//8, N] int32 → unpack → [K, N]
        #   scales:  [K//g, N] float16 → already TM
        #   zeros:   [K//g, N//8] int32 → unpack → [K//g, N] (+1 offset)
        x = x.cuda()
        if x.dtype == torch.int32:
            xs = _get_u4_slices(x, torch.uint8)
            if kind == "weight":
                x = torch.stack(xs, dim=1).view(-1, x.size(-1))
            else:
                x = torch.stack(xs, dim=-1).view(x.size(0), -1) + 1
        if kind == "zeros":
            x = x.to(torch.float16)
        return x

    def pack(self, tensor: Tensor, kind: str) -> Tensor:
        if kind == "weight" and tensor.dtype == torch.uint8:
            return pack_u4_row(tensor)
        return tensor

    def synthesize_zeros(self, scales: Tensor) -> Tensor:
        return _zeros_int4_symmetric(scales)
```

- [ ] **Step 6: Append `CompressedTensorFormat`**

Append the following to `lmdeploy/turbomind/deploy/weight_format.py`:

```python
class CompressedTensorFormat(WeightFormat):
    name           = "compressed-tensors"
    suffix_map     = {".weight_packed":     "weight",
                      ".weight_scale":      "scales",
                      ".weight_zero_point": "zeros",
                      ".bias":              "bias"}
    weight_dtype   = _tm.DataType.TYPE_UINT4
    has_zero_point = True

    def __init__(self, *, block_in: int):
        super().__init__(block_in=block_in, block_out=None)

    def accepts(self, available: dict[str, Tensor]) -> bool:
        wp = available.get(".weight_packed")
        return wp is not None and wp.dtype == torch.int32

    def normalize(self, x: Tensor, kind: str) -> Tensor:
        x = x.cuda()
        if x.dtype == torch.int32:
            xs = _get_u4_slices(x, torch.uint8)
            if kind == "weight":
                x = torch.stack(xs, dim=-1).view(*x.shape[:-1], -1)
            elif kind == "zeros":
                x = torch.stack(xs, dim=1).view(-1, x.size(-1))
        if kind == "zeros":
            x = x.to(torch.float16)
        if x.dim() >= 2:
            x = x.t()
        return x

    def pack(self, tensor: Tensor, kind: str) -> Tensor:
        if kind == "weight" and tensor.dtype == torch.uint8:
            return pack_u4_row(tensor)
        return tensor

    def synthesize_zeros(self, scales: Tensor) -> Tensor:
        return _zeros_int4_symmetric(scales)
```

- [ ] **Step 7: Append `FP8Format`**

Append the following to `lmdeploy/turbomind/deploy/weight_format.py`:

```python
class FP8Format(WeightFormat):
    name           = "fp8"
    suffix_map     = {".weight":           "weight",
                      ".weight_scale_inv": "scales",
                      ".bias":             "bias"}
    weight_dtype   = _tm.DataType.TYPE_FP8_E4M3
    has_zero_point = False

    def __init__(self):
        super().__init__(block_in=128, block_out=128)

    def accepts(self, available: dict[str, Tensor]) -> bool:
        if ".weight_scale_inv" not in available:
            return False
        w = available.get(".weight")
        return w is None or w.dtype in (torch.float8_e4m3fn, torch.uint8)

    def normalize(self, x: Tensor, kind: str) -> Tensor:
        x = x.cuda()
        if x.dtype == torch.float8_e4m3fn:
            x = x.view(dtype=torch.uint8)
        if x.dim() >= 2:
            x = x.t()
        return x

    def dequant(self, tensors, data_type):
        from .builder._base import _CPP_TO_TORCH

        weight = tensors["weight"]
        scales = tensors["scales"]
        block_size = 128
        fp8_weight = weight.view(torch.float8_e4m3fn).float()
        scale = scales.float()
        scale = scale.repeat_interleave(block_size, dim=0)
        scale = scale.repeat_interleave(block_size, dim=1)
        scale = scale[: fp8_weight.shape[0], : fp8_weight.shape[1]]
        target_dtype = _CPP_TO_TORCH[data_type]
        result: dict[str, Tensor] = {"weight": (fp8_weight * scale).to(target_dtype)}
        if "bias" in tensors:
            result["bias"] = tensors["bias"]
        return result
```

- [ ] **Step 8: Append `MXFP4Format`**

Append the following to `lmdeploy/turbomind/deploy/weight_format.py`:

```python
class MXFP4Format(WeightFormat):
    name           = "mxfp4"
    suffix_map     = {".blocks": "weight", ".scales": "scales", ".bias": "bias"}
    weight_dtype   = _tm.DataType.TYPE_FP4_E2M1
    has_zero_point = False

    def __init__(self):
        super().__init__(block_in=32, block_out=None)

    def accepts(self, available: dict[str, Tensor]) -> bool:
        if ".scales" not in available:
            return False
        w = available.get(".blocks")
        return w is None or w.dtype == torch.uint8

    def normalize(self, x: Tensor, kind: str) -> Tensor:
        x = x.cuda()
        if kind == "weight":
            xs = _get_u4_slices(torch.flatten(x, start_dim=-2), torch.uint8)
            x = torch.flatten(torch.stack(xs, dim=-1), start_dim=-2)
        if x.dim() >= 2:
            x = x.t()
        return x

    def pack(self, tensor: Tensor, kind: str) -> Tensor:
        if kind == "weight" and tensor.dtype == torch.uint8:
            return pack_u4_row(tensor)
        return tensor
```

- [ ] **Step 9: Append `WeightFormatResolver`**

Append the following to `lmdeploy/turbomind/deploy/weight_format.py`:

```python
# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class WeightFormatResolver:
    """Resolve a checkpoint prefix to a ``Linear`` bundle in TM layout.

    Holds the model compute dtype and an ordered list of candidate
    formats. ``resolve(params, prefix)`` probes the checkpoint at the
    given prefix, dispatches to the first candidate whose ``accepts``
    returns True, and constructs a ``Linear`` with the format's
    ``make_data_format`` descriptor.

    The suffix probe is scoped to the union of candidate ``suffix_map``
    keys only — not a global "every format ever" list — so adding a new
    format elsewhere does not widen the probe.

    Priority is encoded by list order. The converter puts quantized
    candidates first and ``TrivialFormat()`` last: a prefix that only
    matches trivial (router, norm-like linears in a quantized model)
    deterministically falls through.

    Failure modes are loud and distinct:

    - ``optional=False`` (default) + no tensors at prefix → ``KeyError``
      with candidate suffix list.
    - Tensors present but no candidate accepts → ``ValueError`` with
      available keys and candidate names.
    - Only "no tensors AND optional=True" returns ``None``.
    """

    def __init__(self, *, data_type: "_tm.DataType",
                 formats: list[WeightFormat]):
        self._data_type = data_type
        self._formats   = formats
        self._suffixes  = frozenset(
            s for f in formats for s in f.suffix_map)

    @property
    def data_type(self) -> "_tm.DataType":
        return self._data_type

    def resolve(self, params: dict[str, Tensor], prefix: str, *,
                index: int | None = None,
                optional: bool = False) -> Linear | None:
        available = {s: params[prefix + s]
                     for s in self._suffixes if (prefix + s) in params}
        if index is not None:
            available = {s: t[index] for s, t in available.items()}

        if not available:
            if optional:
                return None
            raise KeyError(
                f"no checkpoint tensors found at prefix {prefix!r} "
                f"(candidate suffixes: {sorted(self._suffixes)})")

        for fmt in self._formats:
            if fmt.accepts(available):
                return self._build_linear(fmt, available)

        raise ValueError(
            f"no weight format accepts tensors at {prefix!r}: "
            f"got {sorted(available)}, "
            f"tried {[f.name for f in self._formats]}")

    def _build_linear(self, fmt: WeightFormat,
                      available: dict[str, Tensor]) -> Linear:
        tensors = {
            kind: fmt.normalize(available[s], kind)
            for s, kind in fmt.suffix_map.items()
            if s in available
        }
        if fmt.has_zero_point and "zeros" not in tensors:
            tensors["zeros"] = fmt.synthesize_zeros(tensors["scales"])
        return Linear(tensors=tensors,
                      weight_format=fmt,
                      data_format=fmt.make_data_format(self._data_type))
```

- [ ] **Step 10: Sanity-import the new module**

Run:

```bash
python -c "from lmdeploy.turbomind.deploy.weight_format import (
    WeightFormat, TrivialFormat, AWQFormat, GPTQFormat,
    CompressedTensorFormat, FP8Format, MXFP4Format,
    WeightFormatResolver, pack_u4_row); print('OK')"
```

Expected output: `OK`.

If this fails with `ImportError: cannot import name 'TYPE_FP4_E2M1'` (or `TYPE_FP8_E4M3`, `TYPE_UINT4`), the C++ bindings in this build don't expose those dtype enum values. Inspect `_turbomind.DataType` attributes and confirm the expected names before proceeding — the current deploy pipeline depends on them, so a missing name is a pre-existing build issue, not a regression.

- [ ] **Step 11: Commit**

```bash
git add lmdeploy/turbomind/deploy/weight_format.py
git commit -m "$(cat <<'EOF'
deploy: add weight_format module with resolver and format hierarchy

Introduces WeightFormat ABC with six concrete subclasses (Trivial, AWQ,
GPTQ, CompressedTensor, FP8, MXFP4) and a WeightFormatResolver that
carries the model compute dtype plus an ordered candidate list.
Standalone addition; no existing consumers updated yet.
EOF
)"
```

---

### Task 2: Unit tests for `WeightFormatResolver`

**Files:**
- Create: `tests/test_lmdeploy/test_turbomind/test_weight_format_resolver.py`

Tests for the resolver's pure-Python dispatch logic, using a lightweight fake `WeightFormat` subclass that bypasses `_turbomind` (matching the stub pattern already established in `test_transform_tensors.py`).

- [ ] **Step 1: Create the test file**

Write the following to `tests/test_lmdeploy/test_turbomind/test_weight_format_resolver.py`:

```python
# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for WeightFormatResolver dispatch logic.

Uses a lightweight fake WeightFormat subclass that stubs out
``make_data_format`` so the resolver can be exercised without the real
``_turbomind`` extension.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types

import pytest
import torch

# ---------------------------------------------------------------------------
# _turbomind stub (same pattern as test_transform_tensors.py)
# ---------------------------------------------------------------------------

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def _setup_fake_tm():
    """Ensure ``_turbomind`` in sys.modules has every attribute the resolver
    and weight_format class bodies touch. Idempotent: augments whatever is
    already there so running after test_transform_tensors.py (which sets up
    a minimal stub) still leaves a usable module.
    """
    tm = sys.modules.get('_turbomind')
    if tm is None:
        tm = types.ModuleType('_turbomind')
        sys.modules['_turbomind'] = tm

    dt = getattr(tm, 'DataType', None)
    if dt is None:
        class DataType:
            pass
        dt = DataType
        tm.DataType = dt

    # Class-body references in weight_format.py and builder/_base.py
    # (_STR_TO_DTYPE, _TORCH_TO_CPP) need these specific names present at
    # module load time.
    for name, val in (('TYPE_FP32', 0), ('TYPE_FP16', 1), ('TYPE_BF16', 2),
                      ('TYPE_INVALID', 3), ('TYPE_INT32', 4),
                      ('TYPE_INT64', 5), ('TYPE_INT8', 6),
                      ('TYPE_UINT8', 7), ('TYPE_UINT4', 10),
                      ('TYPE_FP8_E4M3', 11), ('TYPE_FP4_E2M1', 12)):
        if not hasattr(dt, name):
            setattr(dt, name, val)

    if not hasattr(tm, 'ResolveLinearWeightFormat'):
        tm.ResolveLinearWeightFormat = lambda d, w, bi, bo: ('DataFormat', d, w, bi, bo)


_setup_fake_tm()

# Register package stubs.
import lmdeploy  # noqa: F401
for _pkg in ('lmdeploy.turbomind', 'lmdeploy.turbomind.deploy'):
    if _pkg not in sys.modules:
        mod = types.ModuleType(_pkg)
        mod.__path__ = [os.path.join(_repo_root, *_pkg.split('.'))]
        mod.__package__ = _pkg
        sys.modules[_pkg] = mod


def _load(mod_name, file_rel_path):
    path = os.path.join(_repo_root, *file_rel_path.split('/'))
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_linear_mod = _load('lmdeploy.turbomind.deploy.linear',
                    'lmdeploy/turbomind/deploy/linear.py')
_wf_mod = _load('lmdeploy.turbomind.deploy.weight_format',
                'lmdeploy/turbomind/deploy/weight_format.py')

Linear = _linear_mod.Linear
WeightFormat = _wf_mod.WeightFormat
WeightFormatResolver = _wf_mod.WeightFormatResolver


# ---------------------------------------------------------------------------
# Fake format used by the tests
# ---------------------------------------------------------------------------


class _FakeQuant(WeightFormat):
    """Accepts when a ``.qfoo`` tensor is present. ``normalize`` is identity."""
    name = 'fakeq'
    suffix_map = {'.qfoo': 'weight', '.scales': 'scales', '.bias': 'bias'}
    weight_dtype = 0  # TYPE_FP32 from our stub
    has_zero_point = False

    def __init__(self, *, block_in=None, block_out=None):
        super().__init__(block_in=block_in, block_out=block_out)

    def accepts(self, available):
        return '.qfoo' in available

    def normalize(self, x, kind):
        return x


class _FakeQuantWithZeros(_FakeQuant):
    name = 'fakeqz'
    suffix_map = {'.qfoo': 'weight', '.scales': 'scales',
                  '.qzeros': 'zeros', '.bias': 'bias'}
    has_zero_point = True

    def synthesize_zeros(self, scales):
        return torch.zeros_like(scales)


class _FakeTrivial(WeightFormat):
    name = 'faketr'
    suffix_map = {'.weight': 'weight', '.bias': 'bias'}
    weight_dtype = None
    has_zero_point = False

    def accepts(self, available):
        return available.keys() <= {'.weight', '.bias'} and '.weight' in available

    def normalize(self, x, kind):
        return x

    def dequant(self, tensors, data_type):
        return tensors


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResolveQuantized:

    def _make_resolver(self):
        return WeightFormatResolver(
            data_type=0,
            formats=[_FakeQuant(), _FakeTrivial()])

    def test_quant_prefix_picks_quant_format(self):
        params = {
            'layer.qfoo':   torch.randn(4, 4),
            'layer.scales': torch.randn(1, 4),
        }
        lin = self._make_resolver().resolve(params, 'layer')
        assert isinstance(lin.weight_format, _FakeQuant)
        assert set(lin.tensors) == {'weight', 'scales'}

    def test_trivial_prefix_falls_through(self):
        params = {'layer.weight': torch.randn(4, 4)}
        lin = self._make_resolver().resolve(params, 'layer')
        assert isinstance(lin.weight_format, _FakeTrivial)


class TestResolveFailureModes:

    def _make_resolver(self):
        return WeightFormatResolver(
            data_type=0,
            formats=[_FakeQuant(), _FakeTrivial()])

    def test_missing_prefix_default_raises_key_error(self):
        with pytest.raises(KeyError, match='no checkpoint tensors found'):
            self._make_resolver().resolve({}, 'missing.prefix')

    def test_missing_prefix_optional_returns_none(self):
        assert self._make_resolver().resolve(
            {}, 'missing.prefix', optional=True) is None

    def test_tensors_present_no_match_raises_value_error(self):
        # .qfoo present but dtype not accepted? Our _FakeQuant.accepts is
        # lenient; use a bogus suffix to force no-match.
        params = {'layer.unexpected': torch.randn(4, 4)}
        # The resolver only probes declared suffixes, so .unexpected is
        # filtered out and this reduces to the "no tensors" path. To hit
        # the "tensors present, no match" path, subclass _FakeTrivial to
        # reject a legitimate key pattern.
        class _PickyTrivial(_FakeTrivial):
            def accepts(self, available):
                return False

        resolver = WeightFormatResolver(
            data_type=0,
            formats=[_FakeQuant(), _PickyTrivial()])
        params = {'layer.weight': torch.randn(4, 4)}
        with pytest.raises(ValueError, match='no weight format accepts'):
            resolver.resolve(params, 'layer')


class TestIndexedProbe:

    def test_index_slices_available_tensors(self):
        resolver = WeightFormatResolver(
            data_type=0, formats=[_FakeTrivial()])
        params = {'experts.weight': torch.arange(24).reshape(3, 4, 2).float()}
        lin = resolver.resolve(params, 'experts', index=1)
        assert lin.tensors['weight'].shape == (4, 2)
        torch.testing.assert_close(
            lin.tensors['weight'],
            torch.arange(8, 16).reshape(4, 2).float())


class TestZerosSynthesis:

    def test_synthesize_zeros_called_when_missing(self):
        params = {
            'layer.qfoo':   torch.randn(4, 4),
            'layer.scales': torch.ones(1, 4),
        }
        resolver = WeightFormatResolver(
            data_type=0, formats=[_FakeQuantWithZeros()])
        lin = resolver.resolve(params, 'layer')
        assert 'zeros' in lin.tensors
        torch.testing.assert_close(
            lin.tensors['zeros'], torch.zeros(1, 4))

    def test_synthesize_zeros_skipped_when_present(self):
        scales   = torch.ones(1, 4)
        supplied = torch.full_like(scales, 5.0)
        params = {
            'layer.qfoo':   torch.randn(4, 4),
            'layer.scales': scales,
            'layer.qzeros': supplied,
        }
        resolver = WeightFormatResolver(
            data_type=0, formats=[_FakeQuantWithZeros()])
        lin = resolver.resolve(params, 'layer')
        # supplied zeros were not replaced by synthesis.
        torch.testing.assert_close(lin.tensors['zeros'], supplied)


class TestEquality:

    def test_same_class_same_blocks_equal(self):
        a = _FakeQuant(block_in=128)
        b = _FakeQuant(block_in=128)
        assert a == b
        assert hash(a) == hash(b)
        assert {a, b} == {a}

    def test_different_blocks_unequal(self):
        assert _FakeQuant(block_in=128) != _FakeQuant(block_in=64)

    def test_different_classes_unequal(self):
        assert _FakeQuant() != _FakeTrivial()
```

- [ ] **Step 2: Run the tests**

Run:

```bash
pytest tests/test_lmdeploy/test_turbomind/test_weight_format_resolver.py -v
```

Expected: all tests pass. If any test fails, fix the resolver implementation in Task 1 (the tests describe intended behavior).

- [ ] **Step 3: Commit**

```bash
git add tests/test_lmdeploy/test_turbomind/test_weight_format_resolver.py
git commit -m "$(cat <<'EOF'
deploy: unit tests for WeightFormatResolver dispatch logic

Covers priority dispatch (quant first, trivial fallback), failure modes
(KeyError vs ValueError vs optional-None), indexed-prefix slicing,
has_zero_point synthesis gating, and WeightFormat equality / hashing.
EOF
)"
```

---

### Task 3: Migrate all consumers and delete `kind_map.py`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/linear.py` (TYPE_CHECKING import)
- Modify: `lmdeploy/turbomind/deploy/builder/_base.py` (TRIVIAL_FORMAT import, `_dequant_linear`, `_commit_linear`)
- Modify: `lmdeploy/turbomind/deploy/converter.py` (build resolver)
- Modify: `lmdeploy/turbomind/deploy/spec.py` (`resolver=`, `_linear` with `optional=`, `_cpp_dtype` delegation)
- Modify: `lmdeploy/turbomind/deploy/source_model/utils.py` (signatures + imports)
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py`
- Modify: `tests/test_lmdeploy/test_turbomind/test_transform_tensors.py` (hard-coded path)
- Delete: `lmdeploy/turbomind/deploy/kind_map.py`

All file edits land in a **single commit** — the spec/converter/builder/utils changes are mutually dependent and any partial state leaves the tree non-functional. Verify compile + import at the end, then commit.

- [ ] **Step 1: Update `linear.py` TYPE_CHECKING import**

In `lmdeploy/turbomind/deploy/linear.py`, change line 33:

Before:
```python
if TYPE_CHECKING:
    from .kind_map import WeightFormat
```

After:
```python
if TYPE_CHECKING:
    from .weight_format import WeightFormat
```

- [ ] **Step 2: Update `builder/_base.py` — TRIVIAL_FORMAT import, `_dequant_linear`, `_commit_linear`**

In `lmdeploy/turbomind/deploy/builder/_base.py`, make three edits.

**Edit 1 (line 11):** Replace the import.

Before:
```python
from ..kind_map import TRIVIAL_FORMAT
from ..linear import Linear, pad_out_dim
```

After:
```python
from ..weight_format import TrivialFormat
from ..linear import Linear, pad_out_dim
```

**Edit 2 (lines 122-137):** Rewrite `_dequant_linear` to drop the `fmt.dequant is None` guard and use a fresh `TrivialFormat()` instance.

Before:
```python
def _dequant_linear(linear: Linear, *, data_type) -> Linear:
    """Dequantize a quantized Linear to trivial when the format provides ``dequant``.

    *data_type* is the model's activation dtype; used to construct the new
    trivial ``data_format`` on the result and is threaded into the dequant
    callable so e.g. FP8 produces weights in the caller's activation dtype.
    """
    fmt = linear.weight_format
    if fmt.dequant is None:
        return linear
    new_tensors = fmt.dequant(linear.tensors, data_type)
    return Linear(
        tensors=new_tensors,
        weight_format=TRIVIAL_FORMAT,
        data_format=TRIVIAL_FORMAT.make_data_format(data_type),
    )
```

After:
```python
def _dequant_linear(linear: Linear, *, data_type) -> Linear:
    """Dequantize a quantized Linear to trivial.

    ``TrivialFormat.dequant`` is identity, so already-trivial inputs round-trip
    safely. ``AWQFormat.dequant`` and ``FP8Format.dequant`` do real work.
    GPTQ / CompressedTensor / MXFP4 inherit the base-class
    ``NotImplementedError`` — calling ``_dequant_linear`` on one of those is a
    broken-fusion-group configuration, and the raise names it at the call site.
    """
    fmt = linear.weight_format
    new_tensors = fmt.dequant(linear.tensors, data_type)
    trivial = TrivialFormat()
    return Linear(
        tensors=new_tensors,
        weight_format=trivial,
        data_format=trivial.make_data_format(data_type),
    )
```

**Edit 3 (lines 481-486):** Collapse the packer hoist in `_commit_linear` to a single `fmt.pack(...)` call.

Before:
```python
        packer = fmt.packer if fmt else None
        if packer is not None:
            tensors = {k: packer(t, k) for k, t in linear.tensors.items()}
        else:
            tensors = linear.tensors
        is_quantized = linear.data_format.is_quantized()
```

After:
```python
        tensors = {k: fmt.pack(t, k) for k, t in linear.tensors.items()}
        is_quantized = linear.data_format.is_quantized()
```

The `fmt` local variable is assigned earlier in the function (`fmt = linear.weight_format`, current line 461) and is always a `WeightFormat` instance — no None-check needed because the new `Linear` dataclass field is required.

- [ ] **Step 3: Update `converter.py` — add resolver factory and wire it in**

In `lmdeploy/turbomind/deploy/converter.py`, make three edits.

**Edit 1 (line 1-12):** Swap imports. Remove `dataclasses.replace` + `get_weight_format`; add `_cpp_dtype` and the new weight-format names.

Before:
```python
# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import replace

import torch

from lmdeploy.archs import get_model_arch, search_nested_config
from lmdeploy.messages import TurbomindEngineConfig
from lmdeploy.utils import get_logger

from ...utils import _get_and_verify_max_len, is_bf16_supported
from ..supported_models import SUPPORTED_ARCHS
from .kind_map import get_weight_format
from .source_model.base import INPUT_MODELS
from .source_model.utils import load_model_config
```

After:
```python
# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.archs import get_model_arch, search_nested_config
from lmdeploy.messages import TurbomindEngineConfig
from lmdeploy.utils import get_logger

from ...utils import _get_and_verify_max_len, is_bf16_supported
from ..supported_models import SUPPORTED_ARCHS
from .builder import _cpp_dtype
from .source_model.base import INPUT_MODELS
from .source_model.utils import load_model_config
from .weight_format import (AWQFormat, CompressedTensorFormat, FP8Format,
                            GPTQFormat, MXFP4Format, TrivialFormat,
                            WeightFormat, WeightFormatResolver)
```

**Edit 2:** Insert the `_build_resolver` factory near the top of the file, after the `SUPPORTED_FORMATS` constant definition (which is the line immediately after `logger = get_logger('lmdeploy')`).

Before:
```python
SUPPORTED_FORMATS = ['hf', 'awq', 'gptq', 'compressed-tensors', 'fp8', 'mxfp4', None]
logger = get_logger('lmdeploy')


def _deep_merge(base: dict, override: dict, path: str = '') -> dict:
```

After:
```python
SUPPORTED_FORMATS = ['hf', 'awq', 'gptq', 'compressed-tensors', 'fp8', 'mxfp4', None]
logger = get_logger('lmdeploy')


def _build_resolver(model_format: str | None,
                    group_size: int | None,
                    data_type: "_tm.DataType") -> WeightFormatResolver:
    """Build the active resolver: quantized format (if any) + trivial fallback.

    Called after the int4 fp16 force but before the ``compressed-tensors →
    awq`` rename, so compressed-tensors models get ``CompressedTensorFormat``.
    """
    formats: list[WeightFormat] = []
    if model_format in (None, 'hf'):
        pass
    elif model_format == 'awq':
        formats.append(AWQFormat(block_in=group_size))
    elif model_format == 'gptq':
        formats.append(GPTQFormat(block_in=group_size))
    elif model_format == 'compressed-tensors':
        formats.append(CompressedTensorFormat(block_in=group_size))
    elif model_format == 'fp8':
        formats.append(FP8Format())
    elif model_format == 'mxfp4':
        formats.append(MXFP4Format())
    else:
        raise ValueError(f"unknown model_format: {model_format!r}")
    formats.append(TrivialFormat())
    return WeightFormatResolver(data_type=data_type, formats=formats)


def _deep_merge(base: dict, override: dict, path: str = '') -> dict:
```

**Edit 3 (lines 157-195):** Restructure the resolver / dtype / rename / spec-build sequence. The ordering constraint is that the resolver must be built **after** the int4 fp16 force (so `data_type` is correct) and **before** the CT→AWQ rename (so `CompressedTensorFormat` is used).

Before:
```python
    group_size = _validate_quant_group_size(engine_config.model_format, group_size)
    if engine_config.model_format is None:
        engine_config.model_format = 'hf'

    # Resolve the active WeightFormat before the CT->AWQ rename below, so
    # compressed-tensors models still get COMPRESSED_TENSOR_FORMAT (correct
    # suffixes) rather than AWQ_FORMAT after the rename.
    weight_format = get_weight_format(engine_config.model_format)
    if weight_format.block_in == 0:
        weight_format = replace(weight_format, block_in=group_size)

    # 3. Resolve dtype and format overrides.
    dtype = _resolve_dtype(engine_config.dtype, hf_model_cfg)
    if engine_config.model_format in ('awq', 'gptq', 'compressed-tensors'):
        dtype = 'float16'
        if engine_config.model_format == 'compressed-tensors':
            engine_config.model_format = 'awq'

    # 4. Resolve session_len default.
    session_len_default = _get_and_verify_max_len(hf_model_cfg, None)

    # 5. Mutate engine_config with resolved values.
    engine_config.dtype = dtype
    if engine_config.session_len is None:
        engine_config.session_len = session_len_default
    engine_config.attn_tp_size = engine_config.attn_tp_size or 1
    engine_config.attn_cp_size = engine_config.attn_cp_size or 1
    engine_config.mlp_tp_size = engine_config.mlp_tp_size or 1

    # 6. Build spec (hf_overrides handling unchanged).
    hf_cfg = load_model_config(model_path)
    if engine_config.hf_overrides:
        logger.warning(f'Overriding HF config with {engine_config.hf_overrides}')
        _deep_merge(hf_cfg, engine_config.hf_overrides)
    spec_name = get_spec_registered_name(model_path, engine_config.model_format)
    spec_cls = INPUT_MODELS.get(spec_name)
    spec = spec_cls(hf_cfg, engine_config, weight_format=weight_format)

    return spec, model_path
```

After:
```python
    group_size = _validate_quant_group_size(engine_config.model_format, group_size)
    if engine_config.model_format is None:
        engine_config.model_format = 'hf'

    # 3. Resolve dtype and format overrides.
    dtype = _resolve_dtype(engine_config.dtype, hf_model_cfg)
    if engine_config.model_format in ('awq', 'gptq', 'compressed-tensors'):
        dtype = 'float16'
    engine_config.dtype = dtype

    # Build resolver after dtype is finalized but before the CT→AWQ rename,
    # so compressed-tensors models instantiate CompressedTensorFormat.
    resolver = _build_resolver(engine_config.model_format,
                               group_size, _cpp_dtype(dtype))

    # C++-side label rename (does not affect resolver).
    if engine_config.model_format == 'compressed-tensors':
        engine_config.model_format = 'awq'

    # 4. Resolve session_len default.
    session_len_default = _get_and_verify_max_len(hf_model_cfg, None)

    # 5. Mutate engine_config with remaining resolved values.
    if engine_config.session_len is None:
        engine_config.session_len = session_len_default
    engine_config.attn_tp_size = engine_config.attn_tp_size or 1
    engine_config.attn_cp_size = engine_config.attn_cp_size or 1
    engine_config.mlp_tp_size = engine_config.mlp_tp_size or 1

    # 6. Build spec (hf_overrides handling unchanged).
    hf_cfg = load_model_config(model_path)
    if engine_config.hf_overrides:
        logger.warning(f'Overriding HF config with {engine_config.hf_overrides}')
        _deep_merge(hf_cfg, engine_config.hf_overrides)
    spec_name = get_spec_registered_name(model_path, engine_config.model_format)
    spec_cls = INPUT_MODELS.get(spec_name)
    spec = spec_cls(hf_cfg, engine_config, resolver=resolver)

    return spec, model_path
```

- [ ] **Step 4: Update `spec.py` base class**

In `lmdeploy/turbomind/deploy/spec.py`, make two edits.

**Edit 1 (line 12):** Drop the `_cpp_dtype` import alias since it's no longer used.

Before:
```python
from .builder import _cpp_dtype as _cd, NormBuilder, make_norm_config
```

After:
```python
from .builder import NormBuilder, make_norm_config
```

**Edit 2 (lines 46-138):** Replace the `__init__`, `_linear`, and `_cpp_dtype` methods. The class docstring and other methods (`_parse_base`, `bind_runtime`, `set_params`, `_get`, `_apply_rope`, `norm`, `qk_norm`) stay exactly as they are.

Before:
```python
    def __init__(self, hf_cfg: dict, engine_cfg: 'TurbomindEngineConfig',
                 *, weight_format):
        """Parse HF config into orchestration scalars.

        ``weight_format`` is the resolved `WeightFormat` for the model's
        quantization format, produced by the converter. It lands on
        ``self._weight_format`` so ``build_linear()`` can use it during
        weight loading.

        Subclasses override `_parse_base` (or extend in their own __init__)
        then construct C++ config templates and per-layer lists.
        """
        self.hf_cfg = hf_cfg
        self.engine_cfg = engine_cfg
        self._weight_format = weight_format
        self._parse_base(hf_cfg)
```

After:
```python
    def __init__(self, hf_cfg: dict, engine_cfg: 'TurbomindEngineConfig',
                 *, resolver):
        """Parse HF config into orchestration scalars.

        ``resolver`` is a ``WeightFormatResolver`` built by the converter.
        It carries the model compute dtype and the ordered list of
        candidate weight formats; ``self._linear()`` delegates to
        ``resolver.resolve()`` at weight-loading time.

        Subclasses override `_parse_base` (or extend in their own __init__)
        then construct C++ config templates and per-layer lists.
        """
        self.hf_cfg = hf_cfg
        self.engine_cfg = engine_cfg
        self._resolver = resolver
        self._parse_base(hf_cfg)
```

And replace the `_linear` / `_cpp_dtype` pair:

Before:
```python
    def _linear(self, pfx: str):
        from .kind_map import build_linear
        return build_linear(self.params, pfx,
                            data_type=self._cpp_dtype(),
                            weight_format=self._weight_format)

    def _cpp_dtype(self):
        return _cd(self.engine_cfg.dtype)
```

After:
```python
    def _linear(self, pfx: str, *, optional: bool = False):
        return self._resolver.resolve(self.params, pfx, optional=optional)

    def _cpp_dtype(self):
        return self._resolver.data_type
```

- [ ] **Step 5: Update `source_model/utils.py` — imports and signatures**

In `lmdeploy/turbomind/deploy/source_model/utils.py`, make three edits.

**Edit 1 (lines 12-14):** Replace the kind_map imports.

Before:
```python
from ..kind_map import TRIVIAL_FORMAT, build_linear
from ..linear import Linear
from ..builder._base import _dequant_linear
```

After:
```python
from ..linear import Linear
from ..builder._base import _dequant_linear
```

`TRIVIAL_FORMAT` and `build_linear` aren't referenced anywhere else in the file after the rest of this step's edits — grep will confirm at the end of the task.

**Edit 2 (lines 180-227):** Update `reorder_rotary_emb` signature — replace the `data_type` kwarg with `resolver`. The Tensor path is unchanged.

Before:
```python
def reorder_rotary_emb(x, head_dim: int, rope_dim: int, *, data_type=None):
    """Apply RoPE layout permutation.

    Accepts either a ``Linear`` or a raw ``torch.Tensor``.

    For ``Linear`` inputs the permutation is applied to every tensor in the
    bundle with quantization awareness (block-alignment check, dequant
    fallback, block-level shuffling for scales/zeros).  ``data_type`` is
    required and must not be ``None``.

    For ``torch.Tensor`` inputs the element-level interleave-transpose is
    applied directly.  ``data_type`` is ignored.
    """
    from ..linear import Linear

    if isinstance(x, Linear):
        if data_type is None:
            raise TypeError(
                "data_type is required when passing a Linear to reorder_rotary_emb"
            )
        wfmt = x.weight_format
        block_out = wfmt.block_out or 0

        # If blocks don't align with heads, dequant first
        if block_out and block_out % head_dim != 0:
            x = _dequant_linear(x, data_type=data_type)
            block_out = 0

        new_tensors = {}
        for kind, tensor in x.tensors.items():
            if kind in ("scales", "zeros") and block_out > 0:
                # Block-level shuffle: reinterpret each block as a "head"
                # so _reorder_rotary_emb shuffles at block granularity.
                blocks_per_head = block_out // head_dim
                if blocks_per_head <= 1:
                    new_tensors[kind] = tensor
                else:
                    rope_dim_blocks = rope_dim * blocks_per_head // head_dim
                    new_tensors[kind] = _reorder_rotary_emb(tensor, blocks_per_head, rope_dim_blocks)
            elif tensor.size(-1) % head_dim == 0:
                new_tensors[kind] = _reorder_rotary_emb(tensor, head_dim, rope_dim)
            else:
                new_tensors[kind] = tensor

        return Linear(tensors=new_tensors, weight_format=x.weight_format,
                      data_format=x.data_format)

    return _reorder_rotary_emb(x, head_dim, rope_dim)
```

After:
```python
def reorder_rotary_emb(x, head_dim: int, rope_dim: int, *, resolver=None):
    """Apply RoPE layout permutation.

    Accepts either a ``Linear`` or a raw ``torch.Tensor``.

    For ``Linear`` inputs the permutation is applied to every tensor in the
    bundle with quantization awareness (block-alignment check, dequant
    fallback, block-level shuffling for scales/zeros). ``resolver`` is
    required and must not be ``None`` — it supplies the compute dtype
    threaded into ``_dequant_linear``.

    For ``torch.Tensor`` inputs the element-level interleave-transpose is
    applied directly. ``resolver`` is ignored.
    """
    from ..linear import Linear

    if isinstance(x, Linear):
        if resolver is None:
            raise TypeError(
                "resolver is required when passing a Linear to reorder_rotary_emb"
            )
        data_type = resolver.data_type
        wfmt = x.weight_format
        block_out = wfmt.block_out or 0

        # If blocks don't align with heads, dequant first
        if block_out and block_out % head_dim != 0:
            x = _dequant_linear(x, data_type=data_type)
            block_out = 0

        new_tensors = {}
        for kind, tensor in x.tensors.items():
            if kind in ("scales", "zeros") and block_out > 0:
                # Block-level shuffle: reinterpret each block as a "head"
                # so _reorder_rotary_emb shuffles at block granularity.
                blocks_per_head = block_out // head_dim
                if blocks_per_head <= 1:
                    new_tensors[kind] = tensor
                else:
                    rope_dim_blocks = rope_dim * blocks_per_head // head_dim
                    new_tensors[kind] = _reorder_rotary_emb(tensor, blocks_per_head, rope_dim_blocks)
            elif tensor.size(-1) % head_dim == 0:
                new_tensors[kind] = _reorder_rotary_emb(tensor, head_dim, rope_dim)
            else:
                new_tensors[kind] = tensor

        return Linear(tensors=new_tensors, weight_format=x.weight_format,
                      data_format=x.data_format)

    return _reorder_rotary_emb(x, head_dim, rope_dim)
```

**Edit 3 (lines 264-321):** Update `read_packed_moe_expert` signature and body. The function switches from `(data_type, weight_format)` to a single `resolver` kwarg.

Before:
```python
def read_packed_moe_expert(
    params: dict,
    gate_up_pfx: str,
    down_pfx: str,
    expert_idx: int,
    *,
    data_type,
    weight_format,
    interleaved: bool = False,
    trans: bool = False,
) -> tuple[Linear, Linear, Linear]:
    """Read one packed MoE expert's fused gate_up + down and split into
    (w1, w2, w3) Linears in TM layout.

    ``gate_up_pfx`` and ``down_pfx`` are the full prefixes to the two
    packed tensors (e.g. ``'model.layers.5.mlp.experts.gate_up_proj'``).
    The caller composes these strings; this helper concatenates nothing.

    Parameters
    ----------
    interleaved : bool
        Split scheme for the fused gate_up output dim.
        ``False`` -> contiguous ``[..., :half]`` / ``[..., half:]`` (qwen3.5).
        ``True``  -> stride-2 interleaved ``[..., ::2]`` / ``[..., 1::2]`` (gpt-oss).
    trans : bool
        For trivial-format checkpoints that store the packed tensor in
        ``[n_experts, in, out]`` layout (gpt-oss), transposes the 2D
        ``weight`` tensor to undo the HF-to-TM transpose applied by
        ``_normalize_trivial``. Only affects the ``weight`` kind on
        trivial-format linears; quantized formats use their own normalizers.
    """
    gate_up = build_linear(params, gate_up_pfx, index=expert_idx,
                           data_type=data_type, weight_format=weight_format)
    down    = build_linear(params, down_pfx,    index=expert_idx,
                           data_type=data_type, weight_format=weight_format)

    if trans:
        for lin in (gate_up, down):
            if lin.weight_format.name == 'trivial':
                w = lin.tensors.get('weight')
                if w is not None and w.dim() == 2:
                    lin.tensors['weight'] = w.t().contiguous()

    w1_t: dict[str, torch.Tensor] = {}
    w3_t: dict[str, torch.Tensor] = {}
    for kind, t in gate_up.tensors.items():
        if interleaved:
            w1_t[kind] = t[..., ::2].contiguous()
            w3_t[kind] = t[..., 1::2].contiguous()
        else:
            half = t.shape[-1] // 2
            w1_t[kind] = t[..., :half].contiguous()
            w3_t[kind] = t[..., half:].contiguous()
    w1 = Linear(tensors=w1_t, weight_format=gate_up.weight_format,
                data_format=gate_up.data_format)
    w3 = Linear(tensors=w3_t, weight_format=gate_up.weight_format,
                data_format=gate_up.data_format)
    return w1, down, w3
```

After:
```python
def read_packed_moe_expert(
    params: dict,
    gate_up_pfx: str,
    down_pfx: str,
    expert_idx: int,
    *,
    resolver,
    interleaved: bool = False,
    trans: bool = False,
) -> tuple[Linear, Linear, Linear]:
    """Read one packed MoE expert's fused gate_up + down and split into
    (w1, w2, w3) Linears in TM layout.

    ``gate_up_pfx`` and ``down_pfx`` are the full prefixes to the two
    packed tensors (e.g. ``'model.layers.5.mlp.experts.gate_up_proj'``).
    The caller composes these strings; this helper concatenates nothing.

    Parameters
    ----------
    interleaved : bool
        Split scheme for the fused gate_up output dim.
        ``False`` -> contiguous ``[..., :half]`` / ``[..., half:]`` (qwen3.5).
        ``True``  -> stride-2 interleaved ``[..., ::2]`` / ``[..., 1::2]`` (gpt-oss).
    trans : bool
        For trivial-format checkpoints that store the packed tensor in
        ``[n_experts, in, out]`` layout (gpt-oss), transposes the 2D
        ``weight`` tensor to undo the HF-to-TM transpose applied by
        ``TrivialFormat.normalize``. Only affects the ``weight`` kind on
        trivial-format linears; quantized formats use their own normalizers.
    """
    gate_up = resolver.resolve(params, gate_up_pfx, index=expert_idx)
    down    = resolver.resolve(params, down_pfx,    index=expert_idx)

    if trans:
        for lin in (gate_up, down):
            if lin.weight_format.name == 'trivial':
                w = lin.tensors.get('weight')
                if w is not None and w.dim() == 2:
                    lin.tensors['weight'] = w.t().contiguous()

    w1_t: dict[str, torch.Tensor] = {}
    w3_t: dict[str, torch.Tensor] = {}
    for kind, t in gate_up.tensors.items():
        if interleaved:
            w1_t[kind] = t[..., ::2].contiguous()
            w3_t[kind] = t[..., 1::2].contiguous()
        else:
            half = t.shape[-1] // 2
            w1_t[kind] = t[..., :half].contiguous()
            w3_t[kind] = t[..., half:].contiguous()
    w1 = Linear(tensors=w1_t, weight_format=gate_up.weight_format,
                data_format=gate_up.data_format)
    w3 = Linear(tensors=w3_t, weight_format=gate_up.weight_format,
                data_format=gate_up.data_format)
    return w1, down, w3
```

- [ ] **Step 6: Update `qwen3_spec.py` — constructor kwarg and reorder_rotary_emb call sites**

In `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py`, make two edits.

**Edit 1 (lines 30-31):** Rename the constructor kwarg.

Before:
```python
    def __init__(self, hf_cfg: dict, engine_cfg, *, weight_format):
        super().__init__(hf_cfg, engine_cfg, weight_format=weight_format)
```

After:
```python
    def __init__(self, hf_cfg: dict, engine_cfg, *, resolver):
        super().__init__(hf_cfg, engine_cfg, resolver=resolver)
```

**Edit 2 (lines 124-127):** Swap `data_type=self._cpp_dtype()` for `resolver=self._resolver`.

Before:
```python
        q = reorder_rotary_emb(q, self._head_dim, self._rope.dim,
                               data_type=self._cpp_dtype())
        k = reorder_rotary_emb(k, self._head_dim, self._rope.dim,
                               data_type=self._cpp_dtype())
```

After:
```python
        q = reorder_rotary_emb(q, self._head_dim, self._rope.dim,
                               resolver=self._resolver)
        k = reorder_rotary_emb(k, self._head_dim, self._rope.dim,
                               resolver=self._resolver)
```

- [ ] **Step 7: Update `qwen3_5_spec.py` — constructor, reorder_rotary_emb, `ffn` optional, `read_packed_moe_expert`**

In `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py`, make five edits.

**Edit 1 (lines 34-35):** Rename the constructor kwarg.

Before:
```python
    def __init__(self, hf_cfg: dict, engine_cfg, *, weight_format):
        super().__init__(hf_cfg, engine_cfg, weight_format=weight_format)
```

After:
```python
    def __init__(self, hf_cfg: dict, engine_cfg, *, resolver):
        super().__init__(hf_cfg, engine_cfg, resolver=resolver)
```

**Edit 2 (lines 178-181):** Swap `reorder_rotary_emb` kwarg.

Before:
```python
        q = reorder_rotary_emb(q, self._head_dim, self._rope.dim,
                               data_type=self._cpp_dtype())
        k = reorder_rotary_emb(k, self._head_dim, self._rope.dim,
                               data_type=self._cpp_dtype())
```

After:
```python
        q = reorder_rotary_emb(q, self._head_dim, self._rope.dim,
                               resolver=self._resolver)
        k = reorder_rotary_emb(k, self._head_dim, self._rope.dim,
                               resolver=self._resolver)
```

**Edit 3: `ffn` gains `optional=` and replaces the all-None guard with a strict all-or-nothing check.** This is the core optional-propagation change. The current `ffn` already has a naive `if w1 is None and w2 is None and w3 is None: return None` check that relied on `build_linear` silently returning None; with the resolver defaulting to `raise`, we need the `optional=` kwarg to thread through, and the partial-absence case (one or two missing but not all three) becomes an explicit error instead of a silently mixed bundle.

Before (lines 224-241; exact line numbers may shift after Edits 1-2):
```python
    def ffn(self, pfx, layer, inter_size=None, fused_moe=False):
        w1 = self._linear(f'{pfx}.gate_proj')
        w3 = self._linear(f'{pfx}.up_proj')
        w2 = self._linear(f'{pfx}.down_proj')
        if w1 is None and w2 is None and w3 is None:
            return None

        cfg = self._ffn_cfg.clone()
        cfg.inter_size = (inter_size if inter_size is not None
                          else self._inter_sizes[layer])
        cfg.fuse_silu  = False
        cfg.fused_moe  = fused_moe

        m = FfnBuilder(cfg, self._contexts,
                       tp=self.engine_cfg.mlp_tp_size,
                       ranks=self._mlp_ranks)
        m.add_ffn(w1, w2, w3)
        return m
```

After:
```python
    def ffn(self, pfx, layer, inter_size=None, fused_moe=False, *,
            optional: bool = False):
        w1 = self._linear(f'{pfx}.gate_proj', optional=optional)
        w3 = self._linear(f'{pfx}.up_proj',   optional=optional)
        w2 = self._linear(f'{pfx}.down_proj', optional=optional)

        present = [t is not None for t in (w1, w3, w2)]
        if not any(present):
            return None                                     # all absent → fallback signal
        if not all(present):
            raise ValueError(
                f'{pfx}: partial FFN checkpoint '
                f'(gate_proj={present[0]}, up_proj={present[1]}, down_proj={present[2]})')

        cfg = self._ffn_cfg.clone()
        cfg.inter_size = (inter_size if inter_size is not None
                          else self._inter_sizes[layer])
        cfg.fuse_silu  = False
        cfg.fused_moe  = fused_moe

        m = FfnBuilder(cfg, self._contexts,
                       tp=self.engine_cfg.mlp_tp_size,
                       ranks=self._mlp_ranks)
        m.add_ffn(w1, w2, w3)
        return m
```

**Edit 4: Update `_moe_expert_ffn` to pass `optional=True`.**

Before:
```python
    def _moe_expert_ffn(self, mlp_pfx, layer, expert_idx, inter_size):
        expert_pfx = f'{mlp_pfx}.experts.{expert_idx}'
        return (self.ffn(expert_pfx, layer, inter_size=inter_size, fused_moe=True)
                or self._packed_moe_ffn(mlp_pfx, expert_idx, inter_size))
```

After:
```python
    def _moe_expert_ffn(self, mlp_pfx, layer, expert_idx, inter_size):
        expert_pfx = f'{mlp_pfx}.experts.{expert_idx}'
        return (self.ffn(expert_pfx, layer, inter_size=inter_size,
                         fused_moe=True, optional=True)
                or self._packed_moe_ffn(mlp_pfx, expert_idx, inter_size))
```

**Edit 5: Update `_packed_moe_ffn` to use `resolver=`.** Find the existing `_packed_moe_ffn` definition and replace the `read_packed_moe_expert` call block.

Before (approximately lines 270-278):
```python
    def _packed_moe_ffn(self, mlp_pfx, expert_idx, inter_size):
        w1, w2, w3 = read_packed_moe_expert(
            self.params,
            f'{mlp_pfx}.experts.gate_up_proj',
            f'{mlp_pfx}.experts.down_proj',
            expert_idx,
            data_type=self._cpp_dtype(),
            weight_format=self._weight_format,
        )
```

After:
```python
    def _packed_moe_ffn(self, mlp_pfx, expert_idx, inter_size):
        w1, w2, w3 = read_packed_moe_expert(
            self.params,
            f'{mlp_pfx}.experts.gate_up_proj',
            f'{mlp_pfx}.experts.down_proj',
            expert_idx,
            resolver=self._resolver,
        )
```

- [ ] **Step 8: Update `gpt_oss_spec.py` — constructor, reorder_rotary_emb, `read_packed_moe_expert`**

In `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py`, make three edits.

**Edit 1 (lines 35-36):** Rename the constructor kwarg.

Before:
```python
    def __init__(self, hf_cfg: dict, engine_cfg, *, weight_format):
        super().__init__(hf_cfg, engine_cfg, weight_format=weight_format)
```

After:
```python
    def __init__(self, hf_cfg: dict, engine_cfg, *, resolver):
        super().__init__(hf_cfg, engine_cfg, resolver=resolver)
```

**Edit 2 (lines 132-135):** Swap `reorder_rotary_emb` kwarg.

Before:
```python
        q = reorder_rotary_emb(q, self._head_dim, self._rope.dim,
                               data_type=self._cpp_dtype())
        k = reorder_rotary_emb(k, self._head_dim, self._rope.dim,
                               data_type=self._cpp_dtype())
```

After:
```python
        q = reorder_rotary_emb(q, self._head_dim, self._rope.dim,
                               resolver=self._resolver)
        k = reorder_rotary_emb(k, self._head_dim, self._rope.dim,
                               resolver=self._resolver)
```

**Edit 3 (lines 205-215):** Swap the `read_packed_moe_expert` call.

Before:
```python
    def _packed_moe_ffn(self, mlp_pfx, expert_idx, inter_size):
        w1, w2, w3 = read_packed_moe_expert(
            self.params,
            f'{mlp_pfx}.experts.gate_up_proj',
            f'{mlp_pfx}.experts.down_proj',
            expert_idx,
            data_type=self._cpp_dtype(),
            weight_format=self._weight_format,
            interleaved=True,
            trans=True,
        )
```

After:
```python
    def _packed_moe_ffn(self, mlp_pfx, expert_idx, inter_size):
        w1, w2, w3 = read_packed_moe_expert(
            self.params,
            f'{mlp_pfx}.experts.gate_up_proj',
            f'{mlp_pfx}.experts.down_proj',
            expert_idx,
            resolver=self._resolver,
            interleaved=True,
            trans=True,
        )
```

- [ ] **Step 9: Update `glm4_moe_lite_spec.py` — constructor and MLA `q_b_proj` optional fallback**

In `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py`, make two edits.

**Edit 1 (lines 24-25):** Rename the constructor kwarg.

Before:
```python
    def __init__(self, hf_cfg: dict, engine_cfg, *, weight_format):
        super().__init__(hf_cfg, engine_cfg, weight_format=weight_format)
```

After:
```python
    def __init__(self, hf_cfg: dict, engine_cfg, *, resolver):
        super().__init__(hf_cfg, engine_cfg, resolver=resolver)
```

**Edit 2 (lines 171-172):** Add `optional=True` to the `q_b_proj` probe so the `or`-fallback to `q_proj` stays a legitimate "module absent" signal rather than a broken-checkpoint error.

Before:
```python
        q_b = (self._linear(f'{pfx}.q_b_proj') or
               self._linear(f'{pfx}.q_proj'))
```

After:
```python
        q_b = (self._linear(f'{pfx}.q_b_proj', optional=True) or
               self._linear(f'{pfx}.q_proj'))
```

The second call is non-optional: if both `q_b_proj` and `q_proj` are missing, the checkpoint is broken and we want the resolver's `KeyError`.

- [ ] **Step 10: Update the test file's hard-coded `kind_map.py` path and module-load order**

In `tests/test_lmdeploy/test_turbomind/test_transform_tensors.py`, replace lines 85-92 (the whole "Load kind_map first / Load linear.py" block). The load order must change: `linear.py` now loads **before** `weight_format.py` because the new `weight_format.py` imports `from .linear import Linear` at module level, and letting `weight_format` trigger an implicit first-load of `linear` would race with the test's own explicit `_load_module_from_file('lmdeploy.turbomind.deploy.linear', ...)` and leave two incompatible copies of the `Linear` class in sys.modules.

Before:
```python
# Load kind_map first (needed by linear and _base)
_kind_map_path = os.path.join(_repo_root, 'lmdeploy', 'turbomind', 'deploy', 'kind_map.py')
_load_module_from_file('lmdeploy.turbomind.deploy.kind_map', _kind_map_path)

# Load linear.py
_linear_path = os.path.join(_repo_root, 'lmdeploy', 'turbomind', 'deploy', 'linear.py')
_linear_mod = _load_module_from_file('lmdeploy.turbomind.deploy.linear', _linear_path)
Linear = _linear_mod.Linear
```

After:
```python
# Load linear.py first — weight_format.py imports from .linear at module level.
_linear_path = os.path.join(_repo_root, 'lmdeploy', 'turbomind', 'deploy', 'linear.py')
_linear_mod = _load_module_from_file('lmdeploy.turbomind.deploy.linear', _linear_path)
Linear = _linear_mod.Linear

# Load weight_format (needed by _base for TrivialFormat)
_wf_path = os.path.join(_repo_root, 'lmdeploy', 'turbomind', 'deploy', 'weight_format.py')
_load_module_from_file('lmdeploy.turbomind.deploy.weight_format', _wf_path)
```

- [ ] **Step 11: Delete `kind_map.py`**

Run:

```bash
git rm lmdeploy/turbomind/deploy/kind_map.py
```

- [ ] **Step 12: Sanity-check that all cross-file references resolved**

Run:

```bash
rg -n 'from \.\.kind_map|from \.kind_map|from lmdeploy\.turbomind\.deploy\.kind_map|import kind_map' lmdeploy tests
```

Expected: zero matches. Any remaining reference is a missed migration site.

Run:

```bash
rg -n '_weight_format\b|TRIVIAL_FORMAT\b|\.packer\b|\bbuild_linear\b|get_weight_format' lmdeploy/turbomind
```

Expected: zero matches inside `lmdeploy/turbomind/`. Scoped to `lmdeploy/turbomind/` to avoid false positives from `lmdeploy/pytorch/nn/linear/__init__.py::build_linear`, which is an unrelated PyTorch-backend function in a different subsystem.

Run:

```bash
python -c "import lmdeploy.turbomind.deploy.converter; import lmdeploy.turbomind.deploy.spec; import lmdeploy.turbomind.deploy.source_model.utils; import lmdeploy.turbomind.deploy.source_model; print('OK')"
```

Expected output: `OK`. This forces every updated module to import successfully and runs each `@INPUT_MODELS.register_module` decorator.

Run:

```bash
pytest tests/test_lmdeploy/test_turbomind/test_transform_tensors.py -v
pytest tests/test_lmdeploy/test_turbomind/test_weight_format_resolver.py -v
```

Expected: both test files pass. If `test_transform_tensors.py` fails with `FileNotFoundError` pointing at `weight_format.py`, the path edit in Step 10 is wrong — revisit.

- [ ] **Step 13: Commit the migration**

```bash
git add -A
git commit -m "$(cat <<'EOF'
deploy: migrate to WeightFormatResolver and delete kind_map.py

Routes every checkpoint-loading path through WeightFormatResolver.resolve,
with the resolver built once by the converter from the final dtype and
the model_format (before the CT->AWQ label rename). Specs now accept
resolver= instead of weight_format= and their _linear() delegates to
resolver.resolve() with an optional= kwarg for legitimate-absence sites
(q_b_proj in MLA, per-expert FFN fallback in qwen3.5 MoE).

Utility changes:
- read_packed_moe_expert takes a single resolver= (was data_type + weight_format)
- reorder_rotary_emb's Linear path takes resolver= (was data_type)
- _dequant_linear drops the `fmt.dequant is None` guard; TrivialFormat.dequant
  is identity, AWQ/FP8 do real work, and GPTQ/CT/MXFP4 inherit the
  base-class NotImplementedError (mixed-fusion misuse is now loud)
- _commit_linear's packer hoist collapses to a single fmt.pack(...) call

Test file test_transform_tensors.py updates its hard-coded kind_map path
to weight_format. test_compressed_tensors.py is pre-broken (imports from
the long-removed parameter.py) and is left untouched.

kind_map.py is deleted; weight_format.py is now the single authority.
EOF
)"
```

- [ ] **Step 14: Verify the commit compiled cleanly**

```bash
git log -1 --stat
```

Sanity-check that the commit includes all 10 modified files plus the `kind_map.py` deletion. If any file is missing, the `git add -A` catch-all missed something — investigate.

---

### Task 4: End-to-end verification matrix

**Files:** None (verification only).

Per `AGENTS.md`, deploy-pipeline regressions are caught by end-to-end model inference through `scripts/test_turbomind_model.py`. This task runs one model per resolver path to confirm the refactor didn't break any format.

Requirements for each run (per `AGENTS.md`):
- The conversion completes without errors.
- The response under `--- response begin ---` is meaningful human text (not gibberish, not empty, not truncated).
- Generated token count (the `generated:` line in `--- tokens ---`) is ≥ 128.

Any gibberish, truncation, or crash indicates a regression. Debug and fix before proceeding — per `AGENTS.md`'s debugging guidance, do not stop with active bugs.

- [ ] **Step 1: Check GPU availability**

Use the `get_gpu_usage` MCP tool to find an empty GPU. If none are free, wait or skip to a quieter time.

- [ ] **Step 2: Pick one model per format path**

Use the `list_models` MCP tool to find cached models covering the full matrix. Prefer the smallest variant per path that still exercises the path:

| Path                 | Resolver candidates                         | Example `model_type` / config hint                    |
|----------------------|---------------------------------------------|-------------------------------------------------------|
| trivial (hf)         | `[TrivialFormat]`                           | any non-quantized Qwen3 / Llama-style model           |
| awq                  | `[AWQFormat, TrivialFormat]`                | AWQ-quantized model with `quant_method: awq`          |
| gptq                 | `[GPTQFormat, TrivialFormat]`               | GPTQ model with `quant_method: gptq, sym: true`       |
| compressed-tensors   | `[CompressedTensorFormat, TrivialFormat]`   | compressed-tensors model with pack-quantized int4     |
| fp8                  | `[FP8Format, TrivialFormat]`                | FP8 model (e.g. DeepSeek or a DS-style fp8 variant)   |
| mxfp4 (gpt-oss)      | `[MXFP4Format, TrivialFormat]`              | gpt-oss native mxfp4 release (router stays trivial — canonical mixed-format demonstration) |

- [ ] **Step 3: Run one model test per selected path**

For each selected model, use the `get_model_cache_path` MCP tool to find its cache dir, then run:

```bash
python scripts/test_turbomind_model.py <model_path> <cache_dir> <tp> <gpus>
```

Verify each run against the three requirements listed at the top of this task.

- [ ] **Step 4: Manually exercise the loud-failure paths (optional spot-check)**

These aren't automated; run if you have spare time and want extra confidence in the failure diagnostics.

Wrong `--model-format`: pick any cached trivial (non-quantized) model and force `model_format='awq'` in engine config. Expected: `ValueError` mentioning "no weight format accepts tensors" plus the prefix and candidate list.

Missing prefix without `optional=True`: attempt conversion on a model whose `lm_head` is tied to embeddings but with a buggy spec edit that calls `self._linear('lm_head')` (without `optional=True`) when the key is absent. Expected: `KeyError` mentioning the prefix and candidate suffixes. (Revert the buggy edit after verifying.)

- [ ] **Step 5: Commit only if a fix was needed during verification**

If no regressions were found: no additional commit — the Task 3 commit stands alone.

If a fix was needed, commit it separately with a descriptive message so the history clearly shows what end-to-end testing caught.

---

## Self-review

**Spec coverage:**

| Spec section                            | Implementing task(s)              |
|-----------------------------------------|-----------------------------------|
| WeightFormat class hierarchy            | Task 1 steps 2-8                  |
| WeightFormatResolver                    | Task 1 step 9                     |
| `__eq__` / `__hash__`                   | Task 1 step 2; Task 2 step 1 test |
| Converter build-resolver + ordering     | Task 3 step 3 edits 2-3           |
| Spec base `resolver=` + `_linear`       | Task 3 step 4                     |
| Spec subclasses (all four)              | Task 3 steps 6-9                  |
| `optional=True` propagation through ffn | Task 3 step 7 edits 3-4           |
| MLA `q_b_proj` optional fallback        | Task 3 step 9 edit 2              |
| `read_packed_moe_expert` signature      | Task 3 step 5 edit 3              |
| `reorder_rotary_emb` signature          | Task 3 step 5 edit 2              |
| `_dequant_linear` guard drop            | Task 3 step 2 edit 2              |
| `_commit_linear` packer hoist collapse  | Task 3 step 2 edit 3              |
| `linear.py` TYPE_CHECKING import        | Task 3 step 1                     |
| Test path update                        | Task 3 step 10                    |
| `kind_map.py` deletion                  | Task 3 step 11                    |
| End-to-end verification matrix          | Task 4                            |
