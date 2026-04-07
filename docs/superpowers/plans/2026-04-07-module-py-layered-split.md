# module.py Layered Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `module.py` (1072 lines) into three focused files organized by operational layer (read/assemble, transform, shard/commit), with a backward-compatible facade.

**Architecture:** Three new files — `spec.py`, `transforms.py`, `commit.py` — each with a single responsibility. `module.py` becomes a re-export facade. All external imports continue working unchanged.

**Tech Stack:** Python 3, PyTorch, pybind11 (`_turbomind` extension).

---

### Task 1: Create `spec.py` — Read & Assemble layer

**Files:**
- Create: `lmdeploy/turbomind/deploy/spec.py`
- Reference: `lmdeploy/turbomind/deploy/module.py` (read-only source)

This file contains `TextModelSpec` ABC and all functions that read/assemble weights from checkpoint into `Linear` bundles.

- [ ] **Step 1: Create `spec.py` with imports**

```python
# Copyright (c) OpenMMLab. All rights reserved.
"""Read & Assemble layer: read weights from checkpoint, build Linear bundles,
assemble composite weights (QKV merge, GDN fusion)."""
from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from .kind_map import DENSE_FORMAT
from .linear import Linear

if TYPE_CHECKING:
    from .target_model.base import BaseOutputModel
```

- [ ] **Step 2: Add `SplitSide` enum and tensor helper functions**

Copy these from `module.py` verbatim — they are used by spec-level assembly functions:

```python
class SplitSide(enum.Enum):
    """Semantic TP split direction for ``commit_linear`` / ``commit_tensor``.

    ``OUTPUT`` — column-parallel: split along the output dimension
    ``INPUT``  — row-parallel:    split along the input dimension
    """

    OUTPUT = "output"
    INPUT = "input"


def permute_v2(x: torch.Tensor, size_per_head: int = 128):
    """
        Contract: x.size(-1) is output dims
    """

    assert x.size(-1) > 1

    output_dims = x.size(-1)
    head_num = output_dims // size_per_head

    return x.view(-1, head_num, 2, size_per_head // 2).transpose(2, 3).reshape(x.shape)


def permute_v2_partial(x: torch.Tensor, size_per_head: int, rotary_dim: int):
    """Permute only the first rotary_dim elements of each head.

    Used when partial_rotary_factor < 1.0: only the rotary portion needs interleaving for TurboMind's RoPE kernel
    layout.
    """
    assert x.size(-1) > 1
    assert rotary_dim % 2 == 0, f'rotary_dim must be even, got {rotary_dim}'
    assert rotary_dim <= size_per_head, f'rotary_dim ({rotary_dim}) must be <= size_per_head ({size_per_head})'
    output_dims = x.size(-1)
    assert output_dims % size_per_head == 0, (f'output_dims ({output_dims}) must be divisible by '
                                              f'size_per_head ({size_per_head})')
    head_num = output_dims // size_per_head
    orig_shape = x.shape
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = x.view(x.size(0), head_num, size_per_head)
    rotary = x[:, :, :rotary_dim]
    passthrough = x[:, :, rotary_dim:]
    # Interleave rotary part: [2, rotary_dim//2] -> [rotary_dim//2, 2]
    rotary = rotary.view(x.size(0), head_num, 2, rotary_dim // 2).transpose(2, 3).contiguous()
    rotary = rotary.view(x.size(0), head_num, rotary_dim)
    x = torch.cat([rotary, passthrough], dim=-1)
    return x.reshape(orig_shape)


def merge_qkv_v2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tp: int):
    """
        Contract: x.size(-1) is output dims
    """

    def reshape(x):
        return x.view(x.size(0), tp, -1) if q.dim() == 2 else x.view(tp, -1)

    qkv = torch.cat(tuple(map(reshape, (q, k, v))), dim=-1)

    qkv = qkv.view(-1, qkv.size(-1) * tp)
    if q.dim() == 1:
        qkv.squeeze_()

    return qkv


def merge_qkvg_v2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gate: torch.Tensor, tp: int):
    """Merge Q, K, V, and Gate with gate appended after V.

    Layout per tp-shard: [Q | K | V | Gate].
    """

    def reshape(x):
        return x.view(x.size(0), tp, -1) if q.dim() == 2 else x.view(tp, -1)

    qkvg = torch.cat(tuple(map(reshape, (q, k, v, gate))), dim=-1)

    qkvg = qkvg.view(-1, qkvg.size(-1) * tp)
    if q.dim() == 1:
        qkvg.squeeze_()

    return qkvg
```

Note: `SplitSide` is placed here because `TextModelSpec.raw_layer_tensors()` returns `SplitSide` values. The commit layer will import it from here.

- [ ] **Step 3: Add commit helper functions used by assembly**

These helpers are called by `merge_qkv_linear` and `fuse_gdn_in_proj` during assembly:

```python
# -----------------------------------------------------------------------
# Assembly helpers
# -----------------------------------------------------------------------


def _dequant_linear(linear: Linear) -> Linear:
    """Dequantize a quantized Linear to dense when the format provides ``dequant``."""
    fmt = linear.weight_format
    if fmt is None or fmt.dequant is None:
        return linear
    new_tensors = fmt.dequant(linear.tensors)
    return Linear(tensors=new_tensors, weight_format=DENSE_FORMAT, data_format=None)


def _ensure_compatible_formats(linears: dict[str, Linear]) -> dict[str, Linear]:
    """Dequant linears to a common dense format if a fusion group has mixed formats."""
    formats = {name: lin.weight_format.name for name, lin in linears.items()}
    if len(set(formats.values())) <= 1:
        return linears
    return {name: _dequant_linear(lin) for name, lin in linears.items()}


def _block_ops_need_dequant(
    lin: Linear, head_dim: int,
    repeat_kv: bool, attn_output_gate: bool, permute_qk: bool,
) -> bool:
    """Return True if any planned QKV-merge operation crosses block boundaries."""
    wfmt = lin.weight_format
    if wfmt is None or wfmt.block_out is None:
        return False
    block_out = wfmt.block_out
    if (repeat_kv or attn_output_gate) and head_dim % block_out != 0:
        return True
    if permute_qk and block_out % head_dim != 0:
        return True
    return False
```

- [ ] **Step 4: Add `merge_qkv_linear` function**

Copy verbatim from `module.py` lines 410–518:

```python
# -----------------------------------------------------------------------
# QKV merge
# -----------------------------------------------------------------------


def merge_qkv_linear(
    q: Linear,
    k: Linear,
    v: Linear,
    tp: int,
    head_dim: int,
    rope_dim: int,
    permute_qk: bool = True,
    attn_output_gate: bool = False,
    repeat_kv: int = 0,
    kv_head_num: int = 0,
) -> Linear:
    """Merge Q/K/V ``Linear`` bundles into a single interleaved ``w_qkv`` Linear."""
    group = _ensure_compatible_formats({"q": q, "k": k, "v": v})
    q, k, v = group["q"], group["k"], group["v"]

    # Pre-dequantise if any output-dimension transformation is inter-block.
    if any(_block_ops_need_dequant(lin, head_dim, bool(repeat_kv),
                                   attn_output_gate, permute_qk)
           for lin in (q, k, v)):
        import warnings
        _wfmt = q.weight_format or k.weight_format or v.weight_format
        warnings.warn(
            f"QKV merge with format '{_wfmt.name if _wfmt else None}' "
            f"(block_out={_wfmt.block_out if _wfmt else None}) and "
            f"head_dim={head_dim}: inter-block transformation detected; "
            f"dequantising to dense.")
        q, k, v = _dequant_linear(q), _dequant_linear(k), _dequant_linear(v)

    merged_tensors: dict[str, torch.Tensor] = {}
    all_kinds = sorted(set(q.tensors) | set(k.tensors) | set(v.tensors))
    for kind in all_kinds:
        qt = q.tensors.get(kind)
        kt = k.tensors.get(kind)
        vt = v.tensors.get(kind)
        if qt is None or kt is None or vt is None:
            continue

        wfmt = q.weight_format
        block_out = (wfmt.block_out or 0) if wfmt is not None else 0
        is_block_kind = kind in ("scales", "zeros") and block_out > 0

        full_res = not is_block_kind and (qt.size(-1) % head_dim == 0)
        gate = None

        if repeat_kv:
            n = repeat_kv
            kv_heads = kv_head_num // n
            if is_block_kind:
                kt = kt.repeat_interleave(n, dim=-1)
                vt = vt.repeat_interleave(n, dim=-1)
            elif full_res:
                kt = kt.reshape(-1, kv_heads, head_dim).repeat(1, 1, n).reshape(-1, kv_heads * n * head_dim)
                vt = vt.reshape(-1, kv_heads, head_dim).repeat(1, 1, n).reshape(-1, kv_heads * n * head_dim)

        if attn_output_gate and full_res:
            head_num = qt.size(-1) // (head_dim * 2)
            orig_shape = list(qt.shape)
            if qt.dim() == 1:
                qt = qt.unsqueeze(0)
            qt = qt.view(qt.size(0), head_num, 2, head_dim)
            q_real = qt[:, :, 0, :].contiguous().reshape(-1, head_num * head_dim)
            gate = qt[:, :, 1, :].contiguous().reshape(-1, head_num * head_dim)
            if len(orig_shape) == 1:
                q_real = q_real.squeeze(0)
                gate = gate.squeeze(0)
            qt = q_real

        if permute_qk and full_res:
            if rope_dim < head_dim:
                qt = permute_v2_partial(qt, head_dim, rope_dim)
                kt = permute_v2_partial(kt, head_dim, rope_dim)
            else:
                qt = permute_v2(qt, head_dim)
                kt = permute_v2(kt, head_dim)

        if gate is not None:
            merged_tensors[kind] = merge_qkvg_v2(qt, kt, vt, gate, tp)
        else:
            merged_tensors[kind] = merge_qkv_v2(qt, kt, vt, tp)

    return Linear(tensors=merged_tensors, weight_format=q.weight_format,
                  data_format=q.data_format)
```

- [ ] **Step 5: Add GDN fusion functions**

Copy `_GDN_IN_PROJ_KEYS`, `_tp_interleave_tensor`, and `fuse_gdn_in_proj` from `module.py` lines 379–631:

```python
# -----------------------------------------------------------------------
# GDN fusion
# -----------------------------------------------------------------------

_GDN_IN_PROJ_KEYS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")


def _tp_interleave_tensor(t: torch.Tensor, tp: int, d: int) -> torch.Tensor:
    """Reshape last dim as [tp, per_tp] and flatten to interleave by TP rank."""
    shape = list(t.shape)
    new_shape = shape[:d] + [tp, shape[d] // tp] + shape[d + 1:]
    return t.reshape(new_shape)


def fuse_gdn_in_proj(la_linears: dict[str, Linear], tp: int,
                     qkv_split: tuple[int, int, int] | None = None) -> dict[str, Linear]:
    """Fuse GDN input projections into ``in_proj_all`` with TP interleaving."""
    result = dict(la_linears)
    components: list[Linear] = []
    for key in _GDN_IN_PROJ_KEYS:
        lin = result.pop(key, None)
        if lin is not None:
            components.append(lin)
    if not components:
        return result

    group = {f"c{i}": c for i, c in enumerate(components)}
    group = _ensure_compatible_formats(group)
    components = [group[f"c{i}"] for i in range(len(group))]

    first = components[0]
    if tp <= 1:
        result["in_proj_all"] = Linear.concat_out_dim(components)
        return result

    if qkv_split is not None:
        q_dim, k_dim, v_dim = qkv_split
        qkv_lin = components[0]
        rest = components[1:]

        fused_tensors: dict[str, torch.Tensor] = {}
        for kind in first.tensors:
            qkv_t = qkv_lin.tensors.get(kind)
            if qkv_t is None:
                continue
            d = qkv_t.dim() - 1
            if qkv_t.dim() <= 1:
                parts = [qkv_t[..., :q_dim], qkv_t[..., q_dim:q_dim + k_dim],
                         qkv_t[..., q_dim + k_dim:]]
                rest_ts = [lin.tensors.get(kind) for lin in rest
                           if lin.tensors.get(kind) is not None]
                fused_tensors[kind] = torch.cat(parts + rest_ts, dim=0)
                continue

            q_t = qkv_t[..., :q_dim]
            k_t = qkv_t[..., q_dim:q_dim + k_dim]
            v_t = qkv_t[..., q_dim + k_dim:]

            interleaved = [
                _tp_interleave_tensor(q_t, tp, d),
                _tp_interleave_tensor(k_t, tp, d),
                _tp_interleave_tensor(v_t, tp, d),
            ]
            for lin in rest:
                t = lin.tensors.get(kind)
                if t is not None:
                    interleaved.append(_tp_interleave_tensor(t, tp, d))

            fused = torch.cat(interleaved, dim=d + 1)
            shape = list(fused.shape)
            final = shape[:d] + [shape[d] * shape[d + 1]] + shape[d + 2:]
            fused_tensors[kind] = fused.reshape(final)

        result["in_proj_all"] = Linear(tensors=fused_tensors, weight_format=first.weight_format,
                                       data_format=first.data_format)
        return result

    fused_tensors: dict[str, torch.Tensor] = {}
    for kind in first.tensors:
        tensors = [lin.tensors[kind] for lin in components]
        t0 = tensors[0]
        d = t0.dim() - 1
        if t0.dim() <= 1:
            fused_tensors[kind] = torch.cat(tensors, dim=0)
            continue
        reshaped = []
        for t in tensors:
            reshaped.append(_tp_interleave_tensor(t, tp, d))
        fused = torch.cat(reshaped, dim=d + 1)
        shape = list(fused.shape)
        final = shape[:d] + [shape[d] * shape[d + 1]] + shape[d + 2:]
        fused_tensors[kind] = fused.reshape(final)

    result["in_proj_all"] = Linear(tensors=fused_tensors, weight_format=first.weight_format,
                                   data_format=first.data_format)
    return result
```

- [ ] **Step 6: Add `TextModelSpec` ABC**

Copy `TextModelSpec` from `module.py` lines 109–328. The class references `merge_qkv_linear`, `fuse_gdn_in_proj`, `_GDN_IN_PROJ_KEYS`, `permute_v2`, `permute_v2_partial` — all now in the same file. It also references `SplitSide` (same file) and `build_linear` (from `kind_map`). No cross-file dependencies.

Copy the class verbatim, preserving all methods and docstrings.

- [ ] **Step 7: Verify `spec.py` compiles**

Run: `cd /data/lmdeploy-modeling && PYTHONPATH=lmdeploy:build/lib python -c "from lmdeploy.turbomind.deploy.spec import TextModelSpec, SplitSide, merge_qkv_linear, fuse_gdn_in_proj"`

Expected: No import errors.

- [ ] **Step 8: Commit**

```bash
git add lmdeploy/turbomind/deploy/spec.py
git commit -m "refactor(spec): extract spec.py (Read & Assemble layer) from module.py"
```

---

### Task 2: Create `transforms.py` — Transform layer

**Files:**
- Create: `lmdeploy/turbomind/deploy/transforms.py`
- Reference: `lmdeploy/turbomind/deploy/module.py` (read-only source)

This file operates on `Linear` objects only — no C++ interaction. It handles TP shard extraction and FFN w1/w3 fusion decisions.

- [ ] **Step 1: Create `transforms.py` with imports and helper functions**

```python
# Copyright (c) OpenMMLab. All rights reserved.
"""Transform layer: weight transformations on Linear objects before commit.

Operates on Linear bundles only — no C++ module interaction.
"""
from __future__ import annotations

from .linear import Linear, chunk_linears, interleave_linears


def _should_fuse_silu(w1_linear: Linear, act_type: str, is_moe: bool = False) -> bool:
    """Determine if fused SiLU (interleave) should be used for w1+w3 fusion.

    Gold standard condition (from GEMM kernel constraints — trust it):
        act_type == SiLU && (int4 || mxfp4 || fp8 || moe) && !(fp8 && SM90)
    """
    if act_type not in ('', 'silu', 'SiLU'):
        return False

    weight = w1_linear.tensors.get("weight")
    if weight is None:
        weight = w1_linear.tensors.get("qweight")
    is_quantized = weight is not None and weight.element_size() < 2
    if not is_quantized and not is_moe:
        return False

    fmt = w1_linear.weight_format
    if fmt is not None and fmt.name == "fp8":
        if __import__('torch').cuda.is_available():
            cap = __import__('torch').cuda.get_device_capability()
            if cap == (9, 0):
                return False

    return True


def _can_fuse_w1w3(w1: Linear, tp: int) -> bool:
    """Check whether w1+w3 fusion is safe for the given TP."""
    if tp <= 1:
        return True
    fmt = w1.weight_format
    if fmt is None or fmt.block_out is None:
        return True
    w = w1.tensors.get("weight")
    if w is None:
        w = w1.tensors.get("qweight")
    if w is None:
        return True
    return (w.size(-1) // tp) % fmt.block_out == 0


def _shard_linear_for_tp(linear: Linear, tp: int, rank: int) -> Linear:
    """Extract the TP shard for *rank*, handling block-scale alignment."""
    fmt = linear.weight_format
    if fmt is None or fmt.block_out is None:
        return linear.split_out_dim(tp)[rank]

    scales = linear.tensors.get("scales")
    if scales is None or scales.size(-1) % tp == 0:
        return linear.split_out_dim(tp)[rank]

    weight = linear.tensors.get("weight")
    if weight is None:
        weight = linear.tensors.get("qweight")
    if weight is None:
        return linear.split_out_dim(tp)[rank]

    N = weight.size(-1)
    block_out = fmt.block_out
    W = N // tp

    shard_tensors: dict[str, __import__('torch').Tensor] = {}
    for kind, t in linear.tensors.items():
        if kind in ("scales", "zeros") and t.dim() >= 2:
            start_block = (rank * W) // block_out
            end_block = -(-(rank + 1) * W // block_out)
            shard_tensors[kind] = t[:, start_block:end_block].contiguous()
        elif t.dim() >= 2:
            split_size = t.size(-1) // tp
            shard_tensors[kind] = t[..., rank * split_size:(rank + 1) * split_size].contiguous()
        else:
            split_size = t.size(0) // tp
            shard_tensors[kind] = t[rank * split_size:(rank + 1) * split_size].contiguous()

    return Linear(tensors=shard_tensors, weight_format=fmt,
                  data_format=linear.data_format)
```

Note: The `torch` references in `_shard_linear_for_tp` use inline `__import__` to avoid a top-level `import torch` (matching the original pattern). A cleaner approach: add `import torch` at the top since this file has no import-cycle risk.

- [ ] **Step 2: Add `fuse_ffn_linears` — the new pure transform function**

This is the key new abstraction extracted from `_fuse_and_commit_ffn`. It does TP sharding + w1/w3 interleave/chunk on `Linear` objects only:

```python
def fuse_ffn_linears(
    w1: Linear,
    w3: Linear,
    tp: int,
    rank: int,
    act_type: str,
    is_moe: bool = False,
) -> tuple[Linear | None, Linear | None, Linear | None, bool]:
    """TP-shard and optionally fuse w1/w3 for FFN.

    Returns (fused_w1w3_or_none, w1_shard_or_none, w3_shard_or_none, fused_silu).
    When fusion is possible, fused_w1w3 is set and shards are None.
    When block-scale boundaries prevent fusion, shards are set individually.
    """
    fused_silu = _should_fuse_silu(w1, act_type, is_moe)
    can_fuse = _can_fuse_w1w3(w1, tp)

    # Split by TP rank (handles block-scale alignment for FP8 etc.)
    if tp > 1:
        w1_shard = _shard_linear_for_tp(w1, tp, rank)
        w3_shard = _shard_linear_for_tp(w3, tp, rank)
    else:
        w1_shard = w1
        w3_shard = w3

    if can_fuse:
        if fused_silu:
            w1w3 = interleave_linears(w1_shard, w3_shard)
        else:
            w1w3 = chunk_linears(w1_shard, w3_shard)
        return (w1w3, None, None, fused_silu)
    else:
        return (None, w1_shard, w3_shard, fused_silu)
```

- [ ] **Step 3: Verify `transforms.py` compiles**

Run: `cd /data/lmdeploy-modeling && PYTHONPATH=lmdeploy:build/lib python -c "from lmdeploy.turbomind.deploy.transforms import fuse_ffn_linears, _shard_linear_for_tp"`

Expected: No import errors.

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/transforms.py
git commit -m "refactor(transforms): extract transforms.py (Transform layer) from module.py"
```

---

### Task 3: Create `commit.py` — Shard & Commit layer

**Files:**
- Create: `lmdeploy/turbomind/deploy/commit.py`
- Reference: `lmdeploy/turbomind/deploy/module.py` (read-only source)

This file handles the final step: allocating C++ tensors and copying data to GPU. It owns TP split rules and the commit functions.

- [ ] **Step 1: Create `commit.py` with SplitSide, dtype helpers, and TP rules**

```python
# Copyright (c) OpenMMLab. All rights reserved.
"""Shard & Commit layer: allocate C++ tensors and copy weight data to GPU."""
from __future__ import annotations

import torch

from .linear import Linear
from .spec import SplitSide
from .transforms import fuse_ffn_linears

_SPLIT_SIDE_TO_DIM: dict[SplitSide, int] = {SplitSide.OUTPUT: -1, SplitSide.INPUT: 0}


def _torch_dtype_to_cpp(dtype: torch.dtype):
    """Convert a torch dtype to the C++ ``DataType`` enum, or ``None``."""
    try:
        import _turbomind as _tm
    except ImportError:
        return None
    _MAP = {
        torch.float32:  _tm.DataType.TYPE_FP32,
        torch.float16:  _tm.DataType.TYPE_FP16,
        torch.bfloat16: _tm.DataType.TYPE_BF16,
        torch.int32:    _tm.DataType.TYPE_INT32,
        torch.int64:    _tm.DataType.TYPE_INT64,
        torch.int8:     _tm.DataType.TYPE_INT8,
        torch.uint8:    _tm.DataType.TYPE_UINT8,
    }
    return _MAP.get(dtype)


def _cast_shard_for_tm(shard: torch.Tensor, tm_tensor) -> torch.Tensor:
    """Cast *shard* dtype to match *tm_tensor*'s C++ dtype when needed."""
    try:
        import _turbomind as _tm
    except ImportError:
        return shard

    if tm_tensor.type == _tm.DataType.TYPE_FP32 and shard.dtype in (torch.float16, torch.bfloat16):
        return shard.float()
    if tm_tensor.type == _tm.DataType.TYPE_FP16 and shard.dtype != torch.float16:
        return shard.half()
    if tm_tensor.type == _tm.DataType.TYPE_BF16 and shard.dtype != torch.bfloat16:
        return shard.to(torch.bfloat16)
    return shard


def _infer_cpp_linear_dtype(linear: Linear):
    """Determine C++ DataType and group_size from ``Linear.weight_format``."""
    try:
        import _turbomind as _tm
    except ImportError:
        return None, 0

    fmt = linear.weight_format
    if fmt is not None and fmt.cpp_dtype_name is not None:
        cpp_dtype = getattr(_tm.DataType, fmt.cpp_dtype_name, None)
        if cpp_dtype is not None:
            return cpp_dtype, fmt.block_in or 0

    weight = linear.tensors.get("weight")
    if weight is not None:
        if weight.dtype == torch.bfloat16:
            return _tm.DataType.TYPE_BF16, 0
        if weight.dtype == torch.float16:
            return _tm.DataType.TYPE_FP16, 0
    return None, 0


def _infer_compute_dtype(linear: Linear):
    """Get the model's compute dtype from a Linear's tensors."""
    try:
        import _turbomind as _tm
    except ImportError:
        return None
    _MAP = {
        torch.bfloat16: _tm.DataType.TYPE_BF16,
        torch.float16:  _tm.DataType.TYPE_FP16,
        torch.float32:  _tm.DataType.TYPE_FP32,
    }
    w = linear.tensors.get('weight')
    if w is not None:
        d = _MAP.get(w.dtype)
        if d is not None:
            return d
        _fp8_dtypes = {torch.uint8}
        for _attr in ('float8_e4m3fn', 'float8_e5m2fn'):
            _dt = getattr(torch, _attr, None)
            if _dt is not None:
                _fp8_dtypes.add(_dt)
        if w.dtype in _fp8_dtypes:
            return _tm.DataType.TYPE_BF16
    for key in ('scales', 'bias'):
        t = linear.tensors.get(key)
        if t is not None:
            d = _MAP.get(t.dtype)
            if d is not None:
                return d
    return None


# -----------------------------------------------------------------------
# TP split rules
# -----------------------------------------------------------------------

_ATTN_TP_RULES: dict[str, dict] = {
    "w_qkv":     dict(split_side=SplitSide.OUTPUT),
    "wo":        dict(split_side=SplitSide.INPUT),
    "q_proj":    dict(split_side=SplitSide.OUTPUT),
    "q_b_proj":  dict(split_side=SplitSide.OUTPUT),
    "kv_b_proj": dict(split_side=SplitSide.OUTPUT),
}

_FFN_TP_RULES: dict[str, dict] = {
    "w1": dict(split_side=SplitSide.OUTPUT),
    "w3": dict(split_side=SplitSide.OUTPUT),
    "w2": dict(split_side=SplitSide.INPUT),
}

_LINEAR_ATTN_TP_RULES: dict[str, dict] = {
    "in_proj_qkv": dict(split_side=SplitSide.OUTPUT),
    "in_proj_z":   dict(split_side=SplitSide.OUTPUT),
    "in_proj_b":   dict(split_side=SplitSide.OUTPUT),
    "in_proj_a":   dict(split_side=SplitSide.OUTPUT),
    "in_proj_all": dict(split_side=SplitSide.OUTPUT),
    "out_proj":    dict(split_side=SplitSide.INPUT),
}
```

- [ ] **Step 2: Add `_commit_tensors`, `commit_linear`, `commit_tensor`**

Copy the core commit functions from `module.py` lines 864–1041. These are the functions that allocate C++ tensors and copy data. They reference `SplitSide` (imported from `spec.py`), `_SPLIT_SIDE_TO_DIM` (defined here), and dtype helpers (defined here).

```python
# -----------------------------------------------------------------------
# Core commit functions
# -----------------------------------------------------------------------


def _commit_tensors(handle, linear: Linear, cpp_dtype, group_size: int,
                    split_side: SplitSide | None, split_num: int, rank: int):
    """Commit tensor data from a ``Linear`` to a pre-created C++ LinearWeight handle."""
    split_dim = _SPLIT_SIDE_TO_DIM.get(split_side) if split_side else None

    packer = linear.weight_format.packer if linear.weight_format else None

    def _kind_order(item):
        k, _ = item
        if k in ("weight", "qweight"):
            return (0, k)
        return (1, k)

    for kind, tensor in sorted(linear.tensors.items(), key=_kind_order):
        if packer is not None:
            tensor = packer(tensor, kind)

        tensor_split_dim = split_dim
        if kind == "bias" and split_side == SplitSide.INPUT:
            tensor_split_dim = None

        if tensor_split_dim is not None and split_num > 1:
            split_size = tensor.shape[tensor_split_dim] // split_num
            shard = tensor.split(split_size, dim=tensor_split_dim)[rank]
        else:
            shard = tensor

        shard = shard.cuda().contiguous()

        dst = handle.alloc(kind, cpp_dtype, group_size)
        if dst:
            shard = _cast_shard_for_tm(shard, dst)
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


def commit_linear(module, linear: Linear, name: str,
                         split_side: SplitSide | None = None,
                         split_num: int = 1, rank: int = 0,
                         model_dtype=None):
    """Commit a ``Linear`` bundle to a C++ ``Module`` handle for a specific TP rank."""
    cpp_dtype, group_size = _infer_cpp_linear_dtype(linear)
    if group_size == 0:
        group_size = max(1, 128)

    if linear.data_format is None and linear.weight_format is not None:
        linear = Linear(tensors=linear.tensors,
                        weight_format=linear.weight_format,
                        data_format=linear.weight_format.to_data_format(
                            cpp_dtype.value if cpp_dtype else 0,
                            group_size))

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


def commit_tensor(module, tensor: torch.Tensor | None, name: str,
                         split_side: SplitSide | None = None,
                         split_num: int = 1, rank: int = 0):
    """Commit a raw tensor to a C++ ``Module`` handle for a specific TP rank."""
    if tensor is None:
        return

    split_dim = _SPLIT_SIDE_TO_DIM.get(split_side) if split_side else None

    if split_dim is not None and split_num > 1:
        split_size = tensor.shape[split_dim] // split_num
        shard = tensor.split(split_size, dim=split_dim)[rank]
    else:
        shard = tensor

    shard = shard.cuda().contiguous()
    cpp_dtype = _torch_dtype_to_cpp(shard.dtype)
    if cpp_dtype is None:
        return
    dst = module.alloc(name, cpp_dtype, 0)
    if dst:
        shard = _cast_shard_for_tm(shard, dst)
        dst.copy_from(shard)
```

- [ ] **Step 3: Add `commit_ffn` — the new orchestration function**

This replaces `_fuse_and_commit_ffn`. It calls `fuse_ffn_linears` from the transform layer, then handles C++ allocation:

```python
def commit_ffn(ffn_mod, w1: Linear, w3: Linear, w2: Linear | None,
               tp: int, rank: int, act_type: str, is_moe: bool = False,
               model_dtype=None):
    """Preprocess, split, fuse (interleave or chunk) and commit FFN weights."""
    fused, w1_shard, w3_shard, fused_silu = fuse_ffn_linears(
        w1, w3, tp, rank, act_type, is_moe)

    if fused is not None:
        commit_linear(ffn_mod, fused, "w1w3",
                           model_dtype=model_dtype)
        ffn_mod.set_fused_silu(fused_silu)
    else:
        for name, shard in (("w1", w1_shard), ("w3", w3_shard)):
            rule = _FFN_TP_RULES.get(name, {})
            commit_linear(ffn_mod, shard, name,
                                 split_num=1, rank=0, model_dtype=model_dtype,
                                 **rule)

    if w2 is not None:
        rule = _FFN_TP_RULES.get("w2", {})
        tp2 = tp if "split_side" in rule else 1
        commit_linear(ffn_mod, w2, "w2",
                             split_num=tp2, rank=rank, model_dtype=model_dtype,
                             **rule)


# Backward-compatible alias
_fuse_and_commit_ffn = commit_ffn
```

- [ ] **Step 4: Verify `commit.py` compiles**

Run: `cd /data/lmdeploy-modeling && PYTHONPATH=lmdeploy:build/lib python -c "from lmdeploy.turbomind.deploy.commit import SplitSide, commit_linear, commit_tensor, commit_ffn, _fuse_and_commit_ffn, _ATTN_TP_RULES, _FFN_TP_RULES, _LINEAR_ATTN_TP_RULES"`

Expected: No import errors.

- [ ] **Step 5: Commit**

```bash
git add lmdeploy/turbomind/deploy/commit.py
git commit -m "refactor(commit): extract commit.py (Shard & Commit layer) from module.py"
```

---

### Task 4: Replace `module.py` with backward-compatible facade

**Files:**
- Modify: `lmdeploy/turbomind/deploy/module.py` (replace contents)

This is the critical step. The entire 1072-line file is replaced with a thin re-export facade. All external imports continue working.

- [ ] **Step 1: Write the facade**

Replace the entire contents of `module.py` with:

```python
# Copyright (c) OpenMMLab. All rights reserved.
"""Backward-compatible facade re-exporting from the layered split.

All names previously available from this module are still importable.
New code should import directly from the specific layer:
  - ``spec.py``      — Read & Assemble (TextModelSpec, QKV/GDN merge)
  - ``transforms.py`` — Transform (FFN fusion, TP shard extraction)
  - ``commit.py``     — Shard & Commit (alloc, copy, TP rules)
"""
from __future__ import annotations

# Read & Assemble layer
from .spec import (
    SplitSide,
    TextModelSpec,
    _GDN_IN_PROJ_KEYS,
    _block_ops_need_dequant,
    _dequant_linear,
    _ensure_compatible_formats,
    _tp_interleave_tensor,
    fuse_gdn_in_proj,
    merge_qkvg_v2,
    merge_qkv_linear,
    merge_qkv_v2,
    permute_v2,
    permute_v2_partial,
)

# Transform layer
from .transforms import (
    _can_fuse_w1w3,
    _shard_linear_for_tp,
    _should_fuse_silu,
    fuse_ffn_linears,
)

# Shard & Commit layer
from .commit import (
    _ATTN_TP_RULES,
    _FFN_TP_RULES,
    _LINEAR_ATTN_TP_RULES,
    _SPLIT_SIDE_TO_DIM,
    _commit_tensors,
    _fuse_and_commit_ffn,
    _cast_shard_for_tm,
    _infer_compute_dtype,
    _infer_cpp_linear_dtype,
    _torch_dtype_to_cpp,
    commit_ffn,
    commit_linear,
    commit_tensor,
)
```

- [ ] **Step 2: Verify the facade re-exports correctly**

Run: `cd /data/lmdeploy-modeling && PYTHONPATH=lmdeploy:build/lib python -c "from lmdeploy.turbomind.deploy.module import TextModelSpec, SplitSide, commit_linear, commit_tensor, _fuse_and_commit_ffn, _ATTN_TP_RULES, _FFN_TP_RULES, _LINEAR_ATTN_TP_RULES; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Verify external importers still work**

Run: `cd /data/lmdeploy-modeling && PYTHONPATH=lmdeploy:build/lib python -c "from lmdeploy.turbomind.deploy.source_model.qwen3_spec import Qwen3Spec; from lmdeploy.turbomind.deploy.source_model.gpt_oss_spec import GptOssSpec; from lmdeploy.turbomind.deploy.text_model_loader import TextModelLoader; print('OK')"`

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add lmdeploy/turbomind/deploy/module.py
git commit -m "refactor(module): replace module.py with backward-compatible facade"
```

---

### Task 5: Test with a model — TP=1

**Files:** No code changes. Smoke test only.

- [ ] **Step 1: Check GPU is free**

Run: Use `get_gpu_usage` MCP tool. Identify a free GPU (0% util, near 0 MiB used).

- [ ] **Step 2: Run model test with TP=1**

Use the turbomind-tester agent to test `Qwen/Qwen3-4B` with TP=1 on the free GPU.

Prompt: "Write a short paragraph about artificial intelligence."

Expected: Meaningful response about AI, not gibberish. At least 128 tokens.

- [ ] **Step 3: Verify response quality**

The model must respond with coherent, relevant text. Gibberish indicates a weight loading bug introduced by the refactoring.

---

### Task 6: Test with a model — TP=2

**Files:** No code changes. Smoke test only.

- [ ] **Step 1: Check two GPUs are free**

Run: Use `get_gpu_usage` MCP tool. Identify two free GPUs.

- [ ] **Step 2: Run model test with TP=2**

Use the turbomind-tester agent to test `Qwen/Qwen3-4B` with TP=2 on two free GPUs.

Prompt: "Explain the difference between supervised and unsupervised learning in machine learning."

Expected: Meaningful response, at least 128 tokens.

- [ ] **Step 3: Verify response quality**

The model must respond with coherent, relevant text. TP=2 exercises the TP sharding code path, which is the most refactored part.

---

### Task 7: Test with a quantized model

**Files:** No code changes. Smoke test only.

Quantized models exercise the format/quant path through `_commit_tensors` and `_shard_linear_for_tp`.

- [ ] **Step 1: Check GPU is free**

- [ ] **Step 2: Run model test with a quantized model**

Use the turbomind-tester agent to test `Qwen/Qwen3-4B-AWQ` with TP=1.

Prompt: "What are the main challenges in training large language models?"

Expected: Meaningful response, at least 128 tokens.

- [ ] **Step 3: Verify response quality**

---

## Self-Review

### Spec coverage

| Spec requirement | Task |
|---|---|
| Create `spec.py` with TextModelSpec, QKV merge, GDN fusion | Task 1 |
| Create `transforms.py` with FFN fusion, TP shard extraction | Task 2 |
| Create `commit.py` with commit functions, TP rules | Task 3 |
| New `fuse_ffn_linears` pure function | Task 2 Step 2 |
| New `commit_ffn` orchestration function | Task 3 Step 3 |
| Backward-compat alias `_fuse_and_commit_ffn` | Task 3 Step 3 |
| Replace `module.py` with facade | Task 4 |
| All existing models work | Tasks 5–7 |
| Test TP=1 and TP=2 | Tasks 5–6 |
| No C++ changes | (none needed) |

### Placeholder scan

No TBD, TODO, or placeholder steps. All steps contain complete code.

### Type consistency

- `SplitSide` is defined in `spec.py`, imported by `commit.py` and `transforms.py` (via `spec.py`)
- `fuse_ffn_linears` returns `tuple[Linear | None, Linear | None, Linear | None, bool]` — consistent between Task 2 (definition) and Task 3 (caller)
- `_fuse_and_commit_ffn = commit_ffn` — signature matches the old `_fuse_and_commit_ffn` exactly
- All TP rules dicts use `SplitSide.OUTPUT` / `SplitSide.INPUT` — consistent with the enum location in `spec.py`
