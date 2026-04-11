# Copyright (c) OpenMMLab. All rights reserved.
"""Linear weight bundle and composable dimension operations.

Two weight types flow through the TurboMind weight loading pipeline:

- ``Linear`` -- a bundle of tensors for a single linear layer (weight +
  optional scales, zeros, bias).
- Raw ``torch.Tensor`` -- everything else (norms, embeddings, scalars).

**Tensor functions** accept an explicit ``dim`` argument and operate on a
single ``torch.Tensor``.

**Linear methods** operate on axis 0 (input) and axis -1 (output), which
is the fixed TM layout contract.  ``split``, ``concat`` are safe as
``Linear`` methods -- they work correctly across all component tensors
regardless of quantization-induced dimension scaling.  ``permute`` and
``pad`` remain Tensor-level functions because quantized components (e.g.
FP8 block scales) have reduced dimensions that don't carry per-element
structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor

import _turbomind as _tm

if TYPE_CHECKING:
    from .kind_map import WeightFormat


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _norm(dim: int, ndim: int) -> int:
    """Normalise a possibly-negative dimension index."""
    return dim if dim >= 0 else dim + ndim


def _has_input_dim(t: Tensor) -> bool:
    """1-D tensors (e.g. bias) have no input dimension."""
    return t.dim() > 1


def _pad_1d(t: Tensor, dim: int, target: int) -> Tensor:
    """Pad one dimension of *t* to *target* size with zeros."""
    deficit = target - t.size(dim)
    if deficit <= 0:
        return t
    pad = [0] * (2 * t.dim())
    # F.pad expects pairs in reverse dim order
    pad[2 * (t.dim() - 1 - dim) + 1] = deficit
    return torch.nn.functional.pad(t, pad, "constant", 0)


def _permute_along(t: Tensor, dim: int, shape: list[int], order: list[int]) -> Tensor:
    """View *dim* as *shape*, permute those sub-dims by *order*, flatten back."""
    pre = list(t.shape[:dim])
    post = list(t.shape[dim + 1:])
    t = t.reshape(pre + list(shape) + post)
    n = len(shape)
    perm = list(range(dim)) + [dim + o for o in order] + list(range(dim + n, t.dim()))
    t = t.permute(perm).contiguous()
    new_size = 1
    for s in shape:
        new_size *= s
    return t.reshape(pre + [new_size] + post)


# ---------------------------------------------------------------------------
# Tensor functions
# ---------------------------------------------------------------------------


def split_out_dim(t: Tensor, num: int, dim: int) -> list[Tensor]:
    """Split *t* along *dim* into *num* equal parts."""
    d = _norm(dim, t.dim())
    return list(t.split(t.size(d) // num, dim=d))


def concat_out_dim(ts: list[Tensor], dim: int) -> Tensor:
    """Concatenate tensors along *dim*."""
    return torch.cat(ts, dim=_norm(dim, ts[0].dim()))


def permute_out_dim(t: Tensor, shape: list[int], order: list[int], dim: int) -> Tensor:
    """View *dim* as *shape*, permute sub-dims by *order*, flatten back."""
    return _permute_along(t, _norm(dim, t.dim()), shape, order)


def permute_in_dim(t: Tensor, shape: list[int], order: list[int], dim: int) -> Tensor:
    """View *dim* as *shape*, permute sub-dims by *order*, flatten back."""
    return _permute_along(t, _norm(dim, t.dim()), shape, order)


def pad_out_dim(t: Tensor, target: int, dim: int) -> Tensor:
    """Pad *dim* to *target* size with zeros."""
    return _pad_1d(t, _norm(dim, t.dim()), target)


def pad_in_dim(t: Tensor, target: int, dim: int) -> Tensor:
    """Pad *dim* to *target* size with zeros."""
    return _pad_1d(t, _norm(dim, t.dim()), target)


def transpose(t: Tensor) -> Tensor:
    """Swap dims 0 and 1."""
    return t.t()


# ---------------------------------------------------------------------------
# Linear dataclass with methods
# ---------------------------------------------------------------------------


@dataclass
class Linear:
    """Bundle of tensors for a single linear layer.

    ``tensors`` maps a closed-set TM weight kind (e.g. ``"weight"``,
    ``"scales"``, ``"zeros"``, ``"bias"``, ``"qweight"``) to the actual
    tensor.

    **Layout contract**: all ``Linear`` objects are in TM layout with
    axis 0 as the input dimension and axis -1 as the output dimension.
    ``commit_linear`` assumes this layout and does not re-transpose.
    1-D tensors (e.g. bias) only have an output dimension (axis 0).
    """

    tensors: dict[str, Tensor]
    weight_format: WeightFormat | None = field(default=None, compare=False, repr=False)
    data_format: _tm.DataFormat | None = field(default=None, compare=False, repr=False)

    def split_out_dim(self, num: int) -> list[Linear]:
        """Split along output dim into *num* equal parts."""
        buckets: list[dict[str, Tensor]] = [{} for _ in range(num)]
        for kind, t in self.tensors.items():
            for i, part in enumerate(split_out_dim(t, num, t.dim() - 1)):
                buckets[i][kind] = part
        return [Linear(tensors=b, weight_format=self.weight_format,
                       data_format=self.data_format) for b in buckets]

    def split_in_dim(self, num: int) -> list[Linear]:
        """Split along input dim into *num* equal parts."""
        buckets: list[dict[str, Tensor]] = [{} for _ in range(num)]
        for kind, t in self.tensors.items():
            if not _has_input_dim(t):
                for i in range(num):
                    buckets[i][kind] = t
                continue
            for i, part in enumerate(split_out_dim(t, num, 0)):
                buckets[i][kind] = part
        return [Linear(tensors=b, weight_format=self.weight_format,
                       data_format=self.data_format) for b in buckets]

    @classmethod
    def concat_out_dim(cls, xs: list[Linear]) -> Linear:
        """Concatenate along output dim."""
        first = xs[0]
        result: dict[str, Tensor] = {}
        for kind in first.tensors:
            t = first.tensors[kind]
            result[kind] = torch.cat([x.tensors[kind] for x in xs], dim=t.dim() - 1)
        fmts = {x.weight_format for x in xs}
        wfmt = next(iter(fmts)) if len(fmts) == 1 else None
        dfmts = {x.data_format for x in xs}
        dfmt = next(iter(dfmts)) if len(dfmts) == 1 else None
        return Linear(tensors=result, weight_format=wfmt, data_format=dfmt)

    @classmethod
    def concat_in_dim(cls, xs: list[Linear]) -> Linear:
        """Concatenate along input dim."""
        first = xs[0]
        result: dict[str, Tensor] = {}
        for kind in first.tensors:
            t0 = first.tensors[kind]
            if not _has_input_dim(t0):
                result[kind] = t0
                continue
            result[kind] = torch.cat([x.tensors[kind] for x in xs], dim=0)
        fmts = {x.weight_format for x in xs}
        wfmt = next(iter(fmts)) if len(fmts) == 1 else None
        dfmts = {x.data_format for x in xs}
        dfmt = next(iter(dfmts)) if len(dfmts) == 1 else None
        return Linear(tensors=result, weight_format=wfmt, data_format=dfmt)


# ---------------------------------------------------------------------------
# Fusion helpers (w1 + w3 → w1w3)
# ---------------------------------------------------------------------------


def preprocess_linear(linear: Linear) -> Linear:
    """Expand FP8 block scales to group scales (blockscale -> groupscale)."""
    fmt = linear.data_format
    if fmt is not None and fmt.is_quantized() and fmt.scales.present():
        scales = linear.tensors.get("scales")
        if scales is not None and scales.dim() == 2:
            block_size = fmt.block_sizes[-1] if len(fmt.block_sizes) > 1 else 1
            if block_size > 1:
                new_scales = scales.repeat_interleave(block_size, dim=-1)
                new_tensors = dict(linear.tensors)
                new_tensors["scales"] = new_scales
                return Linear(tensors=new_tensors,
                              weight_format=linear.weight_format,
                              data_format=linear.data_format)
    return linear


def interleave_linears(w1: Linear, w3: Linear) -> Linear:
    """Interleave w1 and w3 along the output dim for fused SiLU epilogue.

    Output layout: ``[w1_out0, w3_out0, w1_out1, w3_out1, ...]``.
    """
    fused: dict[str, Tensor] = {}
    for kind in w1.tensors:
        t1 = w1.tensors[kind]
        t3 = w3.tensors[kind]
        if _has_input_dim(t1):
            # 2-D: interleave along last (output) dim
            fused[kind] = torch.stack([t1, t3], dim=-1).reshape(t1.shape[:-1] + (-1,))
        else:
            # 1-D (bias): interleave along the only dim
            fused[kind] = torch.stack([t1, t3], dim=-1).reshape(-1)
    fused[kind] = fused[kind].contiguous()
    return Linear(tensors={k: v.contiguous() for k, v in fused.items()},
                  weight_format=w1.weight_format,
                  data_format=w1.data_format)


def chunk_linears(w1: Linear, w3: Linear, tp: int = 1) -> Linear:
    """Concatenate w1 and w3 along the output dim (chunk layout).

    When ``tp > 1``, the result is TP-interleaved so that a naive
    output-dim split gives each rank ``[w1_shard | w3_shard]``.
    This matches the pattern used by ``merge_qkv_v2`` and
    ``fuse_gdn_in_proj``.
    """
    fused: dict[str, Tensor] = {}
    for kind in w1.tensors:
        t1 = w1.tensors[kind]
        t3 = w3.tensors[kind]
        if tp <= 1:
            # No TP: simple concatenation
            dim = -1 if _has_input_dim(t1) else 0
            fused[kind] = torch.cat([t1, t3], dim=dim)
        else:
            # TP-aware: reshape [K, N] -> [K, tp, N/tp], cat on inner dim,
            # then flatten back to [K, 2*N].
            d = t1.dim() - 1  # output dim (last)
            shape = list(t1.shape)
            r1 = t1.reshape(shape[:d] + [tp, shape[d] // tp])
            r3 = t3.reshape(shape[:d] + [tp, shape[d] // tp])
            combined = torch.cat([r1, r3], dim=d + 1)
            c_shape = list(combined.shape)
            fused[kind] = combined.reshape(
                c_shape[:d] + [c_shape[d] * c_shape[d + 1]])
    return Linear(tensors={k: v.contiguous() for k, v in fused.items()},
                  weight_format=w1.weight_format,
                  data_format=w1.data_format)
