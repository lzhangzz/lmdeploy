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
