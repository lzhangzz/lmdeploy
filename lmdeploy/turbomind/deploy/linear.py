# Copyright (c) OpenMMLab. All rights reserved.
"""Linear weight bundle and composable dimension operations.

Two weight types flow through the TurboMind weight loading pipeline:

- ``Linear`` -- a bundle of tensors for a single linear layer (weight +
  optional scales, zeros, bias) with input_dim / output_dim tags.
- Raw ``torch.Tensor`` -- everything else (norms, embeddings, scalars).

Every dimension operation in this module is overloaded to accept either type.
``Linear`` ops use the bundle's dim tags and apply to all components;
raw ``Tensor`` ops take an explicit ``dim`` argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import overload

import torch
from torch import Tensor


@dataclass
class Linear:
    """Bundle of tensors for a single linear layer.

    ``tensors`` maps a closed-set TM weight kind (e.g. ``"weight"``,
    ``"scales"``, ``"zeros"``, ``"bias"``, ``"qweight"``) to the actual
    tensor.  ``input_dim`` and ``output_dim`` are dimension indices that
    are consistent across all component tensors (1-D tensors like bias
    only use ``output_dim``).
    """

    tensors: dict[str, Tensor]
    input_dim: int = 0
    output_dim: int = -1


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
# split_out_dim
# ---------------------------------------------------------------------------


@overload
def split_out_dim(x: Linear, num: int) -> list[Linear]: ...


@overload
def split_out_dim(x: Tensor, num: int, dim: int) -> list[Tensor]: ...


def split_out_dim(x, num, dim=None):
    """Split along output dim into *num* equal parts."""
    if isinstance(x, Linear):
        buckets: list[dict[str, Tensor]] = [{} for _ in range(num)]
        for kind, t in x.tensors.items():
            d = _norm(x.output_dim, t.dim())
            parts = t.split(t.size(d) // num, dim=d)
            for i, part in enumerate(parts):
                buckets[i][kind] = part
        return [Linear(tensors=b, input_dim=x.input_dim, output_dim=x.output_dim) for b in buckets]
    assert dim is not None, "dim is required for Tensor"
    d = _norm(dim, x.dim())
    return list(x.split(x.size(d) // num, dim=d))


# ---------------------------------------------------------------------------
# split_in_dim
# ---------------------------------------------------------------------------


@overload
def split_in_dim(x: Linear, num: int) -> list[Linear]: ...


@overload
def split_in_dim(x: Tensor, num: int, dim: int) -> list[Tensor]: ...


def split_in_dim(x, num, dim=None):
    """Split along input dim into *num* equal parts."""
    if isinstance(x, Linear):
        buckets: list[dict[str, Tensor]] = [{} for _ in range(num)]
        for kind, t in x.tensors.items():
            if not _has_input_dim(t):
                for i in range(num):
                    buckets[i][kind] = t
                continue
            d = _norm(x.input_dim, t.dim())
            parts = t.split(t.size(d) // num, dim=d)
            for i, part in enumerate(parts):
                buckets[i][kind] = part
        return [Linear(tensors=b, input_dim=x.input_dim, output_dim=x.output_dim) for b in buckets]
    assert dim is not None, "dim is required for Tensor"
    d = _norm(dim, x.dim())
    return list(x.split(x.size(d) // num, dim=d))


# ---------------------------------------------------------------------------
# concat_out_dim
# ---------------------------------------------------------------------------


@overload
def concat_out_dim(xs: list[Linear]) -> Linear: ...


@overload
def concat_out_dim(xs: list[Tensor], dim: int) -> Tensor: ...


def concat_out_dim(xs, dim=None):
    """Concatenate along output dim."""
    if isinstance(xs[0], Linear):
        first = xs[0]
        result: dict[str, Tensor] = {}
        for kind in first.tensors:
            d = _norm(first.output_dim, first.tensors[kind].dim())
            result[kind] = torch.cat([x.tensors[kind] for x in xs], dim=d)
        return Linear(tensors=result, input_dim=first.input_dim, output_dim=first.output_dim)
    assert dim is not None, "dim is required for Tensor"
    return torch.cat(xs, dim=_norm(dim, xs[0].dim()))


# ---------------------------------------------------------------------------
# concat_in_dim
# ---------------------------------------------------------------------------


@overload
def concat_in_dim(xs: list[Linear]) -> Linear: ...


@overload
def concat_in_dim(xs: list[Tensor], dim: int) -> Tensor: ...


def concat_in_dim(xs, dim=None):
    """Concatenate along input dim."""
    if isinstance(xs[0], Linear):
        first = xs[0]
        result: dict[str, Tensor] = {}
        for kind in first.tensors:
            t0 = first.tensors[kind]
            if not _has_input_dim(t0):
                result[kind] = t0
                continue
            d = _norm(first.input_dim, t0.dim())
            result[kind] = torch.cat([x.tensors[kind] for x in xs], dim=d)
        return Linear(tensors=result, input_dim=first.input_dim, output_dim=first.output_dim)
    assert dim is not None, "dim is required for Tensor"
    return torch.cat(xs, dim=_norm(dim, xs[0].dim()))


# ---------------------------------------------------------------------------
# permute_out_dim
# ---------------------------------------------------------------------------


@overload
def permute_out_dim(x: Linear, shape: list[int], order: list[int]) -> Linear: ...


@overload
def permute_out_dim(x: Tensor, shape: list[int], order: list[int], dim: int) -> Tensor: ...


def permute_out_dim(x, shape, order, dim=None):
    """View output dim as *shape*, permute sub-dims by *order*, flatten back."""
    if isinstance(x, Linear):
        result: dict[str, Tensor] = {}
        for kind, t in x.tensors.items():
            d = _norm(x.output_dim, t.dim())
            result[kind] = _permute_along(t, d, shape, order)
        return Linear(tensors=result, input_dim=x.input_dim, output_dim=x.output_dim)
    assert dim is not None, "dim is required for Tensor"
    return _permute_along(x, _norm(dim, x.dim()), shape, order)


# ---------------------------------------------------------------------------
# permute_in_dim
# ---------------------------------------------------------------------------


@overload
def permute_in_dim(x: Linear, shape: list[int], order: list[int]) -> Linear: ...


@overload
def permute_in_dim(x: Tensor, shape: list[int], order: list[int], dim: int) -> Tensor: ...


def permute_in_dim(x, shape, order, dim=None):
    """View input dim as *shape*, permute sub-dims by *order*, flatten back."""
    if isinstance(x, Linear):
        result: dict[str, Tensor] = {}
        for kind, t in x.tensors.items():
            if not _has_input_dim(t):
                result[kind] = t
                continue
            d = _norm(x.input_dim, t.dim())
            result[kind] = _permute_along(t, d, shape, order)
        return Linear(tensors=result, input_dim=x.input_dim, output_dim=x.output_dim)
    assert dim is not None, "dim is required for Tensor"
    return _permute_along(x, _norm(dim, x.dim()), shape, order)


# ---------------------------------------------------------------------------
# transpose
# ---------------------------------------------------------------------------


@overload
def transpose(x: Linear) -> Linear: ...


@overload
def transpose(x: Tensor) -> Tensor: ...


def transpose(x):
    """Swap input and output dims (Linear) or dims 0 and 1 (Tensor)."""
    if x is None:
        return None
    if isinstance(x, Linear):
        result: dict[str, Tensor] = {}
        for kind, t in x.tensors.items():
            if t.dim() > 1:
                d_in = _norm(x.input_dim, t.dim())
                d_out = _norm(x.output_dim, t.dim())
                dims = list(range(t.dim()))
                dims[d_in], dims[d_out] = dims[d_out], dims[d_in]
                result[kind] = t.permute(dims).contiguous()
            else:
                result[kind] = t
        return Linear(tensors=result, input_dim=x.output_dim, output_dim=x.input_dim)
    return x.t()


# ---------------------------------------------------------------------------
# pad_out_dim
# ---------------------------------------------------------------------------


@overload
def pad_out_dim(x: Linear, target: int) -> Linear: ...


@overload
def pad_out_dim(x: Tensor, target: int, dim: int) -> Tensor: ...


def pad_out_dim(x, target, dim=None):
    """Pad output dim to *target* size with zeros."""
    if isinstance(x, Linear):
        result: dict[str, Tensor] = {}
        for kind, t in x.tensors.items():
            d = _norm(x.output_dim, t.dim())
            result[kind] = _pad_1d(t, d, target)
        return Linear(tensors=result, input_dim=x.input_dim, output_dim=x.output_dim)
    assert dim is not None, "dim is required for Tensor"
    return _pad_1d(x, _norm(dim, x.dim()), target)


# ---------------------------------------------------------------------------
# pad_in_dim
# ---------------------------------------------------------------------------


@overload
def pad_in_dim(x: Linear, target: int) -> Linear: ...


@overload
def pad_in_dim(x: Tensor, target: int, dim: int) -> Tensor: ...


def pad_in_dim(x, target, dim=None):
    """Pad input dim to *target* size with zeros."""
    if isinstance(x, Linear):
        result: dict[str, Tensor] = {}
        for kind, t in x.tensors.items():
            if not _has_input_dim(t):
                result[kind] = t
                continue
            d = _norm(x.input_dim, t.dim())
            result[kind] = _pad_1d(t, d, target)
        return Linear(tensors=result, input_dim=x.input_dim, output_dim=x.output_dim)
    assert dim is not None, "dim is required for Tensor"
    return _pad_1d(x, _norm(dim, x.dim()), target)
