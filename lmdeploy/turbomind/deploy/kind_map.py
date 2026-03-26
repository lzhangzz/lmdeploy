# Copyright (c) OpenMMLab. All rights reserved.
"""Per-format suffix-to-kind mapping and checkpoint normalization.

Each model format (dense, AWQ, GPTQ, compressed-tensors, FP8, mxfp4) defines
exactly the checkpoint suffixes it uses.  ``get_suffix_map(model_format)``
selects the right mapping at init time so that ``read_linear()`` only probes
relevant suffixes.

``get_normalizer(model_format)`` returns a ``(tensor, kind) -> tensor``
callable that normalizes raw checkpoint data (unpack, transpose, dtype cast)
into the canonical layout expected by TurboMind.  This absorbs the per-format
logic that previously lived in ``policy.py``.
"""

from __future__ import annotations

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Per-format suffix -> TM kind mappings
# ---------------------------------------------------------------------------

DENSE_SUFFIXES: dict[str, str] = {
    ".weight": "weight",
    ".bias": "bias",
}

AWQ_SUFFIXES: dict[str, str] = {
    ".qweight": "qweight",
    ".scales": "scales",
    ".qzeros": "zeros",
    ".bias": "bias",
}

GPTQ_SUFFIXES: dict[str, str] = {
    ".qweight": "qweight",
    ".scales": "scales",
    ".qzeros": "zeros",
    ".bias": "bias",
}

COMPRESSED_TENSOR_SUFFIXES: dict[str, str] = {
    ".weight_packed": "qweight",
    ".weight_scale": "scales",
    ".weight_zero_point": "zeros",
    ".bias": "bias",
}

FP8_SUFFIXES: dict[str, str] = {
    ".weight": "weight",
    ".weight_scale_inv": "scales",
    ".bias": "bias",
}

MXFP4_SUFFIXES: dict[str, str] = {
    ".blocks": "weight",
    ".scales": "scales",
    ".bias": "bias",
}

_FORMAT_MAP: dict[str | None, dict[str, str]] = {
    None: DENSE_SUFFIXES,
    "hf": DENSE_SUFFIXES,
    "awq": AWQ_SUFFIXES,
    "gptq": GPTQ_SUFFIXES,
    "compressed-tensors": COMPRESSED_TENSOR_SUFFIXES,
    "fp8": FP8_SUFFIXES,
    "mxfp4": MXFP4_SUFFIXES,
}


def get_suffix_map(model_format: str | None) -> dict[str, str]:
    """Return ``{suffix: kind}`` for the given model format."""
    return dict(_FORMAT_MAP[model_format])


# ---------------------------------------------------------------------------
# Normalizer helpers (absorbed from policy.py)
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


# ---------------------------------------------------------------------------
# Per-format normalizers: (tensor, kind) -> tensor
# ---------------------------------------------------------------------------


def _normalize_dense(x: Tensor, kind: str) -> Tensor:
    return x.cuda()


def _normalize_awq(x: Tensor, kind: str) -> Tensor:
    x = x.cuda()
    if x.dtype == torch.int32:
        x = _unpack_awq_gemm(x)
    if kind == "zeros":
        x = x.to(torch.float16)
    if kind in ("qweight", "zeros", "scales"):
        x = x.t()
    return x


def _normalize_gptq(x: Tensor, kind: str) -> Tensor:
    x = x.cuda()
    if x.dtype == torch.int32:
        xs = _get_u4_slices(x, torch.uint8)
        if kind == "qweight":
            x = torch.stack(xs, dim=1).view(-1, x.size(-1))
        else:
            x = torch.stack(xs, dim=-1).view(x.size(0), -1) + 1
    if kind == "zeros":
        x = x.to(torch.float16)
    if kind in ("qweight", "zeros", "scales"):
        x = x.t()
    return x


def _normalize_mxfp4(x: Tensor, kind: str) -> Tensor:
    x = x.cuda()
    if kind == "weight":
        xs = _get_u4_slices(torch.flatten(x, start_dim=-2), torch.uint8)
        x = torch.flatten(torch.stack(xs, dim=-1), start_dim=-2)
    return x


def _normalize_fp8(x: Tensor, kind: str) -> Tensor:
    x = x.cuda()
    if x.dtype == torch.float8_e4m3fn:
        return x.view(dtype=torch.uint8)
    return x


def _normalize_compressed_tensor(x: Tensor, kind: str) -> Tensor:
    x = x.cuda()
    if x.dtype == torch.int32:
        xs = _get_u4_slices(x, torch.uint8)
        if kind == "qweight":
            x = torch.stack(xs, dim=-1).view(*x.shape[:-1], -1)
        elif kind == "zeros":
            x = torch.stack(xs, dim=1).view(-1, x.size(-1))
    if kind == "zeros":
        x = x.to(torch.float16)
    return x


_NORMALIZER_MAP: dict[str | None, callable] = {
    None: _normalize_dense,
    "hf": _normalize_dense,
    "awq": _normalize_awq,
    "gptq": _normalize_gptq,
    "compressed-tensors": _normalize_compressed_tensor,
    "fp8": _normalize_fp8,
    "mxfp4": _normalize_mxfp4,
}


def get_normalizer(model_format: str | None):
    """Return a ``(tensor, kind) -> tensor`` normalizer for *model_format*."""
    return _NORMALIZER_MAP[model_format]
