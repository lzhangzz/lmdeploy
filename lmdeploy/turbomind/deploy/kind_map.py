# Copyright (c) OpenMMLab. All rights reserved.
"""Per-format suffix-to-kind mapping and checkpoint normalization.

Each model format (trivial, AWQ, GPTQ, compressed-tensors, FP8, mxfp4) defines
exactly the checkpoint suffixes it uses.  ``get_suffix_map(model_format)``
selects the right mapping at init time so that ``read_linear()`` only probes
relevant suffixes.

``get_normalizer(model_format)`` returns a ``(tensor, kind) -> tensor``
callable that normalizes raw checkpoint data (unpack, transpose, dtype cast)
into the canonical layout expected by TurboMind.  This absorbs the per-format
logic that previously lived in ``policy.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import torch
from torch import Tensor

import _turbomind as _tm

# ---------------------------------------------------------------------------
# WeightFormat descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeightFormat:
    """Immutable descriptor for one checkpoint quantization format.

    Fields
    ------
    name : str | None
        Canonical format name (``None`` for trivial/HF).
    suffix_map : dict[str, str]
        Mapping ``{checkpoint_suffix: tm_kind}``.
    normalizer : Callable[[Tensor, str], Tensor]
        Converts a raw checkpoint tensor to TM layout ``[in, out]``.
    packer : Callable[[Tensor, str], Tensor] | None
        Optional commit-time packer (e.g. ``pack_u4_row``). Applied in
        ``commit_linear`` just before saving.  Receives ``(tensor, kind)``.
    cpp_dtype_name : str | None
        Attribute name on ``_turbomind.DataType`` for the C++ weight dtype,
        or ``None`` for trivial formats whose dtype is inferred from the tensor.
    block_in : int | None
        Grouping of input elements per scale entry along the input dim.
        ``None`` → no per-element scale (trivial).  ``0`` → read from
        ``model_config.group_size`` at commit time.
    block_out : int | None
        Grouping of output elements per scale entry along the output dim.
        ``None`` → no per-output scale.  ``0`` → read from model config.
    zeros_factory : Callable[[Tensor], Tensor] | None
        Optional factory called with the ``"scales"`` tensor to synthesize a
        ``"zeros"`` tensor when one is absent from the checkpoint.  The
        factory encapsulates the zero-point dtype, value, and shape so that
        no format-specific logic leaks into ``build_linear``.
        ``None`` means no synthesis (format either always provides zeros or
        has no zero-point concept).
    accepts : Callable[[dict[str, Tensor]], bool]
        Predicate that receives a ``{suffix: raw_tensor}`` dict of the
        checkpoint tensors actually present for a parameter and returns
        ``True`` when the available tensors satisfy this format.  Both key
        presence and tensor dtype are checked so that, e.g., an FP8 layer
        stored without ``weight_scale_inv`` is correctly classified as trivial
        rather than fp8.
    dequant : Callable[[dict[str, Tensor]], dict[str, Tensor]] | None
        Optional fusion-time dequantizer: maps TM ``tensors`` to a trivial
        ``{weight, bias?}`` dict.  ``None`` means the format is not dequantized
        in Python (e.g. GPTQ / MXFP4 stay as-is for mixed-format fusion).
    """

    name: str | None
    suffix_map: dict[str, str]
    normalizer: Callable[[Tensor, str], Tensor]
    packer: Callable[[Tensor, str], Tensor] | None
    cpp_dtype_name: str | None
    block_in: int | None
    block_out: int | None
    zeros_factory: Callable[[Tensor], Tensor] | None
    accepts: Callable[[dict[str, Tensor]], bool]
    dequant: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None

    def __hash__(self) -> int:
        return hash(self.name)

    def to_data_format(self, cpp_dtype: int, group_size: int = 0):
        """Construct a C++ DataFormat from this WeightFormat.

        Returns None when group_size is needed but not yet known (block_in==0
        and group_size==0), or when the format is trivial (block_in is None).
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

    def complete_tensors(self, tensors: dict[str, Tensor]) -> None:
        """Add any synthesizable tensors absent from the checkpoint in-place.

        Each format knows its own completion rules.  Currently the only case
        is symmetric int4 zero-point synthesis for GPTQ / compressed-tensors:
        when ``scales`` are present but ``zeros`` are missing, call
        ``zeros_factory(scales)`` to produce a default zero-point tensor.
        """
        if self.zeros_factory is not None and "scales" in tensors and "zeros" not in tensors:
            tensors["zeros"] = self.zeros_factory(tensors["scales"])


# ---------------------------------------------------------------------------
# Per-format suffix -> TM kind mappings
# ---------------------------------------------------------------------------

TRIVIAL_SUFFIXES: dict[str, str] = {
    ".weight": "weight",
    ".bias": "bias",
}

AWQ_SUFFIXES: dict[str, str] = {
    ".qweight": "weight",
    ".scales": "scales",
    ".qzeros": "zeros",
    ".bias": "bias",
}

GPTQ_SUFFIXES: dict[str, str] = {
    ".qweight": "weight",
    ".scales": "scales",
    ".qzeros": "zeros",
    ".bias": "bias",
}

COMPRESSED_TENSOR_SUFFIXES: dict[str, str] = {
    ".weight_packed": "weight",
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
    None: TRIVIAL_SUFFIXES,
    "hf": TRIVIAL_SUFFIXES,
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


def _normalize_trivial(x: Tensor, kind: str) -> Tensor:
    x = x.cuda()
    if x.dim() >= 2:
        x = x.t()
    return x


def _normalize_awq(x: Tensor, kind: str) -> Tensor:
    # AWQ checkpoints store weights in TM-native layout:
    #   qweight: [K, N//8] int32  → after unpack → [K, N] (TM, no .t() needed)
    #   scales:  [K//g, N] float16 → already TM
    #   zeros:   [K//g, N//8] int32 → after unpack → [K//g, N] (TM, no .t() needed)
    x = x.cuda()
    if x.dtype == torch.int32:
        x = _unpack_awq_gemm(x)
    if kind == "zeros":
        x = x.to(torch.float16)
    return x


def _normalize_gptq(x: Tensor, kind: str) -> Tensor:
    # GPTQ checkpoint stores weights in TM-native layout:
    #   qweight: [K//8, N] int32  → after unpack → [K, N] (TM, no .t() needed)
    #   scales:  [K//g, N] float16 → already TM
    #   zeros:   [K//g, N//8] int32 → after unpack → [K//g, N] (TM, no .t() needed)
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


def _normalize_mxfp4(x: Tensor, kind: str) -> Tensor:
    x = x.cuda()
    if kind == "weight":
        xs = _get_u4_slices(torch.flatten(x, start_dim=-2), torch.uint8)
        x = torch.flatten(torch.stack(xs, dim=-1), start_dim=-2)
    if x.dim() >= 2:
        x = x.t()
    return x


def _normalize_fp8(x: Tensor, kind: str) -> Tensor:
    x = x.cuda()
    if x.dtype == torch.float8_e4m3fn:
        x = x.view(dtype=torch.uint8)
    if x.dim() >= 2:
        x = x.t()
    return x


def _normalize_compressed_tensor(x: Tensor, kind: str) -> Tensor:
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


_NORMALIZER_MAP: dict[str | None, Callable[[Tensor, str], Tensor]] = {
    None: _normalize_trivial,
    "hf": _normalize_trivial,
    "awq": _normalize_awq,
    "gptq": _normalize_gptq,
    "compressed-tensors": _normalize_compressed_tensor,
    "fp8": _normalize_fp8,
    "mxfp4": _normalize_mxfp4,
}


def get_normalizer(model_format: str | None) -> Callable[[Tensor, str], Tensor]:
    """Return a ``(tensor, kind) -> tensor`` normalizer for *model_format*."""
    return _NORMALIZER_MAP[model_format]


# ---------------------------------------------------------------------------
# Packer helpers (applied at commit_linear time)
# ---------------------------------------------------------------------------


def _pack_u4_qweight(tensor: Tensor, kind: str) -> Tensor:
    """Pack uint8 4-bit values into int32 rows; applied to quantized ``weight``."""
    if kind == "weight" and tensor.dtype == torch.uint8:
        return pack_u4_row(tensor)
    return tensor


def _pack_mxfp4_weight(tensor: Tensor, kind: str) -> Tensor:
    """Pack uint8 4-bit values into int32 rows; applied to mxfp4 ``weight``."""
    if kind == "weight" and tensor.dtype == torch.uint8:
        return pack_u4_row(tensor)
    return tensor


# ---------------------------------------------------------------------------
# Zeros factory helpers
# ---------------------------------------------------------------------------


def _zeros_int4_symmetric(scales: Tensor) -> Tensor:
    """Synthesize symmetric int4 zero-points (all 8) matching *scales* shape.

    Used by GPTQ and compressed-tensors, which may omit zero-points when
    the quantization is symmetric (zero-point = 2**(bits-1) = 8 for int4).
    """
    return torch.full(scales.shape, 8, dtype=torch.uint8, device=scales.device)


# ---------------------------------------------------------------------------
# Format acceptance predicates
# ---------------------------------------------------------------------------


def _accepts_trivial(available: dict[str, "Tensor"]) -> bool:
    """Trivial: only .weight and/or .bias present; weight must be floating-point."""
    if not (available.keys() <= {".weight", ".bias"}):
        return False
    w = available.get(".weight")
    return w is None or w.dtype.is_floating_point


def _accepts_awq(available: dict[str, "Tensor"]) -> bool:
    """AWQ: int32 qweight packed along N-dim (qweight.shape[-1] * 8 == scales.shape[-1])."""
    qw = available.get(".qweight")
    if qw is None or qw.dtype != torch.int32:
        return False
    scales = available.get(".scales")
    if scales is not None and qw.ndim >= 2 and scales.ndim >= 2:
        return qw.shape[-1] * 8 == scales.shape[-1]
    return True  # no scales to disambiguate; AWQ has priority


def _accepts_gptq(available: dict[str, "Tensor"]) -> bool:
    """GPTQ: int32 qweight packed along K-dim (qweight.shape[-1] == scales.shape[-1])."""
    qw = available.get(".qweight")
    if qw is None or qw.dtype != torch.int32:
        return False
    scales = available.get(".scales")
    if scales is not None and qw.ndim >= 2 and scales.ndim >= 2:
        return qw.shape[-1] == scales.shape[-1]
    return True


def _accepts_compressed_tensor(available: dict[str, "Tensor"]) -> bool:
    """Compressed-tensors: weight_packed is int32."""
    wp = available.get(".weight_packed")
    return wp is not None and wp.dtype == torch.int32


def _accepts_fp8(available: dict[str, "Tensor"]) -> bool:
    """FP8: weight_scale_inv must be present; weight dtype must be float8_e4m3fn or uint8."""
    if ".weight_scale_inv" not in available:
        return False
    w = available.get(".weight")
    return w is None or w.dtype in (torch.float8_e4m3fn, torch.uint8)


def _accepts_mxfp4(available: dict[str, "Tensor"]) -> bool:
    """MXFP4: packed uint8 blocks (4-bit weights) and scales (E8M0, dtype not assumed) required."""
    if ".scales" not in available:
        return False
    w = available.get(".blocks")
    return w is None or w.dtype == torch.uint8


# ---------------------------------------------------------------------------
# Fusion-time dequantizers: TM tensors -> trivial {weight, bias?}
# ---------------------------------------------------------------------------


def _dequant_awq(tensors: dict[str, Tensor]) -> dict[str, Tensor]:
    from lmdeploy.pytorch.backends.default.awq_modules import dequantize_gemm

    qweight = tensors["weight"]
    scales = tensors["scales"]
    qzeros = tensors["zeros"]
    group_size = qweight.shape[0] // scales.shape[0]
    w = dequantize_gemm(qweight, qzeros, scales, 4, group_size)
    result: dict[str, Tensor] = {"weight": w}
    if "bias" in tensors:
        result["bias"] = tensors["bias"]
    return result


def _dequant_fp8(tensors: dict[str, Tensor]) -> dict[str, Tensor]:
    weight = tensors["weight"]
    scales = tensors["scales"]
    block_size = 128
    fp8_weight = weight.view(torch.float8_e4m3fn).float()
    scale = scales.float()
    scale = scale.repeat_interleave(block_size, dim=0)
    scale = scale.repeat_interleave(block_size, dim=1)
    scale = scale[: fp8_weight.shape[0], : fp8_weight.shape[1]]
    result: dict[str, Tensor] = {"weight": (fp8_weight * scale).to(torch.bfloat16)}
    if "bias" in tensors:
        result["bias"] = tensors["bias"]
    return result


# ---------------------------------------------------------------------------
# WeightFormat singletons
# ---------------------------------------------------------------------------

TRIVIAL_FORMAT = WeightFormat(
    name="trivial",
    suffix_map=TRIVIAL_SUFFIXES,
    normalizer=_normalize_trivial,
    packer=None,
    cpp_dtype_name=None,
    block_in=None,
    block_out=None,
    zeros_factory=None,
    accepts=_accepts_trivial,
    dequant=None,
)

AWQ_FORMAT = WeightFormat(
    name="awq",
    suffix_map=AWQ_SUFFIXES,
    normalizer=_normalize_awq,
    packer=_pack_u4_qweight,
    cpp_dtype_name="TYPE_UINT4",
    block_in=0,   # group_size from model_config
    block_out=None,
    zeros_factory=None,   # AWQ checkpoints always include qzeros
    accepts=_accepts_awq,
    dequant=_dequant_awq,
)

GPTQ_FORMAT = WeightFormat(
    name="gptq",
    suffix_map=GPTQ_SUFFIXES,
    normalizer=_normalize_gptq,
    packer=_pack_u4_qweight,
    cpp_dtype_name="TYPE_UINT4",
    block_in=0,   # group_size from model_config
    block_out=None,
    zeros_factory=_zeros_int4_symmetric,
    accepts=_accepts_gptq,
    dequant=None,
)

COMPRESSED_TENSOR_FORMAT = WeightFormat(
    name="compressed-tensors",
    suffix_map=COMPRESSED_TENSOR_SUFFIXES,
    normalizer=_normalize_compressed_tensor,
    packer=_pack_u4_qweight,
    cpp_dtype_name="TYPE_UINT4",
    block_in=0,
    block_out=None,
    zeros_factory=_zeros_int4_symmetric,
    accepts=_accepts_compressed_tensor,
    dequant=None,
)

FP8_FORMAT = WeightFormat(
    name="fp8",
    suffix_map=FP8_SUFFIXES,
    normalizer=_normalize_fp8,
    packer=None,
    cpp_dtype_name="TYPE_FP8_E4M3",
    block_in=128,
    block_out=128,
    zeros_factory=None,
    accepts=_accepts_fp8,
    dequant=_dequant_fp8,
)

MXFP4_FORMAT = WeightFormat(
    name="mxfp4",
    suffix_map=MXFP4_SUFFIXES,
    normalizer=_normalize_mxfp4,
    packer=_pack_mxfp4_weight,
    cpp_dtype_name="TYPE_FP4_E2M1",
    block_in=32,
    block_out=None,
    zeros_factory=None,
    accepts=_accepts_mxfp4,
    dequant=None,
)

_WEIGHT_FORMAT_MAP: dict[str | None, WeightFormat] = {
    None: TRIVIAL_FORMAT,
    "hf": TRIVIAL_FORMAT,
    "awq": AWQ_FORMAT,
    "gptq": GPTQ_FORMAT,
    "compressed-tensors": COMPRESSED_TENSOR_FORMAT,
    "fp8": FP8_FORMAT,
    "mxfp4": MXFP4_FORMAT,
}


def get_weight_format(model_format: str | None) -> WeightFormat:
    """Return the ``WeightFormat`` singleton for *model_format*."""
    return _WEIGHT_FORMAT_MAP[model_format]


# ---------------------------------------------------------------------------
# Format classification helpers
# ---------------------------------------------------------------------------

#: Ordered list of all formats used by ``build_linear`` for auto-detection.
#: Quantized formats are listed first so they win over trivial when tensors match.
FORMAT_PRIORITY: list[WeightFormat] = [
    AWQ_FORMAT,
    GPTQ_FORMAT,
    COMPRESSED_TENSOR_FORMAT,
    FP8_FORMAT,
    MXFP4_FORMAT,
    TRIVIAL_FORMAT,
]

#: Union of all checkpoint suffixes across every known format.
ALL_SUFFIXES: frozenset[str] = frozenset(s for fmt in FORMAT_PRIORITY for s in fmt.suffix_map)


# ---------------------------------------------------------------------------
# build_linear — produces Linear bundles from checkpoint keys
# ---------------------------------------------------------------------------


def build_linear(
    params: dict[str, torch.Tensor],
    prefix: str,
    *,
    index: int | None = None,
    block_in: int = 0,
    block_out: int = 0,
) -> Linear | None:
    """Build a ``Linear`` bundle from checkpoint tensors at *prefix*.

    Probes every known checkpoint suffix (union of all format suffix maps),
    classifies the format by running each ``WeightFormat.accepts`` predicate
    in ``FORMAT_PRIORITY`` order, then normalises the collected tensors with
    the winning format's normalizer.

    When *index* is given, each collected tensor is sliced by ``[index]``
    before classification and normalisation (used for packed expert tensors
    where the expert dimension is the leading axis).

    ``block_in`` and ``block_out`` resolve the format's quantization block
    sizes at conversion time.  A format with ``block_in == 0`` (AWQ, GPTQ,
    compressed-tensors) declares "use the runtime group_size".  The caller
    passes that value; we clone the format with ``dataclasses.replace`` so
    the returned ``Linear`` carries an authoritative ``WeightFormat``.  The
    same mechanism applies to ``block_out == 0``, reserved for future
    formats.  Passing ``0`` means "no runtime value available" and leaves
    the format's declared sentinel in place.

    The returned ``Linear`` is in TM layout ``[in, out]`` and carries the
    detected ``WeightFormat`` for downstream use in ``_commit_linear``.
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

    replacements: dict[str, int] = {}
    if fmt.block_in == 0 and block_in > 0:
        replacements['block_in'] = block_in
    if fmt.block_out == 0 and block_out > 0:
        replacements['block_out'] = block_out
    if replacements:
        fmt = replace(fmt, **replacements)

    tensors: dict[str, torch.Tensor] = {
        kind: fmt.normalizer(available[s], kind)
        for s, kind in fmt.suffix_map.items()
        if s in available
    }
    if not tensors:
        return None

    fmt.complete_tensors(tensors)
    return Linear(tensors=tensors, weight_format=fmt, data_format=None)


def pack_u4_row(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.uint8, f'x.dtype: {x.dtype}'
    xs = x.view(*x.shape[:-1], -1, 8).split(1, dim=-1)
    a = torch.zeros(xs[0].shape, dtype=torch.int32, device=x.device)
    for t in reversed(xs):
        a = (a << 4) | t
    return a.squeeze(dim=-1)
