# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from .kind_map import DENSE_FORMAT
from .linear import Linear, _norm, pad_out_dim
from .linear import transpose as linear_transpose

if TYPE_CHECKING:
    from .target_model.base import BaseOutputModel


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



class Transformer:

    def __init__(self, model: BaseOutputModel):
        self._v2 = TransformerV2(model)

    def __call__(self, i: int, r: ModelWeightSpec):
        return self._v2(i, r)


# ===================================================================
# New pipeline: ModelWeightSpec + composable-ops Transformer
# ===================================================================


class ModelWeightSpec(ABC):
    """Declarative weight mapping for a model architecture.

    Subclasses define how to read and transform weights for a specific model.
    Methods return ``Linear`` for linear layers and raw ``Tensor`` for norms,
    embeddings, scalars, etc.

    The ``TransformerV2`` consumes a spec: it iterates the returned dicts,
    applies TP split rules, and commits each weight to C++.
    """

    params: dict[str, torch.Tensor]

    # Default values for configure() fields; overwritten by configure().
    _attn_tp: int = 1
    _permute_qk: bool = True
    _repeat_kv: int = 0
    _head_dim: int = 0
    _rope_dim: int = 0
    _attn_output_gate: bool = False
    _kv_head_num: int = 0
    # TODO: there dont belong here
    _linear_qkv_split: tuple[int, int, int] | None = None
    _gdn_qkv_split: tuple[int, int, int] | None = None

    # -- Configuration (called by TransformerV2 before processing) --

    def configure(
        self,
        attn_tp: int = 1,
        permute_qk: bool = True,
        repeat_kv: int = 0,
        head_dim: int = 0,
        rope_dim: int = 0,
        attn_output_gate: bool = False,
        kv_head_num: int = 0,
    ):
        """Set TP and model parameters needed for QKV merge and GDN fusion.

        Called by ``TransformerV2`` before processing each layer batch.
        Idempotent — safe to call repeatedly with the same values.
        """
        self._attn_tp = attn_tp
        self._permute_qk = permute_qk
        self._repeat_kv = repeat_kv
        self._head_dim = head_dim
        self._rope_dim = rope_dim if rope_dim else head_dim
        self._attn_output_gate = attn_output_gate
        self._kv_head_num = kv_head_num

    # -- Common helpers (subclasses may override) --

    def _get(self, key: str) -> torch.Tensor | None:
        """Get a raw tensor from the checkpoint params."""
        return self.params.get(key)

    def _read_linear(self, prefix: str) -> Linear | None:
        """Read a Linear bundle from the checkpoint at *prefix*.

        Probes all known suffixes and auto-detects the format via
        ``WeightFormat.accepts``.  Override for model-specific logic.
        """
        from .parameter import build_linear
        return build_linear(self.params, prefix)

    _FFN_MAP = [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]

    def _read_ffn_linears(self, pfx: str) -> dict[str, Linear]:
        """Read standard gate/down/up FFN linears."""
        result: dict[str, Linear] = {}
        for tm_name, hf_key in self._FFN_MAP:
            lin = self._read_linear(f"{pfx}.{hf_key}")
            if lin is not None:
                result[tm_name] = lin
        return result

    # -- Linear bundles (TP-split by the transformer) --

    def attn_linears(self, layer: int) -> dict[str, Linear]:
        """Return ``{tm_name: Linear}`` for attention weights.

        Wraps ``_read_attn_linears()``.  If the raw dict contains the
        ``w_qkv.{q,k,v}`` pattern, merges them into a single ``w_qkv``
        ``Linear`` (already TP-interleaved and in TM layout).
        MLA specs return non-QKV keys and the merge is skipped.
        """
        raw = self._read_attn_linears(layer)
        if "w_qkv.q" in raw and "w_qkv.k" in raw and "w_qkv.v" in raw:
            q = raw.pop("w_qkv.q")
            k = raw.pop("w_qkv.k")
            v = raw.pop("w_qkv.v")
            merged = merge_qkv_linear(
                q, k, v,
                tp=self._attn_tp,
                head_dim=self._head_dim,
                rope_dim=self._rope_dim,
                permute_qk=self._permute_qk,
                attn_output_gate=self._attn_output_gate,
                repeat_kv=self._repeat_kv,
                kv_head_num=self._kv_head_num,
            )
            raw["w_qkv"] = merged
            # Pad wo bias with zeros if QKV has bias but wo does not
            o_lin = raw.get("wo")
            if o_lin is not None and "bias" in merged.tensors and "bias" not in o_lin.tensors:
                o_lin.tensors["bias"] = torch.zeros_like(merged.tensors["bias"])
        return raw

    def _read_attn_linears(self, layer: int) -> dict[str, Linear]:
        """Override in subclasses to return raw attention ``Linear`` bundles."""
        return {}

    def ffn_linears(self, layer: int) -> dict[str, Linear]:
        """Return ``{tm_name: Linear}`` for dense FFN / shared-expert weights."""
        return {}

    def moe_ffn_linears(self, layer: int, expert: int) -> dict[str, Linear]:
        """Return ``{tm_name: Linear}`` for one MoE routed expert."""
        return {}

    def linear_attn_linears(self, layer: int) -> dict[str, Linear]:
        """Return ``{tm_name: Linear}`` for linear-attention (GDN) weights.

        Wraps ``_read_linear_attn_linears()``.  If the raw dict contains any
        of the GDN in-proj keys, fuses them into ``in_proj_all`` with TP
        interleaving via ``fuse_gdn_in_proj()``.
        """
        raw = self._read_linear_attn_linears(layer)
        if any(k in raw for k in _GDN_IN_PROJ_KEYS):
            raw = fuse_gdn_in_proj(raw, self._attn_tp,
                                   qkv_split=self._linear_qkv_split)
        return raw

    def _read_linear_attn_linears(self, layer: int) -> dict[str, Linear]:
        """Override in subclasses to return raw GDN ``Linear`` bundles."""
        return {}

    # -- Raw tensors (broadcast or simple split) --

    def attn_norm(self, layer: int) -> torch.Tensor | None:
        return None

    def ffn_norm(self, layer: int) -> torch.Tensor | None:
        return None

    def tok_embeddings(self) -> torch.Tensor | None:
        return None

    def output_weight(self) -> torch.Tensor | None:
        return None

    def norm_weight(self) -> torch.Tensor | None:
        return None

    def raw_layer_tensors(
        self, layer: int
    ) -> list[tuple[str, torch.Tensor | None, SplitSide | None]]:
        """Return raw per-layer tensors for ``TransformerV2`` to commit.

        Each entry is ``(tm_path, tensor, split_side)`` where *tm_path* is
        the suffix after ``layers.{layer}.``.  ``split_side`` follows the
        same convention as ``commit_tensor``:

        - ``None``                — broadcast to all TP ranks (no split).
        - ``SplitSide.OUTPUT``    — split along the last axis (dim -1).
        - ``SplitSide.INPUT``     — split along the first axis (dim 0).
        """
        return []

    def _permute_qk_tensors(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE layout permutation to Q and K norm weights.

        Uses the ``_permute_qk``, ``_head_dim``, and ``_rope_dim`` fields
        set by ``configure()``.
        """
        if not self._permute_qk:
            return q, k
        if self._rope_dim < self._head_dim:
            q = permute_v2_partial(q, self._head_dim, self._rope_dim)
            k = permute_v2_partial(k, self._head_dim, self._rope_dim)
        else:
            q = permute_v2(q, self._head_dim)
            k = permute_v2(k, self._head_dim)
        return q, k

    # -- metadata --

    @abstractmethod
    def model_info(self) -> dict:
        """Return model metadata (num_layer, head_num, etc.)."""

    def num_experts(self, layer: int) -> int:
        return 0


# -----------------------------------------------------------------------
# Commit helpers
# -----------------------------------------------------------------------


def _dequant_linear(linear: Linear) -> Linear:
    """Dequantize a quantized Linear to dense when the format provides ``dequant``."""
    fmt = linear.weight_format
    if fmt is None or fmt.dequant is None:
        return linear
    new_tensors = fmt.dequant(linear.tensors)
    return Linear(tensors=new_tensors, weight_format=DENSE_FORMAT)


def _ensure_compatible_formats(linears: dict[str, Linear]) -> dict[str, Linear]:
    """Dequant linears to a common dense format if a fusion group has mixed formats."""
    formats = {name: lin.weight_format.name for name, lin in linears.items()}
    if len(set(formats.values())) <= 1:
        return linears
    return {name: _dequant_linear(lin) for name, lin in linears.items()}


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

    # Dense (or missing format): dtype from weight tensor
    weight = linear.tensors.get("weight")
    if weight is not None:
        if weight.dtype == torch.bfloat16:
            return _tm.DataType.TYPE_BF16, 0
        if weight.dtype == torch.float16:
            return _tm.DataType.TYPE_FP16, 0
    return None, 0


# -----------------------------------------------------------------------
# QKV merge and GDN fusion free functions
# -----------------------------------------------------------------------

_GDN_IN_PROJ_KEYS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")


def _block_ops_need_dequant(
    lin: Linear, head_dim: int,
    repeat_kv: bool, attn_output_gate: bool, permute_qk: bool,
) -> bool:
    """Return True if any planned QKV-merge operation crosses block boundaries.

    For quantised formats with a non-None ``block_out``:

    - KV repetition and output-gate splitting each split along the output
      dimension at head_dim granularity.  This is block-safe only when
      ``head_dim % block_out == 0`` (each KV head = integer number of blocks).
    - RoPE permutation permutes elements within a head.  This is block-safe
      only when ``block_out % head_dim == 0`` (each block = integer number
      of heads, so permuting within one head is intra-block).

    If any condition fails, the caller should dequantise to dense first.
    """
    wfmt = lin.weight_format
    if wfmt is None or wfmt.block_out is None:
        return False
    block_out = wfmt.block_out
    if (repeat_kv or attn_output_gate) and head_dim % block_out != 0:
        return True
    if permute_qk and block_out % head_dim != 0:
        return True
    return False


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
    """Merge Q/K/V ``Linear`` bundles into a single interleaved ``w_qkv`` Linear.

    Applies RoPE permutation, KV head repetition, and output-gate splitting
    as required, then interleaves components for TP.  Returns a ``Linear``
    in TM layout ``[in, out]``.

    Per-kind routing
    ----------------
    - ``weight`` / ``qweight`` / ``bias``: per-element structure along the
      output dim; all transforms applied when ``tensor.size(-1) % head_dim == 0``.
    - ``scales`` / ``zeros`` with ``block_out > 0``: block structure; KV
      repetition uses ``repeat_interleave`` at block granularity.  All other
      per-element transforms (permute, gate-split) are skipped for these kinds.

    Block-unsafe operations
    -----------------------
    If any planned transformation would cross block boundaries (detected via
    ``_block_ops_need_dequant``), the entire group is dequantised to dense
    bf16/fp16 before merging, with a warning.
    """
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

        # Block scales (scales/zeros with block_out > 0) carry one value per
        # block of output elements; they need block-granular operations.
        wfmt = q.weight_format
        block_out = (wfmt.block_out or 0) if wfmt is not None else 0
        is_block_kind = kind in ("scales", "zeros") and block_out > 0

        # Per-element flag: True when the output dim is head_dim-aligned and
        # the kind is not a block scale (so reshape/repeat work correctly).
        full_res = not is_block_kind and (qt.size(-1) % head_dim == 0)
        gate = None

        if repeat_kv:
            n = repeat_kv
            kv_heads = kv_head_num // n
            if is_block_kind:
                # Block-scale KV repetition: repeat each output-block entry n
                # times.  The pre-dequant check above guarantees alignment.
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

    # All three linears share the same format after _ensure_compatible_formats.
    return Linear(tensors=merged_tensors, weight_format=q.weight_format)


def _tp_interleave_tensor(t: torch.Tensor, tp: int, d: int) -> torch.Tensor:
    """Reshape last dim as [tp, per_tp] and flatten to interleave by TP rank."""
    shape = list(t.shape)
    new_shape = shape[:d] + [tp, shape[d] // tp] + shape[d + 1:]
    return t.reshape(new_shape)


def fuse_gdn_in_proj(la_linears: dict[str, Linear], tp: int,
                     qkv_split: tuple[int, int, int] | None = None) -> dict[str, Linear]:
    """Fuse GDN input projections into ``in_proj_all`` with TP interleaving.

    Pops ``in_proj_qkv``, ``in_proj_z``, ``in_proj_b``, ``in_proj_a`` from
    *la_linears* and inserts a single ``in_proj_all``.  Returns the updated
    dict (does not modify the input in place).

    For ``tp=1`` this reduces to a plain ``concat_out_dim``.

    When *qkv_split* ``(q_dim, k_dim, v_dim)`` is provided and ``tp > 1``,
    the in_proj_qkv weight is split into its Q, K, V sub-projections and
    each is TP-interleaved independently before concatenation.  This is
    necessary because Q, K, V may have different output dimensions, so a
    naive column split would mix data from different projections across
    TP ranks.
    """
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

    # sub-projections Q, K, V with different output dims.  Split and
    # interleave each separately to respect head boundaries.
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
                # 1-D tensors (bias): simple split
                parts = [qkv_t[..., :q_dim], qkv_t[..., q_dim:q_dim + k_dim],
                         qkv_t[..., q_dim + k_dim:]]
                rest_ts = [lin.tensors.get(kind) for lin in rest
                           if lin.tensors.get(kind) is not None]
                fused_tensors[kind] = torch.cat(parts + rest_ts, dim=0)
                continue

            # Split QKV into Q, K, V along the output dim
            q_t = qkv_t[..., :q_dim]
            k_t = qkv_t[..., q_dim:q_dim + k_dim]
            v_t = qkv_t[..., q_dim + k_dim:]

            # TP-interleave each sub-projection independently
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

        result["in_proj_all"] = Linear(tensors=fused_tensors, weight_format=first.weight_format)
        return result

    # Default path: all components have compatible output dims for naive split.
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

    # All components share the same format after _ensure_compatible_formats.
    result["in_proj_all"] = Linear(tensors=fused_tensors, weight_format=first.weight_format)
    return result


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


# -----------------------------------------------------------------------
# Module-based commit helpers (new pipeline)
# -----------------------------------------------------------------------


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


def commit_linear_module(module, linear: Linear, name: str,
                         split_side: SplitSide | None = None,
                         split_num: int = 1, rank: int = 0,
                         copy: bool = False):
    """Commit a ``Linear`` bundle to a C++ ``Module`` handle for a specific TP rank.

    Unlike the legacy ``commit_linear`` which drives all GPUs via ``BaseOutputModel``,
    this function operates on a **single** module (one GPU) and writes only the
    shard corresponding to *rank*.

    Parameters
    ----------
    module : C++ Module handle
        Parent module (e.g. an ``AttentionWeight``).  ``module.get(name)``
        lazily creates the child ``LinearWeight``.
    linear : Linear
        The linear bundle to commit.
    name : str
        Child module name within *module* (e.g. ``"w_qkv"``).
    split_side : SplitSide | None
        TP split semantics.
    split_num : int
        Number of TP shards.
    rank : int
        Which shard to extract and copy.
    copy : bool
        If ``True``, copy the tensor as-is (no split).
    """
    linear_mod = module.get(name)

    split_dim = _SPLIT_SIDE_TO_DIM.get(split_side) if split_side else None

    # Block-scale TP split validation
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

    cpp_dtype, group_size = _infer_cpp_linear_dtype(linear)
    if group_size == 0:
        group_size = max(1, 128)  # default; caller should pass correct value

    packer = linear.weight_format.packer if linear.weight_format else None

    # Process "weight"/"qweight" first so that LinearWeight::alloc() triggers
    # do_allocate() before we encounter "scales"/"zeros"/"bias".  Without this,
    # lazy allocation returns an empty tensor for scales if it is iterated first.
    def _kind_order(item):
        k, _ = item
        if k in ("weight", "qweight"):
            return (0, k)
        return (1, k)

    for kind, tensor in sorted(linear.tensors.items(), key=_kind_order):
        if packer is not None:
            tensor = packer(tensor, kind)

        # Bias is NOT split for row-parallel (INPUT-split) linears — it is
        # replicated across all TP ranks and added after the all-reduce.
        tensor_split_dim = split_dim
        if kind == "bias" and split_side == SplitSide.INPUT:
            tensor_split_dim = None

        # Extract the shard for this rank
        if tensor_split_dim is not None and split_num > 1:
            split_size = tensor.shape[tensor_split_dim] // split_num
            shard = tensor.split(split_size, dim=tensor_split_dim)[rank]
        elif copy:
            shard = tensor
        else:
            shard = tensor

        shard = shard.cuda().contiguous()

        # Allocate (first call triggers full allocation) and copy
        dst = linear_mod.alloc(kind, cpp_dtype, group_size)
        if dst:
            shard = _cast_shard_for_tm(shard, dst)
            # Pad shard with zeros when C++ allocation is larger (e.g. due to
            # _pad_inter_size ensuring group_size alignment for TP splitting).
            # Compare byte sizes since Python packed dtype (int32) may differ
            # from C++ allocation dtype (e2m1, etc.) while having matching
            # byte size when dimensions align.
            if dst.byte_size != shard.nbytes and dst.byte_size > shard.nbytes:
                pad_dim = tensor_split_dim if tensor_split_dim is not None else -1
                if pad_dim < 0:
                    pad_dim = shard.dim() + pad_dim
                # Number of elements on the non-padded dimensions
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


def commit_tensor_module(module, tensor: torch.Tensor | None, name: str,
                         split_side: SplitSide | None = None,
                         split_num: int = 1, rank: int = 0,
                         copy: bool = False):
    """Commit a raw tensor to a C++ ``Module`` handle for a specific TP rank.

    Parameters
    ----------
    module : C++ Module handle
        Module that owns the parameter (e.g. a ``DecoderLayerWeight``).
    tensor : torch.Tensor | None
        The tensor data.  ``None`` is a no-op.
    name : str
        Parameter name within *module* (e.g. ``"weight"`` for a norm).
    split_side, split_num, rank, copy
        Same semantics as ``commit_linear_module``.
    """
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


# -----------------------------------------------------------------------
# TP split rules
# -----------------------------------------------------------------------

# TP split rules use SplitSide.OUTPUT (column-parallel) or
# SplitSide.INPUT (row-parallel).  Keys absent from the table are
# broadcast to all TP ranks (no split needed).
_ATTN_TP_RULES: dict[str, dict] = {
    "w_qkv":     dict(split_side=SplitSide.OUTPUT),  # column-parallel
    "wo":        dict(split_side=SplitSide.INPUT),   # row-parallel
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


# -----------------------------------------------------------------------
# TransformerV2
# -----------------------------------------------------------------------


class TransformerV2:
    """Composable-ops transformer that consumes ``ModelWeightSpec``.

    Unlike the legacy ``Transformer``, this class does not contain per-model
    logic.  All model-specific decisions (key mapping, QKV merge, MLA folding,
    RoPE permutation, zero-centered norms, etc.) live in the spec.
    """

    def __init__(self, model: BaseOutputModel):
        self.model = model
        cfg = model.model_config
        self.attn_tp = model.attn_tp_size
        self.mlp_tp = model.mlp_tp_size
        self.num_layer = cfg.num_layer
        self.vocab_size = cfg.vocab_size
        self.head_dim = cfg.size_per_head
        self.head_num = cfg.head_num
        self.permute_qk = getattr(model, "permute_qk", True)
        self.repeat_kv = getattr(model, "repeat_kv", 0)
        self.attn_output_gate = getattr(cfg, "attn_output_gate", False)
        self.group_size = max(1, cfg.group_size)
        rope_param = model.attention_config.rope_param
        self.rope_dim = rope_param.dim if rope_param else self.head_dim

    def __call__(self, layer: int, spec: ModelWeightSpec):
        if layer >= 0:
            if layer >= self.num_layer:
                return 0
            self._process_layer(layer, spec)
            return 1
        else:
            self._process_misc(spec)

    # -- per-layer ---------------------------------------------------------

    def _process_layer(self, layer: int, spec: ModelWeightSpec):
        # Provide TP and model params to spec for merge/fusion (idempotent).
        spec.configure(
            attn_tp=self.attn_tp,
            permute_qk=self.permute_qk,
            repeat_kv=self.repeat_kv,
            head_dim=self.head_dim,
            rope_dim=self.rope_dim,
            attn_output_gate=self.attn_output_gate,
            kv_head_num=self.model.model_config.kv_head_num,
        )

        for gpu_idx in range(self.model.gpu_count):
            root = self.model.root(gpu_idx)
            if root is None:
                break
            attn_tp_rank, mlp_tp_rank = self.model.tp_ranks(gpu_idx)
            layer_mod = root["layers"][layer]

            # Layer norms (broadcast, no TP split)
            commit_tensor_module(layer_mod["attention_norm"],
                                 spec.attn_norm(layer), "weight")
            commit_tensor_module(layer_mod["ffn_norm"],
                                 spec.ffn_norm(layer), "weight")

            # Attention linears (w_qkv already merged+permuted by base class)
            attn_mod = layer_mod["attention"]
            for name, lin in spec.attn_linears(layer).items():
                rule = _ATTN_TP_RULES.get(name, {})
                tp = self.attn_tp if "split_side" in rule else 1
                commit_linear_module(attn_mod, lin, name,
                                     split_num=tp, rank=attn_tp_rank, **rule)

            # Dense FFN linears (skip if no FFN for this layer, e.g. MoE-only models)
            ffn_linears = spec.ffn_linears(layer)
            if ffn_linears:
                ffn_mod = layer_mod["feed_forward"]
                for name, lin in ffn_linears.items():
                    rule = _FFN_TP_RULES.get(name, {})
                    tp = self.mlp_tp if "split_side" in rule else 1
                    commit_linear_module(ffn_mod, lin, name,
                                         split_num=tp, rank=mlp_tp_rank, **rule)

            # MoE experts
            if spec.num_experts(layer) > 0:
                moe_mod = layer_mod["moe_ffn"]
                for e in range(spec.num_experts(layer)):
                    expert_mod = moe_mod["experts"][e]
                    for name, lin in spec.moe_ffn_linears(layer, e).items():
                        rule = _FFN_TP_RULES.get(name, {})
                        tp = self.mlp_tp if "split_side" in rule else 1
                        commit_linear_module(expert_mod, lin, name,
                                             split_num=tp, rank=mlp_tp_rank, **rule)

            # Linear attention linears (in_proj_all already fused by base class)
            for name, lin in spec.linear_attn_linears(layer).items():
                rule = _LINEAR_ATTN_TP_RULES.get(name, {})
                tp = self.attn_tp if "split_side" in rule else 1
                linear_attn_mod = layer_mod["linear_attn"]
                commit_linear_module(linear_attn_mod, lin, name,
                                     split_num=tp, rank=attn_tp_rank, **rule)

            # All raw per-layer tensors (norms, gates, sinks, scalars, etc.)
            # Each entry is (tm_path, tensor, split_side) where tm_path is
            # like "attention.q_norm" or "moe_ffn.gate.weight" and split_side
            # is "output", "input", or None (broadcast).
            for tm_path, tensor, split_side in spec.raw_layer_tensors(layer):
                tp = self.attn_tp if split_side is not None else 1
                rank = attn_tp_rank
                # Split tm_path: last segment is the param name, everything
                # before is the module path within the layer.
                parts = tm_path.split(".")
                mod = layer_mod
                for seg in parts[:-1]:
                    mod = mod[seg]
                commit_tensor_module(mod, tensor, parts[-1],
                                     split_side=split_side,
                                     split_num=tp, rank=rank)

    # -- misc (embeddings, output head, final norm) ------------------------

    def _process_misc(self, spec: ModelWeightSpec):
        tp = self.attn_tp * self.model.attn_cp_size
        padded_vocab = ((self.vocab_size + tp - 1) // tp) * tp

        for gpu_idx in range(self.model.gpu_count):
            root = self.model.root(gpu_idx)
            if root is None:
                break
            attn_tp_rank, _mlp_tp_rank = self.model.tp_ranks(gpu_idx)

            # Token embeddings (column-parallel: split hidden dim)
            emb = spec.tok_embeddings()
            if emb is not None:
                emb_padded = pad_out_dim(emb, padded_vocab, dim=0)
                commit_tensor_module(root["tok_embeddings"], emb_padded, "weight",
                                     split_side=SplitSide.OUTPUT,
                                     split_num=tp, rank=attn_tp_rank)

            # Final norm (broadcast)
            norm = spec.norm_weight()
            commit_tensor_module(root["norm"], norm, "weight")

            # Output head (column-parallel, transposed)
            output = spec.output_weight()
            if output is not None:
                output_padded = pad_out_dim(output, padded_vocab, dim=0)
                output_t = output_padded.t()
                commit_tensor_module(root["output"], output_t, "weight",
                                     split_side=SplitSide.OUTPUT,
                                     split_num=tp, rank=attn_tp_rank)
