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
            raw = fuse_gdn_in_proj(raw, self._attn_tp)
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


def fuse_gdn_in_proj(la_linears: dict[str, Linear], tp: int) -> dict[str, Linear]:
    """Fuse GDN input projections into ``in_proj_all`` with TP interleaving.

    Pops ``in_proj_qkv``, ``in_proj_z``, ``in_proj_b``, ``in_proj_a`` from
    *la_linears* and inserts a single ``in_proj_all``.  Returns the updated
    dict (does not modify the input in place).

    For ``tp=1`` this reduces to a plain ``concat_out_dim``.
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
            shape = list(t.shape)
            new_shape = shape[:d] + [tp, shape[d] // tp] + shape[d + 1:]
            reshaped.append(t.reshape(new_shape))
        fused = torch.cat(reshaped, dim=d + 1)
        shape = list(fused.shape)
        final = shape[:d] + [shape[d] * shape[d + 1]] + shape[d + 2:]
        fused_tensors[kind] = fused.reshape(final)

    # All components share the same format after _ensure_compatible_formats.
    result["in_proj_all"] = Linear(tensors=fused_tensors, weight_format=first.weight_format)
    return result


_SPLIT_SIDE_TO_DIM: dict[SplitSide, int] = {SplitSide.OUTPUT: -1, SplitSide.INPUT: 0}


def commit_linear(model: BaseOutputModel, linear: Linear, name: str,
                  split_side: SplitSide | None = None, split_num: int = 1,
                  copy: bool = False):
    """Export every tensor in a ``Linear`` bundle via ``model.save_split``.

    All ``Linear`` objects are expected in TM layout ``[in, out]``.
    ``split_side`` controls TP partitioning:

    - ``SplitSide.OUTPUT`` — column-parallel: split along the output dimension
      (``dim=-1`` in TM layout, i.e. the output axis).
    - ``SplitSide.INPUT``  — row-parallel: split along the input dimension
      (``dim=0`` in TM layout, i.e. the input axis).
    - ``None``             — broadcast to all TP ranks (no split).

    Packing (e.g. uint8 → int32 for 4-bit weights) is handled by
    ``linear.weight_format.packer`` if present.

    For deferred-emplace models, this also triggers C++ ``allocate()`` so
    the weight tensors are created before ``copy_from``.
    """
    split_dim = _SPLIT_SIDE_TO_DIM.get(split_side) if split_side else None

    # For output-dimension TP splits, verify that block scales divide evenly.
    if split_side == SplitSide.OUTPUT and split_num > 1:
        wfmt = linear.weight_format
        if wfmt is not None and wfmt.block_out:
            for kind, tensor in linear.tensors.items():
                if kind in ("scales", "zeros"):
                    n_blocks = tensor.size(-1)
                    assert n_blocks % split_num == 0, (
                        f"TP split: {name}.{kind} has {n_blocks} output-dimension "
                        f"scale blocks (block_out={wfmt.block_out}), which is not "
                        f"divisible by split_num={split_num}.")

    cpp_dtype, group_size = None, 0
    if hasattr(model, 'model_comm') and model.model_comm is not None:
        cpp_dtype, group_size = _infer_cpp_linear_dtype(linear)
        if cpp_dtype is not None:
            if group_size == 0:
                group_size = max(1, model.model_config.group_size)
            is_qweight = "qweight" in linear.tensors
            weight_kind = "qweight" if is_qweight else "weight"
            if split_dim is not None or copy:
                for rank in range(split_num):
                    alloc_name = f"{name}.{rank}.{weight_kind}"
                    model.allocate_weight(alloc_name, cpp_dtype, group_size)
            else:
                alloc_name = f"{name}.{weight_kind}"
                model.allocate_weight(alloc_name, cpp_dtype, group_size)

    packer = linear.weight_format.packer if linear.weight_format else None
    for kind, tensor in linear.tensors.items():
        if packer is not None:
            tensor = packer(tensor, kind)
        model.save_split(tensor, f"{name}.{kind}",
                         split_dim=split_dim, split_num=split_num, copy=copy)


def commit_tensor(model: BaseOutputModel, tensor: torch.Tensor | None,
                  name: str, split_side: SplitSide | None = None,
                  split_num: int = 1, copy: bool = False):
    """Export a single raw tensor.

    ``split_side`` follows the same convention as ``commit_linear``:
    ``SplitSide.OUTPUT`` splits along the last axis, ``SplitSide.INPUT``
    along the first, and ``None`` broadcasts to all TP ranks.
    """
    if tensor is None:
        return
    split_dim = _SPLIT_SIDE_TO_DIM.get(split_side) if split_side else None
    if split_dim is not None or copy:
        model.save_split(tensor, name, split_dim=split_dim,
                         split_num=split_num, copy=copy)
    else:
        model.export_weight(tensor, name)


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

        # Layer norms (broadcast, no TP split)
        commit_tensor(self.model, spec.attn_norm(layer),
                      f"layers.{layer}.attention_norm.weight")
        commit_tensor(self.model, spec.ffn_norm(layer),
                      f"layers.{layer}.ffn_norm.weight")

        # Attention linears (w_qkv already merged+permuted by base class)
        for name, lin in spec.attn_linears(layer).items():
            rule = _ATTN_TP_RULES.get(name, {})
            tp = self.attn_tp if "split_side" in rule else 1
            commit_linear(self.model, lin,
                          f"layers.{layer}.attention.{name}",
                          split_num=tp, **rule)

        # Dense FFN linears
        for name, lin in spec.ffn_linears(layer).items():
            rule = _FFN_TP_RULES.get(name, {})
            tp = self.mlp_tp if "split_side" in rule else 1
            commit_linear(self.model, lin,
                          f"layers.{layer}.feed_forward.{name}",
                          split_num=tp, **rule)

        # MoE experts
        for e in range(spec.num_experts(layer)):
            for name, lin in spec.moe_ffn_linears(layer, e).items():
                rule = _FFN_TP_RULES.get(name, {})
                tp = self.mlp_tp if "split_side" in rule else 1
                commit_linear(self.model, lin,
                              f"layers.{layer}.moe_ffn.experts.{e}.{name}",
                              split_num=tp, **rule)

        # Linear attention linears (in_proj_all already fused by base class)
        for name, lin in spec.linear_attn_linears(layer).items():
            rule = _LINEAR_ATTN_TP_RULES.get(name, {})
            tp = self.attn_tp if "split_side" in rule else 1
            commit_linear(self.model, lin,
                          f"layers.{layer}.linear_attn.{name}",
                          split_num=tp, **rule)

        # All raw per-layer tensors (norms, gates, sinks, scalars, etc.)
        # Each entry is (tm_path, tensor, split_side) where split_side is
        # "output", "input", or None (broadcast).
        for tm_path, tensor, split_side in spec.raw_layer_tensors(layer):
            tp = self.attn_tp if split_side is not None else 1
            commit_tensor(self.model, tensor,
                          f"layers.{layer}.{tm_path}",
                          split_side=split_side, split_num=tp)

    # -- misc (embeddings, output head, final norm) ------------------------

    def _process_misc(self, spec: ModelWeightSpec):
        tp = self.attn_tp * self.model.attn_cp_size
        padded_vocab = ((self.vocab_size + tp - 1) // tp) * tp

        emb = spec.tok_embeddings()
        if emb is not None:
            emb = pad_out_dim(emb, padded_vocab, dim=0)
            self.model.save_split(emb, "tok_embeddings.weight",
                                  split_dim=1, split_num=tp)

        norm = spec.norm_weight()
        commit_tensor(self.model, norm, "norm.weight")

        output = spec.output_weight()
        if output is not None:
            output = pad_out_dim(output, padded_vocab, dim=0)
            output = output.t()
            self.model.save_split(output, "output.weight",
                                  split_dim=1, split_num=tp)
