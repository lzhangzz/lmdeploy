# Copyright (c) OpenMMLab. All rights reserved.
"""Builder base class and commit internals for spec-driven module loading.

Absorbs Distributor and the core commit logic from commit.py into a unified
Builder hierarchy.  Each Builder wraps N GPU handles for a single logical
module and distributes module creation, child binding, and weight commits
across all GPUs with bound TP configuration.
"""
from __future__ import annotations

import enum
import torch

import _turbomind as _tm

from .kind_map import TRIVIAL_FORMAT
from .linear import Linear, chunk_linears as _chunk_linears, interleave_linears as _interleave_linears
from .module_configs import make_norm_config

# ---------------------------------------------------------------------------
# SplitSide enum (internal -- not exposed to specs)
# ---------------------------------------------------------------------------


class SplitSide(enum.Enum):
    """Semantic TP split direction for commit operations.

    OUTPUT -- column-parallel: split along the output dimension (axis -1)
    INPUT  -- row-parallel:    split along the input dimension  (axis  0)
    """

    OUTPUT = "output"
    INPUT = "input"


# ---------------------------------------------------------------------------
# Canonical dtype mappings (moved from commit.py)
# ---------------------------------------------------------------------------

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

_FP8_DTYPES: set[torch.dtype] = {torch.uint8}
for _fp8_attr in ('float8_e4m3fn', 'float8_e5m2fn'):
    _fp8_dt = getattr(torch, _fp8_attr, None)
    if _fp8_dt is not None:
        _FP8_DTYPES.add(_fp8_dt)

_SPLIT_SIDE_TO_DIM: dict[SplitSide, int] = {SplitSide.OUTPUT: -1, SplitSide.INPUT: 0}


# ---------------------------------------------------------------------------
# Dtype / format helpers (moved from commit.py)
# ---------------------------------------------------------------------------


def _cpp_dtype(dtype_str: str):
    """Convert a model-config data_type string to C++ DataType enum."""
    return _STR_TO_DTYPE[dtype_str]


def _act_type_id(act_str: str) -> int:
    """Convert activation_type string to C++ ActivationType enum value."""
    return {'silu': 0, 'gpt-oss': 1}.get(act_str, 0)


def _torch_dtype_to_cpp(dtype: torch.dtype):
    """Convert a torch dtype to the C++ ``DataType`` enum, or ``None``."""
    return _TORCH_TO_CPP.get(dtype)


def _cast_shard_for_tm(shard: torch.Tensor, tm_tensor) -> torch.Tensor:
    """Cast *shard* dtype to match *tm_tensor*'s C++ dtype when needed."""
    if tm_tensor.type == _tm.DataType.TYPE_FP32 and shard.dtype in (torch.float16, torch.bfloat16):
        return shard.float()
    if tm_tensor.type == _tm.DataType.TYPE_FP16 and shard.dtype != torch.float16:
        return shard.half()
    if tm_tensor.type == _tm.DataType.TYPE_BF16 and shard.dtype != torch.bfloat16:
        return shard.to(torch.bfloat16)
    return shard


def _infer_cpp_linear_dtype(linear: Linear):
    """Determine C++ DataType and group_size from ``Linear.weight_format``."""
    fmt = linear.weight_format
    if fmt is not None and fmt.cpp_dtype_name is not None:
        cpp_dtype = getattr(_tm.DataType, fmt.cpp_dtype_name, None)
        if cpp_dtype is not None:
            return cpp_dtype, fmt.block_in or 0

    # Trivial (or missing format): dtype from weight tensor
    weight = linear.tensors.get("weight")
    if weight is not None:
        return _TORCH_TO_CPP.get(weight.dtype), 0
    return None, 0


def _infer_compute_dtype(linear: Linear):
    """Get the model's compute dtype from a Linear's tensors.

    For trivial formats the weight itself carries the compute dtype.
    For quantized formats we infer from scales or bias.
    """
    w = linear.tensors.get('weight')
    if w is not None:
        d = _TORCH_TO_CPP.get(w.dtype)
        if d is not None:
            return d
        # FP8 weights: compute dtype is BF16 (or FP16 depending on model),
        # not FP32.  Fall through to scales/bias only for non-FP8 dtypes.
        if w.dtype in _FP8_DTYPES:
            # FP8 stored as uint8 after normalization; prefer BF16.
            return _tm.DataType.TYPE_BF16
    for key in ('scales', 'bias'):
        t = linear.tensors.get(key)
        if t is not None:
            d = _TORCH_TO_CPP.get(t.dtype)
            if d is not None:
                return d
    return None


# ---------------------------------------------------------------------------
# TP split rules
# ---------------------------------------------------------------------------

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

_LINEAR_ATTN_TP_RULES: dict[str, dict] = {
    "in_proj_qkv": dict(split_side=SplitSide.OUTPUT),
    "in_proj_z":   dict(split_side=SplitSide.OUTPUT),
    "in_proj_b":   dict(split_side=SplitSide.OUTPUT),
    "in_proj_a":   dict(split_side=SplitSide.OUTPUT),
    "in_proj_all": dict(split_side=SplitSide.OUTPUT),
    "out_proj":    dict(split_side=SplitSide.INPUT),
}

_MLA_TP_RULES: dict[str, dict] = {
    "q_a_proj":  dict(split_side=SplitSide.OUTPUT),
    "q_b_proj":  dict(split_side=SplitSide.OUTPUT),
    "kv_a_proj": dict(split_side=SplitSide.OUTPUT),
    "wo":        dict(split_side=SplitSide.INPUT),
}


# ---------------------------------------------------------------------------
# Core tensor commit (moved from commit.py)
# ---------------------------------------------------------------------------


def _commit_tensors(handle, linear: Linear, cpp_dtype, group_size: int,
                    split_side: SplitSide | None, split_num: int, rank: int,
                    in_dim: int, out_dim: int,
                    model_dtype=None):
    """Commit tensor data from a ``Linear`` to a pre-created C++ LinearWeight handle.

    Handles packing, TP sharding, allocation, dtype casting, and padding.
    """
    split_dim = _SPLIT_SIDE_TO_DIM.get(split_side) if split_side else None

    packer = linear.weight_format.packer if linear.weight_format else None

    # Whether the weight format is quantized (packed).  For quantized formats
    # the weight shard has a different shape/dtype than the allocation, but
    # byte sizes match due to the packing invariant.
    fmt = linear.weight_format
    is_quantized = fmt is not None and fmt.name != 'trivial'

    for kind, tensor in linear.tensors.items():
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

        if not shard.is_cuda:
            shard = shard.cuda(0).contiguous()
        elif not shard.is_contiguous():
            shard = shard.contiguous()

        # Determine allocation shape and dtype.
        if kind == "weight" and is_quantized:
            # Quantized weight: allocate with model dimensions and weight
            # format dtype.  Byte sizes match the packed shard due to the
            # packing invariant (e.g. 8 uint4 values = 1 int32 = 4 bytes).
            alloc_shape = [in_dim, out_dim]
            alloc_dtype = cpp_dtype
        elif kind == "weight" and model_dtype is not None:
            # Trivial weight: use model compute dtype for dtype coercion.
            alloc_shape = list(shard.shape)
            alloc_dtype = model_dtype
        else:
            # Scales, zeros, bias: use shard's own shape and dtype.
            alloc_shape = list(shard.shape)
            alloc_dtype = _torch_dtype_to_cpp(shard.dtype)

        dst = handle.param(kind).alloc(alloc_shape, alloc_dtype)
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


# ---------------------------------------------------------------------------
# FFN fusion helpers (moved from transforms.py)
# ---------------------------------------------------------------------------


def _should_fuse_silu(w1_linear: Linear, act_type: str, is_moe: bool = False) -> bool:
    """Determine if fused SiLU (interleave) should be used for w1+w3 fusion.

    Gold standard condition (from GEMM kernel constraints — trust it):
        act_type == SiLU && (int4 || mxfp4 || fp8 || moe) && !(fp8 && SM90)
    """
    if act_type not in ('', 'silu', 'SiLU'):
        return False

    # Dense bf16/fp16 without MoE -> chunk, not interleave
    weight = w1_linear.tensors.get("weight")
    is_quantized = weight is not None and weight.element_size() < 2
    if not is_quantized and not is_moe:
        return False

    # FP8 on SM90 -> chunk
    fmt = w1_linear.weight_format
    if fmt is not None and fmt.name == "fp8":
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            if cap == (9, 0):
                return False

    return True


def _can_fuse_w1w3(w1: Linear, tp: int) -> bool:
    """Check whether w1+w3 fusion is safe for the given TP.

    Fusion (interleave or chunk) concatenates w1 and w3 along the output dim.
    For block-quantized formats (e.g. FP8 with block_out=128), the fused
    scale count ``2 * cdiv(N/tp, block_out)`` must equal
    ``cdiv(2*N/tp, block_out)``.  This holds iff ``(N/tp) % block_out == 0``.
    When it doesn't, the fused module's C++ allocation won't match the
    concatenated scales and we must commit w1/w3 separately.
    """
    if tp <= 1:
        return True
    fmt = w1.weight_format
    if fmt is None or fmt.block_out is None:
        return True
    w = w1.tensors.get("weight")
    if w is None:
        return True
    return (w.size(-1) // tp) % fmt.block_out == 0


def fuse_ffn_linears(
    w1: Linear,
    w3: Linear,
    tp: int,
    act_type: str,
    is_moe: bool = False,
) -> tuple[Linear | None, bool]:
    """Optionally fuse w1/w3 on full (unsharded) tensors for FFN.

    Returns (fused_w1w3_or_none, fused_silu).
    When fusion is possible, fused_w1w3 is set.
    When block-scale boundaries prevent fusion, returns (None, fused_silu).

    TP sharding is NOT done here — the caller's commit path handles it
    via split_side=SplitSide.OUTPUT.  ``tp`` is only used for the
    block-scale alignment check in ``_can_fuse_w1w3``.
    """
    fused_silu = _should_fuse_silu(w1, act_type, is_moe)
    can_fuse = _can_fuse_w1w3(w1, tp)

    if can_fuse:
        if fused_silu:
            w1w3 = _interleave_linears(w1, w3)
        else:
            w1w3 = _chunk_linears(w1, w3, tp)
        return (w1w3, fused_silu)
    else:
        return (None, fused_silu)


# ---------------------------------------------------------------------------
# Builder base class
# ---------------------------------------------------------------------------


class Builder:
    """Wraps N GPU handles for a single logical module.

    Distributes module creation, child binding, and weight commits
    across all GPUs with bound TP configuration.

    Subclasses specialize for particular module types (e.g. attention,
    FFN, MoE).
    """

    def __init__(self, config, contexts, tp=1, ranks=None):
        """Create C++ modules via ``_tm.create_module`` on each GPU context.

        Parameters
        ----------
        config : module_configs dataclass
            Config with ``to_cpp()`` method and optionally ``for_rank(rank)``.
        contexts : list
            GPU context managers (one per GPU).
        tp : int
            Tensor parallelism degree.
        ranks : list[int] | None
            Per-GPU TP ranks.
        """
        # Use object.__setattr__ to avoid triggering our custom __setattr__
        object.__setattr__(self, '_contexts', contexts)
        object.__setattr__(self, '_tp', tp)
        object.__setattr__(self, '_ranks', ranks)
        object.__setattr__(self, '_children', {})
        object.__setattr__(self, 'config', config)

        object.__setattr__(self, '_handles', None)
        object.__setattr__(self, '_handles_created', False)

    # ------------------------------------------------------------------
    # Child binding via attribute / item assignment
    # ------------------------------------------------------------------

    def __setattr__(self, name: str, value):
        """If *value* is a Builder, bind its handles as named children."""
        if isinstance(value, Builder):
            self._ensure_handles()
            value._ensure_handles()
            for i, (parent_h, child_h) in enumerate(
                    zip(self._handles, value._handles)):
                with self._contexts[i]:
                    parent_h.add_child_raw(name, child_h)
            self._children[name] = value
        else:
            object.__setattr__(self, name, value)

    def __setitem__(self, index: int, value):
        """Bind a Builder as an indexed child (for ModuleList children)."""
        name = str(index)
        if isinstance(value, Builder):
            self._ensure_handles()
            value._ensure_handles()
            for i, (parent_h, child_h) in enumerate(
                    zip(self._handles, value._handles)):
                with self._contexts[i]:
                    parent_h.add_child_raw(name, child_h)
            self._children[name] = value

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def tp_size(self):
        return self._tp

    def _rank_for(self, gpu_idx: int) -> int:
        if self._ranks and self._tp > 1:
            return self._ranks[gpu_idx]
        return 0

    def _ensure_handles(self):
        """Lazily create C++ module handles on first access."""
        if self._handles_created:
            return
        handles = []
        for i, ctx in enumerate(self._contexts):
            with ctx:
                if self._tp > 1:
                    cfg = self.config.clone()
                    cfg.tp_rank = self._ranks[i]
                else:
                    cfg = self.config
                handle = _tm.create_module(cfg)
                handles.append(handle)
        object.__setattr__(self, '_handles', handles)
        object.__setattr__(self, '_handles_created', True)

    # ------------------------------------------------------------------
    # Commit methods (distributed across all GPUs)
    # ------------------------------------------------------------------

    def _commit_linear(self, name: str, linear: Linear,
                       split_side: SplitSide | None = None,
                       model_dtype=None):
        """Commit a ``Linear`` bundle to a named child on all GPUs.

        Creates the LinearWeight child on first call (deferred creation),
        attaches DataFormat, validates block-scale TP splits, then commits
        tensor data.

        This mirrors the logic in ``commit.commit_linear()`` exactly,
        including deferred LinearWeight creation, DataFormat attachment,
        block-scale TP validation, and padding logic.

        Parameters
        ----------
        name : str
            Child module name (e.g. ``"w_qkv"``).
        linear : Linear
            The linear bundle to commit.
        split_side : SplitSide | None
            TP split semantics.  ``None`` means broadcast (no split).
        model_dtype : C++ DataType value | None
            The model's configured compute dtype.  When set, trivial
            (non-quantized) weights use this dtype instead of the weight
            tensor's dtype.  This prevents dtype mismatches when the
            checkpoint stores weights in a different precision than the
            model config (e.g. BF16 weights in an FP16 model).
        """
        self._ensure_handles()
        cpp_dtype, group_size = _infer_cpp_linear_dtype(linear)
        if group_size == 0:
            group_size = max(1, 128)  # default; caller should pass correct value

        # Ensure the Linear has a DataFormat attached (deferred creation for
        # formats like AWQ/GPTQ where group_size is not known at build_linear
        # time).
        if linear.data_format is None and linear.weight_format is not None:
            linear = Linear(tensors=linear.tensors,
                            weight_format=linear.weight_format,
                            data_format=linear.weight_format.to_data_format(
                                cpp_dtype if cpp_dtype else _tm.DataType.TYPE_INVALID,
                                group_size))

        tp = self._tp if split_side else 1

        # Ensure the LinearWeight child exists on each GPU handle.
        # We check only the first handle; if it doesn't exist, create on all.
        # The child is created once; subsequent calls for the same name reuse it.
        for i, handle in enumerate(self._handles):
            with self._contexts[i]:
                rank = self._rank_for(i) if tp > 1 else 0

                linear_mod = handle.child(name)

                if linear_mod is None:
                    w = linear.tensors.get('weight')
                    if w is None:
                        return
                    in_dim = w.shape[0]
                    out_dim = w.shape[-1]
                    if split_side == SplitSide.OUTPUT:
                        out_dim = out_dim // tp
                    elif split_side == SplitSide.INPUT:
                        in_dim = in_dim // tp
                    compute_dtype = _infer_compute_dtype(linear)
                    # Always prefer the model's configured compute dtype.  For
                    # quantized formats, the scales/bias dtype may differ from
                    # the model's actual compute dtype (e.g. AWQ scales stored
                    # as bf16 in an fp16 model), which would cause an
                    # input_dtype mismatch at GEMM time.
                    if model_dtype is not None:
                        compute_dtype = model_dtype
                    lin_cfg = _tm.LinearConfig()
                    lin_cfg.input_dim = in_dim
                    lin_cfg.output_dim = out_dim
                    lin_cfg.data_type = compute_dtype if compute_dtype else _tm.DataType.TYPE_INVALID
                    lin_cfg.has_bias = 'bias' in linear.tensors
                    linear_mod = handle.create_child(name, lin_cfg)

                # Block-scale TP split validation
                if split_side == SplitSide.OUTPUT and tp > 1:
                    wfmt = linear.weight_format
                    if wfmt is not None and wfmt.block_out:
                        for kind, tensor in linear.tensors.items():
                            if kind in ("scales", "zeros"):
                                n_blocks = tensor.size(-1)
                                assert n_blocks % tp == 0, (
                                    f"TP split: {name}.{kind} has {n_blocks} "
                                    f"output-dimension scale blocks "
                                    f"(block_out={wfmt.block_out}), not "
                                    f"divisible by split_num={tp}.")

                linear_mod.set_weight_spec(cpp_dtype, group_size)

                # Get model dimensions for correct weight allocation shape
                w = linear.tensors.get('weight')
                in_dim = w.shape[0] if w is not None else 0
                out_dim = w.shape[-1] if w is not None else 0
                if split_side == SplitSide.OUTPUT:
                    out_dim = out_dim // tp
                elif split_side == SplitSide.INPUT:
                    in_dim = in_dim // tp

                _commit_tensors(linear_mod, linear, cpp_dtype, group_size,
                                split_side, tp, rank, in_dim, out_dim,
                                model_dtype=model_dtype)

    def _commit_tensor(self, name: str, tensor: torch.Tensor | None,
                       split_side: SplitSide | None = None):
        """Commit a raw tensor to a named parameter on all GPUs.

        Parameters
        ----------
        name : str
            Parameter name within the module.
        tensor : torch.Tensor | None
            The tensor data.  ``None`` is a no-op.
        split_side : SplitSide | None
            TP split semantics.  ``None`` means broadcast.
        """
        self._ensure_handles()
        if tensor is None:
            return

        tp = self._tp if split_side else 1
        split_dim = _SPLIT_SIDE_TO_DIM.get(split_side) if split_side else None

        for i, handle in enumerate(self._handles):
            with self._contexts[i]:
                rank = self._rank_for(i) if tp > 1 else 0

                if split_dim is not None and tp > 1:
                    split_size = tensor.shape[split_dim] // tp
                    shard = tensor.split(split_size, dim=split_dim)[rank]
                else:
                    shard = tensor

                if not shard.is_cuda:
                    shard = shard.cuda(0).contiguous()
                elif not shard.is_contiguous():
                    shard = shard.contiguous()

                cpp_dtype = _torch_dtype_to_cpp(shard.dtype)
                dst = handle.param(name).alloc(list(shard.shape), cpp_dtype)
                shard = _cast_shard_for_tm(shard, dst)
                dst.copy_from(shard)

    def _add_norm_child(self, name: str, tensor: torch.Tensor,
                        data_type=None):
        """Create a NormConfig child and commit weight tensor.

        Parameters
        ----------
        name : str
            Child module name (e.g. ``"attention_norm"``).
        tensor : torch.Tensor
            The norm weight tensor.
        data_type : C++ DataType value | None
            Compute dtype for the norm.  Defaults to FP32 if not set.
        """
        self._ensure_handles()
        if data_type is None:
            data_type = _tm.DataType.TYPE_FP32
        norm_cfg = make_norm_config(dim=tensor.shape[-1], data_type=data_type)

        for i, handle in enumerate(self._handles):
            with self._contexts[i]:
                child = handle.create_child(name, norm_cfg)
                shard = tensor
                if not shard.is_cuda:
                    shard = shard.cuda(0).contiguous()
                elif not shard.is_contiguous():
                    shard = shard.contiguous()
                cpp_dtype = _torch_dtype_to_cpp(shard.dtype)
                dst = child.param('weight').alloc(list(shard.shape), cpp_dtype)
                shard = _cast_shard_for_tm(shard, dst)
                dst.copy_from(shard)


# ---------------------------------------------------------------------------
# TextModelBuilder -- wraps pre-existing root handles
# ---------------------------------------------------------------------------


class TextModelBuilder(Builder):
    """Special case Builder that wraps pre-existing root C++ module handles.

    Unlike regular Builders, ``TextModelBuilder`` does NOT create new modules
    in ``__init__``.  The root handles already exist (created by the
    BaseOutputModel / TurboMind runtime).
    """

    def __init__(self, handles, contexts, tp=1, ranks=None):
        # Bypass Builder.__init__ which calls _tm.create_module
        object.__setattr__(self, '_handles', handles)
        object.__setattr__(self, '_contexts', contexts)
        object.__setattr__(self, '_tp', tp)
        object.__setattr__(self, '_ranks', ranks)
        object.__setattr__(self, '_children', {})
        object.__setattr__(self, '_handles_created', True)
        object.__setattr__(self, 'config', None)


# ---------------------------------------------------------------------------
# DecoderLayerBuilder -- pure container
# ---------------------------------------------------------------------------


class DecoderLayerBuilder(Builder):
    """Pure container builder for decoder layers.

    No extra methods beyond the Builder base class.
    """
    pass


# ---------------------------------------------------------------------------
# ModuleListBuilder -- indexed children via __setitem__
# ---------------------------------------------------------------------------


class ModuleListBuilder(Builder):
    """Builder for ModuleList containers.

    Uses ``__setitem__`` from the base class to bind indexed children.
    """
    pass


# ---------------------------------------------------------------------------
# NormBuilder -- single norm weight
# ---------------------------------------------------------------------------


class NormBuilder(Builder):
    """Builder for a single norm weight module."""

    def set_weight(self, tensor: torch.Tensor):
        """Commit the norm weight tensor to all GPU handles."""
        self._ensure_handles()
        if tensor is None:
            return
        for i, handle in enumerate(self._handles):
            with self._contexts[i]:
                shard = tensor
                if not shard.is_cuda:
                    shard = shard.cuda(0).contiguous()
                elif not shard.is_contiguous():
                    shard = shard.contiguous()
                cpp_dtype = _torch_dtype_to_cpp(shard.dtype)
                dst = handle.param('weight').alloc(list(shard.shape), cpp_dtype)
                shard = _cast_shard_for_tm(shard, dst)
                dst.copy_from(shard)


# ---------------------------------------------------------------------------
# LinearBuilder -- standalone linear layers (embeddings, lm_head)
# ---------------------------------------------------------------------------


class LinearBuilder(Builder):
    """Builder for standalone linear layers (embeddings, lm_head).

    Wraps a C++ LinearWeight module. Use ``set_weight()`` to commit
    the weight tensor.
    """

    def set_weight(self, tensor: torch.Tensor, split_side=None):
        """Commit the weight tensor to all GPU handles.

        Parameters
        ----------
        tensor : torch.Tensor
            The weight tensor (already padded/transposed by the spec).
        split_side : SplitSide | None
            TP split semantics. None means broadcast.
        """
        self._commit_tensor('weight', tensor, split_side)


# ---------------------------------------------------------------------------
# QKV merge / RoPE permutation / GDN fusion helpers
# ---------------------------------------------------------------------------


def _reorder_rotary_emb(x: torch.Tensor, head_dim: int, rope_dim: int):
    """Interleave rotary embedding layout for TurboMind's RoPE kernel.

    Combines the former ``permute_v2`` (full permutation when
    ``rope_dim == head_dim``) and ``permute_v2_partial`` (partial
    permutation when ``rope_dim < head_dim``) into a single function.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor whose last dimension is ``head_num * head_dim``.
    head_dim : int
        Full head dimension.
    rope_dim : int
        Rotary embedding dimension (``<= head_dim``).
    """
    assert x.size(-1) > 1
    assert rope_dim % 2 == 0, f'rope_dim must be even, got {rope_dim}'
    assert rope_dim <= head_dim, f'rope_dim ({rope_dim}) must be <= head_dim ({head_dim})'
    output_dims = x.size(-1)
    assert output_dims % head_dim == 0, (f'output_dims ({output_dims}) must be divisible by '
                                          f'head_dim ({head_dim})')
    head_num = output_dims // head_dim
    orig_shape = x.shape
    if x.dim() == 1:
        x = x.unsqueeze(0)

    x = x.view(x.size(0), head_num, head_dim)

    if rope_dim < head_dim:
        # Partial permutation: only interleave the rotary portion
        rotary = x[:, :, :rope_dim]
        passthrough = x[:, :, rope_dim:]
        rotary = rotary.view(x.size(0), head_num, 2, rope_dim // 2).transpose(2, 3).contiguous()
        rotary = rotary.view(x.size(0), head_num, rope_dim)
        x = torch.cat([rotary, passthrough], dim=-1)
    else:
        # Full permutation: interleave all elements
        x = x.view(x.size(0), head_num, 2, head_dim // 2).transpose(2, 3).contiguous()
        x = x.view(x.size(0), head_num, head_dim)

    return x.reshape(orig_shape)


def _merge_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tp: int):
    """Merge Q, K, V with TP interleaving.

    Contract: x.size(-1) is output dims.
    """
    def reshape(x):
        return x.view(x.size(0), tp, -1) if q.dim() == 2 else x.view(tp, -1)

    qkv = torch.cat(tuple(map(reshape, (q, k, v))), dim=-1)
    qkv = qkv.view(-1, qkv.size(-1) * tp)
    if q.dim() == 1:
        qkv.squeeze_()
    return qkv


def _merge_qkvg(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                gate: torch.Tensor, tp: int):
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


def _dequant_linear(linear: Linear) -> Linear:
    """Dequantize a quantized Linear to trivial when the format provides ``dequant``."""
    fmt = linear.weight_format
    if fmt is None or fmt.dequant is None:
        return linear
    new_tensors = fmt.dequant(linear.tensors)
    return Linear(tensors=new_tensors, weight_format=TRIVIAL_FORMAT, data_format=None)


def _ensure_compatible_formats(linears: dict[str, Linear]) -> dict[str, Linear]:
    """Dequant linears to a common trivial format if a fusion group has mixed formats."""
    formats = {name: lin.weight_format.name for name, lin in linears.items()}
    if len(set(formats.values())) <= 1:
        return linears
    return {name: _dequant_linear(lin) for name, lin in linears.items()}


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

    If any condition fails, the caller should dequantise to trivial first.
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
    ``_block_ops_need_dequant``), the entire group is dequantised to trivial
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
            f"dequantising to trivial.")
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
            qt = _reorder_rotary_emb(qt, head_dim, rope_dim)
            kt = _reorder_rotary_emb(kt, head_dim, rope_dim)

        if gate is not None:
            merged_tensors[kind] = _merge_qkvg(qt, kt, vt, gate, tp)
        else:
            merged_tensors[kind] = _merge_qkv(qt, kt, vt, tp)

    # All three linears share the same format after _ensure_compatible_formats.
    return Linear(tensors=merged_tensors, weight_format=q.weight_format,
                  data_format=q.data_format)


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

        result["in_proj_all"] = Linear(tensors=fused_tensors, weight_format=first.weight_format,
                                       data_format=first.data_format)
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
    result["in_proj_all"] = Linear(tensors=fused_tensors, weight_format=first.weight_format,
                                   data_format=first.data_format)
    return result


# ---------------------------------------------------------------------------
# AttentionBuilder -- QKV fusion, O-proj, QK-norm, direct params
# ---------------------------------------------------------------------------


class AttentionBuilder(Builder):
    """Attention weight loading builder."""

    _PARAM_TP_RULES: dict[str, SplitSide] = {
        'sinks': SplitSide.OUTPUT,
    }

    def add_qkv_proj(self, q, k, v):
        """Fuse Q/K/V into a single w_qkv, apply RoPE + TP interleave, commit."""
        merged = merge_qkv_linear(
            q, k, v,
            tp=self._tp,
            head_dim=self.config.head_dim,
            rope_dim=self.config.rope_dim or self.config.head_dim,
            permute_qk=True,
            attn_output_gate=self.config.attn_output_gate,
            repeat_kv=self.config.repeat_kv,
            kv_head_num=self.config.kv_head_num,
        )
        self._commit_linear('w_qkv', merged, SplitSide.OUTPUT,
                            model_dtype=self.config.data_type)

    def add_o_proj(self, o):
        """Shard along input dim, commit."""
        self._commit_linear('wo', o, SplitSide.INPUT,
                            model_dtype=self.config.data_type)

    def add_linear(self, name, linear):
        """Commit a named attention linear using TP rules.

        Looks up ``_ATTN_TP_RULES`` for the split side; absent keys are
        broadcast (no TP split).  Used for MLA projections (q_b_proj,
        kv_b_proj, o_proj) and other non-QKV attention linears.
        """
        rule = _ATTN_TP_RULES.get(name, {})
        split_side = rule.get('split_side')
        self._commit_linear(name, linear, split_side=split_side,
                            model_dtype=self.config.data_type)

    def add_qk_norm(self, q, k):
        """Create NormConfig children for q_norm, k_norm, commit tensors."""
        if q is not None:
            self._add_norm_child('q_norm', q, data_type=self.config.data_type)
        if k is not None:
            self._add_norm_child('k_norm', k, data_type=self.config.data_type)

    def add_param(self, name, tensor):
        """Commit a direct parameter. Builder determines split side."""
        split_side = self._PARAM_TP_RULES.get(name)
        self._commit_tensor(name, tensor, split_side)


# ---------------------------------------------------------------------------
# FfnBuilder -- w1+w3 fusion, w2 commit
# ---------------------------------------------------------------------------


class FfnBuilder(Builder):
    """FFN weight loading builder with w1+w3 fusion."""

    def add_ffn(self, w1, w2, w3):
        """Fuse w1+w3 if possible, update config, then shard and commit.

        The fusion result determines ``fuse_silu`` on the C++ module config.
        Updating ``self.config.fuse_silu`` **before** any ``_commit_linear``
        call ensures the C++ module is lazily created with the correct flag.
        """
        fused = None
        fused_silu = False
        if w1 is not None and w3 is not None:
            act_type = getattr(self.config, 'act_type', 0)
            # act_type is an int in FfnConfig, convert to string for transform
            if isinstance(act_type, int):
                act_type = {0: 'silu', 1: 'gpt-oss'}.get(act_type, 'silu')
            fused, fused_silu = fuse_ffn_linears(
                w1, w3, self._tp, act_type,
                is_moe=getattr(self.config, 'fused_moe', False))

        # Update config BEFORE first _commit_linear triggers _ensure_handles()
        self.config.fuse_silu = fused_silu

        model_dtype = self.config.data_type
        if fused is not None:
            self._commit_linear('w1w3', fused, SplitSide.OUTPUT,
                                model_dtype=model_dtype)
        else:
            if w1 is not None:
                self._commit_linear('w1', w1, SplitSide.OUTPUT,
                                    model_dtype=model_dtype)
            if w3 is not None:
                self._commit_linear('w3', w3, SplitSide.OUTPUT,
                                    model_dtype=model_dtype)
        if w2 is not None:
            self._commit_linear('w2', w2, SplitSide.INPUT,
                                model_dtype=model_dtype)


# ---------------------------------------------------------------------------
# MoeBuilder -- gate, non-expert params
# ---------------------------------------------------------------------------


class MoeBuilder(Builder):
    """MoE weight loading builder."""

    def add_gate(self, name, linear, model_dtype=None):
        """Commit a gate linear (broadcast, no split)."""
        self._commit_linear(name, linear, split_side=None,
                            model_dtype=model_dtype)

    def add_param(self, name, tensor, split_side=None):
        """Commit a non-expert MoE parameter."""
        if split_side is not None and not isinstance(split_side, SplitSide):
            split_side = None  # specs may pass None for broadcast
        self._commit_tensor(name, tensor, split_side)


# ---------------------------------------------------------------------------
# DeltaNetBuilder -- Gated Delta Net input projections, scalar params, conv1d
# ---------------------------------------------------------------------------


class DeltaNetBuilder(Builder):
    """DeltaNet (Gated Delta Net) weight loading builder."""

    def add_input_projections(self, *, in_proj_qkv=None, in_proj_z=None,
                              in_proj_b=None, in_proj_a=None, out_proj=None,
                              qkv_split=None):
        """Fuse GDN input projections, commit all linears with TP rules.

        Internally calls ``fuse_gdn_in_proj`` to merge qkv/z/b/a into a
        single ``in_proj_all`` with TP interleaving.  Commits each resulting
        linear using ``_LINEAR_ATTN_TP_RULES`` for split-side lookup.
        """
        linears = {}
        if in_proj_qkv is not None:
            linears["in_proj_qkv"] = in_proj_qkv
        if in_proj_z is not None:
            linears["in_proj_z"] = in_proj_z
        if in_proj_b is not None:
            linears["in_proj_b"] = in_proj_b
        if in_proj_a is not None:
            linears["in_proj_a"] = in_proj_a
        if out_proj is not None:
            linears["out_proj"] = out_proj

        linears = fuse_gdn_in_proj(linears, self._tp, qkv_split)

        model_dtype = self.config.data_type
        for name, lin in linears.items():
            rule = _LINEAR_ATTN_TP_RULES.get(name, {})
            split_side = rule.get('split_side')
            self._commit_linear(name, lin, split_side=split_side,
                                model_dtype=model_dtype)

    def add_scalar_params(self, a_log=None, dt_bias=None):
        """Commit A_log and dt_bias as OUTPUT-split tensors."""
        if a_log is not None:
            self._commit_tensor("A_log", a_log, split_side=SplitSide.OUTPUT)
        if dt_bias is not None:
            self._commit_tensor("dt_bias", dt_bias, split_side=SplitSide.OUTPUT)

    def add_conv1d(self, conv1d, qkv_split=None):
        """Transpose HF layout to TM layout, TP-reshape if needed, commit.

        HF stores conv1d as [conv_dim, d_conv]; TM kernel expects
        [d_conv, conv_dim].  When tp > 1 and *qkv_split* is provided,
        the Q/K/V sub-dims are TP-interleaved.
        """
        if conv1d is None:
            return
        # Squeeze leading singleton dim if present
        if conv1d.ndim == 3 and conv1d.shape[1] == 1:
            conv1d = conv1d.squeeze(1)
        # Transpose: HF [conv_dim, d_conv] -> TM [d_conv, conv_dim]
        conv1d = conv1d.t().contiguous()
        # TP Q/K/V interleaving
        if self._tp > 1 and qkv_split is not None:
            q_dim, k_dim, v_dim = qkv_split
            d_conv = conv1d.shape[0]
            tp = self._tp
            q_part = conv1d[:, :q_dim]
            k_part = conv1d[:, q_dim:q_dim + k_dim]
            v_part = conv1d[:, q_dim + k_dim:]
            conv1d = torch.cat([
                q_part.reshape(d_conv, tp, q_dim // tp),
                k_part.reshape(d_conv, tp, k_dim // tp),
                v_part.reshape(d_conv, tp, v_dim // tp),
            ], dim=2).reshape(d_conv, -1).contiguous()
        self._commit_tensor("conv1d", conv1d, split_side=SplitSide.OUTPUT)

    def add_norm(self, norm_weight, data_type):
        """Add inline norm child."""
        self._add_norm_child("norm", norm_weight, data_type=data_type)


# ---------------------------------------------------------------------------
# MLABuilder -- MLA projections, fold+pad, norms
# ---------------------------------------------------------------------------


class MLABuilder(Builder):
    """MLA (Multi-head Latent Attention) weight loading builder."""

    def add_projections(self, *, q_a_proj, q_b_proj, kv_a_proj, kv_b_proj,
                        wo):
        """Apply MLA fold+pad, then commit each projection.

        The fold consumes kv_b_proj — its information is absorbed into
        q_b_proj and wo.  After the fold, kv_b_proj is not committed.
        """
        linears = {
            "q_a_proj": q_a_proj,
            "q_b_proj": q_b_proj,
            "kv_a_proj": kv_a_proj,
            "wo": wo,
        }
        if kv_b_proj is not None:
            linears["kv_b_proj"] = kv_b_proj

        self._fold_and_pad(linears)

        model_dtype = self.config.data_type
        for name, lin in linears.items():
            if lin is None:
                continue
            rule = _MLA_TP_RULES.get(name, {})
            split_side = rule.get('split_side')
            self._commit_linear(name, lin, split_side=split_side,
                                model_dtype=model_dtype)

    def add_norms(self, *, q_a_norm, kv_a_norm, data_type=None):
        """Create norm children for q_a_layernorm and kv_a_layernorm."""
        if q_a_norm is not None:
            self._add_norm_child('q_a_layernorm', q_a_norm,
                                 data_type=data_type)
        if kv_a_norm is not None:
            self._add_norm_child('kv_a_layernorm', kv_a_norm,
                                 data_type=data_type)

    # ------------------------------------------------------------------
    # MLA fold+pad (moved from glm4_moe_lite_spec)
    # ------------------------------------------------------------------

    def _fold_and_pad(self, linears: dict[str, Linear]):
        """Fold kv_b_proj into q_b_proj and wo, then pad wo.

        Weight tensors are temporarily transposed to HF layout [out, in]
        for the fold arithmetic, then transposed back to TM layout [in, out].
        """
        # Temporarily convert weight tensors from TM [in, out] to HF [out, in].
        for lin in linears.values():
            for k in list(lin.tensors.keys()):
                t = lin.tensors[k]
                if t.dim() >= 2:
                    lin.tensors[k] = t.t().contiguous()
        try:
            self._fold_and_pad_hf(linears)
        finally:
            # Convert weight tensors back from HF [out, in] to TM [in, out].
            for lin in linears.values():
                for k in list(lin.tensors.keys()):
                    t = lin.tensors[k]
                    if t.dim() >= 2:
                        lin.tensors[k] = t.t().contiguous()

    def _fold_and_pad_hf(self, linears: dict[str, Linear]):
        """Inner fold logic; expects all weight tensors in HF layout [out, in]."""
        cfg = self.config
        head_num = cfg.head_num
        qk_rope_dim = cfg.qk_rope_dim
        qk_nope_dim = cfg.qk_nope_dim
        kv_lora_rank = cfg.kv_lora_rank
        v_head_dim = cfg.v_head_dim
        size_per_head = cfg.head_dim

        q_b_lin = linears.get("q_b_proj")
        kv_b_lin = linears.pop("kv_b_proj", None)
        o_lin = linears.get("wo")

        if q_b_lin is not None and kv_b_lin is not None and o_lin is not None:
            q_b = q_b_lin.tensors.get("weight")
            kv_b = kv_b_lin.tensors.get("weight")
            o = o_lin.tensors.get("weight")

            if (q_b is not None and kv_b is not None and o is not None
                    and torch.is_floating_point(q_b)
                    and torch.is_floating_point(kv_b)):
                orig_q_head_dim = q_b.size(0) // head_num
                orig_qk_nope_dim = orig_q_head_dim - qk_rope_dim
                orig_v_head_dim = o.size(1) // head_num
                target_nope_dim = size_per_head - qk_rope_dim

                if (orig_qk_nope_dim != target_nope_dim
                        or orig_v_head_dim != v_head_dim):
                    # Split kv_b into kc and vc
                    kv_b_per_head = kv_b.reshape(
                        head_num, orig_qk_nope_dim + orig_v_head_dim,
                        kv_lora_rank)
                    kc_w = kv_b_per_head[:, :orig_qk_nope_dim, :]
                    vc_w = kv_b_per_head[:, orig_qk_nope_dim:, :]

                    # Fold kc into q_b_proj
                    q_b_per_head = q_b.reshape(
                        head_num, orig_q_head_dim, q_b.size(1))
                    q_nope_w = q_b_per_head[:, :orig_qk_nope_dim, :]
                    q_rope_w = q_b_per_head[:, orig_qk_nope_dim:, :]
                    q_nope_expanded = torch.bmm(
                        kc_w.transpose(1, 2), q_nope_w)
                    q_b_folded = torch.cat(
                        [q_nope_expanded, q_rope_w], dim=1)
                    q_b_lin.tensors["weight"] = q_b_folded.reshape(
                        head_num * size_per_head, q_b.size(1))

                    # Fold vc into o_proj
                    o_per_head = o.reshape(
                        o.size(0), head_num, orig_v_head_dim)
                    o_folded = torch.bmm(
                        o_per_head.permute(1, 0, 2), vc_w)
                    o_lin.tensors["weight"] = o_folded.permute(
                        1, 0, 2).reshape(
                            o.size(0), head_num * kv_lora_rank)

        # Pad wo from [hidden, head_num*v_head_dim]
        #           to [hidden, head_num*size_per_head]
        if o_lin is not None:
            o_w = o_lin.tensors["weight"]
            cur_v = o_w.size(1) // head_num
            if cur_v < size_per_head:
                o_w = o_w.reshape(o_w.size(0), head_num, cur_v)
                o_w = torch.nn.functional.pad(
                    o_w, (size_per_head - cur_v, 0, 0, 0, 0, 0))
                o_lin.tensors["weight"] = o_w.reshape(
                    o_w.size(0), head_num * size_per_head)
