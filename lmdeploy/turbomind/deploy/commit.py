# Copyright (c) OpenMMLab. All rights reserved.
"""Shard and commit: allocate C++ params, copy tensor data, apply TP splits."""
from __future__ import annotations

import torch

import _turbomind as _tm

from .linear import Linear
from .builder import SplitSide

# Canonical dtype mappings
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


def _cpp_dtype(dtype_str: str):
    """Convert a model-config data_type string to C++ DataType enum."""
    return _STR_TO_DTYPE[dtype_str]


def _act_type_id(act_str: str) -> int:
    """Convert activation_type string to C++ ActivationType enum value."""
    return {'silu': 0, 'gpt-oss': 1}.get(act_str, 0)


_SPLIT_SIDE_TO_DIM: dict[SplitSide, int] = {SplitSide.OUTPUT: -1, SplitSide.INPUT: 0}


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


def commit_linear(module, linear: Linear, name: str,
                         split_side: SplitSide | None = None,
                         split_num: int = 1, rank: int = 0,
                         model_dtype=None):
    """Commit a ``Linear`` bundle to a C++ ``Module`` handle for a specific TP rank.

    Unlike the legacy ``commit_linear`` which drives all GPUs via ``BaseOutputModel``,
    this function operates on a **single** module (one GPU) and writes only the
    shard corresponding to *rank*.

    Parameters
    ----------
    module : C++ Module handle
        Parent module (e.g. an ``AttentionWeight``).
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
    model_dtype : int | None
        The model's configured compute dtype (C++ DataType value).  When set,
        trivial (non-quantized) weights use this dtype instead of the weight
        tensor's dtype.  This prevents dtype mismatches when the checkpoint
        stores weights in a different precision than the model config (e.g.
        BF16 weights in an FP16 model).
    """
    cpp_dtype, group_size = _infer_cpp_linear_dtype(linear)
    if group_size == 0:
        group_size = max(1, 128)  # default; caller should pass correct value

    # Ensure the Linear has a DataFormat attached (deferred creation for formats
    # like AWQ/GPTQ where group_size is not known at build_linear time).
    if linear.data_format is None and linear.weight_format is not None:
        linear = Linear(tensors=linear.tensors,
                        weight_format=linear.weight_format,
                        data_format=linear.weight_format.to_data_format(
                            cpp_dtype if cpp_dtype else _tm.DataType.TYPE_INVALID,
                            group_size))

    # Ensure the LinearWeight child exists
    linear_mod = module.child(name)
    if linear_mod is None:
        w = linear.tensors.get('weight')
        if w is None:
            return
        in_dim = w.shape[0]
        out_dim = w.shape[-1]
        if split_side == SplitSide.OUTPUT:
            out_dim = out_dim // split_num
        elif split_side == SplitSide.INPUT:
            in_dim = in_dim // split_num
        compute_dtype = _infer_compute_dtype(linear)
        # Always prefer the model's configured compute dtype.  For quantized
        # formats, the scales/bias dtype may differ from the model's actual
        # compute dtype (e.g. AWQ scales stored as bf16 in an fp16 model),
        # which would cause an input_dtype mismatch at GEMM time.
        if model_dtype is not None:
            compute_dtype = model_dtype
        lin_cfg = _tm.LinearConfig()
        lin_cfg.input_dim = in_dim
        lin_cfg.output_dim = out_dim
        lin_cfg.data_type = compute_dtype if compute_dtype else _tm.DataType.TYPE_INVALID
        lin_cfg.has_bias = 'bias' in linear.tensors
        linear_mod = module.create_child(name, lin_cfg)

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

    linear_mod.set_weight_spec(cpp_dtype, group_size)

    # Get model dimensions for correct weight allocation shape
    w = linear.tensors.get('weight')
    in_dim = w.shape[0] if w is not None else 0
    out_dim = w.shape[-1] if w is not None else 0
    if split_side == SplitSide.OUTPUT:
        out_dim = out_dim // split_num
    elif split_side == SplitSide.INPUT:
        in_dim = in_dim // split_num

    _commit_tensors(linear_mod, linear, cpp_dtype, group_size,
                    split_side, split_num, rank, in_dim, out_dim,
                    model_dtype=model_dtype)


def commit_tensor(module, tensor: torch.Tensor | None, name: str,
                         split_side: SplitSide | None = None,
                         split_num: int = 1, rank: int = 0):
    """Commit a raw tensor to a C++ ``Module`` handle for a specific TP rank.

    Parameters
    ----------
    module : C++ Module handle
        Module that owns the parameter (e.g. a ``DecoderLayerWeight``).
    tensor : torch.Tensor | None
        The tensor data.  ``None`` is a no-op.
    name : str
        Parameter name within *module* (e.g. ``"weight"`` for a norm).
    split_side, split_num, rank
        Same semantics as ``commit_linear``.
    """
    if tensor is None:
        return

    split_dim = _SPLIT_SIDE_TO_DIM.get(split_side) if split_side else None

    if split_dim is not None and split_num > 1:
        split_size = tensor.shape[split_dim] // split_num
        shard = tensor.split(split_size, dim=split_dim)[rank]
    else:
        shard = tensor

    if not shard.is_cuda:
        shard = shard.cuda(0).contiguous()
    elif not shard.is_contiguous():
        shard = shard.contiguous()
    cpp_dtype = _torch_dtype_to_cpp(shard.dtype)
    dst = module.param(name).alloc(list(shard.shape), cpp_dtype)
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
