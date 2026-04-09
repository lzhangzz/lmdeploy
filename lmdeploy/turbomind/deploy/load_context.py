# Copyright (c) OpenMMLab. All rights reserved.
"""Composable loading primitives for building the C++ module tree from Python."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .linear import Linear
from .spec import SplitSide

if TYPE_CHECKING:
    from .config import ModelConfig


def _cpp_dtype(dtype_str: str) -> int:
    """Convert a model-config data_type string to C++ DataType enum value as int.

    Returns a plain int (not the enum object) so it can be passed in
    ModuleConfig dicts via pybind11 without requiring enum -> int64_t cast.
    """
    import _turbomind as _tm
    return {
        'float32':  _tm.DataType.TYPE_FP32,
        'float16':  _tm.DataType.TYPE_FP16,
        'bfloat16': _tm.DataType.TYPE_BF16,
    }[dtype_str].value


def _act_type_id(act_str: str) -> int:
    """Convert activation_type string to C++ ActivationType enum value."""
    return {'silu': 0, 'gpt-oss': 1}.get(act_str, 0)


# -----------------------------------------------------------------------
# Commit helpers (moved from commit.py)
# -----------------------------------------------------------------------

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

    # Dense (or missing format): dtype from weight tensor
    weight = linear.tensors.get("weight")
    if weight is not None:
        if weight.dtype == torch.bfloat16:
            return _tm.DataType.TYPE_BF16, 0
        if weight.dtype == torch.float16:
            return _tm.DataType.TYPE_FP16, 0
    return None, 0


def _infer_compute_dtype(linear: Linear):
    """Get the model's compute dtype from a Linear's tensors.

    For dense formats the weight itself carries the compute dtype.
    For quantized formats we infer from scales or bias.
    """
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
        # FP8 weights: compute dtype is BF16 (or FP16 depending on model),
        # not FP32.  Fall through to scales/bias only for non-FP8 dtypes.
        _fp8_dtypes = {torch.uint8}
        for _attr in ('float8_e4m3fn', 'float8_e5m2fn'):
            _dt = getattr(torch, _attr, None)
            if _dt is not None:
                _fp8_dtypes.add(_dt)
        if w.dtype in _fp8_dtypes:
            # FP8 stored as uint8 after normalization; prefer BF16.
            return _tm.DataType.TYPE_BF16
    for key in ('scales', 'bias'):
        t = linear.tensors.get(key)
        if t is not None:
            d = _MAP.get(t.dtype)
            if d is not None:
                return d
    return None


def _commit_tensors(handle, linear: Linear, cpp_dtype, group_size: int,
                    split_side: SplitSide | None, split_num: int, rank: int):
    """Commit tensor data from a ``Linear`` to a pre-created C++ LinearWeight handle.

    Handles packing, TP sharding, allocation, dtype casting, and padding.
    This is the shared tensor-commit loop used by both ``commit_linear`` and
    ``LoadContext.load_linear``.
    """
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

        if not shard.is_cuda:
            shard = shard.cuda(0).contiguous()
        elif not shard.is_contiguous():
            shard = shard.contiguous()

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
        dense (non-quantized) weights use this dtype instead of the weight
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
                            cpp_dtype.value if cpp_dtype else 0,
                            group_size))

    # Ensure the LinearWeight child exists
    linear_mod = module.child(name)
    if linear_mod is None:
        import _turbomind as _tm
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
        # For dense (non-quantized) weights, prefer the model's configured
        # compute dtype to avoid dtype mismatches (e.g. BF16 checkpoint
        # weights in an FP16-configured model).
        if model_dtype is not None and compute_dtype is not None:
            fmt = linear.weight_format
            if fmt is None or fmt.name == 'dense':
                model_dt = _tm.DataType(model_dtype) if isinstance(model_dtype, int) else model_dtype
                compute_dtype = model_dt
        lin_cfg = _tm.LinearConfig()
        lin_cfg.input_dim = in_dim
        lin_cfg.output_dim = out_dim
        lin_cfg.data_type = compute_dtype if compute_dtype else _tm.DataType(0)
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

    _commit_tensors(linear_mod, linear, cpp_dtype, group_size,
                    split_side, split_num, rank)


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


def commit_ffn(ffn_mod, w1: Linear, w3: Linear, w2: Linear | None,
               tp: int, rank: int, act_type: str, is_moe: bool = False,
               model_dtype=None):
    """DEPRECATED: Use Distributor + fuse_ffn_linears directly."""
    from .transforms import fuse_ffn_linears

    fused, fused_silu = fuse_ffn_linears(w1, w3, tp, act_type, is_moe)

    if fused is not None:
        commit_linear(ffn_mod, fused, "w1w3",
                           split_side=SplitSide.OUTPUT, split_num=tp,
                           rank=rank, model_dtype=model_dtype)
        ffn_mod.set_fused_silu(fused_silu)
    else:
        commit_linear(ffn_mod, w1, "w1",
                           split_side=SplitSide.OUTPUT, split_num=tp,
                           rank=rank, model_dtype=model_dtype)
        commit_linear(ffn_mod, w3, "w3",
                           split_side=SplitSide.OUTPUT, split_num=tp,
                           rank=rank, model_dtype=model_dtype)

    if w2 is not None:
        commit_linear(ffn_mod, w2, "w2",
                           split_side=SplitSide.INPUT, split_num=tp,
                           rank=rank, model_dtype=model_dtype)


# Backward-compatible alias
_fuse_and_commit_ffn = commit_ffn


# ======================================================================
# LoadContext
# ======================================================================

class LoadContext:
    """Wraps a C++ Module handle and provides composable loading primitives.

    Each LoadContext is rooted at one C++ Module.  The ``create`` method returns
    a new LoadContext rooted at the created child module.  ``load_linear`` and
    ``load_tensor`` handle the full create + commit lifecycle.
    """

    def __init__(self, handle, tp_config: dict,
                 model_config: 'ModelConfig | None' = None):
        """
        Args:
            handle: C++ Module handle (pybind11 object).
            tp_config: Dict with keys: tp_size, rank, head_dim,
                       rope_dim, permute_qk, repeat_kv, attn_output_gate,
                       kv_head_num.
            model_config: The Python ``ModelConfig`` for the model being loaded.
        """
        self._handle = handle
        self._tp_config = tp_config
        self._model_config = model_config

    @property
    def model_config(self) -> 'ModelConfig':
        assert self._model_config is not None, 'model_config not set'
        return self._model_config

    @property
    def cpp_dtype(self):
        """C++ DataType enum for the model's compute dtype."""
        return _cpp_dtype(self.model_config.data_type)

    @property
    def tp_size(self) -> int:
        return self._tp_config['tp_size']

    @property
    def rank(self) -> int:
        return self._tp_config['rank']

    @property
    def head_dim(self) -> int:
        return self._tp_config.get('head_dim', 0)

    @property
    def rope_dim(self) -> int:
        return self._tp_config.get('rope_dim', 0)

    @property
    def repeat_kv(self) -> int:
        return self._tp_config.get('repeat_kv', 0)

    @property
    def attn_output_gate(self) -> bool:
        return self._tp_config.get('attn_output_gate', False)

    @property
    def kv_head_num(self) -> int:
        return self._tp_config.get('kv_head_num', 0)

    def create(self, name: str, module_type: str, **config) -> 'LoadContext':
        """Create a child module via the C++ registry.

        Returns a new LoadContext rooted at the created module.
        """
        child = self._handle.create_child(name, module_type, config)
        return LoadContext(child, self._tp_config, self._model_config)

    def child(self, name: str) -> 'LoadContext':
        """Return a LoadContext for an existing child (no creation)."""
        handle = self._handle.get(name)
        return LoadContext(handle, self._tp_config, self._model_config)

    def load_linear(self, name: str, linear: Linear,
                    tp_rule: str | None = None):
        """Create a LinearWeight child and commit weight data.

        Handles TP splitting, quantization packing, and dtype casting.
        The child is created via create_child, then weights are committed
        using the shared _commit_tensors function.
        """
        import _turbomind as _tm
        tp_side = SplitSide[tp_rule] if tp_rule else None
        split_num = self.tp_size if tp_side else 1

        cpp_dtype, group_size = _infer_cpp_linear_dtype(linear)
        if group_size == 0:
            group_size = max(1, 128)

        weight = linear.tensors.get('weight') or linear.tensors.get('qweight')
        input_dim = weight.shape[0] if weight is not None else 0
        output_dim = weight.shape[-1] if weight is not None else 0

        lin_cfg = _tm.LinearConfig()
        lin_cfg.input_dim = input_dim
        lin_cfg.output_dim = output_dim
        lin_cfg.data_type = cpp_dtype
        lin_cfg.has_bias = 'bias' in linear.tensors
        child_handle = self._handle.create_child(name, lin_cfg)

        _commit_tensors(child_handle, linear, cpp_dtype, group_size,
                        tp_side, split_num, self.rank)

    def load_tensor(self, name: str, tensor: torch.Tensor,
                    module_type: str = 'NormWeight',
                    module_config: dict | None = None,
                    tp_rule: str | None = None):
        """Create a module child and commit tensor data.

        Args:
            name: Child module name.
            tensor: Weight tensor to commit.
            module_type: C++ module type to create (e.g., ``"NormWeight"``).
            module_config: Config dict for module creation.
            tp_rule: ``"output"`` or ``"input"`` for TP split, None for
                broadcast.
        """
        config = module_config or {}
        child_handle = self._handle.create_child(name, module_type, config)

        tp_side = SplitSide[tp_rule] if tp_rule else None
        commit_tensor(child_handle, tensor, 'weight',
                            split_side=tp_side,
                            split_num=self.tp_size if tp_side else 1,
                            rank=self.rank)
