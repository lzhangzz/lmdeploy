# Copyright (c) OpenMMLab. All rights reserved.
"""Builder base class and commit internals for builder-driven module loading.

Absorbs Distributor and the core commit logic from commit.py into a unified
Builder hierarchy.  Each Builder wraps N GPU handles for a single logical
module and distributes module creation, child binding, and weight commits
across all GPUs with bound TP configuration.
"""
from __future__ import annotations

import enum

import torch

import _turbomind as _tm

from .linear import Linear
from .module_configs import NormConfig

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

        handles = []
        for i, ctx in enumerate(contexts):
            with ctx:
                rank = ranks[i] if ranks and tp > 1 else 0
                cfg = config.for_rank(rank).to_cpp()
                handle = _tm.create_module(cfg)
                handles.append(handle)
        object.__setattr__(self, '_handles', handles)

    # ------------------------------------------------------------------
    # Child binding via attribute / item assignment
    # ------------------------------------------------------------------

    def __setattr__(self, name: str, value):
        """If *value* is a Builder, bind its handles as named children."""
        if isinstance(value, Builder):
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
        if data_type is None:
            data_type = _tm.DataType.TYPE_FP32
        norm_cfg = NormConfig(dim=tensor.shape[-1], data_type=data_type)

        for i, handle in enumerate(self._handles):
            with self._contexts[i]:
                child = handle.create_child(name, norm_cfg.to_cpp())
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
# AttentionBuilder -- QKV fusion, O-proj, QK-norm, direct params
# ---------------------------------------------------------------------------


class AttentionBuilder(Builder):
    """Attention weight loading builder."""

    _PARAM_TP_RULES: dict[str, SplitSide] = {
        'sinks': SplitSide.OUTPUT,
    }

    def add_qkv_proj(self, q, k, v):
        """Fuse QKV, shard along output dim, commit."""
        from .spec import merge_qkv_linear
        merged = merge_qkv_linear(
            q, k, v,
            tp=self._tp,
            head_dim=self.config.head_dim,
            rope_dim=self.config.head_dim,
            permute_qk=True,
            attn_output_gate=self.config.attn_output_gate,
            repeat_kv=0,
            kv_head_num=self.config.kv_head_num,
        )
        self._commit_linear('w_qkv', merged, SplitSide.OUTPUT,
                            model_dtype=self.config.data_type)

    def add_o_proj(self, o):
        """Shard along input dim, commit."""
        self._commit_linear('wo', o, SplitSide.INPUT,
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
        """Fuse w1+w3 if possible, shard, commit."""
        from .transforms import fuse_ffn_linears
        fused = None
        fused_silu = False
        if w1 is not None and w3 is not None:
            act_type = getattr(self.config, 'act_type', 'silu')
            # act_type is an int in FfnConfig, convert to string if needed
            if isinstance(act_type, int):
                act_type = {0: 'silu', 1: 'gpt-oss'}.get(act_type, 'silu')
            fused, fused_silu = fuse_ffn_linears(
                w1, w3, self._tp, act_type,
                is_moe=getattr(self.config, 'fused_moe', False))

        model_dtype = getattr(self.config, 'data_type', None)
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
