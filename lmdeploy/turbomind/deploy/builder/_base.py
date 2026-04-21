# Copyright (c) OpenMMLab. All rights reserved.

import enum
import functools
import inspect

import torch

import _turbomind as _tm

from ..kind_map import TRIVIAL_FORMAT
from ..linear import Linear
# make_norm_config imported locally in _add_norm_child to avoid circular import
# (_base -> norm -> _base)

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
# Dequant / format compatibility helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# @transform_tensors decorator
# ---------------------------------------------------------------------------


def transform_tensors(fn):
    """Decorator that lifts a tensor-level transform to Linear-level.

    Convention: args that are ``Linear`` instances are treated as tensor
    inputs; all other args pass through unchanged.  Return type is detected
    at runtime: ``Tensor`` -> single ``Linear``, ``tuple`` -> tuple of
    ``Linear`` objects.
    """
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        first = next(v for v in bound.arguments.values()
                     if isinstance(v, Linear))
        out_buckets = None

        for kind in first.tensors:
            was_1d = False
            fn_kwargs = {}

            for name, val in bound.arguments.items():
                if isinstance(val, Linear):
                    t = val.tensors[kind]
                    if t.dim() == 1:
                        was_1d = True
                        t = t.unsqueeze(0)
                    fn_kwargs[name] = t
                else:
                    fn_kwargs[name] = val

            result = fn(**fn_kwargs)
            if not isinstance(result, tuple):
                result = (result,)
            if out_buckets is None:
                out_buckets = [{} for _ in result]
            for i, item in enumerate(result):
                out_buckets[i][kind] = item.squeeze(0) if was_1d else item

        outputs = tuple(
            Linear(ts, weight_format=first.weight_format,
                   data_format=first.data_format)
            for ts in out_buckets)
        return outputs if len(outputs) > 1 else outputs[0]

    return wrapper


# ---------------------------------------------------------------------------
# Core tensor commit (moved from commit.py)
# ---------------------------------------------------------------------------


def _copy_shard_to_param(handle, param_name: str, shard: torch.Tensor, *,
                         alloc_shape: list[int] | None = None,
                         alloc_dtype=None) -> None:
    """Move shard to GPU, allocate the C++ param slot, cast, and copy.

    Invariant: ``dst.byte_size == shard.nbytes`` after the cast.  Upstream
    is responsible for any padding/reshape needed to satisfy this.  A
    mismatch raises immediately.

    ``alloc_shape`` / ``alloc_dtype`` default to the shard's own shape /
    dtype.  Override only to express shape/dtype *relabels* where byte
    size is preserved (e.g. quantized weight: physical int32
    [in, out/8] stored in a logical UINT4 [in, out] C++ slot).
    """
    if not shard.is_cuda:
        shard = shard.cuda(0).contiguous()
    elif not shard.is_contiguous():
        shard = shard.contiguous()

    if alloc_shape is None:
        alloc_shape = list(shard.shape)
    if alloc_dtype is None:
        alloc_dtype = _torch_dtype_to_cpp(shard.dtype)

    dst = handle.param(param_name).alloc(alloc_shape, alloc_dtype)
    shard = _cast_shard_for_tm(shard, dst)
    assert dst.byte_size == shard.nbytes, (
        f"{param_name}: alloc byte_size={dst.byte_size} != "
        f"shard.nbytes={shard.nbytes}")
    dst.copy_from(shard)


def _shard(tensor: torch.Tensor, split_dim: int | None, tp: int,
           rank: int) -> torch.Tensor:
    """Return the ``rank``-th split along ``split_dim``, or the tensor unchanged.

    Used wherever a TP shard is selected from a broadcast-by-default
    tensor.  A ``split_dim`` of ``None`` or ``tp <= 1`` returns the tensor
    untouched.
    """
    if split_dim is None or tp <= 1:
        return tensor
    return tensor.split(tensor.shape[split_dim] // tp, dim=split_dim)[rank]


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
        config : C++ config struct
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
                if self._tp > 1 and hasattr(self.config, 'tp_rank'):
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
                shard = _shard(tensor, split_dim, tp, rank)
                _copy_shard_to_param(handle, name, shard)

    def _add_norm_child(self, name: str, tensor: torch.Tensor,
                        data_type=None, *, norm_eps):
        """Create a NormConfig child and commit weight tensor.

        Parameters
        ----------
        name : str
            Child module name (e.g. ``"attention_norm"``).
        tensor : torch.Tensor
            The norm weight tensor.
        data_type : C++ DataType value | None
            Compute dtype for the norm.  Defaults to FP32 if not set.
        norm_eps : float
            RMS norm epsilon.  Required.
        """
        self._ensure_handles()
        from .norm import make_norm_config
        if data_type is None:
            data_type = _tm.DataType.TYPE_FP32
        norm_cfg = make_norm_config(dim=tensor.shape[-1],
                                    data_type=data_type,
                                    norm_eps=norm_eps)

        for i, handle in enumerate(self._handles):
            with self._contexts[i]:
                child = handle.create_child(name, norm_cfg)
                _copy_shard_to_param(child, 'weight', tensor)


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
