# Copyright (c) OpenMMLab. All rights reserved.

import enum
import functools
import inspect

import torch

import _turbomind as _tm

from ..weight_format import TrivialFormat
from ..linear import Linear, pad_out_dim

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

_CPP_TO_TORCH: dict[_tm.DataType, torch.dtype] = {v: k for k, v in _TORCH_TO_CPP.items()}

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


def _dequant_linear(linear: Linear, *, data_type) -> Linear:
    """Dequantize a quantized Linear to trivial.

    ``TrivialFormat.dequant`` is identity, so already-trivial inputs round-trip
    safely. ``AWQFormat.dequant`` and ``FP8Format.dequant`` do real work.
    GPTQ / CompressedTensor / MXFP4 inherit the base-class
    ``NotImplementedError`` — calling ``_dequant_linear`` on one of those is a
    broken-fusion-group configuration, and the raise names it at the call site.
    """
    fmt = linear.weight_format
    new_tensors = fmt.dequant(linear.tensors, data_type)
    trivial = TrivialFormat()
    return Linear(
        tensors=new_tensors,
        weight_format=trivial,
        data_format=trivial.make_data_format(data_type),
    )


def _ensure_compatible_formats(linears: dict[str, Linear], *, data_type) -> dict[str, Linear]:
    """Dequant linears to a common trivial format if a fusion group has mixed formats."""
    formats = {name: lin.weight_format.name for name, lin in linears.items()}
    if len(set(formats.values())) <= 1:
        # Weight formats agree; normalize data_format to a single shared object.
        target_df = next(iter(linears.values())).data_format
        return {name: (Linear(lin.tensors, weight_format=lin.weight_format,
                              data_format=target_df)
                       if lin.data_format is not target_df else lin)
                for name, lin in linears.items()}
    result = {name: _dequant_linear(lin, data_type=data_type) for name, lin in linears.items()}
    # Normalize data_format after dequant — each _dequant_linear may produce
    # a distinct DataFormat object even when they represent the same dtype.
    target_df = next(iter(result.values())).data_format
    return {name: (Linear(lin.tensors, weight_format=lin.weight_format,
                          data_format=target_df)
                   if lin.data_format is not target_df else lin)
            for name, lin in result.items()}


# ---------------------------------------------------------------------------
# @transform_output_dim / @transform_input_dim decorators
# ---------------------------------------------------------------------------


def transform_output_dim(fn):
    """Decorator that lifts a tensor-level transform to Linear-level.

    For output-dim operations: 1-D tensors (bias) are unsqueezed to 2-D
    before calling *fn*, then squeezed back.  Convention: args that are
    ``Linear`` instances are treated as tensor inputs; all other args pass
    through unchanged.  Return type is detected at runtime:
    ``Tensor`` -> single ``Linear``, ``tuple`` -> tuple of ``Linear`` objects.
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


def transform_input_dim(fn):
    """Decorator that lifts a tensor-level transform to Linear-level.

    For input-dim operations: 1-D tensors (bias) have no input dimension
    and are **passed through unchanged**.  The inner function only ever
    sees 2-D tensors for each kind.  For multi-output functions, 1-D
    tensors are duplicated into every output bucket.
    """
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        first = next(v for v in bound.arguments.values()
                     if isinstance(v, Linear))
        out_buckets = None
        deferred_1d: list[str] = []

        for kind in first.tensors:
            fn_kwargs = {}
            is_1d = False

            for name, val in bound.arguments.items():
                if isinstance(val, Linear):
                    t = val.tensors[kind]
                    if t.dim() < 2:
                        is_1d = True
                        break
                    fn_kwargs[name] = t
                else:
                    fn_kwargs[name] = val

            if is_1d:
                deferred_1d.append(kind)
                continue

            result = fn(**fn_kwargs)
            if not isinstance(result, tuple):
                result = (result,)
            if out_buckets is None:
                out_buckets = [{} for _ in result]
            for i, item in enumerate(result):
                out_buckets[i][kind] = item

        if out_buckets is None:
            out_buckets = [{}]
        for kind in deferred_1d:
            for bucket in out_buckets:
                bucket[kind] = first.tensors[kind]

        outputs = tuple(
            Linear(ts, weight_format=first.weight_format,
                   data_format=first.data_format)
            for ts in out_buckets)
        return outputs if len(outputs) > 1 else outputs[0]

    return wrapper


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


# ---------------------------------------------------------------------------
# Builder base class
# ---------------------------------------------------------------------------


class BuiltModule:
    """Opaque handle bundle returned by ``Builder.build()``.

    Wraps a list of per-GPU C++ module handles.  Iteration and len delegate
    to the underlying list so callers can ``zip(BuiltModule, contexts)`` etc.
    """

    __slots__ = ('handles',)

    def __init__(self, handles):
        self.handles = handles

    def __iter__(self):
        return iter(self.handles)

    def __len__(self):
        return len(self.handles)


class Builder:
    """Wraps N GPU handles for a single logical module.

    Distributes module creation, child binding, and weight commits
    across all GPUs with bound TP configuration.

    Subclasses specialize for particular module types (e.g. attention,
    FFN, MoE).

    Lifecycle: stage commits -> build() -> BuiltModule (frozen).
    After ``build()`` the Builder is inert — further commits or child
    attachments raise.
    """

    def __init__(self, config, contexts, tp=1, ranks=None):
        """Initialise the builder with staging dicts (no C++ creation yet).

        Parameters
        ----------
        config : C++ config struct
            Config with ``clone()`` method and optionally ``tp_rank`` field.
        contexts : list
            GPU context managers (one per GPU).
        tp : int
            Tensor parallelism degree.
        ranks : list[int] | None
            Per-GPU TP ranks.
        """
        # `_built` must be set first: __setattr__ reads it inside the
        # BuiltModule branch.  Bool is not a BuiltModule, so the normal
        # fall-through assigns it via object.__setattr__ at the end of
        # __setattr__.
        self._built = False
        self._contexts = contexts
        self._tp = tp
        self._ranks = ranks
        self.config = config
        self._pending_linears = {}
        self._pending_tensors = {}
        self._pending_children = {}
        self._handles = None

    # ------------------------------------------------------------------
    # Child binding via attribute assignment
    # ------------------------------------------------------------------

    def __setattr__(self, name: str, value):
        if isinstance(value, Builder):
            raise TypeError(
                f"{type(self).__name__}.{name}: assign .build() output "
                f"(BuiltModule), not the Builder itself")
        if isinstance(value, BuiltModule):
            if self._built:
                raise RuntimeError(
                    f"{type(self).__name__} is built; "
                    f"cannot assign {name!r}")
            self._pending_children[name] = value.handles
            return
        object.__setattr__(self, name, value)

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
    # Staging methods (pre-build only)
    # ------------------------------------------------------------------

    def _commit_linear(self, name: str, linear: Linear,
                       split_side: SplitSide | None = None,
                       model_dtype=None):
        """Stage a ``Linear`` commit under ``name``.  Applied during
        ``build()`` in ``_apply_linear``.
        """
        assert not self._built, (
            f"{type(self).__name__} is built; commit '{name}' rejected")
        self._pending_linears[name] = (linear, split_side, model_dtype)

    def _commit_tensor(self, name: str, tensor: torch.Tensor | None,
                       split_side: SplitSide | None = None, *,
                       model_dtype=None):
        """Stage a raw-tensor commit under ``name``.  Applied during
        ``build()`` in ``_apply_tensor``.
        """
        assert not self._built, (
            f"{type(self).__name__} is built; commit '{name}' rejected")
        if tensor is not None:
            self._pending_tensors[name] = (tensor, split_side, model_dtype)

    # ------------------------------------------------------------------
    # Commit helpers
    # ------------------------------------------------------------------

    def _commit_child(self, name: str, handles: list):
        """Stage pre-created per-GPU ``Module*`` handles under ``name``.
        Applied during ``build()`` in ``_apply_child``.
        """
        assert not self._built, (
            f"{type(self).__name__} is built; commit '{name}' rejected")
        assert name not in self._pending_children, (
            f"{type(self).__name__}: duplicate child commit '{name}'")
        self._pending_children[name] = handles

    # ------------------------------------------------------------------
    # build() — create handles, drain staged state, return BuiltModule
    # ------------------------------------------------------------------

    def build(self) -> BuiltModule:
        """Create C++ module handles and drain all staged state.

        Idempotent on second call — returns the same ``BuiltModule``.
        """
        if self._built:
            return BuiltModule(self._handles)

        self._create_handles()

        # True is not BuiltModule; falls through to plain assignment.
        self._built = True

        # Drain staged linears
        for name, (linear, split_side, model_dtype) in self._pending_linears.items():
            self._apply_linear(name, linear, split_side, model_dtype)

        # Drain staged tensors
        for name, (tensor, split_side, model_dtype) in self._pending_tensors.items():
            self._apply_tensor(name, tensor, split_side, model_dtype)

        # Drain staged children
        for name, child_handles in self._pending_children.items():
            self._attach_handles(name, child_handles)

        return BuiltModule(self._handles)

    def _create_handles(self):
        """Create one C++ module per context via ``_tm.create_module(cfg)``."""
        handles = []
        for i, ctx in enumerate(self._contexts):
            with ctx:
                cfg = self._cfg_for_rank(i)
                handle = _tm.create_module(cfg)
                handles.append(handle)
        self._handles = handles

    def _cfg_for_rank(self, gpu_idx: int):
        """Clone config and set tp_rank if tp > 1."""
        if self._tp > 1 and hasattr(self.config, 'tp_rank'):
            cfg = self.config.clone()
            cfg.tp_rank = self._ranks[gpu_idx]
            return cfg
        return self.config

    def _apply_child(self, name: str, handles: list):
        """Attach pre-created per-GPU child handles to parent handles."""
        for i, (parent_h, child_h) in enumerate(
                zip(self._handles, handles)):
            with self._contexts[i]:
                parent_h.add_child_raw(name, child_h)

    def _attach_handles(self, name: str, child_handles: list):
        """Attach a child's handles to this module's handles."""
        for i, (parent_h, child_h) in enumerate(
                zip(self._handles, child_handles)):
            with self._contexts[i]:
                parent_h.add_child_raw(name, child_h)

    # ------------------------------------------------------------------
    # Apply methods (GPU-invariant prep + per-GPU commit)
    # ------------------------------------------------------------------

    def _apply_linear(self, name: str, linear: Linear,
                      split_side: SplitSide | None = None,
                      model_dtype=None):
        """Commit a ``Linear`` bundle to a named child on all GPUs.

        Creates a ``LinearWeight`` child via ``handle.create_child`` using
        a ``LinearConfig`` derived from the linear's dimensions and compute
        dtype.  Tensor data is sharded per rank (for TP) and copied to the
        C++ slots via ``_copy_shard_to_param``.

        Parameters
        ----------
        name : str
            Child module name (e.g. ``"w_qkv"``).
        linear : Linear
            The linear bundle to commit.
        split_side : SplitSide | None
            TP split semantics.  ``None`` means broadcast (no split).
        model_dtype : C++ DataType value | None
            The model's configured compute dtype.
        """
        w = linear.tensors.get('weight')
        if w is None:
            return

        # --- GPU-invariant preparation -------------------------------------
        assert linear.data_format is not None, (
            f"{name}: Linear.data_format must be populated by "
            f"WeightFormatResolver.resolve or a fusion helper.")
        weight_cpp_dtype = linear.data_format.dtype
        fmt = linear.weight_format

        tp = self._tp if split_side else 1
        split_dim = _SPLIT_SIDE_TO_DIM.get(split_side) if split_side else None

        in_dim, out_dim = w.shape[0], w.shape[-1]
        if split_side == SplitSide.OUTPUT:
            out_dim //= tp
        elif split_side == SplitSide.INPUT:
            in_dim //= tp

        compute_dtype = (model_dtype if model_dtype is not None
                         else _infer_compute_dtype(linear))
        lin_cfg = _tm.LinearConfig()
        lin_cfg.input_dim  = in_dim
        lin_cfg.output_dim = out_dim
        lin_cfg.data_type  = compute_dtype or _tm.DataType.TYPE_INVALID
        lin_cfg.format     = linear.data_format
        lin_cfg.has_bias   = 'bias' in linear.tensors

        tensors = {k: fmt.pack(t, k) for k, t in linear.tensors.items()}
        is_quantized = linear.data_format.is_quantized()

        kind_split_dims = {
            kind: None if (kind == 'bias' and split_side == SplitSide.INPUT)
                  else split_dim
            for kind in tensors
        }

        # Uniform TP-split validation: every kind split along some axis
        # must have that axis evenly divisible by tp.  Covers weight,
        # scales, zeros, bias; covers both INPUT and OUTPUT split_side;
        # respects the bias-on-INPUT no-split rule.  Runs after the
        # packer hoist so it sees the tensors that will actually be
        # split.
        if tp > 1 and split_dim is not None:
            for kind, tensor in tensors.items():
                kind_split_dim = kind_split_dims[kind]
                if kind_split_dim is not None:
                    d = tensor.shape[kind_split_dim]
                    assert d % tp == 0, (
                        f"TP split: {name}.{kind} dim {kind_split_dim} "
                        f"has size {d}, not divisible by tp={tp}.")

        # --- Per-GPU commit ------------------------------------------------
        for i, handle in enumerate(self._handles):
            with self._contexts[i]:
                rank = self._rank_for(i) if tp > 1 else 0

                # get-or-create: create_child fires only on the first commit for this name
                linear_mod = (handle.child(name)
                              or handle.create_child(name, lin_cfg))

                for kind, tensor in tensors.items():
                    shard = _shard(tensor, kind_split_dims[kind], tp, rank)

                    if kind == 'weight' and is_quantized:
                        alloc_shape, alloc_dtype = ([in_dim, out_dim],
                                                    weight_cpp_dtype)
                    elif kind == 'weight' and model_dtype is not None:
                        alloc_shape, alloc_dtype = None, model_dtype
                    else:
                        alloc_shape, alloc_dtype = None, None

                    _copy_shard_to_param(linear_mod, kind, shard,
                                         alloc_shape=alloc_shape,
                                         alloc_dtype=alloc_dtype)

    def _apply_tensor(self, name: str, tensor: torch.Tensor,
                      split_side: SplitSide | None = None,
                      model_dtype=None):
        """Commit a raw tensor to a named parameter on all GPUs.

        Parameters
        ----------
        name : str
            Parameter name within the module.
        tensor : torch.Tensor
            The tensor data.
        split_side : SplitSide | None
            TP split semantics.  ``None`` means broadcast.
        model_dtype : C++ DataType value | None
            Override dtype for the C++ allocation.
        """
        tp = self._tp if split_side else 1
        split_dim = _SPLIT_SIDE_TO_DIM.get(split_side) if split_side else None

        for i, handle in enumerate(self._handles):
            with self._contexts[i]:
                rank = self._rank_for(i) if tp > 1 else 0
                shard = _shard(tensor, split_dim, tp, rank)
                _copy_shard_to_param(handle, name, shard,
                                     alloc_dtype=model_dtype)


# ---------------------------------------------------------------------------
# TextModelBuilder -- creates ModelWeight via standard Builder path
# ---------------------------------------------------------------------------


class TextModelBuilder(Builder):
    """Builder for the root ModelWeight.

    Constructs a ModelWeight via ``_tm.create_module(ModelWeightConfig)``
    on each context (inherited Builder machinery), then attaches it to
    externally-owned ``ModelRoot`` sentinel handles as their
    ``text_model`` child during ``build()``.

    Owns ``tok_embeddings`` (Tensor param) and ``output`` (LinearWeight
    child) commits on the ModelWeight via ``add_token_embeds`` /
    ``add_lm_head``.
    """

    def __init__(self, config, contexts, *, root_handles,
                 tp, ranks, vocab_size, data_type):
        super().__init__(config=config, contexts=contexts, tp=tp, ranks=ranks)
        self._root_handles = root_handles
        self._vocab_size = vocab_size
        self._data_type = data_type

    def build(self) -> BuiltModule:
        """Create ModelWeight via _tm.create_module (via super), then
        attach each per-GPU ModelWeight handle to its sentinel root
        via add_child_raw.
        """
        built = super().build()
        for i, (root, text_model) in enumerate(
                zip(self._root_handles, built.handles)):
            with self._contexts[i]:
                root.add_child_raw('text_model', text_model)
        return built

    def add_token_embeds(self, tensor):
        """Commit the raw embedding lookup as the ``tok_embeddings`` root param.

        Shards along hidden (output) dim by ``self._tp``. No vocab padding —
        embedding lookup never indexes past ``vocab - 1``.
        """
        self._commit_tensor('tok_embeddings', tensor,
                            split_side=SplitSide.OUTPUT,
                            model_dtype=self._data_type)

    def add_lm_head(self, linear):
        """Pad output dim to ``round_up(vocab_size, tp)`` and commit to the
        ``output`` LinearWeight root child.

        Works for every checkpoint format in use today — trivial / AWQ /
        GPTQ / compressed-tensors / MXFP4 all have ``block_out is None``,
        so padding every tensor in the bundle along ``dim=-1`` keeps the
        format-specific block structure intact. FP8 ``lm_head``
        (``block_out == 128``) would misalign scales under naive padding
        but is not a configuration used by any released checkpoint.
        """
        padded_vocab = ((self._vocab_size + self._tp - 1)
                        // self._tp) * self._tp
        padded = Linear(
            tensors={k: pad_out_dim(t, padded_vocab, dim=-1)
                     for k, t in linear.tensors.items()},
            weight_format=linear.weight_format,
            data_format=linear.data_format)
        self._commit_linear('output', padded,
                            split_side=SplitSide.OUTPUT,
                            model_dtype=self._data_type)
