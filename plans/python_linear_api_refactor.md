# Python Linear API refactor

## Status

**Resumed for re-audit. Do not implement until this revised plan is explicitly
approved.**

## Prerequisite status

The dtype/weight-format prerequisite is complete in `6a6dd1d8a` and
`12f59be1d`. `WeightFormat.make_data_format()` is now zero-argument,
`TrivialFormat` carries a concrete weight dtype, quantized formats use
`kGenericFloat` only for floating qparameter metadata, and family planning
returns concrete bridge targets.

This plan predates that contract and is being re-audited against it. Stale
descriptor construction, zero synthesis, normalization, allocation, and copy
snippets must be corrected before approval.

## Goal

Refactor `tests/turbomind/linear/linear.py` into a small standalone Python API
for preparing TurboMind weights and running dense or grouped linear operations.
The API must hide GEMM planning, parameter allocation, packing, TurboMind
context lifetime, stream handoff, and temporary-output lifetime.

An AWQ wrapper passes the tensors it owns and the format it already knows:

```python
linear = Linear(device=0)
weight = None
try:
    plan = linear.plan_weight(
        weight_format=AWQFormat(block_in=group_size),
        dtype=torch.bfloat16,
        input_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
    )
    weight = linear.prepare_weight(
        qweight,
        plan=plan,
        scales=scales,
        zeros=qzeros,
    )
    output_meta, scales_meta = linear.output_spec(x, weight)
    output = torch.empty_strided(
        output_meta.shape,
        output_meta.stride(),
        dtype=output_meta.dtype,
        device=x.device,
    )
    output_scales = None
    if scales_meta is not None:
        output_scales = torch.empty_strided(
            scales_meta.shape,
            scales_meta.stride(),
            dtype=scales_meta.dtype,
            device=x.device,
        )
    output, output_scales = linear(
        x,
        weight,
        out=output,
        out_scales=output_scales,
    )
finally:
    torch.cuda.synchronize(linear.device)
    if weight is not None:
        with torch.cuda.device(linear.device):
            weight.close()
    linear.close()
```

An ordinary floating-point wrapper supplies its known format the same way:

```python
linear = Linear(device=0)
weight = None
try:
    dtype = torch.bfloat16
    plan = linear.plan_weight(
        weight_format=TrivialFormat(dtype=dtype),
        dtype=dtype,
    )
    weight = linear.prepare_weight(
        torch_weight,
        plan=plan,
    )
    output_meta, _ = linear.output_spec(x, weight)
    output = torch.empty_strided(
        output_meta.shape,
        output_meta.stride(),
        dtype=output_meta.dtype,
        device=x.device,
    )
    output, output_scales = linear(x, weight, out=output)
finally:
    torch.cuda.synchronize(linear.device)
    if weight is not None:
        with torch.cuda.device(linear.device):
            weight.close()
    linear.close()
```

No `Gemm`, `WeightQuery`, `GemmPlan`, `LinearConfig`, TurboMind context, or
manual result release appears in user code. The standalone API does not accept
raw PyTorch engine modules. A format-specific wrapper supplies its final
TP-local tensors and its known `WeightFormat`; `Linear` does not detect the
checkpoint format.

## Scope

This change covers:

- `tests/turbomind/linear/linear.py`;
- migration of `fixture.py`, `benchmark.py`, and `test_linear.py` to the new
  interface;
- the Torch-facing `TrivialFormat` constructor and its existing converter,
  model, dequantization, and focused-test call sites;
- resolved weight-shape constraints on `GemmPlan`, reused by the existing
  builder padding paths and exposed through the standalone API;
- family-level CUDA-graph metadata and the C++ GEMM output-layout query;
- a non-owning external-handle constructor in
  `src/turbomind/core/stream.h` and `stream.cc`;
- the caller-stream context and raw byte-copy bindings in
  `src/turbomind/python/bind.cpp`;
- native `torch.float8_e4m3fn` import through the vendored DLPack type code;
- dense, grouped/MoE, and fused gate/up execution already covered by the
  linear test suite;
- every source `DataFormat` currently exercised by the suite except group-16
  NVFP4, with BF16-input block-out-128 FP8 SiLU fusion also excluded;
- tuning-record import/export and measurement mode.

This change does not move or re-export the API into the installed `lmdeploy`
package. The package already has an unrelated checkpoint bundle named
`lmdeploy.turbomind.linear.Linear`; choosing the installed runtime module name,
resolving that naming conflict, and performing package migration are a later
change requiring a separate approved plan. This refactor establishes and
verifies the interface only in `tests.turbomind.linear.linear`.

No CUDA kernel changes are required. The core stream wrapper and Python
context binding are extended to use a caller-owned Torch CUDA stream and the
CUDA device default memory pool. One Python binding is also added for an
asynchronous byte-for-byte tensor copy on a supplied stream. This is required
when physical packed storage such as `int32[K, N / 8]` is copied into a logical
`UINT4[K, N]` parameter: the byte sizes match, but element-wise generic copy is
not applicable.

Group-16 NVFP4 is explicitly excluded from this iteration. There is no public
`NVFP4Format`: its E4M3 group scales, group size 16, and FP32 global scale
cannot be represented by `MXFP4Format`. Do not add a fixture-private fake
format. When the existing fixture encounters
`weight_type == 'fp4_e2m1' and group_size == 16`, it raises
`NotImplementedError('NVFP4 is not supported by the standalone Linear API')`.
The unit test reports the existing skip and the benchmark emits its existing
unsupported-case warning.

BF16-input, block-out-128 FP8 SiLU fusion is also excluded from this iteration.
The selected family interleaves gate/up in 64-column blocks, while one source
scale covers 128 independently quantized output columns. Correct support needs
scale expansion before fusion and replanning with a groupwise source format;
the current C++ bridge runs too late. Native-FP8-input fusion uses 128-column
gate/up blocks and remains supported. No other format or fusion combination is
an intentional skip.

## Format responsibility

`WeightFormatResolver` remains a checkpoint-loader concern and is not part of
this API. A format-specific wrapper already knows whether its tensors are AWQ,
GPTQ, FP8, MXFP4, or ordinary floating point, including the quantization group
size. It passes that concrete `WeightFormat` directly.

`Linear` applies the supplied format's existing normalization and packing
policies, but it does not inspect suffixes or select among formats:

```python
normalized = self._normalize_params(
    weight_format,
    weight,
    scales,
    zeros,
)
data_format = weight_format.make_data_format()
```

This reuses TurboMind's format-specific tensor transformations without a fake
checkpoint dictionary, `Prefix`, resolver, format probing, or dependency on a
PyTorch engine module class.

Zero-point synthesis matches the current resolver contract exactly: it occurs
after scales have been normalized, only when the normalized format contains
zeros and the caller omitted them. Missing scales still fail, and a supplied
zero tensor is normalized normally rather than replaced. Synthesized and
supplied U4 zeros are converted to the normalized scale tensor's concrete
dtype before physical packing.

## Public surface

`tests.turbomind.linear.linear` exports only:

```python
__all__ = [
    'Linear',
    'Weight',
    'WeightPlan',
    'is_available',
]
```

`is_available()` answers only whether the in-tree extension can be imported:

```python
def is_available() -> bool:
    try:
        _tm()
    except ImportError:
        return False
    return True
```

It does not probe CUDA availability, architecture, or family support. Errors
other than `ImportError` propagate instead of making a broken extension appear
unavailable.

The exact public call signatures are:

```python
import copy
import math
import os
from collections.abc import Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from lmdeploy.turbomind.weight_format import WeightFormat


Linear(device: torch.device | str | int = 'cuda')

Linear.plan_weight(
    *,
    weight_format: WeightFormat,
    dtype: torch.dtype,
    input_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
    grouped: bool = False,
    fusion_type: Literal['silu'] | None = None,
) -> WeightPlan

Linear.prepare_weight(
    weight: torch.Tensor | Sequence[torch.Tensor],
    *,
    plan: WeightPlan,
    scales: torch.Tensor | Sequence[torch.Tensor] | None = None,
    zeros: torch.Tensor | Sequence[torch.Tensor] | None = None,
) -> Weight

Linear.fuse_weight(
    weight: tuple[
        torch.Tensor | Sequence[torch.Tensor],
        torch.Tensor | Sequence[torch.Tensor],
    ],
    *,
    plan: WeightPlan,
    scales: tuple[
        torch.Tensor | Sequence[torch.Tensor],
        torch.Tensor | Sequence[torch.Tensor],
    ] | None = None,
    zeros: tuple[
        torch.Tensor | Sequence[torch.Tensor],
        torch.Tensor | Sequence[torch.Tensor],
    ] | None = None,
) -> Weight

Linear.__call__(
    x: torch.Tensor,
    weight: Weight,
    *,
    out: torch.Tensor,
    input_scales: torch.Tensor | None = None,
    out_scales: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]

Linear.forward_moe(
    x: torch.Tensor,
    weight: Weight,
    *,
    offsets: torch.Tensor,
    out: torch.Tensor,
    indices: torch.Tensor | None = None,
    input_scales: torch.Tensor | None = None,
    out_scales: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]

Linear.output_spec(
    x: torch.Tensor,
    weight: Weight,
) -> tuple[torch.Tensor, torch.Tensor | None]

Linear.output_spec_moe(
    x: torch.Tensor,
    weight: Weight,
    *,
    indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]

Linear.tuning()
Linear.import_records(path: str | os.PathLike[str]) -> int
Linear.export_records(path: str | os.PathLike[str]) -> int
Linear.close() -> None
Weight.close() -> None
Weight.is_graph_compatible: bool
WeightPlan.shape_constraints: tuple[
    tuple[int, int],
    tuple[int, int],
]
```

Passing one weight tensor prepares a dense weight. Passing a non-empty sequence
prepares a grouped weight. Each present companion is respectively one tensor or
a sequence with the same length as `weight`. Every grouped expert must have the
same tensor kinds and logical shape. Preparation infers that structure and
requires it to match the `grouped` mode captured by `WeightPlan`; there is no
second `grouped` argument on `prepare_weight` or `fuse_weight`.

Dense and grouped execution remain distinct and routing arguments never alter
the mode implicitly:

```python
dense = linear.prepare_weight(
    weight_tensor,
    plan=linear.plan_weight(
        weight_format=TrivialFormat(dtype=torch.bfloat16),
        dtype=torch.bfloat16,
    ),
)
dense_meta, _ = linear.output_spec(x, dense)
dense_output = torch.empty_strided(
    dense_meta.shape,
    dense_meta.stride(),
    dtype=dense_meta.dtype,
    device=x.device,
)
dense_output, dense_output_scales = linear(
    x,
    dense,
    out=dense_output,
)

grouped = linear.prepare_weight(
    [expert0, expert1, expert2],
    plan=linear.plan_weight(
        weight_format=TrivialFormat(dtype=torch.bfloat16),
        dtype=torch.bfloat16,
        grouped=True,
    ),
)
moe_meta, _ = linear.output_spec_moe(
    x,
    grouped,
    indices=indices,
)
moe_output = torch.empty_strided(
    moe_meta.shape,
    moe_meta.stride(),
    dtype=moe_meta.dtype,
    device=x.device,
)
moe_output, moe_output_scales = linear.forward_moe(
    x,
    grouped,
    offsets=offsets,
    indices=indices,
    input_scales=input_scales,
    out=moe_output,
)
```

`Linear.__call__` rejects grouped weights and `Linear.forward_moe` rejects
dense weights.

The caller retains ownership of the source tensors; preparation copies their
normalized and packed values into TurboMind storage. `WeightPlan` is the opaque
result of the one family-selection call and exposes only its immutable
`shape_constraints`. `Weight` is an opaque handle returned only by
`prepare_weight` or `fuse_weight`. Its public surface is the immutable
`is_graph_compatible` property and the idempotent `close()` method. It
does not expose dimensions, dtypes, format, grouped mode, plan,
`gate_up_block`, parameter slots, matrix descriptors, or the bound
`_turbomind.LinearWeight`. `Linear` uses private fields for validation and
output allocation; a format-specific wrapper already owns the corresponding
model metadata.

Do not add any other configuration or result class. The three public classes
contain this state:

```python
class WeightPlan:
    __slots__ = (
        '_impl',
        '_weight_format',
        '_dtype',
        '_grouped',
        '_fusion_type',
        '_shape_constraints',
    )


class Weight:
    __slots__ = (
        '_impl',
        '_experts',
    )


class Linear:
    __slots__ = (
        'device',
        '_gemm',
        '_impl',
        '_closed',
    )
```

`WeightPlan` is host-side planning state and has no `close()` method. Its sole
public property is:

```python
@property
def shape_constraints(self):
    return self._shape_constraints
```

Planning stores a shallow snapshot of the supplied built-in `WeightFormat`.
The built-in formats contain only concrete dtype and block-size metadata, so
later mutation of the caller's format object cannot change normalization or
packing for an existing plan. Private `WeightPlan` fields are not part of the
public mutation surface.

The prepared-weight property delegates to C++ and is available while the
weight is open:

```python
@property
def is_graph_compatible(self) -> bool:
    if self._impl is None:
        raise RuntimeError('Weight is closed')
    return bool(self._impl.is_graph_compatible)
```

Bias is deliberately absent. `LlamaLinear::Forward` does not consume
`LinearWeight::bias`; accepting a bias would silently promise behavior that the
execution path does not provide. A wrapper may add bias to a non-quantized
result after the call:

```python
output, output_scales = linear(
    x,
    weight,
    out=output_buffer,
    out_scales=scale_buffer,
)
if bias is not None:
    if output_scales is not None:
        raise NotImplementedError(
            'bias is not supported for native FP8 output')
    output = output + bias
```

Adding a floating-point bias to native FP8 output would change its quantized
contract and remains unsupported until a real bias epilogue exists.

## Dtype and source-layout contract

Public dtype arguments are `torch.dtype`, never strings. The conversion table
is private and complete for the accepted tensor dtypes:

```python
_TORCH_TO_TM_NAME = {
    torch.uint8: 'TYPE_UINT8',
    torch.int32: 'TYPE_INT32',
    torch.float16: 'TYPE_FP16',
    torch.bfloat16: 'TYPE_BF16',
    torch.float32: 'TYPE_FP32',
    torch.float8_e4m3fn: 'TYPE_FP8_E4M3',
}


def _to_tm_dtype(dtype):
    try:
        name = _TORCH_TO_TM_NAME[dtype]
    except KeyError:
        raise TypeError(f'unsupported torch dtype: {dtype}') from None
    return getattr(_tm().DataType, name)


def _to_torch_dtype(dtype):
    tm = _tm()
    for torch_dtype, name in _TORCH_TO_TM_NAME.items():
        if dtype == getattr(tm.DataType, name):
            return torch_dtype
    raise TypeError(f'unsupported TurboMind dtype: {dtype}')
```

The map stores enum member names rather than `_turbomind` values so importing
the Python module remains lazy and `is_available()` can handle an absent
extension. Import `WeightFormat` only under `TYPE_CHECKING`; the concrete
object is used through its existing `normalize`, `pack`, and
`make_data_format` methods without a defensive runtime type check.

PyTorch publishes `torch.float8_e4m3fn` with DLPack type code 10. Extend only
the required type in the vendored header:

```cpp
// src/turbomind/python/dlpack.h
typedef enum
{
    kDLInt = 0U,
    kDLUInt = 1U,
    kDLFloat = 2U,
    kDLOpaqueHandle = 3U,
    kDLBfloat = 4U,
    kDLComplex = 5U,
    kDLBool = 6U,
    kDLFloat8_e4m3fn = 10U,
} DLDataTypeCode;
```

Map it in the existing DLPack importer:

```cpp
case DLDataTypeCode::kDLFloat8_e4m3fn:
    return data_type_v<turbomind::fp8_e4m3_t>;
```

This makes both `from_dlpack` and `from_dlpack_with_strides` preserve native
E4M3FN Torch tensors as `kFloat8_e4m3`. Do not change
`TritonTensorToDLManagedTensor`: TurboMind FP8 tensors retain their existing
opaque-uint8 export contract. The standalone API uses caller-owned Torch FP8
outputs, so it only requires the new import direction.

The fixture continues to use the existing CUDA quantizers, but their physical
destinations must be caller-owned Torch tensors. Add an overload of the
existing `from_dlpack` binding that gives contiguous physical storage an
equal-byte-size logical shape and dtype:

```cpp
m.def(
    "from_dlpack",
    [](py::object obj,
       std::vector<ft::core::ssize_t> logical_shape,
       ft::DataType logical_dtype) {
        py::capsule cap = obj.attr("__dlpack__")();
        auto* dlmt = static_cast<DLManagedTensor*>(
            PyCapsule_GetPointer(cap.ptr(), kDlTensorCapsuleName));
        auto physical = DLManagedTensorToTritonTensor(dlmt);
        cap.set_name("used_dltensor");

        ft::core::Layout logical_layout{std::move(logical_shape)};
        if (physical->byte_size()
            != ft::byte_size(
                logical_dtype,
                logical_layout.cosize())) {
            throw py::value_error(
                "physical and logical byte sizes differ");
        }

        return std::make_shared<Tensor>(
            physical->buffer().view(logical_dtype),
            std::move(logical_layout));
    },
    "dl_managed_tensor"_a,
    "logical_shape"_a,
    "logical_dtype"_a);
```

This overload allocates nothing and retains the DLPack owner through the
returned `Tensor` buffer. It is used only by fixture-side data generation; the
standalone `Linear` API still accepts ordinary Torch tensors and never exposes
logical relabeling.

`TrivialFormat` itself is public to format-specific wrappers, so its
constructor accepts a Torch dtype rather than exposing `_turbomind.DataType`.
Update `lmdeploy/turbomind/weight_format.py` with the complete concrete mapping
used by that format:

```python
_TRIVIAL_DTYPES = {
    torch.float16: _tm.DataType.TYPE_FP16,
    torch.bfloat16: _tm.DataType.TYPE_BF16,
    torch.float32: _tm.DataType.TYPE_FP32,
}


class TrivialFormat(WeightFormat):
    name = 'trivial'
    suffix_map = {'.weight': 'weight', '.bias': 'bias'}
    scales_dtype = _tm.DataType.TYPE_INVALID
    zeros_dtype = _tm.DataType.TYPE_INVALID

    def __init__(self, *, dtype: torch.dtype):
        try:
            self.weight_dtype = _TRIVIAL_DTYPES[dtype]
        except KeyError:
            raise TypeError(
                f'unsupported trivial weight dtype: {dtype}'
            ) from None
        self.dtype = dtype
        super().__init__()

    def normalize(self, tensor: Tensor, kind: str) -> Tensor:
        tensor = tensor.to(self.dtype)
        if tensor.dim() >= 2:
            tensor = tensor.t()
        return tensor
```

The cast is part of trivial-format normalization, not standalone-API special
handling. It guarantees that physical source bytes, the concrete
`weight_dtype`, and the logical allocation agree before `copy_bytes_on_stream`.
If the separately requested computation `dtype` is incompatible with that
concrete format, `plan_weight` returns no family; Python does not add a second
dtype-consistency check.

Update every existing constructor call to supply its already available Torch
dtype:

```python
# converter.py: executable-dtype probe
TrivialFormat(dtype=getattr(torch, name))

# converter.py: selected text and vision resolvers
TrivialFormat(dtype=dtype)
TrivialFormat(dtype=vision_dtype)

# linear.py: dequantization result
TrivialFormat(dtype=dtype)

# qwen3_5.py: explicit vision weight
TrivialFormat(dtype=self._ctx.dtype)

# focused format test
fp16 = TrivialFormat(dtype=torch.float16)
bf16 = TrivialFormat(dtype=torch.bfloat16)
```

`dtype` is the model computation dtype. `input_dtype` is only a family
selection preference and defaults to `dtype`. `output_dtype` is a hard
requirement for the family's ordinary GEMM output and also defaults to `dtype`.
`dtype` is required: it cannot be inferred reliably from quantized weight
storage, whose integer/FP8 payload and scale dtype do not define the model
computation dtype.
After `fuse_weight` selects an epilogue, the prepared C++ weight records that
epilogue's native output format. In particular, a fused dynamic-FP8 epilogue
exposes E4M3 and its scales instead of being converted back to the requested
ordinary output dtype.

`prepare_weight` accepts the tensors in the layout native to the supplied
`WeightFormat`. It calls `normalize(tensor, kind)` for each present tensor to
produce the existing TurboMind source contract:

- floating weight: logical `[K, N]`;
- quantized weight: logical unpacked `[K, N]` storage;
- scales and zeros: `[K / block_k, N / block_n]`;
- grouped input: one such tensor per expert.

The supplied `WeightFormat` creates the C++ descriptor, and logical dimensions
come directly from the normalized weight:

```python
normalized_weight = weight_format.normalize(weight, 'weight')
input_dim = normalized_weight.shape[0]
output_dim = normalized_weight.shape[1]
data_format = weight_format.make_data_format()
```

For example, `TrivialFormat.normalize` transposes a PyTorch `[N, K]` weight to
TurboMind `[K, N]`; `AWQFormat.normalize` expands its checkpoint `int32`
storage and leaves its already-TurboMind logical axes unchanged. The
standalone API does not guess layouts or infer quantization formats.

Preparation rejects:

- tensors not on `Linear.device`;
- mismatched grouped sequence lengths or expert shapes;
- destination byte size different from source `tensor.nbytes`;
- unsupported formats, plans, or epilogues.

Family minimum, alignment, and format-block dimension constraints are an
unchecked precondition of preparation. The TurboMind model path enforces them
through the resolved plan constraint in `Builder._add_linear` and
`_pad_ffn_for_tp`. External callers query the same resolved constraint through
`WeightPlan.shape_constraints` and pad before preparation. Violating this
precondition is undefined behavior; preparation does not add another
dimension-validation path.

Do not add Python or fixture checks comparing actual tensor dtypes with the
format, family, activation, scale, or output dtypes. Supported combinations
flow through the TurboMind descriptors and kernel-feasibility path; storage
that violates an explicitly documented unchecked precondition is undefined
behavior. Python checks are limited to structure, lifecycle, device/layout
requirements needed to form descriptors, and byte-size equality for the
physical copy.

## Ownership and context lifetime

`Linear` records one CUDA device and is immediately usable after construction.
It does not capture, create, or own a CUDA stream. Every operation uses
`torch.cuda.current_stream(self.device)` at the time of that operation. It has
no custom `__enter__` or `__exit__`; resource lifetime is controlled by
explicit idempotent `close()`.

TurboMind needs a `core::Stream` in its thread-local `Context`, but the current
`core::Stream` always owns and destroys its CUDA handle. Add a non-owning
constructor:

```cpp
// stream.h
class StreamImpl {
public:
    explicit StreamImpl(int priority): stream_{}, owned_{true}
    {
        TM_CUDA_CHECK(cudaStreamCreateWithPriority(
            &stream_, cudaStreamNonBlocking, priority));
    }

    explicit StreamImpl(cudaStream_t stream):
        stream_{stream}, owned_{false}
    {
    }

    ~StreamImpl()
    {
        if (owned_) {
            if (auto ec = cudaStreamDestroy(stream_); ec != cudaSuccess) {
                TM_LOG_ERROR("{}", cudaGetErrorString(ec));
            }
        }
        stream_ = {};
    }

    void Sync()
    {
        TM_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    void Wait(const Event& event);

    cudaStream_t handle() const
    {
        return stream_;
    }

private:
    cudaStream_t stream_;
    bool         owned_;
};

class Stream {
public:
    Stream() = default;
    explicit Stream(cudaStream_t stream);
    static Stream create(int priority = 0);

    void Sync()
    {
        impl_->Sync();
    }

    void Wait(const Event& event)
    {
        impl_->Wait(event);
    }

    cudaStream_t handle() const
    {
        return TM_CHECK_NOTNULL(impl_)->handle();
    }

    explicit operator cudaStream_t() const
    {
        return handle();
    }

    explicit operator bool() const noexcept
    {
        return static_cast<bool>(impl_);
    }

    friend bool operator==(const Stream& a, const Stream& b)
    {
        return a.impl_ == b.impl_;
    }

    friend bool operator!=(const Stream& a, const Stream& b)
    {
        return !(a == b);
    }

    friend std::ostream& operator<<(std::ostream& os, const Stream& s)
    {
        os << s.impl_;
        return os;
    }

private:
    shared_ptr<StreamImpl> impl_;
};
```

```cpp
// stream.cc
Stream::Stream(cudaStream_t handle)
{
    impl_ = std::make_shared<StreamImpl>(handle);
}
```

The `cudaStream_t` constructor is explicitly non-owning. Copies of the
resulting `Stream` share the wrapper, but none destroys the Torch-owned CUDA
stream. A zero CUDA handle is accepted because it is the valid legacy default
stream.

Replace the no-argument standalone context binding with a binding that accepts
the Torch stream pointer:

```cpp
m.def(
    "create_device_context",
    [](std::uintptr_t stream_ptr) {
        auto stream = ft::core::Stream{
            reinterpret_cast<cudaStream_t>(stream_ptr)};
        return std::make_unique<PyContextGuard>(
            stream,
            ft::core::Allocator{ft::kCPU},
            ft::core::Allocator{stream, true});
    },
    "stream_ptr"_a,
    "Create a ContextGuard over a caller-owned CUDA stream and the "
    "device default memory pool. The caller retains stream ownership.");
```

Do not retain the no-argument form as an overload. Its only repository caller
is the existing `tests/turbomind/linear/linear.py` implementation being
replaced by this plan. Keeping it would preserve the internal-stream and
internal-allocation behavior this API removes.

`Allocator{stream, true}` calls `cudaDeviceGetDefaultMemPool`. It does not
create or destroy a CUDA memory pool. This is the CUDA device default pool; it
does not mean PyTorch's native caching allocator when PyTorch is configured to
use that separate backend.

One private context method is shared by construction, preparation, execution,
and destruction:

```python
@contextmanager
def _activate(self):
    with torch.cuda.device(self.device):
        stream = torch.cuda.current_stream(self.device)
        with _tm().create_device_context(stream.cuda_stream):
            yield stream
```

The C++ `Context` remains thread-local, but it exists only for the duration of
one Python operation. Sequential calls may therefore originate from different
threads or use different streams. Concurrent calls are addressed below.

`__init__` creates the reusable GEMM registry and `LlamaLinear` workspace under
the stream current during construction, then immediately exits that temporary
context:

```python
if isinstance(device, int):
    resolved_device = torch.device('cuda', device)
else:
    resolved_device = torch.device(device)
if resolved_device.type != 'cuda':
    raise ValueError(f'Linear requires a CUDA device, got {resolved_device}')
if resolved_device.index is None:
    resolved_device = torch.device(
        'cuda', torch.cuda.current_device())
self.device = resolved_device
self._closed = False
self._gemm = None
self._impl = None
with self._activate():
    try:
        self._gemm = _tm().Gemm()
        self._impl = _tm().LlamaLinear()
    except Exception:
        self._impl = None
        self._gemm = None
        self._closed = True
        raise
```

The existing `LlamaLinear` constructor synchronizes its workspace allocation,
so the workspace is initialized before a later operation uses another stream.
This constructor synchronization is not a Python stream bridge and is not
expanded by this plan.

If construction fails, destroy every successfully created object while the
temporary TurboMind context is still active and leave the instance closed.

One `Linear` may prepare and execute any number of weights on its device. It is
the shareable execution object for wrappers that want one kernel registry,
dispatch cache, tuning-record set, and workspace. A wrapper receives that
object from its adapter rather than constructing an instance for every model
projection.

A `WeightPlan` is created and consumed by the same open `Linear`. It may be
reused sequentially to prepare multiple weights with the captured contract.
Using it with another executor or concurrently is an unchecked precondition
violation; preparation does not reselect or compare a family. Once prepared, a
`Weight` follows the separate same-device reuse contract below.

The adapter or engine that creates a shared `Linear` owns and closes it. Each
wrapper independently owns and closes its prepared `Weight`; closing the
executor does not close or invalidate weights. A wrapper does not close an
injected `Linear`. Independent caches or simultaneous execution require
separate executor instances.

`Weight.close()` is idempotent. A grouped weight really aliases every expert's
prepared buffers, so `_experts` retains those handles. Close the linked
pointer-table object before releasing its aliased experts. Weight construction
always occurs under `Linear._activate`; explicit destruction must occur while
that same CUDA device is current:

```python
def close(self):
    if self._impl is None:
        return
    experts = self._experts
    self._impl = None
    self._experts = None
    if experts is not None:
        for expert in experts:
            expert.close()
```

Expert handles remain private children of their grouped handle. Dense and
grouped execution reject a weight whose `_impl` is `None`.

`close()` is idempotent and destroys objects under a temporary context for the
stream current at close time:

```python
if self._closed:
    return
with self._activate():
    self._impl = None
    self._gemm = None
self._closed = True
```

Except for another idempotent `close()`, invoking a public operation after
`Linear.close()` is an unchecked precondition violation and undefined
behavior. Public methods do not add repeated `_closed` branches.

Prepared `Weight` objects remain valid after `Linear.close` and may be used by
another open `Linear` on the same device. They retain no dispatch cache or
workspace from the executor that prepared them. Using a prepared weight with a
`Linear` on any other device is an unchecked precondition violation and is
undefined behavior.

Each C++ buffer allocator already captures the non-owning CUDA stream handle
used for its stream-ordered deallocation. The grouped strided-pointer owner
does the same. Python stores no Torch stream on `Weight`. The caller keeps any
externally owned CUDA stream valid through `Weight.close()`; ordinary Torch
CUDA streams remain owned by Torch's process-lifetime stream pool. Destruction
of `LlamaLinear` enqueues its workspace frees on the Torch stream current
during `close()`.

The API adds no stream wait, event, or synchronization. `Weight.close()` and
`Linear.close()` have a strict quiescence precondition: all CUDA work that uses
the object must already have completed. Ordering work onto another stream is
not sufficient. The caller synchronizes before destruction when work may be
in flight. `Weight.close()` must additionally execute with the same CUDA device
current that was current during preparation; this is an unchecked precondition
and `Weight` does not retain or validate a Python device field:

```python
torch.cuda.synchronize(linear.device)
with torch.cuda.device(linear.device):
    weight.close()
```

Prepared packed tensors and pointer tables are persistent state owned only by
the opaque `Weight` until `Weight.close()`; their pointers never cross the API
boundary. Execution and output-query calls create no persistent allocation.
Every forward-time temporary has its ownership released, with any CUDA free
ordered on the current stream, before the call returns. `out` is always
caller-owned, and native FP8 output additionally requires caller-owned
`out_scales`.

One `_tm.LlamaLinear` owns one mutable GEMM workspace. A `Linear` may switch
current streams between calls, but it must not have simultaneous in-flight
forwards, including calls made by different wrappers sharing that `Linear`, on
multiple streams. When switching streams, the caller establishes ordering
explicitly:

```python
with torch.cuda.stream(stream_a):
    output_a, scales_a = linear(
        x_a,
        weight,
        out=output_buffer_a,
        out_scales=scale_buffer_a,
    )

stream_b.wait_stream(stream_a)
with torch.cuda.stream(stream_b):
    output_b, scales_b = linear(
        x_b,
        weight,
        out=output_buffer_b,
        out_scales=scale_buffer_b,
    )
```

Concurrent multi-stream execution requires separate workspaces and is outside
this refactor.

Do not add a CUDA-running `__del__`; interpreter shutdown cannot guarantee an
active CUDA runtime or the originating thread. A live `Weight` reaching Python
garbage collection would still release its pybind-owned native object, and a
live `Linear` would destroy `LlamaLinear` without the temporary TurboMind
context required by its destructor. Callers must explicitly close every weight
on its construction device and explicitly call `Linear.close()`. Allowing
implicit destruction of either live object is an unchecked precondition
violation and undefined behavior. `Linear.close()` activates its stored device
and installs the required temporary context itself. Callers wanting lexical
cleanup may use the standard library without adding a custom protocol:

```python
from contextlib import closing

with torch.cuda.device(device):
    with closing(Linear(device)) as linear:
        dtype = torch.bfloat16
        plan = linear.plan_weight(
            weight_format=TrivialFormat(dtype=dtype),
            dtype=dtype,
        )
        weight = linear.prepare_weight(
            torch_weight,
            plan=plan,
        )
        with closing(weight):
            output_meta, _ = linear.output_spec(x, weight)
            output = torch.empty_strided(
                output_meta.shape,
                output_meta.stride(),
                dtype=output_meta.dtype,
                device=x.device,
            )
            output, output_scales = linear(
                x,
                weight,
                out=output,
            )
            torch.cuda.synchronize(linear.device)
```

## Weight planning and preparation

The selected plan owns the effective local weight-shape constraint resolved
from both the family and the source format. Dimensions are not inputs to family
selection. Add `<array>` to `plan.h` and expand `GemmPlan` to:

```cpp
class GemmPlan {
public:
    const Family& family() const noexcept
    {
        return *family_;
    }

    std::array<int, 4> shape_constraints() const noexcept
    {
        return {min_k_, min_n_, align_k_, align_n_};
    }

    int gate_up_block(ActivationType act_type) const;
    int gate_up(ActivationType act_type, int projection_n);
    void pack(LinearWeight& linear, cudaStream_t stream) const;

private:
    friend class Gemm;

    const Family* family_{};
    WeightBridge  bridge_{};
    Epilogue      epilogue_{Epilogue::kNone};
    DataFormat    output_format_{};
    int           min_k_{};
    int           min_n_{};
    int           align_k_{};
    int           align_n_{};
};
```

`Family::gate_up` does not use its current integer argument. Remove that
argument from its declaration and definition:

```cpp
int Family::gate_up(
    ActivationType act_type,
    Epilogue& epilogue) const
{
    if (gate_up_block_ && act_type == ActivationType::kSilu) {
        epilogue = Epilogue::kGatedSilu;
        return gate_up_block_;
    }
    epilogue = Epilogue::kNone;
    return 0;
}
```

The read-only block query does not mutate the plan. The existing mutating call
continues to publish the selected epilogue and fused output format:

```cpp
int GemmPlan::gate_up_block(ActivationType act_type) const
{
    Epilogue epilogue = Epilogue::kNone;
    return family_->gate_up(act_type, epilogue);
}

int GemmPlan::gate_up(
    ActivationType act_type,
    int projection_n)
{
    epilogue_ = Epilogue::kNone;
    output_format_ = family_->output_format(Epilogue::kNone);

    Epilogue epilogue = Epilogue::kNone;
    const int block = family_->gate_up(act_type, epilogue);
    if (!block || projection_n % block) {
        return 0;
    }

    epilogue_ = epilogue;
    output_format_ = family_->output_format(epilogue);
    return block;
}
```

After `PlanWeight` has selected the family, resolve the constraint from the
same query that selected it. Add `<numeric>` to `gemm.cu`:

```cpp
GemmPlan plan;
plan.family_ = selected;
plan.bridge_ = selected_bridge;
plan.output_format_ = selected->output_format(Epilogue::kNone);
plan.min_k_ = selected->min_k();
plan.min_n_ = selected->min_n();
plan.align_k_ = std::lcm(
    selected->align_k(),
    query.weight_format.block_sizes[0]);
plan.align_n_ = std::lcm(
    selected->align_n(),
    query.weight_format.block_sizes[1]);
return plan;
```

The selected family has already accepted the source `DataFormat`, including
its rank and block-size structure, before this block executes.

Expose the resolved values and the non-mutating fusion block on the existing
`GemmPlan` binding:

```cpp
py::class_<gemm::GemmPlan>(m, "GemmPlan")
    .def_property_readonly(
        "family",
        &gemm::GemmPlan::family,
        py::return_value_policy::reference_internal)
    .def_property_readonly(
        "shape_constraints",
        [](const gemm::GemmPlan& plan) {
            const auto values = plan.shape_constraints();
            return py::make_tuple(
                py::make_tuple(values[0], values[1]),
                py::make_tuple(values[2], values[3]));
        })
    .def(
        "gate_up_block",
        &gemm::GemmPlan::gate_up_block,
        "act_type"_a)
    .def(
        "gate_up",
        &gemm::GemmPlan::gate_up,
        "act_type"_a,
        "projection_n"_a)
    .def(
        "pack",
        [](const gemm::GemmPlan& plan, LinearWeight& linear) {
            plan.pack(
                linear,
                core::Context::stream().handle());
        },
        "linear"_a,
        py::call_guard<py::gil_scoped_release>());
```

`Linear.plan_weight` is the only family-selection call in the standalone API.
It derives the C++ source format from the explicitly supplied `WeightFormat`,
constructs the complete dimension-free query, resolves optional fusion
constraints, and captures the result in one opaque `WeightPlan`:

```python
def plan_weight(
    self,
    *,
    weight_format,
    dtype,
    input_dtype=None,
    output_dtype=None,
    grouped=False,
    fusion_type=None,
):
    weight_format = copy.copy(weight_format)
    data_format = weight_format.make_data_format()
    query = _tm().WeightQuery()
    query.weight_format = data_format
    query.data_type = _to_tm_dtype(dtype)
    query.input_dtype = _to_tm_dtype(
        dtype if input_dtype is None else input_dtype)
    query.output_dtype = _to_tm_dtype(
        dtype if output_dtype is None else output_dtype)
    query.grouped = grouped

    impl = self._gemm.plan_weight(query)
    if impl is None:
        raise NotImplementedError(
            f'no GEMM family accepts format={data_format}, '
            f'dtype={dtype}, input_dtype={input_dtype}, '
            f'output_dtype={output_dtype}, grouped={grouped}')

    minimum, alignment = impl.shape_constraints
    min_k, min_n = minimum
    align_k, align_n = alignment

    if fusion_type is not None and fusion_type != 'silu':
        raise NotImplementedError(
            f'unsupported weight fusion: {fusion_type!r}')
    if fusion_type == 'silu':
        block_out = weight_format.block_out or 1
        gate_up_block = impl.gate_up_block(
            _tm().ActivationType.kSilu)
        if not gate_up_block or gate_up_block % block_out:
            raise NotImplementedError(
                'selected family cannot represent SiLU fusion')
        minimum = (
            min_k,
            (min_n + 1) // 2,
        )
        alignment = (
            align_k,
            math.lcm(
                block_out,
                gate_up_block,
                align_n // math.gcd(align_n, 2),
            ),
        )

    result = WeightPlan()
    result._impl = impl
    result._weight_format = weight_format
    result._dtype = dtype
    result._grouped = grouped
    result._fusion_type = fusion_type
    result._shape_constraints = (minimum, alignment)
    return result
```

Planning does not pad or inspect a tensor. A wrapper preparing a full
pre-shard tensor multiplies the appropriate returned minimum and alignment by
its TP size before padding, exactly as the builder does. A wrapper querying
already-sharded tensors uses the returned local values directly. The exact
`WeightPlan` used to obtain those values is then passed to preparation; neither
preparation method performs another family-selection call.

Replace the family/format calculation in `Builder._add_linear` with the
resolved plan property. Its existing TP adjustment and validation remain in
place:

```python
((min_k, min_n),
 (align_k, align_n)) = plan.shape_constraints
tp = self.tp.size if split_side else 1
k = int(w.shape[0])
n = int(w.shape[-1])
if split_side == SplitSide.INPUT:
    min_k *= tp
    align_k *= tp
elif split_side == SplitSide.OUTPUT:
    min_n *= tp
    align_n *= tp
if k < min_k or k % align_k or n < min_n or n % align_n:
    raise RuntimeError(
        f'{name}: {plan.family} requires K >= {min_k}, '
        f'K % {align_k} == 0, N >= {min_n}, and '
        f'N % {align_n} == 0; got K={k} N={n}')
```

`_pad_ffn_for_tp` consumes the same resolved constraint instead of reading
family fields. The combined gate/up storage sees `2 * projection_n`, while the
down projection sees the same intermediate dimension as K:

```python
def _pad_ffn_for_tp(
    w1: Linear,
    w2: Linear,
    w3: Linear,
    tp: int,
    plan,
) -> tuple[Linear, Linear, Linear]:
    """Pad the intermediate axis to the plan's TP-local shape contract."""
    ((min_k, min_n),
     (align_k, align_n)) = plan.shape_constraints

    fmt = w1.weight_format
    block_in = fmt.block_in or 1
    block_out = fmt.block_out or 1
    unit = math.lcm(block_in, block_out)

    projection_min_n = (min_n + 1) // 2
    projection_align_n = math.lcm(
        block_out,
        align_n // math.gcd(align_n, 2),
    )
    minimum = max(min_k, projection_min_n)
    granularity = math.lcm(align_k, projection_align_n)

    raw = int(w1.tensors['weight'].size(-1))
    divisor = granularity * tp
    target = max(raw, minimum * tp)
    target = (target + divisor - 1) // divisor * divisor

    groups = raw // unit
    target_groups = target // unit
    w1 = pad_output_groups(
        w1,
        src_groups=groups,
        dst_groups=target_groups,
    )
    w3 = pad_output_groups(
        w3,
        src_groups=groups,
        dst_groups=target_groups,
    )
    w2 = pad_input_groups(
        w2,
        src_groups=groups,
        dst_groups=target_groups,
    )
    return w1, w2, w3
```

Pass the selected plan at the existing call site:

```python
plan = self._query_gemm(self._make_gemm_query(
    w1,
    grouped=self.config.is_expert,
))
w1, w2, w3 = _pad_ffn_for_tp(
    w1,
    w2,
    w3,
    self.tp.size,
    plan,
)
```

The private allocation path creates `LinearConfig`, constructs
`_tm.LinearWeight`, sets the selected plan, allocates each present parameter,
copies its bytes on the TurboMind stream, and calls `prepare()`.

Before planning and allocation, normalize each raw wrapper tensor exactly
once. This method is shared by ordinary preparation and by each gate/up
component of fused preparation:

```python
def _normalize_params(
    self,
    weight_format,
    weight,
    scales,
    zeros,
):
    raw = {
        'weight': weight,
        'scales': scales,
        'zeros': zeros,
    }
    raw = {
        kind: tensor
        for kind, tensor in raw.items()
        if tensor is not None
    }
    for kind, tensor in raw.items():
        if tensor.device != self.device:
            raise ValueError(f'{kind} is on the wrong device')

    normalized = {
        kind: weight_format.normalize(tensor, kind).contiguous()
        for kind, tensor in raw.items()
    }
    tm = _tm()
    if (
        weight_format.scales_dtype != tm.DataType.TYPE_INVALID
        and 'scales' not in normalized
    ):
        raise ValueError('the selected weight format requires scales')
    if (
        weight_format.zeros_dtype != tm.DataType.TYPE_INVALID
        and 'zeros' not in normalized
    ):
        # Omission is valid only for formats whose existing implementation
        # synthesizes symmetric zeros. Required asymmetric zeros are an
        # unchecked caller precondition.
        normalized['zeros'] = weight_format.synthesize_zeros(
            normalized['scales']).contiguous()
    if 'zeros' in normalized:
        normalized['zeros'] = normalized['zeros'].to(
            normalized['scales'].dtype).contiguous()

    self._validate_normalized_params(
        weight_format,
        normalized,
    )
    return normalized
```

The shared validation is structural only. It is reused by ordinary and fused
preparation and does not inspect a tensor dtype or duplicate a family's
minimum/alignment requirements:

```python
def _validate_normalized_params(self, weight_format, params):
    weight = params['weight']
    if weight.ndim != 2:
        raise ValueError('normalized weight must be two-dimensional')

    input_dim = weight.shape[0]
    output_dim = weight.shape[1]
    expected_qparams = (
        (input_dim + (weight_format.block_in or 1) - 1)
        // (weight_format.block_in or 1),
        (output_dim + (weight_format.block_out or 1) - 1)
        // (weight_format.block_out or 1),
    )

    tm = _tm()
    expects_scales = (
        weight_format.scales_dtype != tm.DataType.TYPE_INVALID)
    expects_zeros = (
        weight_format.zeros_dtype != tm.DataType.TYPE_INVALID)
    if ('scales' in params) != expects_scales:
        raise ValueError(
            'scales presence does not match the selected weight format')
    if ('zeros' in params) != expects_zeros:
        raise ValueError(
            'zeros presence does not match the selected weight format')

    for kind in ('scales', 'zeros'):
        if kind in params and tuple(params[kind].shape) != expected_qparams:
            raise ValueError(
                f'{kind} shape {tuple(params[kind].shape)} does not '
                f'match normalized weight shape {tuple(weight.shape)}')
```

Family `min_k`, `min_n`, `align_k`, `align_n`, and format-block dimension
constraints are exposed by `WeightPlan.shape_constraints` but are not checked
again during preparation. Python does not reject a structurally valid matrix
based on those values.

This is the same normalization/packing sequence used by the TurboMind model
builder: `WeightFormat.normalize` converts format-native storage into logical
TM layout, then `WeightFormat.pack` performs physical relabeling such as UINT4
`uint8` to `int32` row packing.

Raw format-native tensors may be strided. Normalization is followed by
`.contiguous()` because formats such as `TrivialFormat` deliberately transpose
`[N, K]` into a `[K, N]` view. This is part of the one-time preparation copy,
not a forward-time hidden copy. Every `PackedTensor.tensor` returned by
`WeightFormat.pack` must be contiguous before `_copy_param`; a non-contiguous
packed result is a format implementation error and is rejected.

`_prepare_one` accepts only normalized tensors. It performs physical packing,
allocation, byte copies, and C++ preparation; it never calls
`WeightFormat.normalize`:

```python
def _prepare_one(
    self,
    normalized,
    *,
    weight_format,
    plan,
    dtype,
    stored_output_dim,
    stream,
):
    source_weight = normalized['weight']
    input_dim = source_weight.shape[0]

    data_format = weight_format.make_data_format()
    config = _tm().LinearConfig()
    config.input_dim = input_dim
    config.output_dim = stored_output_dim
    config.data_type = _to_tm_dtype(dtype)
    config.format = data_format
    config.has_bias = False

    impl = _tm().LinearWeight(config)
    impl.set_plan(plan)
    packed = {
        kind: weight_format.pack(tensor, kind)
        for kind, tensor in normalized.items()
    }
    for kind, item in packed.items():
        tensor = item.tensor
        if not tensor.is_contiguous():
            raise ValueError(
                f'{kind}: packed tensor must be contiguous')
        logical_shape = (
            list(tensor.shape)
            if item.alloc_shape is None
            else list(item.alloc_shape)
        )
        logical_dtype = (
            _to_tm_dtype(tensor.dtype)
            if item.alloc_dtype is None
            else item.alloc_dtype
        )
        self._copy_param(
            impl,
            kind,
            tensor,
            logical_shape=logical_shape,
            logical_dtype=logical_dtype,
            stream=stream,
        )

    impl.prepare()

    handle = Weight()
    handle._impl = impl
    handle._experts = None
    return handle
```

The plan is set before any call to `prepare()`. `prepare()` is called exactly
once per expert after all present parameter copies have been enqueued.

The complete ordinary public method classifies the outer structure before
entering one activation scope. A tensor is dense; every other accepted weight
is a non-empty `Sequence` of tensors. Present companions must have the same
outer structure and expert count:

```python
def prepare_weight(
    self,
    weight,
    *,
    plan,
    scales=None,
    zeros=None,
):
    if plan._fusion_type is not None:
        raise ValueError(
            'prepare_weight requires a non-fused WeightPlan')
    weight_format = plan._weight_format
    dtype = plan._dtype
    impl_plan = plan._impl

    if isinstance(weight, torch.Tensor):
        if plan._grouped:
            raise ValueError(
                'grouped WeightPlan requires an expert sequence')
        if scales is not None and not isinstance(scales, torch.Tensor):
            raise TypeError(
                'dense scales must be a tensor or None')
        if zeros is not None and not isinstance(zeros, torch.Tensor):
            raise TypeError(
                'dense zeros must be a tensor or None')

        with self._activate() as stream:
            normalized = self._normalize_params(
                weight_format,
                weight,
                scales,
                zeros,
            )
            return self._prepare_one(
                normalized,
                weight_format=weight_format,
                plan=impl_plan,
                dtype=dtype,
                stored_output_dim=normalized['weight'].shape[1],
                stream=stream,
            )

    if not isinstance(weight, Sequence):
        raise TypeError(
            'weight must be a tensor or a sequence of tensors')
    if not plan._grouped:
        raise ValueError(
            'dense WeightPlan requires one weight tensor')
    weights = list(weight)
    if not weights:
        raise ValueError('grouped weight must contain experts')
    if any(not isinstance(item, torch.Tensor) for item in weights):
        raise TypeError('every grouped weight must be a tensor')

    if scales is None:
        expert_scales = [None] * len(weights)
    else:
        if isinstance(scales, torch.Tensor) or not isinstance(scales, Sequence):
            raise TypeError('grouped scales must be a sequence or None')
        expert_scales = list(scales)
        if len(expert_scales) != len(weights):
            raise ValueError(
                'grouped scales must match the expert count')
        if any(not isinstance(item, torch.Tensor)
               for item in expert_scales):
            raise TypeError('every grouped scale must be a tensor')

    if zeros is None:
        expert_zeros = [None] * len(weights)
    else:
        if isinstance(zeros, torch.Tensor) or not isinstance(zeros, Sequence):
            raise TypeError('grouped zeros must be a sequence or None')
        expert_zeros = list(zeros)
        if len(expert_zeros) != len(weights):
            raise ValueError(
                'grouped zeros must match the expert count')
        if any(not isinstance(item, torch.Tensor)
               for item in expert_zeros):
            raise TypeError('every grouped zero must be a tensor')

    with self._activate() as stream:
        normalized_experts = [
            self._normalize_params(
                weight_format,
                expert_weight,
                expert_scale,
                expert_zero,
            )
            for expert_weight, expert_scale, expert_zero in zip(
                weights,
                expert_scales,
                expert_zeros,
            )
        ]
        return self._prepare_grouped(
            normalized_experts,
            weight_format=weight_format,
            plan=impl_plan,
            dtype=dtype,
            stored_output_dim=(
                normalized_experts[0]['weight'].shape[1]),
            stream=stream,
        )
```

For a grouped handle, `_impl` is the linked strided-pointer view and `_experts`
owns the prepared expert handles. Dense handles set `_experts = None`.

Add this binding beside `generic_copy_on_stream` in
`src/turbomind/python/bind.cpp`:

```cpp
#include <fmt/format.h>
```

```cpp
m.def(
    "copy_bytes_on_stream",
    [](py::object src_obj,
       std::shared_ptr<Tensor> dst,
       std::uintptr_t stream_ptr) {
        py::capsule cap = src_obj.attr("__dlpack__")();
        auto* dlmt = static_cast<DLManagedTensor*>(
            PyCapsule_GetPointer(cap.ptr(), kDlTensorCapsuleName));
        auto src = DLManagedTensorToTritonTensor(dlmt);
        cap.set_name("used_dltensor");

        if (dst->byte_size() != src->byte_size()) {
            throw py::value_error(fmt::format(
                "destination has {} bytes, source has {} bytes",
                dst->byte_size(),
                src->byte_size()));
        }
        TM_CUDA_CHECK(cudaMemcpyAsync(
            dst->raw_data(),
            src->raw_data(),
            dst->byte_size(),
            cudaMemcpyDefault,
            reinterpret_cast<cudaStream_t>(stream_ptr)));
    },
    "src"_a,
    "dst"_a,
    "stream_ptr"_a);
```

This binding deliberately performs no dtype conversion or layout operation.
The Python preparation path requires contiguous input and validates the
logical allocation before calling it.

One shared `_copy_param` function is retained because it is used for weight,
scales, and zeros. It enforces the physical/logical relabel contract:

```python
dst = impl.param(name).alloc(logical_shape, logical_dtype)
if dst.byte_size != src.nbytes:
    raise ValueError(
        f'{name}: destination has {dst.byte_size} bytes, '
        f'source has {src.nbytes} bytes')
_tm().copy_bytes_on_stream(src, dst, stream.cuda_stream)
```

Weight preparation does not synchronize. Normalization, packing, copies, and
later TurboMind preparation are ordered on the current Torch stream. A caller
that produces a source tensor on another stream must order that producer before
the stream current during preparation. The API does not call `record_stream`,
insert a wait, or retain caller tensors after the method returns.

All preparation tensors must already be CUDA tensors on `Linear.device`.
Preparation performs no implicit CPU-to-GPU or cross-device transfer; the
caller owns checkpoint transfer and TP sharding before invoking this API.

Preparation never mutates the supplied tensors and TurboMind always owns the
destination storage; there is no family-dependent zero-copy path. The caller
must nevertheless keep `weight`, `scales`, and `zeros` alive
until the preparation stream has completed their asynchronous uses.
Normalization and packing temporaries created by the API on that same stream
use normal stream-ordered PyTorch deallocation. If caller storage is used on a
stream different from its allocation stream, the caller owns the usual
PyTorch lifetime handling, such as retaining it until completion or recording
that stream.

Grouped weights retain every prepared expert allocation through their private
expert handles until the grouped handle closes. Python retains no stream
object.

Grouped preparation plans once with `query.grouped=True`, prepares every
expert with the same plan, and creates the fused strided-pointer view.
`_prepare_grouped` receives a list of dictionaries already returned by
`_normalize_params`; it calls `_prepare_one` for each expert and never accepts
raw tensors. The pointer and metadata manipulation remains private. Empty
expert sequences and non-uniform expert metadata are rejected.

Keep expert aggregation in the MoE weight layer. Make the existing static
function reusable without turning it into a `LinearWeight` method:

```cpp
// moe_weight.h
#include <vector>

void LinkLinearExperts(
    const std::vector<LinearWeight*>& experts,
    LinearWeight& destination);
```

```cpp
// moe_weight.cc
void LinkLinearExperts(
    const std::vector<LinearWeight*>& experts,
    LinearWeight& destination)
{
    TM_CHECK(!experts.empty());
    const auto& e0 = *TM_CHECK_NOTNULL(experts.front());
    e0.copy_metadata_to(destination);

    const int n = experts.size();
    destination.k_desc.num = destination.q_desc.num = n;

    if (e0.bias) {
        destination.bias = Tensor{
            {n, e0.output_dim},
            e0.bias.dtype(),
            kDEVICE,
        };
    }

    std::vector<std::pair<void*, int>> weights;
    std::vector<std::pair<void*, int>> scales;
    std::vector<std::pair<void*, int>> global_scales;
    weights.reserve(n);
    scales.reserve(n);
    global_scales.reserve(n);

    for (int i = 0; i < n; ++i) {
        auto& expert = *TM_CHECK_NOTNULL(experts[i]);
        weights.emplace_back(
            expert.weight.raw_data(),
            expert.k_desc.ld);
        if (expert.scales) {
            scales.emplace_back(
                expert.scales.raw_data(),
                expert.q_desc.ld);
        }
        if (expert.global_scale) {
            global_scales.emplace_back(
                expert.global_scale.raw_data(),
                1);
        }
        if (expert.bias) {
            Copy(
                expert.bias,
                destination.bias.slice(i, 1).squeeze(0));
        }
    }

    auto stream = core::Context::stream();
    auto make_strided_ptr = [stream](const auto& ptrs) {
        return std::shared_ptr<void>{
            gemm::MakeStridedPtrs(ptrs, stream.handle()),
            [stream](void* p) {
                if (p) {
                    TM_CUDA_CHECK(cudaFreeAsync(
                        p,
                        stream.handle()));
                }
            },
        };
    };

    destination.weight = Tensor{
        make_strided_ptr(weights),
        {n},
        destination.weight_format.dtype,
        kDEVICE,
    };
    if (e0.scales) {
        TM_CHECK_EQ(static_cast<int>(scales.size()), n);
        destination.scales = Tensor{
            make_strided_ptr(scales),
            {n},
            e0.scales.dtype(),
            kDEVICE,
        };
    }
    if (e0.global_scale) {
        TM_CHECK_EQ(
            static_cast<int>(global_scales.size()),
            n);
        destination.global_scale = Tensor{
            make_strided_ptr(global_scales),
            {n},
            e0.global_scale.dtype(),
            kDEVICE,
        };
    }

    destination.k_desc.ld = destination.q_desc.ld = 0;
    destination.k_desc.offsets =
        destination.q_desc.offsets = nullptr;
}
```

This preserves the model path's bias and global-scale behavior even though the
standalone API excludes both. `MoeWeight::prepare` constructs vectors from its
existing expert accessors and calls this same free function.

Expose only one low-level binding that allocates the destination and invokes
the existing MoE operation; do not bind `MakeStridedPtrs`, a default
`LinearWeight` constructor, metadata setters, or a `LinearWeight` linking
method:

```cpp
#include "src/turbomind/models/moe_weight.h"

m.def(
    "LinkLinearExperts",
    [](const std::vector<LinearWeight*>& experts) {
        auto destination = std::make_unique<LinearWeight>();
        LinkLinearExperts(experts, *destination);
        return destination;
    },
    "experts"_a);
```

This binding is called only inside the sequence implementation of the public
`Linear.prepare_weight`/`Linear.fuse_weight`. The fixture and external callers
call those public methods and never see the linking operation. The C++ linked
weight aliases, rather than copies, the experts' prepared weight/scales
buffers; the returned Python `Weight._experts` list is therefore required
ownership and remains alive until grouped close.

The complete private grouped stage is:

```python
def _prepare_grouped(
    self,
    normalized_experts,
    *,
    weight_format,
    plan,
    dtype,
    stored_output_dim,
    stream,
):
    if not normalized_experts:
        raise ValueError('grouped weight must contain experts')

    signature = {
        kind: tuple(tensor.shape)
        for kind, tensor in normalized_experts[0].items()
    }
    for params in normalized_experts[1:]:
        if {
            kind: tuple(tensor.shape)
            for kind, tensor in params.items()
        } != signature:
            raise ValueError(
                'grouped experts must have identical kinds and shapes')

    experts = []
    impl = None
    try:
        for params in normalized_experts:
            experts.append(self._prepare_one(
                params,
                weight_format=weight_format,
                plan=plan,
                dtype=dtype,
                stored_output_dim=stored_output_dim,
                stream=stream,
            ))

        impl = _tm().LinkLinearExperts(
            [expert._impl for expert in experts])
        handle = Weight()
        handle._impl = impl
        handle._experts = experts
        return handle
    except Exception:
        impl = None
        for expert in reversed(experts):
            expert.close()
        raise
```

Its expert handles remain private children. The grouped close path clears
`handle._impl` first, enqueueing the pointer-table frees, and then closes the
expert handles. The failure path follows the same order while the public
method's `_activate()` scope still has the construction device and TurboMind
context active.

## Fused-weight preparation

`fuse_weight` uses one outer `(gate, up)` tuple to represent the two fused
projections. Every present companion has exactly the same outer structure. A
dense call is:

```python
plan = linear.plan_weight(
    weight_format=AWQFormat(block_in=group_size),
    dtype=torch.bfloat16,
    fusion_type='silu',
)
fused = linear.fuse_weight(
    (gate_weight, up_weight),
    plan=plan,
    scales=(gate_scales, up_scales),
    zeros=(gate_zeros, up_zeros),
)
```

A grouped call keeps the same outer tuple; each element is a non-empty expert
sequence:

```python
plan = linear.plan_weight(
    weight_format=AWQFormat(block_in=group_size),
    dtype=torch.bfloat16,
    grouped=True,
    fusion_type='silu',
)
fused = linear.fuse_weight(
    ([gate_weight_0, gate_weight_1],
     [up_weight_0, up_weight_1]),
    plan=plan,
    scales=([gate_scales_0, gate_scales_1],
            [up_scales_0, up_scales_1]),
    zeros=([gate_zeros_0, gate_zeros_1],
           [up_zeros_0, up_zeros_1]),
)
```

There are no `gate_*` or `up_*` keyword arguments. The outer tuple must contain
exactly two elements; the first is gate and the second is up. Both are tensors
for dense fusion or both are equal-length sequences for grouped fusion; mixing
a tensor and a sequence is rejected. The structure and expert counts of
`scales` and `zeros` must exactly match `weight`.

When `zeros` is omitted for a format that supports symmetric-zero synthesis,
synthesize gate and up zeros independently from their normalized scales before
interleaving, exactly as ordinary `prepare_weight` does. AWQ asymmetric zeros
cannot be derived from scales and must be supplied; omitting them violates an
unchecked caller precondition. A supplied `zeros` tuple is normalized and used
instead.

There is no existing public fusion enum, and the existing `ActivationType` is
not sufficient because a fusion also defines how the two projections are
combined. `Literal['silu']` documents the only accepted value statically;
`plan_weight` handles untyped external callers and captures the selected fusion
mode in `WeightPlan`.

For each normalized tensor kind, gate and up are interleaved by the same
logical number of output groups:

```python
def _interleave_gate_up(self, gate, up, groups):
    gate_groups = gate.unflatten(-1, (groups, -1))
    up_groups = up.unflatten(-1, (groups, -1))
    return torch.stack(
        (gate_groups, up_groups),
        dim=-2,
    ).flatten(-3, -1).contiguous()
```

The complete public method converts dense input into a one-element internal
list so dense and grouped fusion share one normalization and interleaving
path. It does not publish a partially prepared handle:

```python
def fuse_weight(
    self,
    weight,
    *,
    plan,
    scales=None,
    zeros=None,
):
    if plan._fusion_type != 'silu':
        raise ValueError(
            'fuse_weight requires a SiLU-fusion WeightPlan')
    weight_format = plan._weight_format
    dtype = plan._dtype
    impl_plan = plan._impl

    if not isinstance(weight, tuple) or len(weight) != 2:
        raise TypeError('fused weight must be a (gate, up) tuple')
    gate_weight, up_weight = weight

    if isinstance(gate_weight, torch.Tensor) \
            and isinstance(up_weight, torch.Tensor):
        grouped = False
        gate_weights = [gate_weight]
        up_weights = [up_weight]
    elif isinstance(gate_weight, torch.Tensor) \
            or isinstance(up_weight, torch.Tensor):
        raise TypeError(
            'gate and up must both be tensors or both be sequences')
    else:
        if not isinstance(gate_weight, Sequence) \
                or not isinstance(up_weight, Sequence):
            raise TypeError(
                'gate and up must both be tensors or both be sequences')
        grouped = True
        gate_weights = list(gate_weight)
        up_weights = list(up_weight)
        if not gate_weights or not up_weights:
            raise ValueError('grouped fusion must contain experts')
        if len(gate_weights) != len(up_weights):
            raise ValueError(
                'gate and up must have the same expert count')
        if any(not isinstance(item, torch.Tensor)
               for item in gate_weights + up_weights):
            raise TypeError(
                'every grouped gate and up weight must be a tensor')

    if grouped != plan._grouped:
        raise ValueError(
            'weight structure does not match WeightPlan grouped mode')

    count = len(gate_weights)

    if scales is None:
        gate_scales = [None] * count
        up_scales = [None] * count
    else:
        if not isinstance(scales, tuple) or len(scales) != 2:
            raise TypeError(
                'fused scales must be a (gate, up) tuple or None')
        gate_scale_arg, up_scale_arg = scales
        if not grouped:
            if not isinstance(gate_scale_arg, torch.Tensor) \
                    or not isinstance(up_scale_arg, torch.Tensor):
                raise TypeError(
                    'dense fused scales must both be tensors')
            gate_scales = [gate_scale_arg]
            up_scales = [up_scale_arg]
        else:
            if isinstance(gate_scale_arg, torch.Tensor) \
                    or isinstance(up_scale_arg, torch.Tensor) \
                    or not isinstance(gate_scale_arg, Sequence) \
                    or not isinstance(up_scale_arg, Sequence):
                raise TypeError(
                    'grouped fused scales must both be sequences')
            gate_scales = list(gate_scale_arg)
            up_scales = list(up_scale_arg)
            if len(gate_scales) != count or len(up_scales) != count:
                raise ValueError(
                    'fused scales must match the expert count')
            if any(not isinstance(item, torch.Tensor)
                   for item in gate_scales + up_scales):
                raise TypeError(
                    'every grouped fused scale must be a tensor')

    if zeros is None:
        gate_zeros = [None] * count
        up_zeros = [None] * count
    else:
        if not isinstance(zeros, tuple) or len(zeros) != 2:
            raise TypeError(
                'fused zeros must be a (gate, up) tuple or None')
        gate_zero_arg, up_zero_arg = zeros
        if not grouped:
            if not isinstance(gate_zero_arg, torch.Tensor) \
                    or not isinstance(up_zero_arg, torch.Tensor):
                raise TypeError(
                    'dense fused zeros must both be tensors')
            gate_zeros = [gate_zero_arg]
            up_zeros = [up_zero_arg]
        else:
            if isinstance(gate_zero_arg, torch.Tensor) \
                    or isinstance(up_zero_arg, torch.Tensor) \
                    or not isinstance(gate_zero_arg, Sequence) \
                    or not isinstance(up_zero_arg, Sequence):
                raise TypeError(
                    'grouped fused zeros must both be sequences')
            gate_zeros = list(gate_zero_arg)
            up_zeros = list(up_zero_arg)
            if len(gate_zeros) != count or len(up_zeros) != count:
                raise ValueError(
                    'fused zeros must match the expert count')
            if any(not isinstance(item, torch.Tensor)
                   for item in gate_zeros + up_zeros):
                raise TypeError(
                    'every grouped fused zero must be a tensor')

    with self._activate() as stream:
        normalized_pairs = []
        signature = None
        for (gate_weight,
             up_weight,
             gate_scale,
             up_scale,
             gate_zero,
             up_zero) in zip(
                 gate_weights,
                 up_weights,
                 gate_scales,
                 up_scales,
                 gate_zeros,
                 up_zeros,
             ):
            gate = self._normalize_params(
                weight_format,
                gate_weight,
                gate_scale,
                gate_zero,
            )
            up = self._normalize_params(
                weight_format,
                up_weight,
                up_scale,
                up_zero,
            )
            if gate.keys() != up.keys():
                raise ValueError(
                    'gate and up must contain the same tensor kinds')
            for kind in gate:
                if gate[kind].shape != up[kind].shape:
                    raise ValueError(
                        f'gate and up {kind} shapes differ')

            pair_signature = {
                kind: tuple(tensor.shape)
                for kind, tensor in gate.items()
            }
            if signature is None:
                signature = pair_signature
            elif pair_signature != signature:
                raise ValueError(
                    'grouped gate/up experts must have identical '
                    'kinds and shapes')
            normalized_pairs.append((gate, up))

        projection_n = normalized_pairs[0][0]['weight'].shape[1]
        activation = _tm().ActivationType.kSilu
        gate_up_block = impl_plan.gate_up_block(activation)
        if gate_up_block == 0:
            raise NotImplementedError(
                'selected family does not support SiLU fusion')
        if projection_n % gate_up_block:
            raise ValueError(
                f'projection {projection_n} is not divisible by '
                f'gate/up block {gate_up_block}')

        impl_plan.gate_up(activation, projection_n)
        groups = projection_n // gate_up_block
        combined_experts = []
        for gate, up in normalized_pairs:
            combined = {
                kind: self._interleave_gate_up(
                    gate[kind],
                    up[kind],
                    groups,
                )
                for kind in gate
            }
            self._validate_normalized_params(
                weight_format,
                combined,
            )
            combined_experts.append(combined)

        stored_output_dim = projection_n * 2
        if grouped:
            return self._prepare_grouped(
                combined_experts,
                weight_format=weight_format,
                plan=impl_plan,
                dtype=dtype,
                stored_output_dim=stored_output_dim,
                stream=stream,
            )
        return self._prepare_one(
            combined_experts[0],
            weight_format=weight_format,
            plan=impl_plan,
            dtype=dtype,
            stored_output_dim=stored_output_dim,
            stream=stream,
        )
```

This structural block check excludes BF16-input block-out-128 FP8 fusion when
the family returns a 64-column gate/up block. Native-FP8-input fusion returns
block 128 and passes unchanged. Neither `_prepare_grouped` nor `_prepare_one`
invokes normalization, so no raw or interleaved tensor can be normalized twice.

## C++ output specification

Allocation layout and CUDA-graph compatibility come from the prepared C++ GEMM
family. Python does not reproduce output-width, epilogue, qparameter-layout,
alignment, or graph-compatibility rules.

Add family-level graph metadata. Compatibility belongs to a family and cannot
change with kernel dispatch, tuning, or record import:

Define the allocation-only result in `family.h`:

```cpp
#include "src/turbomind/core/layout.h"

struct OutputSpec {
    core::Layout output_layout;
    DataType     output_dtype{kNull};

    core::Layout output_scales_layout;
    DataType     output_scales_dtype{kNull};
};
```

The default family implementation owns ordinary output and fused-SiLU width:

```cpp
inline core::Layout
apply_output_epilogue(core::Layout layout, Epilogue epilogue)
{
    if (epilogue == Epilogue::kGatedSilu) {
        auto shape = layout.shape();
        TM_CHECK_EQ(shape.back() % 2, 0);
        shape.back() /= 2;
        return core::Layout{std::move(shape)};
    }
    TM_CHECK_EQ(epilogue, Epilogue::kNone);
    return layout;
}

inline OutputSpec
plain_output_spec(core::Layout      layout,
                  const DataFormat& format,
                  Epilogue          epilogue)
{
    TM_CHECK(!format.scales.present());

    OutputSpec spec;
    spec.output_layout = apply_output_epilogue(
        std::move(layout), epilogue);
    spec.output_dtype = format.dtype;
    return spec;
}
```

Add these complete members to `Family`:

```cpp
bool is_graph_compatible() const noexcept
{
    return is_graph_compatible_;
}

OutputSpec
output_spec(core::Layout output_layout, Epilogue epilogue) const
{
    return output_spec_(
        std::move(output_layout),
        output_format(epilogue),
        epilogue);
}

Family(std::uint32_t id,
       int priority,
       DataFormat input_format,
       DataFormat output_format,
       int align_k,
       int align_n,
       int min_k,
       int min_n,
       bool requires_packing,
       bool grouped,
       std::optional<WeightBridge> (*supports)(
           const DataFormat&, bool),
       void (*pack)(
           LinearWeight&,
           const WeightBridge&,
           cudaStream_t),
       int gate_up_block = 0,
       DataFormat fused_output = {},
       bool is_graph_compatible = true,
       OutputSpec (*output_spec)(
           core::Layout,
           const DataFormat&,
           Epilogue) = plain_output_spec);
```

The corresponding private members are:

```cpp
bool is_graph_compatible_{};
OutputSpec (*output_spec_)(
    core::Layout,
    const DataFormat&,
    Epilogue){};
```

Initialize both new members in `family.cc` and require the full-spec function:

```cpp
is_graph_compatible_{is_graph_compatible},
output_spec_{output_spec}
{
    TM_CHECK(supports_);
    TM_CHECK(pack_);
    TM_CHECK(output_spec_);
}
```

Existing native and dense cuBLAS families use the default `true`. The grouped cuBLAS families
are the only current exceptions because their host-side launch path performs
D2H copies and a stream synchronization:

```cpp
const Family grouped_f16{
    102,
    90,
    kHalf,
    kHalf,
    1,
    1,
    1,
    1,
    false,
    true,
    supports<kHalf, true>,
    pack<kHalf, true>,
    0,
    {},
    false,
};

const Family grouped_bf16{
    103,
    100,
    kBfloat16,
    kBfloat16,
    1,
    1,
    1,
    1,
    false,
    true,
    supports<kBfloat16, true>,
    pack<kBfloat16, true>,
    0,
    {},
    false,
};
```

Publish the family decision through the prepared C++ weight:

```cpp
// linear_weight.h
bool is_graph_compatible() const;

// linear_weight.cc
bool LinearWeight::is_graph_compatible() const
{
    return TM_CHECK_NOTNULL(family)->is_graph_compatible();
}
```

```cpp
.def_property_readonly(
    "is_graph_compatible",
    &LinearWeight::is_graph_compatible)
```

Append this call to the existing single
`py::class_<LinearWeight, core::Module>` declaration.

The dynamic-FP8 output function lives in `kernel/e4m3.h` and is reused by the
two families that produce that output. It returns the entire specification,
including fused-SiLU compaction:

```cpp
inline OutputSpec
fp8_output_spec(core::Layout      layout,
                const DataFormat& format,
                Epilogue          epilogue)
{
    if (!format.scales.present()) {
        return plain_output_spec(
            std::move(layout), format, epilogue);
    }

    TM_CHECK_EQ(format.dtype, kFloat8_e4m3);
    TM_CHECK_EQ(format.block_sizes.size(), 2);
    TM_CHECK_EQ(format.block_sizes[0], 128);
    TM_CHECK_EQ(format.block_sizes[1], 1);
    TM_CHECK_EQ(format.scales.dtype, kFloat);
    TM_CHECK(!format.zeros.present());

    constexpr int kGroupSize = 128;
    constexpr int kRowAlignment = 4;

    OutputSpec spec;
    spec.output_layout = apply_output_epilogue(
        std::move(layout), epilogue);
    spec.output_dtype = kFloat8_e4m3;

    const int output_dim = spec.output_layout.shape(-1);
    const core::ssize_t rows =
        spec.output_layout.size() / output_dim;

    spec.output_scales_layout = core::Layout{
        {cdiv(output_dim, kGroupSize), rows},
        {round_up(rows, kRowAlignment), 1},
    };
    spec.output_scales_dtype = kFloat;
    return spec;
}
```

The complete affected family definitions are:

```cpp
const Family w8a8{
    28,
    300,
    DataFormat{kFloat8_e4m3, {128, 1}, kFloat},
    kBfloat16,
    1,
    1,
    1,
    1,
    false,
    true,
    supports_e4m3<kFloat, 128>,
    pack,
    128,
    DataFormat{kFloat8_e4m3, {128, 1}, kFloat},
    true,
    fp8_output_spec,
};

const Family folded{
    33,
    300,
    DataFormat{kFloat8_e4m3, {128, 1}, kFloat},
    kBfloat16,
    128,
    64,
    256,
    1,
    true,
    true,
    supports_mxfp4,
    pack,
    128,
    DataFormat{kFloat8_e4m3, {128, 1}, kFloat},
    true,
    fp8_output_spec,
};
```

`Gemm` queries a prepared weight; family selection has already occurred in
`PlanWeight`, and `GemmPlan::pack` has published `family`, `epilogue`, and the
selected `output_format` into `LinearWeight`:

```cpp
#include <optional>

OutputSpec GetOutputSpec(
    const LinearWeight&      weight,
    const core::Layout&      input_layout,
    std::optional<core::ssize_t> indexed_rows = std::nullopt) const;
```

The complete calculation is C++-owned:

```cpp
OutputSpec Gemm::GetOutputSpec(
    const LinearWeight&      weight,
    const core::Layout&      input_layout,
    std::optional<core::ssize_t> indexed_rows) const
{
    TM_CHECK(weight.family);
    TM_CHECK_GE(input_layout.rank(), 2);

    auto output_shape = input_layout.shape();
    const bool grouped = weight.k_desc.ld == 0;
    if (grouped) {
        output_shape = {
            indexed_rows.value_or(input_layout.shape(0)),
            weight.output_dim,
        };
    }
    else {
        TM_CHECK(!indexed_rows);
        output_shape.back() = weight.output_dim;
    }

    return weight.family->output_spec(
        core::Layout{std::move(output_shape)},
        weight.epilogue);
}
```

`Gemm::GetOutputSpec` knows only how dense leading dimensions and grouped row
counts map to a provisional full-width output layout. `k_desc.ld == 0` is the
existing `StridedPtr`-table contract and therefore also handles a grouped
weight containing only one expert. The family callback owns epilogue
compaction and every companion allocation.

Delete the one-use FP8-specific `LlamaLinear::Impl::AllocOutputScales`. After
`GetOperandA`, `Impl::Forward` obtains the same complete specification used by
the public query and uses it for its existing optional C++ allocations:

```cpp
std::optional<core::ssize_t> indexed_rows;
if (indices && indices.size() > 0) {
    indexed_rows = desc_A.rows;
}

const auto output_spec = gemm_.GetOutputSpec(
    weight,
    input.layout(),
    indexed_rows);

Tensor& D = output;
if (!D) {
    D = Tensor{
        output_spec.output_layout,
        output_spec.output_dtype,
        kDEVICE,
    };
}

MatrixLayout desc_D{
    D.dtype(),
    kRowMajor,
    desc_A.rows,
    weight.output_dim,
    static_cast<int>(D.stride(0)),
};
if (offsets) {
    desc_D.num = desc_B.num;
    desc_D.offsets = const_cast<int*>(offsets.data());
}

Tensor& W = output_scales;
MatrixLayout desc_W{};
void* W_ptr = nullptr;
if (output_spec.output_scales_dtype != kNull) {
    if (!W) {
        W = Tensor{
            output_spec.output_scales_layout,
            output_spec.output_scales_dtype,
            kDEVICE,
        };
    }
    desc_W = {
        W.dtype(),
        kColMajor,
        static_cast<int>(W.shape(1)),
        static_cast<int>(W.shape(0)),
        static_cast<int>(W.stride(0)),
    };
    W_ptr = W.raw_data();
}
```

`desc_D.cols` deliberately remains the stored full-width
`weight.output_dim`. A fused-SiLU result uses the compact `D` shape and leading
dimension while the GEMM descriptor retains the existing epilogue contract.
The outer `LlamaLinear::Forward` overloads still flatten dense leading
dimensions before entering `Impl::Forward`, so their optional internal output
remains two-dimensional. The standalone `Gemm.output_spec` call sees the
original input layout and preserves its leading dimensions for caller-owned
allocation. The standalone API always supplies those queried allocations;
other C++ callers retain optional allocation.

Bind `OutputSpec` read-only and add `Gemm.output_spec`. The binding receives the
prepared `LinearWeight`, input tensor, and optional indices tensor, then passes
their C++ layout/element count to `GetOutputSpec`. It performs no CUDA launch or
allocation.

```cpp
#include <optional>

py::class_<gemm::OutputSpec>(m, "GemmOutputSpec")
    .def_property_readonly(
        "output_shape",
        [](const gemm::OutputSpec& spec) {
            return spec.output_layout.shape();
        })
    .def_property_readonly(
        "output_stride",
        [](const gemm::OutputSpec& spec) {
            return spec.output_layout.stride();
        })
    .def_readonly(
        "output_dtype",
        &gemm::OutputSpec::output_dtype)
    .def_property_readonly(
        "output_scales_shape",
        [](const gemm::OutputSpec& spec) {
            return spec.output_scales_layout.shape();
        })
    .def_property_readonly(
        "output_scales_stride",
        [](const gemm::OutputSpec& spec) {
            return spec.output_scales_layout.stride();
        })
    .def_readonly(
        "output_scales_dtype",
        &gemm::OutputSpec::output_scales_dtype);

py::class_<gemm::Gemm>(m, "Gemm")
    .def(py::init<>())
    .def("plan_weight", &gemm::Gemm::PlanWeight, "query"_a)
    .def("data_types", &gemm::Gemm::DataTypes, "weight_format"_a)
    .def(
        "output_spec",
        [](const gemm::Gemm&             gemm,
           const LinearWeight&           weight,
           std::shared_ptr<core::Tensor> input,
           std::shared_ptr<core::Tensor> indices) {
            const auto input_tensor =
                TensorFromShared(input, "input");
            std::optional<core::ssize_t> indexed_rows;
            if (indices && *indices && indices->size() > 0) {
                indexed_rows = indices->size();
            }
            return gemm.GetOutputSpec(
                weight,
                input_tensor.layout(),
                indexed_rows);
        },
        "weight"_a,
        "input"_a,
        "indices"_a = py::none());
```

The public Python methods call that binding and only materialize meta tensors
from the returned C++ shape, stride, and dtype:

```python
def output_spec(self, x, weight):
    if weight._impl is None:
        raise RuntimeError('Weight is closed')
    if weight._experts is not None:
        raise RuntimeError(
            'output_spec requires a dense weight')
    self._validate_input(x, grouped=False)
    return self._output_spec(x, weight, indices=None)


def output_spec_moe(self, x, weight, *, indices=None):
    if weight._impl is None:
        raise RuntimeError('Weight is closed')
    if weight._experts is None:
        raise RuntimeError(
            'output_spec_moe requires a grouped weight')
    self._validate_input(x, grouped=True)
    if indices is not None:
        if indices.device != self.device:
            raise ValueError('indices are on the wrong device')
        if indices.ndim != 1:
            raise ValueError('indices must be one-dimensional')
        if indices.numel() == 0:
            raise ValueError(
                'indices must be non-empty; use None for unindexed input')
        if not indices.is_contiguous():
            raise ValueError('indices must be contiguous')
    return self._output_spec(x, weight, indices=indices)


def _output_spec(self, x, weight, *, indices):
    tm = _tm()
    spec = self._gemm.output_spec(
        weight._impl,
        tm.from_dlpack_with_strides(x),
        None if indices is None else
        tm.from_dlpack_with_strides(indices),
    )

    output = torch.empty_strided(
        spec.output_shape,
        spec.output_stride,
        dtype=_to_torch_dtype(spec.output_dtype),
        device='meta',
    )
    output_scales = None
    if spec.output_scales_dtype != tm.DataType.TYPE_INVALID:
        output_scales = torch.empty_strided(
            spec.output_scales_shape,
            spec.output_scales_stride,
            dtype=_to_torch_dtype(spec.output_scales_dtype),
            device='meta',
        )
    return output, output_scales
```

The query requires a prepared, open weight. No kernel is dispatched: the
family was already selected by `PlanWeight`, and per-shape kernel selection
cannot change any field in `OutputSpec`. Graph compatibility is already
available as `weight.is_graph_compatible` and is not part of this query.

## Dense and grouped execution

Dense execution is `linear(x, weight)`. It requires a dense weight,
an input with rank at least two, CUDA storage on the same device, and full
contiguity. It accepts either a floating-point activation with no scales or
native FP8 E4M3 activation with its FP32 group-128 scales. C++ GEMM descriptors
and kernel feasibility validate K/N compatibility. `Gemm.output_spec` is the
sole authority for the output allocation.

For a flattened input with `input_rows = x.numel() // x.shape[-1]`, FP8
input scales have the exact `QuantizeSymm` layout:

```python
input_scale_shape = (
    (x.shape[-1] + 127) // 128,
    input_rows,
)
```

Native FP8 input scales use `torch.float32`, stride 1 is one, and stride 0 is a
multiple of four floats. Input compatibility is an unchecked caller
precondition. `x.shape[-1]` must equal the prepared weight's logical K. The
input must be either the requested model dtype with `input_scales=None`, or the
selected family's native FP8 E4M3 input with scales having exactly the dtype,
shape, and stride above. A model-dtype input may be quantized internally when
required by the selected family, while native FP8 input is consumed with the
scales supplied by the caller. Neither Python nor C++ adds a defensive check
for these standalone-API preconditions; violating one is undefined behavior.

Grouped execution is explicitly `forward_moe`; it is never selected by the
presence of optional arguments on the dense call. It requires a grouped
weight, a fully contiguous two-dimensional input, contiguous one-dimensional
offsets, and optional non-empty contiguous one-dimensional indices.
`Gemm.output_spec` derives its output row count in C++ from the input layout
and optional index count. Routing-tensor dtypes are not compared in Python.

Routing compatibility is also an unchecked caller precondition. `offsets` and
`indices`, when present, must use contiguous CUDA `torch.int32` storage because
the native binding passes their storage directly as `int*`. `offsets` must
contain one entry per prepared expert plus the terminal entry, start at zero,
be nondecreasing, and remain within the number of work rows. With indexed
input, every referenced `indices` entry must name a valid row of `x`; without
indices, `x` must already be in expert-major order. The API retains its existing
device, rank, contiguity, and non-empty-index structural checks but does not
validate routing dtypes, lengths, values, or relationships. Violating any of
these routing preconditions is undefined behavior.

Every TurboMind call is issued on the stream current when the method is called.
Both returned tensors are caller-owned Torch storage. Dense and grouped calls
share this complete structural validation; it deliberately does not compare
tensor dtypes:

```python
def _validate_input(self, x, *, grouped):
    if x.device != self.device:
        raise ValueError('input is on the wrong device')
    if grouped:
        if x.ndim != 2:
            raise ValueError('grouped input must be two-dimensional')
    elif x.ndim < 2:
        raise ValueError('dense input must have rank at least two')
    if not x.is_contiguous():
        raise ValueError('input must be contiguous')


def _validate_output(self, scales_meta, out, out_scales):
    if out.device != self.device:
        raise ValueError('out is on the wrong device')

    if scales_meta is None:
        if out_scales is not None:
            raise ValueError(
                'the selected family has no output scales')
        return

    if out_scales is None:
        raise ValueError(
            'the selected family requires output scales')
    if out_scales.device != self.device:
        raise ValueError('out_scales is on the wrong device')
```

Dense execution is the complete method below. Calling it after
`Linear.close()` is an unchecked precondition violation; no closed-executor
branch is added:

```python
def __call__(
    self,
    x,
    weight,
    *,
    out,
    input_scales=None,
    out_scales=None,
):
    _, scales_meta = self.output_spec(x, weight)
    self._validate_output(scales_meta, out, out_scales)

    with self._activate():
        self._impl.forward_dense(
            _tm().from_dlpack_with_strides(x),
            weight._impl,
            _tm().from_dlpack_with_strides(out),
            None if input_scales is None else
            _tm().from_dlpack_with_strides(input_scales),
            None if out_scales is None else
            _tm().from_dlpack_with_strides(out_scales),
        )
    return out, out_scales
```

Grouped execution is likewise complete and delegates the open-weight and mode
checks to `output_spec_moe`:

```python
def forward_moe(
    self,
    x,
    weight,
    *,
    offsets,
    out,
    indices=None,
    input_scales=None,
    out_scales=None,
):
    if offsets.device != self.device:
        raise ValueError('offsets are on the wrong device')
    if offsets.ndim != 1:
        raise ValueError('offsets must be one-dimensional')
    if not offsets.is_contiguous():
        raise ValueError('offsets must be contiguous')

    _, scales_meta = self.output_spec_moe(
        x,
        weight,
        indices=indices,
    )
    self._validate_output(scales_meta, out, out_scales)

    with self._activate():
        self._impl.forward_moe(
            _tm().from_dlpack_with_strides(x),
            weight._impl,
            None if indices is None else
            _tm().from_dlpack_with_strides(indices),
            _tm().from_dlpack_with_strides(offsets),
            _tm().from_dlpack_with_strides(out),
            None if input_scales is None else
            _tm().from_dlpack_with_strides(input_scales),
            None if out_scales is None else
            _tm().from_dlpack_with_strides(out_scales),
        )
    return out, out_scales
```

No activation is made contiguous implicitly. Rejecting unsupported strides
keeps the call free of hidden copies and guarantees that `LlamaLinear` can
flatten dense leading dimensions with `view(-1, K)`.

The grouped call uses the same Torch stream with `forward_moe` and its routing
tensors. There is no internal CUDA stream, `torch.cuda.ExternalStream`, event,
`wait_stream`, or host synchronization.

All input and output buffers must already be safe to use on the current Torch
stream. When a producer uses another stream, the caller explicitly establishes
the dependency before calling `Linear`. The API does not call `record_stream`;
the caller keeps `x`, input scales, routing tensors, `out`, and `out_scales`
alive until the execution stream completes, and owns normal PyTorch lifetime
handling for any cross-stream allocation:

```python
consumer_stream.wait_stream(producer_stream)
with torch.cuda.stream(consumer_stream):
    output, output_scales = linear(
        x,
        weight,
        out=output_buffer,
        out_scales=scale_buffer,
    )
```

The return value is always `(output, output_scales)`. For native FP8 output,
`output` has dtype `torch.float8_e4m3fn` and `output_scales` has the FP32
group-128 layout returned by C++ `OutputSpec`. For non-quantized output,
`output_scales` is `None`.

`out` is mandatory. `out_scales` must be present exactly when the C++ output
specification contains scales, so `LlamaLinear` never creates an output
allocation. Python checks device and required presence, but does not repeat
C++ dimension, stride, or dtype feasibility. The caller must allocate `out`
and, when present, `out_scales` with exactly the dtype, shape, and stride
returned by the corresponding output-specification call. Passing any other
storage is explicit undefined behavior; neither Python nor C++ validates it.
Return the same objects without dequantizing native FP8 output. These identity
guarantees hold:

```python
returned, returned_scales = linear(
    x,
    weight,
    input_scales=input_scales,
    out=output_buffer,
    out_scales=scale_buffer,
)
assert returned is output_buffer
assert returned_scales is scale_buffer
```

Consequently, remove `_forward_keep_alive`, `_pack_forward_result`, and
`release_forward_result` completely.

## Tuning records

Replace the public boolean switch with an exception-safe context manager:

```python
with linear.tuning():
    linear(
        x,
        weight,
        out=output_buffer,
        out_scales=scale_buffer,
    )
```

Its complete state transition is:

```python
@contextmanager
def tuning(self):
    self._impl.set_measure(True)
    try:
        yield self
    finally:
        self._impl.set_measure(False)
```

Nested tuning scopes and calling it after `Linear.close()` are unsupported
caller behavior and are not checked. Record import and export normalize the
public `str | PathLike[str]` argument before entering the existing pybind
methods and retain their integer return values:

```python
def import_records(
    self,
    path: str | os.PathLike[str],
) -> int:
    return int(self._impl.import_records(os.fspath(path)))


def export_records(
    self,
    path: str | os.PathLike[str],
) -> int:
    return int(self._impl.export_records(os.fspath(path)))
```

These methods operate only on the host-side GEMM dispatch cache and require no
temporary CUDA context. Calling either after `Linear.close()` follows the same
unchecked precondition as every other public operation. The measurement call
itself completes its event timing before returning; changing the host-side
dispatch policy does not require a stream synchronization.

The context manager is deliberately independent of the PyTorch engine. A
format-specific engine wrapper may adapt it to the existing `WarmupManager`
callback protocol after its weights have been loaded:

```python
def warmup(self, meta: WarmupMeta):
    rows_to_tune = {
        1,
        meta.max_batch_size,
        meta.max_num_tokens,
    }
    with self.linear.tuning():
        for rows in sorted(rows_to_tune):
            x = torch.empty(
                (rows, self.in_features),
                dtype=meta.dtype,
                device=self.linear.device,
            )
            output_meta, scales_meta = self.linear.output_spec(
                x,
                self.weight,
            )
            output = torch.empty_strided(
                output_meta.shape,
                output_meta.stride(),
                dtype=output_meta.dtype,
                device=self.linear.device,
            )
            output_scales = None
            if scales_meta is not None:
                output_scales = torch.empty_strided(
                    scales_meta.shape,
                    scales_meta.stride(),
                    dtype=scales_meta.dtype,
                    device=self.linear.device,
                )
            self.linear(
                x,
                self.weight,
                out=output,
                out_scales=output_scales,
            )
```

This snippet defines the adapter boundary, not a complete engine tuning
policy. The wrapper must add the engine's intended decode-capture and prefill
row counts and must construct representative offsets and indices for grouped
weights. TurboMind records are descriptor-specific, and its lower-bound cache
can reuse a larger tuned row count for a smaller call but cannot provide the
same shape-specific optimum. The later full-model warm-up does not implicitly
enter this tuning scope.

The standalone `Linear` module does not import `WarmupManager` or register
callbacks. A dispatch cache belongs to one `Linear`; wrappers using independent
instances must tune or import records into each instance.

`benchmark.py` changes from:

```python
fx.linear.set_measure(True)
fx.run_linear_forward()
fx.sync_tm()
fx.linear.set_measure(False)
fx.release_forward_result()
```

to:

```python
with fx.linear.tuning():
    fx.run_linear_forward()
```

Timed and warmup forwards no longer call `release_forward_result`.

## Remove non-API helpers

Delete the unused `invoke_moe_dispatch` and `invoke_moe_combine` functions.

Do not retain one-line quantization forwarding functions. The fixture calls
the `_turbomind` operations directly where it generates test weights. The only
private helpers retained in `linear.py` are helpers called from multiple API
paths:

- `_tm` for lazy extension import;
- `_to_tm_dtype` for planning and allocation;
- `_to_torch_dtype` for materializing C++ output specifications;
- `_activate` for every temporary caller-stream context;
- `_copy_param` for all parameter kinds;
- `_normalize_params` for ordinary, grouped, and gate/up source tensors;
- `_prepare_one` for dense and grouped preparation;
- `_prepare_grouped` for ordinary and gate/up grouped preparation;
- `_interleave_gate_up` for dense and grouped gate/up components;
- `_output_spec` for dense and grouped C++ output queries;
- `_validate_input` for dense and grouped device/layout structure;
- `_validate_normalized_params` for ordinary and fused weight components;
- `_validate_output` for caller-owned output presence and device validation.

There is no pointer extraction or `MakeStridedPtrs` call in Python.
`prepare_weight` and `fuse_weight` hand their prepared expert handles to the
C++ grouped-preparation implementation behind `_prepare_grouped`; callers and
tests see only the returned grouped `Weight`.

## Fixture migration

Redesign `LinearFixture` as a consumer of the new API rather than adapting the
old fixture object graph. It creates one device-bound `Linear`, obtains one
`WeightPlan` for each prepared contract, and calls only `plan_weight`,
`prepare_weight`, `fuse_weight`, `output_spec`, `output_spec_moe`,
`Linear.__call__`, `forward_moe`, `tuning`, and the two explicit close methods.
It never constructs or inspects `_tm.Gemm`, `_tm.WeightQuery`,
`_tm.GemmPlan`, `_tm.LinearWeight`, family metadata, packed descriptors, or
expert pointer tables. Dense, grouped, and fused cases pass the exact
`WeightPlan` whose `shape_constraints` they used into preparation. Grouped
cases pass sequences directly to `prepare_weight`; fused grouped cases pass the
tuple of expert sequences directly to `fuse_weight`.

Direct `_turbomind` calls in the fixture are limited to generating test data
and numerical references with the existing quantization operations. A reused
fixture-private context installs the currently selected Torch stream for those
operations:

```python
@contextmanager
def _quantization_context(self):
    with torch.cuda.device(self.device):
        stream = torch.cuda.current_stream(self.device)
        with _tm().create_device_context(stream.cuda_stream):
            yield
```

This setup helper is not part of the standalone `Linear` API and does not own a
stream or memory pool.

The fixture continues to generate quantized test data with the existing
TurboMind CUDA quantizers. Their output, scale, zero, source, and dequantized
destinations are caller-owned Torch tensors wrapped for the duration of the
quantizer call; the fixture no longer allocates an unprepared
`_tm.LinearWeight`.

For U4 compressed-tensors storage, allocate the physical checkpoint tensor and
use the logical-view `from_dlpack` overload only while invoking
`QuantizeGroupwise`:

```python
raw_weight = torch.empty(
    (output_dim, input_dim // 8),
    dtype=torch.int32,
    device=self.device,
)
quant = _tm().from_dlpack(
    raw_weight,
    [output_dim, input_dim],
    _tm().DataType.TYPE_UINT4,
)
raw_scales = torch.empty(
    (output_dim, input_dim // group_size),
    dtype=dtype,
    device=self.device,
)
raw_zeros = torch.empty_like(raw_scales)
dequant = torch.empty_like(source)
_tm().QuantizeGroupwise(
    quant=quant,
    scales=_tm().from_dlpack_with_strides(raw_scales),
    zeros=_tm().from_dlpack_with_strides(raw_zeros),
    dequant=_tm().from_dlpack_with_strides(dequant),
    src=_tm().from_dlpack_with_strides(source),
    group_size=group_size,
)
weight_format = CompressedTensorFormat(block_in=group_size)
```

`source` has checkpoint orientation `[N, K]`, matching the quantizer's
groupwise dimension and `CompressedTensorFormat.normalize`. For the existing
U4 cases, build each K group from `(+x, -x)` pairs. The existing integral
quantizer then produces zero point 8. Retain `raw_zeros` for the mathematical
reference but omit it from `prepare_weight`; `CompressedTensorFormat`
synthesizes the same value from the normalized scales.

For MXFP4, the physical Torch destination is its real `.blocks` tensor:

```python
raw_blocks = torch.empty(
    (output_dim, input_dim // 32, 16),
    dtype=torch.uint8,
    device=self.device,
)
quant = _tm().from_dlpack(
    raw_blocks,
    [output_dim, input_dim],
    _tm().DataType.TYPE_FP4_E2M1,
)
raw_scales = torch.empty(
    (output_dim, input_dim // 32),
    dtype=torch.uint8,
    device=self.device,
)
dequant = torch.empty_like(source)
_tm().QuantizeGroupwise(
    quant=quant,
    scales=_tm().from_dlpack_with_strides(raw_scales),
    zeros=None,
    dequant=_tm().from_dlpack_with_strides(dequant),
    src=_tm().from_dlpack_with_strides(source),
    group_size=32,
)
weight_format = MXFP4Format()
```

FP8 generation similarly supplies Torch-owned E4M3 and FP32-scale
destinations to `QuantizeSymmBlock`; ordinary floating-point cases need no
quantizer. Except for the explicit group-16 NVFP4 exclusion, the fixture passes
these generated format-native tensors directly to `prepare_weight` or
`fuse_weight`. It must not construct a resolver or add a second public
weight-construction interface.

The fixture constructs the type combination declared by each existing
`LinearCase` and sends it through the API without preflight dtype comparisons.
It must not duplicate family or kernel feasibility logic to decide whether a
case is supported; `NotImplementedError` from planning remains the unit-test
skip boundary and the benchmark warning boundary. The only
`NotImplementedError` raised before planning is the explicit missing
`NVFP4Format` case above.

The fixture does not inspect a prepared `Weight` to decide how to form an
activation. It sends the input contract declared by the existing `LinearCase`:
model-dtype cases pass the model-dtype tensor directly, while native-E4M3 cases
use caller-owned Torch output and scale buffers with the existing
`QuantizeSymm` operation:

```python
rows = x.shape[0]
groups = (x.shape[1] + 127) // 128
aligned_rows = (rows + 3) // 4 * 4
x_q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
x_scales = torch.empty_strided(
    (groups, rows),
    (aligned_rows, 1),
    dtype=torch.float32,
    device=self.device,
)
x_dequant = torch.empty_like(x)
with self._quantization_context():
    _tm().QuantizeSymm(
        out=_tm().from_dlpack_with_strides(x_q),
        scale=_tm().from_dlpack_with_strides(x_scales),
        src=_tm().from_dlpack_with_strides(x),
    )
    _tm().DequantizeSymm(
        out=_tm().from_dlpack_with_strides(x_dequant),
        src=_tm().from_dlpack_with_strides(x_q),
        scale=_tm().from_dlpack_with_strides(x_scales),
    )
```

The resulting `x_q`, `x_scales`, and `x_dequant` remain caller-owned Torch
storage. No prepared-weight private field is read.

The fixture keeps separate gate and up tensors for its mathematical reference:
`silu(x @ gate) * (x @ up)`. Only `fuse_weight` constructs the family-specific
blocked source; the fixture neither imports a private helper nor duplicates
that block-packing algorithm.

`prepare_batch` calls `output_spec` or `output_spec_moe` and allocates one
caller-owned output tensor from the returned C++ layout. When the returned
scale meta tensor is present, it allocates the caller-owned output-scale tensor
from that C++ layout as well. Validation, warmup, tuning, and timed forwards
pass them through `out=` and `out_scales=`. The benchmark therefore measures
GEMM execution rather than fresh PyTorch allocations or implicit output
dequantization on every iteration.

When native FP8 output scales are returned, the fixture uses those public
outputs as inputs to `DequantizeSymm`, writing into a caller-owned reference
buffer under `_quantization_context`. Ordinary output is compared directly.
The fixture never reads `Weight._impl`, `_input_format`, `_output_format`, or
the family gate/up block.

`close()` drops caller-owned reference/output tensors, then enters
`torch.cuda.device(self.device)`, explicitly closes its top-level prepared
`Weight`, closes its `Linear`, and clears both references. Grouped expert
handles close through their private parent on that same device. There is no
manual `ContextGuard` exit in the fixture.

## Error behavior

Use normal Python exception categories:

- `TypeError` for explicit dtype translation and preparation/fusion outer-
  structure checks; execution does not defensively classify every Python
  argument kind, so ordinary attribute or binding-conversion errors may surface;
- `ValueError` for validated rank, shape, layout, device, companion presence,
  or sequence-length errors; caller-owned output dtype, shape, and stride
  mismatches, input/scale compatibility mismatches, and invalid routing
  storage or contents are unchecked undefined behavior; cross-device weight
  use, destruction on a device different from construction, and implicit
  destruction of a live weight or `Linear` are also unchecked undefined
  behavior; using a `WeightPlan` with another executor, using a `Linear` after
  `close()`, and omitting required asymmetric zeros from a format such as AWQ
  are likewise unchecked preconditions and receive no special exception
  translation;
- `NotImplementedError` when no family supports a valid requested contract;
- `RuntimeError` for a closed weight or dense/grouped mode mismatch.

Unsupported benchmark cases continue to log a warning and skip. Unit tests
continue to translate `NotImplementedError` into `pytest.skip`. Verification
accepts only the explicit group-16 NVFP4 and BF16-input block-out-128 FP8 SiLU
fusion skips; any other unexpected unsupported format or fusion is a failure.

## Verification

1. Run `git diff --check`.
2. Build with `ninja` from `build/` without setting `PYTHONPATH`; the binding
   change must compile and relink `_turbomind`.
3. Set runtime `PYTHONPATH` to
   `/data/lmdeploy-gemm/build/lib:/data/lmdeploy-gemm` before Python commands.
4. Run a non-GPU compile check without importing the CUDA extension:

   ```bash
   PYTHONPATH=/data/lmdeploy-gemm/build/lib:/data/lmdeploy-gemm \
   python -m py_compile \
       tests/turbomind/linear/linear.py \
       tests/turbomind/linear/fixture.py \
       tests/turbomind/linear/benchmark.py \
       tests/turbomind/linear/test_linear.py
   ```
5. Before every GPU command, call `get_gpu_usage`, select an empty SM90 GPU,
   and run outside the sandbox. Substitute that physical ID for
   `<EMPTY_SM90_GPU>` in each command; do not rely on a previously exported
   `CUDA_VISIBLE_DEVICES`.
6. Do not add a test file or a second test function. Migrate
   `LinearFixture` so the existing parameterized
   `test_smoke_linear_correctness` exercises `plan_weight`, the returned
   `WeightPlan.shape_constraints`, `prepare_weight`, `fuse_weight`,
   `Linear.__call__`, `forward_moe`, caller-owned outputs, native FP8 input and
   scales, native FP8 output and scales, `Weight.close()`, and `Linear.close()`.
   The fixture must stop constructing the old public `Weight` and `Linear`
   interfaces.
7. Adjust the existing smoke case selection only where needed to route that one
   parameterized test through the changed API branches. Reuse existing
   `LinearCase` definitions rather than adding special API-only cases:

   - an existing ordinary dense case covers `prepare_weight` and `out=`;
   - an existing `input_type == 'fp8_e4m3'` case supplies actual FP8 input and
     scales instead of BF16 input that is internally quantized;
   - adapt an existing U4 fixture case to generate groupwise-symmetric source
     values, omit source zeros from `prepare_weight`, and verify the normal
     automatic-synthesis path through its existing numerical comparison;
   - supported dense and grouped `fuse_silu` cases call `fuse_weight` with a
     tuple of tensors and a tuple of expert sequences respectively; the
     BF16-input block-out-128 FP8 cases must report the explicit fusion-block
     exclusion;
   - existing grouped cases continue to cover offsets with and without indexed
     input.

   Preserve the fixture's existing stream-boundary coverage using Torch streams:
   prepare weights on one Torch stream, explicitly order the execution stream
   after it, and run the existing forward path on the execution stream. Before
   closing, synchronize both streams to satisfy the API's quiescence
   precondition. Closing afterward verifies that TurboMind neither destroys the
   caller streams nor binds `Linear` to its construction or preparation stream.

   Add assertions to `test_smoke_linear_correctness` only for contract values
   already produced by those runs: returned output identity when `out=` is
   supplied, returned scale identity for FP8 `out_scales=`, scale
   shape/stride needed by the numerical reference, and meaningful numerical
   comparison. Do not add isolated
   tests for getters, error messages, `is_available`, context-manager absence,
   or other behavior already enforced by the execution path.
8. Run the adjusted existing linear smoke suite:

   ```bash
   PYTHONPATH=/data/lmdeploy-gemm/build/lib:/data/lmdeploy-gemm \
   CUDA_VISIBLE_DEVICES=<EMPTY_SM90_GPU> \
   pytest -q tests/turbomind/linear/test_linear.py
   ```
9. Run the full existing benchmark suite in validation-only mode. This routes
   all existing SM90 formats and shape classes except the two explicit
   exclusions through the migrated fixture without creating new tests. Inspect
   the warnings and require every unsupported warning to name either group-16
   NVFP4 or BF16-input block-out-128 FP8 SiLU fusion:

   ```bash
   PYTHONPATH=/data/lmdeploy-gemm/build/lib:/data/lmdeploy-gemm \
   CUDA_VISIBLE_DEVICES=<EMPTY_SM90_GPU> \
   python -m tests.turbomind.linear.bench_linear \
       --suite full \
       --tp 1 \
       --ep 1 \
       --exact-parallel \
       --quiet \
       --iters 0
   ```

10. After checking `get_gpu_usage` again, run the same case through the tuning
    path:

    ```bash
    PYTHONPATH=/data/lmdeploy-gemm/build/lib:/data/lmdeploy-gemm \
    CUDA_VISIBLE_DEVICES=<EMPTY_SM90_GPU> \
    TM_GEMM_TUNE='top_k=0' \
    python -m tests.turbomind.linear.bench_linear \
        --suite full \
        --case llama2_7b_o__bf16_bf16_bf16 \
        --batch 1 \
        --tp 1 \
        --ep 1 \
        --exact-parallel \
        --tune \
        --no-validate \
        --iters 0
    ```

11. After checking `get_gpu_usage` again and selecting an empty SM90 GPU, run
    the unchanged TurboMind model smoke script outside the sandbox. The Python
    API remains test-local, but this plan also changes the shared `Family`,
    `Gemm`, and `LlamaLinear` C++ paths used by the model engine:

    ```bash
    PYTHONPATH=/data/lmdeploy-gemm/build/lib:/data/lmdeploy-gemm \
    python scripts/test_turbomind_model.py \
        --model-id Qwen/Qwen3-8B \
        --cache-dir /mnt_cfs/huggingface_hub/hub/ \
        --gpus <EMPTY_SM90_GPU> \
        --tp 1 \
        --max-new-tokens 128 \
        --prompt 'Explain why CUDA streams require explicit ordering between dependent operations.'
    ```

    Inspect the generated response and require at least 128 meaningful human
    tokens relevant to the prompt. Gibberish or an unrelated response is a
    failure even if the process exits successfully.

12. Run `git diff --check` again.
