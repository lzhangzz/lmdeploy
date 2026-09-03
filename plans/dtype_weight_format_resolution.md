# Generic-float weight-format resolution

## Status

**Approved for implementation.**

The earlier proposed `dlpack.h` dependency is not part of this design:
`kGenericFloat` is format metadata and never crosses DLPack. DLPack continues
to describe concrete tensor storage types only.

This work is a prerequisite for
`plans/python_linear_api_refactor.md`. That plan remains suspended and must be
re-audited after this plan is implemented and verified.

## Problem

The current converter makes `WeightFormat.make_data_format(data_type)` use the
component computation dtype as the declared dtype of scales and zeros. This
couples checkpoint-format resolution to component-dtype selection and does not
represent checkpoints whose floating qparameters are FP16, BF16, or FP32.

The exact concrete qparameter dtype does not affect family eligibility: C++
`WeightBridge` converts the actual scale and zero tensors to the concrete type
required by the selected packer. `DataFormat` therefore needs a metadata-only
floating category for bridgeable qparameters.

## Settled contracts

### `kGenericFloat`

Introduce `kGenericFloat` as an abstract `DataType` value matching exactly:

```text
kHalf | kBfloat16 | kFloat
```

It does not include FP8, FP4, integers, or FP64. FP64 is excluded because the
current `EnsureFloatDtype` conversion path does not support it.

`kGenericFloat` may appear only in format metadata:

```cpp
DataFormat::scales.dtype
DataFormat::zeros.dtype
```

It is never intentionally used as a concrete `Tensor`, `Buffer`, CUDA,
cuBLAS, or DLPack dtype. Passing it to any storage, allocation, byte-size,
dispatch, or execution interface is explicit undefined behavior. Do not add
defensive checks for that misuse.

### Concrete tensor storage

DLPack import continues to produce exact tensor types:

```text
torch.float16  -> kHalf
torch.bfloat16 -> kBfloat16
torch.float32  -> kFloat
```

No DLPack code maps to or from `kGenericFloat`.

Python preserves the checkpoint scale dtype. Packed or synthesized U4 zeros
are normalized to the concrete scale tensor dtype. C++ `WeightBridge` then
converts the actual tensors to the selected family's required type.

### One descriptor per `WeightFormat`

Every `WeightFormat` has one zero-argument `make_data_format()`:

| Format | Weight dtype | Scales dtype | Zeros dtype |
|---|---:|---:|---:|
| Trivial | selected component dtype | absent | absent |
| AWQ | UINT4 | generic float | generic float |
| GPTQ | UINT4 | generic float | generic float |
| Compressed tensors | UINT4 | generic float | generic float |
| FP8 | E4M3 | generic float | absent |
| MXFP4 | E2M1 | UINT8 | absent |

There is no probe descriptor, publisher-dtype parameter, tensor-dependent
descriptor, or builder-side trivial branch.

### Resolver

`WeightFormatResolver` only selects a checkpoint format and normalizes its
tensors. It neither owns nor resolves the component dtype.

## Implementation

### 1. Add the metadata-only `kGenericFloat` type

Update `src/turbomind/core/data_type.h` by adding the metadata-only value at the
end of the enum. Give it an explicit negative value so it cannot alias any
encoded storage type, without moving it ahead of the existing entries:

```cpp
enum class DataType: int {
    kNull        = 0,
    kBool        = 1,
    kUint8       = encode_data_type(0,  0,  8),
    kUint16      = encode_data_type(0,  0, 16),
    kUint32      = encode_data_type(0,  0, 32),
    kUint64      = encode_data_type(0,  0, 64),
    kInt8        = encode_data_type(1,  0,  8),
    kInt16       = encode_data_type(1,  0, 16),
    kInt32       = encode_data_type(1,  0, 32),
    kInt64       = encode_data_type(1,  0, 64),
    kFloat16     = encode_data_type(1,  5, 10),
    kFloat32     = encode_data_type(1,  8, 23),
    kFloat64     = encode_data_type(1, 11, 52),
    kBfloat16    = encode_data_type(1,  8,  7),
    kFloat4_e2m1 = encode_data_type(1,  2,  1),
    kFloat6_e2m3 = encode_data_type(1,  2,  3),
    kFloat6_e3m2 = encode_data_type(1,  3,  2),
    kFloat8_e4m3 = encode_data_type(1,  4,  3),
    kFloat8_e5m2 = encode_data_type(1,  5,  2),
    kUint2       = encode_data_type(0,  0,  2),
    kUint4       = encode_data_type(0,  0,  4),
    kUint6       = encode_data_type(0,  0,  6),
    kPointer,
    kUint        = kUint32,
    kInt         = kInt32,
    kFloat       = kFloat32,
    kHalf        = kFloat16,
    kDouble      = kFloat64,
    kE2m1        = kFloat4_e2m1,
    kE2m3        = kFloat6_e2m3,
    kE3m2        = kFloat6_e3m2,
    kE4m3        = kFloat8_e4m3,
    kE5m2        = kFloat8_e5m2,
    kGenericFloat = -1,
};
```

Expose the constant and string representation:

```cpp
inline constexpr DataType kGenericFloat = DataType::kGenericFloat;
```

```cpp
case kGenericFloat: return "generic_float";
```

Do not add `from_data_type`, `data_type_v`, byte-size, element-count, CUDA,
cuBLAS, dispatch, or DLPack mappings for `kGenericFloat`.

Expose it to Python through the existing enum binding:

```cpp
.value("TYPE_GENERIC_FLOAT", kGenericFloat)
```

### 2. Separate concrete floating types from format-compatible floating types

Keep the existing concrete predicate unchanged and add one metadata predicate
in `src/turbomind/core/data_format.h`:

```cpp
inline bool IsTrivialFloatType(DataType type) noexcept
{
    return type == kFloat || type == kHalf || type == kBfloat16;
}

inline bool IsFloatFormatType(DataType type) noexcept
{
    return type == kGenericFloat || IsTrivialFloatType(type);
}
```

`EnsureFloatDtype` continues using `IsTrivialFloatType`; it must never accept
`kGenericFloat` as a tensor dtype.

Update U4 and E4M3 format feasibility to accept metadata-compatible floating
types:

```cpp
if (!IsFloatFormatType(format.scales.dtype)) {
    return std::nullopt;
}
```

```cpp
if (!IsFloatFormatType(format.zeros.dtype)) {
    return std::nullopt;
}
```

MXFP4 continues requiring exact UINT8 scales.

### 3. Make U4 bridge targets concrete for every family

Change `supports_u4` so each family declares the qparameter type required by
its packer:

```cpp
template<int GroupSize, DataType QparamDtype>
std::optional<WeightBridge>
supports_u4(const DataFormat& format, bool)
{
    if (format.dtype != kUint4) {
        return std::nullopt;
    }
    if (format.block_sizes.size() != 2) {
        return std::nullopt;
    }
    if (format.block_sizes[1] != 1) {
        return std::nullopt;
    }
    if (format.block_sizes[0] % GroupSize != 0) {
        return std::nullopt;
    }
    if (!IsFloatFormatType(format.scales.dtype)) {
        return std::nullopt;
    }
    if (!IsFloatFormatType(format.zeros.dtype)) {
        return std::nullopt;
    }

    WeightBridge bridge;
    bridge.replicate_scales.x = format.block_sizes[0] / GroupSize;
    bridge.convert_scales = QparamDtype;
    bridge.convert_zeros = QparamDtype;
    return bridge;
}
```

All SM70/SM75/SM80 and SM90 HMMA U4 families use concrete FP16 targets:

```cpp
supports_u4<32, kHalf>
supports_u4<128, kHalf>
```

The native SM90 mixed U4 family uses concrete BF16 targets directly:

```cpp
auto bridge = format.block_sizes[0] % 128 == 0
    ? supports_u4<128, kBfloat16>(format, grouped)
    : supports_u4<32, kBfloat16>(format, grouped);
```

Remove the native family's subsequent manual assignments to
`bridge->convert_scales` and `bridge->convert_zeros`.

E4M3 already declares its concrete bridge target through the existing
`ScaleDtype` template parameter. Only its floating-format predicate changes.

### 4. Make Python `WeightFormat` descriptors dtype-independent

Update `lmdeploy/turbomind/weight_format.py`:

```python
class WeightFormat(ABC):
    name: ClassVar[str]
    suffix_map: ClassVar[dict[str, str]]
    weight_dtype: _tm.DataType
    scales_dtype: ClassVar[_tm.DataType]
    zeros_dtype: ClassVar[_tm.DataType]

    block_in: int | None
    block_out: int | None

    def __init__(self, *, block_in: int | None = None,
                 block_out: int | None = None):
        self.block_in = block_in
        self.block_out = block_out

    def make_data_format(self) -> _tm.DataFormat:
        return _tm.DataFormat(
            self.weight_dtype,
            [self.block_in or 1, self.block_out or 1],
            self.scales_dtype,
            self.zeros_dtype,
        )

    def __eq__(self, other) -> bool:
        return (
            type(self) is type(other)
            and self.weight_dtype == other.weight_dtype
            and self.scales_dtype == other.scales_dtype
            and self.zeros_dtype == other.zeros_dtype
            and self.block_in == other.block_in
            and self.block_out == other.block_out
        )

    def __hash__(self) -> int:
        return hash((
            type(self),
            self.weight_dtype,
            self.scales_dtype,
            self.zeros_dtype,
            self.block_in,
            self.block_out,
        ))
```

All fields that determine the normalized `DataFormat` participate in equality
and hashing. In particular, `TrivialFormat(weight_dtype=TYPE_FP16)` and
`TrivialFormat(weight_dtype=TYPE_BF16)` must not pass the format-uniformity
check in `concat_out_dim` as the same format.

Declare the complete format metadata on every concrete class:

```python
class AWQFormat(WeightFormat):
    weight_dtype = _tm.DataType.TYPE_UINT4
    scales_dtype = _tm.DataType.TYPE_GENERIC_FLOAT
    zeros_dtype = _tm.DataType.TYPE_GENERIC_FLOAT
```

```python
class GPTQFormat(WeightFormat):
    weight_dtype = _tm.DataType.TYPE_UINT4
    scales_dtype = _tm.DataType.TYPE_GENERIC_FLOAT
    zeros_dtype = _tm.DataType.TYPE_GENERIC_FLOAT
```

```python
class CompressedTensorFormat(WeightFormat):
    weight_dtype = _tm.DataType.TYPE_UINT4
    scales_dtype = _tm.DataType.TYPE_GENERIC_FLOAT
    zeros_dtype = _tm.DataType.TYPE_GENERIC_FLOAT
```

```python
class FP8Format(WeightFormat):
    weight_dtype = _tm.DataType.TYPE_FP8_E4M3
    scales_dtype = _tm.DataType.TYPE_GENERIC_FLOAT
    zeros_dtype = _tm.DataType.TYPE_INVALID
```

```python
class MXFP4Format(WeightFormat):
    weight_dtype = _tm.DataType.TYPE_FP4_E2M1
    scales_dtype = _tm.DataType.TYPE_UINT8
    zeros_dtype = _tm.DataType.TYPE_INVALID
```

`TrivialFormat` receives the already selected component dtype:

```python
class TrivialFormat(WeightFormat):
    scales_dtype = _tm.DataType.TYPE_INVALID
    zeros_dtype = _tm.DataType.TYPE_INVALID

    def __init__(self, *, weight_dtype: _tm.DataType):
        self.weight_dtype = weight_dtype
        super().__init__()
```

Remove all `make_data_format(data_type)` overrides.

Do not retain `has_zero_point`. `zeros_dtype != TYPE_INVALID` means that zeros
are part of the normalized representation. It does not mean that every such
format can synthesize missing checkpoint zeros. A format that requires zeros
must enforce their presence in `accepts()`; a format that permits their
absence implements `synthesize_zeros()`.

### 5. Preserve concrete qparameter tensors during normalization

Declare the concrete tensor dtypes represented by `TYPE_GENERIC_FLOAT` once at
module scope:

```python
_GENERIC_FLOAT_DTYPES = frozenset({
    torch.float16,
    torch.bfloat16,
    torch.float32,
})
```

Format recognition must require a scale tensor with one of those concrete
dtypes. The generic descriptor intentionally erases the exact scale dtype, so
an unsupported concrete tensor must not pass `accepts()` and reach the C++
bridge.

Keep the existing AWQ shape check while making the scale contract explicit:

```python
def accepts(self, available: dict[str, Tensor]) -> bool:
    weight = available.get(".qweight")
    scales = available.get(".scales")
    zeros = available.get(".qzeros")
    if weight is None or weight.dtype != torch.int32:
        return False
    if scales is None or scales.dtype not in _GENERIC_FLOAT_DTYPES:
        return False
    if zeros is None or zeros.dtype != torch.int32:
        return False
    if weight.ndim >= 2 and scales.ndim >= 2:
        return weight.shape[-1] * 8 == scales.shape[-1]
    return True
```

Keep the corresponding GPTQ shape check:

```python
def accepts(self, available: dict[str, Tensor]) -> bool:
    qw = available.get(".qweight")
    if qw is None or qw.dtype != torch.int32:
        return False
    scales = available.get(".scales")
    if scales is None or scales.dtype not in _GENERIC_FLOAT_DTYPES:
        return False
    zeros = available.get(".qzeros")
    if zeros is not None and zeros.dtype != torch.int32:
        return False
    if qw.ndim >= 2 and scales.ndim >= 2:
        return qw.shape[-1] == scales.shape[-1]
    return True
```

Require compressed-tensors scales through their checkpoint suffix:

```python
def accepts(self, available: dict[str, Tensor]) -> bool:
    weight = available.get(".weight_packed")
    scales = available.get(".weight_scale")
    zeros = available.get(".weight_zero_point")
    if weight is None or weight.dtype != torch.int32:
        return False
    if scales is None or scales.dtype not in _GENERIC_FLOAT_DTYPES:
        return False
    if zeros is not None and zeros.dtype != torch.int32:
        return False
    return True
```

Make FP8 reuse the same set while retaining all of its existing weight, rank,
and shape checks:

```python
def accepts(self, available: dict[str, Tensor]) -> bool:
    scales = available.get(".weight_scale_inv")
    if scales is None or scales.dtype not in _GENERIC_FLOAT_DTYPES:
        return False
    weight = available.get(".weight")
    if weight is None:
        return False
    if weight.dtype not in (torch.float8_e4m3fn, torch.uint8):
        return False
    if weight.dim() < 2 or scales.dim() < 2:
        return False
    expected = (
        (weight.shape[-2] + self.block_out - 1) // self.block_out,
        (weight.shape[-1] + self.block_in - 1) // self.block_in,
    )
    return scales.shape[-2:] == expected
```

Packed checkpoint zeros are not members of `_GENERIC_FLOAT_DTYPES`: AWQ,
GPTQ, and compressed-tensors `pack-quantized` zeros are `torch.int32` before
normalization. This plan does not accept pre-unpacked zero tensors. Once
decoded, U4 zeros are cast to the already accepted scale tensor's concrete
dtype as described below.
AWQ requires `.qzeros` in `accepts()` because it has no symmetric-zero
synthesis. GPTQ and compressed-tensors may omit their checkpoint zero tensors
because both formats implement `synthesize_zeros()`.

Remove unconditional `.to(torch.float16)` conversions from AWQ, GPTQ, and
compressed-tensors normalization. Preserve FP16, BF16, or FP32 scale tensors.
FP8 likewise preserves its concrete floating scale dtype. MXFP4 preserves
UINT8 scales.

After per-format decoding and missing-zero synthesis, normalize U4 zeros to
the concrete scale dtype in `WeightFormatResolver._build_linear`:

```python
def _build_linear(
    self,
    fmt: WeightFormat,
    available: dict[str, Tensor],
) -> Linear:
    tensors = {
        kind: fmt.normalize(available[suffix], kind)
        for suffix, kind in fmt.suffix_map.items()
        if suffix in available
    }

    if (
        fmt.zeros_dtype != _tm.DataType.TYPE_INVALID
        and 'zeros' not in tensors
    ):
        tensors['zeros'] = fmt.synthesize_zeros(tensors['scales'])

    if 'zeros' in tensors:
        tensors['zeros'] = tensors['zeros'].to(
            tensors['scales'].dtype)

    return Linear(tensors=tensors, weight_format=fmt)
```

Bias is not converted by this path. The accepted floating qparameter set must
remain FP16, BF16, and FP32, matching `kGenericFloat`.

### 6. Resolve component dtype from generic descriptors

Replace `converter.py::_build_resolver` with dtype-independent quantized-format
construction:

```python
def _build_quantized_formats(
    model_format: str | None,
    group_size: int | None,
) -> list[WeightFormat]:
    formats: list[WeightFormat] = []

    if model_format in (None, 'hf'):
        pass
    elif model_format == 'awq':
        formats.append(AWQFormat(block_in=group_size))
    elif model_format == 'gptq':
        formats.append(GPTQFormat(block_in=group_size))
    elif model_format == 'compressed-tensors':
        formats.append(CompressedTensorFormat(block_in=group_size))
    elif model_format == 'fp8':
        formats.extend((
            FP8Format(block_out=128),
            FP8Format(block_out=1),
        ))
    elif model_format == 'mxfp4':
        formats.append(MXFP4Format())
    else:
        raise ValueError(f'unknown model_format: {model_format!r}')

    return formats
```

For each candidate component dtype, query every quantized format plus a
concrete trivial format:

```python
def _get_executable_dtypes(
    formats: list[WeightFormat],
    device,
) -> set[str]:
    executable: set[str] = set()

    with torch.cuda.device(device):
        gemm = _tm.Gemm()

        for name in ('bfloat16', 'float16'):
            candidate = _torch_dtype_to_cpp(getattr(torch, name))
            candidate_formats = [
                *formats,
                TrivialFormat(weight_dtype=candidate),
            ]

            if all(
                candidate in gemm.data_types(
                    fmt.make_data_format()
                )
                for fmt in candidate_formats
            ):
                executable.add(name)

    return executable
```

This correctly handles a model containing only trivial weights: `formats` is
empty and each candidate's concrete `TrivialFormat` queries the registered
floating families on the active architecture.

Apply the existing requested/HF preference, then construct the resolver:

```python
quantized_formats = _build_quantized_formats(
    engine_config.model_format,
    group_size,
)
executable = _get_executable_dtypes(
    quantized_formats,
    engine_config.devices[0],
)
dtype_name = _resolve_dtype(
    requested_dtype,
    hf_model_cfg,
    executable,
)
dtype = getattr(torch, dtype_name)
data_type = _torch_dtype_to_cpp(dtype)

resolver = WeightFormatResolver(formats=[
    *quantized_formats,
    TrivialFormat(weight_dtype=data_type),
])
engine_config.dtype = dtype_name
```

Replace the resolver constructor with:

```python
class WeightFormatResolver:
    def __init__(self, *, formats: list[WeightFormat]):
        self._formats = formats
        self._suffixes = frozenset(
            suffix
            for fmt in formats
            for suffix in fmt.suffix_map
        )
```

Delete `_data_type` and the `data_type` property.

### 7. Use zero-argument descriptors in the builder

Update `_make_gemm_query`:

```python
def _make_gemm_query(self, linear: Linear, *, grouped: bool = False):
    query = _tm.WeightQuery()
    query.data_type = self.config.data_type
    query.weight_format = linear.weight_format.make_data_format()
    query.input_dtype = self._ctx.gemm_input_dtype
    query.grouped = grouped
    return query
```

Update `_add_linear`:

```python
compute_dtype = self.config.data_type
lin_cfg = _tm.LinearConfig()
lin_cfg.input_dim = in_dim
lin_cfg.output_dim = out_dim
lin_cfg.data_type = compute_dtype or _tm.DataType.TYPE_INVALID
lin_cfg.format = linear.weight_format.make_data_format()
lin_cfg.has_bias = 'bias' in linear.tensors
```

Keep the existing trivial-weight allocation rule. There is no builder-side
format branch.

### 8. Update explicit trivial formats and dtype ownership

Python dequantization receives `torch.dtype`, not `_tm.DataType`. Change the
base format contract and every override accordingly:

```python
def dequant(
    self,
    tensors: dict[str, Tensor],
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    raise NotImplementedError(
        f'{type(self).__name__}.dequant not implemented'
    )
```

FP8 uses the Torch dtype directly instead of translating a TM enum back
through `_CPP_TO_TORCH`:

```python
def dequant(
    self,
    tensors: dict[str, Tensor],
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    weight = tensors['weight']
    scales = tensors['scales']
    fp8_weight = weight.view(torch.float8_e4m3fn).float()
    scale = scales.float()
    scale = scale.repeat_interleave(self.block_in, dim=0)
    scale = scale.repeat_interleave(self.block_out or 1, dim=1)
    scale = scale[:fp8_weight.shape[0], :fp8_weight.shape[1]]
    result: dict[str, Tensor] = {
        'weight': (fp8_weight * scale).to(dtype),
    }
    if 'bias' in tensors:
        result['bias'] = tensors['bias']
    return result
```

The Python runtime context exposes the Torch representation of its existing
TM computation dtype once, rather than making every caller repeat the mapping:

```python
class Context:
    def __init__(self, devices, gemm, data_type, gemm_input_dtype):
        self.devices = devices
        self.gemm = gemm
        self.data_type = data_type
        self.dtype = _CPP_TO_TORCH[data_type]
        self.gemm_input_dtype = gemm_input_dtype
        self._active_mask_stack = [(True,) * len(devices)]
```

`linear.py::_dequant_linear` converts back to a TM dtype only when constructing
the new trivial descriptor:

```python
def _dequant_linear(
    linear: Linear,
    *,
    dtype: torch.dtype,
) -> Linear:
    from .builders._base import _torch_dtype_to_cpp
    from .weight_format import TrivialFormat

    fmt = linear.weight_format
    new_tensors = {
        kind: tensor.to(dtype)
        for kind, tensor in fmt.dequant(
            linear.tensors,
            dtype,
        ).items()
    }
    trivial = TrivialFormat(
        weight_dtype=_torch_dtype_to_cpp(dtype),
    )
    return Linear(tensors=new_tensors, weight_format=trivial)
```

Change `dequant_mixed` and both builder callers to pass the Torch dtype:

```python
def dequant_mixed(
    *linears: Linear | None,
    dtype: torch.dtype,
) -> tuple[Linear | None, ...]:
    formats = {
        linear.weight_format
        for linear in linears
        if linear is not None
    }
    if len(formats) <= 1:
        return linears
    return tuple(
        _dequant_linear(linear, dtype=dtype)
        if linear is not None else None
        for linear in linears
    )
```

```python
q, k, v, gate = dequant_mixed(
    q,
    k,
    v,
    gate,
    dtype=self._ctx.dtype,
)
```

```python
q, k, v, z, b, a = dequant_mixed(
    q,
    k,
    v,
    in_proj_z,
    in_proj_b,
    in_proj_a,
    dtype=self._ctx.dtype,
)
```

Change `reorder_rotary_emb` to receive that same Torch dtype directly:

```python
def reorder_rotary_emb(
    x,
    head_dim: int,
    rope_dim: int,
    *,
    dtype: torch.dtype,
):
    if isinstance(x, Linear):
        weight_format = x.weight_format
        block_out = weight_format.block_out or 0

        if block_out and block_out % head_dim != 0:
            x = _dequant_linear(x, dtype=dtype)
            block_out = 0

        new_tensors = {}
        for kind, tensor in x.tensors.items():
            if kind in ('scales', 'zeros') and block_out > 0:
                blocks_per_head = block_out // head_dim
                if blocks_per_head <= 1:
                    new_tensors[kind] = tensor
                else:
                    rope_dim_blocks = (
                        rope_dim * blocks_per_head // head_dim
                    )
                    new_tensors[kind] = _reorder_rotary_emb(
                        tensor,
                        blocks_per_head,
                        rope_dim_blocks,
                    )
            elif tensor.size(-1) % head_dim == 0:
                new_tensors[kind] = _reorder_rotary_emb(
                    tensor,
                    head_dim,
                    rope_dim,
                )
            else:
                new_tensors[kind] = tensor

        return Linear(
            tensors=new_tensors,
            weight_format=x.weight_format,
        )

    return _reorder_rotary_emb(x, head_dim, rope_dim)
```

Every caller in Llama, Mixtral, InternLM2, GPT-OSS, Qwen2, Qwen3, and the
Qwen3.5 text and vision paths uses the bound runtime context:

```python
return reorder_rotary_emb(
    x,
    cfg.head_dim,
    cfg.rope.dim,
    dtype=self._ctx.dtype,
)
```

The Qwen3.5 vision attention calls use the same argument with their existing
`real_hd` dimensions:

```python
q = reorder_rotary_emb(
    q,
    real_hd,
    real_hd,
    dtype=self._ctx.dtype,
)
k = reorder_rotary_emb(
    k,
    real_hd,
    real_hd,
    dtype=self._ctx.dtype,
)
```

Every explicit trivial format likewise receives its TM descriptor dtype from
the runtime context. For example, Qwen3.5 vision patch preprocessing returns:

```python
return Linear(
    tensors=tensors,
    weight_format=TrivialFormat(
        weight_dtype=self._ctx.data_type,
    ),
)
```

Return the locally selected `data_type` from `get_tm_config`:

```python
return model, model_path, data_type
```

Qwen3.5 vision keeps its independently selected dtype outside the resolver.
Resolve its trivial format explicitly in `get_tm_config`:

```python
vision_config = getattr(hf_model_cfg, 'vision_config', None)
if (
    not engine_config.language_model_only
    and vision_config is not None
):
    vision_executable = _get_executable_dtypes(
        [],
        engine_config.devices[0],
    )
    vision_dtype_name = _resolve_dtype(
        requested_dtype,
        vision_config,
        vision_executable,
    )
    vision_dtype = getattr(torch, vision_dtype_name)
    vision_data_type = _torch_dtype_to_cpp(vision_dtype)
    vision_resolver = WeightFormatResolver(formats=[
        TrivialFormat(weight_dtype=vision_data_type),
    ])

    init_kwargs['vision_resolver'] = vision_resolver
    init_kwargs['vision_data_type'] = vision_data_type
```

The complete new Qwen3.5 aggregate constructor state is:

```python
def __init__(
    self,
    cfg: Qwen3_5Config | Qwen3_5MoeConfig,
    *,
    resolver,
    vision_resolver=None,
    vision_data_type=None,
    language_model_only: bool = False,
):
    text_cfg = getattr(cfg, 'text_config', cfg)
    if text_cfg is None:
        raise ValueError(
            'Qwen3_5Model requires a checkpoint with text_config.'
        )
    self.text_model = Qwen3_5TextModel(
        text_cfg,
        resolver=resolver,
    )

    vision_cfg = getattr(cfg, 'vision_config', None)
    if language_model_only or vision_cfg is None:
        self.vision_model = None
        self._vision_data_type = None
    else:
        if vision_resolver is None or vision_data_type is None:
            raise TypeError(
                'vision_resolver and vision_data_type are required '
                'when the vision model is enabled'
            )
        self.vision_model = Qwen3_5VisionModel(
            vision_cfg,
            resolver=vision_resolver,
        )
        self._vision_data_type = vision_data_type
```

Its vision runtime context uses that stored TM dtype:

```python
vision_ctx = Context(
    ctx.devices,
    ctx.gemm,
    data_type=self._vision_data_type,
    gemm_input_dtype=ctx.gemm_input_dtype,
)
```

All Qwen3.5 vision C++ configuration fields that currently read
`self._resolver.data_type` use `self._ctx.data_type` instead. Python tensor
conversion and dequantization use `self._ctx.dtype`.

Delete every `resolver.data_type` and `_resolver.data_type` access.

## DLPack scope

`src/turbomind/python/dlpack.h` and concrete DLPack conversion in `bind.cpp`
remain unchanged:

- no abstract DLPack code is introduced;
- `getDataType(DLDataType)` never returns `kGenericFloat`;
- tensor export never accepts `kGenericFloat`;
- all imported and exported tensors retain concrete storage dtypes.

## Exclusions

This plan does not:

- change checkpoint suffix priority;
- create a new test module;
- resume or implement the suspended Python Linear API plan.

## Verification

Use the existing build, linear-test, and model-smoke surfaces. Extend the
existing `tests/turbomind/linear/test_linear.py`; do not create another test
module.

### Focused format-contract coverage

Add one test to the existing linear test file. It directly covers the Python
contract that the existing CUDA fixture bypasses by constructing concrete C++
`DataFormat` objects itself:

```python
@tm_required
def test_weight_format_dtype_contract():
    import _turbomind as tm

    from lmdeploy.turbomind.weight_format import (
        AWQFormat,
        CompressedTensorFormat,
        FP8Format,
        GPTQFormat,
        TrivialFormat,
        WeightFormatResolver,
        _GENERIC_FLOAT_DTYPES,
    )

    expected = frozenset({
        torch.float16,
        torch.bfloat16,
        torch.float32,
    })
    assert _GENERIC_FLOAT_DTYPES == expected

    for dtype in (*expected, torch.float64, torch.int32):
        awq_scales = torch.ones((1, 8), dtype=dtype)
        gptq_scales = torch.ones((1, 8), dtype=dtype)
        compressed_scales = torch.ones((1, 8), dtype=dtype)
        fp8_scales = torch.ones((1, 1), dtype=dtype)

        cases = (
            (
                AWQFormat(block_in=128),
                {
                    '.qweight': torch.zeros(
                        (128, 1),
                        dtype=torch.int32,
                    ),
                    '.scales': awq_scales,
                    '.qzeros': torch.zeros(
                        (1, 1),
                        dtype=torch.int32,
                    ),
                },
            ),
            (
                GPTQFormat(block_in=128),
                {
                    '.qweight': torch.zeros(
                        (16, 8),
                        dtype=torch.int32,
                    ),
                    '.scales': gptq_scales,
                },
            ),
            (
                CompressedTensorFormat(block_in=128),
                {
                    '.weight_packed': torch.zeros(
                        (16, 8),
                        dtype=torch.int32,
                    ),
                    '.weight_scale': compressed_scales,
                },
            ),
            (
                FP8Format(block_out=128),
                {
                    '.weight': torch.zeros(
                        (128, 128),
                        dtype=torch.uint8,
                    ),
                    '.weight_scale_inv': fp8_scales,
                },
            ),
        )

        accepted = dtype in expected
        for weight_format, available in cases:
            assert weight_format.accepts(available) is accepted
            if accepted:
                resolver = WeightFormatResolver(
                    formats=[weight_format],
                )
                linear = resolver._build_linear(
                    weight_format,
                    available,
                )
                assert linear.tensors['scales'].dtype == dtype
                if (
                    weight_format.zeros_dtype
                    != tm.DataType.TYPE_INVALID
                ):
                    assert linear.tensors['zeros'].dtype == dtype
                data_format = weight_format.make_data_format()
                assert (
                    data_format.scales.dtype
                    == tm.DataType.TYPE_GENERIC_FLOAT
                )

    assert not AWQFormat(block_in=128).accepts({
        '.qweight': torch.zeros((128, 1), dtype=torch.int32),
        '.scales': torch.ones((1, 8), dtype=torch.float16),
    })

    fp16 = TrivialFormat(weight_dtype=tm.DataType.TYPE_FP16)
    bf16 = TrivialFormat(weight_dtype=tm.DataType.TYPE_BF16)
    assert fp16 != bf16
    assert len({fp16, bf16}) == 2
```

The GPTQ and compressed-tensors cases intentionally omit their zero tensors
and must be accepted because those formats synthesize symmetric zeros. The AWQ
case separately verifies that its mandatory `.qzeros` cannot be omitted.

### Static checks

Run:

```bash
rg -n 'kGenericFloat|TYPE_GENERIC_FLOAT' src/turbomind lmdeploy/turbomind
rg -n 'resolver\.data_type|_resolver\.data_type' lmdeploy/turbomind
rg -n 'make_data_format\([^)]*(data_type|dtype|tensors)' lmdeploy/turbomind
rg -n 'TrivialFormat\(\)' lmdeploy/turbomind
git diff --check
```

Audit every intentional `kGenericFloat` occurrence. Each must be format
metadata, format-matching logic, Python enum binding, or string formatting. It
must not intentionally appear in DLPack, CUDA, cuBLAS, tensor dispatch,
allocation, or concrete bridge targets. No defensive storage checks are added;
using the abstract type in those paths is undefined behavior.

### Build

Build from `build/` without setting `PYTHONPATH`:

```bash
ninja
```

### Existing Python surface

This test tree executes CUDA-backed fixtures. Before running it, call
`get_gpu_usage`, choose an empty GPU, and run the command outside the sandbox.
Use the in-tree extension:

```bash
get_gpu_usage

CUDA_VISIBLE_DEVICES=<EMPTY_GPU> \
PYTHONPATH=/data/lmdeploy-gemm/build/lib:/data/lmdeploy-gemm \
python -m pytest tests/turbomind/linear
```

Do not count skipped unsupported cases as coverage.

### GPU model smoke tests

Before every GPU command, run `get_gpu_usage` and select an empty GPU. Run GPU
commands outside the sandbox. Do not modify
`scripts/test_turbomind_model.py`, and request at least 128 generated tokens.

Use the locally cached models under `/mnt_cfs/huggingface_hub/hub/`:

1. `Qwen/Qwen3-8B` verifies concrete trivial BF16 materialization.
2. `Qwen/Qwen3-8B-AWQ` verifies generic U4 metadata, concrete qparameter
   tensors, and C++ bridge conversion.
3. `Qwen/Qwen3.5-35B-A3B-FP8` verifies generic FP8 scale metadata and separate
   Qwen3.5 vision/text dtype ownership.

For every run, inspect the response and require meaningful human words
relevant to the prompt. A zero exit status with gibberish is a failure.

## Completion criteria

This prerequisite is complete only when:

- `kGenericFloat` exists only as non-storage format metadata;
- DLPack remains concrete-only;
- every `WeightFormat.make_data_format()` is zero-argument;
- floating checkpoint qparameters preserve their concrete tensor dtype;
- U4 zeros use the concrete scale dtype;
- every family bridge target is concrete;
- dtype discovery uses the same generic descriptors used for family planning;
- `WeightFormatResolver` contains no component dtype;
- trivial formats, allocations, and copied buffers use one selected concrete
  dtype;
- the focused format-contract test and existing linear tests pass;
- all three model smokes produce meaningful responses of at least 128 tokens.

After these criteria are met, re-audit
`plans/python_linear_api_refactor.md` before resuming it.
