# Merge TurboMind DLPack tensor import

## Status

Approved for implementation.

## Goal

Expose one native tensor-import API with one signature:

```python
tensor = _tm.from_dlpack(source)
```

`from_dlpack` preserves the shape and strides published through supported DLPack layouts. Remove `from_dlpack_with_strides`; callers must not choose between a contiguous interpretation and a strided interpretation of the same DLPack tensor.

Remove the existing three-argument `from_dlpack(source, logical_shape, logical_dtype)` overload as well. Importing a DLPack tensor and reinterpreting its storage are separate operations. The two packed UINT4 and FP4 fixture paths must use the existing `Tensor.reinterpret()` method explicitly.

## Current problem

`src/turbomind/python/bind.cpp` currently has two implementations with conflicting behavior:

- `from_dlpack(source)` discards `DLTensor::strides` and constructs a contiguous TurboMind layout.
- `from_dlpack_with_strides(source)` preserves `DLTensor::strides`.

The split forces callers to know an implementation detail of the binding. It also leaves ordinary users of `from_dlpack` with a fabricated layout when the producer exports a non-contiguous tensor.

The strided implementation additionally probes PyTorch-specific `untyped_storage()` and `storage_offset()` information. DLPack publishes a tensor view, not the capacity of its producer's backing allocation. TurboMind only needs the storage span addressed by the published layout.

## Settled interface

The module exposes exactly one function signature:

```python
from_dlpack(source) -> Tensor
```

There is no `from_dlpack_with_strides` alias, three-argument `from_dlpack` overload, or compatibility wrapper.

The contracts are:

1. `from_dlpack(source)` consumes one DLPack capsule and preserves `shape`, `strides`, `dtype`, `device`, and `byte_offset` for a rank-1-or-higher tensor with strictly positive extents and nonnegative element strides.
2. A null DLPack stride pointer means row-major contiguous layout.
3. The imported Buffer spans `layout.cosize()` elements beginning at `data + byte_offset`.
4. The imported TurboMind Tensor owns the `DLManagedTensor` and invokes its deleter when the last native reference is destroyed.
5. Packed storage is imported normally and then explicitly reinterpreted with `Tensor.reinterpret(logical_dtype, logical_shape)`. Contiguous physical storage and equal source/destination byte extents remain requirements of `reinterpret()`.
6. This change does not add a DLPack stream argument or change synchronization. Existing APIs continue to pass the active CUDA stream separately where execution needs one.
7. `Tensor.copy_from(source)` remains the synchronous raw-byte loader operation.
8. `Tensor.copy_from(source, stream_ptr)` is the asynchronous raw-byte form used by functional Linear weight preparation. Source contiguity and byte extent are unchecked caller preconditions. The source storage must remain alive until the supplied stream passes the copy, or its allocator must guarantee ordered reuse on that same stream.
9. Rank-zero tensors and negative-stride layouts are unchecked contract violations. This refactor does not extend `core::Layout` to represent them and does not add a Python validation path.
10. Zero or negative extents are rejected before capsule consumption because `core::Layout::cosize()` does not define storage capacity for arbitrary empty layouts.
11. Supported DLPack devices are CPU, CUDA host, and CUDA. Other device types are rejected before capsule consumption.
12. Supported DLPack dtypes are scalar bool8, signed and unsigned integers of 8/16/32/64 bits, float16/32/64, bfloat16, and float8 E4M3FN. `lanes` must equal one. Other dtype descriptors are rejected before capsule consumption.

## 1. Unify native import

Use a symmetric pair of private bridge names in `src/turbomind/python/bind.cpp`:

```cpp
DLManagedTensor* ToDLPack(Tensor& tensor);
std::shared_ptr<Tensor> FromDLPack(const py::object& object);
```

Rename the existing exporter:

```cpp
DLManagedTensor* TritonTensorToDLManagedTensor(Tensor& tensor)
```

to:

```cpp
DLManagedTensor* ToDLPack(Tensor& tensor)
```

Its body remains unchanged. `Tensor` is `turbomind::core::Tensor`; no native bridge symbol should retain the stale `TritonTensor` name.

Replace `DLManagedTensorToTritonTensor`, `DLManagedTensorToTritonTensorWithStrides`, and `TorchStorageCapacityElements` with one importer used by every binding path:

Replace `getMemoryType` with the following complete implementation so an unknown DLPack device cannot silently become CPU:

```cpp
ft::DeviceType getMemoryType(DLDevice device)
{
    switch (device.device_type) {
        case DLDeviceType::kDLCPU:
            return ft::DeviceType::kCPU;
        case DLDeviceType::kDLCUDAHost:
            return ft::DeviceType::kCPUpinned;
        case DLDeviceType::kDLCUDA:
            return ft::DeviceType::kDEVICE;
        default:
            throw py::value_error("unsupported DLPack device type");
    }
}
```

Replace `getDataType` with the following complete scalar-dtype conversion:

```cpp
ft::DataType getDataType(DLDataType data_type)
{
    if (data_type.lanes != 1) {
        throw py::value_error("DLPack vector dtypes are not supported");
    }

    using ft::data_type_v;
    switch (data_type.code) {
        case DLDataTypeCode::kDLUInt:
            switch (data_type.bits) {
                case 8:
                    return data_type_v<uint8_t>;
                case 16:
                    return data_type_v<uint16_t>;
                case 32:
                    return data_type_v<uint32_t>;
                case 64:
                    return data_type_v<uint64_t>;
                default:
                    break;
            }
            break;
        case DLDataTypeCode::kDLInt:
            switch (data_type.bits) {
                case 8:
                    return data_type_v<int8_t>;
                case 16:
                    return data_type_v<int16_t>;
                case 32:
                    return data_type_v<int32_t>;
                case 64:
                    return data_type_v<int64_t>;
                default:
                    break;
            }
            break;
        case DLDataTypeCode::kDLFloat:
            switch (data_type.bits) {
                case 16:
                    return data_type_v<turbomind::half_t>;
                case 32:
                    return data_type_v<float>;
                case 64:
                    return data_type_v<double>;
                default:
                    break;
            }
            break;
        case DLDataTypeCode::kDLBfloat:
            if (data_type.bits == 16) {
                return data_type_v<turbomind::bfloat16_t>;
            }
            break;
        case DLDataTypeCode::kDLBool:
            if (data_type.bits == 8) {
                return data_type_v<bool>;
            }
            break;
        case DLDataTypeCode::kDLFloat8_e4m3fn:
            if (data_type.bits == 8) {
                return data_type_v<turbomind::fp8_e4m3_t>;
            }
            break;
        default:
            break;
    }
    throw py::value_error("unsupported DLPack dtype");
}
```

Both conversion functions run before `capsule.set_name("used_dltensor")`. A rejected descriptor therefore remains owned by the producer capsule and follows its normal deleter path.

```cpp
std::shared_ptr<Tensor> FromDLPack(const py::object& object)
{
    py::capsule capsule = object.attr("__dlpack__")();
    auto* managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule.ptr(), kDlTensorCapsuleName));
    auto& dl_tensor = managed->dl_tensor;

    const ft::core::Device device{getMemoryType(dl_tensor.device), dl_tensor.device.device_id};
    const auto dtype = getDataType(dl_tensor.dtype);
    assert(dl_tensor.ndim > 0);

    std::vector<ft::core::ssize_t> shape(dl_tensor.shape, dl_tensor.shape + dl_tensor.ndim);
    for (const auto extent : shape) {
        if (extent <= 0) {
            throw py::value_error("DLPack tensor dimensions must be positive");
        }
    }
    std::vector<ft::core::ssize_t> stride;
    if (dl_tensor.strides) {
        stride.assign(dl_tensor.strides, dl_tensor.strides + dl_tensor.ndim);
    }
    else {
        stride.resize(dl_tensor.ndim);
        ft::core::ssize_t value = 1;
        for (int i = dl_tensor.ndim - 1; i >= 0; --i) {
            stride[i] = value;
            value *= shape[i];
        }
    }

    ft::core::Layout layout{std::move(shape), std::move(stride)};
    auto* data = static_cast<char*>(dl_tensor.data) + dl_tensor.byte_offset;

    capsule.set_name("used_dltensor");
    std::shared_ptr<void> owner{data, [managed](void*) {
        if (managed->deleter) {
            managed->deleter(managed);
        }
    }};

    return std::make_shared<Tensor>(std::move(owner), std::move(layout), dtype, device);
}
```

The ownership transfer occurs after all descriptor parsing and layout construction. Once the capsule is renamed to `used_dltensor`, the returned Tensor's shared owner is responsible for the DLPack deleter.

Do not retain any of these names or helpers:

```cpp
TritonTensorToDLManagedTensor
DLManagedTensorToTritonTensor
DLManagedTensorToTritonTensorWithStrides
TorchStorageCapacityElements
```

`layout.cosize()` is already used by the `Tensor(std::shared_ptr<void>, Layout, DataType, Device)` constructor to size the Buffer. Do not pass `Tensor::PreserveBufferCapacity`: DLPack does not expose a larger allocation capacity to preserve.

## 2. Bind one `from_dlpack` function

Replace the three current module definitions with one definition:

```cpp
m.def("from_dlpack", &FromDLPack, "tensor"_a);
```

Delete the `from_dlpack_with_strides` module definition and the three-argument `from_dlpack` module definition completely. Do not leave an alias, warning, or deprecation path.

Do not add a Python property solely to expose `Tensor.device().id`. The stored ordinal remains private native metadata.

Update the Tensor DLPack exporter to call the renamed function:

```cpp
.def(
    "__dlpack__",
    [](Tensor& self, long stream) {
        DLManagedTensor* managed = ToDLPack(self);
        return py::capsule(managed, kDlTensorCapsuleName, [](PyObject* object) {
            DLManagedTensor* managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(object, kDlTensorCapsuleName));
            if (managed) {
                managed->deleter(managed);
            }
            else {
                PyErr_Clear();
            }
        });
    },
    "stream"_a = 0)
```

The `stream` argument and exporter behavior remain unchanged; this is only the exporter rename and corresponding local-variable cleanup.

## 3. Reuse the importer in `Tensor.copy_from`

Use `FromDLPack` in `Tensor.copy_from()`:

```cpp
.def(
    "copy_from",
    [](Tensor& self, const py::object& object) {
        auto src = FromDLPack(object);
        TM_CHECK_EQ(self.byte_size(), src->byte_size()) << self << " " << *src;
        safe_memcpy(self.raw_data(), src->raw_data(), self.byte_size());
    },
    "tensor"_a)
```

Add an asynchronous overload on the destination Tensor:

```cpp
.def(
    "copy_from",
    [](Tensor& self, const py::object& source, std::uintptr_t stream_ptr) {
        auto src = FromDLPack(source);
        TM_CUDA_CHECK(cudaMemcpyAsync(self.raw_data(), src->raw_data(), self.byte_size(), cudaMemcpyDefault, reinterpret_cast<cudaStream_t>(stream_ptr)));
    },
    "tensor"_a,
    "stream_ptr"_a)
```

The one-argument and two-argument forms are both raw storage copies. They differ only in execution:

```python
destination.copy_from(source)
destination.copy_from(source, stream.cuda_stream)
```

The first completes synchronously through the existing `safe_memcpy` path. The second enqueues `cudaMemcpyAsync` on the explicitly supplied stream.

Delete the module-level `copy_bytes_on_stream` binding. Do not duplicate capsule extraction or capsule renaming in either `Tensor.copy_from` overload.

In `lmdeploy/turbomind/linear.py`, `_copy_param` becomes:

```python
def _copy_param(self, impl, name, src, *, logical_shape, logical_dtype, stream):
    """Allocate a native parameter and copy its source bytes on the active stream."""
    dst = impl.param(name).alloc(logical_shape, logical_dtype)
    dst.copy_from(src, stream.cuda_stream)
```

`src` is a named Tensor retained by the `packed` dictionary while all preparation work is enqueued. Normalization, source allocation, copy, and native preparation use the same current Torch stream, so the Torch allocator orders reuse after the copy. Passing a temporary producer whose storage can be reclaimed independently before stream completion is outside the overload's contract.

## 4. Migrate Python callers

Replace every call to `from_dlpack_with_strides` with `from_dlpack` in these files:

```text
lmdeploy/turbomind/linear.py
tests/turbomind/linear/fixture.py
tests/turbomind/linear_attn/turbomind_gated_delta_rule.py
scripts/test_generic_copy.py
```

For example, `NativeBridge` becomes:

```python
@dataclass(frozen=True)
class NativeBridge:
    tm: object

    def tensor(self, x: torch.Tensor | None):
        return None if x is None else self.tm.from_dlpack(x)
```

Its required-symbol declaration becomes:

```python
REQUIRED_NATIVE_BRIDGE_SYMBOLS = (
    'from_dlpack',
    'delta_rule_plan',
    'delta_rule_run',
    'delta_rule_prepare_state_tma_descs',
)
```

Keep the exact remaining symbols already present in that tuple; only replace the old import spelling. Do not introduce a Python wrapper around `_tm.from_dlpack`.

Replace the two packed-weight fixture calls with an ordinary import followed by the existing explicit reinterpretation operation:

```python
quant = tm.from_dlpack(raw_weight).reinterpret(tm.DataType.TYPE_UINT4, [output_dim, case.input_dim])
quant = tm.from_dlpack(raw_blocks).reinterpret(tm.DataType.TYPE_FP4_E2M1, [output_dim, case.input_dim])
```

## 5. Update native diagnostics

In `src/turbomind/kernels/linear_attn/python_bind.cpp`, update the type-error message to name the sole conversion API:

```cpp
throw py::type_error(std::string(name) + " must be a _turbomind.Tensor; use _turbomind.from_dlpack");
```

No change is needed in `src/turbomind/kernels/gemm/moe_gate_python_bind.cpp`; it already names `_turbomind.from_dlpack`.

## 6. Scope boundaries

This refactor does not:

- add GEMM or runtime Tensor validation outside the DLPack descriptor boundary;
- add rank-zero or negative-stride support to `core::Layout`;
- add a DLPack stream parameter;
- change CUDA synchronization;
- change `GenericCopy` or its kernels;
- change the synchronous raw-copy semantics of `Tensor.copy_from(source)`;
- change `Tensor.reinterpret()`;
- add persistent tests;
- retain a compatibility alias for `from_dlpack_with_strides`.

## 7. Verification

Do not install LMDeploy and do not run `setup.py`.

### 7.1 Static checks

The removed APIs and stale helper names must have no remaining references:

```bash
rg -n 'from_dlpack_with_strides|copy_bytes_on_stream|TritonTensor|DLManagedTensorToTritonTensor|TorchStorageCapacityElements' src lmdeploy tests scripts
```

The command must return no matches.

Confirm that the fixture expresses packed reinterpretation explicitly and no multi-argument import remains:

```bash
rg -n 'from_dlpack\([^)]*,' src lmdeploy tests scripts
rg -n 'from_dlpack\([^\n]*\)\.reinterpret\(' tests/turbomind/linear/fixture.py
```

The first command must return no matches. The second must report the UINT4 and FP4 fixture paths.

Inspect the `FromDLPack` construction directly and require the DLPack ordinal to initialize `core::Device`:

```bash
rg -n 'dl_tensor\.device\.device_id' src/turbomind/python/bind.cpp
```

The match must be the `core::Device` construction passed into the imported Tensor. There is no Python device-ID accessor and no test-only observability API.

Confirm formatting and the exact changed surface:

```bash
git diff --check
git diff --stat
```

### 7.2 Python syntax

Run from the repository root:

```bash
PYTHONPATH=/data/lmdeploy-gemm python -m py_compile lmdeploy/turbomind/linear.py tests/turbomind/linear/fixture.py tests/turbomind/linear_attn/turbomind_gated_delta_rule.py scripts/test_generic_copy.py
```

Verify that an empty strided tensor is rejected instead of producing a Tensor with an invalid Buffer capacity:

```bash
PYTHONPATH=/data/lmdeploy-gemm python - <<'PY'
import torch

from lmdeploy.turbomind import _tm

try:
    _tm.from_dlpack(torch.empty_strided((0,), (7,)))
except ValueError as error:
    assert str(error) == 'DLPack tensor dimensions must be positive'
else:
    raise AssertionError('empty DLPack tensor was accepted')
PY
```

### 7.3 Build

Build without setting `PYTHONPATH`:

```bash
cd /data/lmdeploy-gemm/build
ninja _turbomind
```

### 7.4 GPU selection

Immediately before each GPU command, use `get_gpu_usage` and select an empty GPU. Run every GPU command outside the sandbox. If a command reports OOM, query GPU usage again before retrying.

### 7.5 Nonzero byte offset and managed ownership

Run this transient custom producer on an empty GPU:

```bash
CUDA_VISIBLE_DEVICES="$EMPTY_GPU" PYTHONPATH=/data/lmdeploy-gemm python - <<'PY'
import ctypes
import gc

import torch

from lmdeploy.turbomind import _tm


class DLDevice(ctypes.Structure):
    _fields_ = [('device_type', ctypes.c_int32), ('device_id', ctypes.c_int32)]


class DLDataType(ctypes.Structure):
    _fields_ = [('code', ctypes.c_uint8), ('bits', ctypes.c_uint8), ('lanes', ctypes.c_uint16)]


class DLTensor(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.c_void_p),
        ('device', DLDevice),
        ('ndim', ctypes.c_int32),
        ('dtype', DLDataType),
        ('shape', ctypes.POINTER(ctypes.c_int64)),
        ('strides', ctypes.POINTER(ctypes.c_int64)),
        ('byte_offset', ctypes.c_uint64),
    ]


class DLManagedTensor(ctypes.Structure):
    pass


DLDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensor))
DLManagedTensor._fields_ = [
    ('dl_tensor', DLTensor),
    ('manager_ctx', ctypes.c_void_p),
    ('deleter', DLDeleter),
]


get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
get_pointer.restype = ctypes.c_void_p


class Producer:
    def __init__(self, capsule):
        self.capsule = capsule

    def __dlpack__(self):
        return self.capsule


base = torch.arange(16, device='cuda', dtype=torch.int32)
view = base[4:12]
expected = view.clone()

capsule = view.__dlpack__()
managed = ctypes.cast(get_pointer(capsule, b'dltensor'), ctypes.POINTER(DLManagedTensor))
managed.contents.dl_tensor.data = base.data_ptr()
managed.contents.dl_tensor.byte_offset = 4 * base.element_size()

deleter_calls = [0]
original_address = ctypes.cast(managed.contents.deleter, ctypes.c_void_p).value
original_deleter = DLDeleter(original_address)


@DLDeleter
def counting_deleter(pointer):
    deleter_calls[0] += 1
    original_deleter(pointer)


managed.contents.deleter = counting_deleter
producer = Producer(capsule)
imported = _tm.from_dlpack(producer)

del base, view, producer, capsule
gc.collect()
assert deleter_calls[0] == 0

restored = torch.from_dlpack(imported)
assert torch.equal(restored, expected)

del imported
gc.collect()
assert deleter_calls[0] == 0

del restored
gc.collect()
assert deleter_calls[0] == 1

gc.collect()
assert deleter_calls[0] == 1
PY
```

The descriptor retains the shape of `base[4:12]`, but changes its representation from an already-offset pointer to `base.data_ptr()` plus a nonzero byte offset. Equality with `expected` therefore verifies that import uses `data + byte_offset`.

The custom deleter forwards to Torch's original deleter. Copy its address before replacing the structure field; retaining the ctypes field wrapper directly would resolve through the replaced field and recurse. Destroying the Python source references must not invoke the deleter while the imported Tensor owns the capsule payload. The Torch round trip holds another native Tensor reference, so the original managed deleter must run only after both `imported` and `restored` are destroyed, and repeated collection must leave the count at exactly one.

Do not retain the ctypes producer in the repository and do not add a test case.

### 7.6 Strided DLPack import and GenericCopy

Run the existing generic-copy script unchanged:

```bash
CUDA_VISIBLE_DEVICES="$EMPTY_GPU" PYTHONPATH=/data/lmdeploy-gemm python scripts/test_generic_copy.py --dtype f32
```

Require every contiguous, sliced, transposed, permuted, rank-1, rank-3, and rank-4 case to pass. This is the direct verification that ordinary `from_dlpack()` now preserves strides.

### 7.7 Linear API and packed reinterpretation

Run the existing Linear tests:

```bash
CUDA_VISIBLE_DEVICES="$EMPTY_SM90_GPU" PYTHONPATH=/data/lmdeploy-gemm pytest -q tests/turbomind/linear/test_linear.py
```

This covers ordinary stride-preserving imports used by the public Linear API and the FP4 `Tensor.reinterpret()` path.

Run one existing U4 fixture case explicitly:

```bash
CUDA_VISIBLE_DEVICES="$EMPTY_SM90_GPU" PYTHONPATH=/data/lmdeploy-gemm python - <<'PY'
from tests.turbomind.linear.cases import case_by_name
from tests.turbomind.linear.fixture import LinearFixture

case = case_by_name()['llama2_7b_o__bf16_u4k128_bf16']
fixture = LinearFixture(case)
try:
    fixture.prepare_batch(1)
    fixture.run_reference()
    fixture.run_linear()
    fixture.check_tolerances(fixture.compare())
finally:
    fixture.close()
PY
```

This directly covers the explicit UINT4 `Tensor.reinterpret()` path without adding a test.

### 7.8 Linear-attention bridge

Run the existing strided-QKV and generated-smoke coverage:

```bash
CUDA_VISIBLE_DEVICES="$EMPTY_GPU" PYTHONPATH=/data/lmdeploy-gemm pytest -q tests/turbomind/linear_attn/test_gated_delta_rule.py -k 'packed_strided_qkv_matches_normalized_reference or delta_rule_wrapper_all_generated_smoke_matches_reference'
```

No test file or test case is added.

### 7.9 MoE gate regression

The MoE gate already uses ordinary `from_dlpack`; run its existing test file to confirm that contiguous imports retain their behavior:

```bash
CUDA_VISIBLE_DEVICES="$EMPTY_GPU" PYTHONPATH=/data/lmdeploy-gemm pytest -q tests/turbomind/moe_gate/test_moe_gate_v2.py
```

### 7.10 TurboMind model smoke

Select an empty SM90 GPU immediately before the command and run the unchanged repository script outside the sandbox:

```bash
PYTHONPATH=/data/lmdeploy-gemm python scripts/test_turbomind_model.py --model-id Qwen/Qwen3-8B --cache-dir /mnt_cfs/huggingface_hub/hub/ --gpus "$EMPTY_SM90_GPU" --tp 1 --max-new-tokens 128 --prompt "Explain why tensor shape and stride metadata must remain consistent when sharing CUDA tensors between PyTorch and an inference engine."
```

The command must generate at least 128 tokens. Inspect the response and require meaningful human language relevant to the prompt; an exit code alone is insufficient.

## Completion criteria

The work is complete when:

1. `_tm.from_dlpack(source)` preserves DLPack shape and strides.
2. Packed UINT4 and FP4 fixture storage uses `_tm.from_dlpack(source).reinterpret(logical_dtype, logical_shape)`.
3. `_tm.from_dlpack_with_strides` no longer exists.
4. DLPack capsule consumption and lifetime ownership have one C++ implementation.
5. No Torch-specific storage-capacity probe remains in the native importer.
6. All production, fixture, linear-attention, and generic-copy callers use `from_dlpack`.
7. The private conversion pair is named `ToDLPack` and `FromDLPack`; no `TritonTensor` name remains.
8. The module-level `copy_bytes_on_stream` no longer exists; functional Linear preparation uses `Tensor.copy_from(source, stream_ptr)`.
9. `FromDLPack` initializes `core::Device` with both the mapped device type and `DLDevice::device_id`, without adding Python API for the private ordinal.
10. Unsupported DLPack devices, bit widths, dtype codes, and vector lanes are rejected before capsule consumption.
11. Zero and negative extents are rejected before capsule consumption.
12. A nonzero `DLTensor::byte_offset` is applied correctly, the imported Tensor retains the producer allocation after the original Python references are destroyed, and the managed deleter runs exactly once after the last owner is destroyed.
13. The build and all listed existing verification workloads pass.
14. The TurboMind model response is manually confirmed meaningful and relevant.
