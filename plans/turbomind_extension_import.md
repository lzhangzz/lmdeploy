# TurboMind native-module layout and import refactor

## Status

Approved for implementation.

## Goal

Package one native submodule, `_turbomind`, under `lmdeploy.turbomind`, matching PyTorch's `torch._C` layout. Fold the TurboMind XGrammar bindings into that module.

The native imports are always package-relative:

```python
from . import _turbomind as _tm
```

The package performs that import once and records whether the optional native capability is available.

There is no `bootstrap()`, `sys.path` mutation, relative-versus-top-level import branch, dynamic import, cached loader, origin check, proxy, or module object passed through function arguments.

## Module layout

The build and wheel must expose the same module names inside the `lmdeploy.turbomind` package:

```text
# In-tree build.
lmdeploy/turbomind/_turbomind<extension-suffix>
lmdeploy/turbomind/lib/libturbomind_nccl_stub<shared-library-suffix>

# Installed wheel.
site-packages/lmdeploy/turbomind/_turbomind<extension-suffix>
site-packages/lmdeploy/turbomind/lib/libturbomind_nccl_stub<shared-library-suffix>
```

The Python extension module lives directly in `lmdeploy/turbomind`, and its dependent shared libraries live in `lmdeploy/turbomind/lib`, just as PyTorch keeps `torch._C` in `torch` and its dependent libraries in `torch/lib`.

## 1. Build and packaging

### 1.1 Fold XGrammar bindings into `_turbomind`

In `src/turbomind/python/CMakeLists.txt`, compile `xgrammar_bind.cpp` into the existing module and delete the `_xgrammar` target:

```cmake
pybind11_add_module(${PROJECT_NAME}
    bind.cpp
    linear_bind.cpp
    xgrammar_bind.cpp
    ../kernels/linear_attn/python_bind.cpp
    ../kernels/gemm/moe_gate_python_bind.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE turbomind xgrammar)
```

Delete:

```cmake
pybind11_add_module(_xgrammar xgrammar_bind.cpp)
target_link_libraries(_xgrammar PRIVATE core xgrammar)
target_compile_features(_xgrammar PRIVATE cxx_std_14)
```

In `xgrammar_bind.cpp`, replace its module initializer with an ordinary binding function:

```cpp
namespace turbomind::python {

void bind_xgrammar(py::module_& m)
{
```

The existing binding statements remain the body of that function, followed by:

```cpp
}

}  // namespace turbomind::python
```

Declare the function beside the existing binding declarations in `bind.cpp`:

```cpp
namespace turbomind::python {
void bind_xgrammar(pybind11::module_& m);
}
```

Call it as the first registration in the `_turbomind` initializer so `CompiledGrammar` is registered before `ModelRequest::set_grammar`:

```cpp
PYBIND11_MODULE(_turbomind, m)
{
    turbomind::python::bind_xgrammar(m);
    py::module_ multimodal = m.def_submodule("multimodal");
```

The separately linked `libturbomind_nccl_stub` remains unchanged. Its weak compatibility symbols must follow the real NCCL library in dynamic-link order and therefore must not be compiled into `_turbomind`.

### 1.2 `setup.py`

Declare the extension with its real qualified module name. `cmake_build_extension` then sets the CMake install prefix to the staged `lmdeploy/turbomind` package directory:

```python
ext_modules = [
    cmake_build_extension.CMakeExtension(
        name='lmdeploy.turbomind._turbomind',
        cmake_depends_on=['pybind11'],
        source_dir=str(Path(__file__).parent.absolute()),
        cmake_generator=None if os.name == 'nt' else 'Ninja',
        cmake_build_type=os.getenv('CMAKE_BUILD_TYPE', 'Release'),
        cmake_configure_options=[
            f'-DPython3_ROOT_DIR={Path(sys.prefix)}',
            f'-DPYTHON_EXECUTABLE={Path(sys.executable)}',
            '-DBUILD_SHARED_LIBS:BOOL=OFF',
            '-DBUILD_PY_FFI=ON',
            '-DBUILD_MULTI_GPU=' + ('OFF' if os.name == 'nt' else 'ON'),
            '-DUSE_NVTX=' + ('OFF' if os.name == 'nt' else 'ON'),
        ],
    ),
]
```

Remove both of these old settings:

```python
install_prefix='lmdeploy/lib'
'-DCALL_FROM_SETUP_PY:BOOL=ON'
```

### 1.3 Top-level `CMakeLists.txt`

Use one output layout and one install layout. A packaging build sets `CMAKE_INSTALL_PREFIX` to the staged `lmdeploy/turbomind` package directory. The extension modules install at that directory's root and their private runtime library installs under its `lib` directory.

Replace the complete `BUILD_PY_FFI` block after `add_subdirectory(src)` with:

```cmake
if(BUILD_PY_FFI)
  set(_TURBOMIND_PYTHON_PACKAGE_DIR "${CMAKE_SOURCE_DIR}/lmdeploy/turbomind")

  set_target_properties(
    _turbomind
    PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "$<1:${_TURBOMIND_PYTHON_PACKAGE_DIR}>"
      RUNTIME_OUTPUT_DIRECTORY "$<1:${_TURBOMIND_PYTHON_PACKAGE_DIR}>")

  if(TARGET turbomind_nccl_stub)
    set_target_properties(
      turbomind_nccl_stub
      PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "$<1:${_TURBOMIND_PYTHON_PACKAGE_DIR}/lib>"
        RUNTIME_OUTPUT_DIRECTORY "$<1:${_TURBOMIND_PYTHON_PACKAGE_DIR}/lib>")
  endif()

  install(
    TARGETS _turbomind
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .)

  if(TARGET turbomind_nccl_stub)
    install(
      TARGETS turbomind_nccl_stub
      LIBRARY DESTINATION lib
      RUNTIME DESTINATION lib)
  endif()
endif()
```

This removes the `CALL_FROM_SETUP_PY` install branch and does not use an absolute source-tree destination in an install rule.

This is the PyTorch in-tree model: `ninja` builds the extension and its private runtime library directly into the Python package that imports them. It creates one canonical development copy instead of copying or loading another binary from `build/lib`.

The always-true generator expressions prevent Visual Studio and other multi-configuration generators from appending `Release`, `Debug`, or another configuration directory to either output path.

After the Windows build, verify the in-tree output directly:

```powershell
if (-not (Test-Path 'lmdeploy/turbomind/_turbomind*.pyd')) { throw '_turbomind.pyd is missing from lmdeploy/turbomind' }
if (Test-Path 'lmdeploy/turbomind/Release/_turbomind*.pyd') { throw '_turbomind.pyd was placed under Release' }
if (Test-Path 'lmdeploy/turbomind/Debug/_turbomind*.pyd') { throw '_turbomind.pyd was placed under Debug' }
```

Add these generated artifacts to `.gitignore`:

```text
lmdeploy/turbomind/_turbomind*.so
lmdeploy/turbomind/_turbomind*.pyd
lmdeploy/turbomind/lib/
```

### 1.4 Native-module RPATH

The extension modules and their private library now have the same relative layout in-tree and in a wheel. Replace the conditional `CALL_FROM_SETUP_PY` block in `src/turbomind/python/CMakeLists.txt` with:

```cmake
string(REPLACE "." ";" _cuda_version ${CMAKE_CUDA_COMPILER_VERSION})
list(GET _cuda_version 0 CUDA_MAJOR)

if(CUDA_MAJOR GREATER_EQUAL 13)
  set(_INSTALL_CUDA_RPATH
      "\$ORIGIN/lib"
      "\$ORIGIN/../../nvidia/nccl/lib"
      "\$ORIGIN/../../nvidia/cu${CUDA_MAJOR}/lib")
else()
  set(_INSTALL_CUDA_RPATH
      "\$ORIGIN/lib"
      "\$ORIGIN/../../nvidia/nccl/lib"
      "\$ORIGIN/../../nvidia/cuda_runtime/lib"
      "\$ORIGIN/../../nvidia/cublas/lib"
      "\$ORIGIN/../../nvidia/curand/lib")
endif()

set_target_properties(_turbomind PROPERTIES
    BUILD_RPATH "\$ORIGIN/lib"
    INSTALL_RPATH "${_INSTALL_CUDA_RPATH}")
```

`BUILD_RPATH` and `INSTALL_RPATH` both resolve the stub under the owning `lmdeploy/turbomind` package. The install RPATH additionally resolves NVIDIA wheel dependencies from the site-packages-level `nvidia` packages.

After these changes, `CALL_FROM_SETUP_PY` has no definition or use anywhere in the repository.

### 1.5 `MANIFEST.in`

The native binaries are wheel build outputs, not source-package data collected from the checkout. Remove:

```text
include lmdeploy/lib/*.so
include lmdeploy/lib/*.so*
include lmdeploy/lib/*.dll
include lmdeploy/lib/*.pyd
```

Keep the unrelated `lmdeploy/bin/*` rule.

## 2. One import owner

Replace `lmdeploy/turbomind/__init__.py` with:

```python
# Copyright (c) OpenMMLab. All rights reserved.

import torch  # noqa: F401

_import_error = None

try:
    from . import _turbomind as _tm
except (ImportError, OSError) as error:
    _tm = None
    _import_error = error
else:
    from .turbomind import TurboMind


def is_available() -> bool:
    return _tm is not None
```

`torch` is imported first because PyTorch initializes its native dependencies, including its Windows DLL search state, inside `torch.__init__`. LMDeploy does not duplicate that platform loader.

The import attempt happens once when `lmdeploy.turbomind` is initialized. `is_available()` is then a boolean read; it does not import, search, or load anything. `_tm` and `_import_error` are private package attributes used by repository code. `TurboMind` remains the package-level public API when the native module is available. `update_parallel_config` is called only inside `turbomind.py`; removing its unused package re-export is an intentional API cleanup with no compatibility requirement.

`benchmark/profile_throughput.py` keeps the public import:

```python
from lmdeploy.turbomind import TurboMind
```

### Missing native modules

If `_turbomind` or one of its dynamic-library dependencies is absent, `lmdeploy.turbomind` remains importable with `_tm = None`, `_import_error` containing the original loader exception, and `is_available() == False`. `TurboMind` is not exported in that state.

This keeps every PyTorch-capable entry point importable without requiring its own exception wrapper. The availability branch exists once, in the package that owns the optional native capability.

- `lmdeploy.archs.autoget_backend()` checks `is_available()` and selects the PyTorch engine when false.
- automatic AWQ backend selection checks `is_available()` and selects the Triton implementation when false.
- explicitly requesting a TurboMind engine or TurboMind AWQ backend raises a `RuntimeError` when `is_available()` is false.
- tests check `is_available()` before importing native-dependent fixtures.

Native-dependent implementation modules do not accept `_tm = None` as an operating mode. They are imported only after the availability check or through the conditionally exported `TurboMind` class.

## 3. Production consumers

Only `lmdeploy/turbomind/__init__.py` directly imports the binary module. It resolves `lmdeploy.turbomind._turbomind`; there is no top-level native module and no module under `lmdeploy.lib`.

These direct children import the owned objects with:

```python
from . import _tm
```

Apply it to:

- `lmdeploy/turbomind/converter.py`
- `lmdeploy/turbomind/model_loader.py`
- `lmdeploy/turbomind/weight_format.py`
- `lmdeploy/turbomind/turbomind.py`

`lmdeploy/archs.py` performs the one automatic engine fallback through the package capability:

```python
def autoget_backend(model_path: str, trust_remote_code: bool = False):
    from lmdeploy import turbomind

    turbomind_has = False
    if turbomind.is_available():
        from lmdeploy.turbomind.supported_models import is_supported as is_supported_turbomind
        turbomind_has = is_supported_turbomind(model_path, trust_remote_code=trust_remote_code)
        if not turbomind_has:
            logger.warning(f'Fallback to pytorch engine because {model_path!r} not supported by turbomind engine.')
    else:
        logger.warning('Fallback to pytorch engine because turbomind is not built.')

    return 'turbomind' if turbomind_has else 'pytorch'
```

The two explicit TurboMind construction sites remain local to their TurboMind-only branches:

```python
# lmdeploy/serve/core/async_engine.py
def _build_turbomind(self, model_path: str, backend_config: TurbomindEngineConfig | None = None, trust_remote_code: bool = False, **kwargs):
    from lmdeploy import turbomind

    if not turbomind.is_available():
        raise RuntimeError('TurboMind was requested but its native module is unavailable.') from turbomind._import_error
    return turbomind.TurboMind.from_pretrained(model_path, engine_config=backend_config, trust_remote_code=trust_remote_code, **kwargs)
```

```python
# benchmark/profile_throughput.py
if isinstance(engine_config, TurbomindEngineConfig):
    from lmdeploy import turbomind

    if not turbomind.is_available():
        raise RuntimeError('TurboMind was requested but its native module is unavailable.') from turbomind._import_error
    tm_model = turbomind.TurboMind.from_pretrained(model_path, engine_config=engine_config, trust_remote_code=trust_remote_code)
    self.backend = 'turbomind'
```

Every CLI, server, and other benchmark that supports both engines reaches TurboMind through `autoget_backend()` or `AsyncEngine`; it does not import `lmdeploy.turbomind` directly. The static audit in verification enforces that this remains true.

`lmdeploy/turbomind/tokenizer_info.py` uses:

```python
from . import _tm
```

Replace its `class TokenizerInfo(_xgr.TokenizerInfo):` declaration with `class TokenizerInfo(_tm.TokenizerInfo):`. In `lmdeploy/turbomind/turbomind.py`, replace `_xgr.GrammarCompiler(tokenizer_info)` with `_tm.GrammarCompiler(tokenizer_info)`.

Every module under `lmdeploy/turbomind/builders` and `lmdeploy/turbomind/models` that currently imports `_turbomind` uses:

```python
from .. import _tm
```

The affected files are:

- `lmdeploy/turbomind/builders/_base.py`
- `lmdeploy/turbomind/builders/decoder_layer.py`
- `lmdeploy/turbomind/builders/ffn.py`
- `lmdeploy/turbomind/builders/module_list.py`
- `lmdeploy/turbomind/builders/moe.py`
- `lmdeploy/turbomind/builders/norm.py`
- `lmdeploy/turbomind/models/glm4_moe_lite.py`
- `lmdeploy/turbomind/models/internvl.py`
- `lmdeploy/turbomind/models/qwen2_vl.py`
- `lmdeploy/turbomind/models/qwen3_5.py`
- `lmdeploy/turbomind/models/utils.py`
- `lmdeploy/turbomind/models/vision_utils.py`

In `lmdeploy/turbomind/turbomind.py`, delete the complete path workaround:

```python
import sys
import lmdeploy

lmdeploy_dir = osp.split(lmdeploy.__file__)[0]
sys.path.append(osp.join(lmdeploy_dir, 'lib'))
import _turbomind as _tm
import _xgrammar as _xgr
```

Keep `import os.path as osp` because the file uses `osp.exists` independently of native-module loading.

## 4. PyTorch AWQ backend

`lmdeploy/pytorch/backends/cuda/turbomind_awq_modules.py` imports the owned module once:

```python
from lmdeploy.turbomind import _tm
```

Delete `_load_turbomind` and remove these imports, which become unused:

```python
import functools
import importlib
import sys
from pathlib import Path
from types import ModuleType

import lmdeploy
```

Do not replace `_load_turbomind` with another loader. Remove the module parameter throughout the implementation:

```python
class _LinearRuntime:
    def __init__(self, device: torch.device, stream: torch.cuda.Stream):
        self.device = device
        self.stream = stream
        self.context = _tm.create_device_context(stream.cuda_stream)
        with torch.cuda.device(device), self.context:
            self.linear = _tm.LlamaLinear()

    def forward(self, x, weight):
        with torch.cuda.device(self.device), self.context:
            input_tensor = _tm.from_dlpack_with_strides(x)
            exec_plan = self.linear.get_exec_plan(weight, input_tensor)
            if exec_plan is None:
                raise NotImplementedError('No TurboMind GEMM kernel accepts the AWQ execution problem.')
            out = torch.empty_strided(exec_plan.output_shape, exec_plan.output_stride, dtype=torch.float16, device=x.device)
            self.linear.forward_dense(exec_plan, input_tensor, weight, _tm.from_dlpack_with_strides(out))
        return out
```

```python
_runtime_pool = weakref.WeakValueDictionary()


def _get_runtime(device: torch.device, stream: torch.cuda.Stream):
    key = (device.index, stream.cuda_stream)
    runtime = _runtime_pool.get(key)
    if runtime is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Run one eager TurboMind W4A16 forward on this CUDA stream before graph capture.')
        runtime = _LinearRuntime(device, stream)
        _runtime_pool[key] = runtime
    return runtime
```

```python
def _prepare_turbomind_linear(in_features: int, out_features: int, group_size: int, qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor):
    from lmdeploy.turbomind.weight_format import AWQFormat

    weight_format = AWQFormat(block_in=group_size)
    normalized = {
        'weight': weight_format.normalize(qweight.detach(), 'weight').contiguous(),
        'scales': weight_format.normalize(scales.detach(), 'scales').contiguous(),
        'zeros': weight_format.normalize(qzeros.detach(), 'zeros').to(scales.dtype).contiguous(),
    }
    packed = {kind: weight_format.pack(tensor, kind) for kind, tensor in normalized.items()}

    stream = torch.cuda.current_stream(qweight.device)
    runtime = _get_runtime(qweight.device, stream)
    with runtime.context:
        query = _tm.WeightQuery()
        query.weight_format = weight_format.make_data_format()
        query.data_type = _tm.DataType.TYPE_FP16
        query.input_dtype = _tm.DataType.TYPE_FP16
        query.output_dtype = _tm.DataType.TYPE_FP16
        query.grouped = False
        plan = runtime.linear.get_weight_plan(query)
        if plan is None:
            raise NotImplementedError('No TurboMind GEMM family accepts the AWQ weight format.')

        config = _tm.LinearConfig()
        config.input_dim = in_features
        config.output_dim = out_features
        config.data_type = _tm.DataType.TYPE_FP16
        config.format = query.weight_format
        config.has_bias = False

        weight = _tm.LinearWeight(config)
        weight.set_plan(plan)
        for kind, item in packed.items():
            logical_shape = list(item.tensor.shape) if item.alloc_shape is None else item.alloc_shape
            logical_dtype = _tm.DataType.TYPE_FP16 if item.alloc_dtype is None else item.alloc_dtype
            destination = weight.param(kind).alloc(logical_shape, logical_dtype)
            _tm.copy_bytes_on_stream(item.tensor, destination, stream.cuda_stream)
        weight.prepare()

    stream.synchronize()
    return _PreparedLinear(weight=weight, context=runtime.context, stream=stream, runtimes={stream.cuda_stream: runtime}, device=qweight.device)
```

The two call sites become:

```python
self._prepared = _prepare_turbomind_linear(self.in_features, self.out_features, self.group_size, qweight, scales, qzeros)
```

```python
runtime = prepared.runtimes.get(stream_ptr)
if runtime is None:
    runtime = _get_runtime(prepared.device, stream)
    prepared.runtimes[stream_ptr] = runtime
out = runtime.forward(op_input, prepared.weight)
```

`lmdeploy/pytorch/backends/cuda/awq_modules.py` no longer imports or calls `_load_turbomind`. `_turbomind_support_reason` ends with the shared package capability check:

```python
from lmdeploy import turbomind

if not turbomind.is_available():
    return 'TurboMind native modules are not built'
return None
```

The builder imports the implementation only after that check:

```python
if provider in ('auto', 'turbomind'):
    reason = _turbomind_support_reason(in_features, out_features, w_bit, group_size, dtype)
    if reason is None:
        from .turbomind_awq_modules import TurbomindAwqLinearW4A16Impl

    if reason is None:
        impl_cls = TurbomindAwqLinearW4A16Impl
    elif provider == 'turbomind':
        from lmdeploy import turbomind
        raise RuntimeError(f'TurboMind W4A16 linear was requested but is unavailable or incompatible: {reason}.') from turbomind._import_error
    else:
        impl_cls = AwqLinearW4A16Impl
else:
    impl_cls = AwqLinearW4A16Impl
```

The existing shape, dtype, CUDA-availability, and architecture checks remain before the package capability check. A Python error in `turbomind_awq_modules` is not treated as native unavailability and is not silently converted into a Triton fallback.

## 5. Existing test adapters and scripts

No test or verification utility is added, and no workflow is changed solely to retain verification. All verification in this plan is transient. Existing test adapters and scripts change only where they consume the relocated native module.

`tests/turbomind/linear/test_linear.py` skips the complete module before importing any fixture or support module:

```python
from __future__ import annotations

import pytest
import torch

from lmdeploy import turbomind

if not turbomind.is_available():
    pytest.skip('TurboMind is not built', allow_module_level=True)

from lmdeploy.turbomind import _tm

from .cases import case_by_name, expand_suite
from .fixture import LinearFixture
```

This ordering is required: `fixture.py` imports `lmdeploy.turbomind.weight_format`, so applying a marker after fixture import is too late to make collection safe.

Remove `linear_mod`, `tm_required`, every `@tm_required` decorator, and the function-local `import _turbomind as tm`. The module-level skip replaces the marker, and dtype-contract references use the already imported `_tm` object.

`tests/turbomind/linear/linear.py` uses a normal hard import and removes `is_available()`:

```python
from lmdeploy.turbomind import _tm
```

The dtype-contract test uses the module-level `_tm` object instead of importing the native module again.

The optional native bridges in `tests/turbomind/moe_gate/turbomind_moe_gate.py` and `tests/turbomind/linear_attn/turbomind_gated_delta_rule.py` import once at module scope:

```python
from lmdeploy import turbomind


def _load_native_bridge(required_symbols=REQUIRED_NATIVE_BRIDGE_SYMBOLS):
    if not turbomind.is_available():
        return None
    return turbomind._tm if all(hasattr(turbomind._tm, symbol) for symbol in required_symbols) else None
```

`scripts/test_generic_copy.py` uses:

```python
from lmdeploy.turbomind import _tm
```

`tests/turbomind/linear/run_sm90_scan.py` fingerprints the imported module rather than searching for it again:

```python
from lmdeploy.turbomind import _tm


def extension_fingerprint() -> dict[str, str | int]:
    path = Path(_tm.__file__).resolve()
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
```

Remove the now-unused `importlib.util` import from that script.

In both Sphinx configurations, replace:

```python
'_turbomind',
```

with:

```python
'lmdeploy.turbomind._turbomind',
```

This applies to `docs/en/conf.py` and `docs/zh_cn/conf.py`.

## 6. Static completion checks

The repository must contain exactly one direct import of the binary module, regardless of whether the spelling is top-level, package-qualified, or package-relative:

```bash
rg -n '^\s*(from|import)\s+.*\b_turbomind\b' lmdeploy tests scripts benchmark
```

The only match must be `from . import _turbomind as _tm` in `lmdeploy/turbomind/__init__.py`. In particular, `import _turbomind`, `from _turbomind import ...`, `import lmdeploy.turbomind._turbomind`, and `from lmdeploy.turbomind import _turbomind` are forbidden.

These legacy mechanisms must have no matches in the files being changed:

```bash
rg -n '^\s*def bootstrap\(|^\s*bootstrap\(\)|_load_turbomind|CALL_FROM_SETUP_PY|_xgrammar|import_module\(.?_turbomind|find_spec\(.?_turbomind|sys\.path.*lib|^\s*import _turbomind' lmdeploy tests scripts benchmark setup.py CMakeLists.txt src/turbomind/python/CMakeLists.txt MANIFEST.in
```

Verify the Python changes without importing the extension:

```bash
PYTHONPATH=/data/lmdeploy-gemm/build/lib:/data/lmdeploy-gemm python -m compileall -q lmdeploy/turbomind lmdeploy/pytorch/backends/cuda/turbomind_awq_modules.py lmdeploy/pytorch/backends/cuda/awq_modules.py tests/turbomind/linear tests/turbomind/moe_gate tests/turbomind/linear_attn scripts/test_generic_copy.py
```

Simulate a build without either native module and verify the shared fallback boundary:

```bash
PYTHONPATH=/data/lmdeploy-gemm python - <<'PY'
import importlib.abc
import sys


class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'lmdeploy.turbomind._turbomind':
            raise ModuleNotFoundError(fullname)
        return None


sys.meta_path.insert(0, BlockNative())

from lmdeploy.archs import autoget_backend

assert autoget_backend('native-module-absence-does-not-read-a-model') == 'pytorch'
PY
```

This is a one-time verification command, not a persisted test. It checks only the public behavior: automatic backend selection falls back to PyTorch when the package-relative native module cannot be imported.

## 7. Build and module-identity verification

The existing configured tree contains two obsolete outputs from the old target layout. Remove these exact generated files once; do not add cleanup behavior to CMake:

```bash
rm -- /data/lmdeploy-gemm/build/lib/_turbomind.cpython-313-x86_64-linux-gnu.so
rm -- /data/lmdeploy-gemm/build/lib/_xgrammar.cpython-313-x86_64-linux-gnu.so
```

The relocated target does not recreate either path.

Build from the existing configured tree without setting `PYTHONPATH`:

```bash
cd /data/lmdeploy-gemm/build
ninja
```

Verify the in-tree identity:

```bash
PYTHONPATH=/data/lmdeploy-gemm/build/lib:/data/lmdeploy-gemm python - <<'PY'
from pathlib import Path

import lmdeploy.turbomind as package
from lmdeploy.turbomind import _tm
from lmdeploy.turbomind import converter, model_loader, weight_format

package_dir = Path('/data/lmdeploy-gemm/lmdeploy/turbomind').resolve()
assert Path(_tm.__file__).resolve().parent == package_dir
assert converter._tm is _tm
assert model_loader._tm is _tm
assert weight_format._tm is _tm
assert package._tm is _tm
PY
```

## 8. Existing GPU verification

Before each CUDA command, run `get_gpu_usage` and select an empty SM90 GPU. Run CUDA commands outside the sandbox. Do not set `PYTHONPATH` for compilation; use it for Python runtime commands.

Run the existing linear and MoE coverage:

```bash
CUDA_VISIBLE_DEVICES="$EMPTY_SM90_GPU" PYTHONPATH=/data/lmdeploy-gemm/build/lib:/data/lmdeploy-gemm pytest -q tests/turbomind/linear/test_linear.py
CUDA_VISIBLE_DEVICES="$EMPTY_SM90_GPU" PYTHONPATH=/data/lmdeploy-gemm/build/lib:/data/lmdeploy-gemm pytest -q tests/turbomind/moe_gate/test_moe_gate_v2.py
```

Run this transient direct AWQ comparison. It builds one canonical AWQ input, compares the TurboMind result on the default and caller-created streams bitwise, and compares TurboMind with the Triton AWQ implementation using `rtol=1e-3, atol=2e-3`:

```bash
CUDA_VISIBLE_DEVICES="$EMPTY_SM90_GPU" PYTHONPATH=/data/lmdeploy-gemm/build/lib:/data/lmdeploy-gemm python - <<'PY'
import torch

from lmdeploy.pytorch.backends.cuda.awq_modules import AwqLinearW4A16Impl
from lmdeploy.pytorch.backends.cuda.turbomind_awq_modules import TurbomindAwqLinearW4A16Impl


def pack_awq(values):
    shifts = torch.arange(0, 32, 4, dtype=torch.int64, device=values.device).view(2, 4).t().flatten()
    return torch.sum(values.unflatten(-1, (-1, 8)).to(torch.int64) << shifts, dim=-1).to(torch.int32)


torch.manual_seed(0)
device = torch.device('cuda')
k = 4096
n = 4096
group_size = 128

qweight = pack_awq(torch.randint(0, 16, (k, n), device=device))
qzeros = pack_awq(torch.randint(0, 16, (k // group_size, n), device=device))
scales = (torch.rand((k // group_size, n), device=device) * 0.02 + 0.001).to(torch.float16)
x = (torch.randn((1, 17, k), device=device) * 0.1).to(torch.float16)

triton_impl = AwqLinearW4A16Impl(k, n, 4, group_size)
tm_impl = TurbomindAwqLinearW4A16Impl(k, n, 4, group_size)
tm_impl.update_weights(qweight, scales, qzeros)

triton_output = triton_impl.forward(x, qweight, scales, qzeros)
tm_default_output = tm_impl.forward(x, qweight, scales, qzeros)

default_stream = torch.cuda.current_stream(device)
caller_stream = torch.cuda.Stream(device=device)
caller_stream.wait_stream(default_stream)
with torch.cuda.stream(caller_stream):
    tm_caller_stream_output = tm_impl.forward(x, qweight, scales, qzeros)
caller_stream.synchronize()

assert torch.equal(tm_default_output, tm_caller_stream_output)
max_abs_diff = (tm_default_output - triton_output).abs().max().item()
torch.testing.assert_close(tm_default_output, triton_output, rtol=1e-3, atol=2e-3)
print(f'max absolute difference: {max_abs_diff}')
PY
```

Do not add this comparison to a test file or helper script.

Finally run the required TurboMind model smoke outside the sandbox on the selected empty SM90 GPU:

```bash
PYTHONPATH=/data/lmdeploy-gemm/build/lib:/data/lmdeploy-gemm \
python scripts/test_turbomind_model.py \
    --model-id Qwen/Qwen3-8B \
    --cache-dir /mnt_cfs/huggingface_hub/hub/ \
    --gpus "$EMPTY_SM90_GPU" \
    --tp 1 \
    --max-new-tokens 128 \
    --prompt "Explain how matrix multiplication is used in transformer language models, with concrete examples and enough detail to continue for the full response."
```

Read the generated response and require meaningful human language relevant to the prompt; an exit code alone is not sufficient.

## Completion criteria

- The native extension module lives directly in `lmdeploy/turbomind`; its private runtime library lives in `lmdeploy/turbomind/lib`.
- No Python code mutates `sys.path` to find TurboMind binaries.
- No native-module loader, bootstrap function, origin check, or module cache exists in LMDeploy.
- `import lmdeploy.turbomind` and every PyTorch-only entry point remain usable when the native modules are absent.
- Every production consumer, optional backend, test adapter, and script receives the single module object owned by `lmdeploy.turbomind`.
- In-tree build, existing CUDA tests, AWQ stream comparison, and the model smoke all pass.
