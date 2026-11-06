# Calling TileLang kernels from C++ through dynamic loading

## Findings

Verified on November 5, 2026: TileLang RMSNorm kernels can be exported as shared libraries (`.so`) and loaded and executed by a standalone C++ executable without Python at runtime.

The verified path is:

```text
Python TileLang PrimFunc
    -> compile with execution_backend="tvm_ffi"
    -> export_library(kernel.so)
    -> C++ loads TVM runtime support
    -> ffi::Module::LoadFromFile(kernel.so)
    -> GetFunction(exported_symbol)
    -> invoke using TensorView arguments
```

- All 243 RMSNorm correctness cases passed across FP32, FP16, and BF16.
- Nine CUDA Graph capture/replay checks passed on the caller's non-default CUDA stream.
- Compute Sanitizer reported zero memory errors.
- The executable links no RMSNorm kernel libraries and does not invoke a Python interpreter.
- **TVM FFI alone was insufficient for the tested exports.** The installed TileLang wheel's `libtvm.so` must also be loaded to provide CUDA module loading and runtime support.

This is a standalone interoperability verification, not a TurboMind integration or a performance benchmark. See also the [CuTe DSL verification](cute_dsl_cpp_dynamic_loading.md).

## Verified environment

| Component | Version / configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 5080 |
| TileLang | 0.1.9 |
| Apache TVM FFI | 0.1.9 |
| CUDA toolkit and executable's CUDA runtime | 13.3 |
| NVIDIA driver | 615.71.09 |
| Host | Linux x86_64 |
| Python used for export | 3.13 |
| NVCC host compiler and C++ caller compiler | GCC 15 |
| C++ caller build | CMake Release build, C++17 |
| Requested TileLang target | `cuda -arch=sm_120` |

For this target, the observed TileLang NVCC command used `-arch=sm_120a`. Artifacts should be rebuilt for the deployment GPU architecture rather than assuming portability to other GPUs.

TileLang and TVM FFI were already installed and imported successfully. No dependency installation, global package modification, or project dependency manifest change was needed.

## Sources and artifacts

- [Python RMSNorm kernel exporter](../../../scripts/test_tilelang_rmsnorm.py)
- [C++ loader, reference checks, and stream verification](../../../scripts/test_tilelang_rmsnorm.cpp)
- [Standalone CMake project](../../../scripts/CMakeLists.txt)

Python only exports artifacts under `build/tilelang_rmsnorm/`:

```text
rmsnorm_{f32,f16,bf16}_h{33,128,4096}.so
rmsnorm_{f32,f16,bf16}_h{33,128,4096}.cu
rmsnorm_{f32,f16,bf16}_h{33,128,4096}_host.cc
```

There are nine exported shared libraries: one per dtype and hidden-width specialization. TileLang's native `export_library()` operation creates those libraries; the saved CUDA and host sources are for inspection only. CMake builds `bin/test_tilelang_rmsnorm` under its own build directory. The kernel libraries remain runtime inputs, not link-time dependencies, and CTest runs the verification application.

## Kernel and export ABI

The kernel implements weighted RMSNorm without residual addition or a weight offset:

```text
y[row, col] = x[row, col] * weight[col]
              / sqrt(sum(x[row, :] ** 2) / width + eps)
```

It uses one 128-thread CUDA block per row, FP32 fragments and reduction, and a fragment width rounded up to a power of two. Columns beyond the actual hidden width are masked. The result is converted to the output dtype.

The verification distinguishes compile-time and runtime values:

- **Compile-time:** dtype, hidden width, block configuration, and GPU target.
- **Runtime:** batch size, input/output row strides, tensor pointers, and epsilon.
- Inner strides and the weight stride are fixed to one.

The Python kernel uses `T.StridedTensor` for input/output and `T.dynamic` for batch size and row strides. Each function has an explicit `global_symbol`, such as `rmsnorm_f32_h33`.

Export uses the native TVM FFI backend with an explicit output buffer:

```python
kernel = tilelang.compile(
    make_rmsnorm(width, dtype, symbol),
    out_idx=[],
    execution_backend='tvm_ffi',
    target=tvm.target.Target('cuda -arch=sm_120'),
    compile_flags=['-ccbin=/usr/bin/g++-15'],
)
kernel.export_library(f'./build/tilelang_rmsnorm/{symbol}.so')
```

`out_idx=[]` avoids automatic output allocation. The inspected generated host function accepts exactly four positional arguments:

```text
x, weight, y, eps
```

Batch size and row strides are read from the DLPack descriptors; they are not extra positional arguments. The CUDA stream is supplied through the TVM FFI environment, not as a fifth argument.

The driver disables TileLang's kernel cache to compile and export fresh runtime modules. Exporting cached artifacts was not verified.

## C++ loading and invocation

The tested wheel contains `libtvm.so` but no separate `libtvm_runtime.so`. The executable first loads that library with global symbol visibility:

```cpp
if (!dlopen(runtime_library_path, RTLD_NOW | RTLD_GLOBAL)) {
    throw std::runtime_error(dlerror());
}
```

The runtime library remains loaded for the lifetime of the test process. Keep runtime support resident while modules, functions, and registered runtime callbacks can refer to it.

Load a kernel module and resolve its exported symbol:

```cpp
namespace ffi = tvm::ffi;

ffi::Module module = ffi::Module::LoadFromFile(kernel_library_path);
ffi::Function function = module->GetFunction("rmsnorm_f32_h33").value();
```

The C++ caller allocates GPU buffers, constructs `DLTensor` descriptors, and passes non-owning tensor views:

```cpp
function(ffi::TensorView(&x),
         ffi::TensorView(&weight),
         ffi::TensorView(&y),
         eps);
```

The descriptors carry the device, dtype, dimensions, shapes, and element strides. The test supplies explicit stride arrays even for contiguous tensors and sets byte offsets to zero. No tensor copy or ownership transfer is needed to wrap existing device memory. The descriptors and their shape/stride storage must remain valid during invocation, and device buffers must remain alive until GPU execution completes.

## CUDA stream verification

The caller creates a non-blocking, non-default CUDA stream. Its `StreamScope` saves the previous TVM FFI stream and restores it after invocation:

```cpp
TVMFFIEnvSetStream(kDLCUDA, device_id, stream, &previous_stream);
// Invoke the function.
TVMFFIEnvSetStream(kDLCUDA, device_id, previous_stream, nullptr);
```

The source checks errors when setting the stream and checks that the previous stream context is restored after successful calls. The scope also attempts restoration during exception unwinding; exception-path behavior was not separately exercised.

To verify actual launch placement rather than merely setting thread-local state, each dtype/width specialization is tested after warmup using CUDA Graph capture on the selected stream:

1. Run the kernel normally and synchronize.
2. Reset the output buffer to a sentinel.
3. Begin capture on the caller's stream and invoke the exported function.
4. Confirm the captured graph has exactly one node and that it is a CUDA kernel node.
5. Instantiate and replay the graph on that stream.
6. Compare the replay result with the CPU reference and check output padding.

All nine capture/replay checks passed. These checks cover warmed-up RMSNorm kernels, not first-use capture or arbitrary TileLang kernels.

## Correctness results

The C++ reference accumulates squares and computes normalization in double precision using dtype-rounded inputs and weights, then rounds the expected output to the output dtype. It does not use PyTorch.

The acceptance criterion is:

```text
abs(actual - expected) <= atol + rtol * abs(expected)
```

| Dtype | Correctness cases | Maximum absolute error | atol | rtol | Graph replays |
|---|---:|---:|---:|---:|---:|
| FP32 | 81 passed | 4.76837e-7 | 2e-5 | 2e-5 | 3 passed |
| FP16 | 81 passed | 0.00195312 | 2e-3 | 2e-3 | 3 passed |
| BF16 | 81 passed | 0.0078125 | 2e-2 | 2e-2 | 3 passed |

For each dtype, the test covers:

- Hidden-width specializations: 33, 128, and 4096.
- Runtime batch sizes: 1, 3, and 17.
- Zero contiguous inputs, random padded-row inputs, and small contiguous random inputs scaled by `1e-4`.
- Runtime epsilon values: `1e-6`, `1e-5`, and `1e-3`.
- Repeated calls through the same resolved function handle.
- Finite outputs, untouched output padding, and unchanged inputs and weights.

This gives `3 widths * 3 batches * 3 input patterns * 3 epsilons = 81` cases per dtype, or 243 total. The nine graph replays occur within these cases, with their replayed outputs checked against the same reference.

The full C++ test also passed Compute Sanitizer `memcheck` with `ERROR SUMMARY: 0 errors`. The Python driver passed the project's Ruff rules using Ruff 0.15.4.

## Confirmed runtime dependency requirement

Running the same executable without loading `libtvm.so` failed with:

```text
Library binary was created using {cuda} but a loader of that name is not registered.
Make sure to have runtime that registers ffi.Module.load_from_bytes.cuda
```

Supplying the installed wheel's `libtvm.so` made the exported modules load and all tests pass. This establishes that `libtvm_ffi.so` alone is insufficient for these artifacts.

Library inspection showed:

- The C++ executable links `libtvm_ffi.so`, CUDA runtime/driver libraries, and standard host libraries. It links no RMSNorm shared libraries.
- The exported RMSNorm `.so` files have only standard host libraries in their ELF dependency lists. Runtime support is still required: those dependency lists do not capture the CUDA module-loader registration requirement.
- The wheel's `libtvm.so` brings `libtvm_ffi.so`, `libz3.so`, and the wheel's `libcudart_stub.so`, `libnvrtc_stub.so`, and `libcuda_stub.so`, plus standard host libraries.
- No Python library appears in the inspected dependency lists of the caller, exported kernels, or `libtvm.so`.

The resulting deployment needs the native runtime libraries and their dependencies, but not a Python interpreter. A smaller runtime-only build might reduce packaging overhead; that alternative was not built or verified here.

## Reproduction

Run from the repository root with the verified TileLang 0.1.9 and TVM FFI 0.1.9 environment. First export all nine kernels:

```bash
python scripts/test_tilelang_rmsnorm.py
```

Python only performs kernel export. TileLang's native exporter still creates the kernel shared libraries internally; there are no Python compiler commands for the C++ application.

Configure and build the application with CMake, then run it through CTest. The native package paths below are those used in the verification; adjust them for a different environment:

```bash
cmake -S scripts -B build/tilelang_rmsnorm_cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-15 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-15 \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.3 \
  -DENABLE_CUTE_DSL=OFF -DENABLE_TRITON=OFF \
  -DTVM_FFI_ROOT=/home/li/miniconda3/lib/python3.13/site-packages/tvm_ffi \
  -DTILELANG_ROOT=/home/li/miniconda3/lib/python3.13/site-packages/tilelang
cmake --build build/tilelang_rmsnorm_cmake --parallel 4
ctest --test-dir build/tilelang_rmsnorm_cmake --output-on-failure
```

CMake discovers the native libraries without invoking Python. CTest passes the wheel's `libtvm.so` path, removes `PYTHONPATH`, and sets native library search paths.

The system's default compiler was GCC 16, which NVCC rejected because CUDA 13.3 supports GCC versions only through 15. The exporter selects installed `g++-15` for GPU kernel compilation when available, or accepts `--host-compiler`. The C++ application compiler is selected independently through CMake. Neither uses `-allow-unsupported-compiler`.

To run a built executable or sanitizer directly:

```bash
TVM_LIBRARY=/home/li/miniconda3/lib/python3.13/site-packages/tilelang/lib/libtvm.so

env -u PYTHONPATH build/tilelang_rmsnorm_cmake/bin/test_tilelang_rmsnorm \
  "$PWD/build/tilelang_rmsnorm" "$TVM_LIBRARY"

env -u PYTHONPATH compute-sanitizer --tool memcheck --error-exitcode 1 \
  build/tilelang_rmsnorm_cmake/bin/test_tilelang_rmsnorm \
  "$PWD/build/tilelang_rmsnorm" "$TVM_LIBRARY"
```

The executable contains run paths for the tested TVM FFI and CUDA library directories. A relocated deployment must provide suitable library search paths and the transitive dependencies of `libtvm.so`.

Omitting the runtime argument reproduces the expected missing-loader failure:

```bash
env -u PYTHONPATH build/tilelang_rmsnorm_cmake/bin/test_tilelang_rmsnorm \
  "$PWD/build/tilelang_rmsnorm"
```

## Integration implications and limits

A TurboMind adapter could load runtime support once, load the chosen kernel specializations during initialization, retain module/function handles, wrap existing buffers as DLPack tensor views, and invoke within a stream scope using the engine's CUDA stream. Output allocation stays with TurboMind.

Relative to the verified CuTe DSL path, this TileLang wheel has heavier runtime packaging because CUDA module loading requires the larger `libtvm.so` and its dependencies. Do not substitute a different TVM/TVM FFI build without checking compatibility.

Not established by this verification:

- Performance relative to TurboMind's existing RMSNorm.
- Full inference-engine integration, multi-GPU behavior, or concurrent host calls.
- Runtime-dynamic hidden width, arbitrary inner strides, or zero-size inputs.
- General CUDA Graph compatibility beyond the nine warmed-up RMSNorm specializations.
- Comprehensive invalid-input or exception-path testing.
- Minimal runtime-only deployment or newer TileLang/TVM FFI versions.

## References

- [TileLang export implementation](https://github.com/tile-ai/tilelang/blob/main/tilelang/jit/kernel.py)
- [TVM FFI C++ loading quickstart](https://tvm.apache.org/ffi/get_started/quickstart.html)
- [TVM FFI stream environment API](https://github.com/apache/tvm-ffi/blob/main/include/tvm/ffi/extra/c_env_api.h)

Online sources can change. The findings above refer to the installed 0.1.9 packages, the locally exported artifacts, and the standalone tests in this repository.
