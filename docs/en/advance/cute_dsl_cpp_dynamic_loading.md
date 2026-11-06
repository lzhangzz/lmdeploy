# Calling CuTe DSL kernels from C++ through dynamic loading

## Findings

Verified on November 5, 2026: a standalone C++ executable can dynamically load and execute an ahead-of-time (AOT) compiled CuTe DSL RMSNorm kernel from both object files (`.o`) and shared libraries (`.so`).

- Python is required for kernel compilation and export, not for execution in C++.
- The executable links against the CuTe DSL runtime and CUDA libraries, not the RMSNorm kernel objects or shared library.
- The exported tensor descriptors support runtime batch sizes, hidden widths, and row strides when those values are declared dynamic during compilation.
- An existing non-default CUDA stream can be passed to the exported function.
- Both loading methods passed 243 correctness cases each. Compute Sanitizer reported zero memory errors for object-file loading.

This is a standalone interoperability verification, not a TurboMind integration or a performance benchmark.

## Verified environment

| Component | Version / configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 5080 |
| Device code target | `sm_120` |
| CuTe DSL | `nvidia-cutlass-dsl` 4.8.1, CUDA 13 variant |
| CuTe DSL CUDA version | 13.4 |
| CUDA runtime used for final verification | 13.4.92 |
| NVIDIA driver | 615.71.09 |
| Host | Linux x86_64 |
| Python used for export | 3.13 |
| C++ build | CMake Release build, GCC 15, C++17 |

The system CUDA toolkit was 13.3. Initial runs passed against that runtime, but final verification used isolated CUDA 13.4 headers and runtime libraries to match CuTe DSL. Passing with 13.3 is an observation, not a cross-version compatibility guarantee.

Dependencies were installed under `build/`; no project dependency manifests or global Python packages were changed.

## Verification sources and artifacts

- [Python kernel exporter](../../../scripts/test_cute_dsl_rmsnorm.py)
- [C++ runtime loader and correctness checks](../../../scripts/test_cute_dsl_rmsnorm.cpp)
- [Standalone CMake project](../../../scripts/CMakeLists.txt)

Python only exports these artifacts under `build/cute_dsl_rmsnorm/`:

```text
rmsnorm_f32.h / rmsnorm_f32.o
rmsnorm_f16.h / rmsnorm_f16.o
rmsnorm_bf16.h / rmsnorm_bf16.o
```

CMake builds `bin/test_cute_dsl_rmsnorm` and links `lib/libcute_dsl_rmsnorm_kernels.so` under its own build directory. Each dtype has its own exported function; the shared library combines all three objects. The executable includes the generated headers for descriptor definitions but does not call their static-linking wrappers or link the kernel objects. CTest verifies both object-file and shared-library loading.

## Kernel and ABI

The operation is weighted RMSNorm, without a residual addition or weight offset:

```text
y[row, col] = x[row, col] * weight[col]
              / sqrt(sum(x[row, :] ** 2) / width + eps)
```

The example uses one 32-thread warp per row, FP32 accumulation, and a warp shuffle reduction. FP16 and BF16 inputs and weights are converted to FP32 for arithmetic; the final result is converted to the output dtype.

Compilation uses fake tensors, so exporting does not allocate sample input tensors:

```python
x = make_fake_tensor(
    dtype,
    (cute.sym_int32(), cute.sym_int32()),
    (cute.sym_int64(), 1),
    assumed_align=16,
)
weight = make_fake_tensor(dtype, (cute.sym_int32(),), (1,), assumed_align=16)
y = make_fake_tensor(
    dtype,
    (cute.sym_int32(), cute.sym_int32()),
    (cute.sym_int64(), 1),
    assumed_align=16,
)
compiled = cute.compile(
    rmsnorm, x, weight, y, cutlass.Float32(1e-6), cuda.CUstream(0),
    options='--gpu-arch sm_120',
)
compiled.export_to_c(
    file_path='./build/cute_dsl_rmsnorm',
    file_name='rmsnorm_f32',
    function_prefix='rmsnorm_f32',
)
```

Here, `dtype` is `cutlass.Float32` for the FP32 export. The driver also exports FP16 and BF16 variants.

The resulting input/output descriptors contain a device pointer, two dynamic `int32_t` shapes, and one dynamic `int64_t` row stride. The weight descriptor contains a device pointer and one dynamic `int32_t` shape. Inner strides are fixed to one. Strides are measured in elements, not bytes.

Always use the matching generated header: changing static/dynamic dimensions, stride widths, or argument types can change the ABI. The caller must supply compatible shapes and valid buffers; these tests do not establish runtime shape validation.

## Actual C++ loading API

The online AOT guide showed `CuteDSLRT_Module_Load`, but **the installed CuTe DSL 4.8.1 `CuteDSLRuntime.h` did not expose that function**. The verified entry point is `CuteDSLRT_Module_Create_From_Bytes`.

For an object file, read its bytes and pass them to the runtime:

```cpp
CuteDSLRT_Module_t* module = nullptr;
check_dsl(CuteDSLRT_Module_Create_From_Bytes(
    &module, bytes.data(), bytes.size(), nullptr, 0));
```

The runtime header states that the bytes are no longer needed after module creation. The verification loader releases the local byte vector before invoking the function.

For a shared library, pass its path through the same API:

```cpp
const char* libraries[] = {shared_library_path};
CuteDSLRT_Module_t* module = nullptr;
check_dsl(CuteDSLRT_Module_Create_From_Bytes(
    &module, nullptr, 0, libraries, 1));
```

Resolve the exported function by its `function_prefix`, not by its generated internal MLIR symbol:

```cpp
CuteDSLRT_Function_t* function = nullptr;
check_dsl(CuteDSLRT_Module_Get_Function(&function, module, "rmsnorm_f32"));
```

Populate the generated tensor descriptors, then invoke using the argument order shown by the generated wrapper:

```cpp
int32_t launch_status = -1;
void* args[] = {&x, &weight, &y, &eps, &stream, &launch_status};
CuteDSLRT_Error_t runtime_status = CuteDSLRT_Function_Run(function, args, 6);
check_dsl(runtime_status);
check_cuda(static_cast<cudaError_t>(launch_status));
```

There are two statuses: the CuTe DSL runtime status and the trailing CUDA launch status. Execution is asynchronous; check stream completion separately for execution faults. The verification synchronizes after copying results back to the CPU.

Create the module with the intended CUDA device/context active. Cache module and function handles across launches, and finish outstanding GPU work before calling `CuteDSLRT_Module_Destroy`. This test uses device 0 and does not verify multi-GPU handling.

## Correctness results

The C++ reference computes the sum of squares and normalization in double precision from the dtype-rounded inputs and weights, then rounds the result to the output dtype. It does not depend on PyTorch.

The acceptance criterion is:

```text
abs(actual - expected) <= atol + rtol * abs(expected)
```

| Dtype | Cases per loading method | Maximum absolute error | atol | rtol |
|---|---:|---:|---:|---:|
| FP32 | 81 passed | 4.76837e-7 | 2e-5 | 2e-5 |
| FP16 | 81 passed | 0.00195312 | 2e-3 | 2e-3 |
| BF16 | 81 passed | 0.015625 | 2e-2 | 2e-2 |

Each dtype runs nine `(rows, width)` shapes:

```text
(1, 1), (3, 7), (2, 31), (4, 32), (7, 33),
(16, 128), (5, 1000), (8, 4096), (2, 8192)
```

For each shape, the test checks:

- Zero inputs with contiguous rows.
- Random inputs with padded input/output row strides.
- Small random inputs, scaled by `1e-4`, with contiguous rows.
- Runtime epsilon values `1e-6`, `1e-5`, and `1e-3`.
- Repeated invocation of the same resolved function on a non-blocking, non-default CUDA stream.
- Finite outputs, untouched output padding, and unchanged inputs and weights.

This gives `9 shapes * 3 input patterns * 3 epsilons = 81` cases per dtype: 243 through `.o` loading and 243 through `.so` loading, for 486 total checks.

Compute Sanitizer `memcheck` passed all 243 object-file cases with `ERROR SUMMARY: 0 errors`. The Python source also passed the project's Ruff rules using Ruff 0.15.4.

## Reproduction

Run from the repository root. The isolated dependencies from this verification remain under `build/`. To recreate them in fresh target directories:

```bash
python -m pip install --target build/cute_dsl_verify_deps \
  'nvidia-cutlass-dsl[cu13]==4.8.1'

python -m pip install --target build/cute_dsl_verify_cuda \
  'nvidia-cuda-runtime==13.4.92' \
  'nvidia-cuda-nvcc==13.4.92' \
  'nvidia-cuda-crt==13.4.92' \
  'nvidia-nvvm==13.4.92' \
  'nvidia-cuda-cccl==13.3.4.3.1' \
  'ruff==0.15.4'
```

The runtime wheel alone lacks `crt/host_config.h`, so the additional CUDA header packages are required for the C++ build. CUDA 13.4 component versions are pinned explicitly because the NVCC wheel's dependencies otherwise selected newer CRT/NVVM versions during installation.

First export the kernels; Python does not build or run the C++ application:

```bash
PYTHONPATH="$PWD/build/cute_dsl_verify_deps:$PWD/build/cute_dsl_verify_deps/nvidia_cutlass_dsl/dsl_packages" \
CUTE_DSL_USE_CUDA_13=1 \
python scripts/test_cute_dsl_rmsnorm.py
```

Both Python paths are needed with `pip --target`: adding the target to `PYTHONPATH` alone does not process the wheel's `.pth` file that adds `dsl_packages`.

Configure and build the standalone CMake project, then verify both loading methods with CTest:

```bash
cmake -S scripts -B build/cute_dsl_rmsnorm_cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-15 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-15 \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.3 \
  -DENABLE_TILELANG=OFF -DENABLE_TRITON=OFF \
  -DCUTE_DSL_ROOT="$PWD/build/cute_dsl_verify_deps/nvidia_cutlass_dsl/cu13" \
  -DCUTE_DSL_CUDA_ROOT="$PWD/build/cute_dsl_verify_cuda/nvidia/cu13"
cmake --build build/cute_dsl_rmsnorm_cmake --parallel 4
ctest --test-dir build/cute_dsl_rmsnorm_cmake --output-on-failure
```

CMake discovers native libraries from the supplied roots and does not invoke Python. `CUTE_DSL_CUDA_ROOT` selects the matching CUDA 13.4 headers/runtime for CuTe, independently of the system toolkit used by the other applications. CTest removes `PYTHONPATH` and prepends the selected native library directories so an existing `LD_LIBRARY_PATH` cannot silently select the system CUDA 13.3 runtime.

To run a built executable directly or under Compute Sanitizer, use the matching library path explicitly:

```bash
export LD_LIBRARY_PATH="$PWD/build/cute_dsl_verify_deps/nvidia_cutlass_dsl/cu13/lib:$PWD/build/cute_dsl_verify_cuda/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

env -u PYTHONPATH build/cute_dsl_rmsnorm_cmake/bin/test_cute_dsl_rmsnorm \
  "$PWD/build/cute_dsl_rmsnorm"

env -u PYTHONPATH build/cute_dsl_rmsnorm_cmake/bin/test_cute_dsl_rmsnorm \
  "$PWD/build/cute_dsl_rmsnorm" \
  "$PWD/build/cute_dsl_rmsnorm_cmake/lib/libcute_dsl_rmsnorm_kernels.so"

env -u PYTHONPATH compute-sanitizer --tool memcheck --error-exitcode 1 \
  build/cute_dsl_rmsnorm_cmake/bin/test_cute_dsl_rmsnorm \
  "$PWD/build/cute_dsl_rmsnorm"
```

With that environment, `ldd build/cute_dsl_rmsnorm_cmake/bin/test_cute_dsl_rmsnorm` should resolve `libcudart.so.13` to the isolated CUDA 13.4 directory, not `/usr/local/cuda-13.3/lib64`. The same CMake project supports all three DSLs together by enabling their targets and supplying their native library roots.

## Integration implications and limits

A TurboMind adapter could load artifacts during engine initialization, retain function handles, construct descriptors from existing device pointers and strides, and invoke on the engine's existing CUDA stream. Kernel artifacts would remain runtime inputs rather than link-time dependencies of the backend.

However:

- Deployment still needs the matching CuTe DSL runtime and CUDA libraries. No Python interpreter is needed for the C++ path.
- Artifacts must target the deployment GPU architecture, and the generated headers must match those artifacts.
- This example supports contiguous inner dimensions, dynamic row strides, positive non-empty shapes, and the declared pointer alignment. It does not establish support for arbitrary layouts or zero-size inputs.
- The one-warp-per-row kernel is a correctness example, not a tuned replacement for TurboMind's RMSNorm.
- Latency, throughput, CUDA Graph capture, concurrent host calls, multi-GPU behavior, and full inference integration were not verified.
- The test covers successful calls; it does not exercise all runtime failure/error-reporting paths.

## References

- [NVIDIA AOT compilation guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/ahead_of_time_compilation.html)
- [NVIDIA C++ dynamic-loading example](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/dsl_tutorials/export/run_with_dynamic_loading.cpp)
- [NVIDIA CuTe DSL installation guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html)

The online documentation and examples can change. For the reproduced 4.8.1 behavior, the installed `CuteDSLRuntime.h` and the locally generated kernel headers are the authoritative API/ABI references.
