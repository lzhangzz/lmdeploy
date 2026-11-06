# Calling Triton kernels from C++ through dynamic loading

## Findings

Verified on November 5, 2026: Triton CUDA kernels can be ahead-of-time (AOT) compiled into generated C launchers with embedded cubin data, built into a shared library, and loaded from standalone C++ using `dlopen()` and `dlsym()`.

The passing RMSNorm verification used an explicit **FP64 epsilon argument in the exported ABI**, narrowed to FP32 inside the kernel. The original FP32 epsilon ABI failed because of a confirmed scalar-marshalling issue in the installed Triton 3.7.0 AOT exporter.

- All 729 correctness cases passed across FP32, FP16, and BF16 tensor types.
- All 27 CUDA Graph capture/replay checks passed on the caller's non-default stream.
- Compute Sanitizer reported zero memory errors.
- The kernel shared library's direct ELF dependencies are only `libcuda.so.1` and libc.
- No Python interpreter, separate Triton runtime library, or TVM runtime is needed for execution.
- No installed package or generated C launcher was patched.

This verifies the FP64-epsilon loading path, not the broken FP32 scalar ABI. Performance and full TurboMind integration were not verified.

Related findings: [CuTe DSL](cute_dsl_cpp_dynamic_loading.md) and [TileLang](tilelang_cpp_dynamic_loading.md).

## Verified environment

| Component | Version / configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 5080 |
| Triton | 3.7.0 |
| Reported Triton target | `GPUTarget(backend='cuda', arch=120, warp_size=32)` |
| Observed generated PTX target | `sm_120a` |
| CUDA toolkit and caller's runtime libraries | 13.3 |
| NVIDIA driver | 615.71.09 |
| Host | Linux x86_64 |
| Python used for export | 3.13 |
| Generated C launcher build | CMake Release build, GCC 15, C11, shared library |
| C++ caller build | CMake Release build, GCC 15, C++17 |

Triton was already installed. No dependency installation or project dependency manifest change was needed. Triton performs GPU compilation through its own compiler/backend. CUDA header/library discovery and native compiler selection are handled by CMake, not Python.

## Sources and artifacts

- [Python kernel and AOT exporter](../../../scripts/test_triton_rmsnorm.py)
- [C++ shared-library loader and correctness checks](../../../scripts/test_triton_rmsnorm.cpp)
- [Standalone CMake project](../../../scripts/CMakeLists.txt)

Python only exports one kernel per tensor dtype, producing these files under `build/triton_rmsnorm/`:

```text
rmsnorm_{f32,f16,bf16}.<signature-hash>_<specialization-suffix>.c
rmsnorm_{f32,f16,bf16}.<signature-hash>_<specialization-suffix>.h
rmsnorm_symbols.h
rmsnorm_sources.cmake
```

The C files embed the cubin data. A separate cubin file is not required at runtime. The source manifest lists exactly the current exports, avoiding accidental compilation of stale specializations. CMake compiles those sources into `lib/libtriton_rmsnorm_kernels.so` and builds `bin/test_triton_rmsnorm` under its own build directory. The executable uses the symbol-name header but does not link the kernel shared library; CTest passes its path at execution time.

## Kernel and dynamic arguments

The operation is weighted RMSNorm, without residual addition or a weight offset:

```text
y[row, col] = x[row, col] * weight[col]
              / sqrt(sum(x[row, :] ** 2) / width + float32(eps))
```

One Triton program processes one row. The compiled reduction block has 8192 elements, with columns outside the runtime width masked to zero. Input values and weights are converted to FP32, reduction and normalization use FP32, and the result is stored in the output dtype.

Compile-time configuration:

- Tensor dtype: FP32, FP16, or BF16, with one export per dtype.
- Maximum block width: 8192.
- Four warps and one stage.
- Pointer base-alignment hints of 16 bytes.
- The selected GPU architecture.

Runtime arguments:

```text
x, weight, y, rows, cols, x_stride, y_stride, eps
```

Batch size, hidden width, row strides, and epsilon are not compile-time constants. Inner strides and the weight stride are one; row strides are measured in elements. Only positive widths up to 8192 were tested.

The passing AOT signature is:

```text
*<dtype>:16, *<dtype>:16, *<dtype>:16,
i32, i32, i64, i64, fp64, 8192
```

The final `8192` is a constexpr and does not appear in the launcher ABI. The grid is exported as `rows,1,1`, so grid dimensions are computed by the launcher rather than supplied as extra C++ arguments.

## AOT export and C++ invocation

The driver uses the installed native AOT compiler:

```python
symbol, files = compile_kernel(CompileArgs(
    path=str(source_path),
    kernel_name='rmsnorm_kernel',
    signature=signature,
    grid='rows,1,1',
    num_warps=4,
    num_stages=1,
    out_name='rmsnorm_f32',
    out_path=build_dir / 'rmsnorm_f32',
))
```

CMake compiles the generated C files without modifying them, linking against the CUDA driver. Python only exports those files and their source manifest. The generated header defines the exact launch signature. For the passing export, it is equivalent to:

```cpp
using Launch = CUresult (*)(
    CUstream stream,
    CUdeviceptr x,
    CUdeviceptr weight,
    CUdeviceptr y,
    int32_t rows,
    int32_t cols,
    int64_t x_stride,
    int64_t y_stride,
    double eps);
```

Each generated symbol has a signature hash and specialization suffix. The exporter returns the actual name; the driver records it in `rmsnorm_symbols.h`. Do not guess symbol names or assume the launch ABI remains unchanged across Triton versions.

The standalone C++ caller:

1. Selects CUDA device 0 and initializes its context.
2. Opens the shared library with `dlopen(path, RTLD_NOW | RTLD_LOCAL)`.
3. Resolves the launch function and its generated `load_<symbol>` and `unload_<symbol>` functions with `dlsym()`.
4. Explicitly loads each CUDA module before launching kernels.
5. Passes existing device pointers and an explicit CUDA stream to the launch function.
6. Checks launch status and stream completion.
7. Unloads modules after outstanding work completes, then closes the shared library.

The generated launcher uses `cuModuleLoadData`, `cuModuleGetFunction`, `cuLaunchKernel`, and `cuModuleUnload`. No DSL-specific loader or tensor descriptor runtime is involved.

## Confirmed FP32 scalar ABI failure

The initial export specified epsilon as `fp32`. The generated header and C launcher nevertheless declared it as `double`, because the installed NVIDIA backend's `map_python_to_cpp_type` maps `fp32` to `double`.

The generated launcher passes the address of that variable directly:

```cpp
// Excerpt of the failing generated launcher.
CUresult generated_launch(/* other arguments */, double eps) {
    void* args[] = { /* other pointers */, &eps, /* scratch pointers */ };
    return cuLaunchKernel(/* ... */, args, nullptr);
}
```

Inspection of the corresponding PTX showed that epsilon is device parameter 7 with type `.f32`. The driver therefore consumes four bytes from the host `double` storage rather than a converted FP32 value. This is not a C++ caller signature mismatch: the caller followed the generated header's `double` ABI.

The first correctness check failed:

```text
FAIL: FP32 mismatch at 1x1, row 0, col 0: got nan, expected -0.000000
```

The passing variant explicitly exports epsilon as `fp64`, matching the generated C `double` parameter and the device parameter size. The kernel then calls `eps.to(tl.float32)` before normalization. Runtime epsilon remains supported; its arithmetic precision is intentionally FP32.

The test epsilon constants are FP32 values promoted to C++ `double`, so the CPU reference and the kernel's narrowing use the same epsilon value.

This is a version-specific compatibility choice, not a fix to Triton's exporter. The failing FP32 variant remains reproducible with `--epsilon-type fp32`. The findings do not establish that other floating-point scalar types marshal correctly through this exporter.

## Correctness and stream results

The C++ reference computes the sum of squares and normalization in double precision from dtype-rounded inputs and weights, then rounds the expected result to the output dtype. It does not use PyTorch.

Acceptance criterion:

```text
abs(actual - expected) <= atol + rtol * abs(expected)
```

| Tensor dtype | Correctness cases | Maximum absolute error | atol | rtol | Graph replays |
|---|---:|---:|---:|---:|---:|
| FP32 | 243 passed | 4.76837e-7 | 2e-5 | 2e-5 | 9 passed |
| FP16 | 243 passed | 0.00195312 | 2e-3 | 2e-3 | 9 passed |
| BF16 | 243 passed | 0.015625 | 2e-2 | 2e-2 | 9 passed |

For each dtype, the same compiled kernel handles:

- Runtime widths: 1, 7, 31, 32, 33, 128, 1000, 4096, and 8192.
- Runtime batches: 1, 3, and 17.
- Zero contiguous inputs, random padded-row inputs, and small contiguous random inputs scaled by `1e-4`.
- Runtime epsilon values: `1e-6f`, `1e-5f`, and `1e-3f`, promoted to `double`.
- Repeated launches on a non-blocking, non-default CUDA stream.
- Finite outputs, unchanged inputs and weights, and untouched output padding.

This gives `9 widths * 3 batches * 3 patterns * 3 epsilons = 243` cases per dtype, or 729 total.

For each dtype/width pair, the test warms up the kernel, resets output to a sentinel, captures invocation on the caller's stream, verifies that the graph contains exactly one CUDA kernel node, and replays it. The replayed result is compared with the CPU reference. All 27 replays passed; these checks occur within the 729 correctness cases.

Compute Sanitizer `memcheck` passed the complete C++ test with `ERROR SUMMARY: 0 errors`. The Python driver passed the project's Ruff rules using Ruff 0.15.4.

## Runtime dependencies

`readelf` and `ldd` inspection confirmed:

| Artifact | Direct ELF dependencies |
|---|---|
| `libtriton_rmsnorm_kernels.so` | `libcuda.so.1`, `libc.so.6` |
| C++ caller | CUDA runtime/driver and standard C/C++ libraries |

The caller's CUDA runtime dependency comes from allocation, stream, copy, and graph APIs used by the test. The generated kernel library itself uses the CUDA driver only.

Neither artifact depends on Python, a separate Triton runtime, TVM FFI, or TVM runtime support. This is the lightest runtime dependency set among the three locally verified paths, although it does not establish a performance advantage.

## Reproduction

Run from the repository root in the verified Triton 3.7.0 environment. First export kernels and the CMake source manifest:

```bash
python scripts/test_triton_rmsnorm.py
```

The default is `--epsilon-type fp64`. Python does not compile the generated C sources, build the application, or run tests.

Use CMake for both the generated C launcher library and the C++ application, then verify with CTest:

```bash
cmake -S scripts -B build/triton_rmsnorm_cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-15 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-15 \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.3 \
  -DENABLE_CUTE_DSL=OFF -DENABLE_TILELANG=OFF
cmake --build build/triton_rmsnorm_cmake --parallel 4
ctest --test-dir build/triton_rmsnorm_cmake --output-on-failure
```

CMake reads `rmsnorm_sources.cmake` to select the current generated sources and does not invoke Python. CTest removes `PYTHONPATH` and sets the CUDA runtime library search path.

Run the already-built executable without Python, or check it with Compute Sanitizer:

```bash
env -u PYTHONPATH build/triton_rmsnorm_cmake/bin/test_triton_rmsnorm \
  "$PWD/build/triton_rmsnorm_cmake/lib/libtriton_rmsnorm_kernels.so"

env -u PYTHONPATH compute-sanitizer --tool memcheck --error-exitcode 1 \
  build/triton_rmsnorm_cmake/bin/test_triton_rmsnorm \
  "$PWD/build/triton_rmsnorm_cmake/lib/libtriton_rmsnorm_kernels.so"
```

Reproduce the expected FP32 scalar ABI failure in separate artifact and CMake build directories:

```bash
python scripts/test_triton_rmsnorm.py \
  --epsilon-type fp32 \
  --build-dir build/triton_rmsnorm_fp32_probe
cmake -S scripts -B build/triton_rmsnorm_fp32_probe_cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-15 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-15 \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.3 \
  -DENABLE_CUTE_DSL=OFF -DENABLE_TILELANG=OFF \
  -DTRITON_ARTIFACT_DIR="$PWD/build/triton_rmsnorm_fp32_probe"
cmake --build build/triton_rmsnorm_fp32_probe_cmake --parallel 4
ctest --test-dir build/triton_rmsnorm_fp32_probe_cmake --output-on-failure
```

Export and build should succeed; the final CTest command is expected to fail with the NaN mismatch shown above. This does not overwrite the passing artifacts.

The C++ executable embeds a run path for the tested CUDA library directory. A relocated deployment must provide suitable native library search paths. CUDA discovery is controlled by `CUDAToolkit_ROOT`, and native compilers by CMake's `CMAKE_C_COMPILER` and `CMAKE_CXX_COMPILER`.

## Production implications and limits

A TurboMind adapter could load a kernel shared library during initialization, resolve and retain its function pointers, explicitly initialize the CUDA modules, and launch with existing device pointers, runtime dimensions, and the engine's CUDA stream. No Python or tensor-view conversion would be required in the inference path.

However, the stock generated launcher is not a complete production abstraction:

- It stores module/function handles in global variables. Multi-context, multi-GPU, and concurrent initialization behavior were not verified.
- Its load/unload error helper calls `exit()` on CUDA errors instead of returning an error to the application.
- Compilation warns that a zero-grid path reaches the end of the launch function without returning a `CUresult`. Only positive grids were tested; empty workloads need explicit handling before invocation.
- The generated source also emits unused-parameter warnings. No generated code was edited to suppress these warnings.
- The installed AOT compiler rejects kernels requiring nonzero global or profiling scratch buffers. This verification does not establish support for those configurations.
- The caller must honor pointer alignment, buffer bounds, inner-stride, dtype, and maximum-width assumptions. The generated raw-pointer ABI does not supply tensor shape validation.
- Graph verification covers warmed-up kernels, not first-use capture or arbitrary Triton kernels.
- The fixed 8192-element reduction block is a correctness design, not a tuned choice for every width.

Performance, full TurboMind integration, comprehensive failure-path handling, and other Triton versions remain unverified. The FP32 scalar ABI issue remains unresolved in the installed exporter; the passing FP64 interface must not be presented as proof that the FP32 path works.

## References

- [Triton AOT compiler](https://github.com/triton-lang/triton/blob/main/python/triton/tools/compile.py)
- [Triton CUDA AOT launcher template](https://github.com/triton-lang/triton/blob/main/python/triton/tools/extra/cuda/compile.c)
- [Triton NVIDIA backend type mapping](https://github.com/triton-lang/triton/blob/main/python/triton/backends/nvidia/driver.py)

Online sources can change. These findings refer to the installed Triton 3.7.0 code, the generated artifacts, and the standalone verification scripts in this repository.
