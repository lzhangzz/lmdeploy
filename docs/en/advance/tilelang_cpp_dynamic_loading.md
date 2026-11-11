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

| Component                                  | Version / configuration    |
| ------------------------------------------ | -------------------------- |
| GPU                                        | NVIDIA GeForce RTX 5080    |
| TileLang                                   | 0.1.9                      |
| Apache TVM FFI                             | 0.1.9                      |
| CUDA toolkit and executable's CUDA runtime | 13.3                       |
| NVIDIA driver                              | 615.71.09                  |
| Host                                       | Linux x86_64               |
| Python used for export                     | 3.13                       |
| NVCC host compiler and C++ caller compiler | GCC 15                     |
| C++ caller build                           | CMake Release build, C++17 |
| Requested TileLang target                  | `cuda -arch=sm_120`        |

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
| ----- | ----------------: | ---------------------: | ---: | ---: | ------------: |
| FP32  |         81 passed |             4.76837e-7 | 2e-5 | 2e-5 |      3 passed |
| FP16  |         81 passed |             0.00195312 | 2e-3 | 2e-3 |      3 passed |
| BF16  |         81 passed |              0.0078125 | 2e-2 | 2e-2 |      3 passed |

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

## TileLang 0.1.15: pointer state and full FlashQLA

The results above are historical TileLang/FFI **0.1.9 RMSNorm** measurements.
They were not rerun or relabeled after upgrading. The following separate
verification uses TileLang **0.1.15**, TVM FFI **0.1.11**, Z3 **4.15.4.0**, CUDA
13.3 and GCC 15 on the same RTX 5080.

The newer wheel separates `libtvm_runtime.so` from `libtvm_compiler.so`.
The standalone CMake discovery now accepts `tvm_runtime` before legacy `tvm`.
Dictionary CUDA target syntax is required by these exporters. Full FlashQLA
uses `{'kind': 'cuda', 'arch': 'sm_120a'}` explicitly; its fused register
reconfiguration is not supported by a plain `sm_120` target.

### Direct-pointer proof of concept

[`test_tilelang_pointer_state.py`](../../../scripts/test_tilelang_pointer_state.py)
and its [C++ caller](../../../scripts/test_tilelang_pointer_state.cpp) verify
native `T.make_tensor_from_addr` plus ordinary `T.copy`, fp32 accumulator
fragments, and fp32/bf16 in-place state updates through an int64 device pointer
table. There are no injected CUDA helpers or generated-source patches.

- Four exported modules: fp32/bf16 state, 128/512-thread blocks, with the first
  128 threads doing state I/O and separate 64-column value tiles.
- **720 exact C++ cases passed**, including grouped layers/heads, partial groups,
  remapped pointers, guards, store masks, repeats and non-default streams.
- Four warmed-up capture/replay checks passed for these POC modules only.
- Memcheck and racecheck reported zero errors. This POC has no shared-memory
  compute pipeline; its racecheck result is not proof about full FlashQLA.
- The C++ process loads **only `libtvm_runtime.so`** as TVM runtime support,
  in addition to its linked FFI and CUDA libraries. No Python is needed.

Use `ENABLE_TILELANG_POINTER_STATE=ON` in the standalone `scripts/` CMake
project. See [the implementation plan](../../../FLASHQLA_PREFILL_PLAN.md) for
its direct-state addressing contract.

### Full exported FlashQLA kernels

[`lmdeploy/turbomind/kernels/tilelang/flashqla/`](../../../lmdeploy/turbomind/kernels/tilelang/flashqla/)
maintains explicit adapted factories, one module per upstream source file
(`cumsum.py`, `kkt_solve.py`, `fused_fwd.py`), derived from QwenLM/FlashQLA revision
`a14dbf3474514e1674526431e960325f72081e9f`: chunk-32 cumsum, strided KKT and
no-CP fused prefill, exported through TVM FFI. All three factories are
repo-local; exporting needs no installed FlashQLA package or upstream source
hash. Deterministic `flashqla_v2_*` library/function names carry the positional
ABI (`v2`) and every specialization parameter (`Hv`, `Hq`, state dtype). There is
no JSON manifest, no inventory file and no runtime provenance validation.

The fused kernel reads TurboMind's grouped `state_ptrs` directly, using the
existing layer offset in elements converted to bytes for address creation.
It performs masked in-place state stores and accepts interleaved Q/K/V strides.
There are no dense state buffers, gather/scatter, or Q/K/V packing operations.
One maintained KKT change synchronizes inversion rows: shared-memory hazards
were reproduced with unmodified pinned upstream before adding that barrier.

**The full kernels require more runtime support than the pointer POC:** both
`libtvm_runtime.so` and `libtilelang.so`, loaded with `RTLD_GLOBAL` and kept
resident. Without `libtilelang.so`, the full exports fail to resolve the TMA
creation callback `__tvm_tensormap_create_tiled`. That library transitively
loads TVM's compiler and Z3 libraries. A smaller runtime-only package for full
FlashQLA has **not** been built or verified. In TurboMind the Python front end
loads both (see below); the C++ engine itself invokes no Python interpreter or
runtime JIT.

Historical standalone verification (the standalone C++ verifier,
`scripts/test_flashqla_prefill.cpp`, has since been removed; the engine path
below now exercises loading without Python):

- Six C++ head/state configurations passed CMake/CTest against an independent
  sequential GDN reference on non-default streams. Maximum output error was
  `0.000239674`; state error was `0.00172487` for fp32 and `0.00195312` for
  bf16. Guard regions, masked state, inactive layers and output padding stayed
  unchanged.
- Exported modules passed memcheck and racecheck with zero errors. Racecheck
  checks shared-memory hazards, not arbitrary global-memory races.

An exported Python comparison suite (256 cases, since removed from the tree)
passed against kernels built through the model-load cache: pinned upstream and the
repo reference, including `(Hv,Hq)=(16,16)` for Qwen3.5-0.8B, zero/nonzero state,
exact 32/128-token boundaries, irregular varlen lengths, interleaved views,
grouped state, masks and repeated updates.

### Reusable C++ layer for exported TileLang kernels

`src/turbomind/kernels/tilelang/` (built when `ENABLE_TILELANG=ON`) is
kernel-agnostic and is what a new kernel family reuses:

- `tensor.h`: `ToDLTensor` (borrows a `core::Tensor`'s shape/stride storage),
  `ToDLDataType`, and `CollapseBatchStride` (a singleton leading dimension may
  carry any stride in borrowed views; TileLang's TMA setup expects the dense
  value). Exported functions take `DLTensor*` arguments directly; no
  `ffi::TensorView` wrapper is needed.
- `stream.h`: `StreamScope` saves/restores the TVM FFI environment stream around
  a launch, including on the engine's auxiliary prefill stream.
- `library.h`: `tilelang::Library`, a registry mapping each symbol to the
  module file that exports a function of that name. **Each kernel family owns
  its own instance** (FlashQLA's lives in `kernel/flashqla/host.cc`), so
  families never see each other's kernels. `Add(symbol, path)` records the
  file, which must exist; re-adding the same file is a no-op and a different
  file for the same symbol is an error. `Find` loads the module on first use, on the thread that first
  plans with it, and memoizes it; an unregistered symbol returns nullptr.
  `Contains` checks registration without loading. C++ never scans directories.

Contract for a new kernel family:

1. Own a `tilelang::Library` (a never-deleted heap instance, so loaded CUDA
   modules outlive static destruction) and expose one registration binding,
   `_tm.add_<family>_kernel(symbol, path)`, built only with
   `ENABLE_TILELANG=ON`. The family's Python module provides `available()`
   (the binding exists) and `register(paths)` for the generic
   `KernelProvider`.
2. Give every symbol an ABI version and encode every specialization parameter
   in the name, so the C++ side can compute it.
3. Register exactly the modules a model needs. An unregistered symbol means
   "not available" (fall back); a registered module that fails to load or lacks
   its symbol is an error.
4. The TileLang runtime must already be loaded; C++ does not `dlopen` it.

There is no capture-refusal helper. Exported launches do argument checks, set
the device, encode TMA descriptors on the host and launch; no allocation,
memcpy or synchronization. Callers that capture CUDA graphs anyway must run once
before capturing (TVM loads the CUDA module on first call) and must treat
addresses in TMA descriptors, grid sizes and scalar arguments as frozen at
capture. **Kernel-level graph capture is not refused and not validated.**

### Model-driven kernel provisioning

Kernels are built at model load, for exactly the layers the model has:

1. `DeltaNetBuilder` (`lmdeploy/turbomind/builders/deltanet.py`) states its
   layer's `GdnSpec`: post-TP `Hv`/`Hq`, head dimension, input and state
   dtype. It asks the build context's `KernelProvider` (`ctx.kernels`), the same
   way builders ask `ctx.gemm` for weight plans.
2. The provider returns nothing unless `TM_GDR_BACKEND=flashqla` (parsed only in
   Python) and the family can build the spec (sm120, bf16 inputs, head dim 128,
   fp32/bf16 state), and the extension was built with `ENABLE_TILELANG=ON`.
   Otherwise it compiles missing kernels into the cache, loads the TileLang
   runtime with `RTLD_GLOBAL` (`lmdeploy/turbomind/kernels/tilelang/runtime.py`),
   and has the family register each of the spec's modules with its own
   registry (`flashqla.register` → `_tm.add_flashqla_kernel(symbol, path)`). Other kernels in the cache directory
   stay invisible. A requested, buildable spec whose kernels cannot be provided
   raises.
3. In C++, registration is the opt-in: `GatedDeltaRule::Plan` tries the
   `flashqla` registry entry only when the layer's three symbols are registered
   (`flashqla::Registered`, which does not load), and its `Match` loads them. Runtime
   policy (CP level, force-legacy, speculative rows) stays in that `Match`.
   Nothing travels through the layer config, so registration is process-wide:
   two engines in one process cannot pick different GDR backends.

The cache lives at `$LMDEPLOY_KERNEL_CACHE`, else
`$XDG_CACHE_HOME/lmdeploy/tilelang`, else `~/.cache/lmdeploy/tilelang`, as
`<family>-<hash>/<symbol>.so`. The hash covers the target (`sm_120a`), the
TileLang/TVM FFI versions, the host compiler (`$CXX`, else `c++`) and its
version, and the factory sources. Builders lock the directory and publish each
`.so` with an atomic rename. The NVCC host compiler must be one CUDA supports
(GCC 15 for CUDA 13.3), so set `CXX` accordingly.

A cold build of one model's kernels (Qwen3.5-0.8B, bf16 state) took 8.5 s on
the RTX 5080: cumsum 1.3 s, KKT 1.8 s, fused 5.4 s. Later loads build nothing.

Build TurboMind with `-DENABLE_TILELANG=ON -DTVM_FFI_ROOT=<site-packages>/tvm_ffi`.
No artifacts are needed at configure or build time.

### Integration status

The adapter is stateless: Plan retains only existing geometry/workspace fields,
with no opaque metadata payload or extra preparation hook. Cumsum and KKT derive
chunk indexing from the engine's existing device offsets; the caller's workspace
contains only scratch. The Python kernel provider registers the model's kernels
before engine creation, without GPU allocation, private streams or kernel
launches; modules load at the first `Match`, which `DeltaNetLayer` performs during engine
construction. CUDA first-use initialization is left to the engine's existing
model warm-up. Scratch shapes, reshaped one-dimensional state pointers and
adjusted singleton TMA strides use local stack arrays. There is no backend-owned
pinned staging, event polling, synchronization destructor or execution mutex.
GPU buffer lifetime and stream ordering remain with the engine's existing phase
and fork/join protocol.

The native-only build needs no FFI dependency; the enabled Python extension adds
an FFI library search path. Dispatch/lifecycle tests (since removed from the
tree) covered registration opt-in, fallback for missing kernels, errors for
incompatible modules, memoized loading, loading only the needed state dtype,
independent workspace reuse, explicit cross-stream dependencies, device-side
offset updates after planning and nonblocking warmed submission/Plan release.
Launch/lifetime memcheck passed for both state dtypes with zero errors.

Four real Qwen3.5-0.8B cases passed experimental inference checks (the model
test has since been removed), including
fp32/bf16 state, grouped checkpoints, split prefill, native decode, repeated
requests, multi-turn continuation, prefix-cache restore, cold and warm kernel
caches, and native runs that never import TileLang. **Native token/logit parity
does not pass.** The user accepted experimental opt-in operation with reported
drift; the model test's strict parity mode exposed that mismatch. In a small 4,044-token first-token timing sample FlashQLA was
about 2–4% slower than native. See the plan for exact drift, timing and
regression results. This is not a broad model-quality or performance assessment.

Native is still the default. Opt-in requires **both**
`TM_GDR_BACKEND=flashqla` and `TM_GDR_CP_LEVEL=0`; incompatible ordinary
workloads and decode/speculative work remain native.
