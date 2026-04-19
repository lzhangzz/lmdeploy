# CMake Integration for CuTe GEMM Samples

## Overview

Integrate the 7 BF16 GEMM `.cu` samples under `cute-reference/samples/` into the project's CMake build system as individually buildable targets.

## What Changes

### New file: `cute-reference/samples/CMakeLists.txt`

A `foreach` loop defines 7 `add_executable` targets from the bf16_gemm `.cu` source files:

| Target | Source |
|--------|--------|
| `bf16_gemm_sm80` | `bf16_gemm_sm80.cu` |
| `bf16_gemm_sm80_opt` | `bf16_gemm_sm80_opt.cu` |
| `bf16_gemm_sm80_pipe` | `bf16_gemm_sm80_pipe.cu` |
| `bf16_gemm_sm80_pipe_256x128` | `bf16_gemm_sm80_pipe_256x128.cu` |
| `bf16_gemm_sm80_pipe_epilogue` | `bf16_gemm_sm80_pipe_epilogue.cu` |
| `bf16_gemm_sm80_pipe_tma` | `bf16_gemm_sm80_pipe_tma.cu` |
| `bf16_gemm_sm80_pipe_tma_ws` | `bf16_gemm_sm80_pipe_tma_ws.cu` |

Each target:
- Links `nvidia::cutlass::cutlass` (transitive CUTLASS/CUTE includes)
- Compiles with `-O3` for CUDA language
- Compiles with `$<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>` for verbose PTX assembly output
- Uses `90a` CUDA architecture (required for TMA operations)

The file sets `CMAKE_RUNTIME_OUTPUT_DIRECTORY` to `${CMAKE_BINARY_DIR}/bin` so executables land in `build/bin/`.

### Modified: Top-level `CMakeLists.txt`

Add at the bottom, before the test section:

```cmake
option(BUILD_CUTE_SAMPLES "Build CuTe reference GEMM samples" OFF)
if(BUILD_CUTE_SAMPLES)
  add_subdirectory(cute-reference/samples)
endif()
```

Default `OFF` so normal builds are unaffected. Enable with `-DBUILD_CUTE_SAMPLES=ON`.

### Modified: `cute-reference/samples/build_and_run.sh`

Remove all `nvcc` compilation commands and the `CUTLASS_INC` variable. Keep only the run sections, updated to reference `../../build/bin/<binary>`.

## What Stays the Same

- Test utilities (`test_stsm_layout.cu`, `test_tma_load.cu`, `test_tma_store.cu`, `test_tma_store_only.cu`) remain outside CMake
- `cute-reference/verify/` is untouched
- Existing project CUDA flags and architecture settings are not modified (samples set their own arch to `90a`)

## Usage

```bash
# Configure
cmake -B build -DBUILD_CUTE_SAMPLES=ON

# Build all samples
ninja -C build bf16_gemm_sm80_pipe_tma_ws

# Or build all at once
ninja -C build

# Run
./build/bin/bf16_gemm_sm80_pipe_tma_ws 1024 1024 1024

# Or use the script
cd cute-reference/samples && ./build_and_run.sh
```
