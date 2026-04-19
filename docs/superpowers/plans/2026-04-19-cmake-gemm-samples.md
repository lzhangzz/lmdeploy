# CMake GEMM Samples Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the 7 BF16 GEMM `.cu` samples as CMake targets hooked into the project build.

**Architecture:** New `cute-reference/samples/CMakeLists.txt` defines 7 executables in a foreach loop, each linking `nvidia::cutlass::cutlass`. Top-level `CMakeLists.txt` gets an `option(BUILD_CUTE_SAMPLES)` gate. `build_and_run.sh` strips compilation and references `build/bin/`.

**Tech Stack:** CMake 3.11+, CUDA (nvcc), CUTLASS 3.9.2 (via FetchContent)

---

### Task 1: Create `cute-reference/samples/CMakeLists.txt`

**Files:**
- Create: `cute-reference/samples/CMakeLists.txt`

- [ ] **Step 1: Write the CMakeLists.txt**

```cmake
# CuTe reference GEMM sample kernels
cmake_minimum_required(VERSION 3.11)

set(CUTE_GEMM_SAMPLES
    bf16_gemm_sm80
    bf16_gemm_sm80_opt
    bf16_gemm_sm80_pipe
    bf16_gemm_sm80_pipe_256x128
    bf16_gemm_sm80_pipe_epilogue
    bf16_gemm_sm80_pipe_tma
    bf16_gemm_sm80_pipe_tma_ws
)

foreach(sample ${CUTE_GEMM_SAMPLES})
    add_executable(${sample} ${sample}.cu)
    target_link_libraries(${sample} PRIVATE nvidia::cutlass::cutlass)
    target_compile_options(${sample} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
    )
    set_target_properties(${sample} PROPERTIES CUDA_ARCHITECTURES "90a-real")
endforeach()
```

Notes:
- `nvidia::cutlass::cutlass` is the INTERFACE library from the CUTLASS FetchContent (provides CUTLASS and CUTE include paths transitively).
- `CUDA_ARCHITECTURES "90a-real"` overrides the project-wide multi-arch list since TMA operations require SM90a.
- `-O3` is per-target so samples always get optimized CUDA regardless of build type.
- `CMAKE_RUNTIME_OUTPUT_DIRECTORY` is inherited from the top-level (`${CMAKE_BINARY_DIR}/bin`) — no need to set it here.

- [ ] **Step 2: Verify the file exists**

Run: `cat cute-reference/samples/CMakeLists.txt`
Expected: the content above

- [ ] **Step 3: Commit**

```bash
git add cute-reference/samples/CMakeLists.txt
git commit -m "Add CMakeLists.txt for CuTe GEMM samples"
```

---

### Task 2: Hook into top-level CMakeLists.txt

**Files:**
- Modify: `CMakeLists.txt:346-361`

The `add_subdirectory(src)` is at line 346. The BUILD_CUTE_SAMPLES option should go after line 346 (after `add_subdirectory(src)`) and before the install section (line 353).

- [ ] **Step 1: Add the option gate**

Insert after line 346 (`add_subdirectory(src)`):

```cmake
option(BUILD_CUTE_SAMPLES "Build CuTe reference GEMM samples" OFF)
if(BUILD_CUTE_SAMPLES)
  add_subdirectory(cute-reference/samples)
endif()
```

This goes between `add_subdirectory(src)` (line 346) and the commented-out `BUILD_TEST` block (line 348).

- [ ] **Step 2: Reconfigure the build with the option enabled**

Run: `cd /data/lmdeploy-cute/build && cmake .. -DBUILD_CUTE_SAMPLES=ON`
Expected: CMake configures successfully, the 7 sample targets appear in the configuration output.

- [ ] **Step 3: Build one sample to verify compilation**

Run: `ninja -C /data/lmdeploy-cute/build bf16_gemm_sm80_pipe_tma_ws`
Expected: Compiles without errors, produces `build/bin/bf16_gemm_sm80_pipe_tma_ws`.

- [ ] **Step 4: Run it to verify correctness**

Run: `/data/lmdeploy-cute/build/bin/bf16_gemm_sm80_pipe_tma_ws 1024 1024 1024`
Expected: `max error 1.250305e-01 — PASS` (same as the standalone-compiled version).

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt
git commit -m "Add BUILD_CUTE_SAMPLES option to top-level CMakeLists.txt"
```

---

### Task 3: Update `build_and_run.sh` to use CMake-built binaries

**Files:**
- Modify: `cute-reference/samples/build_and_run.sh`

- [ ] **Step 1: Rewrite the script**

Replace the entire file with:

```bash
#!/bin/bash
set -e

BIN=../../build/bin

echo ""
echo "=== Running plain sample ==="
echo "--- 1024x1024x1024 ---"
$BIN/bf16_gemm_sm80 1024 1024 1024

echo ""
echo "=== Running optimized sample (swizzle+LDSM) ==="
echo "--- 1024x1024x1024 ---"
$BIN/bf16_gemm_sm80_opt 1024 1024 1024

echo ""
echo "=== Running pipelined sample (cp.async 3-stage) ==="
echo "--- 1024x1024x1024 ---"
$BIN/bf16_gemm_sm80_pipe 1024 1024 1024

echo ""
echo "=== Running 256x128 tile sample (256 threads) ==="
echo "--- 1024x1024x1024 ---"
$BIN/bf16_gemm_sm80_pipe_256x128 1024 1024 1024

echo ""
echo "=== Running STSM epilogue sample (BF16 output) ==="
echo "--- 1024x1024x1024 ---"
$BIN/bf16_gemm_sm80_pipe_epilogue 1024 1024 1024

echo ""
echo "=== Running TMA load + STSM/TMA store sample ==="
echo "--- 1024x1024x1024 ---"
$BIN/bf16_gemm_sm80_pipe_tma 1024 1024 1024

echo ""
echo "=== Running warp-specialized TMA load/store sample ==="
echo "--- 1024x1024x1024 ---"
$BIN/bf16_gemm_sm80_pipe_tma_ws 1024 1024 1024
```

- [ ] **Step 2: Verify the script runs**

Run: `cd /data/lmdeploy-cute/cute-reference/samples && bash build_and_run.sh`
Expected: All 7 samples run and pass correctness checks.

- [ ] **Step 3: Commit**

```bash
git add cute-reference/samples/build_and_run.sh
git commit -m "Update build_and_run.sh to use CMake-built binaries"
```
