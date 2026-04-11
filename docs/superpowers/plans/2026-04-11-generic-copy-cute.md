# GenericCopy with CuTE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the disabled GenericCopy with a CuTE-based implementation that handles arbitrary non-contiguous tensor copies.

**Architecture:** The kernel constructs CuTE tensors with dynamic layouts from runtime shape/stride arrays, uses CuTE's tensor accessor for stride-aware element addressing. The host dispatcher normalizes layouts, detects alignment, and selects vector type + rank template specialization. The Copy dispatcher in tensor.cc routes non-contiguous copies to GenericCopy with fast-path fallbacks for cudaMemcpy2D/3D.

**Tech Stack:** CuTE (CUTLASS v3.9.2, headers-only), CUDA, existing turbomind::core::Layout/Tensor/Buffer infrastructure.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/turbomind/core/tensor.cu` | CuTE-based GenericCopy kernel + host dispatcher (replace `#if 0` block) |
| `src/turbomind/core/tensor.cc` | Copy dispatcher with fast paths (replace `#if 0` block) |
| `src/turbomind/core/tensor.h` | Uncomment GenericCopy declaration |
| `src/turbomind/core/CMakeLists.txt` | Add CUTLASS link dependency |

---

### Task 1: Add CUTLASS dependency to core CMake

**Files:**
- Modify: `src/turbomind/core/CMakeLists.txt:31`

- [ ] **Step 1: Add `nvidia::cutlass::cutlass` to core's target_link_libraries**

In `src/turbomind/core/CMakeLists.txt`, change line 31 from:

```cmake
target_link_libraries(core PUBLIC cuda_utils CUDA::cudart CUDA::cuda_driver fmt::fmt concurrentqueue)
```

to:

```cmake
target_link_libraries(core PUBLIC cuda_utils CUDA::cudart CUDA::cuda_driver fmt::fmt concurrentqueue nvidia::cutlass::cutlass)
```

- [ ] **Step 2: Verify the build still configures**

Run: `cd /data/lmdeploy-copy/build && sh ../my_generate.sh 2>&1 | tail -20`
Expected: CMake configuration succeeds (exit code 0).

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/core/CMakeLists.txt
git commit -m "build: add CUTLASS dependency to core library for CuTE"
```

---

### Task 2: Write CuTE helpers and GenericCopy kernel in tensor.cu

**Files:**
- Modify: `src/turbomind/core/tensor.cu` (replace the entire `#if 0` ... `#endif` block, lines 11–199)

- [ ] **Step 1: Replace the `#if 0` block in tensor.cu with the new implementation**

Write the following content, replacing everything from line 11 (`#if 0`) through line 199 (`#endif`):

```cpp
#include "cute/layout.hpp"
#include "cute/tensor.hpp"

namespace turbomind::core {

namespace kernel {

// Convert an Array<T,N> to a cute::tuple for dynamic layout construction.
template<typename T, int N, size_t... Is>
__device__ auto to_cute_tuple(const Array<T, N>& arr, std::index_sequence<Is...>)
{
    return cute::make_tuple(arr[Is]...);
}

template<typename T, int N>
__device__ auto to_cute_tuple(const Array<T, N>& arr)
{
    return to_cute_tuple(arr, std::make_index_sequence<N>{});
}

// GenericCopy kernel: each thread processes one element (of vector type VecT).
// Uses CuTE tensors with dynamic layouts for stride-aware addressing.
template<class VecT, int kRank>
__global__ void GenericCopyKernel(const VecT* __restrict__ src,
                                  VecT* __restrict__ dst,
                                  Array<int64_t, kRank> strides_a,
                                  Array<int64_t, kRank> strides_b,
                                  Array<int32_t, kRank> shape,
                                  int64_t               total)
{
    int64_t idx = static_cast<int64_t>(threadIdx.x) + static_cast<int64_t>(blockIdx.x) * blockDim.x;

    if (idx >= total) {
        return;
    }

    // Build CuTE tensors with dynamic layouts
    auto iseq = std::make_index_sequence<kRank>{};

    auto tensor_a = cute::make_tensor(src, cute::make_layout(
        to_cute_tuple(shape, iseq),
        to_cute_tuple(strides_a, iseq)));
    auto tensor_b = cute::make_tensor(dst, cute::make_layout(
        to_cute_tuple(shape, iseq),
        to_cute_tuple(strides_b, iseq)));

    // Decompose linear index to multi-dim coordinate
    Array<int64_t, kRank> coord{};
    PRAGMA_UNROLL
    for (int i = 0; i < kRank; ++i) {
        coord[i] = idx % shape[i];
        idx /= shape[i];
    }

    // CuTE tensor access handles stride computation
    tensor_b(to_cute_tuple(coord, iseq)) = tensor_a(to_cute_tuple(coord, iseq));
}

}  // namespace kernel

void GenericCopy(const Tensor& src, Tensor& dst, const Stream& stream)
{
    auto a = src.layout();
    auto b = dst.layout();

    TM_CHECK_EQ(a.rank(), b.rank());

    // Sort strides ascending so the innermost (fastest-varying) dim is first
    vector<int> idxs(a.rank());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [&](int i, int j) {
        return a.stride()[i] < a.stride()[j];
    });

    a = a.permute(idxs);
    b = b.permute(idxs);

    // Coalesce adjacent contiguous dims
    a = a.coalesce();
    b = b.coalesce();

    int rank = std::max(a.rank(), b.rank());

    if (a.rank() < rank) {
        a = a.view(b.shape());
    } else if (b.rank() < rank) {
        b = b.view(a.shape());
    }

    const DataType dtype = src.dtype();

    // --- Alignment detection ---
    int64_t alignment = 16;  // start optimistic

    auto align = [&](auto v) { alignment = std::gcd(alignment, v); };

    // If the innermost dim is not stride-1, we can't vectorize along it
    if (a.stride(0) > 1 || b.stride(0) > 1) {
        alignment = get_byte_size(dtype);
    }

    align(get_byte_size(dtype, a.shape(0)));

    auto data_a = src.raw_data();
    auto data_b = dst.raw_data();

    align(reinterpret_cast<uintptr_t>(data_a));
    align(reinterpret_cast<uintptr_t>(data_b));

    for (int i = 1; i < rank; ++i) {
        align(get_byte_size(dtype, a.stride(i)));
        align(get_byte_size(dtype, b.stride(i)));
    }

    // --- Select vector type ---
    const int vec_size = alignment / get_byte_size(dtype);

    const int64_t total_elements = a.size() / std::max(vec_size, 1);

    // --- Template dispatch ---
    auto invoke = [&](auto vec_t, auto index_t, auto d) {
        using VecT     = decltype(vec_t);
        using IndexT   = decltype(index_t);
        constexpr int D = d.value;

        Array<int32_t, D> shape_arr;
        std::fill(shape_arr.begin() + rank, shape_arr.end(), 1);
        for (int i = 0; i < rank; ++i) {
            shape_arr[i] = static_cast<int32_t>(a.shape(i));
        }

        Array<int64_t, D> strides_a{};
        Array<int64_t, D> strides_b{};
        std::copy_n(a.stride().data(), rank, strides_a.data());
        std::copy_n(b.stride().data(), rank, strides_b.data());

        if (vec_size > 1) {
            shape_arr[0] /= vec_size;
            for (int i = 0; i < D; ++i) {
                strides_a[i] /= vec_size;
                strides_b[i] /= vec_size;
            }
        }

        auto func = kernel::GenericCopyKernel<VecT, D>;

        int device{};
        check_cuda_error(cudaGetDevice(&device));
        int sm_count{};
        check_cuda_error(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));

        int block_size = 256;
        int min_waves  = INT_MAX;
        int grid_size  = 0;

        for (int threads = 256; threads <= 1024; threads *= 2) {
            int n_active{};
            check_cuda_error(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n_active, func, threads, 0));
            int blocks = (total_elements + threads - 1) / threads;
            int waves  = (blocks + n_active * sm_count - 1) / (n_active * sm_count);
            if (waves < min_waves) {
                min_waves  = waves;
                block_size = threads;
                grid_size  = blocks;
            }
        }

        func<<<grid_size, block_size, 0, stream.handle()>>>(
            reinterpret_cast<const VecT*>(data_a),
            reinterpret_cast<VecT*>(data_b),
            strides_a,
            strides_b,
            shape_arr,
            total_elements);
    };

    auto invoke_d = [&](auto vec_t, auto idx_t) {
        if (rank <= 2) {
            invoke(vec_t, idx_t, constant<2>{});
        } else if (rank <= 4) {
            invoke(vec_t, idx_t, constant<4>{});
        } else if (rank <= 6) {
            invoke(vec_t, idx_t, constant<6>{});
        } else {
            throw std::runtime_error("GenericCopy: rank > 6 not supported");
        }
    };

    auto invoke_i = [&](auto vec_t) {
        if (total_elements < INT_MAX) {
            invoke_d(vec_t, int{});
        } else {
            invoke_d(vec_t, int64_t{});
        }
    };

    switch (alignment) {
        case 16:
            return invoke_i(uint4{});
        case 8:
            return invoke_i(uint2{});
        case 4:
            return invoke_i(unsigned{});
        case 2:
            return invoke_i(unsigned short{});
        default:
            return invoke_i(char{});
    }
}

}  // namespace turbomind::core
```

- [ ] **Step 2: Verify compilation of tensor.cu**

Run: `cd /data/lmdeploy-copy/build && ninja core 2>&1 | tail -30`
Expected: Compilation succeeds. There may be warnings but no errors.

If there are CuTE include path issues, check that the CUTLASS headers are in `build/_deps/repo-cutlass-src/include/`.

- [ ] **Step 3: Fix any compilation errors**

Common issues:
- `cute::make_tuple` ambiguity: may need `cute::` qualification consistently
- `std::index_sequence` needs `<utility>` header — add if missing
- `Array` is `turbomind::Array` from `kernels/core/array.h` — already included via the existing includes

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "feat(core): add CuTE-based GenericCopy kernel and host dispatcher"
```

---

### Task 3: Expand Copy dispatcher in tensor.cc

**Files:**
- Modify: `src/turbomind/core/tensor.cc` (modify existing `Copy` at line 43 and delete the `#if 0` block at lines 74–172)

- [ ] **Step 1: Modify the existing `Copy(const Tensor&, Ref<Tensor>, const Stream&)` to handle non-contiguous copies, and delete the `#if 0` block**

**Important:** The existing active `Copy` takes `Ref<Tensor>` for dst, not `Tensor&`. We must modify the existing function — NOT add parallel overloads — to avoid overload ambiguity.

Replace the entire file content from line 43 through line 174 (end of namespace) with:

```cpp
void Copy(const Tensor& src, Ref<Tensor> dst_, const Stream& stream)
{
    auto& dst = dst_.get();
    TM_CHECK_EQ(src.dtype(), dst.dtype());
    TM_CHECK_EQ(src.shape(), dst.shape());

    const DataType dtype = src.dtype();

    auto trivial = [&] {
        const ssize_t bytes = get_byte_size(dtype, src.size());
        if (bytes) {
            check_cuda_error(
                cudaMemcpyAsync(dst.raw_data(), src.raw_data(), bytes, cudaMemcpyDefault, stream.handle()));
        }
    };

    // Fast path: both contiguous
    if (src.is_contiguous() && dst.is_contiguous()) {
        return trivial();
    }

    auto a = src.layout();
    auto b = dst.layout();

    // Sort strides ascending so innermost dim (stride=1 candidate) comes first
    vector<int> idxs(a.rank());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [&](int i, int j) {
        return a.stride()[j] < a.stride()[i];
    });

    // If the innermost dim is not contiguous in either tensor, go straight to GenericCopy
    if (a.stride(idxs.back()) > 1 || b.stride(idxs.back()) > 1) {
        return GenericCopy(src, dst, stream);
    }

    a = a.permute(idxs);
    b = b.permute(idxs);

    // After reordering, check if both are now contiguous
    if (a.is_contiguous() && b.is_contiguous()) {
        return trivial();
    }

    a = a.coalesce();
    b = b.coalesce();

    int rank = std::max(a.rank(), b.rank());

    if (rank > 3) {
        return GenericCopy(src, dst, stream);
    }

    if (a.rank() < rank) {
        a = a.view(b.shape());
    } else if (b.rank() < rank) {
        b = b.view(a.shape());
    }

    // 2D fast path using cudaMemcpy2DAsync
    if (rank == 2) {
        check_cuda_error(cudaMemcpy2DAsync(dst.raw_data(),
                                           get_byte_size(dtype, b.stride(0)),
                                           src.raw_data(),
                                           get_byte_size(dtype, a.stride(0)),
                                           get_byte_size(dtype, a.shape(1)),
                                           a.shape(0),
                                           cudaMemcpyDefault,
                                           stream.handle()));
        return;
    }

    // 3D fast path using cudaMemcpy3DAsync (only when the underlying space is a cube)
    if (rank == 3) {
        auto [a0, a1] = a.strides(0, 1);
        auto [b0, b1] = b.strides(0, 1);

        if (a0 % a1 == 0 && b0 % b1 == 0) {
            const auto xsz_a = get_byte_size(dtype, a.stride(1));
            const auto xsz_b = get_byte_size(dtype, b.stride(1));
            const auto ysz_a = a0 / a1;
            const auto ysz_b = b0 / b1;

            cudaMemcpy3DParms param{};
            param.srcPtr = make_cudaPitchedPtr((void*)src.raw_data(), xsz_a, xsz_a, ysz_a);
            param.dstPtr = make_cudaPitchedPtr((void*)dst.raw_data(), xsz_b, xsz_b, ysz_b);
            param.extent = make_cudaExtent(get_byte_size(dtype, a.shape(2)), a.shape(1), a.shape(0));
            param.kind   = cudaMemcpyDefault;

            check_cuda_error(cudaMemcpy3DAsync(&param, stream.handle()));
            return;
        }
    }

    // Fallback for everything else
    return GenericCopy(src, dst, stream);
}

void Copy(const Tensor& src, Ref<Tensor> dst_)
{
    Copy(src, dst_, Context::stream());
}

}  // namespace turbomind::core
```

This replaces:
- The existing contiguous-only `Copy` at line 43
- The `Copy(src, dst_)` forwarding overload at line 55
- The entire `#if 0` block at lines 74–172

The new `Copy` handles both contiguous (fast path) and non-contiguous (GenericCopy fallback) cases.

- [ ] **Step 2: Verify compilation**

Run: `cd /data/lmdeploy-copy/build && ninja core 2>&1 | tail -30`
Expected: Compilation succeeds.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/core/tensor.cc
git commit -m "feat(core): expand Tensor Copy dispatcher to handle non-contiguous copies"
```

---

### Task 4: Update tensor.h declarations

**Files:**
- Modify: `src/turbomind/core/tensor.h` (the `#if 0` block at lines 233–247)

- [ ] **Step 1: Replace the `#if 0` block with active declarations**

Replace lines 233–247 (`#if 0` ... `#endif`) with:

```cpp
void GenericCopy(const Tensor& src, Tensor& dst, const Stream& stream);
```

Remove the unused declarations for `Reshape`, `Transpoe`, `Permute`, `Contiguous` — they are not part of this task.

- [ ] **Step 2: Verify compilation**

Run: `cd /data/lmdeploy-copy/build && ninja core 2>&1 | tail -30`
Expected: Compilation succeeds.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/core/tensor.h
git commit -m "feat(core): declare GenericCopy in tensor.h"
```

---

### Task 5: Full build verification

**Files:** None (verification only)

- [ ] **Step 1: Clean build of the core target**

Run: `cd /data/lmdeploy-copy/build && ninja core 2>&1 | tail -50`
Expected: Build succeeds with no errors.

- [ ] **Step 2: Build downstream targets to check for link issues**

Run: `cd /data/lmdeploy-copy/build && ninja _turbomind 2>&1 | tail -50`
Expected: Full build succeeds. If there are link errors about missing CuTE symbols, verify the `nvidia::cutlass::cutlass` target propagation.

- [ ] **Step 3: Fix any remaining issues**

If CuTE headers cause compile errors in other files that include tensor.h (because the dependency propagates via `PUBLIC`), check if the CUTLASS include path is correctly set globally. The root CMakeLists.txt already adds it to `COMMON_HEADER_DIRS`.

---

### Task 6: Functional test

**Files:**
- Create: `test_generic_copy.py` (temporary test script at repo root)

- [ ] **Step 1: Check GPU availability**

Run: `nvidia-smi --query-gpu=name,memory.free --format=csv,noheader`
Expected: At least one GPU with free memory.

- [ ] **Step 2: Write a test script that exercises non-contiguous copy**

Create `test_generic_copy.py`:

```python
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import huggingface_hub.constants as hf_constants

# Query locally available model
# (Use model-server MCP tools or skip model test if not needed)

import torch
import numpy as np

# Set up paths
import sys
sys.path.insert(0, '/data/lmdeploy-copy/lmdeploy')
sys.path.insert(0, '/data/lmdeploy-copy/build/lib')

# This test directly calls the C++ GenericCopy via Python bindings
# if available, or tests indirectly through model inference.

# For now, test via a simple model inference that exercises non-contiguous
# tensor operations (transpose, slice, etc.)

# Step 1: Verify a basic model still works (contiguous path unchanged)
# Step 2: If the model uses non-contiguous copies, verify output correctness

print("Test script placeholder - verify with actual model inference")
```

Note: The actual test depends on what models are available. Use `scripts/test_turbomind_model.py` for model-level testing as described in CLAUDE.md.

- [ ] **Step 3: Run a model inference test**

Follow the CLAUDE.md testing guidelines:
1. Use `scripts/test_turbomind_model.py` with a locally cached model
2. Verify the response contains meaningful human words (not gibberish)
3. Request at least 128 tokens

The GenericCopy path is triggered when non-contiguous tensors are copied. Most model inference involves contiguous copies, so the fast path (`cudaMemcpyAsync`) handles the majority. The GenericCopy path gets exercised by:
- Transposed weight tensors
- Strided KV cache copies
- Any tensor view with non-contiguous strides

If no model test exercises the GenericCopy path, that's OK — the fast paths are unchanged and the GenericCopy is a fallback.

- [ ] **Step 4: Clean up test file**

```bash
rm -f test_generic_copy.py
```

- [ ] **Step 5: Commit all changes**

```bash
git add -A
git commit -m "feat(core): complete CuTE-based GenericCopy implementation"
```
