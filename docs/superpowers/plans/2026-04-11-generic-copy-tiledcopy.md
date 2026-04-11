# GenericCopy CuTe TiledCopy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the plain-CUDA GenericCopy kernel with a CuTe TiledCopy-based implementation for improved performance on pathological stride patterns.

**Architecture:** Each CTA handles one "row" of outer dimensions (mapped via blockIdx.x). Within the CTA, a CuTe TiledCopy distributes the innermost dimension across 256 threads. CuTe layout types are used throughout the kernel interface — no turbomind `Array<>` types in the kernel signature. The host constructs CuTe layouts on the host side and passes them to the kernel. VecT selection (uint4/uint2/uint/ushort/char) remains on the host.

**Tech Stack:** CuTe (CUTLASS v3.9.2, headers-only), CUDA, existing `turbomind::core::Tensor/Layout` infrastructure.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/turbomind/core/tensor.cu` | CuTe TiledCopy GenericCopy kernel + updated host dispatcher |
| `src/turbomind/core/CMakeLists.txt` | Add CUTLASS link dependency to core target |

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
git commit -m "build: add CUTLASS dependency to core library for CuTe TiledCopy"
```

---

### Task 2: Write CuTe TiledCopy kernel and update host dispatcher in tensor.cu

**Files:**
- Modify: `src/turbomind/core/tensor.cu` (replace entire file contents)

This task replaces the entire file. The kernel uses CuTe layout types for all parameters. The host constructs CuTe layouts and passes them as kernel arguments.

- [ ] **Step 1: Replace tensor.cu with CuTe TiledCopy implementation**

Write the following as the complete contents of `src/turbomind/core/tensor.cu`:

```cpp
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/core/meta.h"

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>

#include <numeric>

namespace turbomind::core {

// ============================================================================
// Helpers: Construct CuTe layouts from runtime shape/stride arrays
// ============================================================================
namespace detail {

template<int kRank>
auto make_cute_shape(const ssize_t* shape_data)
{
    if constexpr (kRank == 1) {
        return cute::make_shape(static_cast<int32_t>(shape_data[0]));
    }
    else if constexpr (kRank == 2) {
        return cute::make_shape(static_cast<int32_t>(shape_data[0]),
                                static_cast<int32_t>(shape_data[1]));
    }
    else if constexpr (kRank == 3) {
        return cute::make_shape(static_cast<int32_t>(shape_data[0]),
                                static_cast<int32_t>(shape_data[1]),
                                static_cast<int32_t>(shape_data[2]));
    }
    else if constexpr (kRank == 4) {
        return cute::make_shape(static_cast<int32_t>(shape_data[0]),
                                static_cast<int32_t>(shape_data[1]),
                                static_cast<int32_t>(shape_data[2]),
                                static_cast<int32_t>(shape_data[3]));
    }
    else if constexpr (kRank == 5) {
        return cute::make_shape(static_cast<int32_t>(shape_data[0]),
                                static_cast<int32_t>(shape_data[1]),
                                static_cast<int32_t>(shape_data[2]),
                                static_cast<int32_t>(shape_data[3]),
                                static_cast<int32_t>(shape_data[4]));
    }
    else if constexpr (kRank == 6) {
        return cute::make_shape(static_cast<int32_t>(shape_data[0]),
                                static_cast<int32_t>(shape_data[1]),
                                static_cast<int32_t>(shape_data[2]),
                                static_cast<int32_t>(shape_data[3]),
                                static_cast<int32_t>(shape_data[4]),
                                static_cast<int32_t>(shape_data[5]));
    }
}

template<int kRank>
auto make_cute_stride(const ssize_t* stride_data)
{
    if constexpr (kRank == 1) {
        return cute::make_stride(static_cast<int64_t>(stride_data[0]));
    }
    else if constexpr (kRank == 2) {
        return cute::make_stride(static_cast<int64_t>(stride_data[0]),
                                 static_cast<int64_t>(stride_data[1]));
    }
    else if constexpr (kRank == 3) {
        return cute::make_stride(static_cast<int64_t>(stride_data[0]),
                                 static_cast<int64_t>(stride_data[1]),
                                 static_cast<int64_t>(stride_data[2]));
    }
    else if constexpr (kRank == 4) {
        return cute::make_stride(static_cast<int64_t>(stride_data[0]),
                                 static_cast<int64_t>(stride_data[1]),
                                 static_cast<int64_t>(stride_data[2]),
                                 static_cast<int64_t>(stride_data[3]));
    }
    else if constexpr (kRank == 5) {
        return cute::make_stride(static_cast<int64_t>(stride_data[0]),
                                 static_cast<int64_t>(stride_data[1]),
                                 static_cast<int64_t>(stride_data[2]),
                                 static_cast<int64_t>(stride_data[3]),
                                 static_cast<int64_t>(stride_data[4]));
    }
    else if constexpr (kRank == 6) {
        return cute::make_stride(static_cast<int64_t>(stride_data[0]),
                                 static_cast<int64_t>(stride_data[1]),
                                 static_cast<int64_t>(stride_data[2]),
                                 static_cast<int64_t>(stride_data[3]),
                                 static_cast<int64_t>(stride_data[4]),
                                 static_cast<int64_t>(stride_data[5]));
    }
}

template<int kRank>
auto make_cute_layout(const ssize_t* shape, const ssize_t* stride)
{
    return cute::make_layout(make_cute_shape<kRank>(shape),
                             make_cute_stride<kRank>(stride));
}

}  // namespace detail

// ============================================================================
// CUDA kernel: GenericCopyKernel (CuTe TiledCopy)
// ============================================================================
namespace kernel {

// CTA-level tiling: blockIdx.x maps to one "row" of outer dimensions.
// TiledCopy distributes the innermost dimension across 256 threads.
// Each thread copies VecT-sized elements via CuTe's copy().
//
// Template params:
//   VecT       - vector type for load/store (uint4, uint2, uint, ushort, char)
//   SrcLayoutT - CuTe layout type for source tensor (dynamic rank-kRank layout)
//   DstLayoutT - CuTe layout type for dest tensor (dynamic rank-kRank layout)
template<typename VecT, typename SrcLayoutT, typename DstLayoutT>
__global__ void __launch_bounds__(256)
GenericCopyKernel(const VecT* __restrict__ src_ptr,
                  VecT* __restrict__       dst_ptr,
                  SrcLayoutT               src_layout,
                  DstLayoutT               dst_layout,
                  int64_t                  inner_size,
                  int64_t                  outer_total)
{
    constexpr int kRank        = cute::rank_v<SrcLayoutT>;
    constexpr int kBlockThreads = 256;

    // 1. Map CTA to outer-dim coordinates via blockIdx.x
    int64_t outer_idx = blockIdx.x;
    if (outer_idx >= outer_total) {
        return;
    }

    // 2. Extract shape/stride from CuTe layout into local arrays
    //    for runtime iteration in the outer-dim decomposition loop.
    //    CuTe tuple access requires compile-time indices, so we use
    //    cute::for_each with a compile-time index sequence.
    int32_t shape[kRank];
    int64_t src_strides[kRank];
    int64_t dst_strides[kRank];

    auto src_shape   = cute::shape(src_layout);
    auto src_stride  = cute::stride(src_layout);
    auto dst_stride  = cute::stride(dst_layout);

    cute::for_each(cute::make_seq<kRank>{}, [&](auto i) {
        shape[i]        = cute::get<decltype(i){}>(src_shape);
        src_strides[i]  = cute::get<decltype(i){}>(src_stride);
        dst_strides[i]  = cute::get<decltype(i){}>(dst_stride);
    });

    // 3. Decompose outer_idx into coordinates for dims 1..kRank-1
    //    and compute the base pointer offset for this CTA's slice.
    const VecT* my_src = src_ptr;
    VecT*       my_dst = dst_ptr;
    int64_t     rem    = outer_idx;

    PRAGMA_UNROLL
    for (int i = kRank - 1; i >= 1; --i) {
        int64_t c = rem % shape[i];
        rem /= shape[i];
        my_src += c * src_strides[i];
        my_dst += c * dst_strides[i];
    }

    // 4. Build TiledCopy for the innermost dimension
    //    Copy_Atom: basic VecT-sized load/store (SM70+ compatible)
    //    Thread layout: kBlockThreads threads along inner dim
    //    Value layout: 1 VecT per atom
    auto tiled_copy = make_tiled_copy(
        Copy_Atom<UniversalCopy<VecT>, VecT>{},
        make_layout(make_shape(cute::Int<kBlockThreads>{})),
        make_layout(make_shape(cute::Int<1>{})));

    auto thr_copy = tiled_copy.get_slice(threadIdx.x);

    // 5. Loop over inner-dim tiles of kBlockThreads elements each.
    //    Each tile creates a 1D CuTe tensor, partitions among threads,
    //    and calls copy().
    PRAGMA_UNROLL
    for (int64_t tile = 0; tile < inner_size; tile += kBlockThreads) {
        int64_t remaining = min(static_cast<int64_t>(kBlockThreads), inner_size - tile);

        // Create 1D CuTe tensors for this tile of the inner dim
        auto tile_src = make_tensor(
            my_src + tile * src_strides[0],
            make_layout(make_shape(remaining), make_stride(src_strides[0])));
        auto tile_dst = make_tensor(
            my_dst + tile * dst_strides[0],
            make_layout(make_shape(remaining), make_stride(dst_strides[0])));

        // Only threads within the tile bounds participate
        if (threadIdx.x < remaining) {
            auto thr_src = thr_copy.partition_S(tile_src);
            auto thr_dst = thr_copy.partition_D(tile_dst);
            copy(tiled_copy, thr_src, thr_dst);
        }
    }
}

}  // namespace kernel

// ============================================================================
// Host function: GenericCopy
// ============================================================================
void GenericCopy(const Tensor& src, Tensor& dst, cudaStream_t stream)
{
    auto a = src.layout();
    auto b = dst.layout();

    // Sort strides ascending so the innermost (fastest-varying) dim is first
    vector<int> idxs(a.rank());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [&](int i, int j) {  //
        return a.stride()[i] < a.stride()[j];
    });

    a = a.permute(idxs);
    b = b.permute(idxs);

    a = a.coalesce();
    b = b.coalesce();

    int rank = std::max(a.rank(), b.rank());

    if (a.rank() < rank) {
        a = a.view(b.shape());
    }
    else if (b.rank() < rank) {
        b = b.view(a.shape());
    }

    const DataType dtype = src.dtype();

    // --- Alignment detection ---
    int64_t alignment = 16;

    auto align = [&](auto v) { alignment = std::gcd(alignment, v); };

    // If the innermost dim is not stride-1, we can't vectorize along it
    if (a.stride(0) > 1 || b.stride(0) > 1) {
        alignment = byte_size(dtype);
    }

    align(byte_size(dtype, a.shape(0)));

    auto data_a = src.raw_data();
    auto data_b = dst.raw_data();

    align(reinterpret_cast<uintptr_t>(data_a));
    align(reinterpret_cast<uintptr_t>(data_b));

    for (int i = 1; i < rank; ++i) {
        align(byte_size(dtype, a.stride(i)));
        align(byte_size(dtype, b.stride(i)));
    }

    // --- Select vector type ---
    const int64_t vec_size = alignment / std::max<int64_t>(1, byte_size(dtype));

    // --- Launch kernel with CuTe layout parameters ---
    auto invoke = [&](auto vec_t, auto d) {
        using VecT    = decltype(vec_t);
        constexpr int kRank = d.value;

        // Build CuTe layouts from runtime shape/stride arrays
        auto src_layout = detail::make_cute_layout<kRank>(a.shape().data(), a.stride().data());
        auto dst_layout = detail::make_cute_layout<kRank>(a.shape().data(), b.stride().data());

        // Compute inner/outer split
        int64_t inner_size  = a.shape(0) / vec_size;
        int64_t outer_total = 1;
        for (int i = 1; i < rank; ++i) {
            outer_total *= a.shape(i);
        }

        // Adjust strides for vectorization (outer dims only)
        ssize_t src_stride_adj[kRank];
        ssize_t dst_stride_adj[kRank];
        std::copy_n(a.stride().data(), rank, src_stride_adj);
        std::copy_n(b.stride().data(), rank, dst_stride_adj);

        if (vec_size > 1) {
            for (int i = 1; i < rank; ++i) {
                src_stride_adj[i] /= vec_size;
                dst_stride_adj[i] /= vec_size;
            }
            // Also need the adjusted shape for dim 0
            ssize_t shape_adj[kRank];
            std::copy_n(a.shape().data(), rank, shape_adj);
            shape_adj[0] /= vec_size;
            src_layout = detail::make_cute_layout<kRank>(shape_adj, src_stride_adj);
            dst_layout = detail::make_cute_layout<kRank>(shape_adj, dst_stride_adj);
        }

        auto func = kernel::GenericCopyKernel<VecT, decltype(src_layout), decltype(dst_layout)>;

        // Grid: one CTA per outer-dim "row", block: fixed 256 threads
        int grid_size = static_cast<int>(outer_total);

        func<<<grid_size, 256, 0, stream>>>(
            reinterpret_cast<const VecT*>(data_a),
            reinterpret_cast<VecT*>(data_b),
            src_layout,
            dst_layout,
            inner_size,
            outer_total);
    };

    // Dispatch on exact rank (1-6, no padding)
    auto dispatch_rank = [&](auto vec_t) {
        switch (rank) {
            case 1: invoke(vec_t, constant<1>{}); break;
            case 2: invoke(vec_t, constant<2>{}); break;
            case 3: invoke(vec_t, constant<3>{}); break;
            case 4: invoke(vec_t, constant<4>{}); break;
            case 5: invoke(vec_t, constant<5>{}); break;
            case 6: invoke(vec_t, constant<6>{}); break;
            default: throw std::runtime_error("GenericCopy: rank > 6 not implemented");
        }
    };

    switch (alignment) {
        case 16: return dispatch_rank(uint4{});
        case 8:  return dispatch_rank(uint2{});
        case 4:  return dispatch_rank(uint{});
        case 2:  return dispatch_rank(ushort{});
        default: return dispatch_rank(char{});
    }
}

}  // namespace turbomind::core
```

Key design decisions:

1. **Kernel parameters**: Uses CuTe layout types (`SrcLayoutT`, `DstLayoutT`) instead of `Array<>`. The layout types are deduced from the arguments — the host constructs layouts using `detail::make_cute_layout<kRank>()`.

2. **Outer-dim decomposition**: Uses `cute::for_each(cute::make_seq<kRank>{}, ...)` to extract shape/stride from CuTe tuples into local arrays, then a standard runtime loop for the coordinate decomposition.

3. **Inner-dim TiledCopy**: Same as before — TiledCopy distributes inner dim across 256 threads, `copy()` performs the actual load/store.

4. **Host dispatcher**: Constructs CuTe layouts using if-constexpr helpers in `detail::make_cute_shape/make_stride/make_layout`. The kernel function pointer type uses `decltype(src_layout)` to match the CuTe layout type.

- [ ] **Step 2: Verify compilation**

Run: `cd /data/lmdeploy-copy/build && ninja core 2>&1 | tail -40`

Expected: Compilation succeeds. There may be CuTe-related warnings but no errors.

Common issues to fix if compilation fails:
- **`cute::make_tuple` ambiguity**: If CuTe's `make_tuple` conflicts with `std::make_tuple`, add `using cute::make_tuple;` or qualify all CuTe calls with `cute::`.
- **Missing includes**: If CuTe headers can't be found, verify that `build/_deps/repo-cutlass-src/include/` exists and the `nvidia::cutlass::cutlass` target is resolving correctly.
- **`UniversalCopy` not found**: This is in `cute/arch/copy.hpp` which is transitively included by `cute/algorithm/copy.hpp`.
- **`PRAGMA_UNROLL` not found**: This macro is defined in the existing codebase's headers. If CuTe headers don't see it, add `#pragma unroll` directly.
- **`cute::for_each` not found**: Included transitively via `cute/layout.hpp`.
- **`cute::make_seq` not found**: May be `cute::make_integer_sequence` or in a different header. Check CuTe headers for the correct name — it might be `cute::seq<kRank>` or similar.

- [ ] **Step 3: Fix any compilation errors**

Iterate: modify code, rebuild with `ninja core`, until clean.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "feat(core): rewrite GenericCopy kernel with CuTe TiledCopy"
```

---

### Task 3: Full build verification

**Files:** None (verification only)

- [ ] **Step 1: Build the _turbomind extension to check downstream link**

Run: `cd /data/lmdeploy-copy/build && ninja _turbomind 2>&1 | tail -50`

Expected: Full build succeeds. If there are link errors about missing CuTE/CUTLASS symbols, verify the `nvidia::cutlass::cutlass` target propagation. CUTLASS is headers-only so link errors indicate a cmake configuration issue, not a real link issue.

- [ ] **Step 2: Fix any remaining build issues**

If CuTe headers cause compile errors in other files that include tensor.h (because the dependency propagates via `PUBLIC`), check that the CUTLASS include path is correctly set. The root CMakeLists.txt already adds it to `COMMON_HEADER_DIRS`.

---

### Task 4: Functional and performance testing

**Files:**
- Existing: `test_generic_copy.py`

- [ ] **Step 1: Check GPU availability**

Run: `python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"`

Expected: `True` and GPU name.

- [ ] **Step 2: Run the existing GenericCopy test suite**

Run: `cd /data/lmdeploy-copy && python3 test_generic_copy.py 2>&1`

Expected: ALL TESTS PASSED. Every test case must show PASS. The test suite covers:
- Contiguous baseline (f32)
- 2D: transpose, row-stride, col-stride, narrow
- 3D: permute
- 4D: slice
- Combined: slice+transpose
- Dtype sweep: f16, i8, i32 (transpose)
- Throughput sweep: 1K to 256M elements (contiguous + transpose)

- [ ] **Step 3: Verify throughput numbers**

The test output shows throughput in GB/s for each test. Check that:
- Contiguous copies are at or near the current kernel's throughput (within 10%)
- Transpose copies show throughput comparable to the current kernel
- No test shows significantly degraded throughput (>20% regression)

If throughput regresses significantly, the tile loop or TiledCopy overhead may be the cause. Investigate by:
1. Checking if the inner_size is very small (< 256) causing wasted threads
2. Checking if the tile loop overhead (branch + tensor creation) dominates for large inner_size
3. Comparing PTX output between old and new kernels

- [ ] **Step 4: Fix any correctness or performance issues**

Iterate: modify code, rebuild (`ninja core`), re-run tests until all pass with acceptable throughput.

- [ ] **Step 5: Final commit if any fixes were needed**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "fix(core): address GenericCopy TiledCopy issues from testing"
```
