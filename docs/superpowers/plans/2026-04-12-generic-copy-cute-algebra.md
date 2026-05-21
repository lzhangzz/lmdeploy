# GenericCopy CuTe Shape Algebra Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace manual index arithmetic with CuTe layout algebra (group, slicing, zipped_divide, copy_if) and switch to a 2D grid architecture. Remove VecT dispatch, use raw data type.

**Architecture:** 2D grid where blockIdx.x = inner tile, blockIdx.y = outer row. Each CTA handles one (256)-element tile. No inner loop. CuTe tensors + group + zipped_divide for all indexing. Identity tensor + copy_if for predication.

**Tech Stack:** CuTe (CUTLASS v3.9.2 headers-only), CUDA, existing `turbomind::core::Tensor/Layout` infrastructure.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/turbomind/core/tensor.cu` | CuTe layout algebra GenericCopy kernel + simplified host dispatcher |

---

### Task 1: Rewrite GenericCopy kernel and host dispatcher in tensor.cu

**Files:**
- Modify: `src/turbomind/core/tensor.cu` (replace entire file contents)

This task replaces the entire file. The `detail::make_cute_shape/stride/layout` helpers are kept unchanged. The kernel and host `GenericCopy` are rewritten.

- [ ] **Step 1: Replace tensor.cu with CuTe layout algebra implementation**

Write the following as the complete contents of `src/turbomind/core/tensor.cu`:

```cpp
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/core/meta.h"

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>

#include <algorithm>
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
// CUDA kernel: GenericCopyKernel (CuTe layout algebra)
// ============================================================================
namespace kernel {

template<typename T, typename SrcLayoutT, typename DstLayoutT>
__global__ void __launch_bounds__(256)
GenericCopyKernel(const T* __restrict__ src_ptr,
                  T* __restrict__       dst_ptr,
                  SrcLayoutT             src_layout,
                  DstLayoutT             dst_layout)
{
    constexpr int kBlockThreads = 256;
    constexpr int kRank         = cute::rank_v<SrcLayoutT>;

    // 1. Create CuTe tensors from pointers + layouts
    auto gSrc = cute::make_tensor(cute::make_gmem_ptr(src_ptr), src_layout);
    auto gDst = cute::make_tensor(cute::make_gmem_ptr(dst_ptr), dst_layout);

    // Build TiledCopy: 256 threads, 1 element per thread
    auto tiled_copy = cute::make_tiled_copy(
        cute::Copy_Atom<cute::UniversalCopy<T>, T>{},
        cute::make_layout(cute::make_shape(cute::Int<kBlockThreads>{})),
        cute::make_layout(cute::make_shape(cute::Int<1>{})));
    auto thr_copy = tiled_copy.get_slice(threadIdx.x);

    if constexpr (kRank == 1) {
        // No outer dims — the whole tensor is the inner dim
        if (blockIdx.y > 0) return;

        // 2. Tile the inner dim with zipped_divide
        auto tiler    = cute::Int<kBlockThreads>{};
        auto tiledSrc = cute::zipped_divide(gSrc, tiler);
        auto tiledDst = cute::zipped_divide(gDst, tiler);

        if (blockIdx.x >= cute::size<1>(tiledSrc)) return;

        auto ctaSrc = tiledSrc(_, blockIdx.x);
        auto ctaDst = tiledDst(_, blockIdx.x);

        // 3. Predication via identity tensor
        auto id_row   = cute::make_identity_tensor(cute::shape(gSrc));
        auto id_tiled = cute::zipped_divide(id_row, tiler);
        auto tile_id  = id_tiled(_, blockIdx.x);

        auto thrSrc = thr_copy.partition_S(ctaSrc);
        auto thrDst = thr_copy.partition_D(ctaDst);
        auto thrId  = thr_copy.partition_S(tile_id);

        auto pred = cute::make_tensor<bool>(cute::shape(thrSrc));
        PRAGMA_UNROLL
        for (int i = 0; i < cute::size(pred); ++i) {
            pred(i) = cute::get<0>(thrId(i)) < cute::size(gSrc);
        }

        cute::copy_if(pred, thrSrc, thrDst);
    }
    else {
        // 2. Group outer dims (modes 1..kRank-1) into a single mode
        //    rank-k layout -> rank-2 (inner, outer_flat)
        auto src_layout_g = cute::group<1, kRank>(src_layout);
        auto dst_layout_g = cute::group<1, kRank>(dst_layout);
        auto gSrc_g = cute::make_tensor(cute::make_gmem_ptr(src_ptr), src_layout_g);
        auto gDst_g = cute::make_tensor(cute::make_gmem_ptr(dst_ptr), dst_layout_g);

        if (blockIdx.y >= cute::size<1>(gSrc_g)) return;

        // 3. Slice to get this CTA's 1D inner tensor
        auto rowSrc = gSrc_g(_, blockIdx.y);
        auto rowDst = gDst_g(_, blockIdx.y);

        // 4. Tile the inner dim with zipped_divide
        auto tiler    = cute::Int<kBlockThreads>{};
        auto tiledSrc = cute::zipped_divide(rowSrc, tiler);
        auto tiledDst = cute::zipped_divide(rowDst, tiler);

        if (blockIdx.x >= cute::size<1>(tiledSrc)) return;

        auto ctaSrc = tiledSrc(_, blockIdx.x);
        auto ctaDst = tiledDst(_, blockIdx.x);

        // 5. Predication via identity tensor
        auto id_row   = cute::make_identity_tensor(cute::shape(rowSrc));
        auto id_tiled = cute::zipped_divide(id_row, tiler);
        auto tile_id  = id_tiled(_, blockIdx.x);

        auto thrSrc = thr_copy.partition_S(ctaSrc);
        auto thrDst = thr_copy.partition_D(ctaDst);
        auto thrId  = thr_copy.partition_S(tile_id);

        auto pred = cute::make_tensor<bool>(cute::shape(thrSrc));
        PRAGMA_UNROLL
        for (int i = 0; i < cute::size(pred); ++i) {
            pred(i) = cute::get<0>(thrId(i)) < cute::size(rowSrc);
        }

        cute::copy_if(pred, thrSrc, thrDst);
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

    TM_CHECK_EQ(a.size(), b.size()) << "GenericCopy: src and dst must have the same number of elements";

    // Sort strides ascending so innermost (fastest-varying) dim is first
    vector<int> idxs(a.rank());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [&](int i, int j) {
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
    constexpr int  kBlockThreads = 256;

    // Dispatch on data type T
    auto dispatch_dtype = [&](auto t) {
        using T = decltype(t);

        auto invoke = [&](auto d) {
            constexpr int kRank = d.value;

            // Construct CuTe layouts directly (no vec_size adjustment)
            auto src_layout = detail::make_cute_layout<kRank>(a.shape().data(), a.stride().data());
            auto dst_layout = detail::make_cute_layout<kRank>(a.shape().data(), b.stride().data());

            // Compute 2D grid: (num_inner_tiles, outer_total)
            int64_t inner_size = a.shape(0);
            int64_t outer_total = 1;
            for (int i = 1; i < rank; ++i) {
                outer_total *= a.shape(i);
            }

            int64_t num_inner_tiles = (inner_size + kBlockThreads - 1) / kBlockThreads;
            dim3 grid(static_cast<uint32_t>(num_inner_tiles),
                      static_cast<uint32_t>(outer_total));

            auto func = kernel::GenericCopyKernel<T, decltype(src_layout), decltype(dst_layout)>;

            func<<<grid, 256, 0, stream>>>(
                reinterpret_cast<const T*>(src.raw_data()),
                reinterpret_cast<T*>(dst.raw_data()),
                src_layout,
                dst_layout);
        };

        // Dispatch on exact rank (1-6)
        switch (rank) {
            case 1: invoke(constant<1>{}); break;
            case 2: invoke(constant<2>{}); break;
            case 3: invoke(constant<3>{}); break;
            case 4: invoke(constant<4>{}); break;
            case 5: invoke(constant<5>{}); break;
            case 6: invoke(constant<6>{}); break;
            default: throw std::runtime_error("GenericCopy: rank > 6 not implemented");
        }
    };

    // Dispatch on data type
    switch (dtype) {
        case DataType::kFloat32:  return dispatch_dtype(float{});
        case DataType::kFloat16:  return dispatch_dtype(half_t{});
        case DataType::kBfloat16: return dispatch_dtype(bfloat16_t{});
        case DataType::kInt8:     return dispatch_dtype(int8_t{});
        case DataType::kInt32:    return dispatch_dtype(int32_t{});
        case DataType::kBool:     return dispatch_dtype(bool{});
        case DataType::kUint8:    return dispatch_dtype(uint8_t{});
        default:
            throw std::runtime_error("GenericCopy: unsupported data type " + to_string(dtype));
    }
}

}  // namespace turbomind::core
```

Key design decisions:

1. **Kernel signature**: `template<typename T, typename SrcLayoutT, typename DstLayoutT>` — uses the raw data type T, not VecT. Removed `inner_size` and `outer_total` parameters.

2. **TiledCopy built once**: Moved before the `if constexpr` branch since it's the same for both paths.

3. **group<1, kRank> on layout, not tensor**: CuTe's `group` operates on layouts. We apply it to the layout and create a new tensor with the grouped layout.

4. **Identity tensor predication**: `make_identity_tensor` + `zipped_divide` + `partition_S` gives each thread its original coordinate. Predicate checks `coord < size(rowSrc)`.

5. **Host simplification**: Removed alignment detection, vec_size computation, stride/shape adjustment, and VecT dispatch. Added dtype dispatch with the `TM_DISPATCH_DTYPES`-style switch.

6. **copy_if overload**: Uses the basic `copy_if(pred, src, dst)` overload (line 49 of copy.hpp) which iterates element-by-element with predication. This works on the partitioned per-thread tensors.

- [ ] **Step 2: Build the core target**

Run: `cd /data/lmdeploy-copy/build && ninja core 2>&1 | tail -40`

Expected: Compilation succeeds. There may be CuTe-related warnings but no errors.

Common issues to fix if compilation fails:

- **`cute::group` not found**: Verify `#include <cute/layout.hpp>` is present. `group` is defined in `layout.hpp` and `layout_composed.hpp`.

- **`cute::make_identity_tensor` not found**: It's in `tensor_impl.hpp`, included transitively via `cute/tensor.hpp`.

- **`CUTE_UNROLL` not found**: Replace with `PRAGMA_UNROLL` from the codebase's `kernels/core/common.h` (included via `meta.h`).

- **`cute::make_gmem_ptr` with `const T*`**: Should work — CuTe deduces const-ness. If issues, try `cute::make_gmem_ptr<const T>(src_ptr)`.

- **`group<1, kRank>` with kRank=1**: Guarded by `if constexpr (kRank == 1)` which uses a separate branch. If the compiler still tries to instantiate `group<1,1>`, move the group call behind an explicit `if constexpr (kRank >= 2)`.

- **`copy_if` ambiguity**: If CuTe can't resolve the overload, explicitly call the 3-argument version: `cute::copy_if(pred, thrSrc, thrDst)` (without `tiled_copy`).

- [ ] **Step 3: Fix any compilation errors**

Iterate: modify code, rebuild with `ninja core`, until clean.

- [ ] **Step 4: Build the _turbomind extension**

Run: `cd /data/lmdeploy-copy/build && ninja _turbomind 2>&1 | tail -30`

Expected: Full build succeeds. If there are link errors about missing CuTe/CUTLASS symbols, verify the `nvidia::cutlass::cutlass` target propagation in `src/turbomind/core/CMakeLists.txt`.

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "refactor(core): rewrite GenericCopy with CuTe layout algebra (group, zipped_divide, copy_if)

- Replace manual shape/stride extraction with CuTe tensors throughout
- Replace manual outer-dim decomposition with group<1,k> + slicing
- Replace manual tile loop with zipped_divide + 2D grid
- Replace if (threadIdx.x < remaining) with copy_if predication
- Remove VecT dispatch; use raw data type T
- Remove alignment detection and vec_size computation

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Functional testing

**Files:**
- Existing: `test_generic_copy.py`

- [ ] **Step 1: Check GPU availability**

Run: `python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"`

Expected: `True` and GPU name.

- [ ] **Step 2: Run the existing GenericCopy test suite**

Run: `cd /data/lmdeploy-copy && PYTHONPATH=lmdeploy:build/lib python3 test_generic_copy.py 2>&1`

Expected: ALL TESTS PASSED. Every test case must show PASS. The test suite covers:
- Contiguous baseline (f32)
- 2D: transpose, row-stride, col-stride, narrow
- 3D: permute (2,0,1)
- 4D: slice
- Combined: slice+transpose
- Dtype sweep: f16, i8, i32 (transpose)
- Throughput sweep: 1K to 256M elements (contiguous + transpose)
- Negative strides (flip) — may be SKIP

- [ ] **Step 3: Fix any correctness issues**

If tests fail, the most likely causes are:

1. **Data type mismatch**: The test creates tensors with specific dtypes (f32, f16, i8, i32). Verify the dtype dispatch covers all tested types. Check that `reinterpret_cast<T*>(raw_data)` is correct for each type.

2. **Grid overflow**: `dim3` truncates to `uint32_t`. If `num_inner_tiles` or `outer_total` exceed `UINT32_MAX`, the grid will be wrong. Add bounds checks if needed.

3. **zipped_divide rounding**: The last tile of each row is rounded up by `zipped_divide`. The predication via identity tensor should handle this. Verify the predicate is correct by checking that `get<0>(thrId(i))` gives the expected coordinate.

4. **group<1,k> coordinate mapping**: For rank-3+ tensors, verify that `group<1,k>` correctly flattens the outer dims and that `blockIdx.y` maps to the right outer coordinates.

Iterate: modify code, rebuild (`ninja core && ninja _turbomind`), re-run tests until all pass.

- [ ] **Step 4: Commit any fixes**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "fix(core): address GenericCopy CuTe layout algebra issues from testing

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Performance validation

**Files:**
- Existing: `test_generic_copy.py` (includes throughput benchmarking)

- [ ] **Step 1: Compare throughput numbers**

The test output shows throughput in GB/s for each test. Compare the new numbers against what's expected:

- **Contiguous copies**: Throughput may be lower than the previous VecT approach due to removing 16-byte vectorization. For f32 contiguous copies, expect ~250-400 GB/s (vs. ~900+ GB/s with uint4). The 2D grid compensates somewhat.

- **Transpose copies**: Throughput should be similar to before (~100-300 GB/s for f32 transpose), since both approaches do element-by-element access for non-contiguous layouts.

- **Small tensors**: No significant change expected.

- [ ] **Step 2: Address any unexpected regressions**

If throughput regresses more than expected:

1. Check if the TiledCopy + copy_if overhead is significant for small tiles
2. Verify that the 2D grid isn't causing excessive CTAs for small tensors
3. Consider if `copy_if` (which iterates) is slower than `copy` for the common case

If regression is unacceptable, vectorization can be reintroduced through CuTe's TiledCopy `val_layout` parameter in a follow-up.

- [ ] **Step 3: Final commit if any performance fixes were needed**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "perf(core): optimize GenericCopy CuTe layout algebra performance

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
