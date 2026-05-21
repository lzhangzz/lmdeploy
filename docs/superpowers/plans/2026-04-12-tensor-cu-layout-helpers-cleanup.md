# tensor.cu Layout Helpers Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ~155 lines of `if constexpr` boilerplate in `tensor.cu`'s `detail` namespace with ~35 lines of C++17 index_sequence pack expansion.

**Architecture:** Each of the three helper functions (`make_cute_shape`, `make_cute_stride`, `make_cute_layout_unit_inner`) currently has 6 `if constexpr` branches for ranks 1-6. Replace each with a pair of functions: an `_impl` helper that takes `std::index_sequence<Is...>` and expands the pack, plus a 1-line wrapper that calls it with `std::make_index_sequence<kRank>{}`.

**Tech Stack:** C++17 (`std::index_sequence`, `std::make_index_sequence`), CuTe (`cute::make_shape`, `cute::make_stride`), CUDA.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/turbomind/core/tensor.cu` | Only file modified. Replace `detail` namespace helpers (lines 20-172) |

---

### Task 1: Replace layout helpers with index_sequence versions

**Files:**
- Modify: `src/turbomind/core/tensor.cu`

- [ ] **Step 1: Add `#include <utility>` for `std::index_sequence`**

In `src/turbomind/core/tensor.cu`, add `#include <utility>` after the existing `#include <string>` on line 11. The includes section (lines 1-11) becomes:

```cpp
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/core/meta.h"

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>

#include <algorithm>
#include <numeric>
#include <string>
#include <utility>
```

- [ ] **Step 2: Replace the entire `detail` namespace (lines 20-172)**

Replace everything from `namespace detail {` (line 20) through `}  // namespace detail` (line 172) with:

```cpp
namespace detail {

template<size_t... Is>
auto make_cute_shape_impl(const ssize_t* data, std::index_sequence<Is...>)
{
    return cute::make_shape(static_cast<int32_t>(data[Is])...);
}

template<int kRank>
auto make_cute_shape(const ssize_t* data)
{
    return make_cute_shape_impl(data, std::make_index_sequence<kRank>{});
}

template<size_t... Is>
auto make_cute_stride_impl(const ssize_t* data, std::index_sequence<Is...>)
{
    return cute::make_stride(static_cast<int64_t>(data[Is])...);
}

template<int kRank>
auto make_cute_stride(const ssize_t* data)
{
    return make_cute_stride_impl(data, std::make_index_sequence<kRank>{});
}

template<int kRank>
auto make_cute_layout(const ssize_t* shape, const ssize_t* stride)
{
    return cute::make_layout(make_cute_shape<kRank>(shape),
                             make_cute_stride<kRank>(stride));
}

// Layout with compile-time Int<1> inner stride — needed for CuTe's recast
// in wide Copy_Atom (vectorized path). Only valid when inner stride == 1.
template<size_t... Is>
auto make_unit_inner_stride_impl(const ssize_t* stride, std::index_sequence<Is...>)
{
    return cute::make_stride(cute::Int<1>{}, static_cast<int64_t>(stride[Is + 1])...);
}

template<int kRank>
auto make_cute_layout_unit_inner(const ssize_t* shape, const ssize_t* stride)
{
    return cute::make_layout(
        make_cute_shape<kRank>(shape),
        make_unit_inner_stride_impl(stride, std::make_index_sequence<kRank - 1>{}));
}

}  // namespace detail
```

This replaces ~153 lines with ~46 lines. Every function below the `detail` namespace (kernels, host dispatch) is unchanged and will continue to call these helpers the same way.

- [ ] **Step 3: Build the core target**

Run: `cd /data/lmdeploy-copy/build && ninja core 2>&1 | tail -20`

Expected: Compilation succeeds. The index_sequence pack expansion produces identical template instantiations to the old `if constexpr` branches. CuTe's `make_shape`/`make_stride` accept variadic args, so pack expansion works directly.

If compilation fails with `std::index_sequence` not found, verify `#include <utility>` was added in Step 1.

- [ ] **Step 4: Build the _turbomind extension**

Run: `cd /data/lmdeploy-copy/build && ninja _turbomind 2>&1 | tail -20`

Expected: Full build succeeds.

- [ ] **Step 5: Run the GenericCopy test suite**

Run: `cd /data/lmdeploy-copy && PYTHONPATH=lmdeploy:build/lib python3 test_generic_copy.py 2>&1`

Expected: ALL TESTS PASSED (22 tests). Every test must show PASS. This verifies that the layout helpers produce identical CuTe layouts for all ranks 1-4 (contiguous 1D, 2D transpose, 3D permute, 4D slice) and all data types (f32, f16, i8, i32, u8).

- [ ] **Step 6: Commit**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "refactor(core): replace layout helper if-constexpr chains with index_sequence

- make_cute_shape, make_cute_stride: 6-branch if constexpr -> index_sequence pack expansion
- make_cute_layout_unit_inner: same, with Int<1> first stride and Is+1 offset for rest
- Net reduction: ~155 lines -> ~46 lines, identical generated code

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
