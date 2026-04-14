# CuTe-First Layout Algebra Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all `core::Layout` algebra in `GenericCopy` with array-based normalization, building `cute::Tensor` as early as possible (after rank is known).

**Architecture:** Extract shape/stride arrays from `core::Tensor`, normalize using array operations (pad, stride-sort, coalesce, rank-match), then dispatch on effective rank to build `cute::Tensor` and launch kernels. No `core::Layout::permute/coalesce/view` calls remain in `GenericCopy`.

**Tech Stack:** CuTe (CUTLASS), CUDA, C++17

---

### Task 1: Add array-based layout normalization helpers

**Files:**
- Modify: `src/turbomind/core/tensor.cu` (add helpers to `detail` namespace, lines 24–73)

Add four helper functions to the `detail` namespace. These replace `core::Layout::permute`, `coalesce`, and `view`.

- [ ] **Step 1: Add `pad_arrays` helper**

Extract shape/stride from a `core::Layout`, copy into `std::array<ssize_t, kMax>` with trailing `(1, 0)` padding.

```cpp
// Pad shape/stride from a core::Layout into fixed-size arrays.
// Modes are preserved in their original order; missing modes padded with (1, 0).
template<int kMax>
void pad_arrays(const core::Layout& layout,
                std::array<ssize_t, kMax>& shape,
                std::array<ssize_t, kMax>& stride)
{
    shape.fill(1);
    stride.fill(0);
    const int rank = layout.rank();
    for (int i = 0; i < rank; ++i) {
        shape[i]  = layout.shape(i);
        stride[i] = layout.stride(i);
    }
}
```

- [ ] **Step 2: Add `sort_by_stride` helper**

Sort all four arrays (src shape/stride, dst shape/stride) by ascending src stride. Same permutation applied to both src and dst. After this call, index 0 holds the innermost (smallest stride) dimension, matching CuTe convention.

```cpp
// Sort modes by ascending src stride. Same permutation applied to dst.
// After sorting, index 0 = innermost (CuTe convention).
template<int kMax>
void sort_by_stride(std::array<ssize_t, kMax>& src_shape,
                    std::array<ssize_t, kMax>& src_stride,
                    std::array<ssize_t, kMax>& dst_shape,
                    std::array<ssize_t, kMax>& dst_stride)
{
    int idx[kMax];
    std::iota(idx, idx + kMax, 0);
    std::sort(idx, idx + kMax, [&](int a, int b) {
        return src_stride[a] < src_stride[b];
    });

    auto apply_perm = [&](auto& arr) {
        std::array<ssize_t, kMax> tmp;
        for (int i = 0; i < kMax; ++i) {
            tmp[i] = arr[idx[i]];
        }
        arr = tmp;
    };

    apply_perm(src_shape);
    apply_perm(src_stride);
    apply_perm(dst_shape);
    apply_perm(dst_stride);
}
```

- [ ] **Step 3: Add `coalesce_ascending` helper**

Merge adjacent contiguous modes in ascending-stride order. Returns effective rank. The merge condition is `next_stride == acc_shape * acc_stride` (the next mode starts exactly where the accumulated mode ends).

```cpp
// Coalesce adjacent contiguous modes in ascending-stride order.
// Returns effective rank (number of non-trivial modes after merging).
// Size-1 modes are skipped. Contiguous modes are merged by multiplying
// shapes and keeping the inner (smaller) stride.
template<int kMax>
int coalesce_ascending(std::array<ssize_t, kMax>& shape,
                       std::array<ssize_t, kMax>& stride)
{
    std::array<ssize_t, kMax> out_shape{};
    std::array<ssize_t, kMax> out_stride{};
    int n = 0;

    for (int i = 0; i < kMax; ++i) {
        if (shape[i] == 1) {
            continue;
        }
        if (n == 0) {
            out_shape[0]  = shape[i];
            out_stride[0] = stride[i];
            n = 1;
        }
        else if (stride[i] == out_shape[n - 1] * out_stride[n - 1]) {
            // Contiguous: merge with previous mode
            out_shape[n - 1] *= shape[i];
            // out_stride[n-1] stays (inner stride)
        }
        else {
            out_shape[n]  = shape[i];
            out_stride[n] = stride[i];
            ++n;
        }
    }

    shape  = out_shape;
    stride = out_stride;
    return n;
}
```

- [ ] **Step 4: Add `reshape_to` helper**

Reshape a layout (given as arrays) to match a target shape. Walks from innermost to outermost, consuming input modes. Fails (TM_CHECK) if splitting would cross a non-contiguous boundary. This replaces `core::Layout::view` for the rank-match step.

```cpp
// Reshape (in_shape, in_stride) with in_rank modes to match target_shape.
// Walks from innermost (index 0) to outermost. Input must be coalesced.
// Fails if the reshape would split across a non-contiguous boundary.
template<int kMax>
void reshape_to(int in_rank,
                const std::array<ssize_t, kMax>& in_shape,
                const std::array<ssize_t, kMax>& in_stride,
                int out_rank,
                const std::array<ssize_t, kMax>& out_shape,
                std::array<ssize_t, kMax>& out_stride)
{
    int     p = 0;   // current input mode index
    ssize_t s = 1;   // remaining elements in current input mode
    ssize_t d = 0;   // stride accumulator

    for (int i = 0; i < out_rank; ++i) {
        if (out_shape[i] == 1) {
            out_stride[i] = 0;
        }
        else {
            if (s == 1) {
                TM_CHECK_LT(p, in_rank);
                s = in_shape[p];
                d = in_stride[p];
                ++p;
            }
            TM_CHECK_EQ(s % out_shape[i], 0)
                << "reshape_to: splitting across non-contiguous boundary";
            out_stride[i] = d;
            d *= out_shape[i];
            s /= out_shape[i];
        }
    }
}
```

- [ ] **Step 5: Commit helpers**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "refactor(core): add array-based layout normalization helpers for GenericCopy"
```

---

### Task 2: Rewrite GenericCopy host function

**Files:**
- Modify: `src/turbomind/core/tensor.cu` (lines 199–441, the `GenericCopy` function)

Rewrite `GenericCopy` to use the new array-based helpers instead of `core::Layout` algebra. The kernel code (`CopyKernelND`, `TransposeCopyKernel`) is unchanged.

- [ ] **Step 1: Replace the function body of `GenericCopy`**

Replace everything from line 199 (`void GenericCopy(...) {`) through line 441 (`}` — end of function) with the following. The `#include`s, `namespace`, `detail` helpers, and kernel code are untouched.

```cpp
void GenericCopy(const Tensor& src, Tensor& dst, cudaStream_t stream)
{
    constexpr int kMaxRank = 6;

    TM_CHECK_EQ(src.layout().size(), dst.layout().size())
        << "GenericCopy: src and dst must have the same number of elements";

    // --- 1. Extract, pad, sort, coalesce ---
    std::array<ssize_t, kMaxRank> src_shape{}, src_stride{};
    std::array<ssize_t, kMaxRank> dst_shape{}, dst_stride{};

    detail::pad_arrays<kMaxRank>(src.layout(), src_shape, src_stride);
    detail::pad_arrays<kMaxRank>(dst.layout(), dst_shape, dst_stride);

    detail::sort_by_stride<kMaxRank>(src_shape, src_stride, dst_shape, dst_stride);

    int src_rank = detail::coalesce_ascending<kMaxRank>(src_shape, src_stride);
    int dst_rank = detail::coalesce_ascending<kMaxRank>(dst_shape, dst_stride);

    // --- 2. Rank-match: reshape lower-rank to higher-rank ---
    int rank = std::max(src_rank, dst_rank);

    if (src_rank < rank) {
        std::array<ssize_t, kMaxRank> new_src_stride{};
        detail::reshape_to<kMaxRank>(src_rank, src_shape, src_stride,
                                     rank, dst_shape, new_src_stride);
        src_stride = new_src_stride;
        src_shape  = dst_shape;  // shape now matches
        src_rank   = rank;
    }
    else if (dst_rank < rank) {
        std::array<ssize_t, kMaxRank> new_dst_stride{};
        detail::reshape_to<kMaxRank>(dst_rank, dst_shape, dst_stride,
                                     rank, src_shape, new_dst_stride);
        dst_stride = new_dst_stride;
        dst_shape  = src_shape;  // shape now matches
        dst_rank   = rank;
    }

    TM_CHECK_EQ(src_shape, dst_shape) << "GenericCopy: shapes don't match after rank-match";

    const DataType dtype       = src.dtype();
    constexpr int  kBlockThreads = 256;

    auto data_a = src.raw_data();
    auto data_b = dst.raw_data();

    // --- 3. 2D transpose detection ---
    constexpr int kTileDim = 32;
    bool is_2d_transpose =
        (rank == 2) && (src_stride[0] == 1) && (dst_stride[1] == 1)
        && (src_stride[1] > 1) && (dst_stride[0] > 1)
        && (src_shape[0] % kTileDim == 0) && (src_shape[1] % kTileDim == 0);

    if (is_2d_transpose) {
        int32_t M = static_cast<int32_t>(src_shape[0]);
        int32_t N = static_cast<int32_t>(src_shape[1]);
        dim3 grid(static_cast<uint32_t>(N / kTileDim),
                  static_cast<uint32_t>(M / kTileDim));

        auto tr_dispatch = [&](auto t) {
            using T         = decltype(t);
            auto src_gmem = make_tensor(
                make_gmem_ptr(reinterpret_cast<const T*>(data_a)),
                make_layout(make_shape(M, N),
                            make_stride(Int<1>{}, src_stride[1])));
            auto dst_gmem = make_tensor(
                make_gmem_ptr(reinterpret_cast<T*>(data_b)),
                make_layout(make_shape(M, N),
                            make_stride(dst_stride[0], Int<1>{})));

            kernel::TransposeCopyKernel<kTileDim, 8 * sizeof(T)>
                <<<grid, 256, 0, stream>>>(src_gmem, dst_gmem);
        };

        switch (byte_size(dtype)) {
            case 1: return tr_dispatch(uint8_t{});
            case 2: return tr_dispatch(uint16_t{});
            case 4: return tr_dispatch(uint32_t{});
            case 8: return tr_dispatch(uint64_t{});
            default:
                TM_CHECK(0) << "GenericCopy: unsupported element size " << byte_size(dtype);
                break;
        }
    }

    // --- 4. Alignment detection ---
    int64_t alignment = 16;
    auto     align    = [&](auto v) { alignment = std::gcd(alignment, v); };

    if (src_stride[0] > 1 || dst_stride[0] > 1) {
        alignment = byte_size(dtype);
    }

    align(byte_size(dtype, src_shape[0]));
    align(reinterpret_cast<uintptr_t>(data_a));
    align(reinterpret_cast<uintptr_t>(data_b));

    for (int i = 1; i < rank; ++i) {
        align(byte_size(dtype, src_stride[i]));
        align(byte_size(dtype, dst_stride[i]));
    }

    // --- 5. vec_size computation ---
    const int elem_size = byte_size(dtype);
    int       vec_size  = static_cast<int>(alignment / std::max<int64_t>(1, elem_size));

    if (vec_size * elem_size > 16) {
        vec_size = 16 / elem_size;
    }

    while (vec_size > 1
           && src_shape[0] % (static_cast<int64_t>(vec_size) * kBlockThreads) != 0) {
        vec_size /= 2;
    }

    // --- 6. Dispatch on data type T, vec_size kVec, and rank kRank ---
    auto dispatch_elem_size = [&](auto t) {
        using T           = decltype(t);
        constexpr int kElemBits = sizeof_bits_v<T>;

        auto dispatch_vec = [&](auto v) {
            constexpr int kVec = v.value;

            if constexpr (kVec * kElemBits <= 128) {

            auto invoke_nd = [&](auto d) {
                constexpr int kRank        = d.value;
                constexpr int kCopyThreads = kBlockThreads / kVec;

                // Build CuTe layouts from normalized arrays
                auto src_layout = [&] {
                    if constexpr (kVec > 1)
                        return detail::make_cute_layout_unit_inner<kRank>(
                            src_shape.data(), src_stride.data());
                    else
                        return detail::make_cute_layout<kRank>(
                            src_shape.data(), src_stride.data());
                }();

                auto dst_layout = [&] {
                    if constexpr (kVec > 1)
                        return detail::make_cute_layout_unit_inner<kRank>(
                            src_shape.data(), dst_stride.data());
                    else
                        return detail::make_cute_layout<kRank>(
                            src_shape.data(), dst_stride.data());
                }();

                auto src_gmem = make_tensor(
                    make_gmem_ptr(reinterpret_cast<const T*>(data_a)), src_layout);
                auto dst_gmem = make_tensor(
                    make_gmem_ptr(reinterpret_cast<T*>(data_b)), dst_layout);

                auto src_grouped = [&] {
                    if constexpr (kRank > 1) return group_modes<1, kRank>(src_gmem);
                    else return make_tensor(src_gmem.data(),
                        make_layout(make_shape(src_gmem.size(), Int<1>{}),
                                    make_stride(stride<0>(src_gmem.layout()), Int<0>{})));
                }();
                auto dst_grouped = [&] {
                    if constexpr (kRank > 1) return group_modes<1, kRank>(dst_gmem);
                    else return make_tensor(dst_gmem.data(),
                        make_layout(make_shape(dst_gmem.size(), Int<1>{}),
                                    make_stride(stride<0>(dst_gmem.layout()), Int<0>{})));
                }();

                auto make_tiled_3d = [&](auto tensor_2d) {
                    auto inner_size_val   = size<0>(tensor_2d);
                    auto inner_stride_val = stride<0>(tensor_2d.layout());
                    int   num_tiles       = (inner_size_val + kBlockThreads - 1) / kBlockThreads;
                    auto outer_shape      = shape<1>(tensor_2d.layout());
                    auto outer_stride     = stride<1>(tensor_2d.layout());
                    auto tile_num_stride  = static_cast<int64_t>(inner_stride_val) * kBlockThreads;
                    return make_tensor(tensor_2d.data(),
                        make_layout(
                            make_shape(Int<kBlockThreads>{}, num_tiles, outer_shape),
                            make_stride(inner_stride_val, tile_num_stride, outer_stride)));
                };

                auto tiled_src = make_tiled_3d(src_grouped);
                auto tiled_dst = make_tiled_3d(dst_grouped);

                auto tiled_copy = make_tiled_copy(
                    Copy_Atom<UniversalCopy<uint_bit_t<kVec * kElemBits>>, T>{},
                    make_layout(make_shape(Int<kCopyThreads>{})),
                    make_layout(make_shape(Int<kVec>{})));

                int64_t inner_size = static_cast<int64_t>(src_shape[0]);
                dim3    grid(size<1>(tiled_src), size<2>(tiled_src));

                auto func = kernel::CopyKernelND<
                    kVec == 1,
                    decltype(tiled_src),
                    decltype(tiled_dst),
                    decltype(tiled_copy)>;

                func<<<grid, kBlockThreads, 0, stream>>>(
                    tiled_src, tiled_dst, tiled_copy, inner_size);
            };

            switch (rank) {
                case 1: invoke_nd(constant<1>{}); break;
                case 2: invoke_nd(constant<2>{}); break;
                case 3: invoke_nd(constant<3>{}); break;
                case 4: invoke_nd(constant<4>{}); break;
                default:
                    TM_CHECK(0) << "GenericCopy: rank > 4 not implemented"; break;
            }

            }  // end if constexpr (kVec * kElemBits <= 128)
        };

        switch (vec_size) {
            case 16: dispatch_vec(constant<16>{}); break;
            case 8:  dispatch_vec(constant<8>{}); break;
            case 4:  dispatch_vec(constant<4>{}); break;
            case 2:  dispatch_vec(constant<2>{}); break;
            default: dispatch_vec(constant<1>{}); break;
        }
    };

    switch (byte_size(dtype)) {
        case 1: return dispatch_elem_size(uint8_t{});
        case 2: return dispatch_elem_size(uint16_t{});
        case 4: return dispatch_elem_size(uint32_t{});
        case 8: return dispatch_elem_size(uint64_t{});
        default:
            TM_CHECK(0) << "GenericCopy: unsupported element size " << byte_size(dtype);
            break;
    }
}
```

Key differences from the current code:
- Lines 199–226 (core::Layout algebra) → replaced by `pad_arrays` + `sort_by_stride` + `coalesce_ascending` + `reshape_to`
- Lines 233–272 (transpose detection on core::Layout) → transpose detection on arrays (`src_stride[0] == 1`)
- Lines 279–313 (alignment on core::Layout) → alignment on arrays
- Lines 316–441 (dispatch) → same structure, reading from arrays instead of `a.shape(0)` / `a.stride(0)`

- [ ] **Step 2: Build**

```bash
cd /data/lmdeploy-copy/build && ninja
```

Expected: Clean build with no errors.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "refactor(core): rewrite GenericCopy with array-based layout algebra"
```

---

### Task 3: Test correctness

**Files:**
- Test: `test_generic_copy.py` (existing, no changes needed)

Run the existing test suite to verify correctness after the refactor.

- [ ] **Step 1: Run the test suite**

```bash
cd /data/lmdeploy-copy && python test_generic_copy.py
```

Expected: ALL TESTS PASSED. All test cases (contiguous, rank-1, 2D transformations, 3D, 4D, combined, throughput sweep) must pass.

- [ ] **Step 2: Run with additional dtypes**

```bash
python test_generic_copy.py --dtype f16
python test_generic_copy.py --dtype bf16
python test_generic_copy.py --dtype i8
```

Expected: ALL TESTS PASSED for each dtype.

- [ ] **Step 3: If tests fail, debug and fix**

Common issues:
- `coalesce_ascending` merging logic incorrect → compare output against `core::Layout::coalesce`
- `sort_by_stride` permutation applied incorrectly → verify with simple test
- `reshape_to` splitting across non-contiguous boundary → check stride computation
- `make_cute_layout_unit_inner` called with wrong shape (should use `src_shape` not `a.shape()`) → verify alignment

Iterate: modify → build → test until all tests pass.

- [ ] **Step 4: Commit any fixes**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "fix(core): fix issues found during testing of CuTe-first layout algebra"
```

---

### Task 4: Clean up dead code

**Files:**
- Modify: `src/turbomind/core/tensor.cu`

After the rewrite, verify that no unused code remains in the `detail` namespace or `GenericCopy`.

- [ ] **Step 1: Check for dead code**

Search for any references to `core::Layout` methods that are no longer used in `GenericCopy`:
- `core::Layout::permute` — should not appear in GenericCopy
- `core::Layout::coalesce` — should not appear in GenericCopy
- `core::Layout::view` — should not appear in GenericCopy

The `detail::make_cute_layout` and `detail::make_cute_layout_unit_inner` helpers are still used inside the dispatch (Task 2 step 1, the `invoke_nd` lambda) and must be kept.

- [ ] **Step 2: Remove any dead includes or using-declarations**

If `core::Layout` methods are no longer called directly from `GenericCopy`, check if any includes can be simplified. Keep `layout.h` include since `src.layout()` is still used.

- [ ] **Step 3: Final build and test**

```bash
cd /data/lmdeploy-copy/build && ninja && cd .. && python test_generic_copy.py
```

Expected: Clean build, ALL TESTS PASSED.

- [ ] **Step 4: Commit cleanup**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "cleanup(core): remove dead code after CuTe-first layout algebra refactor"
```
