# Full-Utilization CopyKernelND Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace TiledCopy-based CopyKernelND with manual thread-to-element mapping so all 256 threads participate, fixing ~20% throughput on wide vectorization.

**Architecture:** The kernel receives raw pointers + CuTe shape/stride/partition tuples. It decodes threadIdx/blockIdx into multi-dim coordinates via `idx2crd`, computes memory offsets via `crd2idx`, and applies `Copy_Atom` on per-thread rank-1 gmem tensors. No TiledCopy, no `if constexpr` on rank.

**Tech Stack:** CUDA, CuTe (CUTLASS), CuTe tuple operations (`transform`, `for_each`, `idx2crd`, `crd2idx`)

**Files:**
- Modify: `src/turbomind/core/tensor.cu` (kernel + host dispatch — all changes in this file)
- Test: `test_generic_copy.py` (existing, no changes needed)

---

### Task 1: Add host helpers to detail namespace

Add `compute_thr_partition` and `make_vec_factors` to the `detail` namespace. These go in `src/turbomind/core/tensor.cu` after the existing helpers (after line 73, before `}  // namespace detail`).

**Files:**
- Modify: `src/turbomind/core/tensor.cu:24-73` (detail namespace)

- [ ] **Step 1: Add `make_vec_factors` helper**

Add after line 71 (after `make_cute_layout_unit_inner`), still inside `namespace detail`:

```cpp
// Construct vec_factors tuple: (kVec, 1, 1, ...) — used for element coord scaling.
// Dim 0 (innermost) scales by kVec; all other dims scale by 1.
template<int kVec, int kRank, size_t... Is>
auto make_vec_factors_impl(std::index_sequence<Is...>)
{
    return make_shape((Is == 0 ? Int<kVec>{} : Int<1>{})...);
}

template<int kVec, int kRank>
auto make_vec_factors()
{
    return make_vec_factors_impl<kVec, kRank>(std::make_index_sequence<kRank>{});
}
```

- [ ] **Step 2: Add `compute_thr_partition` helper**

```cpp
// Compute thread partition: (T0, T1, ..., Tk-1) where T0*...*Tk-1 = 256.
// T0 is the largest power-of-2 <= shape[0]/kVec.
// Remaining threads are distributed across outer dims.
template<int kRank>
auto compute_thr_partition(const ssize_t* shape, int kVec)
    -> std::array<ssize_t, kRank>
{
    std::array<ssize_t, kRank> partition{};
    partition.fill(1);

    // Inner dim: largest power-of-2 that divides 256 and <= shape[0]/kVec
    int64_t max_inner = shape[0] / kVec;
    ssize_t T0 = 256;
    while (T0 > 1 && T0 > max_inner) {
        T0 /= 2;
    }
    partition[0] = T0;

    // Distribute remaining threads across outer dims
    ssize_t remaining = 256 / T0;
    for (int i = 1; i < kRank; ++i) {
        partition[i] = std::min<ssize_t>(shape[i], remaining);
        remaining /= partition[i];
        if (remaining < 1) {
            remaining = 1;
        }
    }
    return partition;
}
```

- [ ] **Step 3: Build to verify compilation**

Run: `cd /data/lmdeploy-copy/build && ninja _turbomind 2>&1 | tail -20`
Expected: BUILD SUCCEEDED (new helpers are not yet called, so they just need to compile).

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "feat(core): add thread partition and vec_factors helpers for full-utilization copy"
```

---

### Task 2: Write the new CopyKernelND kernel

Replace the existing kernel (lines 75-135) with the new manual-tiling kernel. Keep `TransposeCopyKernel` unchanged.

**Files:**
- Modify: `src/turbomind/core/tensor.cu:75-135` (kernel namespace)

- [ ] **Step 1: Replace the old CopyKernelND with the new kernel**

Replace everything from the comment block `// ============================================================================` at line 75 through line 135 (the old kernel) with:

```cpp
// ============================================================================
// CUDA kernel: CopyKernelND (full-utilization manual-tiling copy)
// ============================================================================
// All 256 threads participate. Thread-to-element mapping is done manually
// using CuTe tuple operations (idx2crd, crd2idx, transform).
// Copy_Atom is applied to per-thread rank-1 (kVec,) gmem tensors.
namespace kernel {

template<int kVec, class DataShape, class SrcStride, class DstStride,
         class ThrPartition, class TileCounts, class VecFactors, typename T>
__global__ void __launch_bounds__(256)
CopyKernelND(const T* __restrict__ src, T* __restrict__ dst,
             DataShape data_shape,
             SrcStride src_strides, DstStride dst_strides,
             ThrPartition thr_partition, TileCounts tile_counts,
             VecFactors vec_factors)
{
    using namespace cute;

    // 1. Decode threadIdx -> per-dim thread coordinate (colexicographic)
    auto thr_coord = idx2crd(threadIdx.x, thr_partition);

    // 2. Decode blockIdx -> per-dim tile coordinate
    auto tile_coord = idx2crd(int64_t(blockIdx.x), tile_counts);

    // 3. Compute element coordinate
    //    inner_coord[i] = tile_coord[i] * thr_partition[i] + thr_coord[i]
    //    elem_coord[i]  = inner_coord[i] * vec_factors[i]
    auto inner_coord = transform(tile_coord, thr_coord, thr_partition,
        [](auto tc, auto thr, auto tp) { return tc * tp + thr; });
    auto elem_coord = transform(inner_coord, vec_factors,
        [](auto ic, auto vf) { return ic * vf; });

    // 4. Bounds check: elem[i] + vec_factor[i] <= shape[i]
    bool valid = true;
    for_each(transform(elem_coord, vec_factors, data_shape,
        [](auto ec, auto vf, auto s) { return ec + vf <= s; }),
        [&](auto v) { valid = valid && static_cast<bool>(v); });
    if (!valid) return;

    // 5. Compute memory offsets via CuTe's crd2idx
    int64_t src_off = crd2idx(elem_coord, data_shape, src_strides);
    int64_t dst_off = crd2idx(elem_coord, data_shape, dst_strides);

    // 6. Per-thread vectorized copy via Copy_Atom
    auto src_frag = make_tensor(make_gmem_ptr(src + src_off),
        make_layout(make_shape(Int<kVec>{}), make_stride(Int<1>{})));
    auto dst_frag = make_tensor(make_gmem_ptr(dst + dst_off),
        make_layout(make_shape(Int<kVec>{}), make_stride(Int<1>{})));
    copy(Copy_Atom<UniversalCopy<uint_bit_t<kVec * sizeof_bits_v<T>>>, T>{},
         src_frag, dst_frag);
}

}  // namespace kernel
```

- [ ] **Step 2: Build to verify kernel compiles**

Run: `cd /data/lmdeploy-copy/build && ninja _turbomind 2>&1 | tail -30`
Expected: BUILD SUCCEEDED. The kernel is not yet called from host code, but it must compile.

If there are CuTe type errors, fix them. Common issues:
- `transform` with 3 tuples may need explicit include of `<cute/algorithm/tuple_algorithms.hpp>`
- `crd2idx` return type may need explicit cast to `int64_t`

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "refactor(core): replace CopyKernelND with full-utilization manual-tiling kernel"
```

---

### Task 3: Rewrite the host dispatch in GenericCopy

Replace the triple-dispatch section (lines 315-436: `dispatch_elem_size` through the dtype switch) with the new approach that computes thread partitions and passes CuTe tuples to the new kernel.

**Files:**
- Modify: `src/turbomind/core/tensor.cu:315-436` (dispatch section of GenericCopy)

- [ ] **Step 1: Replace the dispatch section**

Replace lines 315 through 436 (from `// --- Dispatch on data type T...` through the closing `}` of GenericCopy, but NOT the `}  // namespace turbomind::core`) with:

```cpp
    // --- Dispatch on data type T and vec_size kVec ---
    auto dispatch_elem_size = [&](auto t) {
        using T = decltype(t);
        constexpr int kElemBits = sizeof_bits_v<T>;

        auto dispatch_vec = [&](auto v) {
            constexpr int kVec = v.value;

            // Guard: CuTe Copy_Atom supports up to 128 bits
            if constexpr (kVec * kElemBits <= 128) {

            auto dispatch_rank = [&](auto d) {
                constexpr int kRank = d.value;

                // 1. Create CuTe layouts (same helpers as before)
                auto src_layout = [&] {
                    if constexpr (kVec > 1)
                        return detail::make_cute_layout_unit_inner<kRank>(
                            a.shape().data(), a.stride().data());
                    else
                        return detail::make_cute_layout<kRank>(
                            a.shape().data(), a.stride().data());
                }();
                auto dst_layout = [&] {
                    if constexpr (kVec > 1)
                        return detail::make_cute_layout_unit_inner<kRank>(
                            a.shape().data(), b.stride().data());
                    else
                        return detail::make_cute_layout<kRank>(
                            a.shape().data(), b.stride().data());
                }();

                // 2. Extract shapes and strides as CuTe tuples
                auto data_shape   = src_layout.shape();
                auto src_strides  = src_layout.stride();
                auto dst_strides  = dst_layout.stride();

                // 3. Compute thread partition
                auto partition_arr = detail::compute_thr_partition<kRank>(
                    a.shape().data(), kVec);
                auto thr_partition = detail::make_cute_shape<kRank>(
                    partition_arr.data());

                // 4. Compute vec_factors and tile_counts
                auto vec_factors = detail::make_vec_factors<kVec, kRank>();
                auto tile_sizes  = transform(thr_partition, vec_factors,
                    [](auto tp, auto vf) { return tp * vf; });
                auto tile_counts = transform(data_shape, tile_sizes,
                    [](auto s, auto ts) -> int64_t {
                        return (static_cast<int64_t>(s) + static_cast<int64_t>(ts) - 1)
                               / static_cast<int64_t>(ts);
                    });

                // 5. Grid: 1D, each block = one flat tile
                int64_t total_tiles = product(tile_counts);
                dim3 grid(static_cast<uint32_t>(total_tiles));

                // 6. Launch kernel
                kernel::CopyKernelND<kVec>
                    <<<grid, kBlockThreads, 0, stream>>>(
                        reinterpret_cast<const T*>(data_a),
                        reinterpret_cast<T*>(data_b),
                        data_shape, src_strides, dst_strides,
                        thr_partition, tile_counts, vec_factors);
            };

            switch (rank) {
                case 1: dispatch_rank(constant<1>{}); break;
                case 2: dispatch_rank(constant<2>{}); break;
                case 3: dispatch_rank(constant<3>{}); break;
                case 4: dispatch_rank(constant<4>{}); break;
                default: TM_CHECK(0) << "GenericCopy: rank > 4 not implemented"; break;
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
```

- [ ] **Step 2: Build**

Run: `cd /data/lmdeploy-copy/build && ninja _turbomind 2>&1 | tail -30`
Expected: BUILD SUCCEEDED.

If there are compilation errors:
- `make_cute_shape` takes `const ssize_t*` but `partition_arr` returns `std::array<ssize_t, N>` — `.data()` returns `ssize_t*`, which is correct.
- CuTe `transform` with runtime tuples may need type adjustments. Cast inside lambdas if needed.
- The `product()` call on tile_counts may need `cute::` prefix or include.

- [ ] **Step 3: Fix any compilation issues and rebuild**

Iterate until the build succeeds. Common fixes:
- Add `using namespace cute;` or explicit `cute::` prefix for `transform`, `for_each`, `idx2crd`, `crd2idx`, `product`.
- If `transform` with 3 args is not found, verify `<cute/algorithm/tuple_algorithms.hpp>` is included (it should be pulled in by `<cute/tensor.hpp>`).
- If `crd2idx` return type issues, add `static_cast<int64_t>(crd2idx(...))`.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "refactor(core): rewrite GenericCopy dispatch for full-utilization kernel"
```

---

### Task 4: Run existing correctness tests

Run the existing test suite to verify all copy operations still produce correct results.

**Files:**
- Test: `test_generic_copy.py` (no modifications)

- [ ] **Step 1: Run tests with f32**

Run: `cd /data/lmdeploy-copy && PYTHONPATH=lmdeploy:build/lib python test_generic_copy.py --dtype f32 2>&1`
Expected: All tests PASS. The output should show `Results: X/X passed` and `ALL TESTS PASSED`.

If any test fails, inspect the failure output. Common issues:
- Shape/stride mismatch in CuTe tuple construction
- Wrong offset computation in `crd2idx`
- Thread partition heuristic producing bad partition (e.g., partition element = 0)
- Bounds check too aggressive (rejecting valid coordinates)

- [ ] **Step 2: Debug and fix any failures**

If tests fail:
1. Note which test case failed (e.g., "col-stride", "4D slice")
2. Check the shape/stride/partition values by adding temporary prints in the host dispatch
3. Verify the kernel receives correct parameters
4. Fix the issue and rebuild: `cd build && ninja _turbomind`
5. Re-run the failing test

Do NOT proceed to Task 5 until all f32 tests pass.

- [ ] **Step 3: Run tests with i8**

Run: `cd /data/lmdeploy-copy && PYTHONPATH=lmdeploy:build/lib python test_generic_copy.py --dtype i8 2>&1`
Expected: All tests PASS. This tests the wide vectorization path (kVec=16).

- [ ] **Step 4: Run tests with f16**

Run: `cd /data/lmdeploy-copy && PYTHONPATH=lmdeploy:build/lib python test_generic_copy.py --dtype f16 2>&1`
Expected: All tests PASS. This tests kVec=8 vectorization.

- [ ] **Step 5: Run tests with remaining dtypes**

Run each:
```bash
cd /data/lmdeploy-copy && PYTHONPATH=lmdeploy:build/lib python test_generic_copy.py --dtype bf16
cd /data/lmdeploy-copy && PYTHONPATH=lmdeploy:build/lib python test_generic_copy.py --dtype f64
cd /data/lmdeploy-copy && PYTHONPATH=lmdeploy:build/lib python test_generic_copy.py --dtype i32
cd /data/lmdeploy-copy && PYTHONPATH=lmdeploy:build/lib python test_generic_copy.py --dtype i64
```
Expected: All PASS for each dtype.

- [ ] **Step 6: Commit (if any fixes were needed)**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "fix(core): address correctness issues in full-utilization CopyKernelND"
```

---

### Task 5: Benchmark and verify throughput improvement

Run the throughput sweep to verify the performance improvement. The test already benchmarks contiguous and transposed copies at various sizes.

**Files:**
- Test: `test_generic_copy.py` (no modifications)

- [ ] **Step 1: Benchmark f32 contiguous throughput**

Run: `cd /data/lmdeploy-copy && PYTHONPATH=lmdeploy:build/lib python test_generic_copy.py --dtype f32 2>&1 | grep -A 20 "Throughput Summary"`
Expected: GenericCopy contiguous throughput should be >80% of PyTorch (up from ~25%). The percentage column should show significant improvement.

- [ ] **Step 2: Benchmark i8 contiguous throughput**

Run: `cd /data/lmdeploy-copy && PYTHONPATH=lmdeploy:build/lib python test_generic_copy.py --dtype i8 2>&1 | grep -A 20 "Throughput Summary"`
Expected: GenericCopy contiguous throughput should be >80% of PyTorch (up from ~20%). This is the widest vectorization case (kVec=16) and should show the largest improvement.

- [ ] **Step 3: Benchmark f16 throughput**

Run: `cd /data/lmdeploy-copy && PYTHONPATH=lmdeploy:build/lib python test_generic_copy.py --dtype f16 2>&1 | grep -A 20 "Throughput Summary"`
Expected: Significant improvement over baseline.

- [ ] **Step 4: Verify strided/transpose performance is not degraded**

Check the throughput summary for transpose and strided cases. These use different code paths:
- Transpose: uses `TransposeCopyKernel` (unchanged) — throughput should be identical to before.
- Strided non-transpose: uses the new kernel with kVec=1 (scalar) — throughput should be similar or slightly improved due to better thread utilization.

- [ ] **Step 5: Record results and commit (if any tuning was done)**

Note the throughput numbers in a comment. If any tuning was needed (e.g., adjusting the thread partition heuristic), commit the changes.

---

### Task 6: Clean up

Remove dead code and clean up the dispatch.

**Files:**
- Modify: `src/turbomind/core/tensor.cu`

- [ ] **Step 1: Remove unused variables and dead code**

Check for and remove:
- The `kCopyThreads` variable (if it still exists)
- The `make_tiled_3d` lambda (if it still exists)
- Any commented-out old code

- [ ] **Step 2: Build and test one final time**

Run: `cd /data/lmdeploy-copy/build && ninja _turbomind && cd .. && PYTHONPATH=lmdeploy:build/lib python test_generic_copy.py --dtype f32 2>&1 | tail -5`
Expected: BUILD SUCCEEDED, all tests PASS.

- [ ] **Step 3: Final commit**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "cleanup(core): remove dead code from full-utilization GenericCopy refactor"
```
