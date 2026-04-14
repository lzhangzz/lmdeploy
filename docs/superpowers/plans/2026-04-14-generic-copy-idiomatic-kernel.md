# Idiomatic CopyKernelND Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor CopyKernelND to match the CUTLASS tiled_copy.cu tutorial pattern — host-side tensor construction/tiling, kernel receives pre-tiled tensors.

**Architecture:** Move all CuTe tensor construction, layout grouping, `tiled_divide`, and `TiledCopy` creation from the kernel to the host dispatch. The kernel becomes a pure slice → partition → fragment → copy loop matching `copy_kernel_vectorized` from `examples/cute/tutorial/tiled_copy.cu`.

**Tech Stack:** CUDA, CuTe (CUTLASS), PyTorch (for tests)

**Spec:** `docs/superpowers/specs/2026-04-14-generic-copy-idiomatic-kernel-design.md`

**Reference:** `build/_deps/repo-cutlass-src/examples/cute/tutorial/tiled_copy.cu`

---

### Task 1: Rewrite CopyKernelND kernel body

**Files:**
- Modify: `src/turbomind/core/tensor.cu:79-165` (the entire `CopyKernelND` kernel)

Replace the kernel signature and body with the tutorial pattern. The new kernel receives pre-tiled tensors + TiledCopy from the host.

- [ ] **Step 1: Replace the CopyKernelND kernel**

Replace lines 79-165 in `src/turbomind/core/tensor.cu` with:

```cpp
// ============================================================================
// CUDA kernel: CopyKernelND (tutorial-pattern vectorized copy)
// ============================================================================
// Matches the copy_kernel_vectorized pattern from
// cutlass/examples/cute/tutorial/tiled_copy.cu:
//   1. Host creates and tiles tensors via tiled_divide
//   2. Kernel slices tile by blockIdx
//   3. Thread partition via TiledCopy
//   4. Register fragment + two-phase copy (gmem→fragment→gmem)
template<bool kPredicated, class TensorS, class TensorD, class TiledCopy>
__global__ void __launch_bounds__(256)
CopyKernelND(TensorS S, TensorD D, TiledCopy tiled_copy, int64_t inner_size)
{
    constexpr int kBlockThreads = 256;

    // Excess threads (kCopyThreads < kBlockThreads when kVec > 1) exit early
    if (threadIdx.x >= size(tiled_copy)) return;

    // Bounds check on tile grid
    if (blockIdx.y >= size<2>(S) || blockIdx.x >= size<1>(S)) return;

    // Slice tile — tutorial pattern
    auto tile_S = S(_, blockIdx.x, blockIdx.y);
    auto tile_D = D(_, blockIdx.x, blockIdx.y);

    // Thread partition — tutorial pattern
    ThrCopy thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
    auto thr_S = thr_copy.partition_S(tile_S);
    auto thr_D = thr_copy.partition_D(tile_D);

    // Register fragment — tutorial pattern
    auto fragment = make_fragment_like(thr_D);

    if constexpr (!kPredicated) {
        // Vectorized: unconditional two-phase copy
        copy(tiled_copy, thr_S, fragment);
        copy(tiled_copy, fragment, thr_D);
    } else {
        // Scalar: predicated copy for partial last tile
        auto id_tile = make_identity_tensor(make_shape(size<0>(tile_S)));
        auto thr_id  = thr_copy.partition_S(id_tile);

        auto pred = make_tensor<bool>(shape(thr_S));
        PRAGMA_UNROLL
        for (int i = 0; i < size(pred); ++i) {
            pred(i) = static_cast<int64_t>(get<0>(thr_id(i)))
                      + static_cast<int64_t>(blockIdx.x) * kBlockThreads
                      < inner_size;
        }

        copy_if(pred, tiled_copy, thr_S, fragment);
        copy_if(pred, tiled_copy, fragment, thr_D);
    }
}
```

- [ ] **Step 2: Build to verify compilation**

Run: `cd /data/lmdeploy-copy/build && ninja 2>&1 | tail -20`
Expected: Build succeeds. The host dispatch still references the old kernel signature so this will fail with type errors — that's expected, we fix it in Task 2.

Note: If compilation fails with errors about the *old* kernel being called from the host with wrong argument types, that confirms the kernel signature change is correct and we need Task 2 next. Do NOT fix host-side errors here.

---

### Task 2: Rewrite host-side dispatch in GenericCopy

**Files:**
- Modify: `src/turbomind/core/tensor.cu:345-383` (the `invoke_nd` lambda inside `dispatch_vec` inside `dispatch_elem_size`)

Restructure the host dispatch to construct CuTe tensors, group outer dims, tile with `tiled_divide`, create `TiledCopy`, and pass them by value to the new kernel.

- [ ] **Step 1: Replace the `dispatch_elem_size` lambda body (lines 346-401)**

Replace the entire `dispatch_elem_size` lambda and its trailing switch statement (lines 346-409) with:

```cpp
    // --- Dispatch on data type T, vec_size kVec, and rank kRank ---
    auto dispatch_elem_size = [&](auto t) {
        using T = decltype(t);

        auto dispatch_vec = [&](auto v) {
            constexpr int kVec = v.value;

            auto invoke_nd = [&](auto d) {
                constexpr int kRank      = d.value;
                constexpr int kBlockThreads = 256;
                constexpr int kCopyThreads  = kBlockThreads / kVec;

                // 1. Create CuTe layouts (same as before)
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

                // 2. Wrap in CuTe gmem tensors
                auto src_gmem = make_tensor(
                    make_gmem_ptr(reinterpret_cast<const T*>(data_a)), src_layout);
                auto dst_gmem = make_tensor(
                    make_gmem_ptr(reinterpret_cast<T*>(data_b)), dst_layout);

                // 3. Group outer dims → rank-2 (inner, outer_product)
                auto src_2d = [&] {
                    if constexpr (kRank > 1) return group_modes<1, kRank>(src_gmem);
                    else return src_gmem;
                }();
                auto dst_2d = [&] {
                    if constexpr (kRank > 1) return group_modes<1, kRank>(dst_gmem);
                    else return dst_gmem;
                }();

                // 4. Tile inner dim
                auto tiled_src = tiled_divide(src_2d, Int<kBlockThreads>{});
                auto tiled_dst = tiled_divide(dst_2d, Int<kBlockThreads>{});

                // 5. Create TiledCopy
                auto tiled_copy = make_tiled_copy(
                    Copy_Atom<UniversalCopy<uint_bit_t<kVec * sizeof_bits_v<T>>>, T>{},
                    make_layout(make_shape(Int<kCopyThreads>{})),
                    make_layout(make_shape(Int<kVec>{})));

                // 6. Grid and launch
                int64_t inner_size = static_cast<int64_t>(a.shape(0));
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
                default: TM_CHECK(0) << "GenericCopy: rank > 4 not implemented"; break;
            }
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
        default: TM_CHECK(0) << "GenericCopy: unsupported element size " << byte_size(dtype); break;
    }
```

- [ ] **Step 2: Build to verify compilation**

Run: `cd /data/lmdeploy-copy/build && ninja 2>&1 | tail -30`
Expected: Build succeeds with no errors.

---

### Task 3: Verify correctness — f32

Run the full test suite for float32 to verify the refactor preserves all existing behavior.

- [ ] **Step 1: Run f32 tests**

Run:
```bash
cd /data/lmdeploy-copy && \
PYTHONPATH=lmdeploy:build/lib python test_generic_copy.py --dtype f32 2>&1
```

Expected: All tests pass (contiguous, rank-1, transpose, row-stride, col-stride, narrow, permute, 4D, slice+transpose, throughput sweep). Output ends with "ALL TESTS PASSED".

If any test fails, debug the failure before proceeding. The most likely issue is the `group_modes` call producing the wrong layout — verify that `group_modes<1,kRank>(tensor)` on a rank-1 tensor is a no-op.

---

### Task 4: Verify correctness — all dtypes

Run the test suite for each supported dtype to verify the refactor works across element sizes.

- [ ] **Step 1: Run f64 tests**

Run: `cd /data/lmdeploy-copy && PYTHONPATH=lmdeploy:build/lib python test_generic_copy.py --dtype f64 2>&1`
Expected: ALL TESTS PASSED

- [ ] **Step 2: Run f16 tests**

Run: `cd /data/lmdeploy-copy && PYTHONPATH=lmdeploy:build/lib python test_generic_copy.py --dtype f16 2>&1`
Expected: ALL TESTS PASSED

- [ ] **Step 3: Run bf16 tests**

Run: `cd /data/lmdeploy-copy && PYTHONPATH=lmdeploy:build/lib python test_generic_copy.py --dtype bf16 2>&1`
Expected: ALL TESTS PASSED

- [ ] **Step 4: Run i8 tests**

Run: `cd /data/lmdeploy-copy && PYTHONPATH=lmdeploy:build/lib python test_generic_copy.py --dtype i8 2>&1`
Expected: ALL TESTS PASSED

- [ ] **Step 5: Run i32 tests**

Run: `cd /data/lmdeploy-copy && PYTHONPATH=lmdeploy:build/lib python test_generic_copy.py --dtype i32 2>&1`
Expected: ALL TESTS PASSED

- [ ] **Step 6: Run i64 tests**

Run: `cd /data/lmdeploy-copy && PYTHONPATH=lmdeploy:build/lib python test_generic_copy.py --dtype i64 2>&1`
Expected: ALL TESTS PASSED

---

### Task 5: Commit

- [ ] **Step 1: Commit the refactor**

```bash
cd /data/lmdeploy-copy && git add src/turbomind/core/tensor.cu && \
git commit -m "refactor(core): rewrite CopyKernelND to match tiled_copy.cu tutorial pattern

Move CuTe tensor construction, layout grouping, tiled_divide, and
TiledCopy creation from kernel to host dispatch. Kernel is now a pure
slice → partition → fragment → copy loop matching copy_kernel_vectorized
from cutlass/examples/cute/tutorial/tiled_copy.cu.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
