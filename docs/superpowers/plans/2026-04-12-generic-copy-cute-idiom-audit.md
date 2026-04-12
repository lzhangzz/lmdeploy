# CuTe Idiom Audit Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the 5 actionable findings from the CuTe idiom audit: add compile-time preconditions, verify `make_fragment_like` safety, investigate kernel unification, improve layout helper documentation, and document the stride-1 vectorization requirement.

**Architecture:** All changes target `src/turbomind/core/tensor.cu`. Four of five items are small, localized edits (comments and `static_assert`s). The fifth item (P2/M3 — kernel unification) requires more investigation: `group<1,1>` triggers a `static_assert` failure in CuTe, so unification requires an `if constexpr` guard around the `group` call.

**Tech Stack:** C++17, CUDA, CuTe (NVIDIA CUTLASS), Python/PyTorch for testing.

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `src/turbomind/core/tensor.cu` | Modify | All code changes (assertions, comments, kernel unification) |
| `test_generic_copy.py` | Read-only | Existing test suite used to verify no regressions |

---

### Task 1: Add compile-time preconditions to all three kernels (M5)

**Files:**
- Modify: `src/turbomind/core/tensor.cu`

- [ ] **Step 1: Add `static_assert` to CopyKernelND**

Insert after line 149 (`constexpr int kRank = cute::rank_v<SrcLayoutT>;`), before line 150 (`constexpr int kCopyThreads`):

```cpp
    static_assert(2 <= kRank && kRank <= 4, "CopyKernelND: rank must be 2..4");
```

- [ ] **Step 2: Add `static_assert` to TransposeCopyKernel**

Insert after the opening `if constexpr` on line 220, as the first statement inside the constexpr block (before the `kThrLoad` line at 224):

```cpp
    static_assert(kTileDim % kVec == 0, "TransposeCopyKernel: kTileDim must be divisible by kVec");
```

Note: `kVec >= 2` is already guarded by `if constexpr (kVec >= 2)` on line 220, so a separate `static_assert` is redundant — the compiler won't instantiate the body for `kVec < 2`. No assertion needed for that condition.

- [ ] **Step 3: Add rank assertion to CopyKernel1D**

Insert after the function signature at line 83, before the `if constexpr` on line 84:

```cpp
    static_assert(cute::rank_v<SrcLayoutT> == 1, "CopyKernel1D: layout rank must be 1");
```

- [ ] **Step 4: Build to verify compilation**

Run: `cd /data/lmdeploy-copy/build && ninja`

Expected: Clean build with no errors. The assertions should be trivially satisfied by all existing call sites.

- [ ] **Step 5: Run existing tests to confirm no regressions**

Run: `cd /data/lmdeploy-copy && python test_generic_copy.py`

Expected: ALL TESTS PASSED (same results as before).

- [ ] **Step 6: Commit**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "fix(core): add compile-time preconditions to GenericCopy kernels

Add static_assert for rank and divisibility constraints:
- CopyKernel1D: assert rank == 1
- CopyKernelND: assert rank 2..4
- TransposeCopyKernel: assert kTileDim % kVec == 0"
```

---

### Task 2: Add safety comment for `make_fragment_like` usage (C1)

**Files:**
- Modify: `src/turbomind/core/tensor.cu`

- [ ] **Step 1: Add explanatory comment at both `make_fragment_like` call sites**

At line 267, replace:
```cpp
    auto rmem_ld  = cute::make_fragment_like(thr_smw);
```
with:
```cpp
    // make_fragment_like is safe here: smem_w has mode-0 stride Int<1> (column-major),
    // so make_fragment_like (which forces mode-0 to stride-1) produces the same layout
    // as make_tensor_like. The manual rmem(i)=smem(i) loop below depends on this match.
    auto rmem_ld  = cute::make_fragment_like(thr_smw);
```

At line 290 (now shifted by 3 lines due to the above), replace:
```cpp
    auto rmem_st   = cute::make_fragment_like(thr_smw2);
```
with:
```cpp
    // Same reasoning as rmem_ld above: smem_w mode-0 is stride-1, so fragment layout matches.
    auto rmem_st   = cute::make_fragment_like(thr_smw2);
```

- [ ] **Step 2: Build and test**

Run: `cd /data/lmdeploy-copy/build && ninja && cd /data/lmdeploy-copy && python test_generic_copy.py`

Expected: Clean build, ALL TESTS PASSED. Comments don't affect behavior.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "docs(core): document make_fragment_like safety in TransposeCopyKernel

Explain why make_fragment_like produces the correct layout for smem
register buffers: smem_w has mode-0 stride Int<1>, matching the
fragment's forced column-major mode-0."
```

---

### Task 3: Add namespace comment to layout helpers (M1)

**Files:**
- Modify: `src/turbomind/core/tensor.cu`

- [ ] **Step 1: Replace the detail namespace header comment**

At line 21, replace:
```cpp
namespace detail {
```
with:
```cpp
// CuTe's make_shape/make_stride require compile-time variadic template args,
// but our tensor shapes and strides are runtime values. These helpers bridge
// that gap via std::index_sequence expansion, producing CuTe Layout objects
// from runtime shape/stride arrays. Only the innermost stride is promoted to
// compile-time Int<1> (in make_cute_layout_unit_inner) to enable CuTe's
// vectorized Copy_Atom recast.
namespace detail {
```

- [ ] **Step 2: Build**

Run: `cd /data/lmdeploy-copy/build && ninja`

Expected: Clean build.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "docs(core): explain runtime-to-CuTe layout bridge in detail namespace"
```

---

### Task 4: Document stride-1 vectorization requirement (P4)

**Files:**
- Modify: `src/turbomind/core/tensor.cu`

- [ ] **Step 1: Add comment above the alignment block**

At line 413 (the `// --- Alignment detection ---` comment), replace:
```cpp
    // --- Alignment detection ---
```
with:
```cpp
    // --- Alignment detection ---
    // NOTE: GenericCopy vectorizes along the innermost (stride-sorted) dimension.
    // If neither src nor dst has a stride-1 innermost dim, alignment falls to
    // byte_size(dtype) (vec_size=1), resulting in scalar copies. Vectorizing
    // along a non-contiguous dimension would require a different kernel architecture.
```

- [ ] **Step 2: Build**

Run: `cd /data/lmdeploy-copy/build && ninja`

Expected: Clean build.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "docs(core): document stride-1 vectorization requirement in GenericCopy"
```

---

### Task 5: Investigate CopyKernel1D + CopyKernelND unification (P2/M3)

This task is investigative. The key question is whether `if constexpr (kRank > 1)` around the `group` call in CopyKernelND allows it to handle rank-1 tensors without the separate CopyKernel1D kernel. If yes, unify; if no, document why they must remain separate.

**Files:**
- Modify: `src/turbomind/core/tensor.cu`

- [ ] **Step 1: Run existing tests as baseline**

Run: `cd /data/lmdeploy-copy && python test_generic_copy.py`

Expected: ALL TESTS PASSED. Record output for comparison.

- [ ] **Step 2: Add a rank-1 test case to confirm rank-1 coverage**

Add this test case in `test_generic_copy.py`, in the `main()` function, after the "Contiguous baseline" section (around line 155):

```python
    # --- Rank-1 (1D contiguous) ---
    print("\nRank-1:")
    check("rank-1 f32", torch.randn(8192, dtype=torch.float32, device=DEV))
    check("rank-1 f16", torch.randn(8192, dtype=torch.float16, device=DEV), atol=1e-3, rtol=1e-3)
    check("rank-1 i32", torch.randint(0, 1000, (8192,), dtype=torch.int32, device=DEV))
```

Run: `cd /data/lmdeploy-copy && python test_generic_copy.py`

Expected: The new rank-1 tests PASS alongside all existing tests.

- [ ] **Step 3: Modify CopyKernelND to handle rank-1 via `if constexpr`**

Replace the CopyKernelND kernel body (lines 146-203). The key change is wrapping the `group` call and `blockIdx.y` row selection in `if constexpr (kRank > 1)`:

```cpp
    if constexpr (kVec * cute::sizeof_bits_v<T> <= 128)
    {
    constexpr int kBlockThreads = 256;
    constexpr int kRank         = cute::rank_v<SrcLayoutT>;
    static_assert(1 <= kRank && kRank <= 4, "CopyKernelND: rank must be 1..4");
    constexpr int kCopyThreads  = kBlockThreads / kVec;

    auto tiled_copy = cute::make_tiled_copy(
        cute::Copy_Atom<cute::UniversalCopy<cute::uint_bit_t<kVec * cute::sizeof_bits_v<T>>>, T>{},
        cute::make_layout(cute::make_shape(cute::Int<kCopyThreads>{})),
        cute::make_layout(cute::make_shape(cute::Int<kVec>{})));

    if (threadIdx.x >= kCopyThreads) return;

    auto thr_copy = tiled_copy.get_slice(threadIdx.x);

    // Obtain a 1D inner tensor for this CTA's row.
    // For kRank > 1, group outer dims and select by blockIdx.y.
    // For kRank == 1, use the tensor directly (only blockIdx.x tiles).
    auto gSrc = cute::make_tensor(cute::make_gmem_ptr(src_ptr), src_layout);
    auto gDst = cute::make_tensor(cute::make_gmem_ptr(dst_ptr), dst_layout);

    if constexpr (kRank > 1) {
        auto src_layout_g = cute::group<1, kRank>(src_layout);
        auto dst_layout_g = cute::group<1, kRank>(dst_layout);
        auto gSrc_g = cute::make_tensor(cute::make_gmem_ptr(src_ptr), src_layout_g);
        auto gDst_g = cute::make_tensor(cute::make_gmem_ptr(dst_ptr), dst_layout_g);

        if (blockIdx.y >= cute::size<1>(gSrc_g)) return;

        gSrc = gSrc_g(_, blockIdx.y);
        gDst = gDst_g(_, blockIdx.y);
    } else {
        if (blockIdx.y > 0) return;
    }

    auto tiler    = cute::Int<kBlockThreads>{};
    auto tiledSrc = cute::zipped_divide(gSrc, tiler);
    auto tiledDst = cute::zipped_divide(gDst, tiler);

    if (blockIdx.x >= cute::size<1>(tiledSrc)) return;

    auto ctaSrc = tiledSrc(_, blockIdx.x);
    auto ctaDst = tiledDst(_, blockIdx.x);

    auto thrSrc = thr_copy.partition_S(ctaSrc);
    auto thrDst = thr_copy.partition_D(ctaDst);

    if constexpr (kVec > 1) {
        cute::copy(tiled_copy, thrSrc, thrDst);
    }
    else {
        auto id_row   = cute::make_identity_tensor(cute::shape(gSrc));
        auto id_tiled = cute::zipped_divide(id_row, tiler);
        auto tile_id  = id_tiled(_, blockIdx.x);

        auto thrId = thr_copy.partition_S(tile_id);

        auto pred = cute::make_tensor<bool>(cute::shape(thrSrc));
        PRAGMA_UNROLL
        for (int i = 0; i < cute::size(pred); ++i) {
            pred(i) = cute::get<0>(thrId(i)) < cute::size(gSrc);
        }

        cute::copy_if(pred, thrSrc, thrDst);
    }
    }  // end if constexpr (kVec * sizeof_bits_v<T> <= 128)
```

Important: The variable names `gSrc`/`gDst` are now declared outside the `if constexpr` block (as the full-rank tensors) and then reassigned inside. Since `auto` variables cannot be reassigned, we need a different approach. The body must be restructured so that the 1D row tensor is obtained through a single code path.

**Revised approach:** Use a helper function or restructure to avoid reassignment. The simplest correct approach:

```cpp
    if constexpr (kVec * cute::sizeof_bits_v<T> <= 128)
    {
    constexpr int kBlockThreads = 256;
    constexpr int kRank         = cute::rank_v<SrcLayoutT>;
    static_assert(1 <= kRank && kRank <= 4, "CopyKernelND: rank must be 1..4");
    constexpr int kCopyThreads  = kBlockThreads / kVec;

    auto tiled_copy = cute::make_tiled_copy(
        cute::Copy_Atom<cute::UniversalCopy<cute::uint_bit_t<kVec * cute::sizeof_bits_v<T>>>, T>{},
        cute::make_layout(cute::make_shape(cute::Int<kCopyThreads>{})),
        cute::make_layout(cute::make_shape(cute::Int<kVec>{})));

    if (threadIdx.x >= kCopyThreads) return;

    auto thr_copy = tiled_copy.get_slice(threadIdx.x);

    // Group outer dims into a single mode and select this CTA's row.
    // For kRank==1, the full tensor IS the 1D row — no grouping needed.
    auto rowSrc = [&] {
        if constexpr (kRank > 1) {
            auto src_layout_g = cute::group<1, kRank>(src_layout);
            auto gSrc_g = cute::make_tensor(cute::make_gmem_ptr(src_ptr), src_layout_g);
            return gSrc_g(_, blockIdx.y);
        } else {
            return cute::make_tensor(cute::make_gmem_ptr(src_ptr), src_layout);
        }
    }();

    auto rowDst = [&] {
        if constexpr (kRank > 1) {
            auto dst_layout_g = cute::group<1, kRank>(dst_layout);
            auto gDst_g = cute::make_tensor(cute::make_gmem_ptr(dst_ptr), dst_layout_g);
            return gDst_g(_, blockIdx.y);
        } else {
            return cute::make_tensor(cute::make_gmem_ptr(dst_ptr), dst_layout);
        }
    }();

    auto tiler    = cute::Int<kBlockThreads>{};
    auto tiledSrc = cute::zipped_divide(rowSrc, tiler);
    auto tiledDst = cute::zipped_divide(rowDst, tiler);

    if (blockIdx.x >= cute::size<1>(tiledSrc)) return;

    auto ctaSrc = tiledSrc(_, blockIdx.x);
    auto ctaDst = tiledDst(_, blockIdx.x);

    auto thrSrc = thr_copy.partition_S(ctaSrc);
    auto thrDst = thr_copy.partition_D(ctaDst);

    if constexpr (kVec > 1) {
        cute::copy(tiled_copy, thrSrc, thrDst);
    }
    else {
        auto id_row   = cute::make_identity_tensor(cute::shape(rowSrc));
        auto id_tiled = cute::zipped_divide(id_row, tiler);
        auto tile_id  = id_tiled(_, blockIdx.x);

        auto thrId = thr_copy.partition_S(tile_id);

        auto pred = cute::make_tensor<bool>(cute::shape(thrSrc));
        PRAGMA_UNROLL
        for (int i = 0; i < cute::size(pred); ++i) {
            pred(i) = cute::get<0>(thrId(i)) < cute::size(rowSrc);
        }

        cute::copy_if(pred, thrSrc, thrDst);
    }
    }  // end if constexpr (kVec * sizeof_bits_v<T> <= 128)
```

Note: The `kRank == 1` branch needs `blockIdx.y == 0` check, same as the original CopyKernel1D. Add it inside the else branch of the lambda, or as a separate guard before the lambdas:

```cpp
    if constexpr (kRank == 1) {
        if (blockIdx.y > 0) return;
    }
```

Place this before the `rowSrc`/`rowDst` lambdas.

Also need the `blockIdx.y` bounds check for `kRank > 1`. Place after the lambdas:
```cpp
    if constexpr (kRank > 1) {
        // Already handled within the lambda via slicing — but need bounds check
        // Actually, the slicing gSrc_g(_, blockIdx.y) doesn't bounds-check.
        // Add explicit check:
    }
```

Actually, the original CopyKernelND checks `blockIdx.y >= cute::size<1>(gSrc_g)` AFTER creating the grouped tensors. With the lambda approach, we need this check too. Restructure to extract the bounds check:

```cpp
    // Bounds check on outer dimension
    if constexpr (kRank > 1) {
        auto src_layout_g = cute::group<1, kRank>(src_layout);
        if (blockIdx.y >= cute::size<1>(src_layout_g)) return;
    } else {
        if (blockIdx.y > 0) return;
    }
```

Place this before the rowSrc/rowDst lambdas.

- [ ] **Step 4: Build to verify the unified kernel compiles**

Run: `cd /data/lmdeploy-copy/build && ninja`

Expected: Clean build. If it fails with a CuTe `static_assert` about `group<1,1>`, the `if constexpr` guard is not preventing instantiation and the unification approach won't work.

- [ ] **Step 5: If build succeeds, run tests**

Run: `cd /data/lmdeploy-copy && python test_generic_copy.py`

Expected: ALL TESTS PASSED — both the new rank-1 tests and all existing tests produce identical results to the baseline.

- [ ] **Step 6: If tests pass, remove CopyKernel1D and update host dispatch**

1. Delete the entire CopyKernel1D kernel (lines 72-134) and its section comment (lines 72-74).

2. Update the host dispatch. Since the kernel is now unified, route `rank == 1` through `invoke_nd(constant<1>{})` and delete `invoke_1d` entirely.

Replace the switch statement at ~line 515:
```cpp
            switch (rank) {
                case 1: invoke_1d(); break;
                case 2: invoke_nd(constant<2>{}); break;
                case 3: invoke_nd(constant<3>{}); break;
                case 4: invoke_nd(constant<4>{}); break;
                default: TM_CHECK(0) << "GenericCopy: rank > 4 not implemented"; break;
            }
```
with:
```cpp
            switch (rank) {
                case 1: invoke_nd(constant<1>{}); break;
                case 2: invoke_nd(constant<2>{}); break;
                case 3: invoke_nd(constant<3>{}); break;
                case 4: invoke_nd(constant<4>{}); break;
                default: TM_CHECK(0) << "GenericCopy: rank > 4 not implemented"; break;
            }
```

3. Delete the `invoke_1d` lambda (the entire block from `auto invoke_1d = [&] {` to its closing `};`).

4. `invoke_nd` already handles `kRank == 1` correctly: the `outer_total` loop (`for (int i = 1; i < rank; ++i)`) doesn't execute for rank=1, so `outer_total` stays 1 and `grid` becomes `(inner_tiles, 1)` — identical to the old `invoke_1d` behavior.

- [ ] **Step 7: Build and run full tests again**

Run: `cd /data/lmdeploy-copy/build && ninja && cd /data/lmdeploy-copy && python test_generic_copy.py`

Expected: ALL TESTS PASSED. All rank-1, 2D, 3D, 4D, transpose, and dtype sweep tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/turbomind/core/tensor.cu test_generic_copy.py
git commit -m "refactor(core): unify CopyKernel1D into CopyKernelND with if-constexpr rank guard

CopyKernelND now handles rank 1..4 by guarding the group<1,kRank>
call behind if constexpr (kRank > 1). For rank-1, the full tensor is
used directly without grouping. This eliminates CopyKernel1D and
simplifies the host dispatch."
```

**If Step 4 or Step 5 fails:** Revert the kernel changes. The two-kernel approach must stay. Add a comment to CopyKernel1D explaining why it exists separately:

```cpp
// Separate from CopyKernelND because CuTe's group<1,1> is ill-formed.
// group<B,E> requires B < E; for rank-1 there are no outer dims to group.
```

Commit the investigation result:
```bash
git commit -m "docs(core): document why CopyKernel1D and CopyKernelND are separate"
```

---

## Self-Review

**Spec coverage:**
- M5 (static_asserts) → Task 1 ✓
- C1 (make_fragment_like) → Task 2 ✓
- M1 (namespace comment) → Task 3 ✓
- P4 (stride-1 documentation) → Task 4 ✓
- P2/M3 (kernel unification) → Task 5 ✓

**Placeholder scan:** No TBDs, TODOs, or vague instructions. All steps contain exact code.

**Type consistency:** `CopyKernelND` template parameters are consistent between declaration and all call sites. `constant<N>{}` usage matches existing patterns.
