# GenericCopy CuTe Idiom Audit

**Date:** 2026-04-12
**Scope:** `src/turbomind/core/tensor.cu` — all kernels, host dispatch, layout helpers
**References:**
- CuTe docs: `build/_deps/repo-cutlass-src/media/docs/cpp/cute/`
- CuTe examples: `build/_deps/repo-cutlass-src/examples/cute/tutorial/`
- CuTe headers: `build/_deps/repo-cutlass-src/include/cute/`

---

## Methodology

Read all CuTe documentation (layouts, tensors, algorithms, predication), the `tiled_copy.cu` example, and the `sgemm_1.cu` GEMM tutorial. Compared against every CuTe API usage in `tensor.cu`. Verified internal dispatch behavior by reading CuTe header sources (`copy.hpp`, `copy_traits.hpp`, `copy_atom.hpp`, `layout.hpp`, `tensor_impl.hpp`, `underscore.hpp`, `tuple_algorithms.hpp`).

---

## Tier 1: Correctness Risks

### C1: `make_fragment_like` used for smem register buffers — verify layout compatibility

**Location:** `TransposeCopyKernel` lines 267, 290

**What the code does:**
```cpp
auto rmem_ld = cute::make_fragment_like(thr_smw);
auto rmem_st = cute::make_fragment_like(thr_smw2);
```

**CuTe behavior:** `make_fragment_like` forces mode-0 to column-major stride-1 (`compact_major<LayoutLeft>`), regardless of the source tensor's stride ordering. Modes 1+ follow the source's ordering. This differs from `make_tensor_like`, which preserves the full source ordering.

**Risk:** If `thr_smw`'s mode-0 is already stride-1 (column-major), the fragment layout matches and data transfer is correct. If not, the fragment would have a different layout than `thr_smw`, and the subsequent `copy(tiled_copy, src, rmem)` would map by logical coordinate — still correct, but the manual `rmem(i) = smem(i)` loop on lines 272-275 would transfer by flat index, which DOES depend on layout matching.

**Assessment:** In this kernel, `thr_smw` is partitioned from a column-major smem layout (`Stride<Int<1>, Int<kTileDim+1>>`), so mode-0 is stride-1. `make_fragment_like` produces the same result as `make_tensor_like`. The code is correct but fragile — a future change to the smem layout could silently break it.

**Recommendation:** Add a comment explaining why `make_fragment_like` is correct here (smem mode-0 is already stride-1). Alternatively, switch to `make_tensor_like` for robustness.

---

### C2: CopyKernel1D has unreachable bounds check

**Location:** `CopyKernel1D` line 101

**What the code does:** `if (blockIdx.y > 0) return;`

**Assessment:** The host always launches with `grid(inner_tiles, 1)` — `blockIdx.y` is always 0. The guard is defensive and correct.

**Recommendation:** Accept as-is. Safety net.

---

### C3: Manual smem↔rmem transfer bypasses CuTe copy

**Location:** `TransposeCopyKernel` lines 272-275, 293-296

**What the code does:**
```cpp
CUTE_UNROLL
for (int i = 0; i < cute::size(thr_smw); ++i) {
    thr_smw(i) = rmem_ld(i);
}
```

**CuTe idiom:** `copy(fragment, smem_tensor)` would dispatch to an optimized path based on memory spaces.

**Assessment:** The comment says "avoids CuTe auto-vectorization on smem." CuTe's `AutoVectorizingCopy` (which is the default for smem access) assumes 128-bit alignment and may generate wide vector loads/stores that could cause bank conflicts on smem. The manual scalar transfer is a deliberate safety measure.

**Recommendation:** Accept as intentional. Consider whether `DefaultCopy` (assumes 8-bit alignment only) via `Copy_Atom<DefaultCopy, T>` could replace the manual loop while remaining safe.

---

## Tier 2: Performance Concerns

### P1: Only innermost stride is compile-time — this is correct and optimal

**Location:** `detail::make_cute_layout_unit_inner` lines 56-68

**What:** Constructs layouts with `Int<1>` innermost stride, all other strides and shapes are runtime.

**Assessment:** CuTe's `Copy_Atom` vectorization requires `Int<1>` on the contiguous stride (verified by reading `copy_traits.hpp` — `recast` calls `downcast` which requires `has_int1<Stride>`). Making outer dimensions compile-time would enable more loop unrolling but is infeasible for generic copy. The current approach provides exactly the compile-time information needed for vectorized loads/stores.

**Recommendation:** Accept as optimal for the generic case.

---

### P2: CopyKernel1D and CopyKernelND could potentially be unified

**Location:** `CopyKernel1D` (lines 77-134), `CopyKernelND` (lines 139-203)

**What:** Two nearly identical kernels. `CopyKernelND` calls `group<1, kRank>` which is invalid for rank-1.

**Gap:** The only structural difference is that CopyKernelND calls `group<1,kRank>` and uses `blockIdx.y` for outer rows. If rank-1 were handled by skipping the `group` call, a single kernel could serve both cases.

**Assessment:** The host already dispatches `rank==1` to `CopyKernel1D` and `rank>=2` to `CopyKernelND`. Unification is a maintainability improvement, not a performance issue. The existing `simplify-dispatch` design doc may already address this.

**Recommendation:** Investigate whether `if constexpr (kRank > 1) { auto src_g = group<1,kRank>(src_layout); ... }` would compile correctly and produce equivalent code. If yes, merge into a single kernel.

---

### P3: kVec=1 scalar path uses full predication — necessary overhead

**Location:** `CopyKernel1D` lines 118-132, `CopyKernelND` lines 187-201

**What:** The kVec=1 path constructs identity tensor, tiles it, partitions it, and builds predicate tensor for boundary handling.

**Assessment:** This is the textbook CuTe predication pattern from `0y_predication.md`. The scalar path is only reached when the host's alignment computation determined vectorization is impossible (non-divisible shape, unaligned pointer, non-unit inner stride). In these cases, predication is mandatory.

**Recommendation:** Accept as necessary and idiomatic.

---

### P4: Conservative alignment — no vectorization along non-contiguous dimensions

**Location:** `GenericCopy` lines 419-420

**What:**
```cpp
if (a.stride(0) > 1 || b.stride(0) > 1) {
    alignment = byte_size(dtype);
}
```

**Gap:** When the innermost dimension (after stride-sorting) doesn't have stride-1, the code falls back to scalar copies. Vectorization along an outer dimension would require a fundamentally different kernel architecture (treating a strided dimension as contiguous after layout transformation).

**Recommendation:** Accept the limitation. Document that `GenericCopy` requires at least one dimension with stride-1 in either src or dst for vectorized copies.

---

### P5: Transpose kernel uses two TiledCopy objects — correct pattern

**Location:** `TransposeCopyKernel` lines 258-285

**What:** `load_copy` vectorizes along mode 0 (gmem read), `store_copy` vectorizes along mode 1 (gmem write). Different thread-value layouts.

**Assessment:** This is the correct pattern for a tiled transpose — the load and store phases need different vectorization directions. CuTe examples don't show this specific pattern (no transpose example exists), but the usage of `make_tiled_copy` + `get_slice` + `partition_S/D` is textbook.

**Recommendation:** Accept as correct.

---

## Tier 3: Maintainability

### M1: Runtime layout helpers — necessary but non-obvious

**Location:** `detail` namespace lines 22-68

**What:** `make_cute_shape_impl`, `make_cute_stride_impl`, `make_cute_layout`, `make_cute_layout_unit_inner` use `std::index_sequence` to expand runtime arrays into CuTe's variadic `make_shape`/`make_stride`.

**Gap:** CuTe examples never do this — they use compile-time literals. A CuTe-experienced reader may not immediately understand why these helpers exist.

**Assessment:** These are correct and necessary. CuTe's layout construction requires variadic template arguments; our shapes are runtime. The `index_sequence` trick is the standard C++ way to bridge this gap. The existing comments (especially line 54-55 about `Int<1>` inner stride) are helpful.

**Recommendation:** Add one comment to the `detail` namespace explaining the pattern: "CuTe's `make_shape`/`make_stride` require compile-time variadic args, but our shapes are runtime. These helpers bridge that gap via `index_sequence` expansion."

---

### M2: Three-level dispatch nesting in GenericCopy

**Location:** `GenericCopy` lines 451-538

**What:** Three nested `switch` + lambda dispatchers: dtype → vec_size → rank.

**Gap:** Each level follows the identical pattern `auto dispatch_X = [&](auto x) { switch(Y) { ... } }`. The nesting makes it hard to follow which (T, kVec, kRank) combination is being compiled.

**Assessment:** The pattern works and is common in CuTe-based code. However, the lambda captures (`[&]`) create a deep reference chain.

**Recommendation:** Consider whether the existing `simplify-dispatch` refactoring plan already addresses this. If not, a flat dispatch table mapping `(dtype, vec_size, rank) → kernel instantiation` would be more readable.

---

### M3: 1D and ND host paths duplicate grid computation

**Location:** `invoke_1d` (lines 457-479), `invoke_nd` (lines 482-513)

**What:** `invoke_1d` computes `grid(inner_tiles, 1)`, `invoke_nd` computes `grid(inner_tiles, outer_total)`. The rest of the setup (layout construction, kernel launch) is identical.

**Assessment:** Same as P2. If kernels are unified, host paths unify too.

**Recommendation:** Address alongside P2.

---

### M4: Transpose kernel uses raw strides, not CuTe layouts, as kernel arguments

**Location:** `TransposeCopyKernel` signature lines 210-215

**What:** Passes `int64_t src_stride_outer, int64_t dst_stride_outer` as kernel arguments, then constructs layouts inside the kernel with partially-static shapes.

**Gap:** Inconsistent with CopyKernel1D/ND which pass CuTe layout objects as template-type parameters.

**Assessment:** The transpose kernel's layout is mixed static-dynamic (`Shape<Int<kTileDim>, Int<kTileDim>>, Stride<Int<1>, int64_t>`), which is harder to construct on the host. The current approach of passing the dynamic part as a kernel arg is simpler.

**Recommendation:** Accept as intentional. The inconsistency reflects a real difference in layout complexity.

---

### M5: No compile-time preconditions in kernels

**Location:** All three kernels

**What:** CuTe examples extensively use `CUTE_STATIC_ASSERT_V` to verify layout properties. Our kernels have no such checks.

**Gap:**
- `CopyKernel1D`: No assertion that layout rank is 1
- `CopyKernelND`: No assertion that `kRank >= 2`
- `TransposeCopyKernel`: No assertion that `kVec >= 2` (guarded by `if constexpr`, but an explicit assertion would be clearer)

**Recommendation:** Add preconditions:
```cpp
// CopyKernelND
static_assert(kRank >= 2 && kRank <= 4, "CopyKernelND: rank must be 2..4");

// TransposeCopyKernel — already has if constexpr guard, add:
static_assert(kTileDim % kVec == 0, "kTileDim must be divisible by kVec");
```

---

### M6: Naming convention — acceptable

**What:** `CopyKernel1D`, `CopyKernelND`, `TransposeCopyKernel` vs CuTe's `copy_kernel`, `copy_kernel_vectorized`.

**Assessment:** Names are clear within the project's `turbomind::core::kernel` namespace.

**Recommendation:** Accept as-is.

---

## Summary Table

| ID | Severity | Finding | Verdict |
|----|----------|---------|---------|
| C1 | Correctness | `make_fragment_like` layout compatibility in transpose | Verify + document |
| C2 | Correctness | Unreachable bounds check in CopyKernel1D | Accept |
| C3 | Correctness | Manual smem transfer bypasses CuTe copy | Accept (intentional) |
| P1 | Performance | Only inner stride is compile-time | Accept (optimal) |
| P2 | Performance | Duplicate CopyKernel1D/ND kernels | Investigate unification |
| P3 | Performance | kVec=1 predication overhead | Accept (necessary) |
| P4 | Performance | Conservative alignment, no non-contiguous vectorization | Accept + document |
| P5 | Performance | Two TiledCopy for transpose | Accept (correct) |
| M1 | Maintainability | Runtime layout helpers not idiomatic | Accept + improve comments |
| M2 | Maintainability | Three-level dispatch nesting | Consider flattening |
| M3 | Maintainability | 1D/ND host paths duplicate logic | Address with P2 |
| M4 | Maintainability | Transpose uses raw strides not layouts | Accept (intentional) |
| M5 | Maintainability | No compile-time preconditions | Add assertions |
| M6 | Maintainability | Naming convention | Accept |

---

## Actionable items (prioritized)

1. **M5** — Add `static_assert` preconditions to all three kernels (low effort, high value)
2. **C1** — Verify `make_fragment_like` layout match in TransposeCopyKernel, add comment (low effort, safety)
3. **P2/M3** — Investigate unifying CopyKernel1D + CopyKernelND (medium effort, reduces duplication)
4. **M1** — Add namespace comment to layout helpers (trivial)
5. **P4** — Document the stride-1 vectorization requirement (trivial)
