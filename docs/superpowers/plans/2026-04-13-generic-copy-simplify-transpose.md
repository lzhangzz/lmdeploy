# TransposeCopyKernel Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify TransposeCopyKernel to use scalar-only cooperative_copy with two naive smem views (no padding) and collapse the host dispatch.

**Architecture:** Remove smem padding and auto-vectorization. Use two views of a flat smem buffer — row-major write view for Phase 1, column-major read view for Phase 2. The transpose happens implicitly via the view swap. Host dispatch collapses from a 5-way vec-bits switch to a direct launch with `8*sizeof(T)`.

**Tech Stack:** CUDA, CuTe (CUTLASS), C++17

**Spec:** `docs/superpowers/specs/2026-04-13-generic-copy-simplify-transpose-design.md`

---

### Task 1: Simplify kernel body

**Files:**
- Modify: `src/turbomind/core/tensor.cu:167-219`

- [ ] **Step 1: Edit the kernel body**

Replace lines 182–218 (everything inside the kernel function body) with:

```cpp
    using T = typename SrcEngine::value_type;
    static_assert(std::is_same_v<T, typename DstEngine::value_type>,
                  "TransposeCopyKernel: src and dst value types must match");

    __shared__ T smem[kTileDim * kTileDim];

    // Write view: row-major (stride-1 on dim 0, matches src)
    auto smem_w = cute::make_tensor(cute::make_smem_ptr(smem),
        cute::make_layout(cute::make_shape(cute::Int<kTileDim>{}, cute::Int<kTileDim>{}),
                          cute::make_stride(cute::Int<1>{}, cute::Int<kTileDim>{})));

    // Read view: column-major (stride-1 on dim 0, matches dst) — transposed
    auto smem_r = cute::make_tensor(cute::make_smem_ptr(smem),
        cute::make_layout(cute::make_shape(cute::Int<kTileDim>{}, cute::Int<kTileDim>{}),
                          cute::make_stride(cute::Int<kTileDim>{}, cute::Int<1>{})));

    // Tile gmem tensors — inner (kTileDim, kTileDim) is static, outer is dynamic
    auto tiler = cute::make_shape(cute::Int<kTileDim>{}, cute::Int<kTileDim>{});
    auto src_tiled = cute::zipped_divide(src, tiler);
    auto dst_tiled = cute::zipped_divide(dst, tiler);

    // Bounds check on tile grid — zipped_divide produces ((inner0,inner1),(outer0,outer1))
    if (blockIdx.y >= cute::size<1, 0>(src_tiled) ||
        blockIdx.x >= cute::size<1, 1>(src_tiled)) return;

    // Per-CTA tile with static shape (kTileDim, kTileDim)
    auto src_tile = src_tiled(cute::_,
                              cute::make_coord(blockIdx.y, blockIdx.x));
    auto dst_tile = dst_tiled(cute::_,
                              cute::make_coord(blockIdx.y, blockIdx.x));

    // Phase 1: gmem(src) -> smem (row-major)
    cute::cooperative_copy<256, kMaxVecBits>(threadIdx.x, src_tile, smem_w);
    __syncthreads();
    // Phase 2: smem -> gmem(dst) (column-major view — transposed)
    cute::cooperative_copy<256, kMaxVecBits>(threadIdx.x, smem_r, dst_tile);
```

Also update the block comment above the kernel (lines 170–174) to reflect the new design:

```cpp
// Copies a 2D tensor from src to dst where src and dst have orthogonal
// contiguous dimensions (src contiguous on dim 0, dst contiguous on dim 1).
// Uses smem staging with two views of a flat buffer: row-major write view
// for Phase 1, column-major read view for Phase 2 (implicit transpose).
// Both phases use kMaxVecBits (scalar = 8*sizeof(T) from host).
```

- [ ] **Step 2: Build to verify compilation**

Run: `ninja -C build _turbomind`
Expected: clean compile with no errors

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "refactor(core): simplify TransposeCopyKernel — dual smem views, scalar cooperative_copy"
```

---

### Task 2: Simplify host dispatch

**Files:**
- Modify: `src/turbomind/core/tensor.cu:258-318`

- [ ] **Step 1: Edit the transpose dispatch block**

Replace lines 266–317 (the body of the `if (is_2d_transpose && ...)` block) with:

```cpp
    {
        auto tr_data_a = src.raw_data();
        auto tr_data_b = dst.raw_data();

        int32_t M = static_cast<int32_t>(a.shape(0));
        int32_t N = static_cast<int32_t>(a.shape(1));
        dim3 grid(static_cast<uint32_t>(N / kTileDim),
                  static_cast<uint32_t>(M / kTileDim));

        auto tr_dispatch_elem_size = [&](auto t) {
            using T = decltype(t);
            constexpr uint32_t kVB = 8 * sizeof(T);

            auto src_gmem = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<const T*>(tr_data_a)),
                cute::make_layout(cute::make_shape(M, N),
                                  cute::make_stride(cute::Int<1>{}, a.stride(1))));

            auto dst_gmem = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<T*>(tr_data_b)),
                cute::make_layout(cute::make_shape(M, N),
                                  cute::make_stride(b.stride(0), cute::Int<1>{})));

            using SrcE = typename decltype(src_gmem)::engine_type;
            using SrcL = typename decltype(src_gmem)::layout_type;
            using DstE = typename decltype(dst_gmem)::engine_type;
            using DstL = typename decltype(dst_gmem)::layout_type;
            kernel::TransposeCopyKernel<kTileDim, kVB, SrcE, SrcL, DstE, DstL>
                <<<grid, 256, 0, stream>>>(src_gmem, dst_gmem);
        };

        switch (byte_size(dtype)) {
            case 1: return tr_dispatch_elem_size(uint8_t{});
            case 2: return tr_dispatch_elem_size(uint16_t{});
            case 4: return tr_dispatch_elem_size(uint32_t{});
            default:
                TM_CHECK(0) << "GenericCopy: unsupported element size " << byte_size(dtype);
                break;
        }
    }
```

- [ ] **Step 2: Build to verify compilation**

Run: `ninja -C build _turbomind`
Expected: clean compile with no errors

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "refactor(core): collapse transpose dispatch — remove alignment computation and vec-bits switch"
```

---

### Task 3: Test correctness

**Files:**
- Test: `test_generic_copy.py`

- [ ] **Step 1: Run existing test suite**

Run: `cd /data/lmdeploy-copy && python test_generic_copy.py`
Expected: ALL TESTS PASSED — all transpose cases (f32, f16, i8, i32, various sizes) must pass with correct results

- [ ] **Step 2: Verify transpose test outputs specifically**

Check that these test cases in particular produce PASS:
- `transpose f32` (64×128)
- `transpose f16` (64×128)
- `transpose i8` (64×128)
- `transpose i32` (64×128)
- `trans 1M (1024x1024)` through `trans 256M (16384x16384)`
- `slice+transpose`
