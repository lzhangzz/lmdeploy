# GenericCopy cooperative_copy Transpose Kernel Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the manual TiledCopy + register staging in `TransposeCopyKernel` with CuTe `cooperative_copy`, and use adaptive smem padding to eliminate bank conflicts for all element sizes.

**Architecture:** The kernel receives CuTe Tensor objects, uses `zipped_divide` for per-CTA tiling, and performs two `cooperative_copy` calls (gmem→smem, smem→gmem) with adaptive padding. The host creates CuTe gmem tensors and dispatches on (dtype, MaxVecBits).

**Tech Stack:** CuTe (CUTLASS v3.9.2 headers-only), CUDA, existing `turbomind::core::Tensor/Layout` infrastructure.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/turbomind/core/tensor.cu:167-268` | Replace `TransposeCopyKernel` (lines 167-268) |
| `src/turbomind/core/tensor.cu:307-378` | Replace host transpose dispatch block (lines 307-378) |
| `test_generic_copy.py` | Existing test suite — no changes, used for verification |

---

### Task 1: Replace TransposeCopyKernel

**Files:**
- Modify: `src/turbomind/core/tensor.cu` (lines 167-268)

Replace the entire `TransposeCopyKernel` with the new `cooperative_copy`-based version.

- [ ] **Step 1: Replace the kernel (lines 167-268)**

Delete lines 167-268 (the old kernel) and insert:

```cpp
// ============================================================================
// CUDA kernel: TransposeCopyKernel (cooperative_copy 2D transpose)
// ============================================================================
template<int kTileDim, uint32_t kMaxVecBits,
         typename SrcEngine, typename SrcLayout,
         typename DstEngine, typename DstLayout>
__global__ void __launch_bounds__(256)
TransposeCopyKernel(cute::Tensor<SrcEngine, SrcLayout> src,
                    cute::Tensor<DstEngine, DstLayout> dst)
{
    using T = typename SrcEngine::value_type;
    static_assert(std::is_same_v<T, typename DstEngine::value_type>,
                  "TransposeCopyKernel: src and dst value types must match");

    // Adaptive padding: ensure byte stride is a multiple of 4 (smem bank width).
    // ceil(4 / sizeof(T)) extra elements guarantee the bank stride is coprime with 32.
    constexpr int kPadded = kTileDim + (4 + sizeof(T) - 1) / sizeof(T);

    __shared__ T smem[kTileDim * kPadded];

    // Load view: row-major padded — stride-1 on mode 0 (matches src contiguous dim)
    auto smem_w = cute::make_tensor(cute::make_smem_ptr(smem),
        cute::make_layout(cute::make_shape(cute::Int<kTileDim>{}, cute::Int<kTileDim>{}),
                          cute::make_stride(cute::Int<1>{}, cute::Int<kPadded>{})));

    // Store view: column-major padded — stride-1 on mode 1 (matches dst contiguous dim)
    auto smem_r = cute::make_tensor(cute::make_smem_ptr(smem),
        cute::make_layout(cute::make_shape(cute::Int<kTileDim>{}, cute::Int<kTileDim>{}),
                          cute::make_stride(cute::Int<kPadded>{}, cute::Int<1>{})));

    // Tile gmem tensors — inner (kTileDim, kTileDim) is static, outer is dynamic
    auto tiler = cute::make_shape(cute::Int<kTileDim>{}, cute::Int<kTileDim>{});
    auto src_tiled = cute::zipped_divide(src, tiler);
    auto dst_tiled = cute::zipped_divide(dst, tiler);

    // Bounds check on tile grid
    if (blockIdx.y >= cute::size<2>(src_tiled) ||
        blockIdx.x >= cute::size<3>(src_tiled)) return;

    // Per-CTA tile with static shape (kTileDim, kTileDim)
    auto src_tile = src_tiled(cute::_, cute::_, blockIdx.y, blockIdx.x);
    auto dst_tile = dst_tiled(cute::_, cute::_, blockIdx.y, blockIdx.x);

    // Phase 1: gmem(src) -> smem (cooperative, vectorized)
    cute::cooperative_copy<256, kMaxVecBits>(threadIdx.x, src_tile, smem_w);
    __syncthreads();

    // Phase 2: smem -> gmem(dst) (cooperative, vectorized, transposed via smem_r view)
    cute::cooperative_copy<256, kMaxVecBits>(threadIdx.x, smem_r, dst_tile);
}
```

- [ ] **Step 2: Build the core target**

Run: `cd /data/lmdeploy-copy/build && ninja core 2>&1 | tail -60`

Expected: Compilation succeeds. The kernel is not yet called from the host (that's Task 2), so only syntax/template errors could appear.

Common issues and fixes:

- **`cooperative_copy` not found:** The header `cute/algorithm/cooperative_copy.hpp` is included transitively via `cute/tensor.hpp` (line 6). If not found, add `#include <cute/algorithm/cooperative_copy.hpp>` after line 7.

- **`zipped_divide` ambiguous overload:** CuTe may need explicit namespace `cute::zipped_divide`. Already qualified in the code above.

- **`Tensor` template not found:** Use `cute::Tensor<SrcEngine, SrcLayout>` (fully qualified).

- **Static assert on smem shape:** `cooperative_copy` requires `is_static<decltype(shape(...))>`. The smem views have shape `(Int<kTileDim>, Int<kTileDim>)` — static. The gmem tiles from `zipped_divide` have inner shape `(Int<kTileDim>, Int<kTileDim>)` — also static.

- **`sizeof(T)` in constexpr context:** `T` comes from `typename SrcEngine::value_type` which is a template-dependent type. `sizeof(T)` is always constexpr for complete types. Should work.

Iterate: modify code, rebuild with `ninja core`, until clean.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "refactor(core): replace TransposeCopyKernel with cooperative_copy

- Use CuTe cooperative_copy instead of manual TiledCopy + register staging
- Pass CuTe Tensor objects instead of raw pointers + strides
- Adaptive smem padding eliminates bank conflicts for all element sizes
- Use zipped_divide for CuTe-native per-CTA tiling

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Replace host transpose dispatch

**Files:**
- Modify: `src/turbomind/core/tensor.cu` (lines 307-378)

Replace the transpose detection + dispatch block. The detection logic is the same; the dispatch creates CuTe Tensor objects and passes them to the new kernel.

- [ ] **Step 1: Replace the transpose dispatch block (lines 307-378)**

Delete lines 307-378 (from `// --- 2D transpose detection ---` to the closing `}` of the `if (is_2d_transpose ...)` block) and insert:

```cpp
    // --- 2D transpose detection ---
    constexpr int kTileDim = 32;
    bool is_2d_transpose = (rank == 2) &&
        (a.stride(0) == 1) && (b.stride(1) == 1) &&
        (a.stride(1) > 1) && (b.stride(0) > 1);

    if (is_2d_transpose &&
        a.shape(0) % kTileDim == 0 && a.shape(1) % kTileDim == 0)
    {
        // Pointer alignment -> MaxVecBits for cooperative_copy
        int64_t tr_alignment = 16;
        auto tr_data_a = src.raw_data();
        auto tr_data_b = dst.raw_data();
        tr_alignment = std::gcd(tr_alignment, reinterpret_cast<uintptr_t>(tr_data_a));
        tr_alignment = std::gcd(tr_alignment, reinterpret_cast<uintptr_t>(tr_data_b));

        int max_vec_bits = std::min(128, static_cast<int>(tr_alignment * 8));

        int32_t M = static_cast<int32_t>(a.shape(0));
        int32_t N = static_cast<int32_t>(a.shape(1));
        dim3 grid(static_cast<uint32_t>(N / kTileDim),
                  static_cast<uint32_t>(M / kTileDim));

        auto tr_dispatch_elem_size = [&](auto t) {
            using T = decltype(t);

            auto src_gmem = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<const T*>(tr_data_a)),
                cute::make_layout(cute::make_shape(M, N),
                                  cute::make_stride(cute::Int<1>{}, a.stride(1))));

            auto dst_gmem = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<T*>(tr_data_b)),
                cute::make_layout(cute::make_shape(M, N),
                                  cute::make_stride(b.stride(0), cute::Int<1>{})));

            auto tr_dispatch_vec = [&](auto v) {
                constexpr uint32_t kVB = v.value;
                auto func = kernel::TransposeCopyKernel<kTileDim, kVB,
                    decltype(src_gmem), decltype(dst_gmem)>;
                func<<<grid, 256, 0, stream>>>(src_gmem, dst_gmem);
            };

            switch (max_vec_bits) {
                case 128: tr_dispatch_vec(constant<128>{}); break;
                case 64:  tr_dispatch_vec(constant<64>{}); break;
                case 32:  tr_dispatch_vec(constant<32>{}); break;
                case 16:  tr_dispatch_vec(constant<16>{}); break;
                default:  tr_dispatch_vec(constant<8>{}); break;
            }
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

- [ ] **Step 2: Build the full extension**

Run: `cd /data/lmdeploy-copy/build && ninja core 2>&1 | tail -60`

Expected: Compilation succeeds.

Common issues and fixes:

- **`make_shape(M, N)` with int32_t:** CuTe's `make_shape` should accept runtime int32_t values producing a dynamic shape. If the compiler complains about implicit conversion, use `cute::make_shape(static_cast<int>(M), static_cast<int>(N))`.

- **`make_stride(Int<1>{}, a.stride(1))` mixed static/dynamic:** CuTe supports mixed compile-time/runtime strides. The `Int<1>{}` enables `cooperative_copy`'s `max_common_vector` to detect vectorization opportunity along mode 0.

- **Kernel launch with CuTe Tensor arguments:** CuTe Tensors are trivially copyable value types (pointer + layout). They can be passed directly as kernel arguments via `<<<>>>` syntax.

Iterate: modify code, rebuild with `ninja core`, until clean.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "refactor(core): update host dispatch for cooperative_copy transpose

- Create CuTe gmem Tensor objects in host dispatch
- Dispatch on (dtype, MaxVecBits) instead of (dtype, kVec)
- Remove vec_size >= 2 guard (cooperative_copy handles any width)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Build full extension and run tests

**Files:**
- Existing: `test_generic_copy.py`

- [ ] **Step 1: Build the _turbomind extension**

Run: `cd /data/lmdeploy-copy/build && ninja _turbomind 2>&1 | tail -30`

Expected: Full build succeeds.

- [ ] **Step 2: Check GPU availability**

Run: `python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"`

Expected: `True` and GPU name.

- [ ] **Step 3: Run the existing GenericCopy test suite**

Run: `cd /data/lmdeploy-copy && PYTHONPATH=lmdeploy:build/lib python3 test_generic_copy.py 2>&1`

Expected: ALL TESTS PASSED. The test suite covers:
- Contiguous baseline (f32) — uses existing vectorized path (NOT transpose kernel)
- 2D transpose f32 — uses new cooperative_copy TransposeCopyKernel (shape 64x128 after .t())
- 2D: row-stride, col-stride, narrow — uses existing scalar/vectorized paths
- 3D: permute (2,0,1) — uses existing scalar path (rank != 2)
- 4D: slice — uses existing scalar path (rank != 2)
- Combined: slice+transpose — uses existing scalar path
- Dtype sweep: f16, i8, i32 (transpose) — uses new TransposeCopyKernel
- Throughput sweep: 1024 to 16384 square matrices (transpose) — uses new TransposeCopyKernel

The test `check()` function compares against PyTorch's `clone()` on the non-contiguous tensor and asserts `torch.allclose`. Any numerical mismatch will fail.

- [ ] **Step 4: Fix any correctness issues**

If tests fail, investigate and fix. Most likely causes:

1. **`zipped_divide` tile indexing:** The 4D result of `zipped_divide` on a 2D tensor is `(inner0, inner1, outer0, outer1)`. Verify that `(_, _, blockIdx.y, blockIdx.x)` maps blockIdx.y to row tiles and blockIdx.x to column tiles, matching the grid `dim3(N/kTileDim, M/kTileDim)`. Grid: x=N/32, y=M/32. So blockIdx.y is the row tile, blockIdx.x is the column tile. For `zipped_divide`, mode 0 (rows) tiles into (inner0, outer0) = (32, M/32), mode 1 (cols) tiles into (inner1, outer1) = (32, N/32). So `(_, _, blockIdx.y, blockIdx.x)` = `(inner0, inner1, outer0, outer1)` — correct.

2. **`cooperative_copy` alignment assert:** If pointers are not aligned to `kMaxVecBits/8` bytes, cooperative_copy may trigger a runtime assert. Verify the host computes `max_vec_bits` correctly from pointer alignment.

3. **Smem bank conflicts for sub-word types:** If adaptive padding doesn't eliminate conflicts, check that `kPadded` is computed correctly. For `uint8_t` (1 byte): `kPadded = 32 + (4+0)/1 = 36`. Smem size: 32*36 = 1152 bytes. Byte stride: 36. Banks: 9. 9 is coprime with 32 — no conflicts.

4. **`constant<N>` not found:** The `constant` type alias should be defined in the existing code. If not, use `cute::Int<N>{}` directly or `std::integral_constant<uint32_t, N>{}`.

Iterate: modify code, rebuild (`ninja core && ninja _turbomind`), re-run tests until all pass.

- [ ] **Step 5: Commit any fixes**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "fix(core): address cooperative_copy transpose issues from testing

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Performance validation

**Files:**
- Existing: `test_generic_copy.py` (includes throughput benchmarking)

- [ ] **Step 1: Compare throughput numbers**

The test output shows throughput in GB/s for both GenericCopy and PyTorch. Check:

- **2D transpose (f32, 4096x4096 and larger):** Should maintain or exceed current ~1300 GB/s
- **2D transpose dtype sweep (f16, i8, i32):** Check that sub-word types (f16, i8) show improvement from bank conflict elimination
- **Contiguous copies:** Should be unchanged (uses existing vectorized path)

- [ ] **Step 2: Address any regressions**

If throughput regresses for transpose:

1. **Check cooperative_copy vectorization:** Add a temporary `printf` inside the kernel to verify `kMaxVecBits`:
   ```cpp
   if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
       printf("TransposeCopyKernel: kMaxVecBits=%u, kPadded=%d, sizeof(T)=%zu\n",
              kMaxVecBits, kPadded, sizeof(T));
   ```

2. **Check coalescing:** Use `cuobjdump -sass` on the compiled kernel to verify vector loads/stores are used.

3. **Check bank conflicts:** For sub-word types, verify the adaptive padding eliminates conflicts by comparing smem load/store throughput.

- [ ] **Step 3: Final commit if any performance fixes were needed**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "perf(core): optimize cooperative_copy transpose performance

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
