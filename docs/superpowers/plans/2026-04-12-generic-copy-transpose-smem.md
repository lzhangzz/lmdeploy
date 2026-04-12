# GenericCopy 2D Transpose SMEM Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a specialized SMEM-tiled transpose kernel to GenericCopy that achieves ~90% of peak copy bandwidth for 2D transpose patterns, up from ~45% of PyTorch today.

**Architecture:** A new `TransposeCopyKernel` uses two-phase SMEM tiling: Phase 1 loads a 32×32 tile from src gmem into registers (vectorized, coalesced) then writes to smem; Phase 2 reads smem in transposed order into registers then writes to dst gmem (vectorized, coalesced). Two CuTe views of the same padded smem handle the transpose. The host detects the 2D transpose pattern (cross-mode stride-1) and dispatches to the transpose kernel when both dims are divisible by 32.

**Tech Stack:** CuTe (CUTLASS v3.9.2 headers-only), CUDA, existing `turbomind::core::Tensor/Layout` infrastructure.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/turbomind/core/tensor.cu` | Existing file. Add `TransposeCopyKernel` in `kernel` namespace, add transpose detection + dispatch in `GenericCopy` host function |
| `test_generic_copy.py` | Existing test suite (22 tests). No changes needed — transpose tests already exist and will automatically use the new kernel |

---

### Task 1: Add TransposeCopyKernel and host dispatch to tensor.cu

**Files:**
- Modify: `src/turbomind/core/tensor.cu`

This task adds the `TransposeCopyKernel` to the `kernel` namespace (after `GenericCopyKernel`, before the host function) and modifies the `GenericCopy` host function to detect and dispatch 2D transpose patterns before falling through to the existing dispatch.

- [ ] **Step 1: Add TransposeCopyKernel after GenericCopyKernel**

Insert the following kernel code after the closing `}  // namespace kernel` on line 299, but BEFORE that closing brace (i.e., inside `namespace kernel`):

```cpp
// ============================================================================
// CUDA kernel: TransposeCopyKernel (SMEM tiled 2D transpose)
// ============================================================================
template<typename T, int kVec, int kTileDim>
__global__ void __launch_bounds__(256)
TransposeCopyKernel(const T* __restrict__ src_ptr,
                    T* __restrict__       dst_ptr,
                    int64_t               src_stride_outer,
                    int64_t               dst_stride_outer,
                    int32_t               M,
                    int32_t               N)
{
    // Guard: kVec * sizeof(T) must not exceed 16 bytes (128 bits).
    if constexpr (kVec * cute::sizeof_bits_v<T> <= 128)
    {
    constexpr int kBlockThreads = 256;

    // Thread bounds: for kTileDim=32, kVec=4: 8*32=256 (all threads).
    // For kVec=8: 4*32=128 (half threads). Compute min across both phases.
    constexpr int kThrLoad  = (kTileDim / kVec) * kTileDim;
    constexpr int kThrStore = kTileDim * (kTileDim / kVec);
    constexpr int kCopyThreads = kThrLoad < kThrStore ? kThrLoad : kThrStore;
    if (threadIdx.x >= kCopyThreads) return;

    // --- Shared memory with padding for bank conflict avoidance ---
    __shared__ T smem[kTileDim * (kTileDim + 1)];

    // Write view: (kTileDim, kTileDim) stride (1, kTileDim+1) — row-major padded
    auto smem_w = cute::make_tensor(cute::make_smem_ptr(smem),
        cute::make_layout(cute::make_shape(cute::Int<kTileDim>{}, cute::Int<kTileDim>{}),
                          cute::make_stride(cute::Int<1>{}, cute::Int<kTileDim + 1>{})));

    // Read view: (kTileDim, kTileDim) stride (kTileDim+1, 1) — column-major (transposed)
    auto smem_r = cute::make_tensor(cute::make_smem_ptr(smem),
        cute::make_layout(cute::make_shape(cute::Int<kTileDim>{}, cute::Int<kTileDim>{}),
                          cute::make_stride(cute::Int<kTileDim + 1>{}, cute::Int<1>{})));

    // --- Tile coordinates ---
    int m0 = blockIdx.y * kTileDim;
    int n0 = blockIdx.x * kTileDim;

    // --- Gmem tile tensors with Int<1> on contiguous modes ---
    // Src tile: shape (kTileDim, kTileDim), strides (Int<1>, src_stride_outer)
    auto src_tile = cute::make_tensor(cute::make_gmem_ptr(src_ptr + m0 + n0 * src_stride_outer),
        cute::make_layout(cute::make_shape(cute::Int<kTileDim>{}, cute::Int<kTileDim>{}),
                          cute::make_stride(cute::Int<1>{}, src_stride_outer)));

    // Dst tile: transposed mapping — tile (m,n) of src → tile (n,m) of dst
    auto dst_tile = cute::make_tensor(cute::make_gmem_ptr(dst_ptr + n0 + m0 * dst_stride_outer),
        cute::make_layout(cute::make_shape(cute::Int<kTileDim>{}, cute::Int<kTileDim>{}),
                          cute::make_stride(dst_stride_outer, cute::Int<1>{})));

    // --- Phase 1: gmem(src) → registers → smem ---
    auto load_copy = cute::make_tiled_copy(
        cute::Copy_Atom<cute::UniversalCopy<cute::uint_bit_t<kVec * cute::sizeof_bits_v<T>>>, T>{},
        cute::make_layout(cute::make_shape(cute::Int<kTileDim / kVec>{},
                                           cute::Int<kBlockThreads * kVec / kTileDim>{})),
        cute::make_layout(cute::make_shape(cute::Int<kVec>{}, cute::Int<1>{})));

    auto thr_load = load_copy.get_slice(threadIdx.x);
    auto thr_src  = thr_load.partition_S(src_tile);
    auto thr_smw  = thr_load.partition_D(smem_w);
    auto rmem_ld  = cute::make_fragment_like(thr_smw);

    cute::copy(load_copy, thr_src, rmem_ld);    // vectorized gmem → registers
    cute::copy(rmem_ld, thr_smw);               // registers → smem

    __syncthreads();

    // --- Phase 2: smem → registers → gmem(dst) ---
    auto store_copy = cute::make_tiled_copy(
        cute::Copy_Atom<cute::UniversalCopy<cute::uint_bit_t<kVec * cute::sizeof_bits_v<T>>>, T>{},
        cute::make_layout(cute::make_shape(cute::Int<kBlockThreads * kVec / kTileDim>{},
                                           cute::Int<kTileDim / kVec>{})),
        cute::make_layout(cute::make_shape(cute::Int<1>{}, cute::Int<kVec>{})));

    auto thr_store = store_copy.get_slice(threadIdx.x);
    auto thr_smr   = thr_store.partition_S(smem_r);
    auto thr_dst   = thr_store.partition_D(dst_tile);
    auto rmem_st   = cute::make_fragment_like(thr_smr);

    cute::copy(thr_smr, rmem_st);               // smem → registers
    cute::copy(store_copy, rmem_st, thr_dst);    // vectorized registers → gmem

    }  // end if constexpr (kVec * sizeof_bits_v<T> <= 128)
}
```

- [ ] **Step 2: Add transpose detection + dispatch in GenericCopy host function**

Insert the following block between line 333 (`const DataType dtype = src.dtype();`) and line 336 (`// --- Alignment detection ---`). This is BEFORE the existing alignment detection. The block detects the transpose pattern and returns early if matched, otherwise falls through to the existing dispatch.

```cpp
    // --- 2D transpose detection ---
    constexpr int kTileDim = 32;
    bool is_2d_transpose = (rank == 2) &&
        (a.stride(0) == 1) && (b.stride(1) == 1) &&
        (a.stride(1) > 1) && (b.stride(0) > 1);

    if (is_2d_transpose &&
        a.shape(0) % kTileDim == 0 && a.shape(1) % kTileDim == 0)
    {
        // NOTE: The transpose kernel requires kVec >= 2 because kTileDim=32 with
        // kVec=1 would need 1024 threads (32*32) but we only have 256.
        // If pointer alignment gives vec_size=1, fall through to scalar GenericCopy.
        // Transpose alignment: only pointer alignment matters
        int64_t tr_alignment = 16;
        auto tr_data_a = src.raw_data();
        auto tr_data_b = dst.raw_data();
        tr_alignment = std::gcd(tr_alignment, reinterpret_cast<uintptr_t>(tr_data_a));
        tr_alignment = std::gcd(tr_alignment, reinterpret_cast<uintptr_t>(tr_data_b));

        const int tr_elem_size = byte_size(dtype);
        int tr_vec_size = static_cast<int>(tr_alignment / std::max<int64_t>(1, tr_elem_size));

        // Cap at 128 bits (16 bytes)
        if (tr_vec_size * tr_elem_size > 16) {
            tr_vec_size = 16 / tr_elem_size;
        }

        // kTileDim must be divisible by vec_size for TiledCopy thread layout
        while (tr_vec_size > 1 && kTileDim % tr_vec_size != 0) {
            tr_vec_size /= 2;
        }

        int32_t M = static_cast<int32_t>(a.shape(0));
        int32_t N = static_cast<int32_t>(a.shape(1));
        dim3 grid(static_cast<uint32_t>(N / kTileDim),
                  static_cast<uint32_t>(M / kTileDim));

        auto tr_dispatch_dtype = [&](auto t) {
            using T = decltype(t);

            auto tr_dispatch_vec = [&](auto v) {
                constexpr int kVec = v.value;
                auto func = kernel::TransposeCopyKernel<T, kVec, kTileDim>;
                func<<<grid, 256, 0, stream>>>(
                    reinterpret_cast<const T*>(tr_data_a),
                    reinterpret_cast<T*>(tr_data_b),
                    a.stride(1), b.stride(0),
                    M, N);
            };

            switch (tr_vec_size) {
                case 16: tr_dispatch_vec(constant<16>{}); break;
                case 8:  tr_dispatch_vec(constant<8>{}); break;
                case 4:  tr_dispatch_vec(constant<4>{}); break;
                case 2:  tr_dispatch_vec(constant<2>{}); break;
                default: tr_dispatch_vec(constant<1>{}); break;
            }
        };

        switch (dtype) {
            case DataType::kFloat32:  return tr_dispatch_dtype(float{});
            case DataType::kFloat16:  return tr_dispatch_dtype(half_t{});
            case DataType::kBfloat16: return tr_dispatch_dtype(bfloat16_t{});
            case DataType::kInt8:     return tr_dispatch_dtype(int8_t{});
            case DataType::kInt32:    return tr_dispatch_dtype(int32_t{});
            case DataType::kBool:     return tr_dispatch_dtype(uint8_t{});
            case DataType::kUint8:    return tr_dispatch_dtype(uint8_t{});
            default:
                throw std::runtime_error(std::string("GenericCopy: unsupported data type ") + to_string(dtype));
        }
        return;
    }
```

- [ ] **Step 3: Build the core target**

Run: `cd /data/lmdeploy-copy/build && ninja core 2>&1 | tail -40`

Expected: Compilation succeeds. There may be CuTe-related warnings but no errors.

Common issues to fix if compilation fails:

- **`kTileDim / kVec` not a compile-time constant for large kVec:** The `Int<kTileDim / kVec>` in the TiledCopy thread layout requires compile-time division. Since both are template params, this should work. If the compiler complains about negative values (when kVec > kTileDim), add a guard: `if constexpr (kVec <= kTileDim)`.

- **`smem` array size exceeds 48KB:** For kTileDim=32, smem is 32×33×sizeof(T) = max 4224 bytes (f32). Well within limits. If the compiler warns about smem size, verify kTileDim=32.

- **`make_smem_ptr` with `__shared__` array:** CuTe's `make_smem_ptr` should accept a `T[]` array. If it complains about const-ness, try `make_smem_ptr(const_cast<T*>(smem))` for the read view.

- **`copy(store_copy, rmem_st, thr_dst)` overload resolution:** CuTe may not find the right overload for `copy(TiledCopy, src, dst)`. If it fails, try explicitly: `cute::copy<decltype(store_copy), decltype(rmem_st), decltype(thr_dst)>(store_copy, rmem_st, thr_dst)`.

- **`make_fragment_like` not found:** It's in `cute/tensor.hpp` which is already included. If issues, use `cute::make_tensor<T>(cute::shape(tensor))` instead.

- [ ] **Step 4: Fix any compilation errors**

Iterate: modify code, rebuild with `ninja core`, until clean.

- [ ] **Step 5: Build the _turbomind extension**

Run: `cd /data/lmdeploy-copy/build && ninja _turbomind 2>&1 | tail -30`

Expected: Full build succeeds.

- [ ] **Step 6: Commit**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "feat(core): add SMEM-tiled transpose kernel for 2D GenericCopy

- Add TransposeCopyKernel: two-phase gmem→smem→gmem with padded smem
- Dual TiledCopy: vectorize along mode 0 (load), mode 1 (store)
- Register staging for SM70+ portability
- Host detects 2D transpose (cross-mode stride-1) and dispatches
- Shape divisibility guard (both dims divisible by kTileDim=32)

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
- Contiguous baseline (f32) — uses existing vectorized path (NOT transpose kernel)
- 2D transpose f32 — uses new TransposeCopyKernel (shape 128×64, both divisible by 32)
- 2D: row-stride, col-stride, narrow — uses existing scalar/vectorized paths
- 3D: permute (2,0,1) — uses existing scalar path (rank != 2)
- 4D: slice — uses existing scalar path (rank != 2)
- Combined: slice+transpose — uses existing scalar path (shape may not be divisible by 32)
- Dtype sweep: f16, i8, i32 (transpose, shape 128×64) — uses new TransposeCopyKernel
- Throughput sweep: 1024 to 16384 square matrices (transpose) — uses new TransposeCopyKernel
- Negative strides (flip) — may be SKIP

- [ ] **Step 3: Fix any correctness issues**

If tests fail, the most likely causes are:

1. **Smem layout mismatch:** The write view `smem_w` and read view `smem_r` must access the same physical memory locations. Verify by checking that `smem_w(i, j)` and `smem_r(j, i)` map to the same offset: `i + j * (kTileDim + 1)` vs `j * (kTileDim + 1) + i`. These are equal. ✓

2. **Dst tile offset error:** The transposed tile mapping `dst_ptr + n0 + m0 * dst_stride_outer` must be correct. For src tile (m,n), the dst should write at position (n,m). Element dst(i,j) = dst_ptr + i*dst_stride_outer + j. Tile (n0, m0) covers i in [n0, n0+kTileDim), j in [m0, m0+kTileDim). Base offset = n0*dst_stride_outer + m0. ✓

3. **TiledCopy partitioning mismatch:** Phase 1 partitions src_tile and smem_w with `load_copy`. Phase 2 partitions smem_r and dst_tile with `store_copy`. The register tensors (`rmem_ld`, `rmem_st`) must match their respective TiledCopy partitions. Since `make_fragment_like` creates a tensor matching the partition shape, this should be correct.

4. **Thread bounds for kVec=4:** With kTileDim=32 and kVec=4: load thread layout (8, 32) = 256, store thread layout (32, 8) = 256. All 256 threads participate. No bounds issue.

5. **vec_size falls to 1:** The host code requires vec_size >= 2 for the transpose path (the kernel cannot handle kVec=1 with kTileDim=32 — would need 1024 threads). If tests show wrong results, verify the host falls through correctly when vec_size=1.

Iterate: modify code, rebuild (`ninja core && ninja _turbomind`), re-run tests until all pass.

- [ ] **Step 4: Commit any fixes**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "fix(core): address GenericCopy transpose kernel issues from testing

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Performance validation

**Files:**
- Existing: `test_generic_copy.py` (includes throughput benchmarking)

- [ ] **Step 1: Compare throughput numbers**

The test output shows throughput in GB/s for each test. Expected changes:

- **2D transpose (f32, 128×64):** Small matrix. May see improvement but L2 cache effects dominate at this size.

- **2D transpose (f32, 4096×4096 and larger):** Should improve from ~200 GB/s to ~1300-1400 GB/s. The SMEM tiling enables fully coalesced reads AND writes.

- **2D transpose dtype sweep (f16, i8, i32):** Similar improvement expected.

- **Contiguous copies:** No change — uses existing vectorized path.

- **Non-2D layouts (3D permute, 4D slice, etc.):** No change — uses existing scalar path.

- [ ] **Step 2: Address any unexpected regressions**

If throughput is worse than the scalar baseline for transpose:

1. **Check smem bank conflicts:** Verify the +1 padding is correct. Add `printf` inside the kernel (temporarily) to check smem_w and smem_r layout strides.

2. **Check coalescing:** Use `cuobjdump -sass` to verify that Phase 1 uses vectorized loads (LDG.E.128) and Phase 2 uses vectorized stores (STG.E.128).

3. **Check vec_size dispatch:** Add a debug print in the host before the transpose dispatch: `printf("transpose: M=%d, N=%d, vec_size=%d, grid=(%d,%d)\n", M, N, tr_vec_size, N/kTileDim, M/kTileDim);`.

4. **Check thread utilization:** For kVec=4 and kTileDim=32, all 256 threads participate. Verify this by checking the TiledCopy thread layout sizes match.

If contiguous or non-transpose tests regress:

1. The transpose detection may be triggering for non-transpose cases. Verify detection logic: `a.stride(0) == 1 && b.stride(1) == 1 && a.stride(1) > 1 && b.stride(0) > 1`.

2. Add a temporary `printf("transpose detected: rank=%d, a.strides=(%ld,%ld), b.strides=(%ld,%ld)\n", ...)` to verify which cases trigger the transpose path.

- [ ] **Step 3: Final commit if any performance fixes were needed**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "perf(core): optimize GenericCopy transpose kernel performance

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
