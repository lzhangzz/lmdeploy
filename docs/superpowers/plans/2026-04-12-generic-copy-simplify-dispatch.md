# GenericCopy Dispatch Simplification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify GenericCopy's 3-level dispatch (7 dtypes x 5 vec_sizes x 6 ranks) down to 3 dtypes x 5 vec_sizes x 4 ranks by dispatching on byte size, capping rank at 4, and splitting the kernel into 1D/ND variants.

**Architecture:** Replace `switch(dtype)` with `switch(byte_size(dtype))` using uint8_t/uint16_t/uint32_t as representative types. Extract the rank-1 branch from GenericCopyKernel into a separate CopyKernel1D. Cap rank dispatch at 4.

**Tech Stack:** CUDA, CuTe (CUTLASS), C++17

---

### Task 1: Split GenericCopyKernel into CopyKernel1D + CopyKernelND

**Files:**
- Modify: `src/turbomind/core/tensor.cu:73-195` (kernel namespace)

Replace the single `GenericCopyKernel` with two kernels. The code is extracted directly from the current `if constexpr (kRank == 1)` / `else` branches.

- [ ] **Step 1: Replace GenericCopyKernel with CopyKernel1D**

Delete lines 73-195 (the entire `GenericCopyKernel` function). In its place, add `CopyKernel1D` — the rank-1 path extracted from the old `if constexpr (kRank == 1)` branch:

```cpp
// ============================================================================
// CUDA kernel: CopyKernel1D (rank-1 vectorized copy)
// ============================================================================
template<typename T, int kVec, typename SrcLayoutT, typename DstLayoutT>
__global__ void __launch_bounds__(256)
CopyKernel1D(const T* __restrict__ src_ptr,
             T* __restrict__       dst_ptr,
             SrcLayoutT             src_layout,
             DstLayoutT             dst_layout)
{
    if constexpr (kVec * cute::sizeof_bits_v<T> <= 128)
    {
    constexpr int kBlockThreads = 256;
    constexpr int kCopyThreads  = kBlockThreads / kVec;

    auto gSrc = cute::make_tensor(cute::make_gmem_ptr(src_ptr), src_layout);
    auto gDst = cute::make_tensor(cute::make_gmem_ptr(dst_ptr), dst_layout);

    auto tiled_copy = cute::make_tiled_copy(
        cute::Copy_Atom<cute::UniversalCopy<cute::uint_bit_t<kVec * cute::sizeof_bits_v<T>>>, T>{},
        cute::make_layout(cute::make_shape(cute::Int<kCopyThreads>{})),
        cute::make_layout(cute::make_shape(cute::Int<kVec>{})));

    if (threadIdx.x >= kCopyThreads) return;

    auto thr_copy = tiled_copy.get_slice(threadIdx.x);

    if (blockIdx.y > 0) return;

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
}
```

- [ ] **Step 2: Add CopyKernelND below CopyKernel1D**

Add the rank-2-4 kernel right after CopyKernel1D (still in `namespace kernel`):

```cpp
// ============================================================================
// CUDA kernel: CopyKernelND (rank-2..4 vectorized copy)
// ============================================================================
template<typename T, int kVec, typename SrcLayoutT, typename DstLayoutT>
__global__ void __launch_bounds__(256)
CopyKernelND(const T* __restrict__ src_ptr,
             T* __restrict__       dst_ptr,
             SrcLayoutT             src_layout,
             DstLayoutT             dst_layout)
{
    if constexpr (kVec * cute::sizeof_bits_v<T> <= 128)
    {
    constexpr int kBlockThreads = 256;
    constexpr int kRank         = cute::rank_v<SrcLayoutT>;
    constexpr int kCopyThreads  = kBlockThreads / kVec;

    auto gSrc = cute::make_tensor(cute::make_gmem_ptr(src_ptr), src_layout);
    auto gDst = cute::make_tensor(cute::make_gmem_ptr(dst_ptr), dst_layout);

    auto tiled_copy = cute::make_tiled_copy(
        cute::Copy_Atom<cute::UniversalCopy<cute::uint_bit_t<kVec * cute::sizeof_bits_v<T>>>, T>{},
        cute::make_layout(cute::make_shape(cute::Int<kCopyThreads>{})),
        cute::make_layout(cute::make_shape(cute::Int<kVec>{})));

    if (threadIdx.x >= kCopyThreads) return;

    auto thr_copy = tiled_copy.get_slice(threadIdx.x);

    // Group outer dims (modes 1..kRank-1) into a single mode
    auto src_layout_g = cute::group<1, kRank>(src_layout);
    auto dst_layout_g = cute::group<1, kRank>(dst_layout);
    auto gSrc_g = cute::make_tensor(cute::make_gmem_ptr(src_ptr), src_layout_g);
    auto gDst_g = cute::make_tensor(cute::make_gmem_ptr(dst_ptr), dst_layout_g);

    if (blockIdx.y >= cute::size<1>(gSrc_g)) return;

    auto rowSrc = gSrc_g(_, blockIdx.y);
    auto rowDst = gDst_g(_, blockIdx.y);

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
}
```

---

### Task 2: Simplify GenericCopy host dispatch

**Files:**
- Modify: `src/turbomind/core/tensor.cu:445-530` (the dispatch_dtype lambda through the outer switch)

Replace the entire dispatch section. The layout preparation code (lines 300-443) stays unchanged.

- [ ] **Step 1: Replace the dispatch_dtype lambda and outer switch**

Replace lines 445–530 (from `auto dispatch_dtype` through the closing `}` of `GenericCopy`) with:

```cpp
    auto dispatch_elem_size = [&](auto t) {
        using T = decltype(t);

        auto dispatch_vec = [&](auto v) {
            constexpr int kVec = v.value;

            auto invoke_1d = [&] {
                int64_t inner_size   = a.shape(0);
                int64_t num_inner_tiles = (inner_size + kBlockThreads - 1) / kBlockThreads;
                dim3 grid(static_cast<uint32_t>(num_inner_tiles), 1u);

                if constexpr (kVec > 1) {
                    auto src_layout = detail::make_cute_layout_unit_inner<1>(a.shape().data(), a.stride().data());
                    auto dst_layout = detail::make_cute_layout_unit_inner<1>(a.shape().data(), b.stride().data());
                    auto func = kernel::CopyKernel1D<T, kVec, decltype(src_layout), decltype(dst_layout)>;
                    func<<<grid, 256, 0, stream>>>(
                        reinterpret_cast<const T*>(data_a),
                        reinterpret_cast<T*>(data_b),
                        src_layout, dst_layout);
                }
                else {
                    auto src_layout = detail::make_cute_layout<1>(a.shape().data(), a.stride().data());
                    auto dst_layout = detail::make_cute_layout<1>(a.shape().data(), b.stride().data());
                    auto func = kernel::CopyKernel1D<T, kVec, decltype(src_layout), decltype(dst_layout)>;
                    func<<<grid, 256, 0, stream>>>(
                        reinterpret_cast<const T*>(data_a),
                        reinterpret_cast<T*>(data_b),
                        src_layout, dst_layout);
                }
            };

            auto invoke_nd = [&](auto d) {
                constexpr int kRank = d.value;

                int64_t inner_size = a.shape(0);
                int64_t outer_total = 1;
                for (int i = 1; i < rank; ++i) {
                    outer_total *= a.shape(i);
                }

                int64_t num_inner_tiles = (inner_size + kBlockThreads - 1) / kBlockThreads;
                dim3 grid(static_cast<uint32_t>(num_inner_tiles),
                          static_cast<uint32_t>(outer_total));

                if constexpr (kVec > 1) {
                    auto src_layout = detail::make_cute_layout_unit_inner<kRank>(a.shape().data(), a.stride().data());
                    auto dst_layout = detail::make_cute_layout_unit_inner<kRank>(a.shape().data(), b.stride().data());
                    auto func = kernel::CopyKernelND<T, kVec, decltype(src_layout), decltype(dst_layout)>;
                    func<<<grid, 256, 0, stream>>>(
                        reinterpret_cast<const T*>(data_a),
                        reinterpret_cast<T*>(data_b),
                        src_layout, dst_layout);
                }
                else {
                    auto src_layout = detail::make_cute_layout<kRank>(a.shape().data(), a.stride().data());
                    auto dst_layout = detail::make_cute_layout<kRank>(a.shape().data(), b.stride().data());
                    auto func = kernel::CopyKernelND<T, kVec, decltype(src_layout), decltype(dst_layout)>;
                    func<<<grid, 256, 0, stream>>>(
                        reinterpret_cast<const T*>(data_a),
                        reinterpret_cast<T*>(data_b),
                        src_layout, dst_layout);
                }
            };

            switch (rank) {
                case 1: invoke_1d(); break;
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
        default: TM_CHECK(0) << "GenericCopy: unsupported element size " << byte_size(dtype); break;
    }
```

---

### Task 3: Simplify TransposeCopy host dispatch

**Files:**
- Modify: `src/turbomind/core/tensor.cu:372-404` (the tr_dispatch_dtype lambda and its switch)

- [ ] **Step 1: Replace the tr_dispatch_dtype switch**

Replace lines 372–404 (from `auto tr_dispatch_dtype =` through the closing `}` of its switch) with:

```cpp
            auto tr_dispatch_elem_size = [&](auto t) {
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

            switch (byte_size(dtype)) {
                case 1: return tr_dispatch_elem_size(uint8_t{});
                case 2: return tr_dispatch_elem_size(uint16_t{});
                case 4: return tr_dispatch_elem_size(uint32_t{});
                default:
                    TM_CHECK(0) << "GenericCopy: unsupported element size " << byte_size(dtype);
                    break;
            }
```

---

### Task 4: Build and verify

- [ ] **Step 1: Build**

Run: `cd build && ninja`
Expected: Clean compilation, no errors or warnings.

- [ ] **Step 2: Run tests**

Run:
```bash
cd /data/lmdeploy-copy
PYTHONPATH=lmdeploy:build/lib python test_generic_copy.py
```

Expected: ALL TESTS PASSED. All dtype sweep tests (f32, f16, i8, i32), all rank tests (1D–4D), transpose and contiguous tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "refactor(core): simplify GenericCopy dispatch — byte-size grouping, kernel split, rank cap at 4

- Dispatch by element byte size (uint8_t/uint16_t/uint32_t) instead of
  7 concrete types — same bit-copy semantics, fewer template instantiations
- Split GenericCopyKernel into CopyKernel1D (rank-1) and CopyKernelND (rank-2..4)
- Drop untested rank-5/6 dispatch paths
- Apply byte-size grouping to TransposeCopyKernel dispatch too"
```
