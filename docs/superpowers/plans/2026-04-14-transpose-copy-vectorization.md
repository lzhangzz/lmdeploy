# TransposeCopy Vectorization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Vectorize TransposeCopyKernel via dual-smem layout conversion so both gmem phases issue wide loads/stores for narrow types (i8, f16).

**Architecture:** 3-phase kernel: vectorized gmem→smem1 (row-major), scalar smem1→smem2 (col-major layout conversion), vectorized smem2→gmem. Single file change to `transpose.cu`.

**Tech Stack:** CUDA, CuTe (CUTLASS), existing test harness (`test_generic_copy.py`)

---

### Task 1: Rewrite the kernel

**Files:**
- Modify: `src/turbomind/kernels/copy/transpose.cu`

- [ ] **Step 1: Replace the entire file with the vectorized kernel**

The complete new file content:

```cpp
#include "src/turbomind/kernels/copy/copy.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/logger.h"
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>

namespace turbomind::core {

using namespace cute;

// ============================================================================
// CUDA kernel: TransposeCopyKernel (vectorized 3-phase smem-staged transpose)
// ============================================================================
namespace kernel {

template<int kTileDim, int kVec,
         typename SrcEngine, typename SrcLayout,
         typename DstEngine, typename DstLayout>
__global__ void __launch_bounds__(256)
TransposeCopyKernel(cute::Tensor<SrcEngine, SrcLayout> src,
                    cute::Tensor<DstEngine, DstLayout> dst)
{
    using T = typename SrcEngine::value_type;
    static_assert(std::is_same_v<T, typename DstEngine::value_type>,
                  "TransposeCopyKernel: src and dst value types must match");

    constexpr int kPad    = kVec;
    constexpr int kStride = kTileDim + kPad;
    using VecT = uint_bit_t<kVec * sizeof_bits_v<T>>;

    extern __shared__ T smem[];

    // Smem1: row-major — contiguous dim 0 (for phase 1 vectorization)
    auto smem1 = make_tensor(make_smem_ptr(smem),
        make_layout(make_shape(Int<kTileDim>{}, Int<kTileDim>{}),
                          make_stride(Int<1>{}, Int<kStride>{})));

    // Smem2: col-major — contiguous dim 1 (for phase 2 vectorization)
    auto smem2 = make_tensor(make_smem_ptr(smem + kTileDim * kStride),
        make_layout(make_shape(Int<kTileDim>{}, Int<kTileDim>{}),
                          make_stride(Int<kStride>{}, Int<1>{})));

    // Tile gmem tensors
    auto tiler = make_shape(Int<kTileDim>{}, Int<kTileDim>{});
    auto src_tiled = tiled_divide(src, tiler);
    auto dst_tiled = tiled_divide(dst, tiler);

    // Bounds check on tile grid
    if (blockIdx.y >= size<1>(src_tiled) ||
        blockIdx.x >= size<2>(src_tiled)) return;

    auto src_tile = src_tiled(make_coord(_, _), blockIdx.y, blockIdx.x);
    auto dst_tile = dst_tiled(make_coord(_, _), blockIdx.y, blockIdx.x);

    // Phase 1: gmem(src) -> smem1, vectorize along dim 0
    auto tc1 = make_tiled_copy(
        Copy_Atom<UniversalCopy<VecT>, T>{},
        make_layout(make_shape(Int<8>{}, Int<kTileDim>{})),
        make_layout(make_shape(Int<kVec>{}, Int<1>{})));
    auto thr1 = tc1.get_slice(threadIdx.x);
    copy(tc1, thr1.partition_S(src_tile), thr1.partition_D(smem1));

    __syncthreads();

    // In-smem: smem1 -> smem2 (physical layout conversion, same logical data)
    auto tc_s = make_tiled_copy(
        Copy_Atom<UniversalCopy<T>, T>{},
        make_layout(make_shape(Int<16>{}, Int<16>{})),
        make_layout(make_shape(Int<1>{}, Int<1>{})));
    auto thr_s = tc_s.get_slice(threadIdx.x);
    copy(tc_s, thr_s.partition_S(smem1), thr_s.partition_D(smem2));

    __syncthreads();

    // Phase 2: smem2 -> gmem(dst), vectorize along dim 1
    auto tc2 = make_tiled_copy(
        Copy_Atom<UniversalCopy<VecT>, T>{},
        make_layout(make_shape(Int<kTileDim>{}, Int<8>{}),
                       make_stride(Int<8>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<kVec>{})));
    auto thr2 = tc2.get_slice(threadIdx.x);
    copy(tc2, thr2.partition_S(smem2), thr2.partition_D(dst_tile));
}

}  // namespace kernel

// ============================================================================
// TransposeCopy: 2D transpose via vectorized smem-staged TiledCopy
// ============================================================================
void TransposeCopy(const void* data_a, void* data_b,
                   const Layout& a, const Layout& b,
                   DataType dtype, cudaStream_t stream)
{
    constexpr int kTileDim = 32;

    int32_t M = static_cast<int32_t>(a.shape(0));
    int32_t N = static_cast<int32_t>(a.shape(1));
    dim3 grid(static_cast<uint32_t>(N / kTileDim),
              static_cast<uint32_t>(M / kTileDim));

    auto dispatch = [&](auto t, auto kvec) {
        using T = decltype(t);
        constexpr int kVec = decltype(kvec)::value;
        constexpr int smem_bytes = 2 * kTileDim * (kTileDim + kVec) * sizeof(T);

        auto src_gmem = make_tensor(make_gmem_ptr(reinterpret_cast<const T*>(data_a)),
            make_layout(make_shape(M, N),
                              make_stride(Int<1>{}, a.stride(1))));

        auto dst_gmem = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(data_b)),
            make_layout(make_shape(M, N),
                              make_stride(b.stride(0), Int<1>{})));

        kernel::TransposeCopyKernel<kTileDim, kVec>
            <<<grid, 256, smem_bytes, stream>>>(src_gmem, dst_gmem);
    };

    switch (byte_size(dtype)) {
        case 1: return dispatch(uint8_t{},  Int<4>{});
        case 2: return dispatch(uint16_t{}, Int<2>{});
        case 4: return dispatch(uint32_t{}, Int<2>{});
        case 8: return dispatch(uint64_t{}, Int<1>{});
        default:
            TM_CHECK(0) << "TransposeCopy: unsupported element size " << byte_size(dtype);
            break;
    }
}

}  // namespace turbomind::core
```

- [ ] **Step 2: Build**

Run: `cd build && ninja _turbomind`
Expected: Clean build with no errors.

### Task 2: Test correctness for all dtypes

**Files:**
- Test: `test_generic_copy.py` (existing, no modification)

- [ ] **Step 1: Test i8 correctness**

Run:
```bash
cd /data/lmdeploy-copy && CUDA_VISIBLE_DEVICES=4 \
  PYTHONPATH=lmdeploy:build/lib python test_generic_copy.py --dtype i8
```

Expected: `ALL TESTS PASSED`. All transpose and slice+transpose tests produce correct output matching PyTorch.

- [ ] **Step 2: Test f16 correctness**

Run:
```bash
cd /data/lmdeploy-copy && CUDA_VISIBLE_DEVICES=4 \
  PYTHONPATH=lmdeploy:build/lib python test_generic_copy.py --dtype f16
```

Expected: `ALL TESTS PASSED`.

- [ ] **Step 3: Test f32 correctness**

Run:
```bash
cd /data/lmdeploy-copy && CUDA_VISIBLE_DEVICES=4 \
  PYTHONPATH=lmdeploy:build/lib python test_generic_copy.py --dtype f32
```

Expected: `ALL TESTS PASSED`.

- [ ] **Step 4: Test i64 correctness**

Run:
```bash
cd /data/lmdeploy-copy && CUDA_VISIBLE_DEVICES=4 \
  PYTHONPATH=lmdeploy:build/lib python test_generic_copy.py --dtype i64
```

Expected: `ALL TESTS PASSED`.

- [ ] **Step 5: Fix any failures**

If any test fails with output mismatches:
1. Check the mismatch pattern (off-by-one, fully wrong, partial corruption)
2. Verify smem indexing: `smem1(i,j) = i + j*kStride`, `smem2(i,j) = i*kStride + j`
3. Verify thread layout dimensions match val layout
4. Rebuild and retest

### Task 3: Benchmark all dtypes

**Files:**
- Test: `test_generic_copy.py` (existing)

- [ ] **Step 1: Benchmark all 4 dtypes**

Run each:
```bash
for dtype in i8 f16 f32 i64; do
  cd /data/lmdeploy-copy && CUDA_VISIBLE_DEVICES=4 \
    PYTHONPATH=lmdeploy:build/lib python test_generic_copy.py --dtype $dtype \
    2>&1 | grep -E "Throughput Summary|Shape|Contiguous:|Transpose:|^  [0-9]"
done
```

Expected: Transpose throughput for i8/f16 significantly higher than the pre-vectorization baseline (838/1053 GB/s). f32/i64 should be at or near previous levels.

- [ ] **Step 2: Compare against baseline**

Baseline (pre-vectorization, coalesced tc2):
- i8 transpose 16K: ~838 GB/s
- f16 transpose 16K: ~1053 GB/s
- f32 transpose 16K: ~1415 GB/s
- i64 transpose 16K: ~1493 GB/s

If i8/f16 do not show improvement, investigate alignment or TiledCopy configuration.

### Task 4: Commit

- [ ] **Step 1: Commit the implementation**

```bash
git add src/turbomind/kernels/copy/transpose.cu
git commit -m "perf(core): vectorize TransposeCopyKernel with dual-smem layout

3-phase approach: vectorized gmem→smem1 (row-major), scalar in-smem
layout conversion smem1→smem2 (col-major), vectorized smem2→gmem.
Enables wide load/store for narrow types (i8 vec=4, f16 vec=2).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
