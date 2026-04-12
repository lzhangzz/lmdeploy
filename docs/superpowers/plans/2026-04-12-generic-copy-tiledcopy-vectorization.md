# GenericCopy CuTe TiledCopy Vectorization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add alignment-gated vectorization to the GenericCopy kernel via CuTe TiledCopy's val_layout parameter, closing the throughput gap on contiguous large copies from ~75% to ~95%+ of PyTorch.

**Architecture:** The kernel keeps its CuTe layout algebra structure (2D grid, group, zipped_divide). A new compile-time `kVec` template param controls vectorization: `make_tiled_copy` uses `Copy_Atom<UniversalCopy<uint_bit_t<kVec * sizeof_bits_v<T>>>, T>` with `val_layout = Shape<kVec>`. Host detects alignment (pointer, stride, shape divisibility) and dispatches on vec_size. When `kVec > 1`, predication is skipped (host guarantees full tiles). When `kVec = 1`, identity tensor + `copy_if` predication is used.

**Tech Stack:** CuTe (CUTLASS v3.9.2 headers-only), CUDA, existing `turbomind::core::Tensor/Layout` infrastructure.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/turbomind/core/tensor.cu` | Vectorized GenericCopy kernel + alignment-gated host dispatcher |

---

### Task 1: Rewrite GenericCopy kernel and host dispatcher with vectorization

**Files:**
- Modify: `src/turbomind/core/tensor.cu` (replace entire file contents)

This task replaces the entire file. The `detail::make_cute_shape/stride/layout` helpers are unchanged. The kernel gains `kVec` template param. The host adds alignment detection and vec_size dispatch.

- [ ] **Step 1: Replace tensor.cu with vectorized implementation**

Write the following as the complete contents of `src/turbomind/core/tensor.cu`:

```cpp
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/core/meta.h"

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>

#include <algorithm>
#include <numeric>
#include <string>

namespace turbomind::core {

using cute::_;

// ============================================================================
// Helpers: Construct CuTe layouts from runtime shape/stride arrays
// ============================================================================
namespace detail {

template<int kRank>
auto make_cute_shape(const ssize_t* shape_data)
{
    if constexpr (kRank == 1) {
        return cute::make_shape(static_cast<int32_t>(shape_data[0]));
    }
    else if constexpr (kRank == 2) {
        return cute::make_shape(static_cast<int32_t>(shape_data[0]),
                                static_cast<int32_t>(shape_data[1]));
    }
    else if constexpr (kRank == 3) {
        return cute::make_shape(static_cast<int32_t>(shape_data[0]),
                                static_cast<int32_t>(shape_data[1]),
                                static_cast<int32_t>(shape_data[2]));
    }
    else if constexpr (kRank == 4) {
        return cute::make_shape(static_cast<int32_t>(shape_data[0]),
                                static_cast<int32_t>(shape_data[1]),
                                static_cast<int32_t>(shape_data[2]),
                                static_cast<int32_t>(shape_data[3]));
    }
    else if constexpr (kRank == 5) {
        return cute::make_shape(static_cast<int32_t>(shape_data[0]),
                                static_cast<int32_t>(shape_data[1]),
                                static_cast<int32_t>(shape_data[2]),
                                static_cast<int32_t>(shape_data[3]),
                                static_cast<int32_t>(shape_data[4]));
    }
    else if constexpr (kRank == 6) {
        return cute::make_shape(static_cast<int32_t>(shape_data[0]),
                                static_cast<int32_t>(shape_data[1]),
                                static_cast<int32_t>(shape_data[2]),
                                static_cast<int32_t>(shape_data[3]),
                                static_cast<int32_t>(shape_data[4]),
                                static_cast<int32_t>(shape_data[5]));
    }
}

template<int kRank>
auto make_cute_stride(const ssize_t* stride_data)
{
    if constexpr (kRank == 1) {
        return cute::make_stride(static_cast<int64_t>(stride_data[0]));
    }
    else if constexpr (kRank == 2) {
        return cute::make_stride(static_cast<int64_t>(stride_data[0]),
                                 static_cast<int64_t>(stride_data[1]));
    }
    else if constexpr (kRank == 3) {
        return cute::make_stride(static_cast<int64_t>(stride_data[0]),
                                 static_cast<int64_t>(stride_data[1]),
                                 static_cast<int64_t>(stride_data[2]));
    }
    else if constexpr (kRank == 4) {
        return cute::make_stride(static_cast<int64_t>(stride_data[0]),
                                 static_cast<int64_t>(stride_data[1]),
                                 static_cast<int64_t>(stride_data[2]),
                                 static_cast<int64_t>(stride_data[3]));
    }
    else if constexpr (kRank == 5) {
        return cute::make_stride(static_cast<int64_t>(stride_data[0]),
                                 static_cast<int64_t>(stride_data[1]),
                                 static_cast<int64_t>(stride_data[2]),
                                 static_cast<int64_t>(stride_data[3]),
                                 static_cast<int64_t>(stride_data[4]));
    }
    else if constexpr (kRank == 6) {
        return cute::make_stride(static_cast<int64_t>(stride_data[0]),
                                 static_cast<int64_t>(stride_data[1]),
                                 static_cast<int64_t>(stride_data[2]),
                                 static_cast<int64_t>(stride_data[3]),
                                 static_cast<int64_t>(stride_data[4]),
                                 static_cast<int64_t>(stride_data[5]));
    }
}

template<int kRank>
auto make_cute_layout(const ssize_t* shape, const ssize_t* stride)
{
    return cute::make_layout(make_cute_shape<kRank>(shape),
                             make_cute_stride<kRank>(stride));
}

}  // namespace detail

// ============================================================================
// CUDA kernel: GenericCopyKernel (CuTe layout algebra + vectorization)
// ============================================================================
namespace kernel {

template<typename T, int kVec, typename SrcLayoutT, typename DstLayoutT>
__global__ void __launch_bounds__(256)
GenericCopyKernel(const T* __restrict__ src_ptr,
                  T* __restrict__       dst_ptr,
                  SrcLayoutT             src_layout,
                  DstLayoutT             dst_layout)
{
    constexpr int kBlockThreads = 256;
    constexpr int kRank         = cute::rank_v<SrcLayoutT>;
    constexpr int kCopyThreads  = kBlockThreads / kVec;

    // 1. Create CuTe tensors from pointers + layouts
    auto gSrc = cute::make_tensor(cute::make_gmem_ptr(src_ptr), src_layout);
    auto gDst = cute::make_tensor(cute::make_gmem_ptr(dst_ptr), dst_layout);

    // Build TiledCopy: kCopyThreads threads, kVec elements per thread
    auto tiled_copy = cute::make_tiled_copy(
        cute::Copy_Atom<cute::UniversalCopy<cute::uint_bit_t<kVec * cute::sizeof_bits_v<T>>>, T>{},
        cute::make_layout(cute::make_shape(cute::Int<kCopyThreads>{})),
        cute::make_layout(cute::make_shape(cute::Int<kVec>{})));
    auto thr_copy = tiled_copy.get_slice(threadIdx.x);

    if constexpr (kRank == 1) {
        // No outer dims — the whole tensor is the inner dim
        if (blockIdx.y > 0) return;

        // 2. Tile the inner dim with zipped_divide
        auto tiler    = cute::Int<kBlockThreads>{};
        auto tiledSrc = cute::zipped_divide(gSrc, tiler);
        auto tiledDst = cute::zipped_divide(gDst, tiler);

        if (blockIdx.x >= cute::size<1>(tiledSrc)) return;

        auto ctaSrc = tiledSrc(_, blockIdx.x);
        auto ctaDst = tiledDst(_, blockIdx.x);

        auto thrSrc = thr_copy.partition_S(ctaSrc);
        auto thrDst = thr_copy.partition_D(ctaDst);

        if constexpr (kVec > 1) {
            // Full tiles guaranteed by host — no predication needed
            cute::copy(tiled_copy, thrSrc, thrDst);
        }
        else {
            // Scalar path with identity tensor predication
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
    }
    else {
        // 2. Group outer dims (modes 1..kRank-1) into a single mode
        //    rank-k layout -> rank-2 (inner, outer_flat)
        auto src_layout_g = cute::group<1, kRank>(src_layout);
        auto dst_layout_g = cute::group<1, kRank>(dst_layout);
        auto gSrc_g = cute::make_tensor(cute::make_gmem_ptr(src_ptr), src_layout_g);
        auto gDst_g = cute::make_tensor(cute::make_gmem_ptr(dst_ptr), dst_layout_g);

        if (blockIdx.y >= cute::size<1>(gSrc_g)) return;

        // 3. Slice to get this CTA's 1D inner tensor
        auto rowSrc = gSrc_g(_, blockIdx.y);
        auto rowDst = gDst_g(_, blockIdx.y);

        // 4. Tile the inner dim with zipped_divide
        auto tiler    = cute::Int<kBlockThreads>{};
        auto tiledSrc = cute::zipped_divide(rowSrc, tiler);
        auto tiledDst = cute::zipped_divide(rowDst, tiler);

        if (blockIdx.x >= cute::size<1>(tiledSrc)) return;

        auto ctaSrc = tiledSrc(_, blockIdx.x);
        auto ctaDst = tiledDst(_, blockIdx.x);

        auto thrSrc = thr_copy.partition_S(ctaSrc);
        auto thrDst = thr_copy.partition_D(ctaDst);

        if constexpr (kVec > 1) {
            // Full tiles guaranteed by host — no predication needed
            cute::copy(tiled_copy, thrSrc, thrDst);
        }
        else {
            // Scalar path with identity tensor predication
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
    }
}

}  // namespace kernel

// ============================================================================
// Host function: GenericCopy (alignment-gated vectorization)
// ============================================================================
void GenericCopy(const Tensor& src, Tensor& dst, cudaStream_t stream)
{
    auto a = src.layout();
    auto b = dst.layout();

    TM_CHECK_EQ(a.size(), b.size()) << "GenericCopy: src and dst must have the same number of elements";

    // Sort strides ascending so innermost (fastest-varying) dim is first
    vector<int> idxs(a.rank());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [&](int i, int j) {
        return a.stride()[i] < a.stride()[j];
    });

    a = a.permute(idxs);
    b = b.permute(idxs);

    a = a.coalesce();
    b = b.coalesce();

    int rank = std::max(a.rank(), b.rank());

    if (a.rank() < rank) {
        a = a.view(b.shape());
    }
    else if (b.rank() < rank) {
        b = b.view(a.shape());
    }

    const DataType dtype = src.dtype();
    constexpr int  kBlockThreads = 256;

    // --- Alignment detection ---
    int64_t alignment = 16;

    auto align = [&](auto v) { alignment = std::gcd(alignment, v); };

    // If the innermost dim is not stride-1, we can't vectorize along it
    if (a.stride(0) > 1 || b.stride(0) > 1) {
        alignment = byte_size(dtype);
    }

    align(byte_size(dtype, a.shape(0)));

    auto data_a = src.raw_data();
    auto data_b = dst.raw_data();

    align(reinterpret_cast<uintptr_t>(data_a));
    align(reinterpret_cast<uintptr_t>(data_b));

    for (int i = 1; i < rank; ++i) {
        align(byte_size(dtype, a.stride(i)));
        align(byte_size(dtype, b.stride(i)));
    }

    // --- vec_size computation ---
    const int elem_size = byte_size(dtype);
    int vec_size = static_cast<int>(alignment / std::max<int64_t>(1, elem_size));

    // Cap at 128 bits (16 bytes) — CuTe's max
    if (vec_size * elem_size > 16) {
        vec_size = 16 / elem_size;
    }

    // Shape divisibility: shape[0] must be divisible by vec_size * kBlockThreads
    while (vec_size > 1 && a.shape(0) % (static_cast<int64_t>(vec_size) * kBlockThreads) != 0) {
        vec_size /= 2;
    }

    // --- Dispatch on data type T, vec_size kVec, and rank kRank ---
    auto dispatch_dtype = [&](auto t) {
        using T = decltype(t);

        auto dispatch_vec = [&](auto v) {
            constexpr int kVec = v.value;

            auto invoke = [&](auto d) {
                constexpr int kRank = d.value;

                // Construct CuTe layouts
                auto src_layout = detail::make_cute_layout<kRank>(a.shape().data(), a.stride().data());
                auto dst_layout = detail::make_cute_layout<kRank>(a.shape().data(), b.stride().data());

                // Adjust layouts for vectorization
                if constexpr (kVec > 1) {
                    ssize_t shape_adj[kRank]{};
                    std::copy_n(a.shape().data(), rank, shape_adj);
                    shape_adj[0] /= kVec;

                    ssize_t src_stride_adj[kRank]{};
                    ssize_t dst_stride_adj[kRank]{};
                    std::copy_n(a.stride().data(), rank, src_stride_adj);
                    std::copy_n(b.stride().data(), rank, dst_stride_adj);

                    for (int i = 1; i < rank; ++i) {
                        src_stride_adj[i] /= kVec;
                        dst_stride_adj[i] /= kVec;
                    }

                    src_layout = detail::make_cute_layout<kRank>(shape_adj, src_stride_adj);
                    dst_layout = detail::make_cute_layout<kRank>(shape_adj, dst_stride_adj);
                }

                // Compute 2D grid: (num_inner_tiles, outer_total)
                int64_t inner_size = a.shape(0) / kVec;
                int64_t outer_total = 1;
                for (int i = 1; i < rank; ++i) {
                    outer_total *= a.shape(i);
                }

                int64_t num_inner_tiles = (inner_size + kBlockThreads - 1) / kBlockThreads;
                dim3 grid(static_cast<uint32_t>(num_inner_tiles),
                          static_cast<uint32_t>(outer_total));

                auto func = kernel::GenericCopyKernel<T, kVec, decltype(src_layout), decltype(dst_layout)>;

                func<<<grid, 256, 0, stream>>>(
                    reinterpret_cast<const T*>(data_a),
                    reinterpret_cast<T*>(data_b),
                    src_layout,
                    dst_layout);
            };

            // Dispatch on exact rank (1-6)
            switch (rank) {
                case 1: invoke(constant<1>{}); break;
                case 2: invoke(constant<2>{}); break;
                case 3: invoke(constant<3>{}); break;
                case 4: invoke(constant<4>{}); break;
                case 5: invoke(constant<5>{}); break;
                case 6: invoke(constant<6>{}); break;
                default: throw std::runtime_error("GenericCopy: rank > 6 not implemented");
            }
        };

        // Dispatch on vec_size (powers of 2, max 128 bits)
        switch (vec_size) {
            case 16: dispatch_vec(constant<16>{}); break;
            case 8:  dispatch_vec(constant<8>{}); break;
            case 4:  dispatch_vec(constant<4>{}); break;
            case 2:  dispatch_vec(constant<2>{}); break;
            default: dispatch_vec(constant<1>{}); break;
        }
    };

    // Dispatch on data type
    switch (dtype) {
        case DataType::kFloat32:  return dispatch_dtype(float{});
        case DataType::kFloat16:  return dispatch_dtype(half_t{});
        case DataType::kBfloat16: return dispatch_dtype(bfloat16_t{});
        case DataType::kInt8:     return dispatch_dtype(int8_t{});
        case DataType::kInt32:    return dispatch_dtype(int32_t{});
        case DataType::kBool:     return dispatch_dtype(uint8_t{});
        case DataType::kUint8:    return dispatch_dtype(uint8_t{});
        default:
            throw std::runtime_error(std::string("GenericCopy: unsupported data type ") + to_string(dtype));
    }
}

}  // namespace turbomind::core
```

Key design decisions:

1. **Kernel `kVec` template param**: Compile-time vectorization factor. When `kVec=4` and `T=float`, the atom is `UniversalCopy<uint128_t>` (128-bit). The thread layout shrinks to `kBlockThreads/kVec` active threads.

2. **Predication branching**: `if constexpr (kVec > 1)` — vectorized path skips predication (host guarantees full tiles via shape divisibility check). Scalar path keeps identity tensor + `copy_if`.

3. **Host alignment detection**: Same logic as the old VecT code. Checks pointer alignment, stride-1 innermost dim, and shape divisibility by `vec_size * kBlockThreads`. Reduces `vec_size` until all conditions are met.

4. **Layout adjustment**: When `kVec > 1`, `shape[0] /= kVec` and outer strides divided by `kVec`. The adjusted layout is fed to the kernel, which operates on "vector elements."

5. **Three-level dispatch**: `dtype` -> `vec_size` -> `rank`. The vec_size switch covers {1, 2, 4, 8, 16} (all powers of 2 up to 128-bit max).

- [ ] **Step 2: Build the core target**

Run: `cd /data/lmdeploy-copy/build && ninja core 2>&1 | tail -40`

Expected: Compilation succeeds. There may be CuTe-related warnings but no errors.

Common issues to fix if compilation fails:

- **`cute::sizeof_bits_v<T>` not found for `__half` or `__nv_bfloat16`**: CuTe should have these via CUTLASS type traits. If not, use `sizeof(T) * 8` instead.

- **`cute::uint_bit_t<N>` with unsupported N**: Valid N values are {1, 2, 4, 6, 8, 16, 32, 64, 128}. Our `kVec * sizeof_bits_v<T>` products are always powers of 2 in this range for supported types. If there's an issue, verify the product is a valid specialization.

- **Static assert in `make_tiled_copy`**: The atom's NumVal (elements per instruction) must divide TiledNumVal. With `kCopyThreads` threads and `kVec` values each, TiledNumVal = kCopyThreads * kVec = kBlockThreads. AtomNumVal = kVec. So kBlockThreads % kVec == 0, which is always true since kVec is a power of 2 <= 256.

- **`copy(tiled_copy, ...)` with non-vectorized path**: When `kVec > 1`, we use `cute::copy(tiled_copy, thrSrc, thrDst)`. When `kVec == 1`, we use `cute::copy_if(pred, thrSrc, thrDst)` (basic overload, no TiledCopy). This should resolve correctly.

- [ ] **Step 3: Fix any compilation errors**

Iterate: modify code, rebuild with `ninja core`, until clean.

- [ ] **Step 4: Build the _turbomind extension**

Run: `cd /data/lmdeploy-copy/build && ninja _turbomind 2>&1 | tail -30`

Expected: Full build succeeds.

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "feat(core): add alignment-gated vectorization to GenericCopy via CuTe TiledCopy

- Add kVec template param for compile-time vectorization factor
- Use Copy_Atom<UniversalCopy<uint_bit_t<N>>, T> for wide memory transactions
- Host detects alignment (pointer, stride, shape) and selects vec_size
- Vectorized path skips predication (host guarantees full tiles)
- Scalar path (kVec=1) retains identity tensor + copy_if predication
- Three-level dispatch: dtype -> vec_size -> rank

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
- Contiguous baseline (f32)
- 2D: transpose, row-stride, col-stride, narrow
- 3D: permute (2,0,1)
- 4D: slice
- Combined: slice+transpose
- Dtype sweep: f16, i8, i32 (transpose)
- Throughput sweep: 1K to 256M elements (contiguous + transpose)
- Negative strides (flip)

- [ ] **Step 3: Fix any correctness issues**

If tests fail, the most likely causes are:

1. **Layout adjustment bug**: When `vec_size > 1`, `shape[0] /= kVec` and outer strides divided by `kVec`. Verify the adjusted layout still produces correct memory addresses. Check that the kernel receives the adjusted layout and operates on the correct elements.

2. **Grid computation mismatch**: `inner_size = a.shape(0) / kVec` must match the adjusted layout's shape[0]. The number of tiles must be consistent.

3. **Vec_size fallback**: For transpose tests, the innermost stride > 1, so `vec_size` should be 1 (scalar fallback). Verify the alignment detection correctly forces scalar for these cases.

4. **Shape divisibility**: If `a.shape(0)` is not divisible by `vec_size * 256`, the while loop should reduce vec_size to 1. Verify this works for all test shapes.

Iterate: modify code, rebuild (`ninja core && ninja _turbomind`), re-run tests until all pass.

- [ ] **Step 4: Commit any fixes**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "fix(core): address GenericCopy vectorization issues from testing

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Performance validation

**Files:**
- Existing: `test_generic_copy.py` (includes throughput benchmarking)

- [ ] **Step 1: Compare throughput numbers**

The test output shows throughput in GB/s for each test. Expected changes:

- **Contiguous large copies (f32)**: Should improve from ~1137 GB/s to ~1400-1500 GB/s. With `vec_size=4` (128-bit), each thread moves 16 bytes instead of 4 bytes per instruction. Target: within 95% of PyTorch's ~1510 GB/s.

- **Contiguous large copies (f16)**: Should improve similarly with `vec_size=8` (128-bit for 16-bit types).

- **Transpose copies**: No change expected (vec_size=1 fallback, same as before).

- **Small tensors**: May see slight improvement if alignment is met, otherwise no change.

- [ ] **Step 2: Address any unexpected regressions**

If throughput is worse than the scalar baseline:

1. Check that `vec_size` is being set correctly for contiguous cases. Add a debug print: `printf("vec_size=%d, alignment=%ld\n", vec_size, alignment);` in the host.

2. Check that the kernel is using the vectorized TiledCopy path. The `cute::copy(tiled_copy, ...)` should emit 128-bit loads/stores. Check PTX with `cuobjdump -sass` if needed.

3. If `vec_size` is always 1 even for contiguous cases, the alignment detection may be too conservative. Check each alignment condition.

- [ ] **Step 3: Final commit if any performance fixes were needed**

```bash
git add src/turbomind/core/tensor.cu
git commit -m "perf(core): optimize GenericCopy vectorization performance

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
