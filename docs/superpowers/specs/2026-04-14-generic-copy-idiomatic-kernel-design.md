# Design: Idiomatic CopyKernelND Refactor

**Date:** 2026-04-14
**Scope:** `src/turbomind/core/tensor.cu` — `CopyKernelND` kernel + `GenericCopy` host dispatch
**Reference:** `build/_deps/repo-cutlass-src/examples/cute/tutorial/tiled_copy.cu`
**Branch:** `generic-copy`

## Motivation

The current `CopyKernelND` (lines 79-165) constructs CuTe tensors, groups outer dims, tiles with
`zipped_divide`, and creates `TiledCopy` — all inside the kernel. The CUTLASS tiled_copy.cu tutorial
demonstrates a cleaner pattern: host creates and tiles tensors, kernel receives pre-tiled data and
performs a minimal slice → partition → fragment → copy sequence.

This refactor moves all CuTe tensor construction, grouping, tiling, and TiledCopy creation to the
host, leaving the kernel as a pure tutorial-pattern copier.

## Design

### 1. Kernel signature

**Before:**
```cpp
template<typename T, int kVec, typename SrcLayoutT, typename DstLayoutT>
__global__ void CopyKernelND(const T* __restrict__ src_ptr,
                              T* __restrict__ dst_ptr,
                              SrcLayoutT src_layout,
                              DstLayoutT dst_layout)
```

**After:**
```cpp
template<bool kPredicated, class TensorS, class TensorD, class TiledCopy>
__global__ void __launch_bounds__(256)
CopyKernelND(TensorS S, TensorD D, TiledCopy tiled_copy, int64_t inner_size)
```

Changes:
- `T` deduced from `TensorS::value_type`
- `kVec` encoded in `TiledCopy` (not a separate template param)
- Raw pointers + layouts replaced by pre-tiled CuTe tensors
- `kPredicated` controls scalar vs vectorized copy path
- `inner_size` always passed but only read when `kPredicated == true` (for last-tile bounds check)

### 2. Kernel body — tutorial pattern

```cpp
{
    constexpr int kBlockThreads = 256;

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

### 3. Host-side construction

Inside the existing rank/vec dispatch lambdas:

```cpp
// 1. Create CuTe gmem tensors (replaces raw pointer + layout args)
auto src_gmem = make_tensor(make_gmem_ptr(reinterpret_cast<const T*>(data_a)), src_layout);
auto dst_gmem = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(data_b)), dst_layout);

// 2. Group outer dims → rank-2 (moved from kernel)
auto src_2d = [&] { if constexpr (kRank > 1) return group<1, kRank>(src_gmem); else return src_gmem; }();
auto dst_2d = [&] { if constexpr (kRank > 1) return group<1, kRank>(dst_gmem); else return dst_gmem; }();

// 3. Tile inner dim (moved from kernel, tiled_divide replaces zipped_divide)
constexpr int kBlockThreads = 256;
auto tiled_src = tiled_divide(src_2d, Int<kBlockThreads>{});
auto tiled_dst = tiled_divide(dst_2d, Int<kBlockThreads>{});

// 4. Create TiledCopy (moved from kernel)
constexpr int kCopyThreads = kBlockThreads / kVec;
auto tiled_copy = make_tiled_copy(
    Copy_Atom<UniversalCopy<uint_bit_t<kVec * sizeof_bits_v<T>>>, T>{},
    make_layout(make_shape(Int<kCopyThreads>{})),
    make_layout(make_shape(Int<kVec>{})));

// 5. Grid dimensions and inner size (for predication)
int64_t inner_size = static_cast<int64_t>(a.shape(0));
dim3 grid(size<1>(tiled_src), size<2>(tiled_src));

// 6. Launch
auto func = CopyKernelND<kVec == 1, decltype(tiled_src), decltype(tiled_dst), decltype(tiled_copy)>;
func<<<grid, 256, 0, stream>>>(tiled_src, tiled_dst, tiled_copy, inner_size);
```

### 4. What gets removed from kernel

- `group<1,kRank>` call and manual outer-dim extraction lambdas
- `make_gmem_ptr` + raw pointer tensor construction
- `zipped_divide` tiling
- `make_tiled_copy` construction
- `if constexpr (kVec * sizeof_bits_v<T> <= 128)` outer guard (host ensures valid kVec)
- Identity tensor construction for non-predicated path

### 5. What stays the same

- `__launch_bounds__(256)`
- Host-side alignment computation and vec_size capping
- Host-side rank dispatch (switch on rank 1-4)
- Host-side element-size dispatch (switch on byte_size)
- `TransposeCopyKernel` — unchanged (already refactored separately)

### 6. `detail` namespace helpers

The `make_cute_layout_unit_inner` and `make_cute_layout` helpers remain — they're still used to
construct CuTe layouts from runtime arrays before wrapping in `make_tensor`. The host now calls
`make_tensor(gmem_ptr, layout)` instead of passing layout + raw pointer separately.

## Scope

Single file: `src/turbomind/core/tensor.cu`

- Kernel `CopyKernelND`: full rewrite of signature and body (~80 lines → ~30 lines)
- Host dispatch in `GenericCopy`: restructure `invoke_nd` lambda to construct CuTe tensors,
  group, tile, and pass by value
- No changes to `TransposeCopyKernel`, `detail` helpers, or the `Tensor` C++ API
