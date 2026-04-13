# TiledCopy TransposeCopyKernel Design

**Date:** 2026-04-13
**Scope:** `src/turbomind/core/tensor.cu` — `TransposeCopyKernel` internal refactor
**Branch:** `generic-copy`

## Motivation

The current `TransposeCopyKernel` uses `cooperative_copy` for both phases (gmem→smem, smem→gmem). While functional, `cooperative_copy` is a closed-box API that auto-determines thread partitioning and vectorization via `heuristic_permutation` + `domain_distribute`. This makes it difficult to tune or extend per-phase.

Switching to explicit `TiledCopy` (via `make_tiled_copy`) provides:

1. **Explicit thread/value layout control** — thread mapping is visible and tunable, not hidden inside `cooperative_copy`
2. **Per-phase independence** — each phase has its own TiledCopy, enabling independent optimization
3. **Consistent API** — matches `CopyKernelND`'s pattern (`make_tiled_copy` → `get_slice` → `partition_S/D` → `copy`)
4. **Future vectorization** — switching from scalar to vectorized is changing the `Copy_Atom` and value layout, no structural changes needed

## Design

### Kernel signature (unchanged)

```cpp
template<int kTileDim, uint32_t kMaxVecBits,
         typename SrcEngine, typename SrcLayout,
         typename DstEngine, typename DstLayout>
__global__ void __launch_bounds__(256)
TransposeCopyKernel(cute::Tensor<SrcEngine, SrcLayout> src,
                    cute::Tensor<DstEngine, DstLayout> dst)
```

`kMaxVecBits` is retained for future vectorization but unused in the scalar implementation.

### smem layout (unchanged)

Flat `T smem[kTileDim * kTileDim]` with two views:
- `smem_w`: row-major, stride `(1, kTileDim)` — write view for Phase 1
- `smem_r`: column-major, stride `(kTileDim, 1)` — read view for Phase 2

### Tiling and bounds checking (unchanged)

`zipped_divide(src/dst, tiler)` with `tiler = (kTileDim, kTileDim)`. Bounds check on tile grid coordinates.

### Phase 1: gmem→smem (TiledCopy)

```cpp
auto tc1 = cute::make_tiled_copy(
    cute::Copy_Atom<cute::UniversalCopy<T>, T>{},
    cute::make_layout(cute::make_shape(cute::Int<256>{})),
    cute::make_layout(cute::make_shape(cute::Int<1>{})));
auto thr1 = tc1.get_slice(threadIdx.x);
cute::copy(tc1, thr1.partition_S(src_tile), thr1.partition_D(smem_w));
```

Scalar `Copy_Atom<UniversalCopy<T>, T>`, 256 threads, 1 element per atom. Each thread copies `kTileDim*kTileDim/256` elements via the TiledCopy loop.

### Phase 2: smem→gmem (TiledCopy)

```cpp
auto tc2 = cute::make_tiled_copy(
    cute::Copy_Atom<cute::UniversalCopy<T>, T>{},
    cute::make_layout(cute::make_shape(cute::Int<256>{})),
    cute::make_layout(cute::make_shape(cute::Int<1>{})));
auto thr2 = tc2.get_slice(threadIdx.x);
cute::copy(tc2, thr2.partition_S(smem_r), thr2.partition_D(dst_tile));
```

Separate `make_tiled_copy` call — same scalar configuration for now, but structurally independent from Phase 1.

### Host dispatch (unchanged)

No changes. The dtype switch, grid computation, and gmem tensor construction remain the same.

## What changes

| Component | Before | After |
|-----------|--------|-------|
| Phase 1 copy | `cooperative_copy<256, kMaxVecBits>(tid, src_tile, smem_w)` | `make_tiled_copy` + `copy` |
| Phase 2 copy | `cooperative_copy<256, kMaxVecBits>(tid, smem_r, dst_tile)` | `make_tiled_copy` + `copy` |
| TiledCopy objects | 0 | 2 (one per phase) |

## What stays the same

- Kernel signature and template parameters
- Host dispatch logic (dtype switch, grid dim, gmem tensor construction)
- smem layout (flat buffer, two views)
- `zipped_divide` tiling and bounds checking
- `__launch_bounds__(256)`

## Future vectorization path

To vectorize Phase 1 along dim 0 (both src and smem_w have stride-1 on dim 0):

```cpp
auto tc1 = make_tiled_copy(
    Copy_Atom<UniversalCopy<uint_bit_t<kVecBits>>, T>{},
    make_layout(make_shape(Int<256/kVec>{})),
    make_layout(make_shape(Int<kVec>{})));
```

To vectorize Phase 2 along dim 1 (both smem_r and dst have stride-1 on dim 1):

```cpp
auto tc2 = make_tiled_copy(
    Copy_Atom<UniversalCopy<uint_bit_t<kVecBits>>, T>{},
    make_layout(make_shape(Int<1>{}, Int<256/kVec>{})),
    make_layout(make_shape(Int<kVec>{}, Int<1>{})));
```

No structural changes needed — only `Copy_Atom` bit width, thread count, and value layout dimensions.

## Scope

~15 lines changed inside the kernel body. 0 lines changed in host dispatch. No new files, no new kernels, no API changes.
