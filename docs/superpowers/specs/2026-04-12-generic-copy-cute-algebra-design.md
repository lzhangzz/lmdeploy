# GenericCopy CuTe Shape Algebra Refactoring Spec

**Goal:** Replace all manual index arithmetic in the GenericCopy kernel with CuTe's layout algebra operations (group, slicing, zipped_divide, copy_if). Rethink the kernel architecture from a 1D grid (CTA-per-row with inner loop) to a 2D grid (CTA-per-inner-tile-per-row with no loop). Remove manual VecT vectorization dispatch — use the raw data type and let CuTe handle the copy.

**Motivation:** The current kernel uses CuTe layout types as parameters but immediately dumps them into C arrays and does manual pointer arithmetic. It also manually dispatches on VecT (uint4/uint2/uint/ushort/char) for vectorization. Both should be replaced with CuTe-idiomatic patterns: layout algebra for indexing and the raw data type for element operations.

---

## Architecture

### Grid mapping

**Current:** 1D grid. `blockIdx.x` maps to an outer-dim "row." Each CTA loops over tiles of the innermost dimension.

**New:** 2D grid. `blockIdx.x` = inner tile index, `blockIdx.y` = outer row index. Each CTA handles exactly one tile of `kBlockThreads` (256) elements. No inner loop.

Example for layout `(8192, 4)`:
- Current grid: `(4, 1)` — 4 CTAs, each loops over 32 tiles of 256 VecT-sized elements
- New grid: `(32, 4)` = 128 — 128 CTAs, each handles 1 tile of 256 elements

### Kernel pipeline

```
1. make_gmem_ptr + make_tensor     → CuTe tensors from ptr + layout
2. group<1, kRank>(tensor)         → rank-2 (inner, outer_flat)  [kRank ≥ 2 only]
3. tensor(_, blockIdx.y)           → 1D inner tensor             [replaces manual decomposition]
4. zipped_divide(inner, Int<256>)  → ((256), (num_tiles))
5. tiled(_, blockIdx.x)            → (256) elements (may be rounded up)
6. TiledCopy partition_S/D         → per-thread elements
7. Identity tensor + zipped_divide → predication coordinates
8. copy_if(pred, src, dst)         → predicated copy
```

### What changes vs. current

| Aspect | Current | New |
|--------|---------|-----|
| Grid | 1D (outer_total) | 2D (inner_tiles × outer_total) |
| Inner dim | Loop within CTA | Separate CTAs |
| Outer dim | Manual modular decomposition | `group<1,k>` + slicing |
| Predication | `if (threadIdx.x < remaining)` | `copy_if` with identity tensor |
| Vectorization | Manual VecT dispatch (uint4/...) | Raw data type, CuTe copy |
| Alignment | Host computes vec_size, adjusts layout | Not needed |

---

## Kernel design

### Signature

```cpp
template<typename T, typename SrcLayoutT, typename DstLayoutT>
__global__ void __launch_bounds__(256)
GenericCopyKernel(const T* __restrict__ src_ptr,
                  T* __restrict__       dst_ptr,
                  SrcLayoutT             src_layout,
                  DstLayoutT             dst_layout);
```

- `T` is the actual data type (e.g., `float`, `__half`, `int8_t`)
- Removed `inner_size` and `outer_total` parameters — kernel derives from layouts
- No VecT — no alignment, no vec_size, no layout adjustment

### Kernel body

The CuTe tensor types differ between the kRank==1 and kRank≥2 branches (different layout types), so a shared `auto` variable isn't possible. The two branches each produce a 1D `rowSrc`/`rowDst`, then call the common TiledCopy+predication logic.

```cpp
constexpr int kBlockThreads = 256;
constexpr int kRank = cute::rank_v<SrcLayoutT>;

// 1. Create CuTe tensors
auto gSrc = cute::make_tensor(cute::make_gmem_ptr(src_ptr), src_layout);
auto gDst = cute::make_tensor(cute::make_gmem_ptr(dst_ptr), dst_layout);

// 2-3. Get 1D inner tensor for this CTA's outer row
if constexpr (kRank == 1) {
    // No outer dims
    if (blockIdx.y > 0) return;

    // 4. Tile the inner dim
    auto tiler = cute::Int<kBlockThreads>{};
    auto tiledSrc = cute::zipped_divide(gSrc, tiler);
    auto tiledDst = cute::zipped_divide(gDst, tiler);

    if (blockIdx.x >= cute::size<1>(tiledSrc)) return;

    auto ctaSrc = tiledSrc(_, blockIdx.x);
    auto ctaDst = tiledDst(_, blockIdx.x);

    // 5-8. TiledCopy + predication
    copy_tile_with_predication<T>(ctaSrc, ctaDst, gSrc, tiler, blockIdx.x);
}
else {
    auto gSrc_g = cute::group<1, kRank>(gSrc);
    auto gDst_g = cute::group<1, kRank>(gDst);

    if (blockIdx.y >= cute::size<1>(gSrc_g)) return;

    auto rowSrc = gSrc_g(_, blockIdx.y);
    auto rowDst = gDst_g(_, blockIdx.y);

    // 4. Tile the inner dim
    auto tiler = cute::Int<kBlockThreads>{};
    auto tiledSrc = cute::zipped_divide(rowSrc, tiler);
    auto tiledDst = cute::zipped_divide(rowDst, tiler);

    if (blockIdx.x >= cute::size<1>(tiledSrc)) return;

    auto ctaSrc = tiledSrc(_, blockIdx.x);
    auto ctaDst = tiledDst(_, blockIdx.x);

    // 5-8. TiledCopy + predication
    copy_tile_with_predication<T>(ctaSrc, ctaDst, rowSrc, tiler, blockIdx.x);
}
```

### TiledCopy + Predication helper

```cpp
template<typename T, typename CtaSrcT, typename CtaDstT, typename RowSrcT, typename TilerT>
__device__ void copy_tile_with_predication(
    CtaSrcT const& ctaSrc, CtaDstT& ctaDst,
    RowSrcT const& rowSrc, TilerT tiler,
    int tile_idx)
{
    constexpr int kBlockThreads = 256;

    // Build TiledCopy: 256 threads, 1 element per thread
    auto tiled_copy = cute::make_tiled_copy(
        cute::Copy_Atom<cute::UniversalCopy<T>, T>{},
        cute::make_layout(cute::make_shape(cute::Int<kBlockThreads>{})),
        cute::make_layout(cute::make_shape(cute::Int<1>{})));
    auto thr_copy = tiled_copy.get_slice(threadIdx.x);

    // Partition data tensors
    auto thrSrc = thr_copy.partition_S(ctaSrc);
    auto thrDst = thr_copy.partition_D(ctaDst);

    // Predication via identity tensor
    auto id_row = cute::make_identity_tensor(cute::shape(rowSrc));
    auto id_tiled = cute::zipped_divide(id_row, tiler);
    auto tile_id = id_tiled(_, tile_idx);
    auto thrId = thr_copy.partition_S(tile_id);

    // Build predicate tensor
    auto pred = cute::make_tensor<bool>(cute::shape(thrSrc));
    CUTE_UNROLL
    for (int i = 0; i < cute::size(pred); ++i) {
        pred(i) = cute::get<0>(thrId(i)) < cute::size(rowSrc);
    }

    // Predicated copy
    cute::copy_if(pred, thrSrc, thrDst);
}
```

---

## Predication details

### Why identity tensor

From CuTe's predication docs: create an identity tensor (maps coord → coord), apply the same tiling, partition identically, then compare coordinates against bounds. This is architecture-independent and works for any layout.

### How it works

1. `make_identity_tensor(shape(rowSrc))` creates a 1D tensor where element `i` has value `i`
2. `zipped_divide(id_row, Int<256>{})` tiles it identically to the data tensor
3. `id_tiled(_, tile_idx)` gives the coordinate values for this tile
4. After TiledCopy partitioning, each thread's `thrId(i)` gives the original coordinate
5. Predicate: `get<0>(thrId(i)) < size(rowSrc)` checks if the coordinate is within bounds

For tiles that are fully within bounds (all but the last partial tile), every predicate is `true` and `copy_if` behaves like `copy` with no overhead.

---

## Host changes

### Layout construction

Simplified: no more alignment detection, vec_size computation, or stride/shape adjustment. The layout is constructed directly from the Tensor's shape and stride.

```cpp
// Current:
// 1. Compute alignment
// 2. Select VecT based on alignment
// 3. Adjust strides by dividing by vec_size
// 4. Adjust shape[0] by dividing by vec_size
// 5. Construct layout with adjusted values

// New: just construct the layout directly
auto src_layout = detail::make_cute_layout<kRank>(a.shape().data(), a.stride().data());
auto dst_layout = detail::make_cute_layout<kRank>(a.shape().data(), b.stride().data());
```

### Grid computation

```cpp
int64_t inner_size = a.shape(0);  // raw element count, not divided by vec_size
int64_t outer_total = 1;
for (int i = 1; i < rank; ++i) outer_total *= a.shape(i);

int64_t num_inner_tiles = (inner_size + kBlockThreads - 1) / kBlockThreads;
dim3 grid(static_cast<uint32_t>(num_inner_tiles), static_cast<uint32_t>(outer_total));
```

### Data type dispatch

Replace the VecT dispatch with a data type dispatch:

```cpp
// Dispatch on the actual data type T
auto dispatch_dtype = [&](auto t) {
    using T = decltype(t);
    auto invoke = [&](auto d) {
        constexpr int kRank = d.value;
        auto src_layout = detail::make_cute_layout<kRank>(...);
        auto dst_layout = detail::make_cute_layout<kRank>(...);
        // ... compute grid ...
        kernel::GenericCopyKernel<T, decltype(src_layout), decltype(dst_layout)>
            <<<grid, 256, 0, stream>>>(reinterpret_cast<const T*>(data_a),
                                        reinterpret_cast<T*>(data_b),
                                        src_layout, dst_layout);
    };
    // dispatch on rank (same as current)
};

switch (dtype) {
    case DataType::kF32: return dispatch_dtype(float{});
    case DataType::kF16: return dispatch_dtype(__half{});
    case DataType::kBF16: return dispatch_dtype(__nv_bfloat16{});
    case DataType::kI8:  return dispatch_dtype(int8_t{});
    case DataType::kI32: return dispatch_dtype(int32_t{});
    // ... other types ...
}
```

### Removed code

The following host code is removed:
- Alignment detection loop
- VecT selection switch (alignment → uint4/uint2/uint/ushort/char)
- Vec_size computation
- Stride adjustment (dividing by vec_size)
- Shape adjustment (dividing shape[0] by vec_size)

---

## Rank-1 handling

For kRank == 1, there are no outer dims:
- `group<1,1>` is not valid (empty group range)
- Grid is `(num_inner_tiles, 1)` — blockIdx.y is always 0
- The full tensor IS the inner dim — use it directly without grouping

Handled via `if constexpr (kRank == 1)` branch.

---

## Performance expectations

### Contiguous copy

Each CTA handles 256 elements of type T. For f32, each thread loads/stores 4 bytes. With 256 threads doing coalesced accesses, each CTA moves 1 KB per copy. This is well within coalescing limits. The 2D grid increases parallelism for large inner dims.

### Transpose / non-contiguous

Same memory access pattern as current. Each thread reads/writes one T element at the stride determined by the layout.

### Performance vs. current VecT approach

Removing VecT means each thread copies 1 element of type T instead of 1 element of type VecT (which could be 16 bytes for uint4). For contiguous f32 copies where the current code uses uint4 (4× vectorization), the new code does 4× more memory transactions. This may reduce throughput for large contiguous copies. The 2D grid partially compensates by exposing more parallelism.

Vectorization can be reintroduced later through CuTe's TiledCopy val_layout parameter if needed, without changing the kernel architecture.

---

## Limitations

- `zipped_divide` rounds up the inner tile size. The last tile of each row may access out-of-bounds memory if not predicated. The `copy_if` pattern handles this correctly.
- The `group<1, kRank>` operation creates a layout with a hierarchical mode. The flattened outer size must fit in `uint32_t` for `blockIdx.y` (max 2^31 - 1 per CUDA spec).
- No manual vectorization — each thread copies one T element. Throughput for contiguous large copies may be lower than the current VecT approach.
