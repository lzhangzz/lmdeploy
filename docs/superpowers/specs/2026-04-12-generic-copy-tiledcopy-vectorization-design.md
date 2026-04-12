# GenericCopy CuTe TiledCopy Vectorization Spec

**Goal:** Add vectorized memory transactions to the CuTe layout algebra GenericCopy kernel via TiledCopy's val_layout parameter. Close the throughput gap on contiguous large copies (currently 75% of PyTorch, target ~95%+).

**Motivation:** The current kernel uses `Copy_Atom<UniversalCopy<T>, T>` with `val_layout = Int<1>`, meaning each thread copies one element of type T per instruction. For f32 contiguous copies, this produces ~1137 GB/s vs PyTorch's ~1510 GB/s. The old VecT approach (before the CuTe algebra refactor) achieved near-parity by using `uint4` (128-bit) loads/stores. We need vectorization back, but expressed through CuTe idioms.

---

## Architecture

### Overview

The kernel architecture (2D grid, group, zipped_divide, copy_if) is unchanged. The only change is the `make_tiled_copy` call, which gains a compile-time vectorization factor `kVec` from the host.

**When `kVec > 1`**: Each thread copies `kVec` contiguous elements per instruction using a wide `Copy_Atom`. For f32 with `kVec=4`, each thread issues 128-bit loads/stores (4 floats = `uint128_t`).

**When `kVec = 1`**: Falls back to the current scalar path. Used when alignment conditions aren't met.

### Constraints

Vectorization is only enabled when ALL of the following hold:
1. Innermost stride == 1 for both src and dst (contiguous in the innermost dim)
2. Pointer alignment: both src and dst pointers are aligned to `vec_size * sizeof(T)` bytes
3. Shape divisibility: `shape[0] % (vec_size * kBlockThreads) == 0`
4. Max vector width: `vec_size * sizeof(T) <= 16` bytes (128 bits, CuTe's max)

If any condition fails, `vec_size` is reduced or falls back to 1.

### What changes vs. current

| Aspect | Current | New |
|--------|---------|-----|
| TiledCopy atom | `UniversalCopy<T>` (scalar) | `UniversalCopy<uint_bit_t<kVec * sizeof_bits_v<T>>>` (vectorized) |
| val_layout | `Shape<Int<1>>` | `Shape<Int<kVec>>` |
| Thread count | 256 threads active | `256 / kVec` threads active for copy |
| Host alignment | None (removed with VecT) | Reintroduced: pointer + stride + shape checks |
| Kernel template params | `<T, SrcLayoutT, DstLayoutT>` | `<T, kVec, SrcLayoutT, DstLayoutT>` |

---

## Kernel design

### Signature

```cpp
template<typename T, int kVec, typename SrcLayoutT, typename DstLayoutT>
__global__ void __launch_bounds__(256)
GenericCopyKernel(const T* __restrict__ src_ptr,
                  T* __restrict__       dst_ptr,
                  SrcLayoutT             src_layout,
                  DstLayoutT             dst_layout)
```

Added `kVec` as a compile-time constant. No new runtime parameters.

### TiledCopy construction

```cpp
constexpr int kBlockThreads = 256;
constexpr int kCopyThreads  = kBlockThreads / kVec;  // threads participating in copy

auto tiled_copy = cute::make_tiled_copy(
    cute::Copy_Atom<cute::UniversalCopy<cute::uint_bit_t<kVec * cute::sizeof_bits_v<T>>>, T>{},
    cute::make_layout(cute::make_shape(cute::Int<kCopyThreads>{})),
    cute::make_layout(cute::make_shape(cute::Int<kVec>{})));
```

When `kVec=1`, this is identical to the current code (256 threads, 1 element each).
When `kVec=4` and `T=float`, this is `UniversalCopy<uint128_t>` with 64 threads each copying 4 floats.

### Identity tensor predication

When `kVec > 1`, the shape divisibility guarantee (condition 3) ensures every tile is fully populated. No partial tiles exist. Therefore:

- **kVec > 1**: No predication needed. Use `cute::copy(tiled_copy, thrSrc, thrDst)` directly.
- **kVec = 1**: Keep the current identity tensor + `copy_if` predication for the scalar fallback.

```cpp
if constexpr (kVec > 1) {
    // Full tiles guaranteed by host — no predication needed
    cute::copy(tiled_copy, thrSrc, thrDst);
} else {
    // Scalar path with identity tensor predication
    auto id_row   = cute::make_identity_tensor(cute::shape(rowSrc));
    auto id_tiled = cute::zipped_divide(id_row, tiler);
    auto tile_id  = id_tiled(_, blockIdx.x);
    auto thrId    = thr_copy.partition_S(tile_id);
    auto pred     = cute::make_tensor<bool>(cute::shape(thrSrc));
    PRAGMA_UNROLL
    for (int i = 0; i < cute::size(pred); ++i) {
        pred(i) = cute::get<0>(thrId(i)) < cute::size(rowSrc);
    }
    cute::copy_if(pred, thrSrc, thrDst);
}
```

### Full kernel body (kRank >= 2 branch)

```cpp
constexpr int kBlockThreads = 256;
constexpr int kCopyThreads  = kBlockThreads / kVec;

// Create CuTe tensors
auto gSrc = cute::make_tensor(cute::make_gmem_ptr(src_ptr), src_layout);
auto gDst = cute::make_tensor(cute::make_gmem_ptr(dst_ptr), dst_layout);

// Build TiledCopy
auto tiled_copy = cute::make_tiled_copy(
    cute::Copy_Atom<cute::UniversalCopy<cute::uint_bit_t<kVec * cute::sizeof_bits_v<T>>>, T>{},
    cute::make_layout(cute::make_shape(cute::Int<kCopyThreads>{})),
    cute::make_layout(cute::make_shape(cute::Int<kVec>{})));
auto thr_copy = tiled_copy.get_slice(threadIdx.x);

// Group outer dims
auto src_layout_g = cute::group<1, kRank>(src_layout);
auto dst_layout_g = cute::group<1, kRank>(dst_layout);
auto gSrc_g = cute::make_tensor(cute::make_gmem_ptr(src_ptr), src_layout_g);
auto gDst_g = cute::make_tensor(cute::make_gmem_ptr(dst_ptr), dst_layout_g);

if (blockIdx.y >= cute::size<1>(gSrc_g)) return;

auto rowSrc = gSrc_g(_, blockIdx.y);
auto rowDst = gDst_g(_, blockIdx.y);

// Tile the inner dim
auto tiler    = cute::Int<kBlockThreads>{};
auto tiledSrc = cute::zipped_divide(rowSrc, tiler);
auto tiledDst = cute::zipped_divide(rowDst, tiler);

if (blockIdx.x >= cute::size<1>(tiledSrc)) return;

auto ctaSrc = tiledSrc(_, blockIdx.x);
auto ctaDst = tiledDst(_, blockIdx.x);

// Partition
auto thrSrc = thr_copy.partition_S(ctaSrc);
auto thrDst = thr_copy.partition_D(ctaDst);

// Copy with or without predication
if constexpr (kVec > 1) {
    cute::copy(tiled_copy, thrSrc, thrDst);
} else {
    auto id_row   = cute::make_identity_tensor(cute::shape(rowSrc));
    auto id_tiled = cute::zipped_divide(id_row, tiler);
    auto tile_id  = id_tiled(_, blockIdx.x);
    auto thrId    = thr_copy.partition_S(tile_id);
    auto pred     = cute::make_tensor<bool>(cute::shape(thrSrc));
    PRAGMA_UNROLL
    for (int i = 0; i < cute::size(pred); ++i) {
        pred(i) = cute::get<0>(thrId(i)) < cute::size(rowSrc);
    }
    cute::copy_if(pred, thrSrc, thrDst);
}
```

---

## Host changes

### Alignment detection

```cpp
int64_t alignment = 16;  // Start with max (128 bits = 16 bytes)

auto align = [&](auto v) { alignment = std::gcd(alignment, v); };

// Stride-1 required for vectorization
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
```

This is the same alignment detection from the old VecT code.

### vec_size computation

```cpp
const int elem_size = byte_size(dtype);
int vec_size = static_cast<int>(alignment / std::max<int64_t>(1, elem_size));

// Cap at 128 bits (16 bytes) — CuTe's max
if (vec_size * elem_size > 16) {
    vec_size = 16 / elem_size;
}

// Shape divisibility: shape[0] must be divisible by vec_size * kBlockThreads
while (vec_size > 1 && a.shape(0) % (vec_size * kBlockThreads) != 0) {
    vec_size /= 2;
}
```

### Layout adjustment

When `vec_size > 1`, adjust the layout for the vectorized view:

```cpp
if (vec_size > 1) {
    // Adjust shape[0]
    ssize_t shape_adj[kRank];
    std::copy_n(a.shape().data(), rank, shape_adj);
    shape_adj[0] /= vec_size;

    // Adjust outer strides
    ssize_t src_stride_adj[kRank];
    ssize_t dst_stride_adj[kRank];
    std::copy_n(a.stride().data(), rank, src_stride_adj);
    std::copy_n(b.stride().data(), rank, dst_stride_adj);
    for (int i = 1; i < rank; ++i) {
        src_stride_adj[i] /= vec_size;
        dst_stride_adj[i] /= vec_size;
    }

    src_layout = detail::make_cute_layout<kRank>(shape_adj, src_stride_adj);
    dst_layout = detail::make_cute_layout<kRank>(shape_adj, dst_stride_adj);
}
```

### Grid computation

```cpp
int64_t inner_size = a.shape(0) / vec_size;
int64_t outer_total = 1;
for (int i = 1; i < rank; ++i) outer_total *= a.shape(i);

int64_t num_inner_tiles = (inner_size + kBlockThreads - 1) / kBlockThreads;
dim3 grid(static_cast<uint32_t>(num_inner_tiles),
          static_cast<uint32_t>(outer_total));
```

### Dispatch

The host dispatches on (dtype, vec_size, rank). Vec_size is passed as a template parameter:

```cpp
auto dispatch_dtype = [&](auto t) {
    using T = decltype(t);

    auto dispatch_vec = [&](auto v) {
        constexpr int kVec = v.value;

        auto invoke = [&](auto d) {
            constexpr int kRank = d.value;
            // ... build layout, compute grid, launch kernel ...
            auto func = kernel::GenericCopyKernel<T, kVec, decltype(src_layout), decltype(dst_layout)>;
            func<<<grid, 256, 0, stream>>>(...);
        };

        switch (rank) {
            case 1: invoke(constant<1>{}); break;
            // ... 2-6 ...
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
```

The vec_size switch only needs to cover {1, 2, 4, 8, 16} (powers of 2 up to 128 bits). In practice, for f32 it's {1, 2, 4}; for f16 it's {1, 2, 4, 8}; for i8 it's {1, 2, 4, 8, 16}.

---

## Performance expectations

### Contiguous copies (the target case)

For f32 contiguous with `vec_size=4`:
- 256 threads / 4 = 64 threads active, each copying 128 bits
- Each CTA moves 64 * 16 = 1024 bytes = same total as before
- Expect ~1400-1500 GB/s (close to PyTorch's ~1510 GB/s)
- Improvement from ~1137 GB/s (current scalar) to ~1400+ GB/s

For f16 contiguous with `vec_size=8`:
- 256 threads / 8 = 32 threads active, each copying 128 bits
- Each CTA still moves 1024 bytes
- Similar throughput improvement expected

### Non-contiguous copies

Stride > 1 on innermost dim → `vec_size = 1` → same as current. No regression.

### Small tensors

Alignment may not be met → scalar fallback. No change from current.

---

## Limitations

- Only vectorizes along the innermost dimension (stride-1 dimension after sorting)
- Max 128-bit vectors (CuTe's `uint128_t` ceiling)
- Shape must be evenly divisible by `vec_size * kBlockThreads` — partial tiles go scalar
- No vectorization for non-contiguous layouts (transpose, etc.) — these stay scalar
