# GenericCopy Transpose Kernel Refactor: cooperative_copy + Adaptive Padding

**Goal:** Refactor the `TransposeCopyKernel` to use CuTe's `cooperative_copy` API with adaptive smem padding, replacing the current manual `TiledCopy` + register staging approach. This simplifies the kernel from ~100 lines to ~50 lines while maintaining correctness and performance.

**Motivation:** The current `TransposeCopyKernel` manually constructs `TiledCopy` objects, partitions tensors per-thread, and stages data through registers with explicit copy loops. CuTe's `cooperative_copy` handles all of this automatically — thread partitioning, vectorization, and the actual copy. Additionally, the current fixed +1 element padding causes smem bank conflicts for sub-word types (float16, int8) because the byte stride isn't aligned to the 4-byte bank width. Adaptive padding fixes this for all element sizes.

---

## Architecture

### Kernel structure

```
Phase 1: cooperative_copy<256, MaxVecBits>(tid, src_gmem_tile, smem_w)  // gmem -> smem
          __syncthreads()
Phase 2: cooperative_copy<256, MaxVecBits>(tid, smem_r, dst_gmem_tile)  // smem -> gmem
```

Two `cooperative_copy` calls separated by one `__syncthreads()`. No register staging, no manual TiledCopy, no smem-to-smem transpose.

### Adaptive smem padding

The padding ensures the byte stride of the smem layout is a multiple of the 4-byte smem bank width, preventing bank conflicts for all element sizes.

Formula (compile-time):
```cpp
constexpr int kPadded = kTileDim + (4 + sizeof(T) - 1) / sizeof(T);
```

| Type | sizeof(T) | kPadded | Byte stride (kPadded * sizeof(T)) | Banks | Coprime with 32? |
|------|-----------|---------|-------------------------------------|-------|-------------------|
| float32 | 4 | 33 | 132 | 33 | Yes |
| float16 | 2 | 34 | 68 | 17 | Yes |
| bf16 | 2 | 34 | 68 | 17 | Yes |
| int8 | 1 | 36 | 36 | 9 | Yes |
| int32 | 4 | 33 | 132 | 33 | Yes |

When the bank stride (byte_stride / 4) is coprime with 32, each row in the transposed view starts at a different bank — zero conflicts across all 32 rows.

### Smem layout

Two views of a single shared memory buffer:

```cpp
__shared__ T smem[kTileDim * kPadded];

// Load view: row-major padded — stride-1 on mode 0 (matches src contiguous dim)
auto smem_w = make_tensor(make_smem_ptr(smem),
    make_layout(make_shape(Int<kTileDim>{}, Int<kTileDim>{}),
                make_stride(Int<1>{}, Int<kPadded>{})));

// Store view: column-major padded — stride-1 on mode 1 (matches dst contiguous dim)
auto smem_r = make_tensor(make_smem_ptr(smem),
    make_layout(make_shape(Int<kTileDim>{}, Int<kTileDim>{}),
                make_stride(Int<kPadded>{}, Int<1>{})));
```

The transpose is implicit: `smem_w(i, j)` and `smem_r(j, i)` map to the same physical address.

---

## Kernel design

### Signature

```cpp
template<int kTileDim, uint32_t kMaxVecBits,
         typename SrcEngine, typename SrcLayout,
         typename DstEngine, typename DstLayout>
__global__ void __launch_bounds__(256)
TransposeCopyKernel(Tensor<SrcEngine, SrcLayout> src,
                    Tensor<DstEngine, DstLayout> dst)
```

- `kTileDim`: tile dimension (32)
- `kMaxVecBits`: maximum vectorization width in bits (8, 16, 32, 64, or 128)
- `src`, `dst`: CuTe Tensor objects encapsulating pointer + layout for the full matrix

### Body

```cpp
{
    using T = typename SrcEngine::value_type;
    constexpr int kPadded = kTileDim + (4 + sizeof(T) - 1) / sizeof(T);

    __shared__ T smem[kTileDim * kPadded];

    // Smem views (load: row-major padded, store: column-major padded)
    auto smem_w = make_tensor(make_smem_ptr(smem),
        make_layout(make_shape(Int<kTileDim>{}, Int<kTileDim>{}),
                    make_stride(Int<1>{}, Int<kPadded>{})));
    auto smem_r = make_tensor(make_smem_ptr(smem),
        make_layout(make_shape(Int<kTileDim>{}, Int<kTileDim>{}),
                    make_stride(Int<kPadded>{}, Int<1>{})));

    // Tile gmem tensors — inner (kTileDim, kTileDim) is static, outer is dynamic
    auto tiler = make_shape(Int<kTileDim>{}, Int<kTileDim>{});
    auto src_tiled = zipped_divide(src, tiler);
    auto dst_tiled = zipped_divide(dst, tiler);

    // Bounds check
    if (blockIdx.y >= size<2>(src_tiled) || blockIdx.x >= size<3>(src_tiled)) return;

    // Per-CTA tile with static shape (kTileDim, kTileDim)
    auto src_tile = src_tiled(_, _, blockIdx.y, blockIdx.x);
    auto dst_tile = dst_tiled(_, _, blockIdx.y, blockIdx.x);

    // Phase 1: gmem -> smem (cooperative, vectorized)
    cooperative_copy<256, kMaxVecBits>(threadIdx.x, src_tile, smem_w);
    __syncthreads();

    // Phase 2: smem -> gmem (cooperative, vectorized, transposed access)
    cooperative_copy<256, kMaxVecBits>(threadIdx.x, smem_r, dst_tile);
}
```

### How vectorization works

`cooperative_copy` internally:
1. Computes `heuristic_permutation` to find a common layout between src and dst
2. Uses `max_common_vector` to determine the maximum vector width based on compile-time stride information
3. Partitions work across threads, recasts to vector types, and issues vector loads/stores

For Phase 1: src has stride (Int<1>, runtime), smem_w has stride (Int<1>, Int<kPadded>). Both have Int<1> on mode 0. `max_common_vector` vectorizes along mode 0.

For Phase 2: smem_r has stride (Int<kPadded>, Int<1>), dst has stride (runtime, Int<1>). Both have Int<1> on mode 1. `max_common_vector` vectorizes along mode 1.

The actual vector width is `min(kMaxVecBits, max_common_vector_result, 128)`.

### Thread utilization

With kTileDim=32 and 256 threads, the tile has 1024 elements:
- float32 + MaxVecBits=128: 4 elements/vector, 256 vectors, 256 threads -> all active
- float16 + MaxVecBits=128: 8 elements/vector, 128 vectors, 128 threads active
- int8 + MaxVecBits=128: 16 elements/vector, 64 vectors, 64 threads active

cooperative_copy handles idle threads internally. Threads with `tid >= required_threads` simply do nothing.

---

## Host dispatch

### Detection (unchanged)

```cpp
constexpr int kTileDim = 32;
bool is_2d_transpose = (rank == 2) &&
    (a.stride(0) == 1) && (b.stride(1) == 1) &&
    (a.stride(1) > 1) && (b.stride(0) > 1);
```

After stride sorting, src has stride-1 on mode 0 and dst has stride-1 on mode 1. This cross-mode stride-1 pattern is the signature of a transpose.

### Alignment and MaxVecBits

```cpp
int64_t tr_alignment = 16;
tr_alignment = gcd(tr_alignment, reinterpret_cast<uintptr_t>(data_a));
tr_alignment = gcd(tr_alignment, reinterpret_cast<uintptr_t>(data_b));
int max_vec_bits = min(128, static_cast<int>(tr_alignment * 8));
```

`cooperative_copy`'s internal recast to `uint_bit_t<N>` requires pointer alignment to `N/8` bytes. The host computes the safe maximum from pointer alignment.

### Dispatch

```cpp
auto dispatch_dtype = [&](auto t) {
    using T = decltype(t);

    // Create CuTe gmem tensors
    auto src_gmem = make_tensor(make_gmem_ptr(reinterpret_cast<const T*>(data_a)),
        make_layout(make_shape(static_cast<int>(M), static_cast<int>(N)),
                    make_stride(Int<1>{}, a.stride(1))));
    auto dst_gmem = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(data_b)),
        make_layout(make_shape(static_cast<int>(M), static_cast<int>(N)),
                    make_stride(b.stride(0), Int<1>{})));

    auto dispatch_vec = [&](auto v) {
        constexpr uint32_t kVB = v.value;
        auto func = kernel::TransposeCopyKernel<kTileDim, kVB,
            decltype(src_gmem), decltype(dst_gmem)>;
        func<<<grid, 256, 0, stream>>>(src_gmem, dst_gmem);
    };

    switch (max_vec_bits) {
        case 128: dispatch_vec(constant<128>{}); break;
        case 64:  dispatch_vec(constant<64>{}); break;
        case 32:  dispatch_vec(constant<32>{}); break;
        case 16:  dispatch_vec(constant<16>{}); break;
        default:  dispatch_vec(constant<8>{}); break;
    }
};

switch (byte_size(dtype)) {
    case 1: return dispatch_dtype(uint8_t{});
    case 2: return dispatch_dtype(uint16_t{});
    case 4: return dispatch_dtype(uint32_t{});
    default:
        TM_CHECK(0) << "GenericCopy: unsupported element size " << byte_size(dtype);
        break;
}
```

### Changes vs current dispatch

| Aspect | Current | New |
|--------|---------|-----|
| Kernel arguments | `(const T*, T*, int64_t, int64_t, int32_t, int32_t)` | `(Tensor<SrcEngine, SrcLayout>, Tensor<DstEngine, DstLayout>)` |
| Dispatch variable | `kVec` (elements per vector) | `kMaxVecBits` (bits per vector) |
| M, N parameters | Passed explicitly | Encoded in Tensor layout |
| vec_size >= 2 guard | Required (kTileDim=32 needs 1024 threads for kVec=1) | Removed (cooperative_copy handles any width) |
| Offset computation | Manual `m0 + n0 * src_stride_outer` | `zipped_divide` handles automatically |

---

## Smem usage per data type

| Type | sizeof(T) | kPadded | SMEM per CTA |
|------|-----------|---------|--------------|
| f32  | 4 bytes   | 33      | 4224 bytes   |
| f16  | 2 bytes   | 34      | 2176 bytes   |
| bf16 | 2 bytes   | 34      | 2176 bytes   |
| i8   | 1 byte    | 36      | 1152 bytes   |
| i32  | 4 bytes   | 33      | 4224 bytes   |

All well within the 48 KB shared memory limit.

---

## What is removed

The following code from the current `TransposeCopyKernel` is replaced by `cooperative_copy`:

1. **Manual TiledCopy construction** (`make_tiled_copy` + `Copy_Atom`) — cooperative_copy auto-constructs these internally
2. **Thread partitioning** (`get_slice` + `partition_S` / `partition_D`) — cooperative_copy partitions automatically
3. **Register staging** (`make_fragment_like` + manual `rmem(i) = smem(i)` loops) — cooperative_copy goes directly gmem<->smem
4. **kVec template parameter** — cooperative_copy determines vectorization from MaxVecBits and layout analysis
5. **M, N kernel parameters** — encoded in the CuTe Tensor layout

---

## Performance expectations

### Bank conflicts

Current kernel: bank conflicts for sub-word types (float16, int8) due to fixed +1 padding not aligning to 4-byte bank width.

New kernel: zero bank conflicts for all types due to adaptive padding ensuring coprime bank stride with 32.

### Memory access pattern

Unchanged from current kernel:
- Phase 1: coalesced vectorized reads from src gmem (stride-1 on mode 0)
- Phase 2: coalesced vectorized writes to dst gmem (stride-1 on mode 1)

### Throughput

Expected to match or exceed current kernel performance. The cooperative_copy path uses the same underlying vector instructions (LDG.128, STG.128). Bank conflict elimination for sub-word types may improve throughput for those types.

---

## Constraints and limitations

Same as current transpose kernel:
- Only rank-2 transpose (cross-mode stride-1)
- Both dimensions must be divisible by kTileDim (32)
- kTileDim is fixed at 32

Additional constraint from cooperative_copy:
- Requires static tensor shapes. The per-CTA tile shapes are (Int<kTileDim>, Int<kTileDim>) — static by construction via zipped_divide.
- Requires pointers byte-aligned to `kMaxVecBits / 8`. The host computes `max_vec_bits` from pointer alignment.
