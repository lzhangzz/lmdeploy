# GenericCopy 2D Transpose SMEM Optimization Spec

**Goal:** Add a specialized SMEM-tiled transpose kernel to GenericCopy that achieves ~90% of peak copy bandwidth for 2D transpose patterns, up from ~45% of PyTorch today.

**Motivation:** The vectorized GenericCopy kernel achieves ~100% of PyTorch for contiguous copies but only ~41-54% for transpose copies (~200 GB/s vs ~450 GB/s). The root cause: after stride sorting, both src and dst have their stride-1 dimensions on different modes, so neither can be vectorized along a common dimension. The SMEM tiling approach decouples the read and write access patterns, enabling coalesced vectorized reads from src AND coalesced vectorized writes to dst.

---

## Architecture

### Overview

A new `TransposeCopyKernel` is added alongside the existing `GenericCopyKernel`. The host detects the 2D transpose pattern and dispatches to the transpose kernel when conditions are met. Otherwise, falls back to the existing vectorized/scalar path.

The kernel uses a two-phase shared memory tiling approach:

```
Phase 1: src gmem → registers (TiledCopy, vectorized) → smem (bare copy)
          __syncthreads()
Phase 2: smem → registers (bare copy) → dst gmem (TiledCopy, vectorized)
```

The "transpose" happens by accessing the same smem with two different layouts:
- **Write view:** row-major padded — coalesced writes from Phase 1
- **Read view:** column-major — coalesced reads for Phase 2

### Constraints

The transpose kernel is only dispatched when ALL of the following hold:

1. **Rank == 2:** Only 2D transpose is optimized
2. **Cross-mode stride-1:** src has stride-1 on mode 0 AND dst has stride-1 on mode 1 (after host stride sorting)
3. **Non-trivial outer strides:** src stride(1) > 1 AND dst stride(0) > 1
4. **Shape divisibility:** Both dimensions divisible by kTileDim (32)
5. **Pointer alignment:** Both pointers aligned to vec_size * sizeof(T)
6. **Max vector width:** vec_size * sizeof(T) <= 16 bytes (128 bits)

If any condition fails, falls back to existing GenericCopy (scalar path for transpose).

### What changes vs. current

| Aspect | Current | New |
|--------|---------|-----|
| Transpose path | Scalar (kVec=1), element-by-element | SMEM tiled, 128-bit vectorized both phases |
| Throughput (4096x4096 f32) | ~200 GB/s (~45% of PyTorch) | ~1300-1400 GB/s (target ~90% of peak copy BW) |
| Kernel | GenericCopyKernel only | GenericCopyKernel + TransposeCopyKernel |
| Host dispatch | dtype → vec_size → rank | Check transpose first, then existing dispatch |

---

## Transpose detection

After the existing stride sort, coalesce, and rank matching, the host checks:

```cpp
bool is_2d_transpose = (rank == 2) &&
    (a.stride(0) == 1) && (b.stride(1) == 1) &&   // stride-1 on different modes
    (a.stride(1) > 1) && (b.stride(0) > 1);         // non-trivial outer strides
```

**Why this detects transpose:** After sorting by src's ascending strides, src has its smallest stride (1) on mode 0. If dst has stride-1 on mode 1 (not mode 0), the two tensors have their contiguous dimensions on different modes — the signature of a transpose.

**Examples:**
- `src.t() → dst`: src strides (1, N), dst strides (M, 1) → detected ✓
- `src[::2, :].t() → dst`: src strides (1, 2*M), dst strides (M/2, 1) → detected ✓
- Contiguous `src → dst`: src strides (1, M), dst strides (1, M) → NOT detected (b.stride(0) == 1) ✓
- Row-stride `src[::2, :] → dst`: src strides (1, 2*M), dst strides (1, M) → NOT detected (b.stride(0) == 1) ✓

---

## Kernel design

### Signature

```cpp
template<typename T, int kVec, int kTileDim>
__global__ void __launch_bounds__(256)
TransposeCopyKernel(const T* __restrict__ src_ptr,
                    T* __restrict__       dst_ptr,
                    int64_t               src_stride_outer,
                    int64_t               dst_stride_outer,
                    int32_t               M,
                    int32_t               N)
```

- `T`: data type (float, half_t, etc.)
- `kVec`: vectorization width (elements per vector instruction)
- `kTileDim`: tile dimension (fixed at 32)
- `src_stride_outer`: stride of src's mode 1 (runtime)
- `dst_stride_outer`: stride of dst's mode 0 (runtime)
- `M`: src shape(0) (number of rows after sorting)
- `N`: src shape(1) (number of columns after sorting)

### Shared memory

Static shared memory with padding for bank conflict avoidance:

```cpp
constexpr int kPaddedDim = kTileDim + 1;
__shared__ T smem[kTileDim * kPaddedDim];
```

Two CuTe views of the same memory:

```cpp
// Write view: row-major padded
// (kTileDim, kTileDim) stride (1, kTileDim+1)
auto smem_w = make_tensor(make_smem_ptr(smem),
    make_layout(make_shape(Int<kTileDim>{}, Int<kTileDim>{}),
                make_stride(Int<1>{}, Int<kTileDim + 1>{})));

// Read view: column-major (transposed)
// (kTileDim, kTileDim) stride (kTileDim+1, 1)
auto smem_r = make_tensor(make_smem_ptr(smem),
    make_layout(make_shape(Int<kTileDim>{}, Int<kTileDim>{}),
                make_stride(Int<kTileDim + 1>{}, Int<1>{})));
```

The +1 padding ensures that concurrent accesses along rows (write) and columns (read) hit different shared memory banks, avoiding bank conflicts.

### Gmem tile tensors

Per-tile tensors with compile-time `Int<1>` strides for CuTe `recast` vectorization:

```cpp
int m0 = blockIdx.y * kTileDim;
int n0 = blockIdx.x * kTileDim;

// Src tile: contiguous along mode 0 (Int<1> stride)
auto src_tile = make_tensor(make_gmem_ptr(src_ptr + m0 + n0 * src_stride_outer),
    make_layout(make_shape(Int<kTileDim>{}, Int<kTileDim>{}),
                make_stride(Int<1>{}, src_stride_outer)));

// Dst tile: contiguous along mode 1 (Int<1> stride)
// Note: transposed tile mapping — tile (m,n) of src maps to tile (n,m) of dst
auto dst_tile = make_tensor(make_gmem_ptr(dst_ptr + n0 + m0 * dst_stride_outer),
    make_layout(make_shape(Int<kTileDim>{}, Int<kTileDim>{}),
                make_stride(dst_stride_outer, Int<1>{})));
```

The dst tile offset `n0 + m0 * dst_stride_outer` implements the transpose: src tile (m,n) maps to dst tile (n,m).

### Phase 1: gmem(src) → smem

Uses register staging (SM70 two-stage pattern from CuTe tutorials):

```cpp
auto load_copy = make_tiled_copy(
    Copy_Atom<UniversalCopy<uint_bit_t<kVec * sizeof_bits_v<T>>>, T>{},
    make_layout(make_shape(Int<kTileDim / kVec>{},           // mode 0: 8 for kVec=4
                        Int<kBlockThreads * kVec / kTileDim>{})),  // mode 1: 32
    make_layout(make_shape(Int<kVec>{}, Int<1>{})));         // vectorize along mode 0

auto thr_load = load_copy.get_slice(threadIdx.x);
auto thr_src  = thr_load.partition_S(src_tile);
auto thr_smw  = thr_load.partition_D(smem_w);
auto rmem_ld  = make_fragment_like(thr_smw);

copy(load_copy, thr_src, rmem_ld);   // vectorized gmem → registers
copy(rmem_ld, thr_smw);              // registers → smem (bare copy)
```

**Why this is coalesced:** src has stride 1 on mode 0. The val_layout `(kVec, 1)` groups kVec contiguous elements per thread. Consecutive threads in the thread layout read consecutive elements along mode 0 → fully coalesced 128-byte warp transactions.

### Phase 2: smem → gmem(dst)

```cpp
__syncthreads();

auto store_copy = make_tiled_copy(
    Copy_Atom<UniversalCopy<uint_bit_t<kVec * sizeof_bits_v<T>>>, T>{},
    make_layout(make_shape(Int<kBlockThreads * kVec / kTileDim>{},  // mode 0: 32
                        Int<kTileDim / kVec>{})),           // mode 1: 8
    make_layout(make_shape(Int<1>{}, Int<kVec>{})));        // vectorize along mode 1

auto thr_store = store_copy.get_slice(threadIdx.x);
auto thr_smr   = thr_store.partition_S(smem_r);
auto thr_dst   = thr_store.partition_D(dst_tile);
auto rmem_st   = make_fragment_like(thr_smr);

copy(thr_smr, rmem_st);                // smem → registers (bare copy)
copy(store_copy, rmem_st, thr_dst);    // vectorized registers → gmem
```

**Why this is coalesced:** The smem_r view has stride 1 on mode 1 (column-major). dst also has stride 1 on mode 1. The val_layout `(1, kVec)` groups kVec contiguous elements along mode 1. Consecutive threads write consecutive elements along mode 1 → fully coalesced 128-byte warp transactions.

### Template instantiation guard

Same as existing kernel — wrap body in:

```cpp
if constexpr (kVec * cute::sizeof_bits_v<T> <= 128)
{
    // ... kernel body ...
}
```

This prevents invalid `uint_bit_t<N>` instantiations (e.g., kVec=16 with int32_t = 512 bits).

### Thread bounds check

```cpp
constexpr int kCopyThreadsLoad  = kBlockThreads;  // All 256 threads participate
constexpr int kCopyThreadsStore = kBlockThreads;  // All 256 threads participate
```

For the transpose kernel with kTileDim=32 and kVec=4: thread layouts are (8, 32) and (32, 8), both totaling 256. All threads participate in both phases. No thread bounds check needed.

For kVec=8 (f16 with 128-bit vectors): thread layout would be (4, 32) = 128 threads per phase. The remaining 128 threads must exit early. The kernel uses `constexpr int kCopyThreads = (kTileDim / kVec) * (kTileDim / 1)` for Phase 1 and `constexpr int kCopyThreads2 = (kTileDim / 1) * (kTileDim / kVec)` for Phase 2. A bounds check `if (threadIdx.x >= min(kCopyThreads, kCopyThreads2)) return;` is placed at kernel entry.

For kVec=1 (scalar fallback): all 256 threads participate, no bounds check.

---

## Host changes

### Dispatch flow

```cpp
void GenericCopy(const Tensor& src, Tensor& dst, cudaStream_t stream)
{
    // ... existing stride sort, coalesce, rank matching ...

    // 1. Check for 2D transpose pattern
    constexpr int kTileDim = 32;
    bool is_2d_transpose = (rank == 2) &&
        (a.stride(0) == 1) && (b.stride(1) == 1) &&
        (a.stride(1) > 1) && (b.stride(0) > 1);

    if (is_2d_transpose &&
        a.shape(0) % kTileDim == 0 && a.shape(1) % kTileDim == 0)
    {
        // 2. Compute alignment and vec_size for transpose
        int64_t alignment = 16;
        auto data_a = src.raw_data();
        auto data_b = dst.raw_data();
        alignment = std::gcd(alignment, reinterpret_cast<uintptr_t>(data_a));
        alignment = std::gcd(alignment, reinterpret_cast<uintptr_t>(data_b));

        const int elem_size = byte_size(dtype);
        int vec_size = static_cast<int>(alignment / elem_size);
        if (vec_size * elem_size > 16) vec_size = 16 / elem_size;

        // Ensure kTileDim is divisible by vec_size for TiledCopy
        while (vec_size > 1 && kTileDim % vec_size != 0) vec_size /= 2;

        // 3. Dispatch to transpose kernel
        int32_t M = a.shape(0);
        int32_t N = a.shape(1);
        dim3 grid(N / kTileDim, M / kTileDim);

        auto dispatch_transpose = [&](auto t, auto v) {
            using T = decltype(t);
            constexpr int kVec = v.value;
            auto func = kernel::TransposeCopyKernel<T, kVec, kTileDim>;
            func<<<grid, 256, 0, stream>>>(
                reinterpret_cast<const T*>(data_a),
                reinterpret_cast<T*>(data_b),
                a.stride(1), b.stride(0),
                M, N);
        };

        // Dispatch on dtype and vec_size
        auto dispatch_dtype = [&](auto t) {
            using T = decltype(t);
            auto dispatch_vec = [&](auto v) {
                constexpr int kVec = v.value;
                constexpr int kCopyThreads = kBlockThreads > (kTileDim / kVec * kTileDim / kVec)
                    ? kTileDim / kVec * kTileDim / kVec : kBlockThreads;
                // Note: for kVec=4, kTileDim=32: kCopyThreads=256 (all threads)
                // for kVec=8, kTileDim=32: kCopyThreads=128 (half threads)
                auto func = kernel::TransposeCopyKernel<T, kVec, kTileDim>;
                func<<<grid, 256, 0, stream>>>(
                    reinterpret_cast<const T*>(data_a),
                    reinterpret_cast<T*>(data_b),
                    a.stride(1), b.stride(0),
                    M, N);
            };
            switch (vec_size) {
                case 16: dispatch_vec(constant<16>{}); break;
                case 8:  dispatch_vec(constant<8>{}); break;
                case 4:  dispatch_vec(constant<4>{}); break;
                case 2:  dispatch_vec(constant<2>{}); break;
                default: dispatch_vec(constant<1>{}); break;
            }
        };

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
        return;
    }

    // 4. Fall through to existing vectorized/scalar dispatch
    // ...
}
```

### Alignment for transpose

For the transpose kernel, alignment only depends on:
1. **Pointer alignment:** Both src and dst pointers aligned to `vec_size * sizeof(T)`
2. **kTileDim divisibility:** kTileDim (32) must be divisible by vec_size

No stride alignment check needed — we already know both tensors have stride-1 on their respective contiguous modes.

### Data type dispatch

Same dtype dispatch as existing GenericCopy:
- `kFloat32` → `float`
- `kFloat16` → `half_t`
- `kBfloat16` → `bfloat16_t`
- `kInt8` → `int8_t`
- `kInt32` → `int32_t`
- `kBool` / `kUint8` → `uint8_t`

---

## Performance expectations

### Contiguous copy (unchanged)

No change to the existing vectorized path. Contiguous copies continue to achieve ~100% of PyTorch.

### 2D transpose (the target case)

For f32 4096x4096 transpose:
- Grid: (128, 128) = 16384 CTAs
- Each CTA: 32x32 = 1024 elements (4 KB)
- Phase 1: 256 vector loads (128-bit each) = 4 KB read, fully coalesced
- Phase 2: 256 vector stores (128-bit each) = 4 KB written, fully coalesced
- Both phases achieve near-peak memory bandwidth
- **Target: ~1300-1400 GB/s** (~90% of peak copy BW on L20Y SM90)
- **Improvement: from ~200 GB/s to ~1300+ GB/s (~6.5x)**

### Non-2D-transpose (unchanged)

3D+ permutes, arbitrary strides, non-divisible shapes → fall back to existing scalar path.

---

## Limitations

- Only optimizes 2D transpose (rank == 2, cross-mode stride-1). 3D+ permutes still use scalar.
- Requires both dimensions divisible by kTileDim (32). Smaller or non-aligned sizes fall back to scalar.
- kTileDim is fixed at 32 (not tunable). This gives 4 elements per thread for 256 threads.
- No vectorization for non-contiguous innermost dimension on either tensor.
- Static shared memory: 32 * 33 * sizeof(T) bytes per CTA (4224 bytes for f32).
- The `copy(store_copy, rmem_st, thr_dst)` uses a different TiledCopy than the load phase. Each thread participates in two different TiledCopy partitionings. This requires register staging (rmem_ld for Phase 1, rmem_st for Phase 2).

---

## SMEM size per data type

| Type | sizeof(T) | SMEM per CTA |
|------|-----------|--------------|
| f32  | 4 bytes   | 4224 bytes   |
| f16  | 2 bytes   | 2112 bytes   |
| bf16 | 2 bytes   | 2112 bytes   |
| i8   | 1 byte    | 1056 bytes   |
| i32  | 4 bytes   | 4224 bytes   |

All well within the 48 KB shared memory limit per SM.
