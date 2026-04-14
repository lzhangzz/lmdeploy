---
name: TransposeCopy Vectorization
date: 2026-04-14
status: approved
---

# TransposeCopy Kernel Vectorization

## Problem

TransposeCopyKernel uses scalar `Copy_Atom<UniversalCopy<T>, T>` for both
gmem phases. Narrow types (i8, f16) issue many 1-2 byte stores, leaving
significant gmem bandwidth on the table:

| dtype | Transpose (GB/s) | Contiguous peak (GB/s) | Gap |
|-------|-----------------|----------------------|-----|
| i8    | 838             | 1496                 | 44% |
| f16   | 1053            | 1514                 | 30% |
| f32   | 1415            | 1523                 | 7%  |
| i64   | 1493            | 1526                 | 2%  |

## Design

### 3-phase data flow

Introduce a second smem buffer with transposed physical layout so that both
gmem phases can vectorize along their respective contiguous dimensions.

```
Phase 1:  src(stride<1,N>) ──vec──▶ smem1 [row-major, stride<1, kStride>]
              │                            │
              │  vectorize dim 0            │  contiguous dim 0
              ▼                            ▼
         __syncthreads()
              │
In-smem:  smem1 ──scalar──▶ smem2 [col-major, stride<kStride, 1>]
              │                 │
              │  physical       │  same logical (i,j)
              │  rearrange      │  different stride
              ▼                 ▼
         __syncthreads()
              │
Phase 2:  smem2 ──vec──▶ dst(stride<M,1>)
              │                       │
              │  contiguous dim 1     │  vectorize dim 1
```

The in-smem copy is a physical layout conversion: element `(i,j)` moves from
byte offset `i + j*kStride` to `i*kStride + j`. Same logical data, different
stride. Smem bandwidth (~19 TB/s on SM90) makes this cheap vs gmem.

### Smem allocation

Dynamic shared memory: `2 * kTileDim * (kTileDim + kVec) * sizeof(T)` bytes.

```
smem1: T[kTileDim * kStride]  row-major stride<1, kStride>
smem2: T[kTileDim * kStride]  col-major stride<kStride, 1>
```

### kPad and alignment

`kPad = kVec`, giving `kStride = kTileDim + kPad = 32 + kVec`.

Vectorized smem access requires `(kTileDim + kPad) % kVec == 0` so that
every thread's vector starting address is aligned to the vector width.
Since `kVec` always divides 32, `kPad = kVec` satisfies this.

Smem2 base sits at offset `kTileDim * kStride * sizeof(T)`, which is always
aligned to `kVec * sizeof(T)` (since `kTileDim * kStride` is a multiple of
`kVec`).

### Vectorization widths

Chosen to be sufficient for gmem coalescing without over-vectorizing types
that already approach peak throughput:

| dtype | kVec | vector bytes | kPad | kStride | P1 iters | P2 iters |
|-------|------|-------------|------|---------|----------|----------|
| i8    | 4    | 4 (uint32)  | 4    | 36      | 1        | 1        |
| f16   | 2    | 4 (uint32)  | 2    | 34      | 2        | 2        |
| f32   | 2    | 8 (uint64)  | 2    | 34      | 2        | 2        |
| i64   | 1    | 8 (scalar)  | 1    | 33      | 4        | 4        |

i64 uses the same code path (vec=1) for simplicity — the in-smem copy
overhead is negligible.

### TiledCopy configurations

Thread layouts are fixed across all types. Only `kVec` changes the val
layout and Copy_Atom width.

#### Phase 1 — src -> smem1, vectorize dim 0

```
Copy_Atom:  UniversalCopy<uint_bit_t<kVec * sizeof_bits_v<T>>>
Thr layout: (8, 32)  col-major (default strides)
Val layout: (kVec, 1)
```

Thread `(r,c)` owns `(kVec*r .. kVec*r+kVec-1, c)` — stride-1 in both src
and smem1.

#### In-smem — smem1 -> smem2, scalar layout conversion

```
Copy_Atom:  UniversalCopy<T>  (scalar)
Thr layout: (16, 16)
Val layout: (1, 1)
```

4 scalar reads + 4 scalar writes per thread.

#### Phase 2 — smem2 -> dst, vectorize dim 1

```
Copy_Atom:  UniversalCopy<uint_bit_t<kVec * sizeof_bits_v<T>>>
Thr layout: (32, 8)  row-major stride(8, 1)
Val layout: (1, kVec)
```

Thread `(r,c)` owns `(r, kVec*c .. kVec*c+kVec-1)` — stride-1 in both smem2
and dst. Consecutive threads along `c` write consecutive gmem addresses.

### Host-side dispatch

```cpp
template<int kTileDim, int kVec, typename SrcE, typename SrcL, typename DstE, typename DstL>
__global__ void __launch_bounds__(256)
TransposeCopyKernel(Tensor<SrcE,SrcL> src, Tensor<DstE,DstL> dst);

void TransposeCopy(...) {
    constexpr int kTileDim = 32;
    // grid, src/dst tensors...
    auto dispatch = [&](auto t, auto kvec) {
        using T = decltype(t);
        constexpr int kVec = decltype(kvec)::value;
        constexpr int smem_bytes = 2 * kTileDim * (kTileDim + kVec) * sizeof(T);
        kernel::TransposeCopyKernel<kTileDim, kVec>
            <<<grid, 256, smem_bytes, stream>>>(src, dst);
    };
    switch (byte_size(dtype)) {
        case 1: return dispatch(uint8_t{},  Int<4>{});
        case 2: return dispatch(uint16_t{}, Int<2>{});
        case 4: return dispatch(uint32_t{}, Int<2>{});
        case 8: return dispatch(uint64_t{}, Int<1>{});
    }
}
```

### Expected impact

For i8 and f16, the dominant improvement comes from:
- Fewer gmem instructions (1 vec store instead of 4 scalar for i8)
- Wider per-warp memory transactions (32 threads * 4B = 128B per warp for i8)

For f32/i64, the change is neutral to slightly positive (these types already
approach peak throughput with scalar access).

### Files changed

- `src/turbomind/kernels/copy/transpose.cu` — kernel and dispatch rewrite
