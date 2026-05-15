---
name: TransposeCopy Multi-Dim Support
date: 2026-05-15
status: approved
---

# TransposeCopy Multi-Dim Support

## Problem

`TransposeCopy` (in `src/turbomind/kernels/copy/transpose.cu`) only handles
strict rank-2 transposes. The dispatcher in `copy.cc::GenericCopy` requires:

```cpp
bool is_2d_transpose = (rank == 2) &&
    (a.stride(0) == 1) && (b.stride(1) == 1) &&
    (a.stride(1) > 1) && (b.stride(0) > 1);
```

Anything rank ≥ 3 with a transpose-like access pattern (e.g.
`(B,H,M,N) → (B,H,N,M)`, common in attention) falls through to
`VectorizedCopy`, which performs poorly on transposed access for narrow
types — the exact case the smem-staged kernel was built to fix.

## Goal

Extend the transpose path so any layout that has

- one dim with src stride 1 (call it `I`), and
- another dim with dst stride 1 (call it `J`),

uses the smem-staged kernel, with all remaining dims treated as batch
(arbitrary strides; not necessarily contiguous between src and dst). `J`
may be at any position after the existing src-stride-ascending sort — not
just the second-innermost.

## Design

### Dispatcher canonicalization (`copy.cc::GenericCopy`)

This block **replaces** the existing `is_2d_transpose` branch in
`GenericCopy`. The rank-2 case is a strict subset (`J = 1`, no batch
dims), so it continues to work unchanged.

After the existing src-stride-ascending sort + permute (so position 0 has
the smallest src stride), find `J` and bring it to position 1:

```cpp
int J = -1;
for (int i = 1; i < rank; ++i) {
    if (b.stride(i) == 1) { J = i; break; }
}

bool ok = (J >= 1) &&
          (a.stride(0) == 1) && (a.stride(J) > 1) && (b.stride(0) > 1) &&
          (a.shape(0) % kTileDim == 0) &&
          (a.shape(J) % kTileDim == 0);

if (ok) {
    if (J != 1) { a = a.transpose(1, J); b = b.transpose(1, J); }
    std::tie(a, b) = coalesce_batch_dims(a, b);  // see below
    if (a.rank() <= 4) {
        TransposeCopy(src.raw_data(), dst.raw_data(), a, b, dtype, stream);
        return;
    }
}
// fall through to VectorizedCopy
```

After this prep, the layout is canonical:

- position 0 = `I` (src stride 1, dst stride > 1, divisible by tile),
- position 1 = `J` (dst stride 1, src stride > 1, divisible by tile),
- positions 2.. = batch dims with arbitrary strides in src and dst.

### Batch-dim coalescing

`coalesce_batch_dims(a, b)` is a small helper local to `copy.cc`. It walks
positions `i = 2 .. rank-2` once, merging position `i` with `i+1` when
the proportional-stride condition holds **simultaneously in both `a` and
`b`**:

```text
a.stride(i) == a.shape(i+1) * a.stride(i+1)  AND
b.stride(i) == b.shape(i+1) * b.stride(i+1)
```

A merge replaces the pair with a single dim of shape `shape(i)*shape(i+1)`
and stride `stride(i+1)` (in both layouts). A single forward pass suffices
in the canonical post-permute form (no re-scan needed).

This collapses the common attention pattern `(B,H,M,N) ↔ (B,H,N,M)` from
rank 4 to rank 3 (single batch dim of size `B*H`). It is
correctness-preserving — the kernel works without it — but the rank-4
instantiation is then reserved for genuinely non-coalesceable layouts
(sliced views, etc.).

The existing `Layout::coalesce()` is not reusable here: it operates on a
single layout and assumes outer-major order, while we need to coalesce
across the canonical (inner-major) form using both `a` and `b` jointly.

### Kernel — rank-generic CuTe body

The kernel template signature is unchanged from today; the rank lives in
the CuTe layout type carried by `SrcLay` / `DstLay`. The body becomes
rank-agnostic by tiling the first two dims (the `I,J` plane) and decoding
`blockIdx.z` into a multi-dim batch coord:

```cpp
template<int kTileDim, int kVec,
         class SrcEng, class SrcLay, class DstEng, class DstLay>
__global__ void __launch_bounds__(256)
TransposeCopyKernel(Tensor<SrcEng, SrcLay> src, Tensor<DstEng, DstLay> dst)
{
    auto tiler = make_shape(Int<kTileDim>{}, Int<kTileDim>{});

    // batch_shape = (B0, B1, ...) — empty tuple when rank == 2
    auto batch_shape = take<2, rank_v<SrcLay>>(shape(src));
    auto batch_coord = idx2crd(int(blockIdx.z), batch_shape);

    auto tile_yx = make_coord(int(blockIdx.y), int(blockIdx.x));
    auto src_tile = local_tile(src, tiler, append(tile_yx, batch_coord));
    auto dst_tile = local_tile(dst, tiler, append(tile_yx, batch_coord));

    // ... existing 3-phase smem-staged 2D body, unchanged, on src_tile/dst_tile
}
```

Properties:

- `take<2, 2>(shape)` on a rank-2 tensor yields the empty tuple, so
  `batch_coord` is empty, `append(tile_yx, ())` is just `tile_yx`, and the
  kernel reduces exactly to today's 2D path.
- The 3-phase smem code (smem1 row-major / smem2 col-major / in-smem
  rearrange / vectorized gmem on both sides) is byte-for-byte unchanged.
- `idx2crd` and `crd2idx` are CuTe primitives — the multi-dim decoding
  expands to a small chain of mul/div per block, dwarfed by the smem
  staging cost.

### Host launch

```cpp
auto dispatch = [&](auto t, auto kvec, auto ktiledim, auto rank_c) {
    using T = decltype(t);
    constexpr int kVec = decltype(kvec)::value;
    constexpr int kTileDim = decltype(ktiledim)::value;
    constexpr int kRank = decltype(rank_c)::value;

    auto data_shape  = detail::make_cute_shape<kRank>(a.shape().data());
    auto src_strides = detail::make_cute_stride<kRank>(a.stride().data());
    auto dst_strides = detail::make_cute_stride<kRank>(b.stride().data());

    auto src = make_tensor(make_gmem_ptr(reinterpret_cast<const T*>(data_a)),
                           make_layout(data_shape, src_strides));
    auto dst = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(data_b)),
                           make_layout(data_shape, dst_strides));

    int64_t total_batch = 1;
    for (int i = 2; i < kRank; ++i) total_batch *= a.shape(i);

    constexpr int smem_bytes = 2 * kTileDim * (kTileDim + kVec) * sizeof(T);
    dim3 grid(static_cast<uint32_t>(N / kTileDim),
              static_cast<uint32_t>(M / kTileDim),
              static_cast<uint32_t>(total_batch));

    kernel::TransposeCopyKernel<kTileDim, kVec>
        <<<grid, 256, smem_bytes, stream>>>(src, dst);
};

// outer switch: byte_size(dtype) → (T, kVec, kTileDim) — same defaults as today
//   case 1: (uint8_t,  Int<16>, Int<64>)
//   case 2: (uint16_t, Int<8>,  Int<64>)
//   case 4: (uint32_t, Int<4>,  Int<32>)
//   case 8: (uint64_t, Int<2>,  Int<32>)

// inner switch: rank → rank_c
//   case 2: constant<2>{}
//   case 3: constant<3>{}
//   case 4: constant<4>{}
//   default: TM_LOG_WARNING and return  // dispatcher already capped at 4
```

The shape/stride helpers (`detail::make_cute_shape`, `detail::make_cute_stride`)
are reused from `copy.cu` — same translation pattern as `VectorizedCopy`.

Total kernel instantiations: **4 dtypes × 3 ranks = 12** (was 4).

### Constraints (unchanged)

- Both transposed dims (`I` and `J` after canonicalization) must be
  divisible by `kTileDim`. Otherwise dispatcher falls through to
  `VectorizedCopy`.
- Per-dtype `(kVec, kTileDim)` defaults unchanged.
- Smem allocation `2 * kTileDim * (kTileDim + kVec) * sizeof(T)` unchanged.

### Grid.z size cap

`total_batch` is bounded by `gridDim.z ≤ 65535` on most CUDA architectures.
With max rank 4 and typical batch dim sizes, this is comfortable
(e.g. `B=128, H=64` → `8192`). If a real workload ever pushes past, the
kernel can switch to `gridDim.x × kTilesY × kTilesX` linearization
(orthogonal change). Document the limit; do not implement preemptively.

### Files changed

- `src/turbomind/kernels/copy/transpose.cu` — rank-generic kernel body and
  dispatch over rank.
- `src/turbomind/kernels/copy/copy.cc` — extend `GenericCopy` dispatch to
  detect the multi-dim transpose pattern (find `J`, swap to position 1,
  coalesce batch dims, cap rank at 4).

## Testing

Extend `test_generic_copy.py` with new cases. The existing harness already
verifies `GenericCopy` against `torch.contiguous()` of the strided source —
so any layout we feed it gets correctness-checked end-to-end.

New cases (added in the same style as existing checks):

```python
# 3D transformations (existing section)
check("3D batched transpose (B,M,N)->(B,N,M)",
      _rand(8, 64, 128).transpose(1, 2))

# 4D transformations (existing section)
check("4D batched transpose (B,H,M,N)->(B,H,N,M)",
      _rand(4, 8, 64, 128).transpose(2, 3))

# Non-adjacent transpose: J != 1 originally (the "1 inner + 1 arbitrary" case)
check("4D non-adjacent transpose (B,M,H,N)->(B,N,H,M)",
      _rand(4, 64, 8, 128).transpose(1, 3))

# Non-coalesceable batch (sliced view → batch strides mismatch between src and dst)
check("4D batched transpose with sliced batch",
      _rand(4, 16, 64, 128)[:, ::2, :, :].transpose(2, 3))

# Tile-unaligned batched transpose — must fall through to VectorizedCopy
# and still produce correct output
check("3D batched transpose, unaligned",
      _rand(8, 60, 100).transpose(1, 2))
```

Add a throughput-sweep entry for batched transpose to confirm the new path
matches or exceeds `VectorizedCopy` for representative shapes (e.g.
`(8, 64, 4096, 128)`-class attention transposes).

Run across all dtypes:

```bash
for dt in i8 f16 f32 i64; do
  python test_generic_copy.py --dtype $dt
done
```

Pass criteria: every check returns `[PASS]` (i.e. `torch.equal(result,
golden)` is true). The throughput numbers should match or improve over the
current rank-fallthrough behavior for the new batched cases.

### Expected impact

- Correctness: identical (the 2D body is byte-for-byte unchanged).
- Performance: batched-transpose call sites that previously took the
  `VectorizedCopy` fallback now use the smem-staged path, narrowing the
  same gmem-instruction-count gap that motivated the original
  vectorization (i8 / f16 benefit most).
