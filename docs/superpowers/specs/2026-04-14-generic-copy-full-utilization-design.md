# Full-Utilization CopyKernelND

Date: 2026-04-14

## Problem

`CopyKernelND` uses CuTe's `TiledCopy` with `kCopyThreads = kBlockThreads / kVec`.
When vectorization is wide, most threads in the block are idle:

| dtype | kVec | active threads | utilization |
|-------|------|----------------|-------------|
| int8  | 16   | 16 / 256       | 6.25%       |
| f16   | 8    | 32 / 256       | 12.5%       |
| f32   | 4    | 64 / 256       | 25%         |

Result: ~20% of peak memory throughput for contiguous data.

Root cause: `TiledCopy` requires compile-time thread layouts (`Int<N>`-based
shapes). The current design works around this by shrinking the active thread
count, which underutilizes the GPU.

## Solution

Replace `TiledCopy` with manual thread-to-element mapping. All 256 threads
participate. Thread partition across tensor dims is dynamic (runtime CuTe
shapes). CuTe's `Copy_Atom` is retained for generating optimal vectorized
load/store instructions.

### Data structures: CuTe tuples throughout

All shapes, strides, coordinates, and partition descriptors are CuTe tuple
types. No C arrays, no `if constexpr` on rank.

- **Data shape/strides**: extracted from host-side CuTe layouts via
  `.shape()` and `.stride()`. May contain mixed compile-time (`Int<N>`) and
  runtime values.
- **Thread partition**: runtime CuTe shape, e.g. `make_shape(64, 4)`.
  Product of elements = 256.
- **Tile counts**: runtime CuTe shape, e.g. `make_shape(int64_t(4),
  int64_t(2))`.
- **Coordinates**: CuTe tuples produced by `idx2crd` decomposition.

### Kernel

```
CopyKernelND<kVec, ThrShape, TileCounts, DataShape, SrcStride, DstStride, T>
```

Parameters:
- `src`, `dst`: raw `T*` pointers
- `shape`: CuTe shape tuple (e.g. `cute::tuple<int, int>`)
- `src_strides`, `dst_strides`: CuTe stride tuples (may contain `Int<N>`)
- `thr_partition`: CuTe shape — thread count per dim, product = 256
- `tile_counts`: CuTe shape — number of tiles per dim

Logic:
1. **Decode thread** — `thr_coord = idx2crd(threadIdx.x, thr_partition)`
2. **Decode tile** — `tile_coord = idx2crd(int64_t(blockIdx.x), tile_counts)`
3. **Compute element coordinate**:
   - `tile_size[0] = thr_partition[0] * kVec`, `tile_size[i>0] = thr_partition[i]`
   - `vec_factor[0] = kVec`, `vec_factor[i>0] = 1`
   - `elem[i] = tile_coord[i] * tile_size[i] + thr_coord[i] * vec_factor[i]`
4. **Bounds check** — iterate over CuTe coordinate and shape tuples.
   Dim 0 checks `elem[0] + kVec <= shape[0]`; other dims check
   `elem[i] < shape[i]`. Implementation: `cute::for_each` with a
   zip-like iteration, or `cute::fold` with an early-exit flag. The
   exact CuTe tuple iteration pattern will be determined during
   implementation based on available CuTe utilities.
5. **Compute offsets** — `src_off = crd2idx(elem_coord, shape, src_strides)`,
   `dst_off = crd2idx(elem_coord, shape, dst_strides)`.
6. **Vectorized copy** — per-thread rank-1 `(kVec,)` gmem tensor with
   stride `Int<1>{}`:
   ```
   auto frag = make_tensor(make_gmem_ptr(ptr + offset),
       make_layout(make_shape(Int<kVec>{}), make_stride(Int<1>{})));
   copy(Copy_Atom<UniversalCopy<uint_bit_t<kVec * kElemBits>>, T>{},
        src_frag, dst_frag);
   ```

Grid: 1D — `gridDim.x = product(tile_counts)`. Each block handles one flat
tile; `blockIdx.x` is decoded into multi-dim tile coordinates.

### Host-side dispatch

The three-level dispatch (dtype x kVec x kRank) collapses to two levels
(dtype x kVec). Rank is encoded in the CuTe tuple types; no separate switch.

Steps:
1. Construct CuTe layouts using existing helpers (`make_cute_layout`,
   `make_cute_layout_unit_inner`).
2. Extract `shape()`, `stride()` tuples from layouts.
3. Determine `kVec` from alignment analysis (unchanged).
4. Select thread partition (see heuristic below).
5. Compute tile counts: `tile_counts[i] = ceil(shape[i] / tile_size[i])`.
6. Launch kernel with CuTe tuples as parameters.

### Thread partition heuristic

Given shape `(D0, D1, ..., Dk-1)` and `kVec`:

```
T0 = round_down_pow2(min(D0 / kVec, 256))
R  = 256 / T0
for i = 1 .. k-1:
    Ti = min(Di, R)
    R /= Ti
if R > 1:
    try increasing T0 or redistributing
```

Goal: `T0 * T1 * ... * Tk-1 = 256`, maximizing tile coverage.

Common cases:
- Large 1D tensor: `(256,)` — all threads on inner dim.
- 2D tensor (M, N) with moderate N: `(64, 4)` — 64 threads inner, 4 outer.
- Small inner dim: `(16, 16)` — spread threads across both dims.

### Predication

- **Vectorized path** (kVec > 1): shape divisibility is guaranteed by the
  alignment check (`shape[0] % (kVec * T0) == 0`). No predication needed.
- **Scalar path** (kVec == 1): bounds check via the coordinate loop. Threads
  whose element is out-of-bounds return early.

### Utility: idx2crd

CuTe has `detail::idx2crd(idx, shape)` internally. We expose a clean version
that decomposes a linear index into a CuTe coordinate tuple given a CuTe
shape. For shape `(S0, S1, ..., Sn)`:
```
cn = idx % Sn
c(n-1) = (idx / Sn) % S(n-1)
...
c0 = idx / (S1 * S2 * ... * Sn)
```

This is a right-to-left row-major decomposition, matching CuTe's
`crd2idx` inverse.

## What this replaces

**Removed from kernel:**
- `TiledCopy` parameter and `get_thread_slice`
- `partition_S` / `partition_D`
- `make_fragment_like`
- Excess-thread early exit (`threadIdx.x >= size(tiled_copy)`)

**Removed from host:**
- `make_tiled_3d` helper
- `kCopyThreads = kBlockThreads / kVec`
- The `invoke_nd` rank-dispatch lambda (rank is now implicit in tuple types)

**Added:**
- `idx2crd` utility (small, ~10 lines)
- Thread partition selection heuristic (~20 lines)
- `crd2idx`-based offset computation in kernel

## Expected performance

For contiguous int8 with kVec=16:
- Current: 16/256 threads active -> ~20% peak bandwidth
- New: 256/256 threads active -> targeting >80% peak bandwidth

For contiguous f32 with kVec=4:
- Current: 64/256 threads active
- New: 256/256 threads active -> ~4x improvement

## Scope

- Applies to all non-transpose copies (vectorized and scalar).
- Transpose path (`TransposeCopyKernel`) is unchanged.
- Supports ranks 1-4 (current limit).
