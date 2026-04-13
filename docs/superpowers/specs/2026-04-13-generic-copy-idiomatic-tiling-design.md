# Idiomatic Tiled Tiling in TransposeCopyKernel

**Date:** 2026-04-13
**Scope:** `src/turbomind/core/tensor.cu` — `TransposeCopyKernel` tiling refactor
**Branch:** `generic-copy`

## Motivation

The current `TransposeCopyKernel` uses `zipped_divide` for gmem tiling, which produces a single zipped inner mode `((32,32))` and a single zipped outer mode. This requires a manual `get<0>(layout)` unwrap to recover rank-2 layout for TiledCopy, and uses awkward `size<1,0>()` / `make_coord(by,bx)` indexing into zipped modes.

Switching to `tiled_divide` (the standard CuTe tutorial pattern) produces `((TM,TN), M/TM, N/TN)` with separate modes, enabling direct slice indexing and eliminating the unwrap.

## Design

### What changes

Replace 6 lines in the tiling/slicing section (lines 192-205):

**Before:**
```cpp
auto src_tiled = zipped_divide(src, tiler);
auto dst_tiled = zipped_divide(dst, tiler);

if (blockIdx.y >= size<1, 0>(src_tiled) ||
    blockIdx.x >= size<1, 1>(src_tiled)) return;

auto src_tile_z = src_tiled(_, make_coord(blockIdx.y, blockIdx.x));
auto dst_tile_z = dst_tiled(_, make_coord(blockIdx.y, blockIdx.x));
auto src_tile = make_tensor(src_tile_z.data(), get<0>(src_tile_z.layout()));
auto dst_tile = make_tensor(dst_tile_z.data(), get<0>(dst_tile_z.layout()));
```

**After:**
```cpp
auto src_tiled = tiled_divide(src, tiler);
auto dst_tiled = tiled_divide(dst, tiler);

if (blockIdx.y >= size<1>(src_tiled) ||
    blockIdx.x >= size<2>(src_tiled)) return;

auto src_tile = src_tiled(_, blockIdx.y, blockIdx.x);
auto dst_tile = dst_tiled(_, blockIdx.y, blockIdx.x);
```

### Why this works

`tiled_divide(src, (TM, TN))` produces layout `((TM, TN), M/TM, N/TN)`:
- Mode 0: `(TM, TN)` — rank-2 tile shape, directly usable by TiledCopy
- Mode 1: `M/TM` — tiles along dim 0, maps to `blockIdx.y`
- Mode 2: `N/TN` — tiles along dim 1, maps to `blockIdx.x`

Slicing `src_tiled(_, by, bx)` returns a rank-2 `(TM, TN)` tensor — no unwrap needed.

### What stays the same

- Kernel signature and template parameters
- Host dispatch logic
- smem layout and padding
- TiledCopy objects (scalar, two phases with orthogonal thread layouts)
- `__launch_bounds__(256)`

## Scope

6 lines changed inside the kernel. 0 lines in host dispatch or smem.
