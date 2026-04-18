# Copy & Swizzling

## Copy Atoms

A Copy_Atom wraps a hardware copy instruction with its thread/data layout metadata.

**Copy_Traits** defines:
- `ThrID` — thread mapping
- `SrcLayout` — `(thread_id, value_id) → bit_offset` in source
- `DstLayout` — `(thread_id, value_id) → bit_offset` in destination
- `RefLayout` — reference layout for the copy

**Copy_Atom** wraps traits with a `call(src, dst)` interface.
Recasts bit layouts to value layouts via `recast_layout`.

### TiledCopy

Tiles a Copy_Atom across threads:
```cpp
template <class Copy_Atom, class LayoutCopy_TV, class ShapeTiler_MN>
struct TiledCopy : Copy_Atom
```

Usage:
```cpp
TiledCopy tiled_copy = make_tile_copy(copy_atom, copy_layout, tiler);
ThrCopy thr_copy = tiled_copy.get_slice(threadIdx.x);
Tensor tAgA = thr_copy.partition_S(gA);   // source partition: (CPY, CPY_M, CPY_K, k)
Tensor tAsA = thr_copy.partition_D(sA);   // destination partition
copy(tiled_copy, tAgA, tAsA);              // perform the copy
```

### The `copy` Algorithm

Two main overloads:
```cpp
// Default: dispatches based on tensor types (memory spaces, layouts)
void copy(Tensor<SrcEngine, SrcLayout> const& src,
          Tensor<DstEngine, DstLayout>      & dst);

// With explicit Copy_Atom override
void copy(Copy_Atom<CopyArgs...> const& copy_atom,
          Tensor<SrcEngine, SrcLayout> const& src,
          Tensor<DstEngine, DstLayout>      & dst);
```

The copy algorithm auto-dispatches to:
- Sequential per-thread copy
- Vectorized copy (e.g., 4×ld.global.b32 → ld.global.b128)
- Async copy (cp.async, TMA)
- Cooperative copy (multi-thread)

### `copy_if` — Predicated Copy

Copies elements only where a predicate tensor is nonzero:
```cpp
Tensor pred = make_identity_tensor(shape);     // pred(i) = i
copy_if(pred < bound, src, dst);               // only copy in-bounds elements
```

## Architecture-Specific Copy Atoms

- [07-sm75.md](07-sm75.md) — SM75 (Turing): `ldsm` shared→register loads
- [08-sm80.md](08-sm80.md) — SM80 (Ampere): `cp.async` gmem→smem, cache control
- [09-sm90.md](09-sm90.md) — SM90 (Hopper): TMA, STSM

## Swizzle

`Swizzle<BBits, MBase, SShift>` — XOR-based address bit manipulation to avoid
shared memory bank conflicts.

```
0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
                               ^--^ MBase: least-sig bits to keep constant
                  ^-^       ^-^     BBits: bits in mask
                    ^---------^     SShift: shift distance for YYY mask
Result: ZZZ ^= YYY (XOR of the two bit groups)
```

### Common swizzle patterns

```cpp
// Custom swizzle composed with a layout
auto swizzle_atom = composition(Swizzle<3,3,3>{},
    Layout<Shape <_8,Shape <_8, _8>>,
           Stride<_8,Stride<_1,_64>>>{});
```

SM90's 128-byte TMA swizzle is in [09-sm90.md](09-sm90.md).

### Position-independent swizzle

For copies that need to handle swizzled shared memory:
```cpp
Tensor sA_ = as_position_independent_swizzle_tensor(sA);
// Treats the swizzle as part of the pointer rather than the layout
// Required when using TiledCopy with swizzled smem layouts
```
