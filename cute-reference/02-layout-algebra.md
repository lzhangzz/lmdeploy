# Layout Algebra

CuTe layouts are functions from integers to integers. The layout algebra provides operations
to combine, split, and replicate these functions — all built on three primitives:
**composition**, **complement**, and **concatenation**.

---

## 0. IntTuple & Layout (Foundations)

### IntTuple

An integer or tuple of IntTuples (recursive). The foundation type for all CuTe shapes and strides.

```cpp
// Static integers (compile-time known)
Int<4>{}, _4{}              // equivalent
// Dynamic integers (runtime)
int{4}, 4
// Hierarchical tuples
make_tuple(4, make_tuple(2, 3))  // (4,(2,3))
```

Key operations: `rank(x)` (number of elements), `get<I>(x)` (Ith element),
`size(x)` (product of all elements), `depth(x)` (nesting depth).

### Layout

`Layout<Shape, Stride>` maps coordinates within Shape to indices via Stride.
The most fundamental CuTe concept. Everything else is built on it.

```cpp
auto layout = make_layout(make_shape(4, 8), make_stride(1, 4));  // (4,8):(1,4)
layout(2, 3)   // => 2*1 + 3*4 = 14  (coordinate to index)
shape(layout)   // => (4,8)
stride(layout)  // => (1,4)
size(layout)    // => 32 (domain size)
cosize(layout)  // => 32 (codomain size = max index + 1)
```

Static vs dynamic: `_4` is compile-time 4, `4` is runtime. CuTe handles both identically.
Prefer static when possible — enables compile-time optimizations and vectorization.

Hierarchical access: `size<0>(layout)`, `shape<1,0>(layout)` — drill into nested modes.

**Recall:** `crd2idx(c, shape, stride)` — the inner product of the natural coordinate with the stride.
CuTe layouts are hierarchical, so `Shape` and `Stride` can be nested tuples.

---

## 1. Coalesce

**What it does:** Simplifies a layout without changing its mathematical function.
Removes size-1 modes and merges adjacent modes where possible.

**Math:** If the layout is a function `f: integer → integer`, then `coalesce(f)` produces
an equivalent `g` where `g(i) = f(i)` for all `i`, but `g` has fewer modes (depth ≤ 1).

**Rules** (applied pairwise to adjacent modes `s0:d0` and `s1:d1`):
```
s0:d0  ++  _1:d1  =>  s0:d0           // size-1 mode is ignored
_1:d0  ++  s1:d1  =>  s1:d1           // size-1 mode is ignored
s0:d0  ++  s1:s0*d0  =>  s0*s1:d0     // contiguous: merge (stride of 2nd = size_1st * stride_1st)
s0:d0  ++  s1:d1  =>  (s0,s1):(d0,d1) // otherwise: keep separate
```

**Example:**
```cpp
// Before coalesce: hierarchical, 3 modes
auto a = Layout<Shape <_2,Shape <_1,_6>>,
                Stride<_1,Stride<_6,_2>>>{};   // (2,(1,6)):(1,(6,2))
// After coalesce: fully flat (all modes contiguous)
auto b = coalesce(a);                          // 12:_1

// By-mode coalesce preserves shape profile (coalesces within each mode):
auto c = coalesce(a, Step<_1,_1>{});           // (2,6):(_1,_2)
```

---

## 2. Composition — `composition(A, B)`

**What it does:** Functional composition `R = A ∘ B`, meaning `R(c) = A(B(c))`.
Layout B defines a new coordinate space; composition produces the layout that maps
B's coordinates to the same indices that A would produce.

**Math:**
```
Given  A: Shape_A → Index   (the "outer" layout)
       B: Shape_B → Index   (the "inner" layout, selecting from A)
Result R: Shape_B → Index   where R(c) = A(B(c))
```

**Key property:** `R` is always compatible with `B` — any coordinate valid for B is valid for R.

**How it works internally** (for 1D B = `s:d` composing with coalesced A):

1. **Divide out the stride:** From A's shape, "remove" the first `d` elements by dividing
   modes from the left. This is the **stride divisibility condition** — `d` must divide
   the stride of each mode it passes through.
   ```
   (6,2) /  2 => (3,2)    // remove 2 elements from shape (6,2)
   (6,2) /  3 => (2,2)    // remove 3 elements
   (6,2) /  6 => (1,2)    // remove 6 elements
   ```

2. **Mod out the shape:** Then "keep" the first `s` elements by modding modes from the left.
   This is the **shape divisibility condition**.
   ```
   (6,2) %  2 => (2,1)    // keep 2 elements from shape (6,2)
   (6,2) %  3 => (3,1)    // keep 3 elements
   (6,2) %  6 => (6,2)    // keep all 6 elements
   ```

3. **Scale the stride:** The new strides are the old strides multiplied by B's stride.

**Example 1 — Reshape a 1D layout as a matrix:**
```
A = 20:2          // 20-element vector, stride 2
B = (5,4):(4,1)   // 5×4 row-major matrix

R = A ∘ B = A(B(c))

Step-by-step for B = (5:4, 4:1):
  20:2 ∘ 5:4  =>  5:8    // trivial: new_shape=5, new_stride=2*4=8
  20:2 ∘ 4:1  =>  4:2    // trivial: new_shape=4, new_stride=2*1=2

R = (5,4):(8,2)
```
```cpp
Layout a = make_layout(20, 2);                              // 20:2
Layout b = make_layout(make_shape(5, 4), LayoutRight{});    // (5,4):(4,1)
Layout c = composition(a, b);                                // (5,4):(8,2)
```
Verification: `R(0,0) = 0*8 + 0*2 = 0`, `R(1,0) = 1*8 + 0*2 = 8`, `R(0,1) = 0*8 + 1*2 = 2`.
These match `A(B(0,0)) = A(0) = 0`, `A(B(1,0)) = A(4) = 8`, `A(B(0,1)) = A(1) = 2`.

**Example 2 — Reshape with non-trivial strides:**
```
A = (10,2):(16,4)   // 10×2 layout with stride 16 down rows, 4 across cols
B = (5,4):(1,5)     // 5×4 column-major matrix

Step-by-step:
  (10,2):(16,4) ∘ 5:1  =>  (5,1):(16,4)   // mod out shape 5 from (10,2)
  (10,2):(16,4) ∘ 4:5  =>  (2,2):(80,4)    // div out stride 5 from (10,2)

R = ((5,1):(16,4), (2,2):(80,4))
  => by-mode coalesce => (5,(2,2)):(16,(80,4))
```
```cpp
Layout a = make_layout(make_shape (Int<10>{}, Int<2>{}),
                       make_stride(Int<16>{}, Int<4>{}));
Layout b = make_layout(make_shape (Int<5>{}, Int<4>{}),
                       make_stride(Int<1>{}, Int<5>{}));
Layout c = composition(a, b);   // (_5,(_2,_2)):(_16,(_80,_4))
```

**By-mode composition (Tiler):** When B is a tuple of layouts/shapes, composition is
applied to each corresponding mode of A independently:
```cpp
auto tiler = make_tile(Layout<_3,_4>{}, Layout<_8,_2>{});
auto result = composition(a, tiler);
// equivalent to: make_layout(composition(layout<0>(a), _3:4), composition(layout<1>(a), _8:2))
```

---

## 3. Complement — `complement(A, cotarget)`

**What it does:** Finds the "rest" — a layout that covers the indices NOT touched by A,
up to a target codomain size. The complement "fills the gaps" in A's codomain.

**Math:**
```
Given A: Shape_A → Index with size(A) elements
       cotarget: the desired total codomain size
Result R: the smallest layout such that:
  1. cosize(A ∘ R) ≥ cotarget  (together they cover enough indices)
  2. R has increasing, positive strides (R is unique and ordered)
  3. A and R have disjoint codomains (R fills only gaps in A)
```

**How it works** (algorithm from source):
1. Sort A's modes by stride (ascending).
2. For each mode (except the last), create a new mode with shape = `current_stride / previous_result_stride` and stride = `current_stride * current_shape`.
3. For the last mode, create a "rest" with shape = `ceil(cotarget / last_stride)` to fill remaining space.

Intuitively: the complement walks through A's modes from smallest stride to largest,
and at each step, fills in the "gaps" between the elements that mode touches.

**Examples:**
```
complement(4:1, 24)    => 6:4
  A touches indices {0,1,2,3}. Stride=1, so 4 elements packed.
  Complement repeats this pattern at stride 4: {0,1,2,3}, {4,5,6,7}, ...
  Need 24/4 = 6 repetitions → 6:4

complement(4:2, 24)    => (2,3):(1,8)
  A touches indices {0,2,4,6} (stride 2, 4 elements).
  Sort by stride: only one mode, stride=2, shape=4.
  Step 1: new_shape = stride / prev_result_stride = 2 / 1 = 2
          new_stride = stride * shape = 2 * 4 = 8
  Step 2: rest_shape = ceil(24 / 8) = 3, rest_stride = 1
  Result: coalesce((2,3):(8,1)) = (2,3):(1,8)
  Verification: cosize((4,(2,3)):(2,(8,1))) = max(2*8+0*1, ...) + 1 = 16+8+1 ≥ 24 ✓

complement((2,2):(1,6), 24)  => (3,2):(2,12)
  A touches indices {0,1,6,7} (2×2 block: stride 1 down cols, stride 6 across rows).
  Sort by stride: mode0 stride=1 shape=2, mode1 stride=6 shape=2.
  Step 1: new_shape = min_stride / prev_result_stride = 6/1 = 6
          new_stride = min_stride * curr_shape = 6*2 = 12
  Step 2: rest_shape = ceil(24 / 12) = 2, rest_stride = 1
  Raw result: (6,2):(12,1). Coalesce with input profile: (3,2):(2,12).
  Verification: cosize with A gives ≥ 24 ✓

complement((4,6):(1,4), 24)  => 1:0
  Already covers 24 unique indices, nothing to fill.
```

**Overloads:**
```cpp
complement(layout)              // cotarget defaults to cosize(layout)
complement(layout, cotarget)    // explicit target codomain size
complement(layout, shape)       // cotarget = size(shape) — shape provides divisibility hints
```

---

## 4. Division (Tiling) — `logical_divide(A, Tiler)`

**What it does:** Splits layout A into two parts — the tile (elements selected by Tiler)
and the rest (everything else). This is the foundation of partitioning data across threads.

**Math:**
```
logical_divide(A, B) := A ∘ (B, B*)
where B* = complement(B, size(A))
```

That is: concatenate the tiler B with its complement B*, then compose with A.

**Result shape:** `((TileShape), (RestShape))` — the first mode is the tile, the second is the layout of tiles.

**Example — 1D divide:**
```
A = (4,2,3):(2,1,8)    // 24-element layout
B = 4:2                 // tile: 4 elements with stride 2

Step 1: B* = complement(4:2, 24) = (2,3):(1,8)
Step 2: (B, B*) = (4,(2,3)):(2,(1,8))
Step 3: A ∘ (B, B*) = (4,2,3):(2,1,8) ∘ (4,(2,3)):(2,(1,8))
  = ((2,2),(2,3)):((4,1),(2,8))

Verification:
  A(0) = 0, A(2) = 2, A(4) = 4, A(6) = 6   (stride 2, shape 4)
  A(1) = 1, A(3) = 3, A(5) = 5, A(7) = 7   (stride 1, shape 2 — the "gap fill")
  Then repeated at stride 8, 3 times.
  Result: first mode (2,2):(4,1) is the tile, second mode (2,3):(2,8) iterates over tiles.
```

**Convenience variants** — rearrange the modes of the result:
```
Layout Shape : (M, N, L, ...)
Tiler Shape  : <TileM, TileN>

logical_divide : ((TileM,RestM), (TileN,RestN), L, ...)    // keeps per-mode semantics
zipped_divide  : ((TileM,TileN), (RestM,RestN,L,...))       // zips tile modes together, rest modes together
tiled_divide   : ((TileM,TileN), RestM, RestN, L, ...)      // zipped + unpacked rest
flat_divide    : (TileM, TileN, RestM, RestN, L, ...)       // everything flat
```

**Example — 2D divide:**
```
A = (9,32):(59,(13,1))       // 9×32 layout (with multi-stride)
T = <3:3, (2,4):(1,8)>       // 3-element stride-3 tile × (2,4) subtile

ld = logical_divide(A, T)    // ((3,3),(8,4)):((177,(13,2)),(59,(26,4)))
// First mode of each mode: the tile. Second mode: iterating over tiles.
```

---

## 5. Product (Replication) — `logical_product(A, B)`

**What it does:** Replicates layout A according to layout B. Where division *extracts*
tiles, product *creates* copies of a tile arranged according to B.

**Math:**
```
logical_product(A, B) := (A, A* ∘ B)
where A* = complement(A, size(A) * cosize(B))
```

**Example — 1D product:**
```
A = (2,2):(4,1)   // 4-element layout: indices {0,1,4,5}
B = 6:1            // 6 repetitions

Step 1: A* = complement((2,2):(4,1), 24)
  Sort by stride: mode0 stride=1 shape=2, mode1 stride=4 shape=2.
  new_shape = 4/1 = 4, new_stride = 4*2 = 8
  rest_shape = ceil(24/8) = 3
  Result: coalesce((4,3):(8,1)) = (4,3):(1,8)

Step 2: A* ∘ B = (2,3):(2,8) ∘ 6:1 = (2,3):(2,8)  (trivial)

Step 3: (A, A* ∘ B) = ((2,2),(2,3)):((4,1),(2,8))

First mode (2,2):(4,1) is the tile.
Second mode (2,3):(2,8) iterates over 6 copies:
  copy 0: indices {0,1,4,5}
  copy 1: indices {2,3,6,7}
  copy 2: indices {8,9,12,13}  (stride 8)
```

**blocked_product vs raked_product:**

These are rank-sensitive wrappers on `logical_product` that reassociate modes for
more intuitive results. Both require A and B to have the same rank.

```cpp
blocked_product(A, B)  // zip(tile_mode_A, tile_mode_B) — tiles appear as contiguous blocks
raked_product(A, B)    // zip(tile_mode_B, tile_mode_A) — tiles are interleaved/cyclic
```

The only difference is which mode comes first in the zip.

**tile_to_shape:** Convenience function that performs a blocked_product to match a target shape.
The most common way to construct shared memory layouts:
```cpp
auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TA>{}, make_shape(bM, bK, bP));
auto sD = tile_to_shape(SmemAtom{}, make_shape(bM, bN, bP), Step<_2, _1, _3>{});
```

---

## 6. Higher-Level Partitioning Functions

These functions wrap composition/divide/product for common kernel patterns.

#### `local_tile(tensor, tiler, coord, step)` — 74 uses

Extracts a single tile from a tensor at the given coordinate.
Wraps `zipped_divide` + slicing. The primary way to partition global tensors across CTAs.

```cpp
Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), dA);
auto cta_tiler = make_shape(bM, bK);

// 3D: (M, N, K) problem, (BLK_M, BLK_N, BLK_K) tiler
Tensor gA = local_tile(mA, cta_tiler, make_coord(blockIdx.x, blockIdx.y, _), Step<_1, X, _1>{});
// Result: (BLK_M, BLK_K, k) — keeps M and K, slices N with blockIdx.y
Tensor gB = local_tile(mB, cta_tiler, make_coord(blockIdx.x, blockIdx.y, _), Step<X, _1, _1>{});
// Result: (BLK_N, BLK_K, k) — keeps N and K, slices M with blockIdx.x
Tensor gC = local_tile(mC, cta_tiler, make_coord(blockIdx.x, blockIdx.y, _), Step<_1, _1, X>{});
// Result: (BLK_M, BLK_N) — keeps M and N
```

The `Step` parameter controls which modes of the tiler are kept (stride-1) vs sliced (X):
- `_1` in Step: keep this mode (it becomes a dimension of the result)
- `X` in Step: slice this mode (it's consumed by the corresponding coordinate element)

#### `local_partition(tensor, layout, idx, step)` — 11 uses

Extracts a single thread's partition from a tiled tensor.

```cpp
Tensor thr_A = local_partition(gA, thread_layout_A, threadIdx.x);
Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});
// Result: (THR_M, BLK_K) — partition mode 0 by thread, keep mode 1
```

#### `tiled_divide` and `zipped_divide`

```cpp
// tiled_divide: partition a tensor by a block shape
Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape);
// Result: ((M, N), m', n')

// Cluster layout partitioning (Blackwell tutorials)
Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                          make_tile(typename TiledMMA::AtomThrID{}));

// Epilogue tiling
Tensor tAcc_epi = zipped_divide(tCtAcc, epi_tiler_v);
// Result: (EpiTile, NumTiles)
```

---

## 7. Mode Selection and Manipulation

#### `select<I...>(tuple)` — 146 uses

Select specific modes. Most common: extracting matrix dimensions from MNK.

```cpp
auto prob_shape = make_shape(M, N, K);                        // (M, N, K)
Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(prob_shape), dA);  // (M, K)
Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(prob_shape), dB);  // (N, K)
Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(prob_shape), dC);  // (M, N)

CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));
```

#### `take<begin, end>(tuple)` — 31 uses

Extract a contiguous range of modes.

```cpp
using EpilogueTile = decltype(take<0,2>(TileShape{}));
CUTE_STATIC_ASSERT_V(shape(tCrC) == take<0,3>(shape(tCgC)));
```

#### `append<N>(tuple, value)` — 16 uses

Add a mode to the end.

```cpp
auto problem_shape_MNKL = cute::append<4>(problem_size, 1);   // add L=1 for batched
Tensor gLSE = local_tile(mLSE, append<3>(cta_tiler_pv, _1{}), ...);
```

#### `replace<I>(tuple, value)` — 13 uses

Replace a specific mode.

```cpp
using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{},
                                           replace<2>(TileShape{}, _2{}),
                                           Step<_2, _1, _3>{}));
```

#### `flatten(layout)` — 7 uses

Flatten all modes into a single mode.

```cpp
auto flat_perm = flatten(perm);
auto c_f = make_tensor(c.data(), flatten(c.layout()));
```

#### `make_tile(layouts...)` — 12 uses

Construct a tiler from per-mode tilers.

```cpp
auto tiler = make_tile(Layout<_4,_1>{}, Layout<_8,_2>{});
```

---

## 8. Construction and Validation Utilities

#### `ceil_div(target, tiler)` — 107 uses

Ceiling division for layouts. Essential for grid dimension calculation.

```cpp
dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
// Implemented as: shape(complement(tiler, shape(target)))
```

#### `congruent(shape_a, shape_b)` — 16 uses

Validates that two shapes have the same structure and size.

```cpp
CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));
```

#### `make_ordered_layout(shape, order)` — 14 uses

Create a layout with strides in a specified order.

```cpp
auto layout = make_ordered_layout(make_shape(8, 4), LayoutRight{});   // (8,4):(4,1)
```

#### `make_identity_tensor(shape)` — 22 uses

Creates a tensor where each element equals its own coordinate. For predication.

```cpp
Tensor cS = make_identity_tensor(take<0,2>(CtaShapeQK{}));
// cS(m, n) == make_coord(m, n)
// copy_if(cS(_,_) < bound, src, dst)
```

#### `make_coord(coords...)` — 99 uses

Construct a coordinate (hierarchical IntTuple).

```cpp
auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);   // (m, n, k-loop)
tensor(make_coord(2, 3));                                   // access element at (2,3)
```

#### `recast<NewType>(tensor)` — 27 uses

Reinterpret a tensor's element type without changing its layout.

```cpp
Tensor tVec = recast<float4>(coalesce(tReg));
Tensor tOut = recast<Array<ElementOut, Alignment>>(coalesce(tFragment));
```
