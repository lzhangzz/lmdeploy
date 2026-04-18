# MMA (Matrix Multiply-Accumulate)

## MMA Atoms

CuTe's abstraction for a single hardware MMA instruction. The atom concept has three layers:

### Operation (arch/)

Wraps a single PTX instruction. Minimal dependencies — no layouts, no tensors.
Defines register types and `fma()` static method.

```cpp
struct MMA_Operation {
  using DRegisters = ...;  // output register type and count
  using ARegisters = ...;  // A operand register type and count
  using BRegisters = ...;  // B operand register type and count
  using CRegisters = ...;  // C accumulator register type and count
  static void fma(DRegisters& d, ARegisters const& a, BRegisters const& b, CRegisters& c);
};
```

### Traits (atom/mma_traits_*.hpp)

Defines metadata about the Operation:

- `ValTypeD/A/B/C` — logical compute types
- `Shape_MNK` — logical MxNxK shape of the operation
- `ThrID` — thread index mapping (warp, warpgroup, etc.)
- `ALayout` — `(thread_id, value_id) → (m, k)` coordinate mapping
- `BLayout` — `(thread_id, value_id) → (n, k)` coordinate mapping
- `CLayout` — `(thread_id, value_id) → (m, n)` coordinate mapping

```cpp
template <>
struct MMA_Traits<MMA_Operation> {
  using ValTypeD = ...;
  using Shape_MNK = Shape<_M, _N, _K>;
  using ThrID   = Layout<...>;       // thread mapping
  using ALayout = Layout<...>;       // (tid, vid) -> (m, k)
  using BLayout = Layout<...>;       // (tid, vid) -> (n, k)
  using CLayout = Layout<...>;       // (tid, vid) -> (m, n)
};
```

### MMA_Atom

Wraps Traits with `fma()` and fragment allocation:
```cpp
MMA_Atom atom = MMA_Atom<MMA_Operation>{};
// atom.fma(regA, regB, regC, regD) — execute the instruction
// atom.make_fragment_A/B/C(layout) — allocate register fragments
```

## TiledMMA

Tiles an MMA_Atom across multiple threads and values to create larger MMA operations.

```cpp
template <class MMA_Atom, class AtomLayoutMNK, class PermutationMNK>
struct TiledMMA : MMA_Atom
```

- `AtomLayoutMNK` — how to tile atoms across M, N, K dimensions (replicate across threads)
- `PermutationMNK` — reordering to apply before tiling (reorder values)

### Construction

```cpp
// Single atom (default)
TiledMMA mma = make_tiled_mma(mma_atom{});

// Tile atom across threads (e.g., 2×2 layout of atoms)
TiledMMA mma = make_tiled_mma(mma_atom{},
                              Layout<Shape<_2,_2>, Stride<_2,_1>>{});

// Tile across threads AND values
TiledMMA mma = make_tiled_mma(mma_atom{},
                              Layout<Shape<_2,_2>, Stride<_2,_1>>{},
                              Tile<_32,_32,_4>{});

// With permutation (reorder modes for better smem access)
TiledMMA mma = make_tiled_mma(mma_atom{},
                              Layout<Shape<_2,_2>, Stride<_2,_1>>{},
                              Tile<Layout<Shape<_4,_4,_2>, Stride<_1,_8,_4>>,
                                   _32, _4>{});
```

### Usage — Partitioning Tensors

```cpp
TiledMMA mma = make_tiled_mma(mma_atom{});

// Get this thread's slice
ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);

// Partition shared memory tensors
Tensor tCsA = thr_mma.partition_A(sA);    // (MMA, MMA_M, MMA_K, PIPE)
Tensor tCsB = thr_mma.partition_B(sB);    // (MMA, MMA_N, MMA_K, PIPE)
Tensor tCgC = thr_mma.partition_C(gC);    // (MMA, MMA_M, MMA_N)

// Allocate register fragments
Tensor tCrA = thr_mma.make_fragment_A(tCsA);  // register A fragment
Tensor tCrB = thr_mma.make_fragment_B(tCsB);  // register B fragment
Tensor tCrC = thr_mma.make_fragment_C(tCgC);  // accumulator (register C)

// Execute
clear(tCrC);                                    // zero accumulators
gemm(mma, tCrA, tCrB, tCrC);                   // tCrC += tCrA * tCrB
```

## Architecture-Specific MMA Atoms

Architecture-specific atoms (with full details):
- [06-sm70.md](06-sm70.md) — SM70 (Volta): quadpair 8×8×4 FP16
- [07-sm75.md](07-sm75.md) — SM75 (Turing): warp 16×8×8 FP16/INT8
- [08-sm80.md](08-sm80.md) — SM80/SM89 (Ampere/Ada): FP16/BF16/TF32/INT8/FP8
- [09-sm90.md](09-sm90.md) — SM90 (Hopper): WGMMA warpgroup-level

## Debugging MMA

```cpp
print(mma)                    // print TiledMMA structure
print_latex(mma)              // LaTeX visualization of partitioning pattern
print(tCrC)                   // print accumulator fragment values
```
