# CuTe Basics & Navigation

CuTe is NVIDIA's C++ CUDA template library for defining and operating on hierarchically
multidimensional layouts of threads and data. The core insight: **everything is a Layout** —
a function from coordinates to indices. All higher-level abstractions (Tensor, MMA atoms,
Copy atoms, TiledMMA) are built on Layout composition.

Official docs: `build/_deps/repo-cutlass-src/media/docs/cpp/cute/`

## Library Map

### Core headers (`include/cute/`)

| Header | Defines |
|--------|---------|
| `int_tuple.hpp` | IntTuple concept: `rank()`, `get<I>()`, `size()`, `depth()` |
| `stride.hpp` | `crd2idx(coord, shape, stride)` — the fundamental coordinate→index mapping |
| `layout.hpp` | `Layout<Shape, Stride>`, `make_layout()`, `make_shape()`, `make_stride()`, `make_coord()`, `make_tile()`, `LayoutLeft`, `LayoutRight` |
| `layout_composed.hpp` | `ComposedLayout<LayoutA, Offset, LayoutB>` — for non-trivially composable layouts |
| `swizzle.hpp` | `Swizzle<BBits, MBase, SShift>` — XOR-based shared memory bank conflict avoidance |
| `tensor_impl.hpp` | `Tensor<Engine, Layout>`, `ArrayEngine`, `ViewEngine` |
| `tensor.hpp` | Entrypoint: includes tensor_impl + all algorithms (fill, copy, gemm, etc.) |
| `pointer.hpp` | `make_gmem_ptr()`, `make_smem_ptr()`, `make_rmem_ptr()`, `recast_ptr<T>()` |
| `underscore.hpp` | `Underscore` (`_` or `X`) — slice sentinel for tensor slicing |

### Algorithm headers (`include/cute/algorithm/`)

| Header | Defines |
|--------|---------|
| `copy.hpp` | `copy(src, dst)`, `copy_if(pred, src, dst)` — element-wise or CopyAtom-based |
| `gemm.hpp` | `gemm(D, A, B, C)` — D = A*B+C, dispatches by tensor rank |
| `fill.hpp` | `fill(tensor, value)` |
| `clear.hpp` | `clear(tensor)` — zeros a tensor |
| `axpby.hpp` | `axpby(alpha, x, beta, y)` — y = alpha*x + beta*y |
| `cooperative_copy.hpp` | Multi-threaded cooperative copy |
| `cooperative_gemm.hpp` | Cooperative shared-memory GEMM |
| `tuple_algorithms.hpp` | `transform`, `fold`, `for_each` on hierarchical tuples |
| `tensor_algorithms.hpp` | `for_each`, `transform`, `accumulate` on tensors |

### Architecture headers (`include/cute/arch/`)

Low-level PTX instruction wrappers. Key files:
- `mma_sm90_gmma.hpp` — SM90 WGMMA (warpgroup MMA) instructions, `GmmaDescriptor`
- `copy_sm90_tma.hpp` — SM90 TMA load/store (1D through 5D)
- `copy_sm90.hpp` — `SM90_U32x4_STSM_N`, `SM90_U16x8_STSM_T` (register→smem stores)
- `cluster_sm90.hpp` — `cluster_sync()`, `block_id_in_cluster()`, `cluster_id_in_grid()`
- `mma_sm90_desc.hpp` — MMA descriptor types

### Atom headers (`include/cute/atom/`)

| Header | Defines |
|--------|---------|
| `mma_traits.hpp` | `MMA_Traits` concept, `MMA_Atom`, `TiledMMA`, `make_tiled_mma()` |
| `mma_traits_sm90_gmma.hpp` | SM90 WGMMA traits (Shape_MNK, ThrID, ALayout/BLayout/CLayout) |
| `copy_atom.hpp` | `Copy_Atom`, `TiledCopy`, `ThrCopy` |
| `copy_traits.hpp` | `Copy_Traits` concept (ThrID, SrcLayout/DstLayout/RefLayout) |
| `partitioner.hpp` | `TV_Tiler` — generic thread-value tiling |

### Official documentation files

| File | Covers |
|------|--------|
| `01_layout.md` | Layout fundamentals: IntTuple, Shape, Stride, coordinate mapping, index mapping |
| `02_layout_algebra.md` | Coalesce, Composition, Complement, Division (tiling), Product (replication) |
| `03_tensor.md` | Tensor creation (owning/nonowning), access, slicing, partitioning |
| `04_algorithms.md` | copy, copy_if, gemm, axpby, fill, clear |
| `0t_mma_atom.md` | MMA atoms, traits, Volta/Hopper examples, TiledMMA construction |
| `0x_gemm_tutorial.md` | Full GEMM walkthrough from scratch |
| `0y_predication.md` | Handling non-even tiling with copy_if |
| `0z_tma_tensors.md` | TMA tensor support |

### Tutorial examples

| File | Demonstrates |
|------|-------------|
| `examples/cute/tutorial/sgemm_1.cu` | Basic GEMM: CTA tiling, thread partitioning, mainloop |
| `examples/cute/tutorial/sgemm_2.cu` | GEMM with TiledCopy and TiledMMA |
| `examples/cute/tutorial/hopper/wgmma_sm90.cu` | SM90 GEMM with cp.async + WGMMA |
| `examples/cute/tutorial/hopper/wgmma_tma_sm90.cu` | SM90 GEMM with TMA + WGMMA + cluster pipeline |

## Core Concepts

IntTuple and Layout are CuTe's foundational types. See [02-layout-algebra.md](02-layout-algebra.md) for details.
