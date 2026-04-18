# CuTe Reference

Quick-reference for NVIDIA's CuTe (CUDA Templates) library.
Each file covers one topic. Read in order or jump to what you need.

| File | Topic |
|------|-------|
| [01-basics.md](01-basics.md) | IntTuple, Layout concepts, library map, official docs index |
| [02-layout-algebra.md](02-layout-algebra.md) | coalesce, composition, complement, divide, product, mode manipulation |
| [03-tensors.md](03-tensors.md) | Tensor creation, memory spaces, slicing, partitioning, algorithms |
| [04-copy-swizzle.md](04-copy-swizzle.md) | Copy atoms, TiledCopy, `copy`/`copy_if`, swizzle |
| [05-mma.md](05-mma.md) | MMA atoms, TiledMMA (architecture-agnostic) |
| [08-sm80.md](08-sm80.md) | SM80/SM89 (Ampere/Ada): cp.async, FP16/BF16/TF32/INT8/FP8 MMA |
| [09-sm90.md](09-sm90.md) | SM90 (Hopper): TMA, WGMMA, clusters, STSM |
| [06-sm70.md](06-sm70.md) | SM70 (Volta): quadpair 8×8×4 HMMA |
| [07-sm75.md](07-sm75.md) | SM75 (Turing): warp-level MMA, ldsm copy |

Official CuTe docs: `build/_deps/repo-cutlass-src/media/docs/cpp/cute/`
