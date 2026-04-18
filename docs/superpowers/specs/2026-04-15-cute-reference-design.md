# CuTe Reference Document for LLM Agents — Design Spec

## Goal

Create a standalone context document (`cute-reference.md` at repo root) that helps LLM agents understand NVIDIA's CuTe (CUDA Templates) library well enough to write new CUDA kernels.

## Context

- CuTe source: `build/_deps/repo-cutlass-src/include/cute/` — 109 header files
- Official CuTe docs: `build/_deps/repo-cutlass-src/media/docs/cpp/cute/` — 9 markdown files (tutorials, not quick references)
- CuTe examples: `build/_deps/repo-cutlass-src/examples/cute/tutorial/` — including Hopper-specific WGMMA+TMA examples
- No existing CuTe documentation or reference in the project

## Scope

- **In scope**: CuTe library concepts, patterns, and composition for kernel writing
- **Out of scope**: lmdeploy-specific usage patterns, custom wrappers, project internals

## Document Structure (5 sections, ~270 lines)

### 1. Library Map (~30 lines)
File-to-concept table mapping each key CuTe header to what it defines, plus pointers to the 9 official doc files with one-line descriptions.

### 2. Concept Cheat Sheet (~80 lines)
Six core concepts (IntTuple, Layout, Tensor, MMA Atom, Copy Atom, Swizzle), each 10-15 lines: what it is, key types, key operations, which official doc to read.

### 3. Layout Algebra Quick Reference (~40 lines)
Four key operations (composition, complement, divide, product) with one-paragraph explanations and code snippets.

### 4. Kernel Composition Patterns (~60 lines)
How concepts combine for a full GEMM kernel:
- CTA-level tiling (`local_tile`)
- Thread-level partitioning (TiledCopy)
- MMA partitioning (TiledMMA, `partition_fragment`)
- Mainloop (WGMMA + TMA pipelining, referencing wgmma_tma_sm90.cu)
- Epilogue (register → global via shared memory)

### 5. SM90/Hopper Specifics (~40 lines)
WGMMA atoms, TMA operations, cluster synchronization, STSM operations — the primitives most relevant for Hopper kernel development.

## References

- Official docs: `build/_deps/repo-cutlass-src/media/docs/cpp/cute/`
  - `01_layout.md` — Layout fundamentals
  - `02_layout_algebra.md` — Composition, complement, divide, product
  - `03_tensor.md` — Tensor creation, slicing, partitioning
  - `04_algorithms.md` — copy, gemm, fill, clear
  - `0t_mma_atom.md` — MMA atoms, Volta/Hopper examples
  - `0x_gemm_tutorial.md` — Full GEMM walkthrough
  - `0y_predication.md` — Predication for non-even tiling
  - `0z_tma_tensors.md` — TMA tensor support
- Tutorial examples: `build/_deps/repo-cutlass-src/examples/cute/tutorial/hopper/wgmma_tma_sm90.cu`
