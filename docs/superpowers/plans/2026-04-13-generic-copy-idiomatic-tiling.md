# Idiomatic Tiled Tiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `zipped_divide` with `tiled_divide` in TransposeCopyKernel for idiomatic CuTe tiling.

**Architecture:** Single 6-line edit in the kernel's tiling section. `tiled_divide` produces `((TM,TN), M/TM, N/TN)` with separate modes, enabling direct slice `(_, by, bx)` and eliminating the `get<0>` unwrap.

**Tech Stack:** CuTe `tiled_divide`, CUDA

---

### Task 1: Replace zipped_divide with tiled_divide

**Files:**
- Modify: `src/turbomind/core/tensor.cu:192-205`

- [ ] **Step 1: Replace tiling and slicing code**

Replace lines 192–205:

```cpp
    // Tile gmem tensors — inner (kTileDim, kTileDim) is static, outer is dynamic
    auto tiler = make_shape(Int<kTileDim>{}, Int<kTileDim>{});
    auto src_tiled = zipped_divide(src, tiler);
    auto dst_tiled = zipped_divide(dst, tiler);

    // Bounds check on tile grid
    if (blockIdx.y >= size<1, 0>(src_tiled) ||
        blockIdx.x >= size<1, 1>(src_tiled)) return;

    // Per-CTA tile — unwrap zipped rank-1 ((32,32)) to rank-2 (32,32) for TiledCopy
    auto src_tile_z = src_tiled(_, make_coord(blockIdx.y, blockIdx.x));
    auto dst_tile_z = dst_tiled(_, make_coord(blockIdx.y, blockIdx.x));
    auto src_tile = make_tensor(src_tile_z.data(), get<0>(src_tile_z.layout()));
    auto dst_tile = make_tensor(dst_tile_z.data(), get<0>(dst_tile_z.layout()));
```

with:

```cpp
    // Tile gmem tensors — tiled_divide produces ((TM,TN), M/TM, N/TN)
    auto tiler = make_shape(Int<kTileDim>{}, Int<kTileDim>{});
    auto src_tiled = tiled_divide(src, tiler);
    auto dst_tiled = tiled_divide(dst, tiler);

    // Bounds check on tile grid
    if (blockIdx.y >= size<1>(src_tiled) ||
        blockIdx.x >= size<2>(src_tiled)) return;

    // Per-CTA tile — direct slice gives rank-2 (TM,TN)
    auto src_tile = src_tiled(_, blockIdx.y, blockIdx.x);
    auto dst_tile = dst_tiled(_, blockIdx.y, blockIdx.x);
```

---

### Task 2: Build

- [ ] **Step 1: Build**

```bash
cd /data/lmdeploy-copy/build && ninja
```

Expected: Build succeeds.

---

### Task 3: Correctness test

- [ ] **Step 1: Run all tests**

```bash
cd /data/lmdeploy-copy && PYTHONPATH=lmdeploy:build/lib python test_generic_copy.py
```

Expected: `ALL TESTS PASSED`, including transpose and slice+transpose cases.

---

### Task 4: Commit

- [ ] **Step 1: Commit**

```bash
cd /data/lmdeploy-copy && git add src/turbomind/core/tensor.cu && \
git commit -m "refactor(core): replace zipped_divide with tiled_divide in TransposeCopyKernel"
```
