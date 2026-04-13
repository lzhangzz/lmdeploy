# TiledCopy TransposeCopyKernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `cooperative_copy` calls in `TransposeCopyKernel` with explicit `TiledCopy` (two `make_tiled_copy` objects, one per phase).

**Architecture:** Internal kernel refactor — same signature, same smem layout, same host dispatch. Only the two `cooperative_copy` calls (lines 213-217 of `tensor.cu`) are replaced with `make_tiled_copy` + `get_slice` + `partition_S/D` + `copy` sequences.

**Tech Stack:** CuTe TiledCopy API (`make_tiled_copy`, `Copy_Atom`, `UniversalCopy`), CUDA

---

### Task 1: Replace cooperative_copy with TiledCopy

**Files:**
- Modify: `src/turbomind/core/tensor.cu:168-217`

- [ ] **Step 1: Update the kernel comment header**

Replace the comment block at lines 168-174:

```cpp
// ============================================================================
// CUDA kernel: TransposeCopyKernel (cooperative_copy 2D layout conversion)
// ============================================================================
// Copies a 2D tensor from src to dst where src and dst have orthogonal
// contiguous dimensions (src contiguous on dim 0, dst contiguous on dim 1).
// Uses smem staging with two views of a flat buffer: row-major write view
// for Phase 1, column-major read view for Phase 2 (implicit transpose).
// Both phases use kMaxVecBits (scalar = 8*sizeof(T) from host).
```

with:

```cpp
// ============================================================================
// CUDA kernel: TransposeCopyKernel (TiledCopy 2D layout conversion)
// ============================================================================
// Copies a 2D tensor from src to dst where src and dst have orthogonal
// contiguous dimensions (src contiguous on dim 0, dst contiguous on dim 1).
// Uses smem staging with two views of a flat buffer: row-major write view
// for Phase 1, column-major read view for Phase 2 (implicit transpose).
// Both phases use explicit TiledCopy (scalar Copy_Atom, 256 threads).
```

- [ ] **Step 2: Replace Phase 1 and Phase 2 cooperative_copy calls**

Replace lines 213-217:

```cpp
    // Phase 1: gmem(src) -> smem (row-major)
    cute::cooperative_copy<256, kMaxVecBits>(threadIdx.x, src_tile, smem_w);
    __syncthreads();
    // Phase 2: smem -> gmem(dst) (column-major view — transposed)
    cute::cooperative_copy<256, kMaxVecBits>(threadIdx.x, smem_r, dst_tile);
```

with:

```cpp
    // Phase 1: gmem(src) -> smem (row-major) via TiledCopy
    auto tc1 = cute::make_tiled_copy(
        cute::Copy_Atom<cute::UniversalCopy<T>, T>{},
        cute::make_layout(cute::make_shape(cute::Int<256>{})),
        cute::make_layout(cute::make_shape(cute::Int<1>{})));
    auto thr1 = tc1.get_slice(threadIdx.x);
    cute::copy(tc1, thr1.partition_S(src_tile), thr1.partition_D(smem_w));

    __syncthreads();

    // Phase 2: smem -> gmem(dst) (column-major view — transposed) via TiledCopy
    auto tc2 = cute::make_tiled_copy(
        cute::Copy_Atom<cute::UniversalCopy<T>, T>{},
        cute::make_layout(cute::make_shape(cute::Int<256>{})),
        cute::make_layout(cute::make_shape(cute::Int<1>{})));
    auto thr2 = tc2.get_slice(threadIdx.x);
    cute::copy(tc2, thr2.partition_S(smem_r), thr2.partition_D(dst_tile));
```

- [ ] **Step 3: Verify the full kernel reads correctly**

Read `src/turbomind/core/tensor.cu:167-225` and verify:
- Kernel signature unchanged: `template<int kTileDim, uint32_t kMaxVecBits, ...>`
- smem views unchanged (row-major write, column-major read)
- `zipped_divide` tiling and bounds check unchanged
- Two `make_tiled_copy` calls with `Copy_Atom<UniversalCopy<T>, T>` (scalar)
- No remaining references to `cooperative_copy` in the kernel

---

### Task 2: Build

**Files:** None (build only)

- [ ] **Step 1: Build the tensor target**

```bash
cd /data/lmdeploy-copy/build && ninja turbomind-core-static
```

Expected: Build succeeds with no errors.

---

### Task 3: Correctness tests

**Files:**
- Test: `test_generic_copy.py`

- [ ] **Step 1: Run transpose correctness tests for all dtypes**

```bash
cd /data/lmdeploy-copy && \
PYTHONPATH=lmdeploy:build/lib python test_generic_copy.py 2>&1 | grep -E '(transpose|PASS|FAIL|Error)'
```

Expected: All transpose test cases pass:
- `transpose f32` — PASS
- `slice+transpose` — PASS
- `transpose f16` — PASS
- `transpose i8` — PASS
- `transpose i32` — PASS

---

### Task 4: Throughput benchmark

**Files:**
- Test: `test_generic_copy.py`

- [ ] **Step 1: Run transpose throughput sweep**

```bash
cd /data/lmdeploy-copy && \
PYTHONPATH=lmdeploy:build/lib python test_generic_copy.py 2>&1 | grep -E '(trans N K|GB/s)'
```

Expected: Throughput numbers comparable to the previous `cooperative_copy` implementation (within ~5%). Record the numbers for comparison.

---

### Task 5: Commit

- [ ] **Step 1: Stage and commit**

```bash
cd /data/lmdeploy-copy && \
git add src/turbomind/core/tensor.cu && \
git commit -m "refactor(core): replace cooperative_copy with TiledCopy in TransposeCopyKernel"
```
