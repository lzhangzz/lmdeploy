# Iteration 02: Optimize WGMMA Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace scalar gmem loads and `warpgroup_wait<0>()` in the iter 01 WGMMA kernel with `cp.async.bulk` smem pipeline for packed A, restoring full WGMMA overlap.

**Architecture:** Add 3-stage smem pipeline for packed A (flat register layout) alongside existing B TMA pipeline. Both transfers arrive at a single mbarrier per stage. Consumer does per-thread S2R loads from smem, enabling `warpgroup_wait<2>()` instead of `wait<0>()`.

**Tech Stack:** CUDA 12.8, SM90 (Hopper), CuTe/CUTLASS, `SM90_BULK_COPY_G2S`, `PipelineTmaAsync`

---

### Task 1: Create kernel file with all changes

**Files:**
- Create: `cute-reference/mixed-gemm/02_bf16_gemm_sm90_split_a_wgmma.cu`
- Reference: `cute-reference/mixed-gemm/01_bf16_gemm_sm90_split_a_wgmma.cu` (copy and modify)

- [ ] **Step 1: Copy iter 01 file**

```bash
cp cute-reference/mixed-gemm/01_bf16_gemm_sm90_split_a_wgmma.cu \
   cute-reference/mixed-gemm/02_bf16_gemm_sm90_split_a_wgmma.cu
```

- [ ] **Step 2: Update header comment**

Replace the header comment block (lines 1-17) with:

```cpp
/***************************************************************************************************
 * BF16 GEMM using SM90 WGMMA with pre-packed operand A (iteration 02).
 *
 * Optimized version of iteration 01: loads packed A via cp.async.bulk into a smem pipeline
 * (3 stages) instead of scalar gmem loads. This enables warpgroup_wait<2>() instead of
 * warpgroup_wait<0>(), restoring WGMMA overlap across k_tiles.
 *
 * Changes from iteration 01:
 *   - A smem pipeline: cp.async.bulk gmem->smem + per-thread smem->register loads
 *   - Single mbarrier per stage for both A (bulk) and B (TMA) arrivals
 *   - warpgroup_wait<2>() instead of warpgroup_wait<0>()
 *
 * Target: SM90
 **************************************************************************************************/
```

- [ ] **Step 3: Update SharedStorage to add A smem allocation**

Replace the SharedStorage struct (lines 25-32) with:

```cpp
template <class ElementA, int AStageElements, class ElementB, class ElementC,
          class SmemLayoutB, class SmemLayoutC, int Stages>
struct WgmmaSharedStorage
{
  alignas(128) cute::ArrayEngine<ElementA, AStageElements * Stages> A;
  alignas(128) cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
  alignas(128) cute::ArrayEngine<ElementC, cute::cosize_v<SmemLayoutC>> C;
  typename cutlass::PipelineTmaAsync<Stages>::SharedStorage pipeline;
};
```

`AStageElements = bM * bK = 8192` bf16 elements per stage.

- [ ] **Step 4: Update smem tensors and transaction_bytes in kernel**

In the kernel function body, update the SharedStorage type and add transaction_bytes for A.

Replace lines 62-73:

```cpp
  extern __shared__ char shared_memory[];
  using SharedStorage = WgmmaSharedStorage<TB, TC, SmemLayoutB, SmemLayoutC, cute::size<2>(SmemLayoutB{})>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});
  Tensor sC = make_tensor(make_smem_ptr(smem.C.begin()), SmemLayoutC{});
```

With:

```cpp
  extern __shared__ char shared_memory[];
  static_assert(decltype(size<0>(cta_tiler))::value == 128);
  static_assert(decltype(size<2>(cta_tiler))::value == 64);
  constexpr int a_stage_elements = 128 * 64;
  using SharedStorage = WgmmaSharedStorage<bf16_t, a_stage_elements, TB, TC,
      SmemLayoutB, SmemLayoutC, cute::size<2>(SmemLayoutB{})>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});
  Tensor sC = make_tensor(make_smem_ptr(smem.C.begin()), SmemLayoutC{});
```

`a_stage_elements = bM * bK = 8192` bf16 elements per A pipeline stage. Static asserts verify the
cta_tiler dimensions match. `regs_per_thread * 256 == a_stage_elements` (confirmed: 32 * 256 = 8192).

Replace the transaction_bytes computation (lines 71-72):

```cpp
  constexpr int tma_transaction_bytes =
      sizeof(TB) * cute::cosize_v<SmemLayoutB> / cute::size<2>(SmemLayoutB{});
```

With:

```cpp
  constexpr int tma_transaction_bytes =
      sizeof(bf16_t) * a_stage_elements
    + sizeof(TB) * cute::cosize_v<SmemLayoutB> / cute::size<2>(SmemLayoutB{});
```

This adds A's bulk copy bytes (16 KB) to B's TMA bytes (32 KB) = 48 KB total per stage.

- [ ] **Step 5: Update producer — add bulk copy for A**

Replace the producer block (lines 99-127) with:

```cpp
  if (warp_group_idx == 2) {
    // ==================================================================
    // Producer warp group — bulk copy for A + TMA for B (persistent)
    // ==================================================================
    cutlass::arch::warpgroup_reg_dealloc<40>();

    if (warp_group_thread_idx == 0) {
      using BarrierType = typename MainloopPipeline::ProducerBarrierType;

      while (linear_idx < total_tiles) {
        int m_idx = linear_idx / n_tiles;
        int n_idx = linear_idx % n_tiles;

        Tensor gB = local_tile(mB, cta_tiler, make_coord(m_idx, n_idx, _), Step< X,_1, _1>{});
        auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                           group_modes<0,2>(sB), group_modes<0,2>(gB));

        CUTE_NO_UNROLL
        for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
          pipeline.producer_acquire(smem_pipe_write);
          BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
          int write_stage = smem_pipe_write.index();

          // Issue bulk copy for packed A: gmem -> smem_A[write_stage]
          {
            int a_tile_idx = m_idx * k_tile_count + k_tile;
            SM90_BULK_COPY_G2S::copy(
                packed_A + a_tile_idx * a_stage_elements,
                tma_barrier,
                smem.A.begin() + write_stage * a_stage_elements,
                a_stage_elements * sizeof(bf16_t));
          }

          // Issue TMA for B
          copy(tma_b.with(*tma_barrier), tBgB(_,k_tile), tBsB(_,write_stage));

          ++smem_pipe_write;
        }

        linear_idx += grid_size;
      }
      pipeline.producer_tail(smem_pipe_write);
    }

    cute::tma_store_wait<0>();
```

Key points:
- `SM90_BULK_COPY_G2S::copy(gmem, mbar, smem, bytes)` is a cute wrapper that issues
  `cp.async.bulk...mbarrier::complete_tx::bytes` with correct pointer conversions
- It arrives at the same mbarrier as TMA for B — both use `mbarrier.complete_tx::bytes` to
  track completion. The barrier was initialized with `transaction_bytes = A_bytes + B_bytes`
  (set in Step 4), so it flips only after both transfers complete
- `producer_get_barrier()` returns `uint64_t*` (ProducerBarrierType), matching the function signature

- [ ] **Step 6: Update consumer — smem S2R for A, remove warpgroup_wait<0>()**

Replace the consumer's k_tile loop (lines 177-213):

```cpp
        CUTE_NO_UNROLL
        for (int k_tile_iter = 0; k_tile_iter < k_tile_count; ++k_tile_iter)
        {
          pipeline.consumer_wait(smem_pipe_read);
          int read_stage = smem_pipe_read.index();
          ++smem_pipe_read;

          // Load packed A directly from gmem into registers
          int a_tile_idx = m_idx * k_tile_count + k_tile_iter;
          const bf16_t* packed_ptr = packed_A + a_tile_idx * 256 * regs_per_thread;
          CUTE_UNROLL
          for (int i = 0; i < regs_per_thread; ++i) {
            tCrA(i) = packed_ptr[i * 256 + threadIdx.x];
          }

          warpgroup_fence_operand(tCrC);

          // Batched WGMMA: issue all k_blocks with wait<2> for overlap
          CUTLASS_PRAGMA_UNROLL
          for (int k_block = 0; k_block < k_block_count; ++k_block)
          {
            warpgroup_arrive();
            gemm(mma, tCrA(_,_,k_block),
                      tCrB(_,_,k_block,read_stage), tCrC);
            warpgroup_commit_batch();
            warpgroup_wait<2>();
          }

          // Wait for ALL WGMMA to complete before overwriting tCrA and releasing B stage.
          // This is required because tCrA is reused across k_tiles (no pipeline dimension),
          // unlike sample 13 where A also flows through the smem pipeline.
          warpgroup_fence_operand(tCrC);
          warpgroup_wait<0>();

          pipeline.consumer_release(smem_pipe_release);
          ++smem_pipe_release;
        }
```

With:

```cpp
        CUTE_NO_UNROLL
        for (int k_tile_iter = 0; k_tile_iter < k_tile_count; ++k_tile_iter)
        {
          pipeline.consumer_wait(smem_pipe_read);
          int read_stage = smem_pipe_read.index();
          ++smem_pipe_read;

          // Load packed A from smem pipeline stage into registers
          const bf16_t* smem_a_ptr = smem.A.begin() + read_stage * a_stage_elements;
          CUTE_UNROLL
          for (int i = 0; i < regs_per_thread; ++i) {
            tCrA(i) = smem_a_ptr[i * 256 + threadIdx.x];
          }

          warpgroup_fence_operand(tCrC);

          // Batched WGMMA: issue all k_blocks with wait<2> for overlap
          CUTLASS_PRAGMA_UNROLL
          for (int k_block = 0; k_block < k_block_count; ++k_block)
          {
            warpgroup_arrive();
            gemm(mma, tCrA(_,_,k_block),
                      tCrB(_,_,k_block,read_stage), tCrC);
            warpgroup_commit_batch();
            warpgroup_wait<2>();
          }

          warpgroup_fence_operand(tCrC);

          pipeline.consumer_release(smem_pipe_release);
          ++smem_pipe_release;
        }
```

Changes:
- A loads from `smem.A.begin() + read_stage * a_stage_elements` instead of packed gmem pointer
- Removed `warpgroup_wait<0>()` — `consumer_wait` on the next stage provides the sync barrier before `tCrA` is overwritten
- Removed the `a_tile_idx` computation (A stage is identified by `read_stage`)

- [ ] **Step 7: Update host function — smem size and SharedStorage type**

In the host function `split_a_wgmma`, the SmemLayoutA for fragment creation stays unchanged:

```cpp
  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16_t>{}, make_shape(bM, bK, Int<1>{}));
```

Update the smem_size computation. Replace line 309:

```cpp
  int smem_size = int(sizeof(WgmmaSharedStorage<bf16_t, bf16_t, decltype(sB), decltype(sC_layout), cute::size<2>(decltype(sB){})>));
```

With:

```cpp
  constexpr int a_stage_elements_host = 128 * 64; // must match kernel's a_stage_elements
  int smem_size = int(sizeof(WgmmaSharedStorage<bf16_t, a_stage_elements_host, bf16_t, bf16_t,
      decltype(sB), decltype(sC_layout), cute::size<2>(decltype(sB){})>));
```

The rest of the host function (kernel launch, test harness, benchmark) is unchanged.

- [ ] **Step 8: Update kernel function signature comment**

The kernel signature itself doesn't change. The `SmemLayoutA` parameter is still used for
dummy fragment creation (partition_A / make_fragment_A). The actual A smem is in the
SharedStorage's flat buffer.

- [ ] **Step 9: Commit**

```bash
git add cute-reference/mixed-gemm/02_bf16_gemm_sm90_split_a_wgmma.cu
git commit -m "Add iter 02 WGMMA kernel: cp.async.bulk pipeline for packed A"
```

---

### Task 2: Update build system

**Files:**
- Modify: `cute-reference/mixed-gemm/CMakeLists.txt`

- [ ] **Step 1: Add iter 02 target**

Add `02_bf16_gemm_sm90_split_a_wgmma` to the `MIXED_GEMM_TARGETS` list:

```cmake
set(MIXED_GEMM_TARGETS
    01_bf16_gemm_sm90_split_a_pack
    01_bf16_gemm_sm90_split_a_wgmma
    02_bf16_gemm_sm90_split_a_wgmma
)
```

- [ ] **Step 2: Commit**

```bash
git add cute-reference/mixed-gemm/CMakeLists.txt
git commit -m "Add iter 02 WGMMA kernel to build targets"
```

---

### Task 3: Build and verify correctness

- [ ] **Step 1: Reconfigure build**

```bash
cd /data/lmdeploy-cute/build && sh ../my_generate.sh 2>&1 | tail -5
```

Expected: cmake configuration succeeds.

- [ ] **Step 2: Build iter 02**

```bash
cd /data/lmdeploy-cute/build && ninja 02_bf16_gemm_sm90_split_a_wgmma 2>&1
```

Expected: compiles without errors. PTX warnings about register usage are acceptable.

- [ ] **Step 3: Check GPU availability**

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
```

Expected: at least one GPU with minimal memory usage (< 1000 MiB used).

- [ ] **Step 4: Run correctness tests**

```bash
cd /data/lmdeploy-cute/build && ./02_bf16_gemm_sm90_split_a_wgmma
```

Expected output: all test sizes (128x256x64 through 2048x1024x512) report PASS with max error < 0.5.
Benchmark results print GFLOP/s for sizes 256-4096.

If any test fails, debug and fix before proceeding.

- [ ] **Step 5: Commit (if any fixes were needed)**

```bash
git add cute-reference/mixed-gemm/02_bf16_gemm_sm90_split_a_wgmma.cu
git commit -m "Fix iter 02 correctness: <description>"
```

---

### Task 4: Benchmark and compare

- [ ] **Step 1: Run iter 01 benchmark for comparison**

```bash
cd /data/lmdeploy-cute/build && ninja 01_bf16_gemm_sm90_split_a_wgmma && ./01_bf16_gemm_sm90_split_a_wgmma
```

Record GFLOP/s for each size.

- [ ] **Step 2: Run iter 02 benchmark**

```bash
cd /data/lmdeploy-cute/build && ./02_bf16_gemm_sm90_split_a_wgmma
```

Record GFLOP/s for each size.

- [ ] **Step 3: Compare and document results**

Update the design doc (`docs/superpowers/specs/2026-04-28-iter02-optimize-wgmma-design.md`)
with observed performance numbers and comparison with iter 01 and sample 13.

- [ ] **Step 4: Commit results**

```bash
git add docs/superpowers/specs/2026-04-28-iter02-optimize-wgmma-design.md
git commit -m "Add iter 02 performance results"
```
