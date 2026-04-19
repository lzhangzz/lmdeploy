# BF16 GEMM SM80 — Persistent Warp-Specialized TMA Kernel

## Overview

Add a persistent variant of `07_bf16_gemm_sm80_pipe_tma_ws.cu` using CUTLASS's stride-based work distribution. Each CTA processes multiple output tiles in a loop, striding by the grid size between tiles. This eliminates kernel launch overhead for small GEMMs and improves occupancy for problems that don't saturate the GPU.

New file: `08_bf16_gemm_sm80_pipe_tma_ws_persistent.cu`.

Based on study of CUTLASS 3.x's `sm90_gemm_tma_warpspecialized_cooperative.hpp` persistent kernel.

## What Changes

### Kernel Structure

The kernel gains an outer `while` loop wrapping the existing producer and consumer branches:

```
pipeline init + __syncthreads()  // once, before the loop
producer_state = make_producer_start_state()
consumer_read_state = PipelineState()
consumer_release_state = PipelineState()

while (linear_idx < total_tiles) {
    m_idx = linear_idx / n_tiles
    n_idx = linear_idx % n_tiles

    // recompute gmem tile views for (m_idx, n_idx)
    // recompute TMA gmem partitions (smem partitions are constant)

    if (warp_group_idx == 2) {
        // Producer: TMA loads — ++smem_pipe_write inside the k_tile loop
        //           advances state naturally (no separate .advance() needed)
    } else {
        // Consumer: ++smem_pipe_read / ++smem_pipe_release inside the k_tile loop
        //           advance states naturally (no separate .advance() needed)
        // Epilogue: scale, STSM, TMA store
    }

    linear_idx += grid_size
}
producer_tail(producer_state)  // drain once after loop
```

**Pipeline state advancement:** Unlike CUTLASS's cooperative kernel (which passes state by value to `load()`/`mma()` functions that internally increment their local copy), our inlined loops directly increment `smem_pipe_write`, `smem_pipe_read`, and `smem_pipe_release` via `++` inside the per-k_tile for loops. These `++` operations ARE the tile's advancement. We do NOT call `.advance()` after the inner loops — that would double-advance.

### Tile Scheduling: CUTLASS Stride-Based

Each CTA starts at `linear_idx = blockIdx.x` (1D grid, one block per SM) and strides by `gridDim.x` each iteration. Linear-to-2D mapping:

```
m_idx = linear_idx / n_tiles
n_idx = linear_idx % n_tiles
```

where `n_tiles = ceil_div(N, bN)`.

If `m_idx >= ceil_div(M, bM)`, the tile is invalid and the CTA exits the loop.

No atomics, no global counter, no inter-CTA synchronization. The striding guarantees even work distribution as long as `total_tiles >= grid_size`.

### Pipeline Lifecycle

- `PipelineTmaAsync` constructor runs once at kernel entry (all 384 threads participate in barrier init + `__syncthreads()`).
- `PipelineState` objects are created once before the loop and **never reset**. The `++` operators inside the per-tile k_tile loops advance the states naturally across tiles. CUTLASS's cooperative kernel passes state by value to `load()`/`mma()` and calls `.advance()` at the caller, but since we inline these loops, the `++` inside the loops serves the same purpose. The index wraps modulo pipeline depth, and the phase flips accordingly — the barrier phase-parity mechanism is self-sustaining across tiles.
- `producer_tail` is called once after the while loop exits (not per tile). It drains remaining barrier waits.
- `tma_store_wait<0>()` at the end of each tile's epilogue acts as the inter-tile sync: ensures the TMA store completes before smem is overwritten for the next tile.

### Per-Tile vs Once: What Gets Recomputed

| Computation | Frequency | Why |
|---|---|---|
| TMA descriptors (`tma_a`, `tma_b`, `tma_store_c`) | **Once** (host-side) | Immutable, encode full tensor geometry |
| `mA`, `mB`, `mC` (full gmem tensors) | **Once** | Same shape/strides every tile |
| `sA`, `sB`, `sC` (smem tensors) | **Once** | Same physical smem, same layout |
| `local_tile(mA, cta_tiler, make_coord(m_idx, n_idx, _), ...)` | **Per-tile** | Different CTA coordinate each tile |
| `tma_partition` gmem side (`tAgA`, `tBgB`) | **Per-tile** | Depends on per-tile gmem tensor |
| `tma_partition` smem side (`tAsA`, `tBsB`) | **Per-tile** | Easiest to recompute alongside gmem side (pure function, cheap) |
| S2R copy setup (`thr_s2r_a`, `tXsA`, etc.) | **Once** | Thread mapping and smem layout don't change |
| MMA setup (`thr_mma`, `tCrA`, `tCrB`, `tCrC`) | **Once** | Same register allocation, same smem |
| R2S copy setup (`thr_r2s`, `tRS_sC`) | **Once** | Same smem layout, same thread mapping |
| TMA store partitions (`tSsC`, `tSgC`) | **Once** | Cover full (M,N) tensor, indexed by `linear_idx` |
| `tCgC` (beta-load partition from `gC`) | **Per-tile** | `gC` changes per tile, so `tCgC` must be recomputed |
| `k_tile_count` | **Once** | `ceil_div(K, bK)` — same for all tiles since K is constant |
| `clear(tCrC)` | **Per-tile** | Must reset accumulators before each tile |
| `PipelineState` objects | **Advanced** per-tile | `++` inside k_tile loops advances state across tiles — never reset or `.advance()` |
| `producer_tail` | **Once** after loop | Drain remaining barriers at kernel exit |

### Device Kernel Changes (from 07)

1. **New parameter:** `int total_tiles` — total number of output tiles. Computed on host as `ceil_div(M, bM) * ceil_div(N, bN)`.

2. **Grid is 1D:** The kernel uses `blockIdx.x` directly as the linear tile index. Grid shape is `dim3(num_SMs)`.

3. **Persistent while loop** wraps producer and consumer branches:
   - At loop top: compute `m_idx, n_idx` from `linear_idx` via divmod. Exit if `m_idx >= m_tiles`.
   - Recompute `gA, gB, gC` via `local_tile` with `(m_idx, n_idx)`.
   - Recompute TMA partitions (`tAgA/tAsA`, `tBgB/tBsB`) from the new gmem tensors.
   - Recompute `tCgC` (beta-load partition) from the new `gC`.
   - Producer: uses `producer_state` (advanced between tiles, not reset).
   - Consumer: uses `consumer_read_state`/`consumer_release_state` (advanced between tiles).
   - After epilogue TMA store + `tma_store_wait<0>()`: advance all pipeline states by `k_tile_count`, advance `linear_idx += gridDim.x`.
   - After while loop: call `producer_tail` once.

4. **gC for beta-load recomputed per tile:** `tCgC = thr_mma.partition_C(gC)` where `gC` is the per-tile `local_tile` result.

5. **Accumulator clearing:** `clear(tCrC)` moves inside the while loop (before each tile's computation).

### Host Function Changes

1. **Query SM count:** `cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, device)`.

2. **Grid shape:** `dim3 dimGrid(min(num_SMs, total_tiles))` — 1D grid, capped at SM count. Never launch more CTAs than tiles.

3. **New kernel argument:** `total_tiles` passed after `beta`.

4. **Everything else unchanged:** TMA descriptors, smem layouts, TiledMMA, r2s_copy, s2r_atoms, shared memory size, kernel attributes.

### Benchmark Changes

Same benchmark harness as 07 (warmup + 100 iterations, TFLOP/s calculation). Added size 256³ for small-GEMM coverage. Full sweep: 256³, 512³, 1024³, 2048³, 4096³, 8192³.

## What Stays the Same

- MMA atom: `SM80_16x8x16_F32BF16BF16F32_TN`
- MMA layout: `Layout<Shape<_4, _2>>` (8 warps = 256 consumer threads)
- Tile override: `Tile<Underscore, _64, Underscore>`
- CTA tile: (256, 128, 64)
- Pipeline depth: 3 stages
- Pipeline: `PipelineTmaAsync`
- S2R atoms: `SM75_U32x4_LDSM_N`
- Smem layouts: `GMMA::Layout_K_SW128_Atom` for A/B, plain column-major for C
- STSM register-to-smem copy: `make_tiled_copy_C(SM90_U16x8_STSM_T, mma)`
- TMA store: `make_tma_copy(SM90_TMA_STORE{}, mC, sC_layout, ...)`
- Epilogue: alpha/beta scaling, F32->BF16, STSM, TMA store
- SharedStorage struct: unchanged
- Register allocation: `warpgroup_reg_dealloc<40>` / `warpgroup_reg_alloc<232>`
- Consumer-only NamedBarrier(256, 6) for epilogue sync
- Warp group dispatch: wg 2 = producer, wg 0+1 = consumer

## Kernel Pseudocode

```cpp
__global__ static
__launch_bounds__(384, 1)
void
bf16_gemm_persistent_device(..., int total_tiles)
{
  extern __shared__ char shared_memory[];
  SharedStorage& smem = ...;

  using namespace cute;
  auto [M, N, K] = shape_MNK;
  int m_tiles = ceil_div(M, size<0>(cta_tiler));
  int n_tiles = ceil_div(N, size<1>(cta_tiler));

  // Full gmem tensors (once)
  Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));
  Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);

  // Smem tensors (once)
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});

  int k_tile_count = ceil_div(K, size<2>(cta_tiler));  // same for all tiles

  // Pipeline init (once, all 384 threads)
  MainloopPipeline pipeline(smem.pipeline, pipeline_params, ...);
  __syncthreads();

  // Pipeline states (once, advanced between tiles — NOT reset)
  auto smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
  typename MainloopPipeline::PipelineState smem_pipe_read;
  typename MainloopPipeline::PipelineState smem_pipe_release;

  uint64_t linear_idx = blockIdx.x;
  uint64_t grid_size  = gridDim.x;

  if (warp_group_idx == 2) {
    // ================================================================
    // Producer warp group
    // ================================================================
    cutlass::arch::warpgroup_reg_dealloc<40>();

    if (warp_group_thread_idx == 0) {
      using BarrierType = typename MainloopPipeline::ProducerBarrierType;

      while (linear_idx < total_tiles) {
        int m_idx = linear_idx / n_tiles;
        int n_idx = linear_idx % n_tiles;
        if (m_idx >= m_tiles) break;

        auto cta_coord = make_coord(m_idx, n_idx, _);
        Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
        Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
        auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                           group_modes<0,2>(sA), group_modes<0,2>(gA));
        auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                           group_modes<0,2>(sB), group_modes<0,2>(gB));

        CUTE_NO_UNROLL
        for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
          pipeline.producer_acquire(smem_pipe_write);
          BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
          copy(tma_a.with(*tma_barrier), tAgA(_,k_tile), tAsA(_,smem_pipe_write.index()));
          copy(tma_b.with(*tma_barrier), tBgB(_,k_tile), tBsB(_,smem_pipe_write.index()));
          ++smem_pipe_write;
        }

        // Do NOT call .advance() — ++smem_pipe_write in the loop already advanced by k_tile_count
        linear_idx += grid_size;
      }

      // Drain once after loop
      pipeline.producer_tail(smem_pipe_write);
    }

    cute::tma_store_wait<0>();

  } else {
    // ================================================================
    // Consumer warp groups (wg 0 and 1)
    // ================================================================
    cutlass::arch::warpgroup_reg_alloc<232>();

    // Consumer setup (once — thread mapping doesn't change)
    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));
    Tensor tCrC = thr_mma.make_fragment_C(thr_mma.partition_C(
        make_tensor(make_gmem_ptr(C), make_shape(M, N), dC)));  // shape only, no tile
    // Note: tCrC layout is the same regardless of which tile — only the gmem data differs

    // S2R copy setup (once)
    TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
    ThrCopy thr_s2r_a = s2r_copy_a.get_slice(threadIdx.x);
    Tensor tXsA = thr_s2r_a.partition_S(sA);
    Tensor tXrA = thr_s2r_a.retile_D(tCrA);

    TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
    ThrCopy thr_s2r_b = s2r_copy_b.get_slice(threadIdx.x);
    Tensor tXsB = thr_s2r_b.partition_S(sB);
    Tensor tXrB = thr_s2r_b.retile_D(tCrB);

    // R2S copy setup (once)
    ThrCopy thr_r2s = r2s_copy.get_slice(threadIdx.x);
    Tensor sC = make_tensor(
        make_smem_ptr(reinterpret_cast<bf16_t*>(smem.A.begin())),
        SmemLayoutC{});
    Tensor tRS_sC = thr_r2s.partition_D(sC);

    // TMA store setup (once — covers full (M,N), indexed by linear_idx)
    auto cta_tile_mn = product_each(shape(SmemLayoutC{}));
    Tensor mC_tma = tma_store_c.get_tma_tensor(make_shape(M, N));
    Tensor gC_tma_full = flat_divide(mC_tma, cta_tile_mn);
    auto cta_tma_store = tma_store_c.get_slice(Int<0>{});
    Tensor tSsC = group_modes<1, rank(cta_tma_store.partition_S(sC))>(cta_tma_store.partition_S(sC));
    Tensor tSgC = group_modes<1, rank(cta_tma_store.partition_D(gC_tma_full))>(cta_tma_store.partition_D(gC_tma_full));

    auto K_BLOCK_MAX = size<2>(tCrA);

    while (linear_idx < total_tiles) {
      int m_idx = linear_idx / n_tiles;
      int n_idx = linear_idx % n_tiles;
      if (m_idx >= m_tiles) break;

      // Per-tile: gmem views and beta-load partition
      auto cta_coord = make_coord(m_idx, n_idx, _);
      Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});
      Tensor tCgC = thr_mma.partition_C(gC);

      // Clear accumulators
      clear(tCrC);

      // Consumer main loop (same structure as 07, using advanced pipeline states)
      CUTE_NO_UNROLL
      for (int k_tile_iter = 0; k_tile_iter < k_tile_count; ++k_tile_iter) {
        pipeline.consumer_wait(smem_pipe_read);
        Tensor tXsA_p = tXsA(_,_,_,smem_pipe_read.index());
        Tensor tXsB_p = tXsB(_,_,_,smem_pipe_read.index());

        // Prefetch k_block 0
        copy(s2r_atom_a, tXsA_p(_,_,Int<0>{}), tXrA(_,_,Int<0>{}));
        copy(s2r_atom_b, tXsB_p(_,_,Int<0>{}), tXrB(_,_,Int<0>{}));

        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
          auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;
          if (k_block < K_BLOCK_MAX - 1) {
            copy(s2r_atom_a, tXsA_p(_,_,k_block_next), tXrA(_,_,k_block_next));
            copy(s2r_atom_b, tXsB_p(_,_,k_block_next), tXrB(_,_,k_block_next));
          }
          gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
        }

        pipeline.consumer_release(smem_pipe_release);
        ++smem_pipe_read;
        ++smem_pipe_release;
      }

      // Epilogue: alpha/beta scaling
      CUTE_UNROLL
      for (int i = 0; i < size(tCrC); ++i) {
        tCrC(i) = alpha * tCrC(i) + beta * static_cast<float>(tCgC(i));
      }

      // Convert F32 -> BF16, STSM write to smem
      Tensor tCrC_bf16 = make_tensor<bf16_t>(tCrC.layout());
      CUTE_UNROLL
      for (int i = 0; i < size(tCrC); ++i) {
        tCrC_bf16(i) = static_cast<bf16_t>(tCrC(i));
      }
      Tensor tRS_rAcc = thr_r2s.retile_S(tCrC_bf16);
      copy(r2s_copy, tRS_rAcc, tRS_sC);

      // Consumer-only sync
      cutlass::arch::NamedBarrier consumer_sync(256, 6);
      consumer_sync.sync();

      // TMA store
      if (threadIdx.x == 0) {
        tma_store_fence();
        copy(tma_store_c, tSsC(_, 0), tSgC(_, linear_idx));
        tma_store_arrive();
      }
      tma_store_wait<0>();  // inter-tile sync

      // Do NOT call .advance() — ++smem_pipe_read/++smem_pipe_release in the loop already advanced
      linear_idx += grid_size;
    }
  }
}
```

## Host Pseudocode

```cpp
void bf16_gemm_persistent(int m, int n, int k, ...) {
  // Same setup as 07: prob_shape, strides, cta_tiler, smem layouts, TMA descriptors, etc.

  int num_SMs;
  cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, device);

  int m_tiles = ceil_div(M, bM);
  int n_tiles = ceil_div(N, bN);
  int total_tiles = m_tiles * n_tiles;

  dim3 dimBlock(384);
  dim3 dimCluster(1, 1, 1);
  dim3 dimGrid(min(num_SMs, total_tiles));  // don't launch more CTAs than tiles

  // Same smem_size, kernel attributes as 07

  cutlass::launch_kernel_on_cluster(params, kernel_ptr,
      prob_shape, cta_tiler,
      A, tmaA, s2r_atom_a,
      B, tmaB, s2r_atom_b,
      C, sC_layout, tma_store_c, r2s_copy, dC, mma,
      alpha, beta, total_tiles);
}
```

## Testing

- Verify correctness at 1024³ with BF16 C (tolerance 0.5f, same as 07)
- Benchmark at 256³, 512³, 1024³, 2048³, 4096³, 8192³
- Compare with 07 (non-persistent) to show cross-over point
