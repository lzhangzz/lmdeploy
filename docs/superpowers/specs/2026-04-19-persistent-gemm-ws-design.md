# BF16 GEMM SM80 — Persistent Warp-Specialized TMA Kernel

## Overview

Add a persistent variant of `07_bf16_gemm_sm80_pipe_tma_ws.cu` using CUTLASS's stride-based work distribution. Each CTA processes multiple output tiles in a loop, striding by the grid size between tiles. This eliminates kernel launch overhead for small GEMMs and improves occupancy for problems that don't saturate the GPU.

New file: `08_bf16_gemm_sm80_pipe_tma_ws_persistent.cu`.

## What Changes

### Kernel Structure

The kernel gains an outer `while` loop wrapping the existing producer and consumer branches:

```
pipeline init + __syncthreads()  // once, before the loop

while (linear_idx < total_tiles) {
    m_idx = linear_idx / n_tiles
    n_idx = linear_idx % n_tiles

    // compute tile coordinates from m_idx, n_idx
    // re-partition gmem tensors for this tile

    if (warp_group_idx == 2) {
        // Producer: TMA loads (fresh PipelineState per tile)
    } else {
        // Consumer: LDSM + MMA + epilogue (fresh PipelineState per tile)
    }

    linear_idx += grid_size  // all threads stride together
}
```

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
- Fresh `PipelineState` objects created per tile at the top of the while loop. The underlying barriers don't need re-initialization — the producer/consumer acquire/release pattern returns them to the initial state after all stages are consumed.
- `tma_store_wait<0>()` at the end of each tile's epilogue acts as the inter-tile sync: ensures the TMA store completes before smem is overwritten for the next tile.

### Device Kernel Changes (from 07)

1. **New parameter:** `int total_tiles` — total number of output tiles. Computed on host as `ceil_div(M, bM) * ceil_div(N, bN)`.

2. **Grid is 1D:** The kernel uses `blockIdx.x` directly as the linear tile index. Grid shape is `dim3(num_SMs)`.

3. **Persistent while loop** wraps producer and consumer branches:
   - At loop top: compute `m_idx, n_idx` from `linear_idx` via divmod. Exit if `m_idx >= m_tiles`.
   - TMA partitioning uses `m_idx, n_idx` instead of `blockIdx.x, blockIdx.y`.
   - Producer: fresh `make_producer_start_state()` per tile.
   - Consumer: fresh `PipelineState` per tile, clear accumulators per tile.
   - After epilogue TMA store + `tma_store_wait<0>()`: advance `linear_idx += gridDim.x`.

4. **gC/gA/gB tensors recomputed per tile:** The `local_tile` calls change from `make_coord(blockIdx.x, blockIdx.y, _)` to `make_coord(m_idx, n_idx, _)`. Similarly, the TMA store `rest_idx` changes from `blockIdx.x + blockIdx.y * gridDim.x` to `linear_idx`.

5. **Accumulator clearing:** `clear(tCrC)` moves from before the k-tile loop to the top of the while loop (before each tile's computation).

### Host Function Changes

1. **Query SM count:** `cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, device)`.

2. **Grid shape:** `dim3 dimGrid(num_SMs)` — 1D grid, one block per SM.

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

  // Pipeline init (once, all 384 threads)
  MainloopPipeline pipeline(smem.pipeline, pipeline_params, ...);
  __syncthreads();

  int n_tiles = ceil_div(N, bN);
  uint64_t linear_idx = blockIdx.x;
  uint64_t grid_size  = gridDim.x;

  // Consumer-only setup (once, outside loop)
  ThrMMA thr_mma, s2r copies, r2s copy — setup using threadIdx.x

  // Persistent loop
  while (linear_idx < total_tiles) {
    int m_idx = linear_idx / n_tiles;
    int n_idx = linear_idx % n_tiles;

    if (m_idx >= m_tiles) break;  // invalid tile

    // Partition gmem for this tile (m_idx, n_idx)
    auto cta_coord = make_coord(m_idx, n_idx, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

    // TMA partitioning
    auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                       group_modes<0,2>(sA), group_modes<0,2>(gA));
    auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                       group_modes<0,2>(sB), group_modes<0,2>(gB));

    int k_tile_count = size<1>(tAgA);

    if (warp_group_idx == 2) {
      // Producer — fresh PipelineState per tile
      if (warp_group_thread_idx == 0) {
        auto smem_pipe_write = make_producer_start_state<MainloopPipeline>();
        for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
          pipeline.producer_acquire(smem_pipe_write);
          BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
          copy(tma_a.with(*tma_barrier), tAgA(_,k_tile), tAsA(_,smem_pipe_write.index()));
          copy(tma_b.with(*tma_barrier), tBgB(_,k_tile), tBsB(_,smem_pipe_write.index()));
          ++smem_pipe_write;
        }
        pipeline.producer_tail(smem_pipe_write);
      }
      cute::tma_store_wait<0>();
    } else {
      // Consumer — fresh PipelineState per tile, clear accumulators
      clear(tCrC);

      PipelineState smem_pipe_read, smem_pipe_release;
      for (int k_tile_iter = 0; k_tile_iter < k_tile_count; ++k_tile_iter) {
        pipeline.consumer_wait(smem_pipe_read);
        // LDSM + MMA loop (same as 07)
        pipeline.consumer_release(smem_pipe_release);
        ++smem_pipe_read;
        ++smem_pipe_release;
      }

      // Epilogue (same as 07): scale, STSM, NamedBarrier, TMA store
      // rest_idx = linear_idx (instead of blockIdx.x + blockIdx.y * gridDim.x)

      if (threadIdx.x == 0) {
        tma_store_fence();
        copy(tma_store_c, tSsC(_, 0), tSgC(_, linear_idx));
        tma_store_arrive();
      }
      tma_store_wait<0>();  // inter-tile sync
    }

    linear_idx += grid_size;
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
