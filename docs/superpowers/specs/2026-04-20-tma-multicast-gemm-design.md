# TMA Multicast GEMM — Sample 10

Add TMA multicast support to the cute-reference GEMM sample series as a new sample 10, parameterized over cluster shape so it works with any 2D cluster configuration (e.g., 2x1, 1x2, 2x2).

## Context

Sample 09 (`09_bf16_gemm_sm90_pipe_tma_ws_persistent.cu`) is a persistent warp-specialized GEMM using SM90 WGMMA + TMA load/store with `PipelineTmaAsync`. It uses a single-CTA cluster (`dimCluster(1,1,1)`), so every TMA load is unicast — each CTA independently loads its own A and B tiles.

TMA multicast allows one TMA load to broadcast data to multiple CTAs in a thread block cluster simultaneously. In multicast mode, each CTA loads a *different portion* of the shared tile and broadcasts it to the cluster. The combination of all CTAs' cooperative loads fills each CTA's smem completely, while the total TMA bandwidth for the shared operand is reduced.

## Problem Shape

Same as sample 09: `C = alpha * A * B^T + beta * C` where A is (M,K) row-major BF16, B is (N,K) K-contiguous BF16, C is (M,N) column-major BF16.

## SM90 vs SM100 Cluster Layout

**Critical:** SM90 and SM100 compute the cluster layout differently.

SM100 (Blackwell) uses `tiled_divide(make_layout(cluster_shape), make_tile(AtomThrID{}))` because SM100 MMA atoms are CTA-level (`AtomThrID = Layout<_1>`, size 1). This produces a 4-mode (V,M,N,K) layout.

SM90 (Hopper) WGMMA atoms are CTA-internal (`AtomThrID = Layout<_128>`, 128 threads), so `tiled_divide` with AtomThrID makes no sense. Instead, SM90 uses a simple 3-mode `make_layout(ClusterShape{})` producing layout (M,N,K).

**This spec follows the SM90 pattern** (matching CUTLASS SM90 collectives like `sm90_mma_tma_gmma_ss_warpspecialized.hpp` and `sm90_sparse_mma_tma_gmma_ss_warpspecialized.hpp`).

Mode numbering for the 3-mode layout:
- Mode 0 = M dimension
- Mode 1 = N dimension
- Mode 2 = K dimension (always 1 for GEMM)

## Cluster Configuration

A compile-time `ClusterShape` parameter: `Shape<Int<CM>, Int<CN>, _1>` where CM and CN are the number of CTAs along M and N dimensions. Default: `Shape<_2, _1, _1>` (2x1 cluster).

Two constexpr booleans derived at compile time:

```cpp
constexpr bool multicast_A = (size<1>(ClusterShape{}) > 1);  // CTAs along N share A
constexpr bool multicast_B = (size<0>(ClusterShape{}) > 1);  // CTAs along M share B
```

TMA atom types selected via `std::conditional`:

```cpp
using GmemTiledCopyA = conditional_t<multicast_A, SM90_TMA_LOAD_MULTICAST, SM90_TMA_LOAD>;
using GmemTiledCopyB = conditional_t<multicast_B, SM90_TMA_LOAD_MULTICAST, SM90_TMA_LOAD>;
```

## Host-Side Changes

### Cluster Layout

SM90 uses `make_layout(ClusterShape{})` directly (not `tiled_divide` with AtomThrID):

```cpp
auto cluster_layout = make_layout(cluster_shape);  // Layout<Shape<CM,CN,_1>>
```

### TMA Atom Creation

`make_tma_atom` accepts an optional `cluster_size` parameter (defaults to `Int<1>{}`). For multicast operands, pass the number of CTAs along the multicast dimension:

```cpp
Copy_Atom tmaA = make_tma_atom(GmemTiledCopyA{}, tA, sA(_,_,0),
                               make_shape(bM, bK), size<1>(cluster_shape));
Copy_Atom tmaB = make_tma_atom(GmemTiledCopyB{}, tB, sB(_,_,0),
                               make_shape(bN, bK), size<0>(cluster_shape));
```

- A multicast count = `size<1>(cluster_shape)` = cluster_N. For a 2x1 cluster: `_1` (no multicast).
- B multicast count = `size<0>(cluster_shape)` = cluster_M. For a 2x1 cluster: `_2` (multicast to 2 CTAs).

The `cluster_size` parameter causes `make_tma_atom` to truncate the TMA SMEM box shape by the multicast factor. Each CTA will load 1/N of the tile via `tma_partition`, and the multicast distributes it to all CTAs.

### Grid Sizing

Persistent grid covers clusters, not individual CTAs:

```cpp
int cluster_size = size<0>(cluster_shape) * size<1>(cluster_shape);
int cluster_m_tiles = m_tiles / size<0>(cluster_shape);
int cluster_n_tiles = n_tiles / size<1>(cluster_shape);
int total_cluster_tiles = cluster_m_tiles * cluster_n_tiles;
dim3 dimGrid(std::min(num_SMs / cluster_size, total_cluster_tiles));
dim3 dimCluster(size<0>(cluster_shape), size<1>(cluster_shape), 1);
```

Shared memory layout is unchanged — each CTA has its own smem buffers.

## Device-Side Changes

### Cluster Coordinates and Multicast Masks

SM90 uses `make_layout(ClusterShape{})` directly with `block_rank_in_cluster()`:

```cpp
auto cluster_layout = make_layout(cluster_shape);
auto cta_coord = cluster_layout.get_flat_coord(int(block_rank_in_cluster()));
```

For a 2x1 cluster (`Shape<_2,_1,_1>`):
- CTA 0: `block_rank_in_cluster() = 0` → `cta_coord = (0, 0, 0)`
- CTA 1: `block_rank_in_cluster() = 1` → `cta_coord = (1, 0, 0)`

Multicast masks use `create_tma_multicast_mask` with the 3-mode layout. Mode 0 = M, mode 1 = N:

```cpp
uint16_t tma_mcast_mask_a = multicast_A
    ? create_tma_multicast_mask<1>(cluster_layout, cta_coord) : 0;  // along N (mode 1)
uint16_t tma_mcast_mask_b = multicast_B
    ? create_tma_multicast_mask<0>(cluster_layout, cta_coord) : 0;  // along M (mode 0)
```

For a 2x1 cluster: `tma_mcast_mask_a = 0x0001` (self only, no A multicast), `tma_mcast_mask_b = 0x0003` (both CTAs participate in B multicast).

### Cluster-Aware TMA Partitioning

`tma_partition` takes the CTA's coordinate along the multicast dimension and the cluster sub-layout for that dimension. With the 3-mode layout:

```cpp
// A: multicast along N (mode 1). For 2x1 cluster, coord=0 and layout=_1 — same as non-multicast.
auto [tAgA, tAsA] = tma_partition(tma_a, get<1>(cta_coord),
                                  make_layout(size<1>(cluster_layout)),
                                  group_modes<0,2>(sA), group_modes<0,2>(gA));

// B: multicast along M (mode 0). Each CTA loads a different portion of the B tile.
auto [tBgB, tBsB] = tma_partition(tma_b, get<0>(cta_coord),
                                  make_layout(size<0>(cluster_layout)),
                                  group_modes<0,2>(sB), group_modes<0,2>(gB));
```

For non-multicast operands, the coordinate is 0 and the layout is `_1` — equivalent to sample 09's 3-arg `tma_partition`.

### Pipeline

Pass `cluster_shape` to `PipelineTmaAsync` constructor (already supported). Use `cluster_sync()` / `cluster_arrive_relaxed()` + `cluster_wait()` instead of `__syncthreads()` for pipeline initialization:

```cpp
MainloopPipeline pipeline(smem.pipeline, pipeline_params, cluster_shape);
if constexpr (size(cluster_shape) > 1) {
    cluster_arrive_relaxed();
    cluster_wait();
} else {
    __syncthreads();
}
```

### Persistent Work Distribution

Iterate at cluster granularity. Each CTA derives its own tile coordinates from its intra-cluster rank:

```cpp
int my_rank_m = get<0>(cta_coord);  // M-rank within cluster (mode 0)
int my_rank_n = get<1>(cta_coord);  // N-rank within cluster (mode 1)
int num_clusters = gridDim.x / cluster_size;

while (cluster_linear_idx < total_cluster_tiles) {
    int cluster_m = cluster_linear_idx / cluster_n_tiles;
    int cluster_n = cluster_linear_idx % cluster_n_tiles;
    int m_idx = cluster_m * size<0>(cluster_shape) + my_rank_m;
    int n_idx = cluster_n * size<1>(cluster_shape) + my_rank_n;
    // ... process tile (m_idx, n_idx)
    cluster_linear_idx += num_clusters;
}
```

### Producer (TMA Loads) — Cooperative Multicast

**Every CTA issues its own TMA loads independently.** For multicast operands, each CTA loads a different portion of the shared tile (offset via `tma_partition`) and broadcasts it to the cluster via the multicast mask. The combination of all CTAs' loads fills each CTA's smem completely.

```cpp
// All CTAs issue both A and B loads (coordinates differ per CTA via tma_partition)
pipeline.producer_acquire(smem_pipe_write);
BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
copy(tma_a.with(*tma_barrier, tma_mcast_mask_a), tAgA(_,k_tile_next), tAsA(_,smem_pipe_write.index()));
copy(tma_b.with(*tma_barrier, tma_mcast_mask_b), tBgB(_,k_tile_next), tBsB(_,smem_pipe_write.index()));
```

For non-multicast operands, the mask is 0 (or the `SM90_TMA_LOAD` variant ignores it). For multicast operands, the mask broadcasts each CTA's portion to the full multicast group.

### Consumer (WGMMA)

Unchanged from sample 09. Each CTA independently runs WGMMA over the pipeline stages. The consumer doesn't know or care whether the data arrived via multicast or unicast.

### Epilogue

Unchanged. Per-CTA TMA store (no multicast store exists on SM90).

### Cluster Synchronization After Mainloop

After the persistent loop ends, add a cluster-wide sync to ensure all CTAs finish before any CTA exits:

```cpp
if constexpr (size(cluster_shape) > 1) {
    cluster_arrive();
    cluster_wait();
}
```

## What Does NOT Change

- Tile size: 128x256x64 per CTA (same as sample 09)
- Thread count: 384 (256 MMA + 128 producer)
- Smem layout per CTA (GMMA swizzled layouts for A and B)
- WGMMA atom (`SM90_64x256x16_F32BF16BF16_SS`)
- R2S STSM epilogue pattern
- Pipeline depth (3 stages)
- Problem shape and correctness verification

## File

New file: `cute-reference/samples/10_bf16_gemm_sm90_pipe_tma_ws_persistent_mcast.cu`

## Verification

- Correctness: Same CPU reference check as sample 09 (1024^3)
- Benchmark: Same sizes (256^3 through 8192^3)
- Additional: Print cluster shape in the banner line
