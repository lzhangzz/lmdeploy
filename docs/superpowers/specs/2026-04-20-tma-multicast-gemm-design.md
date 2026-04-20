# TMA Multicast GEMM — Sample 10

Add TMA multicast support to the cute-reference GEMM sample series as a new sample 10, parameterized over cluster shape so it works with any 2D cluster configuration (e.g., 2x1, 1x2, 2x2).

## Context

Sample 09 (`09_bf16_gemm_sm90_pipe_tma_ws_persistent.cu`) is a persistent warp-specialized GEMM using SM90 WGMMA + TMA load/store with `PipelineTmaAsync`. It uses a single-CTA cluster (`dimCluster(1,1,1)`), so every TMA load is unicast — each CTA independently loads its own A and B tiles.

TMA multicast allows one TMA load to broadcast data to multiple CTAs in a thread block cluster simultaneously. When CTAs share an operand tile (e.g., same B tile when CTAs are stacked along M), this halves the TMA bandwidth for that operand.

## Problem Shape

Same as sample 09: `C = alpha * A * B^T + beta * C` where A is (M,K) row-major BF16, B is (N,K) K-contiguous BF16, C is (M,N) column-major BF16.

## Cluster Configuration

A compile-time `ClusterShape` parameter: `Shape<Int<CM>, Int<CN>, _1>` where CM and CN are the number of CTAs along M and N dimensions of the grid. Default: `Shape<_2, _1, _1>` (2x1 cluster).

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

Computed on host (and identically on device):

```cpp
auto cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                        make_tile(typename decltype(mma)::AtomThrID{}));
```

### TMA Atom Creation

Multicast atoms receive an extra parameter — the number of CTAs along the multicast dimension:

```cpp
Copy_Atom tmaA = make_tma_atom(GmemTiledCopyA{}, tA, sA(_,_,0),
                               make_shape(bM, bK), size<2>(cluster_layout_vmnk));
Copy_Atom tmaB = make_tma_atom(GmemTiledCopyB{}, tB, sB(_,_,0),
                               make_shape(bN, bK), size<1>(cluster_layout_vmnk));
```

For non-multicast operands, the extra parameter is 1 (harmless for `SM90_TMA_LOAD`).

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

Shared memory layout is unchanged — each CTA has its own smem buffers. TMA multicast writes the same data to each CTA's smem independently.

## Device-Side Changes

### Cluster Coordinates and Multicast Masks

```cpp
auto cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                        make_tile(typename TiledMma::AtomThrID{}));
auto cta_coord = cluster_layout_vmnk.get_flat_coord(int(block_rank_in_cluster()));

uint16_t tma_mcast_mask_a = multicast_A ? create_tma_multicast_mask<2>(cluster_layout_vmnk, cta_coord) : 0;
uint16_t tma_mcast_mask_b = multicast_B ? create_tma_multicast_mask<1>(cluster_layout_vmnk, cta_coord) : 0;
```

### Cluster-Aware TMA Partitioning

```cpp
auto [tAgA, tAsA] = tma_partition(tma_a, get<2>(cta_coord),
                                  make_layout(size<2>(cluster_layout_vmnk)),
                                  group_modes<0,2>(sA), group_modes<0,2>(gA));
auto [tBgB, tBsB] = tma_partition(tma_b, get<1>(cta_coord),
                                  make_layout(size<1>(cluster_layout_vmnk)),
                                  group_modes<0,2>(sB), group_modes<0,2>(gB));
```

For non-multicast operands, the coordinate is 0 and the layout is `_1`, identical to sample 09.

### Pipeline

Pass `cluster_shape` to `PipelineTmaAsync` constructor (already supported). Add `cluster_sync()` after pipeline initialization.

### Persistent Work Distribution

Iterate at cluster granularity. Each CTA derives its own tile coordinates from its intra-cluster rank:

```cpp
int my_rank_m = get<1>(cta_coord);
int my_rank_n = get<2>(cta_coord);
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

### Producer (TMA Loads)

- **Multicast operand**: Only the CTA with coordinate 0 along the multicast dimension issues the TMA load with the multicast mask: `copy(tma.with(barrier, mask), gB, sB)`. All other CTAs in the multicast group skip the TMA issue — the hardware broadcasts the data to their smem and signals their barriers automatically. This is gated by checking `get<1>(cta_coord) == 0` for B multicast or `get<2>(cta_coord) == 0` for A multicast.
- **Non-multicast operand**: Every CTA independently loads: `copy(tma.with(barrier), gA, sA)`.

### Consumer (WGMMA)

Unchanged from sample 09. Each CTA independently runs WGMMA over the pipeline stages. The consumer doesn't know or care whether the data arrived via multicast or unicast.

### Epilogue

Unchanged. Per-CTA TMA store (no multicast store exists on SM90).

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
