# Separate Epilogue Smem for Tile-Level Pipelining

## Overview

Add a dedicated shared memory buffer for the epilogue C output tile so the producer warp group can load tile N+1's A/B data while the consumer warp groups process tile N's epilogue. This eliminates the smem overlap race condition and enables true tile-level pipelining in the persistent kernel.

Modifies: `08_bf16_gemm_sm80_pipe_tma_ws_persistent.cu`.

## Problem

The current persistent kernel (08) reuses `smem.A` as scratch space for the epilogue C output. `sC` is constructed as:

```cpp
Tensor sC = make_tensor(make_smem_ptr(reinterpret_cast<bf16_t*>(smem.A.begin())), SmemLayoutC{});
```

This creates a race: after the consumer releases all pipeline stages in the k_tile loop, the producer can start loading tile N+1's A data into the same smem that the consumer's epilogue STSM writes C to. No synchronization prevents this overlap.

### Race detail

`smem.A` has 3 pipeline stages of 16384 bf16 elements each (32768 bytes per stage). `sC` overlays `smem.A.begin()` with 32768 bf16 elements (65536 bytes) — covering stages 0 and 1. After the consumer releases all stages via `consumer_release` during the k_tile loop, the producer's `producer_acquire` can succeed for stage 0 or 1. The producer's TMA load writes to that stage while the consumer's STSM writes C to the same addresses. The race is masked in practice by TMA load latency (~100-200 cycles) being longer than the STSM+epilogue path, but it is a latent correctness bug that could manifest with small `k_tile_count`.

### CUTLASS comparison

CUTLASS's persistent kernels (`sm90_gemm_tma_warpspecialized_cooperative.hpp`) use separate `MainloopTensorStorage` and `EpilogueTensorStorage` in `SharedStorage` (no union). The non-persistent kernel uses a union because mainloop and epilogue don't overlap in time. Persistent kernels MUST use separate smem because the producer loads tile N+1 concurrently with the consumer's epilogue for tile N.

## Solution

Allocate a dedicated `C` buffer in `SharedStorage`. The epilogue writes to this buffer instead of overlaying on `smem.A`. The A/B pipeline smem and the C epilogue smem are disjoint, so the producer can safely load ahead while the consumer does the epilogue.

## Changes

### 1. SharedStorage

Add a `C` member after `A` and `B`:

```cpp
template <class ElementA, class ElementB, class ElementC,
          class SmemLayoutA, class SmemLayoutB, class SmemLayoutC, int Stages>
struct SharedStorage
{
  alignas(128) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
  alignas(128) cute::ArrayEngine<ElementC, cute::cosize_v<SmemLayoutC>> C;  // NEW
  typename cutlass::PipelineTmaAsync<Stages>::SharedStorage pipeline;
};
```

New template parameter `ElementC` (bf16_t) and `SmemLayoutC` for the C array engine size.

**Smem budget:**

| Component | Elements | Bytes |
|-----------|----------|-------|
| A (256×64×3, GMMA swizzled) | 49,152 bf16 | 98,304 B |
| B (128×64×3, GMMA swizzled) | 24,576 bf16 | 49,152 B |
| C (256×128, plain col-major) | 32,768 bf16 | 65,536 B |
| Pipeline barriers (3×FullBarrier + 3×EmptyBarrier) | — | 48 B |
| **Total** | | **213,040 B (~208 KB)** |
| L20Y/H800 shared memory per SM | | **228 KB max** |
| Headroom | | **~15 KB** |

No inter-member padding needed: A (98304 B) is 128-byte aligned, B follows naturally, C follows naturally, pipeline barriers are 8-byte aligned.

### 2. sC Tensor Construction

Change from overlaying on `smem.A`:

```cpp
Tensor sC = make_tensor(make_smem_ptr(reinterpret_cast<bf16_t*>(smem.A.begin())), SmemLayoutC{});
```

To using the dedicated C buffer:

```cpp
Tensor sC = make_smem_ptr(smem.C.begin()), SmemLayoutC{});
```

No `reinterpret_cast` needed since `smem.C` is already typed as `ElementC` (bf16_t).

### 3. SharedStorage Type Instantiation (kernel)

The `using SharedStorage = ...` line in the kernel must include the new template parameters:

```cpp
using SharedStorage = SharedStorage<TA, TB, TC, decltype(sA), decltype(sB), SmemLayoutC, cute::size<2>(SmemLayoutA{})>;
```

Note: `SmemLayoutC` and `TC` are already template parameters of the kernel. `TC` is `bf16_t`.

### 4. Host Function SharedStorage Instantiation

The host function computes smem_size using the same SharedStorage type. Update to match the new template parameters:

```cpp
int smem_size = int(sizeof(SharedStorage<bf16_t, bf16_t, bf16_t,
    decltype(sA), decltype(sB), decltype(sC_layout), cute::size<2>(decltype(sA){})>));
```

## What Stays the Same

- Pipeline init, states, advancement — unchanged
- Producer loop structure — unchanged (already loads ahead)
- Consumer MMA loop — unchanged
- Epilogue logic (alpha/beta, STSM, NamedBarrier, TMA store) — unchanged (just different sC base pointer)
- Host function logic (TMA descriptors, grid, launch) — unchanged
- Benchmark and correctness test — unchanged
- No new kernel parameters
- No new synchronization mechanisms needed

## Why No Additional Sync Is Needed

**A/B pipeline sync:** The pipeline's `producer_acquire`/`consumer_release` handles A/B smem synchronization. The producer blocks when all 3 pipeline stages are full. With separate C smem, the consumer's epilogue doesn't conflict with the producer's loads.

**TMA store/load independence:** TMA loads (cp.async.bulk.tensor...mbarrier) and TMA stores (cp.async.bulk.tensor...bulk_group) use different hardware paths and completion mechanisms. They can operate simultaneously on different smem regions without interference. CUTLASS's persistent kernels rely on this.

**C smem reuse between tiles:** The consumer's `tma_store_wait<0>()` at the end of each tile ensures the TMA store from C smem completes before C smem is overwritten for the next tile's epilogue. This acts as the inter-tile sync for the C buffer.

**CTA exit:** The producer's `tma_store_wait<0>()` after the while loop (line 190) ensures the last tile's TMA store completes before the CTA exits and smem is freed.

## Alignment and Safety

- The C buffer uses `alignas(128)`, satisfying STSM's `stmatrix.sync.aligned` 128-byte alignment requirement.
- NamedBarrier(256, 6) for epilogue sync uses ID 6, which does not conflict with pipeline barriers (ClusterTransactionBarrier objects in smem, not NamedBarrier IDs).
- `tma_store_fence()` before the TMA store (already present) ensures STSM writes are visible to the TMA unit.

## Testing

- Correctness at 1024^3 (same tolerance 0.5f)
- Full benchmark sweep: 256^3, 512^3, 1024^3, 2048^3, 4096^3, 8192^3
- Compare with previous 08 results to verify no regression
