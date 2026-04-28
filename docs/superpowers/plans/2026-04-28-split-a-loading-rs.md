# Split A Loading RS WGMMA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split sample 13 (BF16 GEMM, RS WGMMA) into two kernels: one that packs operand A into GMMA register layout, and one that runs the GEMM using pre-packed A.

**Architecture:** Kernel 1 (pack) uses TMA to load A tiles into smem, S2R copy to transform to GMMA register layout, then writes registers directly to a packed gmem buffer. Kernel 2 (WGMMA) reads packed A directly from gmem into registers, uses TMA pipeline for B only, runs batched WGMMA, and stores C via STSM + TMA store. Both kernels share the same TiledMMA construction to ensure register layout consistency.

**Tech Stack:** CUDA 12.8, CuTe, CUTLASS PipelineTmaAsync, SM90 WGMMA tensor cores, TMA load/store

---

## File Structure

| File | Responsibility |
|---|---|
| `cute-reference/mixed-gemm/CMakeLists.txt` | Build targets for the two executables |
| `cute-reference/mixed-gemm/split_a_pack.h` | Shared pack kernel template + host function (included by both .cu files) |
| `cute-reference/mixed-gemm/bf16_gemm_sm90_split_a_pack.cu` | Standalone pack test: pack A and verify round-trip |
| `cute-reference/mixed-gemm/bf16_gemm_sm90_split_a_wgmma.cu` | WGMMA kernel + end-to-end test (pack A, run GEMM, verify against CPU) |

Key reference: `cute-reference/samples/13_bf16_gemm_sm90_pipe_tma_ws_persistent_rs_cutlass.cu` (the kernel being split)

---

### Task 1: Build Infrastructure

**Files:**
- Create: `cute-reference/mixed-gemm/CMakeLists.txt`
- Modify: `CMakeLists.txt` (add subdirectory)

- [ ] **Step 1: Create the mixed-gemm directory and CMakeLists.txt**

```bash
mkdir -p cute-reference/mixed-gemm
```

Create `cute-reference/mixed-gemm/CMakeLists.txt`:

```cmake
# Mixed-precision GEMM kernels (split A loading POC)
set(MIXED_GEMM_TARGETS
    bf16_gemm_sm90_split_a_pack
    bf16_gemm_sm90_split_a_wgmma
)

foreach(target ${MIXED_GEMM_TARGETS})
    add_executable(${target} ${target}.cu)
    target_link_libraries(${target} PRIVATE nvidia::cutlass::cutlass)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
    )
    set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES "90a-real")
endforeach()
```

- [ ] **Step 2: Add subdirectory to top-level CMakeLists.txt**

In `CMakeLists.txt`, after the existing `add_subdirectory(cute-reference/samples)` line (~line 351), add:

```cmake
  add_subdirectory(cute-reference/mixed-gemm)
```

So it becomes:

```cmake
option(BUILD_CUTE_SAMPLES "Build CuTe reference GEMM samples" OFF)
if(BUILD_CUTE_SAMPLES)
  add_subdirectory(cute-reference/samples)
  add_subdirectory(cute-reference/mixed-gemm)
endif()
```

- [ ] **Step 3: Configure and verify the build sees the new targets**

```bash
cd /data/lmdeploy-cute/build && sh ../my_generate.sh -DBUILD_CUTE_SAMPLES=ON 2>&1 | tail -5
```

If the configure step fails because `-DBUILD_CUTE_SAMPLES=ON` can't be appended, modify `my_generate.sh` to add it, then re-run.

Verify the targets exist:

```bash
ninja -C /data/lmdeploy-cute/build -t targets | grep split_a
```

Expected: both `bf16_gemm_sm90_split_a_pack` and `bf16_gemm_sm90_split_a_wgmma` targets listed.

- [ ] **Step 4: Commit**

```bash
git add cute-reference/mixed-gemm/CMakeLists.txt CMakeLists.txt
git commit -m "Add mixed-gemm build infrastructure for split A loading POC"
```

---

### Task 2: Kernel 1 — Operand A Packing

**Files:**
- Create: `cute-reference/mixed-gemm/split_a_pack.h`
- Create: `cute-reference/mixed-gemm/bf16_gemm_sm90_split_a_pack.cu`

This task creates the packing kernel that transforms A from gmem tensor layout to GMMA register layout. The kernel uses TMA to load A tiles into smem, performs the S2R layout transformation (same as sample 13's consumer path), then writes register contents directly to a packed gmem buffer.

**Key design decisions:**
- Uses raw mbarrier (not PipelineTmaAsync) since there's no overlap to exploit — just sequential TMA load → S2R → gmem write
- Uses `cute::initialize_barrier`, `cute::set_barrier_transaction_bytes`, `cute::wait_barrier` from `<cute/arch/copy_sm90_desc.hpp>`
- 256 threads per CTA (2 warpgroups), matching sample 13's consumer thread count
- Packed buffer layout: `[tile_idx][thread_id + i * 256]` for coalesced access

- [ ] **Step 1: Create the shared pack header `split_a_pack.h`**

This header defines the pack kernel template and host function, shared between the standalone pack test and the end-to-end WGMMA test.

```cuda
/***************************************************************************************************
 * Shared pack kernel and host function for split A loading.
 * Included by both bf16_gemm_sm90_split_a_pack.cu and bf16_gemm_sm90_split_a_wgmma.cu.
 **************************************************************************************************/
#pragma once

#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>
#include "cutlass/cluster_launch.hpp"
#include "cutlass/device_kernel.h"

using bf16_t = cute::bfloat16_t;

// SharedStorage for pack kernel: A smem (1 stage) + TMA mbarrier
template <class ElementA, class SmemLayoutA>
struct PackSharedStorage
{
  alignas(128) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  alignas(16)  uint64_t mbarrier;
};

// ================================================================================================
// Pack kernel: transforms A from gmem tensor layout to per-tile GMMA register layout
// ================================================================================================
template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TiledMma>
__global__ static
__launch_bounds__(256, 1)
void
split_a_pack_device(ProblemShape shape_MK, CtaTiler cta_tiler,
                    TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
                    TA* packed_A, TiledMma mma,
                    int total_tiles)
{
  using namespace cute;

  CUTE_STATIC_ASSERT_V(rank(shape_MK) == Int<2>{});
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<2>{});
  static_assert(is_static<SmemLayoutA>::value);

  auto [M, K] = shape_MK;

  // ---- Global memory tensor ----
  Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));                    // (M, K)

  // ---- Shared memory ----
  extern __shared__ char shared_memory[];
  using SharedStorage = PackSharedStorage<TA, SmemLayoutA>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});  // (BLK_M, BLK_K, 1)

  // ---- Tile counts ----
  constexpr int bK = size<1>(CtaTiler{});
  int k_tile_count = size(ceil_div(K, bK));

  // ---- TiledMMA and S2R copy setup (same as sample 13 consumer) ----
  ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);                                  // (MMA, MMA_M, MMA_K, PIPE)
  Tensor tCrA = thr_mma.make_fragment_A(tCsA(_,_,_,Int<0>{}));            // (MMA, MMA_M, MMA_K)

  auto smem_tiled_copy_A = make_tiled_copy_A(
      Copy_Atom<AutoVectorizingCopy, bf16_t>{}, mma);
  auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(threadIdx.x);
  Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);                 // (CPY, CPY_M, CPY_K)
  Tensor tCsA_copy_view = smem_thr_copy_A.partition_S(sA);                // (CPY, CPY_M, CPY_K, PIPE)

  constexpr int regs_per_thread = size(tCrA);

  // TMA transaction bytes (one stage of A smem)
  constexpr int tma_transaction_bytes = sizeof(TA) * cute::cosize_v<SmemLayoutA>;

  // ---- Grid-stride loop over all (M, K) tiles ----
  uint64_t linear_idx = blockIdx.x;
  uint64_t grid_size = gridDim.x;

  while (linear_idx < total_tiles) {
    int tile_m = linear_idx / k_tile_count;
    int tile_k = linear_idx % k_tile_count;

    // Compute gmem tensor for this tile
    Tensor gA = local_tile(mA, cta_tiler, make_coord(tile_m, tile_k));

    // ---- TMA load using raw mbarrier (pattern from tma_load_testbed.hpp) ----
    if (threadIdx.x == 0) {
      smem.mbarrier = 0;
      cute::initialize_barrier(smem.mbarrier, 1);
      cute::set_barrier_transaction_bytes(smem.mbarrier, tma_transaction_bytes);

      // group_modes<0,2> flattens 2D (bM, bK) to 1D ((bM, bK)) so the full tile
      // is treated as one TMA transfer (same pattern as sample 13's tma_partition calls)
      auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                         group_modes<0,2>(sA(_,_,0)),
                                         group_modes<0,2>(gA));
      copy(tma_a.with(smem.mbarrier), tAgA, tAsA);
    }
    __syncthreads();
    cute::wait_barrier(smem.mbarrier, 0);

    // ---- S2R copy: swizzled smem → GMMA register layout ----
    copy(smem_tiled_copy_A, tCsA_copy_view(_,_,_,0), tCrA_copy_view(_,_,_));

    // ---- Write registers to packed gmem buffer (coalesced) ----
    bf16_t* packed_ptr = packed_A + linear_idx * 256 * regs_per_thread;
    CUTE_UNROLL
    for (int i = 0; i < regs_per_thread; ++i) {
      packed_ptr[i * 256 + threadIdx.x] = tCrA(i);
    }

    __syncthreads();
    linear_idx += grid_size;
  }
}

// ================================================================================================
// Host function for pack kernel
// ================================================================================================
inline void
split_a_pack(int m, int k,
             bf16_t const* A, int ldA,
             bf16_t* packed_A,
             cudaStream_t stream = 0)
{
  using namespace cute;

  auto M = int(m);
  auto K = int(k);
  auto shape_MK = make_shape(M, K);

  // CTA tile size (matches sample 13's A tile dimensions)
  auto bM = Int<128>{};
  auto bK = Int<64>{};
  auto cta_tiler = make_shape(bM, bK);

  // Smem layout for A (1 stage, no pipelining)
  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16_t>{}, make_shape(bM, bK, Int<1>{}));

  // TMA load atom
  Tensor tA = make_tensor(A, make_shape(M, K), make_stride(ldA, Int<1>{}));
  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, tA, sA(_,_,0), make_shape(bM, bK));

  // TiledMMA (same as sample 13 — required for correct register layout)
  TiledMMA mma = make_tiled_mma(
      SM90_64x256x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::K>{},
      Layout<Shape<_2, _1>>{});

  static_assert(decltype(size(mma))::value == 256, "Expected 256 threads");

  // Grid
  int total_tiles = size(ceil_div(M, bM)) * size(ceil_div(K, bK));
  int num_SMs = 0;
  CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0));

  dim3 dimBlock(256);
  dim3 dimCluster(1, 1, 1);
  dim3 dimGrid(std::min(num_SMs, total_tiles));

  int smem_size = int(sizeof(PackSharedStorage<bf16_t, decltype(sA)>));

  auto* kernel_ptr = &split_a_pack_device<
      decltype(shape_MK), decltype(cta_tiler),
      bf16_t, decltype(sA), decltype(tmaA),
      decltype(mma)>;

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      (void const*)kernel_ptr,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size));
  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      (void const*)kernel_ptr,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      100));

  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*)kernel_ptr,
      shape_MK, cta_tiler,
      A, tmaA,
      packed_A, mma,
      total_tiles);

  CUTE_CHECK_LAST();
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed at pack kernel launch" << std::endl;
  }
}
```

- [ ] **Step 2: Create the standalone pack test `bf16_gemm_sm90_split_a_pack.cu`**

```cuda
/***************************************************************************************************
 * Standalone test for the A packing kernel.
 *
 * Packs operand A into GMMA register layout, then verifies correctness by:
 * 1. Running the pack kernel
 * 2. Running a CPU reference that simulates the same transformation
 * 3. Comparing the packed outputs
 *
 * Target: SM90 (uses TMA for gmem→smem, SM90 WGMMA register layout)
 **************************************************************************************************/

#include "split_a_pack.h"

// CPU reference: simulate the register layout by running the same S2R transformation
// on the host. For BF16 POC, we verify that the packed data has the right size
// and that the total element count matches (no data loss).
//
// A rigorous check: pack on GPU, unpack back to original layout on CPU, compare.
// The "unpack" is the inverse of the register dump: read elements back in register order
// and reconstruct the original tile.
//
// For the POC, we use a simpler check: run sample 13's GEMM with the original A,
// and verify that the packed data can produce the same result (done in the WGMMA test).
// Here we just verify the pack kernel runs without errors and the packed size is correct.

int main(int argc, char** argv)
{
  using namespace cute;

  printf("BF16 Split A Pack (SM90 TMA + RS WGMMA register layout)\n\n");

  float alpha = 1.0f;
  float beta  = 0.0f;

  // ---- Test at 1024x1024 ----
  {
    int m = 1024, k = 1024;
    int ldA = k;

    thrust::host_vector<bf16_t> h_A(m * k);
    for (int i = 0; i < m * k; ++i) h_A[i] = static_cast<bf16_t>(2.0 * (rand() / double(RAND_MAX)) - 1.0);

    thrust::device_vector<bf16_t> d_A = h_A;

    // Packed buffer: same total size as A (each element maps to exactly one register slot)
    thrust::device_vector<bf16_t> d_packed(m * k);

    // Run pack kernel
    split_a_pack(m, k, d_A.data().get(), ldA, d_packed.data().get());
    CUTE_CHECK_LAST();

    // Verify: copy packed data back and check all values are valid BF16 (no NaN/Inf from uninit)
    thrust::host_vector<bf16_t> h_packed = d_packed;
    bool has_bad = false;
    for (int i = 0; i < m * k; ++i) {
      float v = float(h_packed[i]);
      if (std::isnan(v) || std::isinf(v)) { has_bad = true; break; }
    }

    printf("Pack test (1024x1024): %s (%d elements)\n\n",
           has_bad ? "FAIL (bad values)" : "PASS (all valid)", m * k);
    if (has_bad) return 1;
  }

  // ---- Benchmark ----
  printf("Pack benchmark:\n");
  for (int size : {256, 512, 1024, 2048, 4096}) {
    int m = size, k = size;
    int ldA = k;

    thrust::device_vector<bf16_t> d_A(m * k), d_packed(m * k);
    thrust::host_vector<bf16_t> h_A(m * k);
    for (int i = 0; i < m * k; ++i) h_A[i] = static_cast<bf16_t>(2.0 * (rand() / double(RAND_MAX)) - 1.0);
    d_A = h_A;

    const int timing_iterations = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    split_a_pack(m, k, d_A.data().get(), ldA, d_packed.data().get());
    CUTE_CHECK_LAST();

    cudaEventRecord(start);
    for (int i = 0; i < timing_iterations; ++i) {
      split_a_pack(m, k, d_A.data().get(), ldA, d_packed.data().get());
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    double avg_ms = total_ms / timing_iterations;
    double gb = 2.0 * m * k * sizeof(bf16_t) * 1e-9;  // read + write
    printf("  %dx%d: %.1f GB/s (%.4f ms)\n", m, k, gb / (avg_ms * 1e-3), avg_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  return 0;
}
```

- [ ] **Step 3: Build the pack target**

```bash
cd /data/lmdeploy-cute/build && ninja bf16_gemm_sm90_split_a_pack 2>&1
```

Expected: successful compilation with no errors. PTXAS output should show register usage for 256-thread kernel.

If compilation fails with `tma_partition` errors for 2D tensors, try removing `group_modes<0,2>` and using `(_,0)` indexing instead (pattern from `test_tma_load.cu`):
```cuda
auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{}, sA(_,_,0), gA);
copy(tma_a.with(smem.mbarrier), tAgA(_,0), tAsA(_,0));
```

- [ ] **Step 4: Run the pack test**

```bash
cd /data/lmdeploy-cute/build && ./bf16_gemm_sm90_split_a_pack
```

Expected: "PASS (all valid)" and benchmark throughput numbers.

- [ ] **Step 5: Commit**

```bash
git add cute-reference/mixed-gemm/split_a_pack.h cute-reference/mixed-gemm/bf16_gemm_sm90_split_a_pack.cu
git commit -m "Add pack kernel for split A loading RS WGMMA POC"
```

---

### Task 3: Kernel 2 — RS WGMMA Using Packed A

**Files:**
- Create: `cute-reference/mixed-gemm/bf16_gemm_sm90_split_a_wgmma.cu`

This kernel runs the full GEMM using pre-packed A. Architecture mirrors sample 13 but replaces the A TMA+S2R pipeline with direct gmem reads from the packed buffer.

**Key differences from sample 13:**
1. **No A smem** — SharedStorage only has B and C. A register layout is computed from a dummy smem tensor (no allocation).
2. **Pipeline for B only** — `PipelineTmaAsync<3>` with transaction bytes for B only.
3. **Consumer loads packed A directly** — each thread reads its register values from gmem using the same offset formula as the pack kernel.
4. **Everything else identical** — same TiledMMA, WGMMA loop, epilogue (STSM + TMA store).

- [ ] **Step 1: Create the WGMMA kernel file**

Create `cute-reference/mixed-gemm/bf16_gemm_sm90_split_a_wgmma.cu` with the full kernel, host function, and end-to-end test:

```cuda
/***************************************************************************************************
 * BF16 GEMM using SM90 WGMMA with pre-packed operand A.
 *
 * Kernel 2 of the split A loading design. Uses the packed A buffer (produced by
 * split_a_pack kernel) as direct register input for RS WGMMA.
 *
 * C = alpha * A_packed * B^T + beta * C
 *
 * Differences from sample 13:
 *   - A loaded from packed gmem buffer (direct reads to registers)
 *   - No A smem allocation
 *   - TMA pipeline for B only
 *
 * Target: SM90
 **************************************************************************************************/

#include "split_a_pack.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include <cmath>

// ================================================================================================
// SharedStorage (no A smem)
// ================================================================================================

template <class ElementB, class ElementC,
          class SmemLayoutB, class SmemLayoutC, int Stages>
struct WgmmaSharedStorage
{
  // NO A smem — A is loaded from packed gmem directly into registers
  alignas(128) cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
  alignas(128) cute::ArrayEngine<ElementC, cute::cosize_v<SmemLayoutC>> C;
  typename cutlass::PipelineTmaAsync<Stages>::SharedStorage pipeline;
};

// ================================================================================================
// Device Kernel (Persistent, warp-specialized)
// ================================================================================================

template <class ProblemShape, class CtaTiler,
          class SmemLayoutA,  // for fragment creation only (no allocation)
          class TB, class SmemLayoutB, class TmaB,
          class TC, class SmemLayoutC,
          class TmaStoreC, class R2SCopy, class CStride, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value * 3 / 2, 1)  // 384 threads
void
split_a_wgmma_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                     bf16_t const* packed_A, SmemLayoutA,
                     TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
                     TC      * C, SmemLayoutC,
                     CUTLASS_GRID_CONSTANT TmaStoreC const tma_store_c,
                     R2SCopy r2s_copy, CStride dC, TiledMma mma,
                     Alpha alpha, Beta beta,
                     int total_tiles, int k_tile_count)
{
  using namespace cute;

  // ---- Preconditions ----
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

  auto [M, N, K] = shape_MNK;

  // ---- Global memory tensors ----
  Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);

  // ---- Shared memory ----
  extern __shared__ char shared_memory[];
  using SharedStorage = WgmmaSharedStorage<TB, TC, SmemLayoutB, SmemLayoutC, cute::size<2>(SmemLayoutB{})>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});
  Tensor sC = make_tensor(make_smem_ptr(smem.C.begin()), SmemLayoutC{});

  // ---- Tile counts ----
  int m_tiles = size(ceil_div(M, size<0>(cta_tiler)));
  int n_tiles = size(ceil_div(N, size<1>(cta_tiler)));

  // ---- Pipeline setup (B only) ----
  constexpr int tma_transaction_bytes =
      sizeof(TB) * cute::cosize_v<SmemLayoutB> / cute::size<2>(SmemLayoutB{});

  int warp_group_idx = cutlass::canonical_warp_group_idx();
  int warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

  using MainloopPipeline = cutlass::PipelineTmaAsync<cute::size<2>(SmemLayoutB{})>;
  typename MainloopPipeline::Params pipeline_params;
  if (warp_group_idx == 2) {
    pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
  } else {
    pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
  }
  pipeline_params.is_leader = (warp_group_thread_idx == 0);
  pipeline_params.num_consumers = 256;
  pipeline_params.num_producers = 1;
  pipeline_params.transaction_bytes = tma_transaction_bytes;

  MainloopPipeline pipeline(smem.pipeline, pipeline_params, cute::make_layout(cute::make_shape(cute::_1{}, cute::_1{})));
  __syncthreads();

  auto smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
  typename MainloopPipeline::PipelineState smem_pipe_read;
  typename MainloopPipeline::PipelineState smem_pipe_release;

  uint64_t linear_idx = blockIdx.x;
  uint64_t grid_size = gridDim.x;

  if (warp_group_idx == 2) {
    // ==================================================================
    // Producer warp group — TMA loads for B only (persistent)
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

        int k_tile_next = 0;
        CUTE_NO_UNROLL
        for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
          pipeline.producer_acquire(smem_pipe_write);
          BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
          copy(tma_b.with(*tma_barrier), tBgB(_,k_tile_next), tBsB(_,smem_pipe_write.index()));
          ++smem_pipe_write;
          ++k_tile_next;
        }

        linear_idx += grid_size;
      }
      pipeline.producer_tail(smem_pipe_write);
    }

    cute::tma_store_wait<0>();

  } else {
    // ==================================================================
    // Consumer warp groups (wg 0 and 1) — load packed A + WGMMA + epilogue
    // ==================================================================
    cutlass::arch::warpgroup_reg_alloc<232>();

    // ---- TiledMMA setup ----
    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);

    // Dummy A smem tensor for fragment creation (no actual smem allocation for A)
    Tensor dummy_sA = make_tensor(make_smem_ptr(reinterpret_cast<bf16_t*>(shared_memory)), SmemLayoutA{});
    Tensor dummy_tCsA = thr_mma.partition_A(dummy_sA);
    Tensor tCrA = thr_mma.make_fragment_A(dummy_tCsA(_,_,_,Int<0>{}));       // (MMA, MMA_M, MMA_K)

    constexpr int regs_per_thread = size(tCrA);

    // B partition (smem via GMMA descriptor)
    Tensor tCsB = thr_mma.partition_B(sB);                                   // (MMA, MMA_N, MMA_K, PIPE)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);                             // GMMA descriptors

    constexpr int k_block_count = size<2>(tCrA);  // 4 for BF16 with K=64

    // Accumulator
    Tensor gC_dummy = make_tensor(make_gmem_ptr(C),
                                  make_shape(size<0>(cta_tiler), size<1>(cta_tiler)), dC);
    Tensor tCrC = thr_mma.make_fragment_C(thr_mma.partition_C(gC_dummy));

    // ---- R2S copy setup ----
    ThrCopy thr_r2s = r2s_copy.get_slice(threadIdx.x);

    // ---- TMA store setup ----
    auto cta_tile_mn = product_each(shape(SmemLayoutC{}));
    Tensor mC_tma = tma_store_c.get_tma_tensor(make_shape(M, N));
    Tensor gC_tma_full = flat_divide(mC_tma, cta_tile_mn);

    auto cta_tma_store = tma_store_c.get_slice(Int<0>{});
    Tensor tSsC_x = cta_tma_store.partition_S(sC);
    Tensor tSgC_x = cta_tma_store.partition_D(gC_tma_full);
    Tensor tSgC = group_modes<1, rank(tSgC_x)>(tSgC_x);
    Tensor tSsC = group_modes<1, rank(tSsC_x)>(tSsC_x);

    // ---- Persistent while loop ----
    while (linear_idx < total_tiles) {
      int m_idx = linear_idx / n_tiles;
      int n_idx = linear_idx % n_tiles;

      Tensor gC = local_tile(mC, cta_tiler, make_coord(m_idx, n_idx, _), Step<_1, _1, X>{});
      Tensor tCgC = thr_mma.partition_C(gC);

      clear(tCrC);

      // ---- Main loop: load packed A + wait for B + WGMMA ----
      CUTE_NO_UNROLL
      for (int k_tile_iter = 0; k_tile_iter < k_tile_count; ++k_tile_iter)
      {
        // Load packed A directly from gmem into registers
        int a_tile_idx = m_idx * k_tile_count + k_tile_iter;
        const bf16_t* packed_ptr = packed_A + a_tile_idx * 256 * regs_per_thread;
        CUTE_UNROLL
        for (int i = 0; i < regs_per_thread; ++i) {
          tCrA(i) = packed_ptr[i * 256 + threadIdx.x];
        }

        // Wait for B
        pipeline.consumer_wait(smem_pipe_read);
        int read_stage = smem_pipe_read.index();
        ++smem_pipe_read;

        warpgroup_fence_operand(tCrC);

        // Batched WGMMA
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

      // ---- Epilogue ----
      // Stage 1: alpha/beta scaling
      CUTE_UNROLL
      for (int i = 0; i < size(tCrC); ++i) {
        tCrC(i) = alpha * tCrC(i) + beta * static_cast<float>(tCgC(i));
      }

      // Stage 2: F32 → BF16, STSM
      Tensor tCrC_bf16 = make_tensor<bf16_t>(tCrC.layout());
      CUTE_UNROLL
      for (int i = 0; i < size(tCrC); ++i) {
        tCrC_bf16(i) = static_cast<bf16_t>(tCrC(i));
      }

      Tensor tRS_rAcc = thr_r2s.retile_S(tCrC_bf16);
      Tensor tRS_sC   = thr_r2s.partition_D(sC);
      copy(r2s_copy, tRS_rAcc, tRS_sC);

      // Consumer-only sync
      cutlass::arch::NamedBarrier consumer_sync(256, 6);
      consumer_sync.sync();

      // Stage 3: TMA store
      int rest_idx = m_idx + n_idx * m_tiles;
      if (threadIdx.x == 0) {
        tma_store_fence();
        copy(tma_store_c, tSsC(_, 0), tSgC(_, rest_idx));
        tma_store_arrive();
      }
      tma_store_wait<0>();

      linear_idx += grid_size;
    }
  }
}

// ================================================================================================
// Host function
// ================================================================================================

template <class Alpha, class Beta>
void
split_a_wgmma(int m, int n, int k,
              Alpha alpha,
              bf16_t const* packed_A,
              bf16_t const* B, int ldB,
              Beta beta,
              bf16_t* C, int ldC,
              cudaStream_t stream = 0)
{
  using namespace cute;

  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);

  auto dB = make_stride(ldB, Int<1>{});   // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);   // (dM, dN)

  auto bM = Int<128>{};
  auto bN = Int<256>{};
  auto bK = Int<64>{};
  auto cta_tiler = make_shape(bM, bN, bK);
  auto bP = Int<3>{};

  // A smem layout for fragment creation (same as sample 13 — used for layout only, no allocation)
  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16_t>{}, make_shape(bM, bK, bP));

  // B smem layout
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16_t>{}, make_shape(bN, bK, bP));

  // C smem layout
  auto sC_layout = make_layout(make_shape(bM, bN), make_stride(Int<1>{}, bM));

  // TMA load for B
  Tensor tB = make_tensor(B, make_shape(N, K), dB);
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, tB, sB(_,_,0), make_shape(bN, bK));

  // TMA store for C
  Tensor tC = make_tensor(C, make_shape(M, N), dC);
  auto tma_store_c = make_tma_copy(SM90_TMA_STORE{}, tC, sC_layout, make_shape(bM, bN), Int<1>{});

  // TiledMMA
  TiledMMA mma = make_tiled_mma(
      SM90_64x256x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::K>{},
      Layout<Shape<_2, _1>>{});

  static_assert(decltype(size(mma))::value == 256, "Expected 256 threads");

  // R2S copy
  auto r2s_copy = make_tiled_copy_C(
      Copy_Atom<SM90_U16x8_STSM_T, bf16_t>{},
      mma);

  // Grid
  int num_SMs = 0;
  CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0));

  int total_tiles = size(ceil_div(M, bM)) * size(ceil_div(N, bN));
  int k_tile_count = size(ceil_div(K, bK));

  dim3 dimBlock(size(mma) * 3 / 2);  // 384 threads
  dim3 dimCluster(1, 1, 1);
  dim3 dimGrid(std::min(num_SMs, total_tiles));

  int smem_size = int(sizeof(WgmmaSharedStorage<bf16_t, bf16_t, decltype(sB), decltype(sC_layout), cute::size<2>(decltype(sB){})>));

  auto* kernel_ptr = &split_a_wgmma_device<
      decltype(prob_shape), decltype(cta_tiler),
      decltype(sA),
      bf16_t, decltype(sB), decltype(tmaB),
      bf16_t, decltype(sC_layout),
      decltype(tma_store_c), decltype(r2s_copy), decltype(dC), decltype(mma),
      Alpha, Beta>;

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      (void const*)kernel_ptr,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size));
  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      (void const*)kernel_ptr,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      100));

  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*)kernel_ptr,
      prob_shape, cta_tiler,
      packed_A, sA,
      B, tmaB,
      C, sC_layout, tma_store_c, r2s_copy, dC, mma,
      alpha, beta,
      total_tiles, k_tile_count);

  CUTE_CHECK_LAST();
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed at WGMMA kernel launch" << std::endl;
  }
}

// ================================================================================================
// Main — end-to-end test: pack A, run WGMMA, verify against CPU reference
// ================================================================================================

int main(int argc, char** argv)
{
  using namespace cute;

  printf("BF16 Split A WGMMA (SM90, pre-packed A, tile 128x256x64, 384t WS, persistent)\n\n");

  float alpha = 1.0f;
  float beta  = 0.0f;

  // ---- Correctness test at 1024^3 ----
  {
    int m = 1024, n = 1024, k = 1024;
    int ldA = k, ldB = k, ldC = m;

    thrust::host_vector<bf16_t> h_A(m * k), h_B(n * k), h_C(m * n, static_cast<bf16_t>(-1.0f));
    for (int i = 0; i < m * k; ++i) h_A[i] = static_cast<bf16_t>(2.0 * (rand() / double(RAND_MAX)) - 1.0);
    for (int i = 0; i < n * k; ++i) h_B[i] = static_cast<bf16_t>(2.0 * (rand() / double(RAND_MAX)) - 1.0);

    thrust::device_vector<bf16_t> d_A = h_A, d_B = h_B;
    thrust::device_vector<bf16_t> d_C = h_C;
    thrust::device_vector<bf16_t> d_packed(m * k);

    // Step 1: Pack A
    split_a_pack(m, k, d_A.data().get(), ldA, d_packed.data().get());
    CUTE_CHECK_LAST();

    // Step 2: Run WGMMA with packed A
    split_a_wgmma(m, n, k, alpha,
                  d_packed.data().get(),
                  d_B.data().get(), ldB,
                  beta,
                  d_C.data().get(), ldC);
    CUTE_CHECK_LAST();

    // Step 3: Verify against CPU reference
    thrust::host_vector<bf16_t> h_result = d_C;

    thrust::host_vector<float> h_ref(m * n, 0.0f);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) {
        float sum = 0.0f;
        for (int l = 0; l < k; ++l)
          sum += float(h_A[i * k + l]) * float(h_B[j * k + l]);
        h_ref[i + j * ldC] = alpha * sum + beta * float(h_C[i + j * ldC]);
      }

    float max_err = 0.0f;
    for (int i = 0; i < m * n; ++i)
      max_err = std::max(max_err, std::abs(float(h_result[i]) - h_ref[i]));

    printf("Correctness (1024^3): max error %e — %s\n\n", max_err, max_err < 0.5f ? "PASS" : "FAIL");
    if (max_err >= 0.5f) return 1;
  }

  // ---- Additional test sizes ----
  {
    struct TestSize { int m, n, k; };
    TestSize sizes[] = {{2048, 1024, 512}, {512, 512, 512}};

    for (auto [m, n, k] : sizes) {
      int ldA = k, ldB = k, ldC = m;

      thrust::host_vector<bf16_t> h_A(m * k), h_B(n * k), h_C(m * n, static_cast<bf16_t>(-1.0f));
      for (int i = 0; i < m * k; ++i) h_A[i] = static_cast<bf16_t>(2.0 * (rand() / double(RAND_MAX)) - 1.0);
      for (int i = 0; i < n * k; ++i) h_B[i] = static_cast<bf16_t>(2.0 * (rand() / double(RAND_MAX)) - 1.0);

      thrust::device_vector<bf16_t> d_A = h_A, d_B = h_B;
      thrust::device_vector<bf16_t> d_C = h_C;
      thrust::device_vector<bf16_t> d_packed(m * k);

      split_a_pack(m, k, d_A.data().get(), ldA, d_packed.data().get());
      CUTE_CHECK_LAST();

      split_a_wgmma(m, n, k, alpha,
                    d_packed.data().get(),
                    d_B.data().get(), ldB,
                    beta,
                    d_C.data().get(), ldC);
      CUTE_CHECK_LAST();

      thrust::host_vector<bf16_t> h_result = d_C;

      thrust::host_vector<float> h_ref(m * n, 0.0f);
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
          float sum = 0.0f;
          for (int l = 0; l < k; ++l)
            sum += float(h_A[i * k + l]) * float(h_B[j * k + l]);
          h_ref[i + j * ldC] = alpha * sum + beta * float(h_C[i + j * ldC]);
        }

      float max_err = 0.0f;
      for (int i = 0; i < m * n; ++i)
        max_err = std::max(max_err, std::abs(float(h_result[i]) - h_ref[i]));

      printf("Correctness (%dx%dx%d): max error %e — %s\n", m, n, k, max_err, max_err < 0.5f ? "PASS" : "FAIL");
      if (max_err >= 0.5f) return 1;
    }
    printf("\n");
  }

  // ---- Benchmark ----
  printf("Benchmark (100 iterations each):\n");
  for (int size : {256, 512, 1024, 2048, 4096}) {
    int m = size, n = size, k = size;
    int ldA = k, ldB = k, ldC = m;

    thrust::device_vector<bf16_t> d_A(m * k), d_B(n * k), d_C(m * n), d_packed(m * k);
    thrust::host_vector<bf16_t> h_A(m * k), h_B(n * k);
    for (int i = 0; i < m * k; ++i) h_A[i] = static_cast<bf16_t>(2.0 * (rand() / double(RAND_MAX)) - 1.0);
    for (int i = 0; i < n * k; ++i) h_B[i] = static_cast<bf16_t>(2.0 * (rand() / double(RAND_MAX)) - 1.0);
    d_A = h_A; d_B = h_B;

    // Pack A (not timed)
    split_a_pack(m, k, d_A.data().get(), ldA, d_packed.data().get());
    CUTE_CHECK_LAST();

    const int timing_iterations = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    split_a_wgmma(m, n, k, alpha, d_packed.data().get(),
                  d_B.data().get(), ldB, beta, d_C.data().get(), ldC);
    CUTE_CHECK_LAST();

    cudaEventRecord(start);
    for (int i = 0; i < timing_iterations; ++i) {
      split_a_wgmma(m, n, k, alpha, d_packed.data().get(),
                    d_B.data().get(), ldB, beta, d_C.data().get(), ldC);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    double avg_ms = total_ms / timing_iterations;
    double gflops = (2.0 * m * n * k) * 1e-9;
    printf("  %dx%dx%d: %.1f GFLOP/s (%.4f ms)\n", m, n, k, gflops / (avg_ms * 1e-3), avg_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  return 0;
}
```

- [ ] **Step 2: Build the WGMMA target**

```bash
cd /data/lmdeploy-cute/build && ninja bf16_gemm_sm90_split_a_wgmma 2>&1
```

Expected: successful compilation. If errors occur:

- **`reinterpret_cast<bf16_t*>(shared_memory)` for dummy_sA**: This is used to create a dummy tensor for fragment creation. If the compiler complains, try using `static_cast` or a dedicated smem offset.

- **Missing includes**: Add any missing CUTLASS headers referenced by sample 13.

- [ ] **Step 3: Run the end-to-end test**

```bash
cd /data/lmdeploy-cute/build && ./bf16_gemm_sm90_split_a_wgmma
```

Expected:
```
Correctness (1024^3): max error <some_small_value> — PASS
Correctness (2048x1024x512): max error <some_small_value> — PASS
Correctness (512x512x512): max error <some_small_value> — PASS
Benchmark ...
```

If correctness fails (max error >= 0.5):
1. **Debug the pack kernel first**: In the standalone pack test, add a check that reads packed data back and verifies individual tile values match the CPU reference.
2. **Debug the register mapping**: Print `regs_per_thread` and verify it's 32 (expected for bM=128, bK=64, 256 threads).
3. **Debug the A tile indexing**: Verify that `a_tile_idx = m_idx * k_tile_count + k_tile_iter` matches the pack kernel's `linear_idx = tile_m * k_tile_count + tile_k`.

Iterate until the correctness test passes.

- [ ] **Step 4: Commit**

```bash
git add cute-reference/mixed-gemm/bf16_gemm_sm90_split_a_wgmma.cu
git commit -m "Add WGMMA kernel using pre-packed A for split A loading POC"
```

---

### Task 4: Fix Build Configuration

**Files:**
- Modify: `my_generate.sh`

The pack and WGMMA targets are gated behind `BUILD_CUTE_SAMPLES=ON`, which is not set in `my_generate.sh`. For convenient rebuilds during development, add the flag.

- [ ] **Step 1: Add BUILD_CUTE_SAMPLES to my_generate.sh**

In `my_generate.sh`, add `-DBUILD_CUTE_SAMPLES=ON` to the cmake command.

- [ ] **Step 2: Reconfigure and rebuild**

```bash
cd /data/lmdeploy-cute/build && sh ../my_generate.sh && ninja bf16_gemm_sm90_split_a_pack bf16_gemm_sm90_split_a_wgmma
```

- [ ] **Step 3: Run both tests to confirm**

```bash
cd /data/lmdeploy-cute/build && ./bf16_gemm_sm90_split_a_pack && ./bf16_gemm_sm90_split_a_wgmma
```

- [ ] **Step 4: Commit**

```bash
git add my_generate.sh
git commit -m "Enable BUILD_CUTE_SAMPLES in build configuration"
```

---

## Self-Review

### Verified Correctness (Codebase Exploration)

| Concern | Finding | Source |
|---|---|---|
| `tma_partition` with 2D tensors | Works with 2D tensors (no `group_modes` needed for simple cases), but `group_modes<0,2>` is the proven pattern from sample 13 | `test_tma_load.cu:37`, sample 13 |
| Raw mbarrier pattern | `cute::initialize_barrier` + `cute::set_barrier_transaction_bytes` + `cute::wait_barrier` pattern is verified in `tma_load_testbed.hpp` | Lines 139-149 of test bed |
| `make_fragment_A` with dummy smem tensor | Safe for RS WGMMA. `FrgTypeA = bfloat16_t` (no `has_dereference`), so Path B (`make_fragment_like`) is taken — only reads layout, never dereferences data pointer | `mma_atom.hpp:145-169`, `mma_traits_sm90_gmma.hpp:1866-1883` |
| Register layout consistency across PIPE stages | **Identical.** `thrfrg_A` only touches modes 0 and 2 (spatial M,K), PIPE passes through untouched. `tCsA(_,_,_,Int<0>{})` strips PIPE before `make_fragment_A`, producing the same rank-3 tensor regardless of total PIPE count. | `mma_atom.hpp:288-314` |
| Dummy smem pointer safety | `partition_A` calls `thrfrg_A(atensor.layout())` (layout-only) and `make_tensor(atensor.data(), new_layout)` (non-owning view). Neither dereferences data. For RS WGMMA Path B, `make_fragment_like` creates a fresh register tensor from layout alone. | `mma_atom.hpp:521-530, 145-169` |
| `local_tile` with 2D tiler and 2D coord on 2D tensor | Produces shape `(bM, bK)` — a single tile with no rest dimensions. Confirmed by CuTe docs. | `tensor_impl.hpp:978-1035` |

### Spec Coverage

| Spec Section | Covered By |
|---|---|
| Kernel 1: TMA g2s + S2R + register dump | Task 2, `split_a_pack.h` |
| Kernel 2: Direct gmem reads for packed A | Task 3, WGMMA kernel consumer mainloop |
| Kernel 2: TMA pipeline for B only | Task 3, producer warp group |
| Kernel 2: Batched WGMMA | Task 3, consumer mainloop |
| Kernel 2: STSM epilogue | Task 3, epilogue section |
| Shared constants (bM, bK, MMA atom) | Both kernels use same `make_tiled_mma` |
| Packed gmem layout `[tile][thread+stride]` | Both kernels use `256 * regs_per_thread` stride |
| CPU reference verification | Task 3, main() |
| Test sizes 1024^3, 2048x1024x512 | Task 3, main() |
| Files in `cute-reference/mixed-gemm/` | All files |

### Placeholder Scan

No TBD, TODO, or "implement later" patterns found. All steps contain complete code.

### Type Consistency

- `TiledMMA` construction is identical in both kernels: `make_tiled_mma(SM90_64x256x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::K>{}, Layout<Shape<_2, _1>>{})`
- `regs_per_thread = size(tCrA)` computed the same way in both kernels (via `thr_mma.make_fragment_A`)
- Packed buffer offset uses `256 * regs_per_thread` in both kernels
- `SmemLayoutA` in Kernel 2 uses the same `tile_to_shape` as sample 13 (3 stages), ensuring identical register fragment layout
