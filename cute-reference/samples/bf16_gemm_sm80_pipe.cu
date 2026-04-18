/***************************************************************************************************
 * BF16 GEMM using SM80 tensor cores with CuTe — Pipelined Sample (cp.async 3-stage)
 *
 * Demonstrates (on top of the optimized sample):
 *   - cp.async asynchronous global-to-shared memory copy (SM80 feature)
 *   - 3-stage software pipeline overlapping gmem->smem with MMA compute
 *   - Dynamic shared memory allocation (extern __shared__)
 *   - TiledCopy for gmem->smem replacing synchronous local_partition
 *   - partition_fragment_A/B for register allocation from pipelined smem
 *
 * C = alpha * A * B^T + beta * C
 *   A: bf16, M x K, row-major (TN layout)
 *   B: bf16, K x N, stored as (N, K) in CuTe with K-contiguous stride
 *   C: f32, M x N, column-major
 *
 * Target: SM80+ tensor cores (runs on SM90 L20Y GPU)
 *
 * Pipeline overview:
 *   The key insight is that cp.async allows the GPU to load the next K-tile from global
 *   memory into shared memory while the current K-tile is being processed by tensor cores.
 *   We use 3 pipe stages (bP=3): while the MMA works on stage N, cp.async fills stage N+1
 *   and stage N+2 may already be in flight.
 *
 *   Timeline (conceptual):
 *     T0: cp.async load pipe[0] from gmem
 *     T1: cp.async load pipe[1] from gmem | wait for pipe[0] | smem->regs | MMA on pipe[0]
 *     T2: cp.async load pipe[2] from gmem | wait for pipe[1] | smem->regs | MMA on pipe[1]
 *     T3: cp.async load pipe[0] from gmem | wait for pipe[2] | smem->regs | MMA on pipe[2]
 *         ^-- pipe[0] is now free because MMA finished with it
 *
 * Compared to bf16_gemm_sm80_opt.cu (synchronous copy):
 *   - gmem->smem: synchronous local_partition -> asynchronous TiledCopy with cp.async
 *   - Shared memory: static __shared__ -> dynamic extern __shared__ with SharedStorage struct
 *   - Smem layouts: 2D (BLK_M, BLK_K) -> 3D (BLK_M, BLK_K, PIPE) to hold multiple stages
 *   - Main loop: simple for -> pipelined while with circular buffer indexing
 *   - Register allocation: make_fragment_A/B -> partition_fragment_A/B (slice from 3D smem)
 *   - Added: cp_async_fence(), cp_async_wait<N>() for async copy ordering
 *   - Added: cudaFuncSetAttribute for dynamic smem size and L1 carveout
 **************************************************************************************************/
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cmath>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

using bf16_t = cute::bfloat16_t;

// ================================================================================================
// SharedStorage struct
// ================================================================================================
//
// Encapsulates shared memory allocation for both A and B matrices.
// Uses CuTe's ArrayEngine which provides properly aligned storage for the layout's elements.
// cosize_v<SmemLayout> computes the total number of elements the layout addresses,
// including any padding introduced by swizzle patterns.
//
// This struct is used with dynamic shared memory (extern __shared__ char[]) and is
// reinterpret_cast'd into place. This is the standard CuTe/CUTLASS pattern for
// pipelined kernels that need explicit control over shared memory layout.
//
// The SharedStorage approach has two advantages over static __shared__ arrays:
//   1. Supports 3D pipelined layouts where the third dimension (PIPE) requires
//      multiple back-to-back copies of the 2D tile.
//   2. Allows cudaFuncSetAttribute to query/set the exact shared memory requirement,
//      which is critical for kernels that need >48KB of shared memory.

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage
{
  cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
};

// ================================================================================================
// Device Kernel
// ================================================================================================
//
// Each thread block computes a (BLK_M x BLK_N) tile of output C.
// The K dimension is processed in chunks of BLK_K using a 3-stage pipeline.
//
// Data flow (pipelined, per K-tile):
//   1. cp.async issues async gmem -> smem copy for the NEXT pipe stage
//      (overlaps with MMA compute on the CURRENT pipe stage)
//   2. cp_async_fence() commits pending async copies
//   3. cp_async_wait<N>() + __syncthreads() ensures the target pipe stage is ready
//   4. LDSM vectorized smem -> register copy for the CURRENT pipe stage
//   5. Tensor core MMA on registers
//   6. Pipe indices rotate: read advances, write takes the slot just freed by MMA
//
// Template parameters follow the sgemm_sm80.cu canonical pattern:
//   - AG2SCopy/BG2SCopy: TiledCopy objects for async gmem->smem (replaces thread layouts)
//   - S2RCopyAtomA/B: LDSM copy atoms for smem->register (moved after their G2S counterpart)

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AG2SCopy, class S2RCopyAtomA,
          class TB, class BStride, class BSmemLayout, class BG2SCopy, class S2RCopyAtomB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
bf16_gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                 TA const* A, AStride dA, ASmemLayout sA_layout, AG2SCopy g2s_copy_a, S2RCopyAtomA s2r_atom_a,
                 TB const* B, BStride dB, BSmemLayout sB_layout, BG2SCopy g2s_copy_b, S2RCopyAtomB s2r_atom_b,
                 TC      * C, CStride dC, CSmemLayout          , TiledMma mma,
                 Alpha alpha, Beta beta)
{
  using namespace cute;

  // ---- Preconditions ----
  //
  // These compile-time checks verify that the kernel parameters are consistent.
  // CUTE_STATIC_ASSERT_V works on dynamic values (runtime-known tensor dimensions).
  // static_assert works on types only (compile-time-known layout properties).

  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

  // The TiledCopy for gmem->smem must involve the same number of threads as the TiledMMA.
  // This ensures the kernel launches with the correct number of threads.
  // (Replaces the old size(tA) == size(mma) check for thread layouts.)
  CUTE_STATIC_ASSERT_V(size(g2s_copy_a) == size(mma));                 // NumThreads
  CUTE_STATIC_ASSERT_V(size(g2s_copy_b) == size(mma));                 // NumThreads

  // Smem layouts must be static (compile-time known) for shared memory addressing.
  // Note: The PIPE dimension is included in the smem layout, so size<0>/size<1> still
  // correctly capture BLK_M/BLK_K and BLK_N/BLK_K — the PIPE dim is size<2>.
  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(size<0>(sA_layout) == size<0>(cta_tiler));      // BLK_M
  CUTE_STATIC_ASSERT_V(size<1>(sA_layout) == size<2>(cta_tiler));      // BLK_K
  CUTE_STATIC_ASSERT_V(size<0>(sB_layout) == size<1>(cta_tiler));      // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(sB_layout) == size<2>(cta_tiler));      // BLK_K
  CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));
  CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));
  CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));

  // ---- Step 1: Global memory tensors ----
  //
  // make_tensor wraps a raw pointer with shape and stride information.
  // select<0,2>(shape_MNK) extracts (M, K) from (M, N, K).
  //
  // local_tile extracts this CTA's subtensor from the full tensor.
  //   Step<_1, X, _1> means "tile M and K dimensions, skip N"
  //   Step< X,_1, _1> means "skip M, tile N and K dimensions"
  //   Step<_1,_1,  X> means "tile M and N, skip K"
  //
  // gA and gB have a third mode "k" representing the number of K-tiles to process.

  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M, K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N, K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M, N)

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});  // (BLK_M, BLK_K, k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});  // (BLK_N, BLK_K, k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});  // (BLK_M, BLK_N)

  // ---- Step 2: Shared memory tensors (dynamic, 3D with pipeline dimension) ----
  //
  // Unlike the non-pipelined version which uses static __shared__ arrays,
  // the pipelined kernel uses dynamic shared memory via extern __shared__ char[].
  //
  // The SharedStorage struct provides two ArrayEngine members for A and B.
  // The smem layouts are now 3D: (BLK_M, BLK_K, bP) where bP=3 is the pipeline depth.
  // This means shared memory holds 3 copies of each tile — one being loaded,
  // one being consumed by MMA, and one as a buffer.
  //
  // make_smem_ptr creates a CuTe tensor pointer into shared memory with proper
  // alignment guarantees. The swizzle in the layout is transparent to all users.

  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);   // (BLK_M, BLK_K, bP)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);   // (BLK_N, BLK_K, bP)

  // ---- Step 3: gmem -> smem copy partitioning (async via TiledCopy) ----
  //
  // The non-pipelined version used local_partition with thread layouts for synchronous copy.
  // Here we use TiledCopy objects (g2s_copy_a, g2s_copy_b) which encapsulate:
  //   - The copy atom: SM80_CP_ASYNC_CACHEALWAYS<uint128_t> — async 128-bit copy
  //   - Thread layout: how threads map to the tile
  //   - Value layout: how many values each thread copies per instruction
  //
  // get_slice(threadIdx.x) gives this thread's view of the TiledCopy.
  // partition_S(gA) partitions the source (global memory) tensor among threads.
  // partition_D(sA) partitions the destination (shared memory) tensor among threads.
  //
  // Resulting shapes:
  //   tAgA: (CPY, CPY_M, CPY_K, k) — 4D: copy mode, M tiles, K tiles, K-pipeline
  //   tAsA: (CPY, CPY_M, CPY_K, PIPE) — 4D: same but 4th dim is pipe stage, not K-tile index
  //
  // The async copy instruction (cp.async) will write to the pipe stage indicated by the
  // 4th dimension of tAsA, and read from the K-tile indicated by the 4th dim of tAgA.

  ThrCopy thr_g2s_a = g2s_copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_g2s_a.partition_S(gA);   // (CPY, CPY_M, CPY_K, k)
  Tensor tAsA = thr_g2s_a.partition_D(sA);   // (CPY, CPY_M, CPY_K, PIPE)

  ThrCopy thr_g2s_b = g2s_copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_g2s_b.partition_S(gB);   // (CPY, CPY_N, CPY_K, k)
  Tensor tBsB = thr_g2s_b.partition_D(sB);   // (CPY, CPY_N, CPY_K, PIPE)

  // Verify that the source and destination partition shapes match in M/N and K dimensions.
  // The 4th dimension differs: k (total tiles) vs PIPE (3 stages).
  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K

  // ---- Step 3b: Prefetch phase — fill pipeline stages before MMA starts ----
  //
  // Before the MMA loop begins, we need data already in shared memory.
  // We launch K_PIPE_MAX - 1 = 2 async copies to fill pipeline stages 0 and 1.
  //
  // cp_async_fence() acts as a barrier for async copies — it ensures all cp.async
  // instructions issued before the fence complete before any after the fence.
  // This is needed because cp.async is non-blocking; the GPU schedules the actual
  // memory transfer independently of the issuing thread.
  //
  // k_tile_count tracks remaining K-tiles to process (decremented as we schedule loads).
  // k_tile_next tracks which K-tile in gmem to load next.
  //
  // After this loop:
  //   - pipe[0] and pipe[1] have async copies in flight
  //   - The main loop will wait for pipe[0], process it, and keep the pipeline full

  auto K_PIPE_MAX = size<3>(tAsA);   // = bP = 3
  int k_tile_count = size<3>(tAgA);  // total K-tiles
  int k_tile_next  = 0;

  CUTE_UNROLL
  for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe) {
    copy(g2s_copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
    copy(g2s_copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
    cp_async_fence();
    --k_tile_count;
    if (k_tile_count > 0) { ++k_tile_next; }
  }

  // ---- Step 4: TiledMMA setup and register allocation ----
  //
  // ThrMMA (mma.get_thread_slice) gives this specific thread's view of the TiledMMA.
  //
  // Key difference from the non-pipelined version:
  //   - partition_fragment_A/B is used instead of make_fragment_A/B.
  //   - partition_fragment_A takes a *slice* of smem (sA(_,_,0)) — i.e., one pipe stage.
  //     This is because make_fragment_A/B expects a 2D smem view, but our smem is now 3D.
  //     Slicing with (_,_,0) gives the 2D view for one pipeline stage.
  //   - partition_fragment_A allocates registers matching the MMA partition shape
  //     without needing to go through partition_A first (which would give a 3D view
  //     with the PIPE dimension, confusing the register allocation).
  //
  // tCrA, tCrB are register tensors organized as (MMA, MMA_M, MMA_K) where:
  //   - MMA mode: values that the MMA instruction processes simultaneously
  //   - MMA_M, MMA_K: logical coordinates within the MMA tile
  // The MMA atom processes K in sub-tiles of K_BLOCK_MAX iterations.

  ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);

  // Partition gmem for MMA output
  Tensor tCgC = thr_mma.partition_C(gC);                                // (MMA, MMA_M, MMA_N)

  // Allocate register fragments using partition_fragment_A/B.
  // We slice sA and sB to remove the PIPE dimension (take pipe stage 0).
  // This gives a 2D view that partition_fragment_A/B can work with.
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));  // (MMA, MMA_M, MMA_K) in regs
  Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));  // (MMA, MMA_N, MMA_K) in regs

  // Allocate the accumulator registers — same logical shape as the gmem output partition.
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA, MMA_M, MMA_N) accum

  // Zero the accumulators before the MMA loop.
  clear(tCrC);

  // ---- Step 4b: S2R (smem->register) copy setup ----
  //
  // Same make_tiled_copy_A/B pattern as the non-pipelined version, but now sA and sB
  // are 3D tensors (BLK_M, BLK_K, PIPE). The partition_S will preserve the PIPE dimension:
  //   tXsA: (CPY, MMA_M, MMA_K, PIPE) — 4D with PIPE dimension
  //   tXrA: (CPY, MMA_M, MMA_K) — 3D without PIPE (retile_D works on tCrA which is 3D)
  //
  // In the main loop, we'll slice tXsA along the PIPE dimension to select which
  // pipeline stage to read from: tXsA(_,_,_,smem_pipe_read).
  //
  // make_tiled_copy_A bridges the LDSM instruction with the MMA:
  //   - Takes an LDSM copy atom (SM75_U32x4_LDSM_N — the hardware instruction)
  //   - Takes the TiledMMA (which knows the thread-value layout)
  //   - Produces a TiledCopy whose thread mapping matches the MMA's expectations
  //
  // retile_D reshapes the MMA register fragment to match the LDSM destination layout.
  // No data movement — just a layout reinterpretation.

  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
  ThrCopy  thr_s2r_a   = s2r_copy_a.get_slice(threadIdx.x);
  Tensor tXsA = thr_s2r_a.partition_S(sA);                              // (CPY, MMA_M, MMA_K, PIPE)
  Tensor tXrA = thr_s2r_a.retile_D(tCrA);                              // (CPY, MMA_M, MMA_K)

  TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
  ThrCopy  thr_s2r_b   = s2r_copy_b.get_slice(threadIdx.x);
  Tensor tXsB = thr_s2r_b.partition_S(sB);                              // (CPY, MMA_N, MMA_K, PIPE)
  Tensor tXrB = thr_s2r_b.retile_D(tCrB);                              // (CPY, MMA_N, MMA_K)

  // ---- Step 5: Pipelined main loop ----
  //
  // This is the heart of the pipelined kernel. The loop implements a circular buffer
  // over shared memory pipeline stages:
  //
  //   smem_pipe_read:  which pipe stage we're currently reading from (for smem->regs)
  //   smem_pipe_write: which pipe stage we're currently writing to (for gmem->smem)
  //
  // The pipeline operates as a producer-consumer system:
  //   Producer: cp.async loads data from gmem into smem at smem_pipe_write
  //   Consumer: LDSM loads data from smem at smem_pipe_read into registers, then MMA
  //
  // After each K-tile is processed:
  //   - smem_pipe_write takes the value of smem_pipe_read (the stage MMA just finished with
  //     is now free to be overwritten)
  //   - smem_pipe_read advances circularly: (read + 1) % K_PIPE_MAX
  //
  // The loop condition k_tile_count > -(K_PIPE_MAX - 1) accounts for the drain phase:
  //   After the last gmem load is issued, we still need K_PIPE_MAX - 1 more iterations
  //   to process the remaining pipeline stages. The count goes negative during drain.
  //
  // K_BLOCK_MAX is the number of register sub-tiles within each smem K-tile.
  //   This corresponds to the MMA_K dimension of tCrA/tCrB. For our configuration
  //   (BLK_K=64, MMA tile K=16), K_BLOCK_MAX = 4.
  //   The inner loop over k_block further overlaps smem->regs with MMA:
  //     - At k_block == K_BLOCK_MAX - 1: wait for the next pipe stage, update smem pointers
  //     - At k_block == 0: issue async gmem->smem copy for the next K-tile
  //     - Load k_block+1 while computing on k_block (double-buffering within registers)

  int smem_pipe_read  = 0;
  int smem_pipe_write = K_PIPE_MAX - 1;

  // Pipe slice: tXsA_p and tXsB_p point to the current pipe stage for reading.
  // Updated each iteration to select the correct stage from the 4D tensor.
  Tensor tXsA_p = tXsA(_,_,_,smem_pipe_read);
  Tensor tXsB_p = tXsB(_,_,_,smem_pipe_read);

  // Number of register sub-tiles in the K dimension.
  auto K_BLOCK_MAX = size<2>(tCrA);   // = 4

  // Prefetch the first register block from the first pipeline stage.
  // cp_async_wait<K_PIPE_MAX - 2> waits until at least K_PIPE_MAX - 2 = 1 async
  // copies have completed. Since we prefetched 2 stages, at least stage 0 is ready.
  // __syncthreads() ensures all threads in the block see the completed smem writes.
  if (K_BLOCK_MAX > 1) {
    cp_async_wait<K_PIPE_MAX - 2>();
    __syncthreads();
    copy(s2r_atom_a, tXsA_p(_,_,Int<0>{}), tXrA(_,_,Int<0>{}));
    copy(s2r_atom_b, tXsB_p(_,_,Int<0>{}), tXrB(_,_,Int<0>{}));
  }

  // CUTE_NO_UNROLL prevents the compiler from unrolling the outer while loop.
  // This is intentional: the loop body is already unrolled (CUTE_UNROLL on k_block),
  // and unrolling the outer loop would exponentially increase code size for large K.
  CUTE_NO_UNROLL
  while (k_tile_count > -(K_PIPE_MAX - 1))
  {
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
    {
      // On the last k_block of the inner loop:
      //   - The current smem pipe stage is about to be fully consumed.
      //   - Prepare for the next pipe stage: update smem pointers and wait for it.
      if (k_block == K_BLOCK_MAX - 1)
      {
        // Point to the next pipe stage we'll read from.
        tXsA_p = tXsA(_,_,_,smem_pipe_read);
        tXsB_p = tXsB(_,_,_,smem_pipe_read);

        // Wait until the async copy into this pipe stage is complete.
        // cp_async_wait<N> blocks until at most N async copies are still pending.
        // K_PIPE_MAX - 2 = 1 means "wait until at most 1 async copy is in flight."
        // __syncthreads() ensures all threads see the same shared memory state.
        cp_async_wait<K_PIPE_MAX - 2>();
        __syncthreads();
      }

      // Load the NEXT k_block's data from smem to registers while we compute on the current.
      // k_block_next wraps around via modular arithmetic (Int<1>{} ensures compile-time eval).
      // This is the register-level double-buffering: load k_block+1 while MMA works on k_block.
      auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;      // static
      copy(s2r_atom_a, tXsA_p(_,_,k_block_next), tXrA(_,_,k_block_next));
      copy(s2r_atom_b, tXsB_p(_,_,k_block_next), tXrB(_,_,k_block_next));

      // On the first k_block of the inner loop:
      //   - Issue the async gmem->smem copy for the NEXT K-tile.
      //   - This copy will write to smem_pipe_write (the pipe stage MMA just finished with).
      //   - Advance pipe indices for the next outer loop iteration.
      if (k_block == 0)
      {
        copy(g2s_copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
        copy(g2s_copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
        cp_async_fence();

        // Advance the gmem tile index.
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }

        // Rotate pipe indices circularly.
        // write takes read's slot (MMA just finished consuming it, so it's free).
        // read advances to the next stage in the circular buffer.
        smem_pipe_write = smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == K_PIPE_MAX - 1) ? 0 : smem_pipe_read + 1;
      }

      // Tensor core MMA: tCrC += tCrA[:,:,k_block] * tCrB[:,:,k_block]
      // This operates on the register fragment for the current k_block.
      gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
    }
  }

  // ---- Step 6: Epilogue ----
  //
  // Write accumulators to global memory: C = alpha * accum + beta * C
  // axpby does: dst = alpha * src + beta * dst (element-wise)

  axpby(alpha, tCrC, beta, tCgC);
}

// ================================================================================================
// Host Function — configure and launch the kernel
// ================================================================================================
//
// This function sets up all the parameters for the pipelined BF16 GEMM:
//   - Problem shape and strides (dynamic, from arguments)
//   - CTA tile sizes (static: 128 x 128 x 64)
//   - Smem layouts (static, swizzled with Swizzle<3,3,3>, 3D with PIPE dimension)
//   - TiledCopy for async gmem->smem (static, cp.async 128-bit)
//   - TiledMMA (static, wraps SM80 BF16 tensor core atom)
//   - S2R copy atoms (static, SM75_U32x4_LDSM_N)
//
// Key changes from the non-pipelined host function:
//   - bP = Int<3>{} added for pipeline depth
//   - Smem layouts are 3D: tile_to_shape(swizzle_atom, make_shape(bM, bK, bP))
//   - Thread layouts (tA, tB) replaced by TiledCopy (g2s_copy_a, g2s_copy_b)
//   - S2R atoms are constructed locally (not template parameters of the host function)
//   - Dynamic shared memory size computed and set via cudaFuncSetAttribute
//   - L1 cache carveout set to 100% shared memory (maximizes smem capacity)

template <class Alpha, class Beta>
void
bf16_gemm_tn(int m, int n, int k,
             Alpha alpha,
             bf16_t const* A, int ldA,
             bf16_t const* B, int ldB,
             Beta beta,
             float* C, int ldC,
             cudaStream_t stream = 0)
{
  using namespace cute;

  // Problem shape (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                                // (M, N, K)

  // TN strides (mixed static/dynamic)
  // A: (M, K) row-major — stride (ldA, 1)
  // B: (N, K) K-contiguous — stride (ldB, 1)
  // C: (M, N) column-major — stride (1, ldC)
  auto dA = make_stride(ldA, Int<1>{});
  auto dB = make_stride(ldB, Int<1>{});
  auto dC = make_stride(Int<1>{}, ldC);

  // CTA tile sizes (static)
  // bK=64 required by the swizzle pattern: the Swizzle<3,3,3> base layout has
  // K-dimension 64, so tile_to_shape needs bK to be a multiple of 64.
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<64>{};
  auto cta_tiler = make_shape(bM, bN, bK);                              // (128, 128, 64)

  // Pipeline depth (static)
  // 3 stages: one being consumed by MMA, one being loaded by cp.async, one buffer.
  // This is a tunable parameter; 3 is a good balance between overlap and smem usage.
  auto bP = Int<3>{};

  // Smem layouts (static, swizzled, 3D with pipeline dimension)
  //
  // XOR swizzle eliminates shared memory bank conflicts.
  // Swizzle<3,3,3>: 3-bit XOR, base position 3, shift 3.
  //
  // The base layout is 8 x (8 x 8) with strides (8, (1, 64)):
  //   - Outer dimension: 8 rows of swizzle tiles
  //   - Inner (8, 8): 8 rows x 8 columns per tile, column-major within each tile
  //   - Stride (1, 64): column-major inner, 64-element gap between inner tile groups
  //
  // tile_to_shape now takes a 3D shape: (bM, bK, bP) for A and (bN, bK, bP) for B.
  // This creates 3 back-to-back copies of the 2D swizzled layout in shared memory.
  // The PIPE dimension is contiguous, so pipe stage p starts at offset p * BLK_M * BLK_K
  // (adjusted for swizzle padding).

  auto swizzle_atom = composition(
      Swizzle<3, 3, 3>{},
      Layout<Shape <_8, Shape<_8, _8>>,
             Stride<_8, Stride<_1, _64>>>{});

  auto sA = tile_to_shape(swizzle_atom, make_shape(bM, bK, bP));         // (128, 64, 3) swizzled
  auto sB = tile_to_shape(swizzle_atom, make_shape(bN, bK, bP));         // (128, 64, 3) swizzled
  auto sC = make_layout(make_shape(bM, bN));                             // (128, 128) — unused in kernel

  // G2S (gmem->smem) TiledCopy (static)
  //
  // Replaces the synchronous thread layouts (tA, tB) from the non-pipelined version.
  //
  // Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, bf16_t>:
  //   - SM80_CP_ASYNC_CACHEALWAYS: async copy from global to shared memory, always cache in L2
  //   - uint128_t: each copy operation moves 128 bits = 8 x bf16 values at once
  //   - bf16_t: the element type
  //
  // Thread layout Layout<Shape<_16, _8>, Stride<_8, _1>>{}:
  //   - 16 threads in M dimension, 8 threads in K dimension = 128 threads total
  //   - Stride <_8, _1>: K-contiguous (threads are packed along K)
  //
  // Value layout Layout<Shape<_1, _8>>{}:
  //   - Each thread copies 1 value in M, 8 values in K per instruction
  //   - This matches the uint128_t atom: 8 x bf16 = 128 bits
  //
  // Total per-thread copy: (128/16, 64/(8*8)) = (8, 1) elements per K-tile.
  // Wait, that's not quite right: the TiledCopy tiles the copy across the full shape,
  // so each thread handles a subset of (BLK_M, BLK_K) determined by the tile.

  auto g2s_copy_a = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, bf16_t>{},
      Layout<Shape<_16, _8>, Stride<_8, _1>>{},
      Layout<Shape<_1, _8>>{});

  auto g2s_copy_b = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, bf16_t>{},
      Layout<Shape<_16, _8>, Stride<_8, _1>>{},
      Layout<Shape<_1, _8>>{});

  // TiledMMA (static)
  //
  // Atom: SM80_16x8x16_F32BF16BF16F32_TN — 16x8x16 BF16*BF16->F32, 32 threads
  // Atom layout: 2x2 in (M, N) -> 128 threads total (4 warps)
  //
  // Tile<_32, _32, _16> override expands the MMA tile to (32, 32, 16):
  //   - N expands from 16 (2 atoms of 8) to 32 (4 atoms of 8)
  //   - This gives each thread 4 values in the B partition instead of 2
  //   - Required for SM75_U32x4_LDSM_N which needs 4 values per thread
  TiledMMA mma = make_tiled_mma(
      SM80_16x8x16_F32BF16BF16F32_TN{},
      Layout<Shape<_2, _2>>{},
      Tile<_32, _32, _16>{});

  static_assert(decltype(size(mma))::value == 128, "Expected 128 threads");

  // S2R (smem->register) copy atoms (static)
  //
  // SM75_U32x4_LDSM_N wraps the ldmatrix.sync.aligned.x4.m8n8.shared.b16 PTX instruction.
  // One warp (32 threads) cooperatively loads a 8x8 matrix of 16-bit values (128 bytes)
  // from smem into registers. Each thread receives 4 x 32-bit = 128 bits.
  //
  // _N suffix: normal (non-transposing) layout. Our smem is K-contiguous, and the
  // MMA atom (SM80_16x8x16_F32BF16BF16F32_TN) expects TN layout, so no transpose needed.
  //
  // These are constructed inside the host function (not as template parameters)
  // because the host function template only needs Alpha and Beta.

  Copy_Atom<SM75_U32x4_LDSM_N, bf16_t> s2r_atom_a;
  Copy_Atom<SM75_U32x4_LDSM_N, bf16_t> s2r_atom_b;

  // Grid and block dimensions
  dim3 dimBlock(size(mma));
  dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));

  // Compute shared memory requirement.
  // The SharedStorage struct sizes itself based on the smem layouts, which now include
  // the PIPE dimension. For (128, 64, 3) bf16 swizzled layout, this is approximately
  // 128 * 64 * 3 * 2 bytes * 2 matrices = ~96 KB. The exact size may be larger due
  // to swizzle padding.
  int smem_size = int(sizeof(SharedStorage<bf16_t, bf16_t, decltype(sA), decltype(sB)>));

  // Get the kernel function pointer for cudaFuncSetAttribute.
  // This is needed because cudaFuncSetAttribute requires a function pointer, not a
  // triple-chevron launch. We must spell out all template arguments explicitly.
  auto kernel_fptr = bf16_gemm_device<
      decltype(prob_shape), decltype(cta_tiler),
      bf16_t, decltype(dA), decltype(sA), decltype(g2s_copy_a), decltype(s2r_atom_a),
      bf16_t, decltype(dB), decltype(sB), decltype(g2s_copy_b), decltype(s2r_atom_b),
      float, decltype(dC), decltype(sC), decltype(mma),
      Alpha, Beta>;

  // Set the maximum dynamic shared memory size for this kernel.
  // Required when shared memory exceeds the default (usually 48 KB).
  // The driver uses this hint to reserve enough shared memory for the kernel.
  cudaFuncSetAttribute(
      kernel_fptr,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  // Set L1 cache carveout to 100% shared memory.
  // This tells the hardware to maximize shared memory capacity at the expense of L1 cache.
  // Critical for our kernel which uses ~96 KB of shared memory (well above the 48 KB default).
  cudaFuncSetAttribute(
      kernel_fptr,
      cudaFuncAttributePreferredSharedMemoryCarveout, 100);

  // Launch kernel with dynamic shared memory size.
  // The third argument to <<<>>> is the number of bytes of dynamic shared memory to allocate.
  // This is what the extern __shared__ char shared_memory[] in the kernel will receive.
  kernel_fptr<<<dimGrid, dimBlock, smem_size, stream>>>(
      prob_shape, cta_tiler,
      A, dA, sA, g2s_copy_a, s2r_atom_a,
      B, dB, sB, g2s_copy_b, s2r_atom_b,
      C, dC, sC, mma,
      alpha, beta);
}

// ================================================================================================
// Main — allocate, run, verify, benchmark
// ================================================================================================

// Run benchmark at a single size (no verification)
//
// No template parameters needed for bf16_gemm_tn — the S2R atoms are created internally.
// This simplifies the benchmark code compared to the non-pipelined version.

void benchmark_size(int m, int n, int k,
                    float alpha, float beta,
                    cudaStream_t stream)
{
  using namespace cute;

  assert(m % 128 == 0 && n % 128 == 0 && k % 64 == 0);

  int ldA = k, ldB = k, ldC = m;

  thrust::device_vector<bf16_t> d_A(m * k), d_B(n * k);
  thrust::device_vector<float>  d_C(m * n);
  thrust::host_vector<bf16_t> h_A(m * k), h_B(n * k);
  for (int i = 0; i < m * k; ++i) h_A[i] = static_cast<bf16_t>(2.0 * (rand() / double(RAND_MAX)) - 1.0);
  for (int i = 0; i < n * k; ++i) h_B[i] = static_cast<bf16_t>(2.0 * (rand() / double(RAND_MAX)) - 1.0);
  d_A = h_A; d_B = h_B;

  const int timing_iterations = 100;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warmup
  bf16_gemm_tn(m, n, k, alpha,
               d_A.data().get(), ldA,
               d_B.data().get(), ldB,
               beta,
               d_C.data().get(), ldC, stream);
  CUTE_CHECK_LAST();

  cudaEventRecord(start);
  for (int i = 0; i < timing_iterations; ++i) {
    bf16_gemm_tn(m, n, k, alpha,
                 d_A.data().get(), ldA,
                 d_B.data().get(), ldB,
                 beta,
                 d_C.data().get(), ldC, stream);
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

int main(int argc, char** argv)
{
  using namespace cute;

  printf("BF16 GEMM (SM80 tensor cores, swizzle+LDSM, cp.async 3-stage pipeline)\n\n");

  float alpha = 1.0f;
  float beta  = 0.0f;

  // ---- Verify correctness once at 1024^3 ----
  {
    int m = 1024, n = 1024, k = 1024;
    int ldA = k, ldB = k, ldC = m;

    thrust::host_vector<bf16_t> h_A(m * k), h_B(n * k);
    thrust::host_vector<float>  h_C(m * n);
    for (int i = 0; i < m * k; ++i) h_A[i] = static_cast<bf16_t>(2.0 * (rand() / double(RAND_MAX)) - 1.0);
    for (int i = 0; i < n * k; ++i) h_B[i] = static_cast<bf16_t>(2.0 * (rand() / double(RAND_MAX)) - 1.0);
    for (int i = 0; i < m * n; ++i) h_C[i] = -1.0f;

    thrust::device_vector<bf16_t> d_A = h_A, d_B = h_B;
    thrust::device_vector<float>  d_C = h_C;

    bf16_gemm_tn(m, n, k, alpha,
                 d_A.data().get(), ldA,
                 d_B.data().get(), ldB,
                 beta,
                 d_C.data().get(), ldC);
    CUTE_CHECK_LAST();

    thrust::host_vector<float> h_result = d_C;

    // CPU reference: C[m,n] = alpha * sum_k A[m,k] * B[n,k] + beta * C[m,n]
    thrust::host_vector<float> h_ref(m * n, 0.0f);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) {
        float sum = 0.0f;
        for (int l = 0; l < k; ++l)
          sum += float(h_A[i * k + l]) * float(h_B[j * k + l]);
        h_ref[i + j * ldC] = alpha * sum + beta * h_C[i + j * ldC];
      }

    float max_err = 0.0f;
    for (int i = 0; i < m * n; ++i)
      max_err = std::max(max_err, std::abs(h_result[i] - h_ref[i]));

    printf("Correctness (1024^3): max error %e -- %s\n\n", max_err, max_err < 0.01f ? "PASS" : "FAIL");
    if (max_err >= 0.01f) return 1;
  }

  // ---- Benchmark ----
  printf("Benchmark (100 iterations each):\n");
  benchmark_size(512,  512,  512,  alpha, beta, 0);
  benchmark_size(1024, 1024, 1024, alpha, beta, 0);
  benchmark_size(2048, 2048, 2048, alpha, beta, 0);
  benchmark_size(4096, 4096, 4096, alpha, beta, 0);

  return 0;
}
