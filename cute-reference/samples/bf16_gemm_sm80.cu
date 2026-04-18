/***************************************************************************************************
 * BF16 GEMM using SM80 tensor cores with CuTe — Learning Sample
 *
 * Demonstrates:
 *   - CuTe tensors and layouts for GEMM
 *   - TiledMMA for SM80 BF16 tensor core operations
 *   - Register fragments and the MMA programming model
 *   - Simple (non-pipelined) GEMM mainloop
 *
 * C = alpha * A * B^T + beta * C
 *   A: bf16, M x K, row-major (TN layout)
 *   B: bf16, K x N, stored as (N, K) in CuTe with K-contiguous stride
 *   C: f32, M x N, column-major
 *
 * Target: SM80+ tensor cores (runs on SM90 L20Y GPU)
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
// Device Kernel
// ================================================================================================
//
// Each thread block computes a (BLK_M x BLK_N) tile of output C.
// The K dimension is processed in chunks of BLK_K.
//
// Data flow (per K-tile):
//   1. gmem A,B --> shared memory  (all threads cooperate via thread layout)
//   2. __syncthreads()
//   3. smem A,B --> registers      (MMA-partitioned copy)
//   4. tensor core MMA             (gemm on register fragments)
//   5. __syncthreads()
//
// After all K-tiles: accumulators --> gmem C (epilogue)

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
bf16_gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                 TA const* A, AStride dA, ASmemLayout sA_layout, AThreadLayout tA,
                 TB const* B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB,
                 TC      * C, CStride dC, CSmemLayout          , TiledMma mma,
                 Alpha alpha, Beta beta)
{
  using namespace cute;

  // ---- Preconditions ----
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)
  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);
  static_assert(is_static<AThreadLayout>::value);
  static_assert(is_static<BThreadLayout>::value);
  CUTE_STATIC_ASSERT_V(size(tA) == size(mma));                         // NumThreads
  CUTE_STATIC_ASSERT_V(size(tB) == size(mma));                         // NumThreads
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

  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M, K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N, K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M, N)

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});  // (BLK_M, BLK_K, k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});  // (BLK_N, BLK_K, k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});  // (BLK_M, BLK_N)

  // ---- Step 2: Shared memory tensors ----
  //
  // Static __shared__ arrays sized by cosize_v (number of elements the layout addresses).
  // Plain layouts — no swizzle, no padding. K-contiguous for efficient gmem loading.

  __shared__ TA smem_a[cosize_v<ASmemLayout>];
  __shared__ TB smem_b[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smem_a), sA_layout);            // (BLK_M, BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smem_b), sB_layout);            // (BLK_N, BLK_K)

  // ---- Step 3: gmem -> smem copy partitioning ----
  //
  // local_partition divides a tensor among threads using a thread layout.
  // Thread layout (32, 4) means 32 threads in the first mode, 4 in the second.
  // Each thread gets (BLK_M/32, BLK_K/4) elements to copy.

  Tensor tAgA = local_partition(gA, tA, threadIdx.x);                   // (THR_M, THR_K, k)
  Tensor tAsA = local_partition(sA, tA, threadIdx.x);                   // (THR_M, THR_K)
  Tensor tBgB = local_partition(gB, tB, threadIdx.x);                   // (THR_N, THR_K, k)
  Tensor tBsB = local_partition(sB, tB, threadIdx.x);                   // (THR_N, THR_K)

  // ---- Step 4: TiledMMA setup ----
  //
  // TiledMMA wraps a hardware MMA (tensor core) instruction and tiles it across threads.
  //
  // ThrMMA (mma.get_thread_slice) gives this specific thread's view:
  //   partition_A/B — which smem elements this thread reads for MMA
  //   partition_C   — which gmem output elements this thread writes
  //   make_fragment  — allocate register buffers matching the MMA's expected layout
  //
  // The MMA atom SM80_16x8x16_F32BF16BF16F32_TN computes a 16x8x16 (MxNxK) tile
  // of BF16*BF16 -> F32 using 32 threads (1 warp).
  // With Layout<Shape<_2,_2>>{}, we replicate 2x in M and N -> 128 threads (4 warps).

  ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);

  // Partition smem for MMA input
  Tensor tCsA = thr_mma.partition_A(sA);                                // (MMA, MMA_M, MMA_K)
  Tensor tCsB = thr_mma.partition_B(sB);                                // (MMA, MMA_N, MMA_K)

  // Partition gmem for MMA output
  Tensor tCgC = thr_mma.partition_C(gC);                                // (MMA, MMA_M, MMA_N)

  // Allocate register fragments.
  //
  // make_fragment_A/B creates a register tensor with the same logical shape as
  // the smem partition (tCsA/tCsB), but with a register layout that matches what
  // the MMA hardware instruction expects. This is the key correctness mechanism:
  //
  //   partition_A gives a smem layout — where data sits in shared memory
  //   make_fragment_A gives a register layout — which register holds which (m,k) value
  //
  // Both share the same logical coordinate space (MMA, MMA_M, MMA_K), so
  // copy(tCsA, tCrA) maps each logical element to the correct register.
  // The copy iterates over logical coordinates, not physical addresses —
  // it reads tCsA(v,m,k) from smem and writes to tCrA(v,m,k) in the register
  // that the MMA expects for that (m,k) position.
  //
  // Tradeoff: without a dedicated smem→regs copy atom (e.g. SM75_U32x4_LDSM_N),
  // the copy generates scalar loads. Correct but slow. The production path uses
  // make_tiled_copy_A(s2r_atom, mma) to vectorize this step.
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);                         // (MMA, MMA_M, MMA_K) in regs
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);                         // (MMA, MMA_N, MMA_K) in regs
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA, MMA_M, MMA_N) accum

  // Zero the accumulators
  clear(tCrC);

  // ---- Step 5: Main loop ----
  //
  // Simple non-pipelined loop over K-tiles:
  //   1. All threads copy gmem -> smem (via thread layout partitioning)
  //   2. __syncthreads() — wait for smem writes to complete
  //   3. Each thread copies its MMA-partitioned smem view -> registers
  //   4. Each thread issues tensor core gemm on registers
  //   5. __syncthreads() — wait for smem reads to complete (safe to overwrite next tile)

  auto K_TILE_MAX = size<2>(tAgA);

  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
  {
    // gmem -> smem
    copy(tAgA(_,_,k_tile), tAsA);
    copy(tBgB(_,_,k_tile), tBsB);
    __syncthreads();

    // smem -> registers (see make_fragment_A/B comment above for why this works)
    copy(tCsA, tCrA);
    copy(tCsB, tCrB);

    // Tensor core MMA: tCrC += tCrA * tCrB
    gemm(mma, tCrA, tCrB, tCrC);

    __syncthreads();
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
// This function sets up all the static (compile-time) parameters:
//   - Problem shape and strides (dynamic, from arguments)
//   - CTA tile sizes (static: 128 x 128 x 32)
//   - Smem layouts (static, plain)
//   - Thread layouts for copy (static)
//   - TiledMMA (static, wraps SM80 BF16 tensor core atom)
//
// Then computes grid dimensions and launches the device kernel.

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
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<32>{};
  auto cta_tiler = make_shape(bM, bN, bK);                              // (128, 128, 32)

  // Smem layouts (static, plain — no swizzle)
  // make_layout(shape) produces a layout with default contiguous strides.
  // (128, 32) -> stride (32, 1) — K-contiguous, efficient for row-major loading
  auto sA = make_layout(make_shape(bM, bK));                            // (128, 32) stride (32, 1)
  auto sB = make_layout(make_shape(bN, bK));                            // (128, 32) stride (32, 1)
  auto sC = make_layout(make_shape(bM, bN));                            // (128, 128) — unused in kernel

  // Thread layouts for gmem -> smem copy (static)
  // (32, 4) = 128 threads. Each thread copies (128/32, 32/4) = (4, 8) elements.
  auto tA = make_layout(make_shape(Int<32>{}, Int<4>{}));
  auto tB = make_layout(make_shape(Int<32>{}, Int<4>{}));

  // TiledMMA (static)
  // Atom: SM80_16x8x16_F32BF16BF16F32_TN — 16x8x16 BF16*BF16->F32, 32 threads
  // Atom layout: 2x2 in (M, N) -> 128 threads total (4 warps)
  TiledMMA mma = make_tiled_mma(
      SM80_16x8x16_F32BF16BF16F32_TN{},
      Layout<Shape<_2, _2>>{});

  static_assert(decltype(size(mma))::value == 128, "Expected 128 threads");

  // Grid and block dimensions
  dim3 dimBlock(size(mma));
  dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));

  // Launch kernel
  bf16_gemm_device<<<dimGrid, dimBlock, 0, stream>>>(
      prob_shape, cta_tiler,
      A, dA, sA, tA,
      B, dB, sB, tB,
      C, dC, sC, mma,
      alpha, beta);
}

// ================================================================================================
// Main — allocate, run, verify, benchmark
// ================================================================================================

int main(int argc, char** argv)
{
  int m = 1024;
  if (argc >= 2) sscanf(argv[1], "%d", &m);

  int n = 1024;
  if (argc >= 3) sscanf(argv[2], "%d", &n);

  int k = 1024;
  if (argc >= 4) sscanf(argv[3], "%d", &k);

  printf("BF16 GEMM (SM80 tensor cores): M=%d, N=%d, K=%d\n", m, n, k);

  // Alignment: M, N should be multiples of 128; K should be a multiple of 32
  assert(m % 128 == 0 && "M must be a multiple of 128");
  assert(n % 128 == 0 && "N must be a multiple of 128");
  assert(k % 32 == 0   && "K must be a multiple of 32");

  float alpha = 1.0f;
  float beta  = 0.0f;

  // TN layout leading dimensions
  int ldA = k;    // A is M x K row-major
  int ldB = k;    // B is N x K K-contiguous
  int ldC = m;    // C is M x N column-major

  // Allocate host tensors
  thrust::host_vector<bf16_t> h_A(m * k);
  thrust::host_vector<bf16_t> h_B(n * k);
  thrust::host_vector<float>  h_C(m * n);

  // Fill A, B with random bf16 in [-1, 1]; C with -1
  for (int i = 0; i < m * k; ++i) h_A[i] = static_cast<bf16_t>(2.0 * (rand() / double(RAND_MAX)) - 1.0);
  for (int i = 0; i < n * k; ++i) h_B[i] = static_cast<bf16_t>(2.0 * (rand() / double(RAND_MAX)) - 1.0);
  for (int i = 0; i < m * n; ++i) h_C[i] = -1.0f;

  // Allocate device tensors and copy data
  thrust::device_vector<bf16_t> d_A = h_A;
  thrust::device_vector<bf16_t> d_B = h_B;
  thrust::device_vector<float>  d_C = h_C;

  // ---- Run kernel ----
  bf16_gemm_tn(m, n, k, alpha,
               d_A.data().get(), ldA,
               d_B.data().get(), ldB,
               beta,
               d_C.data().get(), ldC);
  CUTE_CHECK_LAST();

  // Download result
  thrust::host_vector<float> h_result = d_C;

  // ---- CPU reference GEMM ----
  //
  // C[m, n] = alpha * sum_k A[m, k] * B[n, k] + beta * C[m, n]
  // A: row-major, A[m, k] = h_A[m * K + k]
  // B: K-contiguous, B[n, k] = h_B[n * K + k]
  // C: column-major, C[m, n] = h_ref[m + n * M]

  thrust::host_vector<float> h_ref(m * n, 0.0f);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int l = 0; l < k; ++l) {
        sum += float(h_A[i * k + l]) * float(h_B[j * k + l]);
      }
      h_ref[i + j * ldC] = alpha * sum + beta * h_C[i + j * ldC];
    }
  }

  // ---- Verify ----
  float max_err = 0.0f;
  for (int i = 0; i < m * n; ++i) {
    float err = std::abs(h_result[i] - h_ref[i]);
    if (err > max_err) max_err = err;
  }
  printf("Max error: %e\n", max_err);

  bool passed = (max_err < 0.01f);
  printf("%s\n", passed ? "PASS" : "FAIL");

  // ---- Benchmark ----
  const int timing_iterations = 100;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < timing_iterations; ++i) {
    bf16_gemm_tn(m, n, k, alpha,
                 d_A.data().get(), ldA,
                 d_B.data().get(), ldB,
                 beta,
                 d_C.data().get(), ldC);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float total_ms = 0.0f;
  cudaEventElapsedTime(&total_ms, start, stop);
  double avg_ms = total_ms / timing_iterations;
  double gflops = (2.0 * m * n * k) * 1e-9;
  printf("Performance: %.1f GFLOP/s (%.4f ms per GEMM)\n", gflops / (avg_ms * 1e-3), avg_ms);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return passed ? 0 : 1;
}
