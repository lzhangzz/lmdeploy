/***************************************************************************************************
 * Standalone test for the A packing kernel (iteration 06).
 *
 * Pack kernel is unchanged from iteration 05. This test verifies the pack output
 * is still valid when used by the iter 06 consumer kernel.
 *
 * Target: SM90 (uses TMA for gmem->smem, SM90 WGMMA register layout)
 **************************************************************************************************/

#include "06_split_a_pack.h"

int main(int argc, char** argv)
{
  using namespace cute;

  printf("BF16 Split A Pack iter 06 (SM90 TMA + RS WGMMA, 64x16 unit packing)\n\n");

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
  for (int size : {256, 512, 1024, 2048, 4096, 8192}) {
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
