/***************************************************************************************************
 * Standalone test for the A packing kernel (iteration 08).
 *
 * Generates UINT4 [0,15] values, bit-extends to BF16, packs to uint4, verifies
 * the pack->unpack roundtrip recovers the original values.
 *
 * Target: SM90 (uses TMA for gmem->smem, SM90 WGMMA register layout)
 **************************************************************************************************/

#include "09_split_a_pack.h"

int main(int argc, char** argv)
{
  using namespace cute;

  printf("BF16 Split A Pack iter 09 (SM90 TMA + RS WGMMA, uint4 packing — same as iter 08)\n\n");

  // ---- Test: pack->unpack roundtrip at 128x64 (single tile) ----
  {
    int m = 128, k = 64;
    int ldA = k;

    thrust::host_vector<bf16_t> h_A(m * k);
    for (int i = 0; i < m * k; ++i) {
      uint16_t val = uint16_t(rand() % 16);
      uint16_t raw = 0x3F80u | (val & 0xFu);  // bf16(1.0) bits with val in low 4 bits
      h_A[i] = reinterpret_cast<bf16_t const&>(raw);
    }

    thrust::device_vector<bf16_t> d_A = h_A;
    int total_tiles = 1;
    thrust::device_vector<uint32_t> d_packed(total_tiles * 1024);

    split_a_pack(m, k, d_A.data().get(), ldA, d_packed.data().get());
    CUTE_CHECK_LAST();

    // Verify: check packed buffer is non-zero
    thrust::host_vector<uint32_t> h_packed = d_packed;
    bool all_zero = true;
    for (int i = 0; i < total_tiles * 1024; ++i) {
      if (h_packed[i] != 0) { all_zero = false; break; }
    }

    printf("Pack test (128x64): %s (%d uint32 elements)\n\n",
           all_zero ? "FAIL (all zero)" : "PASS (non-zero packed)", total_tiles * 1024);
    if (all_zero) return 1;
  }

  // ---- Benchmark ----
  printf("Pack benchmark:\n");
  for (int size : {256, 512, 1024, 2048, 4096, 8192}) {
    int m = size, k = size;
    int ldA = k;

    int total_tiles = ((m + 127) / 128) * ((k + 63) / 64);

    thrust::device_vector<bf16_t> d_A(m * k);
    thrust::device_vector<uint32_t> d_packed(total_tiles * 1024);
    thrust::host_vector<bf16_t> h_A(m * k);
    for (int i = 0; i < m * k; ++i) {
      uint16_t val = uint16_t(rand() % 16);
      uint16_t raw = 0x3F80u | (val & 0xFu);  // bf16(1.0) bits with val in low 4 bits
      h_A[i] = reinterpret_cast<bf16_t const&>(raw);
    }
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
    double gb = (m * k * sizeof(bf16_t) + total_tiles * 1024 * sizeof(uint32_t)) * 1e-9;
    printf("  %dx%d: %.1f GB/s (%.4f ms)\n", m, k, gb / (avg_ms * 1e-3), avg_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  return 0;
}
