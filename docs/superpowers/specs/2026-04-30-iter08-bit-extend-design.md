# Iteration 08: On-the-fly Bit-Extend Design

**Goal:** Validate the full quantization roundtrip — bit-extend UINT4 to BF16, pack as uint4, dequantize in registers before WGMMA.

**Architecture:** The pack kernel bit-truncates BF16 to uint4 via `pack_bf16_to_u4` and stores 1×uint32 per thread per k_block. The consumer bulk-copies the 4x-smaller packed buffer to smem, S2R loads 1×uint32, applies `unpack_u4_to_bf16` (lop3 + subtract 128) to produce 8×BF16, and feeds into WGMMA unchanged.

**Tech Stack:** CUDA 12+, CuTe, CUTLASS pipeline primitives, SM90 WGMMA, LOP3 for fast I2F

---

## Standalone Pack/Unpack Functions

These are extracted from `src/turbomind/kernels/gemm/format.h` and
`src/turbomind/kernels/attention/quantization.h`, rewritten without `Array<T>`.

### `pack_bf16_to_u4` — Pack 8 BF16 values into 1×uint32

Takes 8 BF16 values (each holding a quantized uint4 value in [0,15]) and packs
them into a single uint32. Each value's low 4 bits are extracted and interleaved
via OR-shift + byte_perm.

```cpp
// Pack 8 BF16 values into 1 uint32 containing 8 uint4 nibbles.
// Each input value must be in [0, 15].
__device__ uint32_t
pack_bf16_to_u4(const uint16_t v[8])
{
    uint32_t w0 = uint32_t(v[0] & 0xF)
                | (uint32_t(v[1] & 0xF) << 8)
                | (uint32_t(v[2] & 0xF) << 16)
                | (uint32_t(v[3] & 0xF) << 24);
    uint32_t w1 = uint32_t(v[4] & 0xF)
                | (uint32_t(v[5] & 0xF) << 8)
                | (uint32_t(v[6] & 0xF) << 16)
                | (uint32_t(v[7] & 0xF) << 24);
    w0 |= (w0 >> 12);
    w1 |= (w1 >> 12);
    return __byte_perm(w0, w1, 0x5140);
}
```

### `unpack_u4_to_bf16` — Unpack 1×uint32 to 8 BF16 values

Inverse of `pack_bf16_to_u4`. Uses 4 `lop3.b32` instructions for fast integer-
to-float conversion (each lop3 produces 2 BF16 values), then subtracts the
implicit zero point (128).

```cpp
// Unpack 1 uint32 (8 uint4 nibbles) to 8 BF16 values via fast I2F.
// Subtracts implicit zero point 128 from each output value.
__device__ void
unpack_u4_to_bf16(uint32_t packed, nv_bfloat16 out[8])
{
    static constexpr uint32_t TEMPLATE = 0x43004300;  // bf162(128, 128)
    static constexpr uint32_t MASK     = 0x000f000f;
    static constexpr uint32_t immLut   = (0xf0 & 0xcc) | 0xaa;

    uint32_t* h = reinterpret_cast<uint32_t*>(out);
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;" : "=r"(h[0]) : "r"(packed),       "n"(MASK), "n"(TEMPLATE), "n"(immLut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;" : "=r"(h[1]) : "r"(packed >> 4),  "n"(MASK), "n"(TEMPLATE), "n"(immLut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;" : "=r"(h[2]) : "r"(packed >> 8),  "n"(MASK), "n"(TEMPLATE), "n"(immLut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;" : "=r"(h[3]) : "r"(packed >> 12), "n"(MASK), "n"(TEMPLATE), "n"(immLut));

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[i] -= nv_bfloat16(128.f);
    }
}
```

### Order-Shuffle Cancellation

The pack function rearranges 8 nibbles into a specific bit pattern via
`__byte_perm(0x5140)`. The unpack function reads the same 4 bytes and extracts
nibbles via right-shifts (`>> 0, >> 4, >> 8, >> 12`), positioning exactly 2
nibbles per lop3 call. The byte_perm shuffle in pack and the shift-based
extraction in unpack are designed to be inverses — no explicit unshuffle is
needed. This is confirmed by production usage in turbomind where the original
`Converter::pack` and `cvt_bf16x8_u4` are used together.

---

## Data Flow

```
Host: generate UINT4 [0,15] → bit-extend to BF16 → feed as "original A"

Pack kernel:
  gmem A (BF16) → TMA → smem (swizzled) → S2R to registers (BF16)
  → pack_bf16_to_u4 (8 BF16 → 1×uint32)
  → store 1×uint32 to gmem packed_A

Consumer kernel:
  gmem packed_A (uint4) → bulk copy → smem (packed)
  → S2R: load 1×uint32 per thread per k_block
  → unpack_u4_to_bf16 (1×uint32 → 8×BF16, zero-point subtracted)
  → copy BF16 into tCrA registers
  → WGMMA (unchanged)
```

## Packed Format Change

**Current (iter 07):**
- Per thread per k_block: 8 BF16 = 16 bytes (1×128-bit store)
- Per warpgroup per tile: 4 × 128 × 16 = 8192 bytes

**New (iter 08):**
- Per thread per k_block: 8 uint4 packed into 1×uint32 = 4 bytes
- Per warpgroup per tile: 4 × 128 × 4 = 2048 bytes
- Packed buffer is **4× smaller**

Packed smem layout per warpgroup: `(THREAD=128, K_BLOCK=4)` with strides
`(1, 128)` in uint32. Each thread loads 1 contiguous uint32 per k_block.

## Pack Kernel Changes

Only the register-to-gmem store changes. Everything upstream (TMA load, smem
layout, S2R copy) is identical to iter 04.

**Register-to-gmem store (per thread, per k_block):**
1. Reinterpret 8 BF16 registers as `uint16_t[8]`
2. Call `pack_bf16_to_u4()` → 1×uint32
3. Store 1×uint32 to packed gmem via `AutoVectorizingCopy`

**Packed gmem tensor per warpgroup:**
```
Shape: (THREAD=128, K_BLOCK=4)
Strides: (1, 128)
Element: uint32_t
Total: 2048 bytes
```

Each thread stores 1 uint32 per k_block at offset `wg_offset + local_tid + kb * 128`.

## Consumer Kernel Changes

### S2R Load

Load 1×uint32 from smem instead of 8×BF16 (32 bits vs 128 bits):
```cpp
auto load_k_block = [&](int kb, int stage) {
    uint32_t* smem_base = reinterpret_cast<uint32_t*>(
        smem.A.begin() + stage * a_stage_bytes + wg_id * 512);  // 512 = 128 threads × 4 bytes
    uint32_t packed = smem_base[local_tid + kb * 128];
    nv_bfloat16 dequant[8];
    unpack_u4_to_bf16(packed, dequant);
    // Copy dequant into tCrA registers for k_block kb
    copy(AutoVectorizingCopy{},
         make_tensor(make_bfloat16_ptr(dequant), make_shape(Int<8>{})),
         make_tensor(tCrA.data() + kb * 8, make_shape(Int<8>{})));
};
```

### Bulk Copy

The bulk copy transfers 4× less data per stage:
- `a_stage_bytes` = 2048 (down from 8192)
- `tma_transaction_bytes` = a_stage_bytes + B TMA transaction size
- Smem A array size shrinks accordingly

### What Doesn't Change

- B pipeline (BF16 via TMA, same smem layout, same S2R)
- WGMMA instruction (`SM90_64x256x16_F32BF16BF16_RS`)
- Producer/consumer pipeline structure (3 stages, same barrier coordination)
- k_block interleaving (iter 05 optimization)
- Deferred TMA store wait (iter 07 optimization)
- Host function signature (M, K, ldA; packed_A is now uint4 buffer)

## Test Strategy

1. **Host generates UINT4 [0,15]** values, bit-extends to BF16 (cast uint16 → BF16)
2. **Pack kernel**: packs BF16 → uint4, stores to gmem
3. **Consumer kernel**: loads uint4 from gmem, dequantizes to BF16, runs WGMMA with B (BF16)
4. **Reference**: run cuBLAS with the bit-extended BF16 A matrix to get expected output
5. **Compare**: consumer output vs cuBLAS output, max error should match iter 07 levels

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `cute-reference/mixed-gemm/08_split_a_pack.h` | Create | Pack kernel + host function with uint4 packing |
| `cute-reference/mixed-gemm/08_bf16_gemm_sm90_split_a_pack.cu` | Create | Pack test (bit-extend → pack → verify packed values) |
| `cute-reference/mixed-gemm/08_bf16_gemm_sm90_split_a_wgmma.cu` | Create | Consumer with uint4 dequant in registers |
| `cute-reference/mixed-gemm/CMakeLists.txt` | Modify | Add 08 targets |
