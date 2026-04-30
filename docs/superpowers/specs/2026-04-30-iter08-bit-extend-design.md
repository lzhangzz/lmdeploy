# Iteration 08: On-the-fly Bit-Extend Design

**Goal:** Validate the full quantization roundtrip — bit-extend UINT4 to BF16, pack as uint4, dequantize in registers before WGMMA.

**Architecture:** The pack kernel bit-truncates BF16 to uint4 using `Converter<uint16_t, uint4_t>::pack` (from `format.h`) and stores 1×uint32 per thread per k_block. The consumer bulk-copies the 4x-smaller packed buffer to smem, S2R loads 1×uint32, applies `cvt_bf16x8_u4` (lop3 + subtract 128) to produce 8×BF16, and feeds into WGMMA unchanged.

**Tech Stack:** CUDA 12+, CuTe, CUTLASS pipeline primitives, SM90 WGMMA, LOP3 for fast I2F

---

## Data Flow

```
Host: generate UINT4 [0,15] → bit-extend to BF16 → feed as "original A"

Pack kernel:
  gmem A (BF16) → TMA → smem (swizzled) → S2R to registers (BF16)
  → Converter::pack (BF16 → uint4, 8 values into 1×uint32)
  → store 1×uint32 to gmem packed_A

Consumer kernel:
  gmem packed_A (uint4) → bulk copy → smem (packed)
  → S2R: load 1×uint32 per thread per k_block
  → cvt_bf16x8_u4<true> (lop3.b32 × 4 + subtract 128)
  → copy BF16 result into tCrA registers
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

The packed smem layout per warpgroup region changes from `(REG=8, THREAD=128, K_BLOCK=4)` with strides `(1, 8, 1024)` in BF16 to `(THREAD=128, K_BLOCK=4)` with strides `(1, 128)` in uint32. Each thread loads 1 contiguous uint32 per k_block.

## Pack Kernel Changes

Only the register-to-gmem store changes. Everything upstream (TMA load, smem layout, S2R copy) is identical to iter 04.

**Register-to-gmem store (per thread, per k_block):**
1. Reinterpret 8 BF16 registers as `Array<uint16_t, 8>`
2. Apply `Converter<uint16_t, uint4_t>::pack()` → `Array<uint4_t, 8>` = 4 bytes = 1×uint32
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
    uint32_t packed = smem_base[local_tid + kb * 128];  // 1×uint32
    // packed contains 8 uint4 values in Converter::pack format
    ...
};
```

### Dequantization in Registers

Apply `cvt_bf16x8_u4<true>` (from `quantization.h` L172-209) to convert the packed uint4 to BF16:

```cpp
Array<uint4_t, 8> packed_u4 = reinterpret_cast<Array<uint4_t, 8>&>(packed);
Array<nv_bfloat16, 8> dequant = cvt_bf16x8_u4<true>(packed_u4);
// dequant now holds 8 BF16 values with zero-point subtracted
// Copy into tCrA registers for k_block kb
copy(AutoVectorizingCopy{},
     make_tensor(make_bfloat16_ptr(dequant.data()), make_shape(Int<8>{})),
     make_tensor(tCrA.data() + kb * 8, make_shape(Int<8>{})));
```

The `cvt_bf16x8_u4<true>` function:
1. Takes 4 bytes (`uint32_t`) containing 8 uint4 nibbles
2. Uses 4 `lop3.b32` instructions (one per 2 BF16 output values) with TEMPLATE=0x43004300, MASK=0x000f000f, immLut for fast integer-to-float conversion
3. Subtracts `nv_bfloat16(128.f)` from each of the 8 output values (zero-point subtraction)

### Bulk Copy

The bulk copy now transfers 4× less data per stage:
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

## Order-Shuffle Cancellation

`Converter::pack` rearranges 8 nibbles into a specific bit pattern via `__byte_perm(0x5140)`:
- Input: bytes `[v0, v1, v2, v3, v4, v5, v6, v7]` (each byte holds a 4-bit value)
- After OR-shift: bytes are interleaved into `ui[0]` and `ui[1]`
- `__byte_perm(0x5140)` selects 4 bytes: positions 0, 1, 4, 5 from the 8-byte input
- Output: 1×uint32 with nibbles in a specific order

`cvt_bf16x8_u4` reads the same 4 bytes as `uint32_t` and extracts nibbles via shifts:
- `i4s` = raw uint32 (same layout as pack output)
- `i4s_4 = i4s >> 4`, `i4s_8 = i4s >> 8`, `i4s_12 = i4s >> 12`
- Each shift positions 2 nibbles for the lop3.b32 to convert to BF16

The pack shuffle and the cvt extraction are designed to be inverses — no explicit unshuffle is needed between them. This is confirmed by the production usage in turbomind where these two functions are used together.

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
