# Iteration 04: Reduce Packing Unit to 64×16

**Goal:** Reduce the packed A operand unit from (128, 64) to (64, 16) — one WGMMA instruction's A operand — for maximum composability in the production pipeline.

**Approach:** Keep the (128, 64) processing tile in the pack kernel (same TMA+S2R pipeline), but restructure the output into 8 independent (64, 16) units. Consumer bulk copy is unchanged; only the smem-to-register mapping is updated.

---

## Packed Format

### Per-unit layout (64×16 = 1024 bf16)

```
Shape <_8, _128>        // (REG, THREAD)
Stride<_1,   _8>
```

128 threads (1 warpgroup), each holding 8 contiguous bf16 (128-bit vector).

### Full packed tensor

```cpp
auto packed_shape = make_shape(
    Int<8>{},                              // REG
    Int<128>{},                            // THREAD
    make_shape(Int<4>{}, K_tiles),         // K: (k_block, k_tile) → k_unit = k_tile*4 + k_block
    make_shape(Int<2>{}, M_tiles)          // M: (m_wg, m_tile)   → m_unit = m_tile*2 + m_wg
);

auto packed_stride = make_stride(
    Int<1>{},                              // REG
    Int<8>{},                              // THREAD
    make_stride(Int<1024>{}, Int<4096>{}), // K
    make_stride(4096 * K_tiles, 8192 * K_tiles)  // M
);
```

Offset formula:
```
m_tile*(8192*K_tiles) + m_wg*(4096*K_tiles) + k_tile*4096 + k_block*1024 + thread*8 + reg
```

Composite modes `(4, K_tiles)` and `(2, M_tiles)` flatten to `K_units = ceil_div(K, 16)` and `M_units = ceil_div(M, 64)`. The 128×64 tile is an addressing convenience, not a structural boundary.

### Within one (128, 64) tile

8 units ordered as:
```
unit_0: m_wg=0, k_block=0  |  1024 bf16
unit_1: m_wg=0, k_block=1  |  1024 bf16
unit_2: m_wg=0, k_block=2  |  1024 bf16
unit_3: m_wg=0, k_block=3  |  1024 bf16
unit_4: m_wg=1, k_block=0  |  1024 bf16
unit_5: m_wg=1, k_block=1  |  1024 bf16
unit_6: m_wg=1, k_block=2  |  1024 bf16
unit_7: m_wg=1, k_block=3  |  1024 bf16
```

Each warpgroup's 4 units are contiguous (4096 bf16). Total per tile: 8192 bf16 (same as iter 03).

---

## Pack Kernel Changes

Keep 256 threads and the (128, 64) TMA+S2R pipeline. Only the output write section changes.

**Current (iter 03):**
```cpp
Tensor gPacked = make_tensor(make_gmem_ptr(packed_A + linear_idx * 256 * regs_per_thread),
                              make_shape(Int<8>{}, Int<256>{}, Int<4>{}));
Tensor gP = gPacked(_, threadIdx.x, _);   // (8, 4)
```

**New (iter 04):**
```cpp
int wg_id = threadIdx.x / 128;
int local_tid = threadIdx.x % 128;

// Per-warpgroup region: (REG=8, THREAD=128, K_BLOCK=4) strides (1, 8, 1024)
Tensor gPacked_wg = make_tensor(
    make_gmem_ptr(packed_A + linear_idx * 8192 + wg_id * 4096),
    make_shape(Int<8>{}, Int<128>{}, Int<4>{}));
Tensor gP = gPacked_wg(_, local_tid, _);  // (8, 4) stride (1, 1024)
Tensor rA = make_tensor(tCrA.data(), make_shape(Int<8>{}, Int<4>{}));
copy(AutoVectorizingCopy{}, rA, gP);
```

Each thread writes to its warpgroup's contiguous 4096 bf16 region within the tile.

---

## Consumer Kernel Changes

The producer's bulk copy is unchanged (still copies 8192 bf16 per k_tile). Only the consumer S2R section changes.

**Current (iter 03):**
```cpp
Tensor sA_packed = make_tensor(
    make_smem_ptr(smem.A.begin() + read_stage * a_stage_elements),
    make_shape(Int<8>{}, Int<256>{}, Int<4>{}));
Tensor sP = sA_packed(_, threadIdx.x, _);  // (8, 4) stride (1, 2048)
```

**New (iter 04):**
```cpp
int wg_id = warp_group_idx;  // 0 or 1
int local_tid = threadIdx.x % 128;

Tensor sA_packed = make_tensor(
    make_smem_ptr(smem.A.begin() + read_stage * a_stage_elements + wg_id * 4096),
    make_shape(Int<8>{}, Int<128>{}, Int<4>{}));  // strides (1, 8, 1024)
Tensor sP = sA_packed(_, local_tid, _);  // (8, 4) stride (1, 1024)
Tensor rA = make_tensor(tCrA.data(), make_shape(Int<8>{}, Int<4>{}));
copy(AutoVectorizingCopy{}, sP, rA);
```

Each consumer warpgroup loads from its own 4096 bf16 region within the smem stage. The WGMMA loop, pipeline staging, B TMA, and epilogue are unchanged.

---

## Files

- `cute-reference/mixed-gemm/04_split_a_pack.h` — shared pack header (from `03_split_a_pack.h`, modify output section)
- `cute-reference/mixed-gemm/04_bf16_gemm_sm90_split_a_pack.cu` — pack test/benchmark
- `cute-reference/mixed-gemm/04_bf16_gemm_sm90_split_a_wgmma.cu` — consumer WGMMA (from iter 03, modify S2R section)
- `cute-reference/mixed-gemm/CMakeLists.txt` — add `04_*` targets

## Validation

- **Correctness:** Run pack then WGMMA, compare output against sample 13 reference. Test sizes 128×256×64 through 8192×8192×8192.
- **Performance:** Compare TFLOP/s against iter 03 and cuBLAS. Expect parity with iter 03 (same total data moved, same WGMMA schedule).
