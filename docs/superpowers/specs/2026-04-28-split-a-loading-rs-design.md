# Split A Loading RS WGMMA Design

Split sample 13 (BF16 GEMM, RS WGMMA) into two separate kernels: one that packs operand A into GMMA register layout, and one that runs the GEMM using the pre-packed A.

## Motivation

This is a POC for a mixed-precision GEMM pipeline. The full pipeline will:
1. **Offline:** Pack quantized weights through the RS WGMMA A data path (TMA g2s + S2R), producing GMMA register-layout data
2. **Runtime:** Load packed weights directly to registers, dequantize, feed to RS WGMMA

This POC validates the split in BF16 (no quantization): correctness of the packing format and the ability to run WGMMA from pre-packed data.

## Design Decision: Direct gmem Register Dump

The packed A data is stored as a flat register dump in gmem. Kernel 1 writes each thread's register values directly; Kernel 2 reads them back directly. No TMA or smem intermediate for packed A in Kernel 2.

**Why:** Simplest POC approach. The gmem layout is `[m_tile][k_tile][thread_offset + reg_idx]`, which is trivially indexable. Once correctness is verified, a TMA-based loading optimization can be added in a follow-up.

## Kernel 1: Operand A Loading and Packing

**File:** `cute-reference/mixed-gemm/bf16_gemm_sm90_split_a_pack.cu`

**Purpose:** Transform the full A matrix from gmem tensor layout to per-tile GMMA register layout, stored in a packed gmem buffer.

### Grid and Threading

- 1 CTA = 1 warpgroup (128 threads)
- Grid size: enough CTAs to cover all (M, K) tiles
- Grid-stride loop over (M, K) tiles
- Single thread (thread 0) issues TMA loads; all threads participate in S2R copy and gmem writes

### Per-Tile Pipeline

1. **TMA gmem→smem:** Thread 0 issues `copy(tma_a.with(barrier), gA_tile, sA)` — same TMA load as sample 13. Wait on barrier.
2. **S2R copy:** `copy(smem_tiled_copy_A, tCsA, tCrA)` — same S2R copy as sample 13. Transforms swizzled smem layout (`GMMA::Layout_K_SW128_Atom`) to GMMA register layout.
3. **Register dump to gmem:** Each thread writes its portion of `tCrA` to the packed buffer using direct stores. Offset: `(tile_m * num_k_tiles + tile_k) * packed_tile_size + thread_id * regs_per_thread`.

### Smem

- `GMMA::Layout_K_SW128_Atom<bf16_t>` for A, single pipeline stage (no pipelining needed — no overlap with WGMMA)
- No smem for output (registers write directly to gmem)

### Shared Constants

| Constant | Value |
|---|---|
| `bM` | 128 |
| `bK` | 64 |
| MMA atom | `SM90_64x256x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::K>` |
| S2R tiled copy | `make_tiled_copy_A(Copy_Atom<AutoVectorizingCopy, bf16_t>{}, mma)` |

### Packed A Gmem Layout

```
packed_A: [num_m_tiles][num_k_tiles][packed_tile_bytes]
  packed_tile_bytes = warpgroup_size * regs_per_thread * sizeof(bf16)
```

Each tile is a contiguous block. Within a tile, threads write sequentially (thread 0's values, then thread 1's, etc.).

## Kernel 2: RS WGMMA Using Packed Operand A

**File:** `cute-reference/mixed-gemm/bf16_gemm_sm90_split_a_wgmma.cu`

**Purpose:** Run the full GEMM `C = alpha * A_packed * B^T + beta * C` using the pre-packed A.

### Grid and Threading

- Same as sample 13: 384 threads = 128 TMA producer (WG2) + 256 consumer (WG0, WG1)
- Persistent scheduling: 1 CTA per SM, grid-stride loop over output tiles
- Register budget: `warpgroup_reg_alloc<232>()` for consumers, `warpgroup_reg_dealloc<40>()` for producer

### Producer (Warp Group 2)

- Issues TMA loads for **B only** — A is pre-packed, loaded by consumers directly from gmem
- Same pipeline: `PipelineTmaAsync<3>` for B
- Same persistent loop as sample 13

### Consumer Mainloop (Warp Groups 0 and 1)

For each output tile (m_tile, n_tile), iterate over k_tiles:

1. **Load packed A to registers:** Each thread reads its portion of `packed_A[m_tile][k_tile]` directly from gmem. Data goes into `tCrA` registers — identical layout to what Kernel 1 wrote. Vectorized loads for coalescing.
2. **Wait for B:** `pipeline.consumer_wait()` for B in smem
3. **Batched WGMMA:** Same as sample 13:
   ```
   for k_block in 0..3:
     warpgroup_arrive()
     gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block,read_stage), tCrC)
     warpgroup_commit_batch()
     warpgroup_wait<2>()
   ```
4. **Release B:** `pipeline.consumer_release()`

### Epilogue

Same as sample 13:
1. Alpha/beta scaling in F32
2. F32→BF16 conversion
3. STSM (register→smem) via `Copy_Atom<SM90_U16x8_STSM_T, bf16_t>{}`
4. Consumer-only sync via named barrier
5. TMA store (smem→gmem) by thread 0

### Key Difference from Sample 13

A is loaded from the packed gmem buffer (direct reads to registers) instead of from smem via TMA + S2R copy. The B pipeline and WGMMA computation are unchanged.

## Verification

- CPU reference BF16 GEMM computes expected C
- Run Kernel 1 to pack A, then Kernel 2 to compute C
- Compare output C against CPU reference with BF16 tolerance
- Test sizes: 1024x1024x1024, 2048x1024x512

## POC Scope and Limitations

- BF16 only — no quantization
- No performance tuning — correctness first
- Single-GPU
- Direct gmem loads for packed A (TMA optimization deferred)
