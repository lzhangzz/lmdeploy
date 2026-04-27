# SM90 WGMMA RS Variant — Design Spec

**Goal:** Create sample 11, an educational variant of sample 09 that swaps the WGMMA atom from SS (both operands in shared memory) to RS (A in registers, B in shared memory). All other components — tile sizes, TMA load/store, PipelineTmaAsync, persistent scheduling, STSM epilogue — remain identical.

**Motivation:** Compare the RS mechanism against SS to understand the register-vs-descriptor trade-off. This is purely educational; RS is not expected to outperform SS.

## Architecture

Copy sample 09 → sample 11, then apply the minimal set of changes:

### 1. MMA atom swap (host function)

```cpp
// Old (SS):
SM90_64x256x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{}

// New (RS):
SM90_64x256x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::K>{}
```

The RS atom imposes `static_assert(tnspA == GMMA::Major::K)` — register-source A must be K-major. We already pass `GMMA::Major::K` for both operands, so no change needed.

### 2. Consumer setup: partition_A behavior changes (device kernel)

With SS, `thr_mma.partition_A(sA)` produces a GMMA descriptor view of smem, and `make_fragment_A` creates a `uint64_t[1]` descriptor.

With RS, `partition_A` partitions smem according to the register layout `ALayout_64x16` — each thread gets a view of its specific smem elements needed for the S2R copy. `make_fragment_A` creates real register fragments: `uint32_t[4]` per atom (8 bf16 values per thread).

B is unchanged — still a GMMA descriptor.

The existing variables `tCsA`, `tCrA` keep their names and types adapt automatically via CuTe dispatch.

### 3. S2R copy added to consumer main loop (device kernel)

Before each `gemm()`, copy A from smem into registers:

```cpp
// New: S2R copy for A (RS requires A in registers)
copy(tCsA(_,_,_,smem_pipe_read.index()), tCrA);

warpgroup_fence_operand(tCrC);
warpgroup_arrive();
gemm(mma, tCrA,                        // register values (was descriptor)
          tCrB(_,_,_,smem_pipe_read.index()), tCrC);
```

`tCrA` is now a register fragment, so it does NOT take a pipe index — the data was already loaded into registers above. The `gemm()` call signature stays the same; CuTe dispatches on the runtime type difference.

### 4. Register allocation (device kernel)

SS uses `warpgroup_reg_alloc<232>()`. RS needs slightly more registers since A is `uint32_t[4]` (16 bytes) instead of `uint64_t[1]` (8 bytes) per warpgroup. Net increase is modest (~3 registers). The exact value will be determined from ptxas output at compile time. Starting estimate: 240.

### 5. Comment updates

File header and inline comments updated to describe RS vs SS. The "no S2R copies" claim in the original header is replaced with "S2R copy for A operand (RS variant)."

## What stays the same

- Tile: 128×256×64, Layout<Shape<_2, _1>> (2 warpgroups), 384 threads
- Smem layouts: GMMA::Layout_K_SW128_Atom swizzled, 3 pipeline stages (~213 KB total)
- TMA load for A and B, TMA store for C
- PipelineTmaAsync (producer warp group + 2 consumer warp groups)
- Persistent scheduling over tiles (1D grid)
- STSM BF16 epilogue (F32 accum → BF16 → smem → TMA store)
- `warpgroup_reg_dealloc<40>()` for producer warp group
- All host-side configuration except the MMA atom type
- Benchmark parameters and correctness check

## Register budget analysis

| Component | SS (current) | RS (new) |
|-----------|-------------|----------|
| A operand | 1× uint64_t descriptor (8B) | 4× uint32_t registers (16B) |
| B operand | 1× uint64_t descriptor (8B) | 1× uint64_t descriptor (8B) |
| C accum | Same | Same |
| Everything else | Same | Same |

Per-thread register delta: ~+2-3 registers. The `warpgroup_reg_alloc` will be tuned based on ptxas output.

## File structure

- Create: `cute-reference/samples/11_bf16_gemm_sm90_pipe_tma_ws_persistent_rs.cu` (copy from `09_bf16_gemm_sm90_pipe_tma_ws_persistent.cu`)
- Modify: `cute-reference/samples/CMakeLists.txt` (add entry for sample 11)

## Self-Review

1. **Placeholder scan**: No TBDs. Register alloc value to be determined empirically from compile output — this is a known unknown, not a placeholder.
2. **Internal consistency**: Data flow is consistent — A moves smem→registers→WGMMA, B stays smem→descriptor→WGMMA, C stays accumulators→STSM→smem→TMA store.
3. **Scope check**: Single-file change with minimal diff. CMakeLists.txt one-line addition.
4. **Ambiguity check**: The S2R copy uses CuTe's standard `copy()` with MMA-partitioned views — same mechanism as the existing STSM R2S copy in reverse.
