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

### 2. Consumer setup: partition_A and make_fragment_A behavior changes

The atom shape is 64×256×16 (M×N×K). With our smem K=64 and atom K=16, both variants have MMA_K = 4 K-tiles. The critical difference is in how `partition_A` and `make_fragment_A` work:

**SS variant:**
- `ALayout = GMMA::ABLayout<64, 16>` — smem layout with `Stride<_0, ...>` so all 128 threads map to the same coordinate, producing 4 GMMA descriptors (one per K-slice)
- `FrgTypeA = GMMA::smem_desc<Major>` (defined in MMA_Traits) — `make_fragment_A` takes the `has_dereference` branch, wrapping the smem view in a descriptor iterator. No data copied; each thread holds 4 `uint64_t` descriptors.

**RS variant:**
- `ALayout = GMMA::ALayout_64x16` — register layout; each thread maps to a unique (M,K) position (128 threads × 8 values = 1024 = 64×16)
- `FrgTypeA` is NOT defined in MMA_Traits — falls back to `ValTypeA = bfloat16_t`. `make_fragment_A` takes the `else` branch, calling `make_fragment_like` to allocate actual register storage for bf16 values.
- Each thread gets 4 K-tiles × 8 bf16 values = 32 bf16 values in registers for A.

B is unchanged — still `FrgTypeB = GMMA::smem_desc<Major>`, 4 descriptors per thread.

The existing variables `tCsA`, `tCrA` keep their names and types adapt automatically via CuTe dispatch.

### 3. S2R copy added to consumer main loop (device kernel)

Before each `gemm()`, copy all 4 K-slices of A from smem into registers in one shot:

```cpp
// New: S2R copy for A (RS requires A in registers)
copy(tCsA(_,_,_,smem_pipe_read.index()), tCrA);

warpgroup_fence_operand(tCrC);
warpgroup_arrive();
gemm(mma, tCrA,                        // register values (was descriptor)
          tCrB(_,_,_,smem_pipe_read.index()), tCrC);
```

`tCrA` is now a register fragment, so it does NOT take a pipe index in `gemm()` — the data was already loaded above. CuTe's `gemm()` handles the 4 MMA_K iterations internally, consuming each K-slice of `tCrA` and `tCrB` in turn. `tCrB` still takes a pipe index because each K-slice is a separate GMMA descriptor.

The `copy()` from ALayout_64x16-partitioned swizzled smem to registers generates plain `ld.shared` instructions. The swizzle is transparent — the composition of ALayout_64x16 with the swizzled smem layout produces correct physical addresses.

### 4. Register allocation (device kernel)

SS uses `warpgroup_reg_alloc<232>()`. RS stores 32 bf16 values per thread for A (vs 4 uint64_t descriptors for SS), roughly doubling A's register footprint. The exact `warpgroup_reg_alloc` value will be determined from ptxas output at compile time. Starting estimate: 248.

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

Atom shape: 64×256×16. Smem K=64 → MMA_K = 4 K-tiles.

| Component | SS (current) | RS (new) |
|-----------|-------------|----------|
| A operand | 4× uint64_t descriptors (32B) | 32× bf16 values (64B) |
| B operand | 4× uint64_t descriptors (32B) | 4× uint64_t descriptors (32B) |
| C accum | Same | Same |
| Everything else | Same | Same |

Per-thread register delta: ~+8 registers for A (64B vs 32B, at 4B/register). Note the compiler may pack bf16 pairs into registers and optimize descriptor liveness; the actual register count is determined by ptxas. The `warpgroup_reg_alloc` will be tuned based on compile output.

## Layout comparison

| Property | SS (ABLayout<64,16>) | RS (ALayout_64x16) |
|----------|---------------------|---------------------|
| Type | Smem layout | Register layout |
| Thread mapping | All 128 threads → same (M,K) | Each thread → unique (M,K) |
| Stride mode-0 | `_0` (broadcast) | `_128` (unique per thread) |
| Per-thread values | n/a (descriptor) | 8 bf16 per K-tile |
| `FrgTypeA` | `smem_desc<Major>` | `ValTypeA` (= bf16) |
| `make_fragment_A` | Wraps smem in descriptor | Allocates register storage |

`ALayout_64x16` definition (`mma_traits_sm90_gmma.hpp:451`):
```cpp
using ALayout_64x16 = Layout<Shape<Shape<_4,_8,_4>, Shape<_2,_2,_2>>,
                              Stride<Stride<_128,_1,_16>, Stride<_64,_8,_512>>>;
```
Maps 128 threads × 8 values → flat (M=64, K=16) coordinate space. The (M,K) coordinates compose with the swizzled smem layout to produce correct physical addresses for the S2R copy.

## File structure

- Create: `cute-reference/samples/11_bf16_gemm_sm90_pipe_tma_ws_persistent_rs.cu` (copy from `09_bf16_gemm_sm90_pipe_tma_ws_persistent.cu`)
- Modify: `cute-reference/samples/CMakeLists.txt` (add entry for sample 11)

## Self-Review

1. **Placeholder scan**: No TBDs. Register alloc value to be determined empirically from compile output — this is a known unknown, not a placeholder.
2. **Internal consistency**: Data flow is consistent — A moves smem→registers→WGMMA (S2R via ALayout_64x16), B stays smem→descriptor→WGMMA, C stays accumulators→STSM→smem→TMA store.
3. **Scope check**: Single-file change with minimal diff. CMakeLists.txt one-line addition.
4. **Ambiguity check**: The S2R copy uses CuTe's standard `copy()` with MMA-partitioned views — same mechanism as the existing STSM R2S copy in reverse. The swizzled smem addresses are computed correctly by layout composition.
