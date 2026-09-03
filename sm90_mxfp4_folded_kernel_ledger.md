# SM90 MXFP4 x FP8 folded-kernel ledger

## Goal and acceptance contract

Rewrite the folded SM90 MXFP4 x FP8 kernel so that all of the following are true:

1. Persistent weights are 4-bit E2M1 values, and persistent scales are 8-bit values.
2. The kernel unpacks each 4-bit weight and converts it to E4M3 on the fly.
3. The kernel applies each folded per-K32-group scale to the converted weight on the fly.
4. The kernel applies the folded base scale to the accumulator exactly once per K128 tile.
5. The tuner reports at least 1050 TFLOP/s for the synthetic 16384 x 16384 x 16384 case.

The performance criterion refers to the tuner's own `measured` result, not a separately timed wrapper or a kernel with an 8-bit persistent weight cache.

Peak-performance benchmarking must run with tuning enabled. An untuned benchmark may be used only as a diagnostic and cannot satisfy the 1050 TFLOP/s acceptance criterion.

Kernel-shape constraint: keep two math warpgroups. Do not pursue a three-math-warpgroup folded kernel.

Implementation reference: follow the existing SM90 mixed GEMM's register-source weight packing and on-the-fly dequantization design. Reuse its ownership/layout derivation and hot-loop structure where the MXFP4 folded arithmetic permits it.

Clarified folded metadata: store four precomputed 8-bit relative exponent scales and one 8-bit base exponent code. Each relative scale byte is the high byte of the exact FP16 power of two, `(15 + shift) << 2`; the hot loop does not add the FP16 bias or reconstruct anything from the original UE8M0 group codes. Within every K32 group, the two scale bytes for a WGMMA-owned row pair are adjacent and aligned. The four N-lanes sharing that row pair use one `uint16_t` shared-memory load indexed by `local_tid / 4`.

Metadata planes and loads: store all per-group relative shifts in one plane and all record-wide base codes in a separate plane. Load the relative-shift plane with TMA and load the 16-byte-aligned base records with `cp.async` into a separate shared-memory stage buffer. This supersedes the earlier two-TMA-load direction.

## Implementation and verification record

| Revision | Persistent layout | K128 arithmetic | Build | Correctness | 16K cubed tuner |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 8-bit E4M3 weights; 16-bit folded metadata | Direct E4M3 load; base applied after four K32 MMAs | Existing | Existing | Invalid for this contract |
| Valid rewrite, N128 atom | 4-bit E2M1 weights; 8-bit shift and base planes | On-the-fly E4M3 conversion and shifts; base applied once per K128 scratch | 168 registers, no spills | `max_abs=0.001953125` vs dequant reference | 648.3 TFLOP/s wrapper; rejected |
| Comparison-free integer decode | 4-bit E2M1 weights; minimum base plus upward shifts | Fixed masks/shifts; no branch or compare | 168 registers, no spills | `max_abs=0.001953125` | Wrapper: 694.5 TFLOP/s with tuning enabled; not a tuner `measured` result |
| Mixed-style packed FP16x2 decode | 4-bit mixed-GEMM nibble pack; minimum base plus upward shifts | Fixed lane construction, FP16x2 normalization and scale, then E4M3 conversion | 168 registers, no spills | `max_abs=0.001953125` | Wrapper: 779.1 TFLOP/s with tuning enabled; not a tuner `measured` result |
| Prebiased FP16 exponent byte | Same 4-bit pack; each 8-bit relative scale stores `15 + shift` | Hot loop places the stored byte at FP16 exponent bit 10; no bias addition | 168 registers, no spills | `max_abs=0.001953125` | Tuner: `measured=10.3615` ms, 848.9 TFLOP/s (swizzle 0); retained by direction |
| WGMMA-owned packed scale pair | Same 4-bit pack; two FP16 exponent high bytes packed per `local_tid / 4` row pair | One aligned `uint16_t` scale load supplies both on-the-fly row conversions | 168 registers, no spills | `max_abs=0.001953125` | Tuner: `measured=10.2638` ms, 857.0 TFLOP/s (swizzle 3); target not met |
| Contiguous packed RS fragment | Two packed E2M1 words converted together into the native four-word K32 A fragment | One helper consumes the complete packed input and writes `{row_lo_k0, row_hi_k0, row_lo_k16, row_hi_k16}` through contiguous fragment storage | 168 registers, no spills | `max_abs=0.001953125` | Tuner: `measured=10.1324` ms, 868.1 TFLOP/s (swizzle 0); target not met |
