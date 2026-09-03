# SM90 U4 zero-point storage plan

## Status

**Implemented and verified on SM90.**

## Settled requirements

- The persistent SM90 U4 representation must store each zero point as an
  unsigned 4-bit value instead of BF16.
- The dequantization meaning remains

  ```cpp
  dequantized = (quantized - zero) * scale;
  ```

  Changing storage must not change this arithmetic.
- Scales remain BF16 in the persistent SM90 U4 representation.
- The change applies to the native SM90 U4 path only, for both
  `Sm90U4Format<32>` and `Sm90U4Format<128>`. The legacy HMMA U4 paths are
  unchanged.
- Each `(K group, OUT64 fragment)` uses the compact 160-byte plane layout:
  128 bytes of BF16 scales followed by 32 bytes of packed U4 zero points.
- Both planes remain in the existing combined persistent qparam allocation
  owned by `linear.scales`; `linear.zeros` is cleared after packing. The
  runtime keeps one qparam pointer and one qparam TMA descriptor.
- Weight packing trusts the source-format contract that every zero point is an
  exact integer in `[0, 15]`. It does not add a content scan, device result, or
  host synchronization. Silent rounding and clamping are not part of the
  design; values outside the source-format contract have undefined results.
- This document records the approved implementation and verification contract.

## Verified current contract

The native SM90 U4 family currently accepts floating-point scale and zero
tensors, converts both to BF16, and fuses them before packing:

```cpp
bridge->convert_scales = kBfloat16;
bridge->convert_zeros  = kBfloat16;

fuse_scales_and_zeros_bf16(tmp_q.data<bfloat16_t>(),
                           linear.scales.raw_data(),
                           linear.zeros ? linear.zeros.raw_data() : nullptr,
                           linear.scales.dtype(),
                           linear.scales.size(),
                           stream);
```

For each logical `(K group, output channel)`, the persistent qparam record is
one `uint32_t`:

```text
bits  0..15: BF16 scale
bits 16..31: BF16 effective_zero, where effective_zero = zero + 128
```

The `+128` matches the weight decoder. The decoder first materializes each U4
weight as the exactly representable BF16 value `128 + q`, then computes:

```cpp
(BF16(128 + q) - BF16(128 + zero)) * scale
```

This is exactly `(q - zero) * scale` for U4 zero points in `[0, 15]`.

The persistent qparam payload per OUT64 fragment and K-group is currently:

```text
64 outputs * 4 bytes = 256 bytes
```

The qparam TMA tensor is one homogeneous, contiguous payload:

```cpp
using QparamCtaTile = cute::Shape<
    cute::Int<Format::kQparamValuesFragment>,
    cute::Int<kOutputFragments>,
    cute::Int<kQGroupsPerStage>>;
```

Only one qparam pointer and one qparam TMA descriptor are passed to the mixed
kernel. Grouped GEMM also publishes exactly one qparam descriptor and links
exactly one qparam pointer per expert. A design with separate persistent scale
and zero tensors would therefore change the kernel ABI, grouped descriptor
preparation, and MoE expert linking.

The checkpoint-facing formats normalize U4 zero points to floating-point
tensors before `LinearWeight::prepare()`. AWQ and GPTQ unpack nibble zero
points; symmetric GPTQ/compressed-tensors weights synthesize the value `8`.
The existing C++ family nevertheless accepts any trivial floating-point zero
tensor and does not currently enforce that every value is an integer in
`[0, 15]`.

Both native SM90 U4 group sizes use the same persistent representation and
dequantizer template:

```cpp
Sm90U4Format<32>
Sm90U4Format<128>
```

For a `TILE_N=128`, `TILE_K=64` stage, the current qparam traffic is:

| Group size | K-groups per stage | OUT64 fragments | Qparam bytes |
|---:|---:|---:|---:|
| 32 | 2 | 2 | 1024 |
| 128 | 1 | 2 | 512 |

## Settled design

Keep the existing single persistent qparam allocation and single TMA
descriptor, but change each `(K group, OUT64 fragment)` payload from 256 bytes
to 160 bytes:

```text
byte   0..127: 64 BF16 scales in consumer-fragment order
byte 128..159: 64 U4 zero points, two zero points per byte
```

Within the zero plane:

```text
zero byte p, bits 0..3: output value 2*p
zero byte p, bits 4..7: output value 2*p + 1
```

This plane layout is preferred over interleaved five-byte records because the
consumer can load each pair of BF16 scales with one aligned 32-bit load. The
zero plane begins 128 bytes into the fragment. The four lanes owning one scale
pair read the same zero byte, so the shared-memory access is a broadcast:

```cpp
const auto* scale_pairs = reinterpret_cast<const uint32_t*>(fragment);
const auto* zero_bytes  = fragment + 128;

const int pair       = local_tid / 4;
const uint32_t scales = scale_pairs[pair];
const uint32_t zeros  = zero_bytes[pair];

const uint32_t scale_lo = scales & 0xffffu;
const uint32_t scale_hi = scales >> 16;
const uint32_t zero_lo  = zeros & 0xfu;
const uint32_t zero_hi  = (zeros >> 4) & 0xfu;
```

The zero point can be expanded once during qparam register loading. BF16
integers `128..143` have bit patterns `0x4300..0x430f`, so the existing
dequantization loop can continue to consume replicated BF16x2 zero registers:

```cpp
constexpr uint32_t kBf16x2_128 = 0x43004300u;
const uint32_t zero_lo_pair =
    kBf16x2_128 | zero_lo | (zero_lo << 16);
const uint32_t zero_hi_pair =
    kBf16x2_128 | zero_hi | (zero_hi << 16);
```

Expanding during `Sm90MixedDequant<Sm90U4Format<GroupSize>>::load()` avoids
repeating the expansion for every K16 dequantization call. It keeps the current
zero-register count; the benefit is persistent-memory, TMA, and
shared-memory traffic reduction rather than register reduction.

The compact payload changes the stage quantities to:

| Group size | Qparam data bytes | 128-byte-aligned stage storage | Reduction in TMA bytes |
|---:|---:|---:|---:|
| 32 | 640 | 640 | 37.5% |
| 128 | 320 | 384 | 37.5% |

The TMA payload uses byte elements:

```cpp
using QparamType = uint8_t;
static constexpr int kQparamValuesFragment = 160;
```

The corresponding TMA dimensions remain valid: the contiguous 160-byte box is
no larger than 256 bytes, and its 160-byte fragment stride and 320-byte
group stride are multiples of 16 bytes. The qparam layout remains unswizzled.

The existing pack entry point has this signature rather than adding a parallel
abstraction:

```cpp
void PackSm90U4QParams(uint8_t*          dst,
                       const bfloat16_t* scales,
                       const bfloat16_t* zeros,
                       int               output_dim,
                       int               group_count,
                       cudaStream_t      stream);
```

It writes the BF16 scale plane and packed U4 zero plane directly in
consumer-fragment order. A null `zeros` pointer would encode zero point `0`,
preserving current behavior.

After packing, the metadata is:

```cpp
linear.zeros         = {};
linear.q_desc        = transpose(MatrixLayout{
    kUint8,
    kColMajor,
    linear.output_dim,
    linear.input_dim / group_size,
    linear.output_dim / kSm90MixedFragmentN
        * kSm90U4QparamValuesFragment,
    kSm90MixedQParamPack,
});
linear.weight_format = DataFormat{
    kUint4,
    {group_size, 1},
    kBfloat16,
    kUint4,
};
```

`linear.scales` continues to own the combined persistent qparam payload, and
`linear.zeros` is cleared after packing. This avoids a second runtime operand
and a fifth grouped TMA descriptor. The obsolete
`fuse_scales_and_zeros_bf16` entry point was removed rather than left as dead
code.

## Runtime dequantization pipeline

The mainloop remains one `PipelineTmaAsync<Stages>` pipeline. Packed weights,
activations for the non-indexed path, and compact qparams share each stage's
full barrier. Only the qparam transfer size and qparam decoding change.

```text
persistent qparams
  [K/group][OUT64][128-byte BF16 scale plane | 32-byte U4 zero plane]
        |
        | TMA load, on the existing mainloop stage barrier
        v
stage shared memory
        |
        | consumer_wait(stage)
        v
qparam register load
  BF16 scales -> replicated BF16x2 scale registers
  U4 zeros    -> BF16(128 + zero) zero registers
        |
        | one packed weight word per lane and K16 fragment
        v
LOP3: eight U4 weights -> eight BF16 values biased by +128
        |
        v
four BF16x2 subtractions: (128 + q) - (128 + zero)
        |
        v
four BF16x2 multiplies by scale
        |
        v
BF16 register-source A fragment -> WGMMA
```

The TMA producer continues to issue qparams with packed weights on the same
barrier:

```cpp
pipeline.producer_acquire(write_state);
auto* bar = pipeline.producer_get_barrier(write_state);

cute::copy(tm_v.with(Vdesc, *bar, mask_B, cute::TMA::CacheHintSm90::EVICT_LAST),
           qparam_src(cute::_, qparam_tile),
           qparam_dst(cute::_, 0));

issue_packed_copy(..., bar, ...);
```

The barrier's expected transaction count must use the compact qparam bytes:

```cpp
main_params.transaction_bytes =
    (kIndexedGather ? kPackedBytesStage
                    : kPackedBytesStage + kInputBytesStage)
    + kQparamDataBytesStage;
```

The existing `consumer_wait` therefore still guarantees that the packed
weights, activation tile where applicable, scale plane, and zero plane are all
resident before any qparam or weight-fragment load.

Group-size behavior remains:

- **G32:** one K64 stage contains two K32 qparam groups. `load_qparams(stage)`
  loads both compact groups. K16 fragments `kb=0,1` use group 0; `kb=2,3`
  use group 1.
- **G128:** one qparam group spans two K64 stages. The consumer refreshes its
  qparam registers on the first stage and reuses them on the second. The
  producer currently issues the same qparam TMA transfer for both stages so
  every mainloop stage has a fixed transaction count. This existing duplicate
  G128 transfer remains unless a separate pipeline redesign is approved.

Within one OUT64 fragment, four adjacent lanes use the same qparam pair. The
scale plane is loaded as one 32-bit pair of BF16 scales. Those four lanes also
load the same byte from the zero plane and select its two nibbles, so the byte
access is a shared-memory broadcast.

The originally proposed 32-bit zero-word load plus lane-variable shift was
rejected during generated-code verification. `ptxas` serialized WGMMA in the
G128 production kernel and the 16384-cubed tuner sample fell to about 704
TFLOP/s. Directly loading the pair's byte removed the serialization warning
and restored the same kernel to about 820 TFLOP/s.

The arithmetic is exact for the settled source contract:

```cpp
BF16(128 + q) - BF16(128 + zero) == BF16(q - zero)
```

for `q, zero` in `[0, 15]`, before multiplication by the BF16 scale.

The consumer expands each U4 zero into the existing replicated BF16x2 zero
register during `Dequant::load()`. The K16 hot loop and its qparam register
count therefore remain unchanged.

## Implementation and verification

Implementation changes only the native SM90 U4 family packer, qparam format,
and its existing dequantizer specialization. The legacy HMMA U4 path and the
SM90 registry structure remain unchanged. The obsolete
`fuse_scales_and_zeros_bf16` entry point is removed after its only caller is
replaced. No permanent test is added.

Verification covers both `Sm90U4Format<32>` and `Sm90U4Format<128>`:

1. Build the existing targets with `ninja` from `build`, and inspect the CUDA
   compiler output for register count and spills.
2. Run the existing linear fixture's reference comparison for a native SM90
   U4 dense case at each group size. This exercises packing, TMA transfer,
   qparam loading, and dequantization together.
3. Re-run the existing `16384 x 16384 x 16384` tuning benchmark with the
   already established policy:

   ```bash
   TM_GEMM_TUNE='max_splits=1,max_waves=10,top_k=0,swizzle=[0,1,2,3],clusters=0,min_iter=1,max_iter=1,max_time=10'
   TM_GEMM_TUNE_VERBOSE=1
   ```

   Report the selected tuning parameters and TFLOP/s for both group sizes.

Verification results on an H200:

- G32 and G128 reference comparisons both produced exactly zero
  `quant_vs_dequant` max and mean absolute error.
- The packer used 32 registers with no stack frame or spills.
- The G128 `384x128 S4 2x1` kernel used 168 registers with no stack frame or
  spills and no WGMMA-serialization diagnostic.
- The G32 `384x128 S3 1x2` kernel used 168 registers and reported a
  24-byte stack frame, 44-byte spill stores, and 32-byte spill loads.
- At 16384 cubed with `splits=1`, G32 selected `swizzle=0` at 10.6320 ms
  (827.3 TFLOP/s), and G128 selected `swizzle=0` at 10.7243 ms
  (820.2 TFLOP/s).
