# DeepSeek-V4-Flash MXFP4 weight-range report

## Summary

A full CPU scan of every MXFP4 scale tensor in the locally cached
DeepSeek-V4-Flash checkpoint, followed by decoding the packed E2M1 groups at
the extreme scale codes, found that the nonzero, fully dequantized MXFP4
weights have the absolute-value range

```text
2^-10 <= |weight| <= 2^2
```

The corresponding normalized unbiased exponent range is therefore `[-10, 2]`.
Zero weights are present but do not have a finite exponent.

This range does not fit exactly in E4M3 when the fully dequantized weights are
materialized directly. E4M3's smallest positive subnormal is `2^-9`, so the
checkpoint values at `2^-10` are below its nonzero range. In total, 2,665
individual scalar weight values are not exactly representable by a direct
E4M3 conversion.

This does not prevent exact use of E4M3 as the SM90 WGMMA operand format. Both
of the kernel's decomposed representations keep a separate power-of-two scale:

- The exact-scale path converts the unscaled E2M1 payload to E4M3 and applies
  each UE8M0 weight scale during FP32 accumulation.
- The folded path converts a relatively shifted E2M1 payload to E4M3 and
  applies the common base scale during FP32 accumulation.

All scale records in this checkpoint satisfy the folded path's exponent-span
constraint.

## Checkpoint and scope

The scan used this local Hugging Face snapshot:

```text
/mnt_cfs/huggingface_hub/hub/models--deepseek-ai--DeepSeek-V4-Flash/
  snapshots/6e763230a9d263eca2023f1d4a5ce1bfe126cf48
```

The checkpoint contains two quantized weight families. They were distinguished
from the safetensors metadata by pairing each `.scale` tensor with its
corresponding `.weight` tensor:

| Packed weight dtype | Scale dtype | Tensor count | Interpretation |
|---|---|---:|---|
| `I8` | `F8_E8M0` | 33,792 | Two packed E2M1 values per byte: MXFP4 |
| `F8_E4M3` | `F8_E8M0` | 375 | FP8 weights, excluded from this report |

The MXFP4 scan covered:

- 33,792 scale tensors for expert `w1`, `w2`, and `w3` weights
- 8,858,370,048 UE8M0 scale bytes
- 283,467,841,536 scalar E2M1 weight values covered by those scales
- all 46 safetensors shards

No model weights were loaded on a GPU.

## Decoding method

Each MXFP4 scale tensor has shape `[N, K / 32]`. One UE8M0 scale applies to 32
E2M1 values along K. The paired packed weight tensor has shape `[N, K / 2]`.

The checkpoint's own conversion code defines the E2M1 nibble table as:

```text
code:   0    1    2    3    4    5    6    7
value:  0   .5    1   1.5   2    3    4    6
```

Codes 8 through 15 have the corresponding negative values, with code 8 being
negative zero. For a finite UE8M0 byte `c`, the scale is

```text
scale(c) = 2^(c - 127)
```

Each scalar was interpreted using

```text
dequantized_weight = E2M1_value * scale(UE8M0_code)
```

The global scale scan established the possible bounds. The packed E2M1 groups
at the extreme scale codes were then decoded to establish the actual, rather
than merely theoretical, dequantized bounds.

## Stored UE8M0 scale range

The stored scale codes range from 118 through 127:

```text
118 <= code <= 127
-9 <= code - 127 <= 0
2^-9 <= scale <= 1
```

There are no code-255 UE8M0 NaNs.

| UE8M0 code | Scale exponent | Count | Percentage |
|---:|---:|---:|---:|
| 118 | -9 | 1,070 | 0.000012079% |
| 119 | -8 | 857,063 | 0.009675177% |
| 120 | -7 | 3,521,636,456 | 39.754903407% |
| 121 | -6 | 5,325,191,528 | 60.114801020% |
| 122 | -5 | 10,386,589 | 0.117251695% |
| 123 | -4 | 257,005 | 0.002901267% |
| 124 | -3 | 37,024 | 0.000417955% |
| 125 | -2 | 3,183 | 0.000035932% |
| 126 | -1 | 129 | 0.000001456% |
| 127 | 0 | 1 | 0.000000011% |

Codes 120 and 121, corresponding to scale exponents -7 and -6, account for
99.869704427% of all MXFP4 scale values.

## Dequantized weight range

### Minimum nonzero magnitude

The minimum stored scale is `2^-9`. Among the 1,070 K32 groups with that scale,
1,399 scalar weights have E2M1 magnitude `0.5`. Therefore the exact minimum
nonzero magnitude is

```text
0.5 * 2^-9 = 2^-10 = 0.0009765625
```

No smaller nonzero value is possible because `0.5` is the smallest nonzero
E2M1 magnitude and `2^-9` is the smallest scale present in the checkpoint.

### Maximum magnitude

The maximum stored scale is `1`, and it occurs for exactly one K32 group:

```text
mtp.0.ffn.experts.208.w2.scale[row=2237, k32_group=2]
```

That group contains 31 zero weights and one weight with E2M1 magnitude `4`.
Its largest dequantized magnitude is therefore

```text
4 * 1 = 4 = 2^2
```

Although E2M1 can represent magnitude `6`, the scale-1 group does not contain
it. A group at the next lower scale, `2^-1`, cannot exceed `6 * 2^-1 = 3`.
Consequently, `4` is the exact global maximum rather than a bound inferred
only from the two formats.

## Direct E4M3 representability

E4M3 has:

```text
smallest positive subnormal = 2^-9
largest finite value        = 448
subnormal quantum           = 2^-9
```

The upper end of the checkpoint's dequantized range fits easily, but the lower
end does not. At scale exponent -9, the following E2M1 magnitudes are not on
the E4M3 subnormal grid:

| E2M1 magnitude | Scale | Dequantized magnitude | Scalar count | Exact in E4M3? |
|---:|---:|---:|---:|---|
| 0.5 | `2^-9` | `2^-10` | 1,399 | No: below the minimum subnormal |
| 1.5 | `2^-9` | `3 * 2^-10` | 1,266 | No: halfway between subnormal grid points |

Thus 2,665 means 2,665 individual scalar weight values, not 2,665 tensors or
scale groups. Every other nonzero dequantized MXFP4 value in this checkpoint is
exactly representable in E4M3.

A direct cast of all fully dequantized weights to E4M3 would therefore alter
these 2,665 values. The exact rounded result depends on the conversion's
rounding policy.

## K128 folding analysis

The SM90 FP8 x MXFP4 GEMM uses one activation scale per K128 and one weight
scale per K32. Folding the four K32 weight scales into a shared base requires
their exponent span to be at most six:

```text
max(weight_scale_exponent[0:4])
  - min(weight_scale_exponent[0:4]) <= 6
```

The scan checked this condition for each output row and each consecutive set of
four K32 groups:

| Exponent span | Row/K128 groups | Percentage |
|---:|---:|---:|
| 0 | 419,538,294 | 18.944265897% |
| 1 | 1,789,991,506 | 80.827127171% |
| 2 | 5,034,496 | 0.227332838% |
| 3 | 24,550 | 0.001108556% |
| 4 | 3,362 | 0.000151811% |
| 5 | 291 | 0.000013140% |
| 6 | 13 | 0.000000587% |

Results:

- 2,214,592,512 row/K128 groups checked
- maximum exponent span: 6
- zero row/K128 groups exceed a span of 6
- zero row/K128 groups contain UE8M0 code 255

The packing-time OUT64 x K128 criterion also passes universally:

- 34,603,008 OUT64 x K128 records checked
- 34,603,008 records foldable
- zero non-foldable records

One maximum-span example is:

```text
tensor:     mtp.0.ffn.experts.208.w2.scale
row:        257
K128 group: 0
codes:      [120, 121, 126, 121]
exponents:  [-7,  -6,  -1,  -6]
span:       6
```

## Kernel implications

### Directly materialized dequantized E4M3 weights

This representation is not lossless for the checkpoint. It cannot exactly
encode the 2,665 scalar values identified above.

### Exact per-K32 scale path

This representation is lossless with respect to the MXFP4 source values:

```text
E2M1 payload -> exact unscaled E4M3 operand
UE8M0 scale  -> separate FP32 accumulation scale
```

The unscaled E2M1 values from `0.5` through `6` are exactly representable in
E4M3. The fully dequantized value is never required to fit in the E4M3 operand.

### Folded K128 scale path

This representation is also lossless for every scale record in this
checkpoint. After choosing the minimum of the four K32 exponents as the base,
the relative shifts range from zero through six. The largest possible shifted
E4M3 operand is

```text
6 * 2^6 = 384 < 448
```

The shared base scale remains separate and is applied during FP32
accumulation. Therefore the `2^-10` fully dequantized values do not need to be
stored directly in E4M3.

## Conclusion

For this DeepSeek-V4-Flash snapshot:

1. Stored MXFP4 weight-scale exponents range from -9 through 0.
2. Nonzero dequantized weight exponents range from -10 through 2.
3. Directly materializing the dequantized weights in E4M3 is not fully exact;
   2,665 scalar weights are affected.
4. Keeping UE8M0 scales separate from E4M3 operands is exact.
5. Every checkpoint record also satisfies the folded K128 representation's
   maximum relative exponent shift of six.

These findings describe this exact checkpoint snapshot. They do not establish
that arbitrary MXFP4 checkpoints will satisfy the same stored exponent range
or K128 folding constraint.
