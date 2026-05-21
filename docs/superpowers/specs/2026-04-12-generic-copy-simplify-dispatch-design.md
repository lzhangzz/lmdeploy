# Simplify GenericCopy Dispatch

Date: 2026-04-12

## Problem

`GenericCopy` in `src/turbomind/core/tensor.cu` has a 3-level nested dispatch: 7 dtypes x 5 vec sizes x 6 ranks = up to 210 kernel template instantiations. The kernel treats all data as opaque bit patterns — the concrete C++ type only matters for `sizeof_bits_v<T>`. Dispatching by concrete type creates redundant instantiations.

Rank 5 and 6 are dispatched but untested and unused. The kernel has a `if constexpr (kRank == 1)` branch that splits two fundamentally different code paths inside one function.

## Design

Three changes, all in `tensor.cu`:

### 1. Byte-size dtype dispatch

Replace the outer `switch(dtype)` (7 cases: float, half, bfloat16, int8, int32, bool, uint8) with `switch(byte_size(dtype))`:

| byte_size | Representative type | Covers |
|-----------|-------------------|--------|
| 1 | `uint8_t` | int8, bool, uint8 |
| 2 | `uint16_t` | half (fp16), bfloat16 |
| 4 | `uint32_t` | float (fp32), int32 |

Default case: `TM_CHECK(0) << "GenericCopy: unsupported element size"`.

Rationale: `Copy_Atom<UniversalCopy<uint_bit_t<kVec * sizeof_bits_v<T>>>, T>` copies bits regardless of T's semantic meaning. Using integer types reinforces the opaque-copy semantics. `reinterpret_cast<uintN_t*>(data)` is valid for any pointer.

Both the GenericCopy host and TransposeCopy host dispatch use this pattern.

### 2. Split rank-1 and rank>1 into separate kernels

**`CopyKernel1D<T, kVec, LayoutT>`** — extracted from current lines 108-144:
- Single inner dimension, no `group<>` call
- `blockIdx.x` = tile index, `blockIdx.y` unused (early return if > 0)
- Same kVec>1 / kVec==1 predication logic

**`CopyKernelND<T, kVec, LayoutT>`** — extracted from current lines 147-193:
- Ranks 2-4, uses `group<1, kRank>` to flatten outer dims
- `blockIdx.y` = outer flat index, `blockIdx.x` = inner tile index
- Same tiling and predication

Both kernels keep the `if constexpr (kVec * sizeof_bits_v<T> <= 128)` guard.

### 3. Drop rank 5/6, cap at 4

The host rank dispatch becomes:

```cpp
switch (rank) {
    case 1: invoke_kernel<CopyKernel1D>(constant<1>{}); break;
    case 2: invoke_kernel<CopyKernelND>(constant<2>{}); break;
    case 3: invoke_kernel<CopyKernelND>(constant<3>{}); break;
    case 4: invoke_kernel<CopyKernelND>(constant<4>{}); break;
    default: TM_CHECK(0) << "GenericCopy: rank > 4 not implemented";
}
```

Ranks 5/6 are untested and unused. Throwing with a clear message is preferable to silently instantiating untested template combos.

## Files changed

- `src/turbomind/core/tensor.cu` — all changes in this file. No header changes needed.

## Verification

- Run `test_generic_copy.py` — all existing tests must pass (covers dtypes fp32, fp16, i8, i32; ranks 1-4; transpose and contiguous)
- Build with `ninja` to verify compilation
