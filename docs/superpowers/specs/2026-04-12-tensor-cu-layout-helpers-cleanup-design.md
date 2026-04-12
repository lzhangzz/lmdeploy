# tensor.cu Layout Helpers Cleanup Spec

**Goal:** Eliminate ~120 lines of boilerplate in the `detail` namespace of `tensor.cu` by replacing rank-by-rank `if constexpr` chains with C++17 index_sequence pack expansion.

**Motivation:** Three helper functions (`make_cute_shape`, `make_cute_stride`, `make_cute_layout_unit_inner`) each have 6 `if constexpr` branches for ranks 1-6, producing ~150 lines of repetitive code. Each branch is identical except for the number of arguments. C++17's `std::make_index_sequence` can collapse all branches into one.

---

## Approach

Use `std::make_index_sequence<kRank>` to generate a parameter pack of indices, then expand the pack into `cute::make_shape(...)` and `cute::make_stride(...)` calls. This is standard C++17, no CuTe internal API dependencies.

### What changes

Replace 3 functions in the `detail` namespace:

| Function | Before | After |
|----------|--------|-------|
| `make_cute_shape<kRank>` | 6 `if constexpr` branches, ~35 lines | `_impl` helper + 1-line wrapper, ~10 lines |
| `make_cute_stride<kRank>` | 6 `if constexpr` branches, ~35 lines | `_impl` helper + 1-line wrapper, ~10 lines |
| `make_cute_layout_unit_inner<kRank>` | 6 `if constexpr` branches, ~60 lines | Reuses `make_cute_shape` + offset index stride helper, ~15 lines |

`make_cute_layout<kRank>` is unchanged (already 3 lines).

### What stays the same

- `kernel::GenericCopyKernel` — no changes
- `kernel::TransposeCopyKernel` — no changes
- `GenericCopy` host function — no changes
- All produced CuTe layout types are identical

### Implementation pattern

Each helper follows this pattern:

```cpp
template<size_t... Is>
auto make_cute_shape_impl(const ssize_t* data, std::index_sequence<Is...>)
{
    return cute::make_shape(static_cast<int32_t>(data[Is])...);
}

template<int kRank>
auto make_cute_shape(const ssize_t* data)
{
    return make_cute_shape_impl(data, std::make_index_sequence<kRank>{});
}
```

For `make_cute_layout_unit_inner`, the stride helper uses `Is + 1` offset to skip index 0 (replaced by `cute::Int<1>{}`):

```cpp
template<size_t... Is>
auto make_unit_inner_stride_impl(const ssize_t* stride, std::index_sequence<Is...>)
{
    return cute::make_stride(cute::Int<1>{}, static_cast<int64_t>(stride[Is + 1])...);
}
```

With `make_index_sequence<kRank - 1>`, `Is` ranges over `0..kRank-2`, so `Is + 1` covers `1..kRank-1`. For `kRank=1`, the pack is empty, producing just `make_stride(Int<1>{})`.

---

## Scope

**In scope:**
- `detail` namespace helpers only (lines 17-172 of tensor.cu)
- Net reduction from ~155 lines to ~35 lines

**Out of scope:**
- Host dispatch switch patterns (not boilerplate — necessary for compile-time template specialization)
- Kernel code (no duplication to eliminate)
- File splitting (not requested)

---

## Risk

Very low. The change is purely structural — same template instantiations, same generated code. The `std::index_sequence` pattern is well-established C++17. CuTe's `make_shape`/`make_stride` accept variadic arguments, so pack expansion works directly.

One subtlety: `make_cute_layout_unit_inner` uses `cute::Int<1>{}` (compile-time) vs `int64_t(1)` (runtime) for the first stride. This distinction must be preserved — the index_sequence approach does this by hardcoding `Int<1>{}` as the first argument and only expanding the remaining strides.
