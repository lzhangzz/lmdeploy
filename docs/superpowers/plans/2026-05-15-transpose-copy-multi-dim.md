# TransposeCopy Multi-Dim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the smem-staged transpose path so any layout with `src.stride==1` at one dim and `dst.stride==1` at any other dim uses the fast kernel — not just strict rank-2.

**Architecture:** The dispatcher in `copy.cc::GenericCopy` finds J (any dim with dst stride 1) after the existing src-stride-ascending sort, swaps it to position 1, jointly coalesces batch dims, caps post-coalesce rank at 4, and calls multi-rank `TransposeCopy`. The kernel body becomes rank-generic by computing per-batch pointer offsets at the top, then running the existing 2D smem-staged transpose unchanged on a 2D view of the (I, J) plane.

**Tech Stack:** CUDA, CuTe (CUTLASS), C++17, PyTorch (test harness via `_turbomind` pybind11 bindings).

**Spec:** `docs/superpowers/specs/2026-05-15-transpose-copy-multi-dim-design.md`

---

## File Structure

**Modified files:**

- `src/turbomind/kernels/copy/transpose.cu` — kernel becomes rank-generic; host dispatches over rank ∈ {2, 3, 4}; adds local `make_unit_stride` helper for static-1 stride at chosen position.
- `src/turbomind/kernels/copy/copy.cc` — dispatcher finds J, transposes to canonical (I=0, J=1), coalesces batch dims; adds local `coalesce_batch_dims` helper; switches the divisibility check from a hardcoded 32 to the per-dtype tile size.
- `test_generic_copy.py` — adds five correctness cases plus a batched-throughput sweep entry.

**Reused (read only):**

- `src/turbomind/kernels/copy/copy.cu` — `detail::make_cute_shape` is duplicated locally in transpose.cu (small helper, two namespaces are separate translation units).
- `src/turbomind/core/layout.h` — `Layout::transpose(int, int)`, `Layout::shape()`, `Layout::stride()`.

**Per-task build/test commands:**

- Build: `ninja` from `build/`. (Configure with `sh ../my_generate.sh` if not already configured.)
- Run tests: `PYTHONPATH=/data/lmdeploy-copy/lmdeploy:/data/lmdeploy-copy/build/lib python test_generic_copy.py [--dtype <dt>]`. Verify a free GPU first via the `get_gpu_usage` MCP tool.

---

## Task 1: Add multi-dim regression test cases (baseline)

These cases currently pass via the `VectorizedCopy` fallback. Adding them now establishes a regression baseline so any later breakage during the refactor is caught at the right step.

**Files:**

- Modify: `test_generic_copy.py` (after line 198, in the existing 3D / 4D sections)

- [ ] **Step 1: Edit `test_generic_copy.py`**

Add these checks immediately after the existing 4D check on line 198. Insert in the "3D transformations" and "4D transformations" sections respectively:

```python
    # --- 3D transformations ---
    print("\n3D transformations:")
    check("permute (2,0,1)", _rand(16, 32, 64).permute(2, 0, 1))
    check("3D batched transpose (B,M,N)->(B,N,M)",
          _rand(8, 64, 128).transpose(1, 2))
    check("3D batched transpose unaligned (must fall through)",
          _rand(8, 60, 100).transpose(1, 2))

    # --- 4D transformations ---
    print("\n4D transformations:")
    check("4D slice", _rand(4, 8, 32, 64)[:, :, ::3, :])
    check("4D batched transpose (B,H,M,N)->(B,H,N,M)",
          _rand(4, 8, 64, 128).transpose(2, 3))
    check("4D non-adjacent transpose (B,M,H,N)->(B,N,H,M)",
          _rand(4, 64, 8, 128).transpose(1, 3))
    check("4D batched transpose with sliced batch (non-coalesceable)",
          _rand(4, 16, 64, 128)[:, ::2, :, :].transpose(2, 3))
```

- [ ] **Step 2: Run the test, confirm new cases PASS via fallback**

Run:

```bash
cd /data/lmdeploy-copy
PYTHONPATH=$PWD/lmdeploy:$PWD/build/lib python test_generic_copy.py
```

Expected: every new case prints `[PASS]` (the `VectorizedCopy` fallback handles them today). If any of the new cases prints `[FAIL]` *before* any kernel changes, that indicates a pre-existing bug in `VectorizedCopy` for that layout — investigate before continuing.

- [ ] **Step 3: Commit**

```bash
git add test_generic_copy.py
git commit -m "test(copy): add multi-dim batched-transpose regression cases"
```

---

## Task 2: Add `coalesce_batch_dims` helper

Pure refactor. The function is added but not called yet.

**Files:**

- Modify: `src/turbomind/kernels/copy/copy.cc` (after the `void TransposeCopy` forward declaration, before `GenericCopy`)

- [ ] **Step 1: Add the helper to `copy.cc`**

Insert the following just below the forward declarations of `VectorizedCopy` / `TransposeCopy` (around line 19), before the `GenericCopy` definition:

```cpp
// Merge adjacent batch dims (positions ≥ 2) of (a, b) when their strides are
// proportional in BOTH a and b. Single forward pass over positions 3..rank-1.
//
// Precondition: a and b have the same shape and rank, with positions 0 and 1
// being the (I, J) transpose pair (not coalesceable). Only positions 2.. are
// considered batch dims.
static std::pair<Layout, Layout>
coalesce_batch_dims(const Layout& a, const Layout& b)
{
    const int rank = a.rank();
    if (rank < 4) return {a, b};  // need ≥ 2 batch dims to merge

    std::vector<ssize_t> ash(a.shape().begin(),  a.shape().begin()  + 3);
    std::vector<ssize_t> ast(a.stride().begin(), a.stride().begin() + 3);
    std::vector<ssize_t> bsh(b.shape().begin(),  b.shape().begin()  + 3);
    std::vector<ssize_t> bst(b.stride().begin(), b.stride().begin() + 3);

    for (int i = 3; i < rank; ++i) {
        const ssize_t ai_sh = a.shape(i),  ai_st = a.stride(i);
        const ssize_t bi_sh = b.shape(i),  bi_st = b.stride(i);

        // Merge with the previously accumulated batch dim if its stride equals
        // shape * stride of that dim, in BOTH a and b.
        if (ai_st == ash.back() * ast.back() &&
            bi_st == bsh.back() * bst.back()) {
            ash.back() *= ai_sh;
            bsh.back() *= bi_sh;
            // strides at the back stay unchanged (they remain the inner stride)
        } else {
            ash.push_back(ai_sh); ast.push_back(ai_st);
            bsh.push_back(bi_sh); bst.push_back(bi_st);
        }
    }

    return {Layout{ash, ast}, Layout{bsh, bst}};
}
```

- [ ] **Step 2: Build to confirm it compiles**

```bash
cd /data/lmdeploy-copy/build
ninja _turbomind
```

Expected: clean build. The helper is unused for now, so you may see an `-Wunused-function` warning (acceptable) — turn into an error only if the project's warning flags already do so. If unused-function fails the build, prefix the function with `[[maybe_unused]]`.

- [ ] **Step 3: Run the test for regression**

```bash
PYTHONPATH=/data/lmdeploy-copy/lmdeploy:/data/lmdeploy-copy/build/lib \
  python /data/lmdeploy-copy/test_generic_copy.py
```

Expected: all cases pass (this task does not change behavior).

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/kernels/copy/copy.cc
git commit -m "refactor(copy): add coalesce_batch_dims helper (unused)"
```

---

## Task 3: Make `TransposeCopyKernel` rank-generic

Add batch offset decoding at the top of the kernel and refactor the existing 2D body to operate on a 2D view sliced at the per-block batch offset. For rank == 2, `take<2,2>` yields an empty tuple, `crd2idx` returns 0, and the kernel reduces exactly to today's path.

**Files:**

- Modify: `src/turbomind/kernels/copy/transpose.cu` (lines 19–87)

- [ ] **Step 1: Replace the kernel body**

Replace the kernel definition (lines 19–87) with:

```cpp
template<int kTileDim, int kVec,
         typename SrcEngine, typename SrcLayout,
         typename DstEngine, typename DstLayout>
__global__ void __launch_bounds__(256)
TransposeCopyKernel(cute::Tensor<SrcEngine, SrcLayout> src,
                    cute::Tensor<DstEngine, DstLayout> dst)
{
    using T = typename SrcEngine::value_type;
    static_assert(std::is_same_v<T, typename DstEngine::value_type>,
                  "TransposeCopyKernel: src and dst value types must match");

    constexpr int kPad      = kVec;
    constexpr int kStride   = kTileDim + kPad;
    constexpr int kThrRows  = 256 / kTileDim;
    using VecT = uint_bit_t<kVec * sizeof_bits_v<T>>;

    T* smem_base = reinterpret_cast<T*>(smem_buf);

    // Smem1: row-major — contiguous dim 0 (for phase 1 vectorization)
    auto smem1 = make_tensor(make_smem_ptr(smem_base),
        make_layout(make_shape(Int<kTileDim>{}, Int<kTileDim>{}),
                          make_stride(Int<1>{}, Int<kStride>{})));

    // Smem2: col-major — contiguous dim 1 (for phase 2 vectorization)
    auto smem2 = make_tensor(make_smem_ptr(smem_base + kTileDim * kStride),
        make_layout(make_shape(Int<kTileDim>{}, Int<kTileDim>{}),
                          make_stride(Int<kStride>{}, Int<1>{})));

    // Decode blockIdx.z → multi-dim batch coord → per-block pointer offsets.
    // For rank == 2, batch_shape is empty, batch_coord is empty, and offsets are 0.
    constexpr int kRank = rank_v<SrcLayout>;
    auto batch_shape   = take<2, kRank>(shape(src));
    auto src_batch_str = take<2, kRank>(stride(src));
    auto dst_batch_str = take<2, kRank>(stride(dst));
    auto batch_coord   = idx2crd(int64_t(blockIdx.z), batch_shape);
    int64_t src_off = crd2idx(batch_coord, batch_shape, src_batch_str);
    int64_t dst_off = crd2idx(batch_coord, batch_shape, dst_batch_str);

    // 2D view of the (I, J) plane at this batch's offset. The static Int<1>
    // strides at src dim 0 / dst dim 1 are preserved (they're part of the
    // CuTe stride tuple's static type), keeping the smem partitioning
    // identical to the original 2D path.
    auto src_2d = make_tensor(
        make_gmem_ptr(raw_pointer_cast(src.data()) + src_off),
        make_layout(make_shape(shape<0>(src), shape<1>(src)),
                    make_stride(stride<0>(src), stride<1>(src))));
    auto dst_2d = make_tensor(
        make_gmem_ptr(raw_pointer_cast(dst.data()) + dst_off),
        make_layout(make_shape(shape<0>(dst), shape<1>(dst)),
                    make_stride(stride<0>(dst), stride<1>(dst))));

    // Tile the 2D plane (existing logic, unchanged from here on)
    auto tiler = make_shape(Int<kTileDim>{}, Int<kTileDim>{});
    auto src_tiled = tiled_divide(src_2d, tiler);
    auto dst_tiled = tiled_divide(dst_2d, tiler);

    // Bounds check on tile grid
    if (blockIdx.y >= size<1>(src_tiled) ||
        blockIdx.x >= size<2>(src_tiled)) return;

    auto src_tile = src_tiled(make_coord(_, _), blockIdx.y, blockIdx.x);
    auto dst_tile = dst_tiled(make_coord(_, _), blockIdx.y, blockIdx.x);

    // Phase 1: gmem(src) -> smem1, vectorize along dim 0
    auto tc1 = make_tiled_copy(
        Copy_Atom<UniversalCopy<VecT>, T>{},
        make_layout(make_shape(Int<kThrRows>{}, Int<kTileDim>{})),
        make_layout(make_shape(Int<kVec>{}, Int<1>{})));
    auto thr1 = tc1.get_slice(threadIdx.x);
    copy(tc1, thr1.partition_S(src_tile), thr1.partition_D(smem1));

    __syncthreads();

    // In-smem: smem1 -> smem2 (physical layout conversion, same logical data)
    auto tc_s = make_tiled_copy(
        Copy_Atom<UniversalCopy<T>, T>{},
        make_layout(make_shape(Int<16>{}, Int<16>{})),
        make_layout(make_shape(Int<1>{}, Int<1>{})));
    auto thr_s = tc_s.get_slice(threadIdx.x);
    copy(tc_s, thr_s.partition_S(smem1), thr_s.partition_D(smem2));

    __syncthreads();

    // Phase 2: smem2 -> gmem(dst), vectorize along dim 1
    auto tc2 = make_tiled_copy(
        Copy_Atom<UniversalCopy<VecT>, T>{},
        make_layout(make_shape(Int<kTileDim>{}, Int<kThrRows>{}),
                       make_stride(Int<kThrRows>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<kVec>{})));
    auto thr2 = tc2.get_slice(threadIdx.x);
    copy(tc2, thr2.partition_S(smem2), thr2.partition_D(dst_tile));
}
```

- [ ] **Step 2: Build**

```bash
cd /data/lmdeploy-copy/build && ninja _turbomind
```

Expected: clean build. The host `TransposeCopy` (Task 4) still passes a rank-2 tensor at this point, so the new code paths reduce to identity for rank 2.

- [ ] **Step 3: Run tests for regression**

```bash
PYTHONPATH=/data/lmdeploy-copy/lmdeploy:/data/lmdeploy-copy/build/lib \
  python /data/lmdeploy-copy/test_generic_copy.py
```

Expected: every case (including the rank-2 transpose cases that exercise this kernel today) prints `[PASS]`. The new multi-dim cases still go through `VectorizedCopy` at this point — they remain `[PASS]` via the fallback.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/kernels/copy/transpose.cu
git commit -m "refactor(copy): rank-generic TransposeCopyKernel body

Decode blockIdx.z into per-block pointer offsets via idx2crd/crd2idx
on the batch dims (positions 2..rank-1). For rank == 2 the batch tuple
is empty and offsets are zero, reducing to the original 2D path."
```

---

## Task 4: Add per-dtype unit-stride helper + rank dispatch in host `TransposeCopy`

**Files:**

- Modify: `src/turbomind/kernels/copy/transpose.cu` (the namespace and host function around lines 91–137)

- [ ] **Step 1: Add the local helpers**

Add this `detail` namespace just above the host `TransposeCopy` definition (replacing the empty space between `}  // namespace kernel` on line 89 and the host comment block on line 91):

```cpp
namespace detail {

// Build a CuTe Shape tuple from a runtime ssize_t array.
template<size_t... Is>
auto make_cute_shape_impl(const ssize_t* data, std::index_sequence<Is...>)
{
    return make_shape(static_cast<int32_t>(data[Is])...);
}

template<int kRank>
auto make_cute_shape(const ssize_t* data)
{
    return make_cute_shape_impl(data, std::make_index_sequence<kRank>{});
}

// Per-element selector: Int<1>{} at UnitPos, otherwise dynamic int64_t.
template<int UnitPos, size_t I>
auto stride_elem(const ssize_t* data)
{
    if constexpr (I == UnitPos) {
        return cute::Int<1>{};
    } else {
        return static_cast<int64_t>(data[I]);
    }
}

// Build a CuTe Stride tuple of length kRank with Int<1>{} at UnitPos and
// dynamic int64_t at all other positions.
template<int UnitPos, size_t... Is>
auto make_unit_stride_impl(const ssize_t* data, std::index_sequence<Is...>)
{
    return cute::make_stride(stride_elem<UnitPos, Is>(data)...);
}

template<int UnitPos, int kRank>
auto make_unit_stride(const ssize_t* data)
{
    return make_unit_stride_impl<UnitPos>(data, std::make_index_sequence<kRank>{});
}

}  // namespace detail
```

- [ ] **Step 2: Replace the host `TransposeCopy` body**

Replace lines 94–137 (the entire `TransposeCopy` host function) with:

```cpp
void TransposeCopy(const void* data_a, void* data_b,
                   const Layout& a, const Layout& b,
                   DataType dtype, cudaStream_t stream)
{
    const int rank = a.rank();
    int32_t M = static_cast<int32_t>(a.shape(0));
    int32_t N = static_cast<int32_t>(a.shape(1));

    auto launch = [&](auto t, auto kvec, auto ktiledim, auto rank_c) {
        using T = decltype(t);
        constexpr int kVec     = decltype(kvec)::value;
        constexpr int kTileDim = decltype(ktiledim)::value;
        constexpr int kRank    = decltype(rank_c)::value;

        if (M % kTileDim || N % kTileDim) {
            TM_LOG_WARNING("TransposeCopy: shape ({}, {}) not divisible by tile {}",
                           M, N, kTileDim);
            return;
        }

        auto data_shape  = detail::make_cute_shape<kRank>(a.shape().data());
        auto src_strides = detail::make_unit_stride<0, kRank>(a.stride().data());
        auto dst_strides = detail::make_unit_stride<1, kRank>(b.stride().data());

        auto src_gmem = make_tensor(
            make_gmem_ptr(reinterpret_cast<const T*>(data_a)),
            make_layout(data_shape, src_strides));
        auto dst_gmem = make_tensor(
            make_gmem_ptr(reinterpret_cast<T*>(data_b)),
            make_layout(data_shape, dst_strides));

        int64_t total_batch = 1;
        for (int i = 2; i < kRank; ++i) total_batch *= a.shape(i);

        constexpr int smem_bytes = 2 * kTileDim * (kTileDim + kVec) * sizeof(T);
        dim3 grid(static_cast<uint32_t>(N / kTileDim),
                  static_cast<uint32_t>(M / kTileDim),
                  static_cast<uint32_t>(total_batch));

        kernel::TransposeCopyKernel<kTileDim, kVec>
            <<<grid, 256, smem_bytes, stream>>>(src_gmem, dst_gmem);
    };

    auto dispatch_rank = [&](auto t, auto kvec, auto ktiledim) {
        switch (rank) {
            case 2: return launch(t, kvec, ktiledim, std::integral_constant<int, 2>{});
            case 3: return launch(t, kvec, ktiledim, std::integral_constant<int, 3>{});
            case 4: return launch(t, kvec, ktiledim, std::integral_constant<int, 4>{});
            default:
                TM_LOG_WARNING("TransposeCopy: rank {} not supported", rank);
                return;
        }
    };

    switch (byte_size(dtype)) {
        case 1: return dispatch_rank(uint8_t{},  Int<16>{}, Int<64>{});
        case 2: return dispatch_rank(uint16_t{}, Int<8>{},  Int<64>{});
        case 4: return dispatch_rank(uint32_t{}, Int<4>{},  Int<32>{});
        case 8: return dispatch_rank(uint64_t{}, Int<2>{},  Int<32>{});
        default:
            TM_CHECK(0) << "TransposeCopy: unsupported element size " << byte_size(dtype);
            break;
    }
}
```

- [ ] **Step 3: Build**

```bash
cd /data/lmdeploy-copy/build && ninja _turbomind
```

Expected: clean build. There are now 12 kernel instantiations (4 dtypes × 3 ranks); compile time may rise slightly.

- [ ] **Step 4: Run tests for regression**

```bash
PYTHONPATH=/data/lmdeploy-copy/lmdeploy:/data/lmdeploy-copy/build/lib \
  python /data/lmdeploy-copy/test_generic_copy.py
```

Expected: all cases `[PASS]`. The rank-2 cases now go through the new dispatch (with `kRank=2`); the multi-dim cases still go through `VectorizedCopy` (Task 5 enables them).

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/kernels/copy/transpose.cu
git commit -m "feat(copy): TransposeCopy host dispatch over rank in {2,3,4}

Add detail::make_unit_stride to build CuTe stride tuples with a static
Int<1> at a chosen position; dispatch to TransposeCopyKernel with
kRank ∈ {2,3,4}. Behavior for rank 2 is unchanged."
```

---

## Task 5: Update `copy.cc::GenericCopy` dispatcher

Replace the existing `is_2d_transpose` block with the multi-dim canonicalization. Also fix the latent divisibility-check mismatch by using the per-dtype tile size (`64` for byte sizes ≤ 2, `32` otherwise) instead of the hardcoded `32`.

**Files:**

- Modify: `src/turbomind/kernels/copy/copy.cc` (lines 53–65, the `is_2d_transpose` block)

- [ ] **Step 1: Replace the dispatch block**

Replace lines 53–65 (from the `// --- 2D transpose detection ---` comment through the `}` closing the `if (is_2d_transpose && ...)` block) with:

```cpp
    // --- Transpose detection (2D + batched) ---
    // After the src-stride-ascending sort above, position 0 holds the smallest
    // src stride (innermost). We dispatch to TransposeCopy when:
    //   - position 0 has src stride 1 (call it I),
    //   - some position J ∈ [1, rank-1] has dst stride 1,
    //   - both shape(0) and shape(J) are divisible by the per-dtype tile.
    // Then swap J → position 1 to get canonical (I=0, J=1, batch...) and
    // coalesce adjacent batch dims that are proportional in both a and b.
    const int kTileDim = byte_size(dtype) <= 2 ? 64 : 32;

    int J = -1;
    for (int i = 1; i < rank; ++i) {
        if (b.stride(i) == 1) { J = i; break; }
    }

    bool is_transpose =
        (J >= 1) &&
        (a.stride(0) == 1) && (a.stride(J) > 1) && (b.stride(0) > 1) &&
        (a.shape(0) % kTileDim == 0) &&
        (a.shape(J) % kTileDim == 0);

    if (is_transpose) {
        if (J != 1) { a = a.transpose(1, J); b = b.transpose(1, J); }
        std::tie(a, b) = coalesce_batch_dims(a, b);
        if (a.rank() <= 4) {
            TransposeCopy(src.raw_data(), dst.raw_data(), a, b, dtype, stream);
            return;
        }
    }
```

(Note: the `coalesce_batch_dims` declaration must be visible at this point — Task 2 added it as a `static` function above `GenericCopy`, so this is satisfied.)

- [ ] **Step 2: Build**

```bash
cd /data/lmdeploy-copy/build && ninja _turbomind
```

Expected: clean build. If `[[maybe_unused]]` was added in Task 2, it's now redundant — leave it in (it's a no-op when used).

- [ ] **Step 3: Run the full test, all dtypes**

```bash
cd /data/lmdeploy-copy
for dt in i8 f16 f32 i64; do
  echo "=== dtype=$dt ==="
  PYTHONPATH=$PWD/lmdeploy:$PWD/build/lib python test_generic_copy.py --dtype $dt
done
```

Expected for each dtype: `ALL TESTS PASSED`.

This is the milestone task: the multi-dim batched cases now flow through `TransposeCopy` (rank 3 or 4) rather than `VectorizedCopy`. If a previously-passing case starts to `[FAIL]`, the bug is in this change — investigate by:

- Confirming the canonicalization preserves shape (`a.size()` should be unchanged after `transpose` and `coalesce_batch_dims`).
- Confirming the kernel sees an `I != J` layout: `a.stride(0) == 1`, `b.stride(1) == 1`, `a.shape(0) % kTileDim == 0`, `a.shape(1) % kTileDim == 0`.
- Adding a `TM_LOG_INFO` line in `TransposeCopy` printing `rank` and `total_batch` to verify dispatch.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/kernels/copy/copy.cc
git commit -m "feat(copy): dispatch batched transposes to TransposeCopy

GenericCopy now finds J (any dim with dst stride 1), swaps it to
position 1, coalesces adjacent batch dims jointly in src/dst, and
dispatches to multi-rank TransposeCopy when post-coalesce rank ≤ 4.
The divisibility check uses the per-dtype tile size (64 for ≤2-byte
dtypes, 32 otherwise), fixing a latent dispatch mismatch with
transpose.cu's per-dtype tile constants."
```

---

## Task 6: Add a batched-transpose throughput sweep

Confirms the new path is being exercised AND faster than the prior `VectorizedCopy` fallback.

**Files:**

- Modify: `test_generic_copy.py` (the throughput sweep section around lines 204–227)

- [ ] **Step 1: Add a batched sweep**

Insert this block in the `--- Throughput sweep ---` section, immediately after the existing 2D contig/trans loop (after line 227, before the summary table on line 229):

```python
    # --- Batched transpose throughput sweep ---
    sweep_batched = []
    for (b_, m, n) in [(8, 1024, 1024), (32, 512, 512), (8, 4096, 128)]:
        numel = b_ * m * n
        size_label = f"{numel // (1024 * 1024)}M" if numel >= 1024 * 1024 else f"{numel // 1024}K"
        shape_label = f"{b_}x{m}x{n}"

        total += 1
        ok, bench = run_test(f"batched-trans {size_label} ({shape_label})",
                             _rand(b_, m, n).transpose(1, 2))
        if ok:
            passed += 1
            sweep_batched.append((shape_label, *bench))
        else:
            all_passed = False
```

And add a corresponding section in the summary table, after the existing `print("  Transpose:")` block (around line 237):

```python
    print("  Batched transpose:")
    for shape, gc, pt, pct in sweep_batched:
        print(f"  {shape:<14} {gc:>12.1f} {pt:>12.1f} {pct:>7.1f}%")
```

- [ ] **Step 2: Run the sweep, observe perf**

```bash
PYTHONPATH=/data/lmdeploy-copy/lmdeploy:/data/lmdeploy-copy/build/lib \
  python /data/lmdeploy-copy/test_generic_copy.py --dtype f16
```

Expected:

- All checks `[PASS]`.
- The "Batched transpose" rows in the summary should show GenericCopy GB/s comparable to the 2D "Transpose" rows of the same total byte size (i.e., the smem-staged path is being used).
- `%PT` (vs PyTorch) should be in the same ballpark as the 2D transpose `%PT` for the same dtype.

If the batched throughput is materially lower than the 2D throughput at the same total byte volume, the multi-dim path likely is NOT being taken — re-check the dispatch logic. Add a temporary `TM_LOG_INFO` in `TransposeCopy` to confirm.

- [ ] **Step 3: Run the same sweep on i8 (the dtype with the largest expected gain)**

```bash
PYTHONPATH=/data/lmdeploy-copy/lmdeploy:/data/lmdeploy-copy/build/lib \
  python /data/lmdeploy-copy/test_generic_copy.py --dtype i8
```

Expected: batched-transpose GenericCopy GB/s should be **substantially higher** than the same shape/volume would have been with `VectorizedCopy` (which was ≈25-50% of peak for transposed i8). A reasonable target is ≥80% of the 2D transpose throughput at the same dtype/byte volume.

- [ ] **Step 4: Commit**

```bash
git add test_generic_copy.py
git commit -m "test(copy): add batched-transpose throughput sweep"
```

---

## Self-Review

**1. Spec coverage:**

- Spec: "find J at any position, treat others as batch" → Task 5 (`for i in [1, rank-1]: if b.stride(i)==1: J=i`).
- Spec: "swap J to position 1, canonical form" → Task 5 (`if (J != 1) a = a.transpose(1, J)`).
- Spec: "joint batch coalescing" → Task 2 (`coalesce_batch_dims`), wired in Task 5.
- Spec: "rank-generic kernel body via take/idx2crd/crd2idx" → Task 3.
- Spec: "host dispatch over rank ∈ {2,3,4}" → Task 4.
- Spec: "12 kernel instantiations" → Task 4 (4 dtypes × 3 ranks switch).
- Spec: "static Int<1> stride at I (src dim 0) and J (dst dim 1)" → Task 4 (`make_unit_stride<0,…>` and `<1,…>`).
- Spec: "per-dtype tile size (64 for ≤2-byte, 32 otherwise)" → Task 5 (replaces hardcoded 32).
- Spec: "fall through to VectorizedCopy when J not found / rank>4 / not aligned" → Task 5 (the `if (is_transpose)` guard around the dispatch + the rank cap).
- Spec: "regression tests cover 3D batched / 4D batched / non-adjacent J / non-coalesceable batch / unaligned" → Task 1.
- Spec: "throughput sweep entry" → Task 6.

All spec requirements covered.

**2. Placeholder scan:** No "TBD"/"TODO"/"add appropriate"/"similar to Task N"/"write tests for the above" patterns. All steps include exact code or exact commands.

**3. Type consistency:**

- `coalesce_batch_dims(const Layout&, const Layout&) -> std::pair<Layout, Layout>` — declared in Task 2, called in Task 5 with matching signature.
- `TransposeCopy(const void*, void*, const Layout&, const Layout&, DataType, cudaStream_t)` — unchanged from existing forward decl; matches caller in Task 5.
- Kernel template params `<int kTileDim, int kVec, SrcEng, SrcLay, DstEng, DstLay>` unchanged; the rank lives inside `SrcLay`/`DstLay`.
- `detail::make_unit_stride<UnitPos, kRank>(const ssize_t*)` declared in Task 4, called twice in Task 4 with `UnitPos = 0` (src) and `UnitPos = 1` (dst).
- `detail::make_cute_shape<kRank>(const ssize_t*)` declared in Task 4, called once in Task 4. Note: this is a local copy of the same-named helper in `copy.cu`'s `detail` namespace — they're separate translation units, so no link conflict.

All consistent.
