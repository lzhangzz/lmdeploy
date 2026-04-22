# DataFormat / WeightFormat adaption — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the `LinearWeight` lifecycle into single-phase construction from a `LinearConfig` that carries a full weight `DataFormat`. Delete `set_weight_spec` / `configure` / `LinearPolicy` / `MakeLinearWeightFormat`'s overloaded signature. Revive `Linear.data_format` as the authoritative Python-to-C++ bridge. Purge the `group_size` thread on the Python side. Fix the latent `_dequant_fp8` hardcoded-BF16 bug by plumbing `data_type` through dequant pipelines.

**Architecture:** One bridge across the seam — `WeightFormat.make_data_format(data_type) -> _tm.DataFormat` — populates `Linear.data_format` at `build_linear` time; `LinearConfig.format` carries the DataFormat into C++; the `LinearWeight` ctor derives all three DataFormats (weight / input / output) via a new `DeriveActivationFormats` function. `DataFormat.block_sizes` is stored in tensor-shape order `{block_in, block_out}`. `converter.py` resolves the active `WeightFormat` exactly once with concrete block sizes, then hands it to the spec; specs no longer carry `_group_size`.

**Tech Stack:** C++ (CUDA host code, pybind11), Python 3, pytest, catch2 (C++ unit tests), ninja build system.

**Spec:** `docs/superpowers/specs/2026-04-22-data-format-weight-format-adaption-design.md`.

**Commit cadence:** five atomic commits in order, one per task. Each commit is independently testable; the structural collapse lives in Task 4.

---

## Files changed overview

| File | Tasks that touch it |
|---|---|
| `src/turbomind/core/data_format.h` | 1 |
| `src/turbomind/core/data_format.cc` | 1 |
| `src/turbomind/core/test_data_format.cc` | 1 |
| `src/turbomind/core/CMakeLists.txt` | 1 (wire up test_data_format binary) |
| `src/turbomind/models/linear_weight.h` | 1, 4 |
| `src/turbomind/models/linear_weight.cc` | 1, 4 |
| `src/turbomind/models/moe_weight.cc` | 4 |
| `src/turbomind/python/bind.cpp` | 1, 4 |
| `lmdeploy/turbomind/deploy/kind_map.py` | 2, 3, 5 |
| `lmdeploy/turbomind/deploy/linear.py` | 3 |
| `lmdeploy/turbomind/deploy/builder/_base.py` | 2, 3, 4 |
| `lmdeploy/turbomind/deploy/builder/attention.py` | 3 |
| `lmdeploy/turbomind/deploy/builder/deltanet.py` | 3 |
| `lmdeploy/turbomind/deploy/builder/mla.py` | 3 |
| `lmdeploy/turbomind/deploy/spec.py` | 2, 5 |
| `lmdeploy/turbomind/deploy/converter.py` | 5 |
| `lmdeploy/turbomind/deploy/source_model/utils.py` | 3, 5 |
| `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` | 5 |
| `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` | 2 (direct build_linear callers), 5 |
| `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` | 2, 3, 5 |
| `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` | 3, 5 |
| `tests/test_lmdeploy/test_turbomind/test_transform_tensors.py` | 3 |

**Out of scope:** `src/turbomind/kernels/gemm/test/testbed_v3.h` — stale; references the removed `LinearWeight::resolved_` field; treat as if it does not exist.

---

## Task 1: `ResolveLinearWeightFormat` factory — rename, explicit block sizes, tensor-shape ordering

**Files:**
- Modify: `src/turbomind/core/data_format.h`
- Modify: `src/turbomind/core/data_format.cc`
- Modify: `src/turbomind/core/test_data_format.cc`
- Modify: `src/turbomind/core/CMakeLists.txt` (add `test_data_format` executable target so tests actually run)
- Modify: `src/turbomind/models/linear_weight.h` (declaration sync)
- Modify: `src/turbomind/models/linear_weight.cc` (readers flip to `block_sizes[0]` for K-axis; `set_weight_spec` bridges to new factory)
- Modify: `src/turbomind/python/bind.cpp` (rename factory binding)

**Out of scope:** `LinearConfig.format`, `set_weight_spec` deletion — those come in Task 4. Task 1 keeps `set_weight_spec` alive as a thin shim that translates old single-arg `group_size` to new explicit `(block_in, block_out)` per weight_dtype.

- [ ] **Step 1: Wire `test_data_format` into CMake**

Modify `src/turbomind/core/CMakeLists.txt` — replace the existing `if (BUILD_TEST)` block with:

```cmake
if (BUILD_TEST)
    add_executable(test_core test_core.cc)
    target_link_libraries(test_core PRIVATE core logger Catch2::Catch2WithMain)

    add_executable(test_data_format test_data_format.cc)
    target_link_libraries(test_data_format PRIVATE core logger Catch2::Catch2WithMain)
endif ()
```

- [ ] **Step 2: Rebuild to confirm the existing test_data_format binary builds from source**

```bash
cd build && ninja test_data_format
```

Expected: binary built successfully at `build/bin/test_data_format`. If CMake reconfiguration is needed first:

```bash
cd build && cmake --build . --target test_data_format
```

- [ ] **Step 3: Run the old (pre-refactor) tests to establish a baseline**

```bash
./build/bin/test_data_format
```

Expected: all tests pass (old signatures, old `[N,K]` ordering).

- [ ] **Step 4: Update `src/turbomind/core/data_format.h` — new factory declaration**

Replace the existing `MakeLinearWeightFormat` declaration (line 38) with:

```cpp
/// Construct the DataFormat for a linear weight tensor in TM [in, out] layout.
/// block_sizes stored in tensor-shape order: {block_in, block_out}, so
/// block_sizes[0] is the K-axis group size and block_sizes[1] is the N-axis.
/// Scales / zeros dtypes are derived from (data_type, weight_dtype) per the
/// format's GEMM convention. Validates that the combination is supported.
DataFormat ResolveLinearWeightFormat(DataType data_type,
                                     DataType weight_dtype,
                                     int      block_in,
                                     int      block_out);
```

- [ ] **Step 5: Update `src/turbomind/core/data_format.cc` — new factory body, tensor-shape ordering**

Replace the entire `MakeLinearWeightFormat` function body with:

```cpp
DataFormat ResolveLinearWeightFormat(DataType data_type,
                                     DataType weight_dtype,
                                     int      block_in,
                                     int      block_out)
{
    DataFormat fmt;
    fmt.dtype = weight_dtype;

    if (IsTrivialFloatType(weight_dtype)) {
        TM_CHECK(block_in == 1 && block_out == 1)
            << "Trivial float weight requires block_in==1 and block_out==1, got "
            << block_in << ", " << block_out;
        fmt.block_sizes = {1, 1};
        return fmt;
    }

    if (weight_dtype == kFloat8_e4m3) {
        TM_CHECK(block_in == 128 && block_out == 128)
            << "FP8 weight format requires block_in==128 and block_out==128, got "
            << block_in << ", " << block_out;
        fmt.block_sizes  = {128, 128};
        fmt.scales.dtype = kFloat;
        return fmt;
    }

    if (weight_dtype == kFloat4_e2m1) {
        TM_CHECK(block_in > 0 && block_out == 1)
            << "FP4 weight format requires block_in>0 and block_out==1, got "
            << block_in << ", " << block_out;
        fmt.block_sizes  = {block_in, 1};
        fmt.scales.dtype = kUint8;
        return fmt;
    }

    const bool is_qweight = weight_dtype == kUint4 || weight_dtype == kUint8;
    if (is_qweight) {
        TM_CHECK(block_in > 0 && block_in <= 256 && block_out == 1)
            << "Quantized integer weight requires 0 < block_in <= 256 and block_out==1, got "
            << block_in << ", " << block_out;
        fmt.block_sizes  = {block_in, 1};
        fmt.scales.dtype = data_type;
        fmt.zeros.dtype  = data_type;
        return fmt;
    }

    TM_CHECK(0) << "Unsupported weight format: " << to_string(weight_dtype);
    return fmt;
}
```

Note: `block_sizes[0]` is now the K-axis group size; `block_sizes[1]` is the N-axis. This is a behavior change for anything that reads `block_sizes` by index.

- [ ] **Step 6: Update `src/turbomind/core/test_data_format.cc` — new signature + ordering**

Replace each `MakeLinearWeightFormat(...)` call and REQUIRE tuple:

```cpp
TEST_CASE("DataFormat trivial is not quantized", "[data_format]")
{
    DataFormat fmt = ResolveLinearWeightFormat(kHalf, kHalf, 1, 1);
    REQUIRE(!fmt.is_quantized());
    REQUIRE(fmt.rank() == 2);
    REQUIRE(fmt.block_sizes == std::vector<int>{1, 1});
    REQUIRE(!fmt.scales.present());
    REQUIRE(!fmt.zeros.present());
}

TEST_CASE("DataFormat FP8 blocked", "[data_format]")
{
    DataFormat fmt = ResolveLinearWeightFormat(kHalf, kFloat8_e4m3, 128, 128);
    REQUIRE(fmt.is_quantized());
    REQUIRE(fmt.dtype == kFloat8_e4m3);
    REQUIRE(fmt.block_sizes == std::vector<int>{128, 128});
    REQUIRE(fmt.scales.present());
    REQUIRE(fmt.scales.dtype == kFloat);
    REQUIRE(!fmt.zeros.present());
}

TEST_CASE("DataFormat FP4", "[data_format]")
{
    DataFormat fmt = ResolveLinearWeightFormat(kHalf, kFloat4_e2m1, 128, 1);
    REQUIRE(fmt.is_quantized());
    REQUIRE(fmt.dtype == kFloat4_e2m1);
    REQUIRE(fmt.block_sizes == std::vector<int>{128, 1});
    REQUIRE(fmt.scales.present());
    REQUIRE(fmt.scales.dtype == kUint8);
    REQUIRE(!fmt.zeros.present());
}

TEST_CASE("DataFormat AWQ uint4", "[data_format]")
{
    DataFormat fmt = ResolveLinearWeightFormat(kHalf, kUint4, 128, 1);
    REQUIRE(fmt.is_quantized());
    REQUIRE(fmt.dtype == kUint4);
    REQUIRE(fmt.block_sizes == std::vector<int>{128, 1});
    REQUIRE(fmt.scales.present());
    REQUIRE(fmt.scales.dtype == kHalf);
    REQUIRE(fmt.zeros.present());
    REQUIRE(fmt.zeros.dtype == kHalf);
}

TEST_CASE("DataFormat uint8 quantized", "[data_format]")
{
    DataFormat fmt = ResolveLinearWeightFormat(kBfloat16, kUint8, 64, 1);
    REQUIRE(fmt.is_quantized());
    REQUIRE(fmt.block_sizes == std::vector<int>{64, 1});
    REQUIRE(fmt.scales.dtype == kBfloat16);
    REQUIRE(fmt.zeros.dtype == kBfloat16);
}

TEST_CASE("DataFormat trivial BF16", "[data_format]")
{
    DataFormat fmt = ResolveLinearWeightFormat(kBfloat16, kBfloat16, 1, 1);
    REQUIRE(!fmt.is_quantized());
    REQUIRE(fmt.dtype == kBfloat16);
}
```

The first test case (`DataFormat default is not quantized`) does not call the factory and stays unchanged.

- [ ] **Step 7: Build and run test_data_format to verify new signature + ordering**

```bash
cd build && ninja test_data_format && ./bin/test_data_format
```

Expected: all 7 test cases pass. If a `TM_CHECK` fires during tests, the factory validation is inconsistent with the expected block_sizes — reread Step 5 carefully.

- [ ] **Step 8: Update `LinearWeight::set_weight_spec` to call the new factory**

`set_weight_spec` keeps its old single-arg `(DataType weight_dtype, int group_size)` signature through Task 1 (called from Python). Its body now translates single `group_size` into `(block_in, block_out)` via the same per-weight_dtype rules the old factory used.

In `src/turbomind/models/linear_weight.cc`, replace the body of `set_weight_spec` with:

```cpp
void LinearWeight::set_weight_spec(DataType weight_dtype, int group_size)
{
    // For trivial float weights, coerce to model compute dtype
    if (weight_dtype != data_type && IsTrivialFloatType(weight_dtype) && IsTrivialFloatType(data_type)) {
        weight_dtype = data_type;
    }
    weight_format = weight_dtype;
    this->group_size = group_size;

    // Translate the legacy single-arg group_size to explicit (block_in, block_out).
    // This shim survives only until Task 4 deletes set_weight_spec entirely.
    int block_in, block_out;
    if (IsTrivialFloatType(weight_dtype)) {
        block_in = block_out = 1;
    }
    else if (weight_dtype == kFloat8_e4m3) {
        block_in = block_out = 128;
    }
    else {  // kFloat4_e2m1 / kUint4 / kUint8 — K-grouped
        block_in = group_size;
        block_out = 1;
    }
    format = ResolveLinearWeightFormat(data_type, weight_format, block_in, block_out);
    policy = ResolveLinearPolicy(format, data_type, getSMVersion());
}
```

- [ ] **Step 9: Flip `block_sizes[1]` → `block_sizes[0]` in `ResolveLinearPolicy` (K-axis reads)**

In `src/turbomind/models/linear_weight.cc::ResolveLinearPolicy`, each `format.block_sizes[1]` referring to the K-axis group size becomes `format.block_sizes[0]`. Three sites:

```cpp
if (format.dtype == kFloat8_e4m3) {
    int gs = format.block_sizes[0];                            // was [1]
    p.weight_quant = gemm::QuantDesc{gemm::QuantType::kB, gs};
    if (sm == 90) {
        p.input_dtype  = kFloat8_e4m3;
        p.input_quant  = gemm::QuantDesc{gemm::QuantType::kK, gs};
    }
    return p;
}

if (format.dtype == kFloat4_e2m1) {
    int gs = format.block_sizes[0];                            // was [1]
    p.weight_quant = gemm::QuantDesc{gemm::QuantType::kK, gs};
    return p;
}

if (format.dtype == kUint4 || format.dtype == kUint8) {
    int gs = format.block_sizes[0];                            // was [1]
    p.weight_quant = gemm::QuantDesc{gemm::QuantType::kK, gs};
    return p;
}
```

- [ ] **Step 10: Update `prepare()` reader to read K-axis group size from `block_sizes[0]`**

In `src/turbomind/models/linear_weight.cc::prepare()`, the `s_desc` layout uses `group_size` (member field). Under Task 1 `group_size` is still populated by `set_weight_spec`, so this line does not change yet. Task 4 replaces it. No edit this step — this checkbox documents the intentional no-op.

- [ ] **Step 11: Update pybind binding — rename factory**

In `src/turbomind/python/bind.cpp` around line 440, replace:

```cpp
m.def("MakeLinearWeightFormat", &turbomind::MakeLinearWeightFormat,
      py::arg("data_type"), py::arg("weight_format"), py::arg("group_size"));
```

with:

```cpp
m.def("ResolveLinearWeightFormat", &turbomind::ResolveLinearWeightFormat,
      py::arg("data_type"),
      py::arg("weight_dtype"),
      py::arg("block_in"),
      py::arg("block_out"));
```

- [ ] **Step 12: Build the whole project**

```bash
cd build && ninja
```

Expected: clean build. The only Python caller of `_tm.MakeLinearWeightFormat` is the dead `WeightFormat.to_data_format` method (kind_map.py line 104) which is not invoked at runtime — but it will `AttributeError` the moment someone *does* call it. That's acceptable for Task 1; Task 2 replaces the method.

- [ ] **Step 13: Run `test_data_format` one more time**

```bash
./build/bin/test_data_format
```

Expected: all pass.

- [ ] **Step 14: Integration smoke test with any trivial model**

Per AGENTS.md: check GPU availability first (`get_gpu_usage` MCP tool), then pick a small trivial-BF16 model via `list_models`. Run `scripts/test_turbomind_model.py` with a simple prompt; require ≥128 tokens of coherent, prompt-relevant output.

Rationale for Task 1: Task 1 preserves behavior on the Python-facing side (`set_weight_spec` still works; Python still calls it the same way). The smoke test confirms the new factory + flipped indexing inside `ResolveLinearPolicy` produce bit-identical kernel selections.

- [ ] **Step 15: Commit**

```bash
cd /data/lmdeploy-modeling && git add \
    src/turbomind/core/data_format.h \
    src/turbomind/core/data_format.cc \
    src/turbomind/core/test_data_format.cc \
    src/turbomind/core/CMakeLists.txt \
    src/turbomind/models/linear_weight.cc \
    src/turbomind/python/bind.cpp \
  && git commit -m "refactor(core): ResolveLinearWeightFormat with explicit block sizes; tensor-shape ordered block_sizes"
```

---

## Task 2: Python bridge — `WeightFormat.make_data_format`; populate `Linear.data_format`; `build_linear` takes `data_type`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/kind_map.py`
- Modify: `lmdeploy/turbomind/deploy/builder/_base.py`
- Modify: `lmdeploy/turbomind/deploy/spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` (direct `build_linear` calls)
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` (direct `build_linear` call)

**Out of scope:** `_group_size` elimination (Task 5), dequant-pipeline `data_type` plumbing (Task 3), `Linear` dataclass tightening to required fields (Task 3).

- [ ] **Step 1: Rewrite `WeightFormat.to_data_format` as `make_data_format(data_type)` in `kind_map.py`**

Replace the existing `to_data_format` method (lines 89-105) with:

```python
def make_data_format(self, data_type) -> "_tm.DataFormat":
    """Construct the C++ DataFormat describing this checkpoint format's
    weight storage, resolved for the given model activation dtype.

    Delegates format-validity rules to the C++ factory
    (``ResolveLinearWeightFormat``), which is the source of truth on which
    (weight_dtype, block_sizes) combinations are supported on this GPU.
    """
    if self.cpp_dtype_name is None:
        # trivial: weight dtype equals activation dtype, no blocking
        return _tm.ResolveLinearWeightFormat(data_type, data_type, 1, 1)
    weight_dtype = getattr(_tm.DataType, self.cpp_dtype_name)
    return _tm.ResolveLinearWeightFormat(
        data_type,
        weight_dtype,
        self.block_in  or 1,
        self.block_out or 1,
    )
```

- [ ] **Step 2: Add `data_type` kwarg to `build_linear` and populate `Linear.data_format`**

In `kind_map.py`, modify the `build_linear` function signature and body:

```python
def build_linear(
    params: dict[str, torch.Tensor],
    prefix: str,
    *,
    data_type,                      # new — the model activation dtype
    index: int | None = None,
    block_in: int = 0,
    block_out: int = 0,
) -> Linear | None:
    """Build a Linear bundle from checkpoint tensors at *prefix*.

    *data_type* is the model's activation dtype; passed through to
    ``WeightFormat.make_data_format`` to construct the C++ DataFormat that
    is stashed on ``Linear.data_format`` and later handed to C++ via
    ``LinearConfig.format``.
    """
    from .linear import Linear

    available: dict[str, torch.Tensor] = {
        s: params[prefix + s] for s in ALL_SUFFIXES if (prefix + s) in params
    }
    if index is not None:
        available = {s: t[index] for s, t in available.items()}

    fmt = next((f for f in FORMAT_PRIORITY if f.accepts(available)), None)
    if fmt is None:
        return None

    replacements: dict[str, int] = {}
    if fmt.block_in == 0 and block_in > 0:
        replacements['block_in'] = block_in
    if fmt.block_out == 0 and block_out > 0:
        replacements['block_out'] = block_out
    if replacements:
        fmt = replace(fmt, **replacements)

    tensors: dict[str, torch.Tensor] = {
        kind: fmt.normalizer(available[s], kind)
        for s, kind in fmt.suffix_map.items()
        if s in available
    }
    if not tensors:
        return None

    fmt.complete_tensors(tensors)
    return Linear(
        tensors=tensors,
        weight_format=fmt,
        data_format=fmt.make_data_format(data_type),
    )
```

The `block_in` / `block_out` kwargs and `FORMAT_PRIORITY` iteration survive to Task 5.

- [ ] **Step 3: Thread `data_type` through `TextModelSpec._linear`**

In `lmdeploy/turbomind/deploy/spec.py`, update `_linear`:

```python
def _linear(self, pfx: str):
    from .kind_map import build_linear
    return build_linear(self.params, pfx,
                        data_type=self._cpp_dtype(),
                        block_in=self._group_size,
                        block_out=self._group_size)
```

- [ ] **Step 4: Thread `data_type` through direct `build_linear` callers in subclass specs**

In `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` (around line 226), update the direct call:

```python
def _read_packed_expert(self, prefix: str, expert: int):
    lin = build_linear(self.params, prefix, index=expert,
                       data_type=self._cpp_dtype(),
                       block_in=self._group_size,
                       block_out=self._group_size)
    ...
```

In `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` (around lines 291, 295), update both direct calls:

```python
gate_up_lin = build_linear(self.params, f'{pfx}.gate_up_proj',
                           index=expert_idx,
                           data_type=self._cpp_dtype(),
                           block_in=self._group_size,
                           block_out=self._group_size)
down_lin = build_linear(self.params, f'{pfx}.down_proj',
                        index=expert_idx,
                        data_type=self._cpp_dtype(),
                        block_in=self._group_size,
                        block_out=self._group_size)
```

- [ ] **Step 5: Update `_commit_linear` to read dtype from `linear.data_format` instead of inferring from WeightFormat**

In `lmdeploy/turbomind/deploy/builder/_base.py`, modify `_commit_linear`. The line around 456 (`cpp_dtype = _infer_cpp_linear_dtype(linear)`) becomes a read from `linear.data_format.dtype`:

```python
def _commit_linear(self, name: str, linear: Linear,
                   split_side: SplitSide | None = None,
                   model_dtype=None):
    self._ensure_handles()
    w = linear.tensors.get('weight')
    if w is None:
        return

    fmt = linear.weight_format
    weight_cpp_dtype = linear.data_format.dtype if linear.data_format is not None else None
    block_in = (fmt.block_in or 0) if fmt is not None else 0

    tp = self._tp if split_side else 1
    split_dim = _SPLIT_SIDE_TO_DIM.get(split_side) if split_side else None

    in_dim, out_dim = w.shape[0], w.shape[-1]
    if split_side == SplitSide.OUTPUT:
        out_dim //= tp
    elif split_side == SplitSide.INPUT:
        in_dim //= tp

    compute_dtype = (model_dtype if model_dtype is not None
                     else _infer_compute_dtype(linear))
    lin_cfg = _tm.LinearConfig()
    lin_cfg.input_dim = in_dim
    lin_cfg.output_dim = out_dim
    lin_cfg.data_type = compute_dtype or _tm.DataType.TYPE_INVALID
    lin_cfg.has_bias = 'bias' in linear.tensors

    packer = fmt.packer if fmt else None
    if packer is not None:
        tensors = {k: packer(t, k) for k, t in linear.tensors.items()}
    else:
        tensors = linear.tensors
    is_quantized = (linear.data_format is not None
                    and linear.data_format.is_quantized())

    kind_split_dims = {
        kind: None if (kind == 'bias' and split_side == SplitSide.INPUT)
              else split_dim
        for kind in tensors
    }

    if tp > 1 and split_dim is not None:
        for kind, tensor in tensors.items():
            kind_split_dim = kind_split_dims[kind]
            if kind_split_dim is not None:
                d = tensor.shape[kind_split_dim]
                assert d % tp == 0, (
                    f"TP split: {name}.{kind} dim {kind_split_dim} "
                    f"has size {d}, not divisible by tp={tp}.")

    for i, handle in enumerate(self._handles):
        with self._contexts[i]:
            rank = self._rank_for(i) if tp > 1 else 0

            linear_mod = (handle.child(name)
                          or handle.create_child(name, lin_cfg))
            linear_mod.set_weight_spec(weight_cpp_dtype, block_in)

            for kind, tensor in tensors.items():
                shard = _shard(tensor, kind_split_dims[kind], tp, rank)

                if kind == 'weight' and is_quantized:
                    alloc_shape, alloc_dtype = ([in_dim, out_dim],
                                                weight_cpp_dtype)
                elif kind == 'weight' and model_dtype is not None:
                    alloc_shape, alloc_dtype = None, model_dtype
                else:
                    alloc_shape, alloc_dtype = None, None

                _copy_shard_to_param(linear_mod, kind, shard,
                                     alloc_shape=alloc_shape,
                                     alloc_dtype=alloc_dtype)
```

Delete the `_infer_cpp_linear_dtype` function (lines 92-109 of `_base.py`) — no callers remain. The imports for it (if any) and the docstring referencing it may be removed too.

Note: the `set_weight_spec(weight_cpp_dtype, block_in)` call survives Task 2. It is deleted in Task 4.

- [ ] **Step 6: Build Python side**

Python-only changes; no ninja rebuild needed. Spot-check with:

```bash
cd /data/lmdeploy-modeling && python -c "from lmdeploy.turbomind.deploy.kind_map import TRIVIAL_FORMAT; import _turbomind as _tm; f = TRIVIAL_FORMAT.make_data_format(_tm.DataType.TYPE_BF16); print(f.dtype, f.block_sizes)"
```

Expected: prints something like `DataType.TYPE_BF16 [1, 1]`. If `_turbomind` is missing, rebuild C++ side first.

- [ ] **Step 7: Run `test_compressed_tensors.py` as a regression guard**

```bash
cd /data/lmdeploy-modeling && pytest tests/test_lmdeploy/test_turbomind/test_compressed_tensors.py -v
```

Expected: all pass. If a test fails mentioning `data_format` being `None`, check Step 2 — every `build_linear` call site must pass `data_type`.

- [ ] **Step 8: Integration smoke — run turbomind on an AWQ model**

Pick a small AWQ model via `list_models` / `get_model_cache_path`. Run `scripts/test_turbomind_model.py`. Require ≥128 tokens coherent output. This exercises the `Linear.data_format` bridge through the commit path (even though `set_weight_spec` is still the actual transport).

- [ ] **Step 9: Integration smoke — trivial BF16 model**

Same recipe with a trivial model. Validates that `make_data_format` handles the `block_in is None` / trivial case.

- [ ] **Step 10: Commit**

```bash
cd /data/lmdeploy-modeling && git add \
    lmdeploy/turbomind/deploy/kind_map.py \
    lmdeploy/turbomind/deploy/spec.py \
    lmdeploy/turbomind/deploy/builder/_base.py \
    lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py \
    lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py \
  && git commit -m "refactor(deploy): WeightFormat.make_data_format; populate Linear.data_format; build_linear takes data_type"
```

---

## Task 3: Explicit `data_type` in dequant pipelines; fix `_dequant_fp8` hardcoded BF16; tighten `Linear` invariants

**Files:**
- Modify: `lmdeploy/turbomind/deploy/kind_map.py`
- Modify: `lmdeploy/turbomind/deploy/linear.py`
- Modify: `lmdeploy/turbomind/deploy/builder/_base.py`
- Modify: `lmdeploy/turbomind/deploy/builder/attention.py`
- Modify: `lmdeploy/turbomind/deploy/builder/deltanet.py`
- Modify: `lmdeploy/turbomind/deploy/builder/mla.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/utils.py` (delete duplicate `_dequant_linear`, thread `data_type` into `_fold_head_dim`)
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` (MoE gate ad-hoc Linear construction; callers of `_fold_head_dim`)
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` (MoE gate ad-hoc Linear construction)
- Modify: `tests/test_lmdeploy/test_turbomind/test_transform_tensors.py` (`_make_linear` helper)

- [ ] **Step 1: Add `_CPP_TO_TORCH` inverse map in `_base.py`**

In `lmdeploy/turbomind/deploy/builder/_base.py`, after the `_TORCH_TO_CPP` dict (around line 50), add:

```python
_CPP_TO_TORCH: dict[_tm.DataType, torch.dtype] = {v: k for k, v in _TORCH_TO_CPP.items()}
```

- [ ] **Step 2: Change `WeightFormat.dequant` callable signature to take `data_type`**

In `lmdeploy/turbomind/deploy/kind_map.py`, update the `WeightFormat` type annotation:

```python
@dataclass(frozen=True)
class WeightFormat:
    ...
    dequant: Callable[[dict[str, Tensor], "_tm.DataType"],
                      dict[str, Tensor]] | None
```

And update the docstring for the `dequant` field accordingly (line ~69-72).

- [ ] **Step 3: Rewrite `_dequant_awq` and `_dequant_fp8` with the new signature**

In `kind_map.py`, replace both functions:

```python
def _dequant_awq(tensors: dict[str, Tensor], data_type) -> dict[str, Tensor]:
    from lmdeploy.pytorch.backends.default.awq_modules import dequantize_gemm

    qweight = tensors["weight"]
    scales = tensors["scales"]
    qzeros = tensors["zeros"]
    group_size = qweight.shape[0] // scales.shape[0]
    w = dequantize_gemm(qweight, qzeros, scales, 4, group_size)
    result: dict[str, Tensor] = {"weight": w}
    if "bias" in tensors:
        result["bias"] = tensors["bias"]
    return result


def _dequant_fp8(tensors: dict[str, Tensor], data_type) -> dict[str, Tensor]:
    from .builder._base import _CPP_TO_TORCH

    weight = tensors["weight"]
    scales = tensors["scales"]
    block_size = 128
    fp8_weight = weight.view(torch.float8_e4m3fn).float()
    scale = scales.float()
    scale = scale.repeat_interleave(block_size, dim=0)
    scale = scale.repeat_interleave(block_size, dim=1)
    scale = scale[: fp8_weight.shape[0], : fp8_weight.shape[1]]
    target_dtype = _CPP_TO_TORCH[data_type]
    result: dict[str, Tensor] = {"weight": (fp8_weight * scale).to(target_dtype)}
    if "bias" in tensors:
        result["bias"] = tensors["bias"]
    return result
```

`_dequant_awq` accepts `data_type` for uniform signature but does not use it (AWQ's `dequantize_gemm` infers dtype from `scales.dtype`).

- [ ] **Step 4: Update `_dequant_linear` in `_base.py` to take `data_type` explicitly**

In `lmdeploy/turbomind/deploy/builder/_base.py`, replace `_dequant_linear`:

```python
def _dequant_linear(linear: Linear, *, data_type) -> Linear:
    """Dequantize a quantized Linear to trivial when the format provides ``dequant``.

    *data_type* is the model's activation dtype; used to construct the new
    trivial ``data_format`` on the result and is threaded into the dequant
    callable so e.g. FP8 produces weights in the caller's activation dtype.
    """
    fmt = linear.weight_format
    if fmt is None or fmt.dequant is None:
        return linear
    new_tensors = fmt.dequant(linear.tensors, data_type)
    return Linear(
        tensors=new_tensors,
        weight_format=TRIVIAL_FORMAT,
        data_format=TRIVIAL_FORMAT.make_data_format(data_type),
    )
```

The `if fmt is None` branch is retained in Task 3 (defensively). It is removed when the Linear dataclass tightens (Step 9) — but removing it prematurely would break ordering. Keep the branch until Step 9.

- [ ] **Step 5: Update `_ensure_compatible_formats` to take `data_type`**

```python
def _ensure_compatible_formats(linears: dict[str, Linear], *, data_type) -> dict[str, Linear]:
    """Dequant linears to a common trivial format if a fusion group has mixed formats."""
    formats = {name: lin.weight_format.name for name, lin in linears.items() if lin is not None}
    if len(set(formats.values())) <= 1:
        return linears
    return {name: _dequant_linear(lin, data_type=data_type) if lin is not None else lin
            for name, lin in linears.items()}
```

- [ ] **Step 6: Update `dequant_mixed` in `attention.py`**

In `lmdeploy/turbomind/deploy/builder/attention.py` (line 22), replace `dequant_mixed`:

```python
def dequant_mixed(*linears: Linear, data_type) -> tuple[Linear, ...]:
    """When a fusion group has mixed formats (e.g. AWQ qkv + trivial norm bias),
    dequantize all non-trivial args so formats match for fusion.
    """
    formats = {l.weight_format.name for l in linears if l is not None}
    if len(formats) <= 1:
        return linears
    return tuple(_dequant_linear(l, data_type=data_type) if l is not None else l
                 for l in linears)
```

Then update the call site (line ~116) to pass `data_type=self.config.data_type`:

```python
q, k, v, gate = dequant_mixed(q, k, v, gate, data_type=self.config.data_type)
```

- [ ] **Step 7: Update `deltanet.py` caller of `_ensure_compatible_formats`**

In `lmdeploy/turbomind/deploy/builder/deltanet.py` (line 123), update:

```python
group = _ensure_compatible_formats(
    {"q": q, "k": k, "v": v, "z": in_proj_z, "b": in_proj_b, "a": in_proj_a},
    data_type=self.config.data_type)
```

- [ ] **Step 8: Delete duplicate `_dequant_linear` in `source_model/utils.py`; import from `_base`; thread `data_type` into `_fold_head_dim`**

In `lmdeploy/turbomind/deploy/source_model/utils.py`:

Delete the local `_dequant_linear` function (lines 178-189).

At the top of the file, add:

```python
from ..builder._base import _dequant_linear
```

Modify `_fold_head_dim` (around line 195) to accept `data_type`:

```python
def _fold_head_dim(linear: "Linear", *, head_num: int, head_dim: int,
                   kv_head_num: int | None = None, data_type) -> "Linear":
    """Fold (head_num, head_dim) into a single output dimension.

    - If quantized and block_out % head_dim != 0, dequantizes first using
      *data_type* to resolve the trivialized format.
    """
    ...
    # existing body, but wherever _dequant_linear(linear) is called, replace with:
    #     linear = _dequant_linear(linear, data_type=data_type)
```

Find each `_dequant_linear(linear)` call in this function and add `data_type=data_type`.

- [ ] **Step 9: Tighten `Linear` dataclass — make `weight_format` and `data_format` required**

In `lmdeploy/turbomind/deploy/linear.py`, modify the `@dataclass` (lines 122-138):

```python
@dataclass
class Linear:
    """Bundle of tensors for a single linear layer.

    ``tensors`` maps a closed-set TM weight kind (e.g. ``"weight"``,
    ``"scales"``, ``"zeros"``, ``"bias"``, ``"qweight"``) to the actual
    tensor.

    **Layout contract**: all ``Linear`` objects are in TM layout with
    axis 0 as the input dimension and axis -1 as the output dimension.
    ``commit_linear`` assumes this layout and does not re-transpose.
    1-D tensors (e.g. bias) only have an output dimension (axis 0).

    ``weight_format`` and ``data_format`` are both required — any
    construction site that doesn't know them is a bug.
    """

    tensors: dict[str, Tensor]
    weight_format: "WeightFormat" = field(compare=False, repr=False)
    data_format: "_tm.DataFormat" = field(compare=False, repr=False)
    # ...
```

The `| None` annotations and `default=None` go away. `field(...)` keeps `compare=False`, `repr=False` per today.

Note: any existing `Linear(tensors=...)` call without `weight_format` / `data_format` arguments will `TypeError: Linear.__init__() missing 2 required positional-or-keyword arguments`. Those are addressed in Steps 11-13.

- [ ] **Step 10: Update `Linear.concat_*_dim` — assert uniform formats instead of silent `None`**

In `lmdeploy/turbomind/deploy/linear.py`, rewrite both `concat_out_dim` and `concat_in_dim`:

```python
@classmethod
def concat_out_dim(cls, xs: list[Linear]) -> Linear:
    """Concatenate along output dim. Requires all inputs have the same
    ``weight_format`` and ``data_format`` — callers must call
    ``dequant_mixed`` first if formats differ."""
    first = xs[0]
    result: dict[str, Tensor] = {}
    for kind in first.tensors:
        t = first.tensors[kind]
        result[kind] = torch.cat([x.tensors[kind] for x in xs], dim=t.dim() - 1)
    wfmts = {x.weight_format for x in xs}
    dfmts = {x.data_format  for x in xs}
    assert len(wfmts) == 1 and len(dfmts) == 1, (
        "concat_out_dim requires uniform weight_format and data_format; "
        "call dequant_mixed first if formats differ.")
    return Linear(tensors=result,
                  weight_format=next(iter(wfmts)),
                  data_format=next(iter(dfmts)))

@classmethod
def concat_in_dim(cls, xs: list[Linear]) -> Linear:
    """Concatenate along input dim. Requires uniform formats (same as
    ``concat_out_dim``)."""
    first = xs[0]
    result: dict[str, Tensor] = {}
    for kind in first.tensors:
        t0 = first.tensors[kind]
        if not _has_input_dim(t0):
            result[kind] = t0
            continue
        result[kind] = torch.cat([x.tensors[kind] for x in xs], dim=0)
    wfmts = {x.weight_format for x in xs}
    dfmts = {x.data_format  for x in xs}
    assert len(wfmts) == 1 and len(dfmts) == 1, (
        "concat_in_dim requires uniform weight_format and data_format; "
        "call dequant_mixed first if formats differ.")
    return Linear(tensors=result,
                  weight_format=next(iter(wfmts)),
                  data_format=next(iter(dfmts)))
```

- [ ] **Step 11: Fix ad-hoc `Linear(tensors={...})` construction sites in `mla.py`**

In `lmdeploy/turbomind/deploy/builder/mla.py`, update three sites (lines 57, 60, 74):

```python
# Around line 57 — q_b + wo fold result:
return (Linear(tensors={"weight": q_folded.contiguous()},
               weight_format=q_b.weight_format,
               data_format=q_b.data_format),
        Linear(tensors={"weight": o_folded.contiguous()},
               weight_format=wo.weight_format,
               data_format=wo.data_format))

# Around line 74 — single-output variant:
return Linear(tensors={"weight": w.contiguous()},
              weight_format=wo.weight_format,
              data_format=wo.data_format)
```

- [ ] **Step 12: Fix ad-hoc MoE gate `Linear(tensors)` in `gpt_oss_spec.py` and `glm4_moe_lite_spec.py`**

In `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` (line 200):

```python
m.add_gate('gate', Linear(
    tensors,
    weight_format=TRIVIAL_FORMAT,
    data_format=TRIVIAL_FORMAT.make_data_format(self._cpp_dtype()),
), model_dtype=dtype)
```

The file needs a new import at the top:

```python
from ..kind_map import TRIVIAL_FORMAT
```

In `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` (line 239), apply the same change with the same import.

- [ ] **Step 13: Thread `data_type` into `_fold_head_dim` callers**

`_fold_head_dim` is called from spec subclasses. Find every caller (search `_fold_head_dim(`) and add `data_type=self._cpp_dtype()`. The known callers live in the same files already touched (qwen3_5_spec.py, glm4_moe_lite_spec.py, etc.) — verify via:

```bash
cd /data/lmdeploy-modeling && rg -n '_fold_head_dim\(' --type py
```

Update each call site:

```python
# Before:
lin = _fold_head_dim(lin, head_num=..., head_dim=..., kv_head_num=...)
# After:
lin = _fold_head_dim(lin, head_num=..., head_dim=..., kv_head_num=...,
                     data_type=self._cpp_dtype())
```

- [ ] **Step 14: Add `data_format` assertion in `_commit_linear`**

In `lmdeploy/turbomind/deploy/builder/_base.py::_commit_linear`, after the `w = linear.tensors.get('weight'); if w is None: return` guard, add:

```python
assert linear.data_format is not None, (
    f"{name}: Linear.data_format must be populated by build_linear or "
    f"by a fusion helper with explicit data_type.")
```

This catches any remaining construction site that slipped through.

Also, remove the `if linear.data_format is not None` guard from the `weight_cpp_dtype` and `is_quantized` assignments (they were defensive against the old `None`-legal world):

```python
weight_cpp_dtype = linear.data_format.dtype
is_quantized = linear.data_format.is_quantized()
```

- [ ] **Step 15: Update `_make_linear` test helper**

In `tests/test_lmdeploy/test_turbomind/test_transform_tensors.py`, find `_make_linear` (line 114) and update to supply `weight_format` + `data_format`:

```python
from lmdeploy.turbomind.deploy.kind_map import TRIVIAL_FORMAT
import _turbomind as _tm

def _make_linear(out_dim: int, in_dim: int | None = None,
                 has_bias: bool = False) -> Linear:
    """Build a trivial BF16 Linear for tests."""
    tensors: dict[str, torch.Tensor] = {}
    if in_dim is None:
        tensors['weight'] = torch.randn(out_dim)  # 1-D (embed/lm_head style)
    else:
        tensors['weight'] = torch.randn(in_dim, out_dim)
    if has_bias:
        tensors['bias'] = torch.randn(out_dim)
    return Linear(
        tensors=tensors,
        weight_format=TRIVIAL_FORMAT,
        data_format=TRIVIAL_FORMAT.make_data_format(_tm.DataType.TYPE_BF16),
    )
```

The existing `test_format_propagation` test (line 274) uses `object.__setattr__` to overwrite the format fields with strings — it continues to work because the decorator doesn't interpret the format values.

- [ ] **Step 16: Run `test_transform_tensors.py`**

```bash
cd /data/lmdeploy-modeling && pytest tests/test_lmdeploy/test_turbomind/test_transform_tensors.py -v
```

Expected: all pass.

- [ ] **Step 17: Run `test_compressed_tensors.py`**

```bash
cd /data/lmdeploy-modeling && pytest tests/test_lmdeploy/test_turbomind/test_compressed_tensors.py -v
```

Expected: all pass.

- [ ] **Step 18: Integration smoke — mixed-format attention (AWQ proj + trivial bias)**

Pick a small AWQ model that has biases in attention (AWQ often omits biases, so this may require picking a model that has them — e.g. an AWQ-ified Qwen variant). Run the turbomind test script; require ≥128 tokens coherent.

If no such model is locally available, skip this specific matrix entry and document the gap in the commit message's test plan.

- [ ] **Step 19: Integration smoke — FP8 model (latent bug fix)**

Pick a small FP8 blocked model (e.g. any Llama-family FP8 variant). Run the test script. Require ≥128 tokens coherent.

This verifies that the new `_dequant_fp8` (honoring `data_type`) still produces correct weights when fusion-time dequant is triggered. Most FP8 models never hit the Python dequant path (kernels stay in FP8), so this is a regression guard not a behavior-change test.

- [ ] **Step 20: Integration smoke — deltanet model (if available)**

If a GDN-style model (e.g. qwen3.5-gdn variant) is locally available, run it. Exercises `_ensure_compatible_formats` + `dequant_mixed` threading.

- [ ] **Step 21: Commit**

```bash
cd /data/lmdeploy-modeling && git add \
    lmdeploy/turbomind/deploy/kind_map.py \
    lmdeploy/turbomind/deploy/linear.py \
    lmdeploy/turbomind/deploy/builder/_base.py \
    lmdeploy/turbomind/deploy/builder/attention.py \
    lmdeploy/turbomind/deploy/builder/deltanet.py \
    lmdeploy/turbomind/deploy/builder/mla.py \
    lmdeploy/turbomind/deploy/source_model/utils.py \
    lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py \
    lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py \
    tests/test_lmdeploy/test_turbomind/test_transform_tensors.py \
  && git commit -m "$(cat <<'EOF'
refactor(deploy): explicit data_type in dequant pipelines; fix _dequant_fp8 hardcoded BF16

Plumb data_type through _dequant_linear, _ensure_compatible_formats,
dequant_mixed, and _fold_head_dim. Make WeightFormat.dequant signature
uniform with a data_type arg. Rewrite _dequant_fp8 to honor data_type
(fixes latent bug: FP8-activation-FP16 models were silently coerced
to BF16). Dedupe _dequant_linear — utils.py now imports from _base.

Tighten Linear.weight_format and Linear.data_format to required
(non-optional). Update ad-hoc Linear constructions in mla.py,
gpt_oss_spec.py, glm4_moe_lite_spec.py to supply both explicitly.
concat_*_dim asserts uniform formats, replacing silent None fallback.
Add data_format assertion in _commit_linear.
EOF
)"
```

---

## Task 4: Single-phase `LinearConfig.format`; delete `set_weight_spec` / `configure` / `LinearPolicy`

**Files:**
- Modify: `src/turbomind/models/linear_weight.h`
- Modify: `src/turbomind/models/linear_weight.cc`
- Modify: `src/turbomind/models/moe_weight.cc`
- Modify: `src/turbomind/python/bind.cpp`
- Modify: `lmdeploy/turbomind/deploy/builder/_base.py`

This is the structural collapse. C++ and Python changes must go together — before this commit, `_commit_linear` calls `set_weight_spec`; after, `set_weight_spec` doesn't exist and `LinearConfig.format` carries the same information.

- [ ] **Step 1: Extend `LinearConfig` with `format: DataFormat`**

In `src/turbomind/models/linear_weight.h` (lines 11-24), update the field list:

```cpp
struct LinearConfig: ModuleConfig {
    LinearConfig(): ModuleConfig{"LinearWeight"} {}

    #define LINEAR_FIELDS(X) \
        X(int,        input_dim)  \
        X(int,        output_dim) \
        X(DataType,   data_type)  \
        X(DataFormat, format)     \
        X(bool,       has_bias)

    LINEAR_FIELDS(TM_MEMBER)
    TM_FOR_EACH(LinearConfig, LINEAR_FIELDS)

    #undef LINEAR_FIELDS
};
```

No change needed in `bind.cpp`'s `bind_config<LinearConfig>` call — the X-macro's `for_each` exposes the new field automatically.

- [ ] **Step 2: Replace `struct LinearPolicy` and `ResolveLinearPolicy` with `DeriveActivationFormats`**

In `src/turbomind/models/linear_weight.h`, delete the `LinearPolicy` struct (lines 34-40) and the `ResolveLinearPolicy` declaration (line 43). Replace with:

```cpp
/// Derive (input_format, output_format) for a GEMM whose weight uses
/// `weight_format`, given the model's activation dtype and hardware SM.
std::pair<DataFormat, DataFormat>
DeriveActivationFormats(const DataFormat& weight_format,
                        DataType          data_type,
                        int               sm);
```

In `src/turbomind/models/linear_weight.cc`, replace the body of `ResolveLinearPolicy` (lines 23-57) with:

```cpp
std::pair<DataFormat, DataFormat>
DeriveActivationFormats(const DataFormat& weight_format, DataType data_type, int sm)
{
    DataFormat in_fmt;
    DataFormat out_fmt;
    in_fmt.dtype       = data_type;
    in_fmt.block_sizes = {1, 1};
    out_fmt.dtype      = data_type;
    out_fmt.block_sizes = {1, 1};

    // Empty weight_format (from LinearBuilder.set_weight path for embeddings /
    // lm_head): treat as trivial. No quantization on I/O.
    if (weight_format.dtype == DataType{}) {
        return {in_fmt, out_fmt};
    }

    if (!weight_format.is_quantized()) {
        return {in_fmt, out_fmt};
    }

    if (weight_format.dtype == kFloat8_e4m3) {
        if (sm == 90) {
            int gs = weight_format.block_sizes[0];          // K-axis, tensor-shape order
            in_fmt.dtype        = kFloat8_e4m3;
            in_fmt.block_sizes  = {gs, 1};
            in_fmt.scales.dtype = kFloat;
        }
        return {in_fmt, out_fmt};
    }

    // FP4 / U4 / U8: input stays in model activation dtype — the GEMM
    // upcasts / dequants on the fly. output_format is also activation dtype.
    return {in_fmt, out_fmt};
}
```

- [ ] **Step 3: Rewrite `LinearWeight` class declaration**

In `src/turbomind/models/linear_weight.h`, replace the `LinearWeight` class body (lines 45-105) with:

```cpp
class LinearWeight: public core::Module {
public:
    const char* type() const override { return "LinearWeight"; }

    LinearWeight() = default;
    LinearWeight(const core::LinearConfig& cfg);

    void prepare() override;
    void copy_metadata_to(LinearWeight& dst) const;

    /// Set grouped-GEMM mode (for MoE expert weights that need row-major layout).
    void set_grouped(bool grouped) { is_grouped_ = grouped; }

    explicit operator bool() const noexcept { return static_cast<bool>(weight); }

    // --- three DataFormats fully describe the GEMM ---
    DataFormat weight_format{};  // from cfg.format
    DataFormat input_format{};   // derived in ctor
    DataFormat output_format{};  // derived in ctor

    DataType input_dtype()  const { return input_format.dtype;  }
    DataType output_dtype() const { return output_format.dtype; }

    // --- dimensions + model activation dtype ---
    int      input_dim  = 0;
    int      output_dim = 0;
    DataType data_type{};   // model activation dtype, copied from cfg.data_type

    // --- GEMM knobs ---
    Epilogue     epilogue{};
    MatrixLayout k_desc{};
    MatrixLayout q_desc{};

#define LINEAR_WEIGHT_CHILDREN(X)

#define LINEAR_WEIGHT_PARAMS(X) \
    X(weight) \
    X(bias)   \
    X(scales) \
    X(zeros)

    TM_MODULE_DECLARE(LinearWeight, LINEAR_WEIGHT_CHILDREN, LINEAR_WEIGHT_PARAMS)

private:
    bool has_bias_   = false;
    bool is_grouped_ = false;
};
```

Deletions (relative to today's declaration): `configure`, `set_weight_spec`, `preprocess` methods; scalar `weight_format: DataType`, `group_size: int`, `policy: LinearPolicy` fields.

- [ ] **Step 4: Rewrite `LinearWeight` ctor and helper in `.cc`**

In `src/turbomind/models/linear_weight.cc`, replace the old `configure` + `set_weight_spec` + `preprocess` functions and the constructor with:

```cpp
LinearWeight::LinearWeight(const core::LinearConfig& cfg)
    : input_dim(cfg.input_dim)
    , output_dim(cfg.output_dim)
    , data_type(cfg.data_type)
    , weight_format(cfg.format)
    , has_bias_(cfg.has_bias)
{
    std::tie(input_format, output_format) =
        DeriveActivationFormats(weight_format, data_type, getSMVersion());
}
```

Delete:
- `void LinearWeight::configure(...)` (lines 63-73 of the old file)
- `void LinearWeight::set_weight_spec(...)` (lines 95-105)
- `void LinearWeight::preprocess()` (lines 111-114)

- [ ] **Step 5: Update `LinearWeight::copy_metadata_to`**

In `src/turbomind/models/linear_weight.cc`, replace the `copy_metadata_to` body with:

```cpp
void LinearWeight::copy_metadata_to(LinearWeight& dst) const
{
    dst.input_dim     = input_dim;
    dst.output_dim    = output_dim;
    dst.data_type     = data_type;
    dst.weight_format = weight_format;
    dst.input_format  = input_format;
    dst.output_format = output_format;
    dst.epilogue      = epilogue;
    dst.has_bias_     = has_bias_;
    dst.is_grouped_   = is_grouped_;
    dst.k_desc        = k_desc;
    dst.q_desc        = q_desc;
}
```

- [ ] **Step 6: Update `LinearWeight::prepare()` — read field names that no longer exist as scalars**

In `src/turbomind/models/linear_weight.cc::prepare()`, replace every scalar `weight_format` access (it's now the `DataFormat` struct):

- Line 122 `if (!weight) { return; }` — unchanged.
- Line 135 `if (weight_format == DataType{}) {` → `if (weight_format.dtype == DataType{}) {`
- Line 145 `if (weight_format == kFloat8_e4m3 && input_dtype() == kFloat8_e4m3) {` → `if (weight_format.dtype == kFloat8_e4m3 && input_dtype() == kFloat8_e4m3) {`
- Line 164 `else if (weight_format == kFloat8_e4m3) {` → `else if (weight_format.dtype == kFloat8_e4m3) {`
- Line 172 `auto [conv_w, conv_s] = GetConverters(data_type, weight_format, input_dtype(), is_grouped_, getSMVersion());` → pass `weight_format.dtype` as the second arg: `GetConverters(data_type, weight_format.dtype, input_dtype(), is_grouped_, getSMVersion())`.
- Line 179 `const int bits = byte_size(weight_format, 8);` → `const int bits = byte_size(weight_format.dtype, 8);`
- Line 216 `kd.type = weight_format;` → `kd.type = weight_format.dtype;`
- Line 229 `kd.type = weight_format;` → `kd.type = weight_format.dtype;`
- Line 252 `else if (weight_format == kFloat8_e4m3) {` → `else if (weight_format.dtype == kFloat8_e4m3) {`
- Line 263 `if (data_type == kHalf && weight_format == kFloat4_e2m1) {` → `if (data_type == kHalf && weight_format.dtype == kFloat4_e2m1) {`

Replace the `group_size` references (line 272 in `s_desc`):

```cpp
int gs = weight_format.block_sizes[0];  // K-axis, tensor-shape order
MatrixLayout s_desc{
    scale_type,
    order_s,
    (int)output_dim,
    (int)input_dim / gs,
    (int)output_dim,
};
```

- [ ] **Step 7: Update `moe_weight.cc::LinkExperts`**

In `src/turbomind/models/moe_weight.cc`, line 65:

```cpp
if (d.weight_format.dtype == kFloat8_e4m3 && d.input_dtype() == kFloat8_e4m3) {
```

And line 77:

```cpp
d.weight = Tensor{make_strided_ptr(weights), {n}, d.weight_format.dtype, kDEVICE};
```

- [ ] **Step 8: Remove `set_weight_spec` pybind binding**

In `src/turbomind/python/bind.cpp` around lines 668-675, delete:

```cpp
py::class_<turbomind::LinearWeight, ft::core::Module>(m, "LinearWeight")
    .def("set_weight_spec",
         [](turbomind::LinearWeight& lw, ft::DataType dtype, int group_size) {
             lw.set_weight_spec(dtype, group_size);
         },
         "dtype"_a,
         "group_size"_a);
```

Entire block removed. `LinearWeight` is no longer exposed with any Python-callable methods beyond what `Module` already provides.

- [ ] **Step 9: Python `_commit_linear` — write `lin_cfg.format` instead of calling `set_weight_spec`**

In `lmdeploy/turbomind/deploy/builder/_base.py::_commit_linear`, replace the block that builds `lin_cfg` + calls `set_weight_spec` with:

```python
    lin_cfg = _tm.LinearConfig()
    lin_cfg.input_dim  = in_dim
    lin_cfg.output_dim = out_dim
    lin_cfg.data_type  = compute_dtype or _tm.DataType.TYPE_INVALID
    lin_cfg.format     = linear.data_format
    lin_cfg.has_bias   = 'bias' in linear.tensors
```

And delete the `linear_mod.set_weight_spec(weight_cpp_dtype, block_in)` call inside the per-GPU loop. The `block_in = (fmt.block_in or 0) if fmt is not None else 0` local becomes unused; delete it.

After this step, `_commit_linear` no longer references `fmt.block_in` at all. The `fmt = linear.weight_format` line is retained only if `fmt.packer` is used — verify and keep if so.

- [ ] **Step 10: Build**

```bash
cd build && ninja
```

Expected: clean build. Any compilation error likely means a straggling `set_weight_spec` / `configure` call site was missed — fix and rebuild.

- [ ] **Step 11: Run `test_data_format`**

```bash
./build/bin/test_data_format
```

Expected: all pass. (No behavior change in the factory itself; this step guards against accidental regressions.)

- [ ] **Step 12: Integration — trivial BF16 model**

Smoke test. Verifies the single-phase ctor works for the dominant trivial path.

- [ ] **Step 13: Integration — AWQ U4**

Smoke test. Verifies the quantized-path commit through `lin_cfg.format` + single-phase ctor.

- [ ] **Step 14: Integration — FP8 on SM90 (if available)**

Smoke test. Verifies `DeriveActivationFormats` sets `input_format.dtype = kFloat8_e4m3` on SM90 and the `prepare()` native-FP8 branch still engages.

- [ ] **Step 15: Integration — MoE (gpt_oss or qwen3_5)**

Smoke test. Verifies `copy_metadata_to` correctly propagates the three DataFormats from experts to the block view, and `set_grouped()` still works.

- [ ] **Step 16: Commit**

```bash
cd /data/lmdeploy-modeling && git add \
    src/turbomind/models/linear_weight.h \
    src/turbomind/models/linear_weight.cc \
    src/turbomind/models/moe_weight.cc \
    src/turbomind/python/bind.cpp \
    lmdeploy/turbomind/deploy/builder/_base.py \
  && git commit -m "$(cat <<'EOF'
refactor(linear_weight): single-phase LinearConfig.format; drop set_weight_spec / configure / LinearPolicy

LinearConfig gains a DataFormat format field carrying the weight storage
descriptor. LinearWeight ctor is now single-phase: derives
input_format / output_format via DeriveActivationFormats (replacing
ResolveLinearPolicy) from (weight_format, data_type, SM). LinearWeight
carries three DataFormat fields; LinearPolicy struct, ResolveLinearPolicy,
set_weight_spec, configure, preprocess, and the redundant scalar
weight_format / group_size / policy fields are gone.

Python _commit_linear writes lin_cfg.format = linear.data_format and no
longer calls set_weight_spec (the pybind binding is removed).
EOF
)"
```

---

## Task 5: Resolve `WeightFormat` at converter; purge `Spec._group_size`

**Files:**
- Modify: `lmdeploy/turbomind/deploy/converter.py`
- Modify: `lmdeploy/turbomind/deploy/spec.py`
- Modify: `lmdeploy/turbomind/deploy/kind_map.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/utils.py` (any subclass that threads `_group_size` indirectly — verify)

- [ ] **Step 1: Resolve active `WeightFormat` in `converter.py`**

In `lmdeploy/turbomind/deploy/converter.py`, add the import at the top:

```python
from dataclasses import replace
from .kind_map import get_weight_format
```

Then in `get_tm_config`, replace the tail (lines 154-185) — specifically where `group_size` is threaded into `spec_cls(...)` — with:

```python
    group_size = _validate_quant_group_size(engine_config.model_format, group_size)
    if engine_config.model_format is None:
        engine_config.model_format = 'hf'

    # 3. Resolve dtype and format overrides.
    dtype = _resolve_dtype(engine_config.dtype, hf_model_cfg)
    if engine_config.model_format in ('awq', 'gptq', 'compressed-tensors'):
        dtype = 'float16'
        if engine_config.model_format == 'compressed-tensors':
            engine_config.model_format = 'awq'

    # 4. Resolve session_len default.
    session_len_default = _get_and_verify_max_len(hf_model_cfg, None)

    # 5. Mutate engine_config with resolved values.
    engine_config.dtype = dtype
    if engine_config.session_len is None:
        engine_config.session_len = session_len_default
    engine_config.attn_tp_size = engine_config.attn_tp_size or 1
    engine_config.attn_cp_size = engine_config.attn_cp_size or 1
    engine_config.mlp_tp_size = engine_config.mlp_tp_size or 1

    # 6. Resolve the active WeightFormat (sentinel block sizes → concrete ints).
    #    Only ``block_in == 0`` is a sentinel today (AWQ / GPTQ / CT).
    #    ``block_out`` is either a concrete int (FP8 = 128) or ``None`` (no
    #    blocking on that axis); neither needs converter-time resolution.
    weight_format = get_weight_format(engine_config.model_format)
    if weight_format.block_in == 0:
        weight_format = replace(weight_format, block_in=group_size)

    # 7. Build spec.
    hf_cfg = load_model_config(model_path)
    if engine_config.hf_overrides:
        logger.warning(f'Overriding HF config with {engine_config.hf_overrides}')
        _deep_merge(hf_cfg, engine_config.hf_overrides)
    spec_name = get_spec_registered_name(model_path, engine_config.model_format)
    spec_cls = INPUT_MODELS.get(spec_name)
    spec = spec_cls(hf_cfg, engine_config, weight_format=weight_format)

    return spec, model_path
```

Note the keyword change: `group_size=group_size or 0` → `weight_format=weight_format`.

- [ ] **Step 2: Update `TextModelSpec.__init__` to take `weight_format` instead of `group_size`**

In `lmdeploy/turbomind/deploy/spec.py` (lines 52-68), replace `__init__`:

```python
def __init__(self, hf_cfg: dict, engine_cfg: 'TurbomindEngineConfig',
             *, weight_format):
    """Parse HF config into orchestration scalars.

    *weight_format* is the pre-resolved checkpoint format (block sizes
    already concrete); carried on ``self._weight_format`` so ``_linear()``
    and subclass specs can pass it into ``build_linear``.
    """
    self.hf_cfg = hf_cfg
    self.engine_cfg = engine_cfg
    self._weight_format = weight_format
    self._parse_base(hf_cfg)
```

Remove the `group_size` kwarg, `self._group_size = group_size`, and the docstring clause about `group_size`.

- [ ] **Step 3: Update `TextModelSpec._linear` to pass `weight_format` (drop `block_in` / `block_out`)**

In `lmdeploy/turbomind/deploy/spec.py` (lines 147-151), replace `_linear`:

```python
def _linear(self, pfx: str):
    from .kind_map import build_linear
    return build_linear(self.params, pfx,
                        data_type=self._cpp_dtype(),
                        weight_format=self._weight_format)
```

- [ ] **Step 4: Update `build_linear` — take `weight_format` kwarg, drop `block_in` / `block_out`, simplify priority**

In `lmdeploy/turbomind/deploy/kind_map.py`, replace `build_linear` entirely:

```python
def build_linear(
    params: dict[str, torch.Tensor],
    prefix: str,
    *,
    data_type,
    weight_format: WeightFormat,
    index: int | None = None,
) -> Linear | None:
    """Build a ``Linear`` bundle from checkpoint tensors at *prefix*.

    *weight_format* is the pre-resolved active format for this model
    (block sizes concrete — converter.py is the single resolution point).
    The format-matching loop collapses to: try the active format first,
    fall back only to TRIVIAL_FORMAT.  (Other quantized formats can't
    match a checkpoint of a different format, so they're not tried.)

    *data_type* is the model activation dtype; passed to
    ``WeightFormat.make_data_format`` to construct the C++ DataFormat on
    ``Linear.data_format``.
    """
    from .linear import Linear

    available: dict[str, torch.Tensor] = {
        s: params[prefix + s] for s in ALL_SUFFIXES if (prefix + s) in params
    }
    if index is not None:
        available = {s: t[index] for s, t in available.items()}

    if weight_format.accepts(available):
        fmt = weight_format
    elif TRIVIAL_FORMAT.accepts(available):
        fmt = TRIVIAL_FORMAT
    else:
        return None

    tensors: dict[str, torch.Tensor] = {
        kind: fmt.normalizer(available[s], kind)
        for s, kind in fmt.suffix_map.items()
        if s in available
    }
    if not tensors:
        return None

    fmt.complete_tensors(tensors)
    return Linear(
        tensors=tensors,
        weight_format=fmt,
        data_format=fmt.make_data_format(data_type),
    )
```

Removed: the `replace(fmt, block_in=..., block_out=...)` sentinel logic, the `FORMAT_PRIORITY` iteration at runtime, and the `block_in` / `block_out` kwargs. `FORMAT_PRIORITY` is retained in the file (still used to derive `ALL_SUFFIXES`) but no longer iterated here.

- [ ] **Step 5: Update subclass spec `__init__`s — drop `group_size`, forward `weight_format`**

For each subclass spec, replace the `__init__` signature. Four files:

`lmdeploy/turbomind/deploy/source_model/qwen3_spec.py` (line 34):

```python
def __init__(self, hf_cfg: dict, engine_cfg, *, weight_format):
    super().__init__(hf_cfg, engine_cfg, weight_format=weight_format)
    # ... body unchanged ...
```

`lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` (line 45):

```python
def __init__(self, hf_cfg: dict, engine_cfg, *, weight_format):
    super().__init__(hf_cfg, engine_cfg, weight_format=weight_format)
    # ... body unchanged ...
```

`lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py` (line 42):

```python
def __init__(self, hf_cfg: dict, engine_cfg, *, weight_format):
    super().__init__(hf_cfg, engine_cfg, weight_format=weight_format)
    # ... body unchanged ...
```

`lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py` (line 28):

```python
def __init__(self, hf_cfg: dict, engine_cfg, *, weight_format):
    super().__init__(hf_cfg, engine_cfg, weight_format=weight_format)
    # ... body unchanged ...
```

- [ ] **Step 6: Update direct `build_linear` call sites in subclass specs to use `weight_format`**

In `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py` (the two `build_linear` calls added in Task 2):

```python
gate_up_lin = build_linear(self.params, f'{pfx}.gate_up_proj',
                           index=expert_idx,
                           data_type=self._cpp_dtype(),
                           weight_format=self._weight_format)
down_lin = build_linear(self.params, f'{pfx}.down_proj',
                        index=expert_idx,
                        data_type=self._cpp_dtype(),
                        weight_format=self._weight_format)
```

In `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py::_read_packed_expert`:

```python
lin = build_linear(self.params, prefix, index=expert,
                   data_type=self._cpp_dtype(),
                   weight_format=self._weight_format)
```

The `block_in` / `block_out` kwargs (threaded in Task 2) are gone from every call site.

- [ ] **Step 7: Verify no `_group_size` references remain**

```bash
cd /data/lmdeploy-modeling && rg -n '_group_size' lmdeploy/turbomind/deploy/
```

Expected: no matches. If any appear, they're bugs — fix them.

```bash
cd /data/lmdeploy-modeling && rg -n 'group_size=' lmdeploy/turbomind/deploy/
```

Expected: only matches inside `converter.py` (the local `group_size` variable resolved from quant_config; not a kwarg passed anywhere).

- [ ] **Step 8: Run unit tests**

```bash
cd /data/lmdeploy-modeling && \
    pytest tests/test_lmdeploy/test_turbomind/test_transform_tensors.py -v && \
    pytest tests/test_lmdeploy/test_turbomind/test_compressed_tensors.py -v
```

Expected: all pass. `test_compressed_tensors` exercises the spec construction path and the commit pipeline; it's the most informative regression guard.

- [ ] **Step 9: Integration — trivial BF16 model**

Smoke test.

- [ ] **Step 10: Integration — AWQ U4 model**

Smoke test. Exercises the `replace(fmt, block_in=128, block_out=128)` path in converter.py + the `weight_format=self._weight_format` threading into `_linear`.

- [ ] **Step 11: Integration — compressed-tensors int4 g=32 (if available)**

This is the regression guard for the 2026-04-21-commit-simplification bug fix. If a compressed-tensors int4 model with `group_size=32` is locally available, run it. Otherwise flag the gap in the commit message.

- [ ] **Step 12: Integration — MoE (gpt_oss or qwen3_5)**

Smoke test. Exercises direct `build_linear(weight_format=self._weight_format, index=...)` in subclass specs.

- [ ] **Step 13: Commit**

```bash
cd /data/lmdeploy-modeling && git add \
    lmdeploy/turbomind/deploy/converter.py \
    lmdeploy/turbomind/deploy/spec.py \
    lmdeploy/turbomind/deploy/kind_map.py \
    lmdeploy/turbomind/deploy/source_model/qwen3_spec.py \
    lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py \
    lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py \
    lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py \
  && git commit -m "$(cat <<'EOF'
refactor(deploy): resolve WeightFormat at converter; purge Spec._group_size

converter.py resolves the active WeightFormat once via
dataclasses.replace (sentinel block_in == 0 → concrete group_size).
TextModelSpec takes weight_format instead of group_size; every
subclass spec drops its group_size kwarg forwarding. build_linear
gains a required weight_format kwarg, drops block_in / block_out,
and collapses the priority loop to active-format → TRIVIAL fallback
(FORMAT_PRIORITY is no longer iterated at runtime; retained only
for ALL_SUFFIXES derivation).

self._group_size disappears from every spec.
EOF
)"
```

---

## Post-task verification

After all five commits land:

- [ ] **Full integration matrix** — run `scripts/test_turbomind_model.py` across the spec's test matrix:
  - Trivial BF16 dense (TP=1, TP=2)
  - AWQ U4 (TP=1, TP=2)
  - FP8 blocked (TP=1, TP=2; SM90 if available)
  - Compressed-tensors int4 g=32 (TP=1) — if locally available
  - Mixed-format fusion (AWQ attention with trivial biases) — if locally available
  - MoE (gpt_oss or qwen3_5) (TP=1)

  Each run: ≥128 tokens coherent, prompt-relevant. Gibberish = silent bug.

- [ ] **Grep for leftovers:**

  ```bash
  cd /data/lmdeploy-modeling && \
      rg -n 'set_weight_spec' && \
      rg -n 'MakeLinearWeightFormat' && \
      rg -n 'ResolveLinearPolicy' && \
      rg -n 'struct LinearPolicy' && \
      rg -n '_group_size' lmdeploy/ && \
      rg -n '_infer_cpp_linear_dtype'
  ```

  Expected: no live matches. Matches inside `docs/superpowers/`, `.md` files, or the out-of-scope `testbed_v3.h` are expected and fine.

- [ ] **Grep for invariant violations:**

  ```bash
  cd /data/lmdeploy-modeling && \
      rg -n 'weight_format=None|data_format=None' lmdeploy/
  ```

  Expected: no matches.

---

## Self-review checklist (done during plan-writing; documented for implementers)

Each spec section maps to a task:
- Spec §1 (Architecture) → frame for the entire plan; no single task.
- Spec §2 (DataFormat / factory) → Task 1 (C++ rename/signature/ordering) + Task 2 (Python bridge `make_data_format`).
- Spec §3 (LinearConfig / LinearWeight single-phase) → Task 4.
- Spec §4 (Python data flow + invariants + `_dequant_fp8` fix) → Task 3 + (the `_commit_linear` final form in) Task 4.
- Spec §5 (Spec `_group_size` purge) → Task 5.
- Spec §6 (invariants / validation / testing) → implicit across all tasks; assertions added in Task 3 Step 14 and Task 3 Step 10.

Placeholder scan: every step either shows the exact code change, runs a specific command, or makes a concrete commit. No TBDs, no "similar to above" (Task 5 Step 5 and Task 3 Step 12 repeat the subclass patterns explicitly).

Type consistency: the factory is named `ResolveLinearWeightFormat` from its first introduction (Task 1) through every caller (Task 2, 5). `DeriveActivationFormats` appears first in Task 4 and is the only name used. `make_data_format(data_type)` is consistent across Task 2, 3, 5. `_CPP_TO_TORCH` defined in Task 3 Step 1 and used in Task 3 Step 3.
