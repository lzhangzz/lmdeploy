# Linear-Module Lifecycle Rethink Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse TurboMind's four-phase `LinearWeight` lifecycle (construct + set_weight_spec + per-param alloc + prepare) into single-phase construction from a `LinearConfig` that carries the full `DataFormat`. Pre-allocate param slots C++-side; commit becomes byte-copy only. Drop `LinearPolicy` for three `DataFormat`s (weight / input / output). Make Python `WeightFormat` a constant `DataFormat` factory with `FormatKind` enum and `BlockSize` NamedTuple. Purge `group_size` defaults; delete `LinearBuilder` / `set_weight` / `make_linear_config`. Move `tok_embeddings` from `LinearWeight` child to plain Tensor parameter. Load `lm_head` via `self._linear` like every other linear. Derive grouped-GEMM selection from the module tree instead of plumbing `is_grouped` through every commit.

**Architecture:** The change replaces several parallel information channels (`LinearConfig.data_type` + `set_weight_spec` + per-param `alloc(shape, dtype)`) with a single channel (`LinearConfig.format: DataFormat`). C++ `LinearWeight` constructor derives everything from `cfg.format` (activation formats via a new `DeriveActivationFormats` function, scales/zeros/bias slot shapes from `format.block_sizes`, grouped-GEMM layout from walking the parent chain at `prepare()` time). Python `WeightFormat` becomes a frozen descriptor with no mutable block-size state; concrete per-linear `DataFormat` lives on `Linear.data_format` and is manufactured by `WeightFormat.make_data_format(compute_dtype, block_in, block_out)`. `converter.py`'s quant-config parse is the sole translation point from `group_size` to `BlockSize`.

**Tech Stack:** C++17, CUDA, pybind11, Python 3.10+, `dataclasses`, `enum`, `typing.NamedTuple`, `torch`, `_turbomind` extension module, `scripts/test_turbomind_model.py` for end-to-end verification.

**Spec:** `docs/superpowers/specs/2026-04-21-linear-lifecycle-rethink-design.md`

---

## Preliminaries

This is a structural refactor. There are no unit tests covering the linear lifecycle today; verification is end-to-end model inference per `AGENTS.md`:

- Before any model run: call the `get_gpu_usage` MCP tool, pick empty GPU id(s).
- Discover test models via the model-server MCP `list_models` tool; pick the smallest model per format family.
- For each model, call `get_model_cache_path` for its cache dir.
- Run template (from repo root):
  ```
  cd /data/lmdeploy-modeling && PYTHONPATH=.:build/lib \
    python scripts/test_turbomind_model.py <model_path> <cache_dir> <tp> <gpus>
  ```
- A good run prints `--- response begin ---` ... `--- response end ---` with coherent prose answering "Write a short paragraph about the importance of reading books." and reports `generated: N` with N ≥ 100 (aim for 128). Gibberish or truncation at a few tokens is a silent bug.
- Record which models you pick so the same models can be re-run across tasks.

Build commands (from repo root):
```
cd /data/lmdeploy-modeling/build && ninja
```
(Configure via `sh ../my_generate.sh` from `build/` if the dir is empty; normally already configured.)

**NEVER** install lmdeploy as a pip package. **NEVER** run `setup.py`. The in-tree build is the only valid path per `AGENTS.md`.

### File-structure map (post-plan)

| File | Responsibility after the refactor |
| --- | --- |
| `src/turbomind/core/data_format.{h,cc}` | `DataFormat` struct unchanged; `MakeLinearWeightFormat(compute, weight, block_in, block_out)` signature |
| `src/turbomind/core/test_data_format.cc` | Tests updated to new signature |
| `src/turbomind/python/bind.cpp` | `MakeLinearWeightFormat` and `LinearConfig` bindings track new signatures |
| `src/turbomind/models/linear_weight.{h,cc}` | Single-phase `LinearConfig{input_dim, output_dim, format, has_bias, epilogue}`; `configure(in, out, DataFormat, has_bias)`; pre-alloc; `prepare()` derives grouped from parent chain; no `set_weight_spec`, `set_grouped`, `preprocess`, `LinearPolicy`, public `data_type`/`weight_format`/`group_size` fields; three DataFormats (`format`, `input_format`, `output_format`) |
| `src/turbomind/models/ffn_weight.cc` | Post-hoc `set_grouped` loop and `epilogue = kGatedSilu` assignment deleted |
| `src/turbomind/models/moe_weight.cc` | `LinkLinearExperts` reads via `input_format.dtype` instead of `input_dtype()` policy path |
| `src/turbomind/models/model_weight.{h,cc}` | `tok_embeddings` moves from `CHILDREN` (LinearWeight) to `PARAMS` (Tensor); `verify()` + `vocab_size` derivation updated |
| `src/turbomind/models/language_model.cc` | `weights_.tok_embeddings->weight` → `weights_.tok_embeddings` (Tensor access) |
| `src/turbomind/models/llama/LlamaLinear.cu` | Synthesizes `gemm::QuantDesc` from `input_format` / `format` at dispatch time |
| `src/turbomind/kernels/gemm/test/testbed_v3.h` | Three `configure`/`set_weight_spec` call sites collapse to single `configure(in, out, DataFormat, false)` |
| `lmdeploy/turbomind/deploy/kind_map.py` | `FormatKind` enum; `WeightFormat` frozen descriptor (no block sizes); `make_data_format` factory; `build_linear(..., compute_dtype, blocks: BlockSize)` |
| `lmdeploy/turbomind/deploy/linear.py` | `Linear.data_format: _tm.DataFormat` mandatory; `__post_init__` guards; `preprocess_linear` removed |
| `lmdeploy/turbomind/deploy/converter.py` | `_parse_quant_blocks` is sole `group_size` → `BlockSize` translator; no defaults |
| `lmdeploy/turbomind/deploy/spec.py` | `__init__(..., blocks: BlockSize)`; `_linear` passes it; `token_embeds` / `lm_head` become `_commit_tensor` / `_commit_linear` with tp/ranks kwargs |
| `lmdeploy/turbomind/deploy/builder/_base.py` | `_commit_linear(..., *, tp, ranks, epilogue)`; `_commit_tensor(..., *, tp, ranks)`; `_make_linear_config_for(..., *, epilogue)`; `_copy_linear_to_handle` helper; no alloc at commit time |
| `lmdeploy/turbomind/deploy/builder/linear.py` | **DELETED** |
| `lmdeploy/turbomind/deploy/builder/__init__.py` | Exports of `LinearBuilder` / `make_linear_config` removed |
| `lmdeploy/turbomind/deploy/builder/ffn.py` | `add_ffn` supplies `epilogue=kGatedSilu` kwarg on fused `w1w3` commit |
| `lmdeploy/turbomind/deploy/builder/attention.py`, `mla.py`, `deltanet.py` | `Linear(...)` constructors pass a real `data_format`; `fmt.block_out` readers switch to `linear.data_format.block_sizes[...]` |
| `lmdeploy/turbomind/deploy/source_model/*_spec.py` | `model()` calls `self.token_embeds(root, ...)`, `self.lm_head(root, ...)`; `_embed_key` / `_norm_key` normalized to prefixes; direct `build_linear` call sites updated |

### Test models

Pick via model-server MCP, cache once, reuse across all tasks:

| Family | Purpose | Suggested model | Tasks exercising it |
| --- | --- | --- | --- |
| trivial dense | baseline | smallest Qwen3 dense | 1, 3, 4, 5, 6 |
| AWQ | int4 groupwise | small AWQ Qwen / Llama | 1, 3, 6 |
| FP8 | FP8-native on SM90 | GLM-4.7-Flash-FP8 (if cached) or other FP8 | 1, 2, 3 |
| MXFP4 | fixed 32 block_in | GPT-OSS | 3 |
| MLA | MLA fold+pad | GLM-4.7-Flash via `glm4_moe_lite_spec` | 3 |
| DeltaNet | split_qkv + fuse_gdn | Qwen3.5 with linear-attention layers | 3, 6 |
| compressed-tensors gs=32 | latent-bug regression | any CT-gs32 model if cached | 3 |

Record the resolved `$MODEL_PATH` and `$CACHE_DIR` per family so you can recall them without re-querying.

---

## Task 1: MakeLinearWeightFormat takes (compute_dtype, weight_dtype, block_in, block_out); block_sizes tensor-shape-ordered

**Goal:** Change the C++ factory signature from `(compute, weight, group_size)` to `(compute, weight, block_in, block_out)`. Internally store `block_sizes = {block_in, block_out}` where index `i` corresponds to the described weight tensor's `shape[i]` (TM layout `[in, out]`). Update all readers that today hardcode `block_sizes[1]` as "the grouping dim" to read the tensor-correct index (pre-transpose: `block_sizes[0]` for K-grouped formats). This commit leaves `set_weight_spec` in place; it internally calls the new signature. Python's one caller in `kind_map.py`'s `WeightFormat.to_data_format` switches over.

**Files:**
- Modify: `src/turbomind/core/data_format.h:38`
- Modify: `src/turbomind/core/data_format.cc:21-59`
- Modify: `src/turbomind/core/test_data_format.cc:20-73` (six test call sites)
- Modify: `src/turbomind/python/bind.cpp:439-440`
- Modify: `src/turbomind/kernels/gemm/test/testbed_v3.h:298-308` (three call sites: keep using set_weight_spec)
- Modify: `src/turbomind/models/linear_weight.cc` (set_weight_spec internal call; any `block_sizes[1]` reads for K-block)
- Modify: `src/turbomind/models/moe_weight.cc` (reads via tensor layout, no behavioral change today)
- Modify: `lmdeploy/turbomind/deploy/kind_map.py:89-106` (`WeightFormat.to_data_format`)

- [ ] **Step 1.1: Update C++ factory declaration**

In `src/turbomind/core/data_format.h`, replace the declaration at line 38:

```cpp
/// Factory: create a DataFormat for linear weight storage.
/// block_sizes is stored in TM weight-tensor [in, out] order:
/// block_sizes[0] = block_in (along K), block_sizes[1] = block_out (along N).
DataFormat MakeLinearWeightFormat(DataType compute_dtype,
                                  DataType weight_dtype,
                                  int      block_in,
                                  int      block_out);
```

- [ ] **Step 1.2: Rewrite the factory body**

In `src/turbomind/core/data_format.cc`, replace lines 21-59 with:

```cpp
DataFormat MakeLinearWeightFormat(DataType compute_dtype,
                                  DataType weight_dtype,
                                  int      block_in,
                                  int      block_out)
{
    DataFormat fmt;
    fmt.dtype       = weight_dtype;
    fmt.block_sizes = {block_in, block_out};

    if (IsTrivialFloatType(weight_dtype)) {
        TM_CHECK(block_in == 1 && block_out == 1)
            << "Trivial weight requires block_in == block_out == 1, got ("
            << block_in << ", " << block_out << ")";
        return fmt;
    }

    if (weight_dtype == kFloat8_e4m3) {
        TM_CHECK(block_in == 128 && block_out == 128)
            << "FP8 requires block_in == block_out == 128, got ("
            << block_in << ", " << block_out << ")";
        fmt.scales.dtype = kFloat;
        return fmt;
    }

    if (weight_dtype == kFloat4_e2m1) {
        TM_CHECK(block_in >= 1 && block_out >= 1)
            << "FP4 requires block_in >= 1 and block_out >= 1, got ("
            << block_in << ", " << block_out << ")";
        fmt.scales.dtype = kUint8;
        return fmt;
    }

    if (weight_dtype == kUint4 || weight_dtype == kUint8) {
        TM_CHECK(block_in >= 1 && block_in <= 256 && block_out >= 1)
            << "Quantized weight requires block_in in [1, 256] and block_out >= 1, got ("
            << block_in << ", " << block_out << ")";
        fmt.scales.dtype = compute_dtype;
        fmt.zeros.dtype  = compute_dtype;
        return fmt;
    }

    TM_CHECK(0) << "Unsupported weight format: " << to_string(weight_dtype);
    return fmt;
}
```

- [ ] **Step 1.3: Update the tests**

In `src/turbomind/core/test_data_format.cc`, replace each call site:

- Line 20 (trivial half): `MakeLinearWeightFormat(kHalf, kHalf, 1, 1)`
- Line 30 (FP8): `MakeLinearWeightFormat(kHalf, kFloat8_e4m3, 128, 128)`
- Line 41 (FP4): `MakeLinearWeightFormat(kHalf, kFloat4_e2m1, 1, 128)` (grouping on N, block_in=1 per today's semantics for FP4 with scales at the output dim — inspect the test assertions and adjust block_in/block_out to match pre-existing expected shapes)
- Line 52 (Uint4 gs=128): `MakeLinearWeightFormat(kHalf, kUint4, 128, 1)` (grouping on K)
- Line 64 (Uint8 gs=64): `MakeLinearWeightFormat(kBfloat16, kUint8, 64, 1)`
- Line 73 (bf16 trivial): `MakeLinearWeightFormat(kBfloat16, kBfloat16, 1, 1)`

Note: the choice `(block_in=128, block_out=1)` for U4/U8 encodes "grouped along K only", which matches AWQ/GPTQ standard. `(block_in=1, block_out=128)` for FP4 matches today's `block_sizes = {1, gs}` layout. Inspect each existing test's post-factory assertions (e.g. `EXPECT_EQ(fmt.block_sizes[...], ...)`) and flip the expected indices if the test asserts on specific positions.

- [ ] **Step 1.4: Update pybind binding signature**

In `src/turbomind/python/bind.cpp`, replace lines 439-440:

```cpp
m.def("MakeLinearWeightFormat", &turbomind::MakeLinearWeightFormat,
      py::arg("compute_dtype"),
      py::arg("weight_dtype"),
      py::arg("block_in"),
      py::arg("block_out"));
```

- [ ] **Step 1.5: Update `LinearWeight::set_weight_spec` to call the new signature**

In `src/turbomind/models/linear_weight.cc`, locate the body of `set_weight_spec` (around line 95) that currently reads:

```cpp
format = MakeLinearWeightFormat(data_type, weight_format, group_size);
```

Replace with:

```cpp
// Map legacy group_size scalar to (block_in, block_out) per format-specific convention.
// This stays only until Task 3 removes set_weight_spec entirely.
int block_in, block_out;
if (IsTrivialFloatType(weight_format)) {
    block_in = 1;
    block_out = 1;
} else if (weight_format == kFloat8_e4m3) {
    block_in = 128;
    block_out = 128;
} else if (weight_format == kFloat4_e2m1) {
    // Keep today's semantics: scales on N dim, block_sizes = {1, gs} in TM [in, out] order.
    block_in = 1;
    block_out = group_size;
} else {
    // Uint4 / Uint8: grouped along K.
    block_in = group_size;
    block_out = 1;
}
format = MakeLinearWeightFormat(data_type, weight_format, block_in, block_out);
policy = ResolveLinearPolicy(format, data_type, getSMVersion());
```

- [ ] **Step 1.6: Update `block_sizes[1]` readers to read tensor-correct index**

Search `src/turbomind/models/linear_weight.cc` and `src/turbomind/models/moe_weight.cc` for `block_sizes[1]`. Every read where the `block_sizes` is on a tensor in TM `[in, out]` layout and we want "the K-dim block" must now read `block_sizes[0]`. The FP8 case (`block_sizes = {128, 128}`) is numerically invariant.

Concretely, in `linear_weight.cc::ResolveLinearPolicy`:

```cpp
if (format.dtype == kFloat8_e4m3) {
    int gs = format.block_sizes[0];      // was [1]; FP8 is symmetric so value unchanged
    p.weight_quant = gemm::QuantDesc{gemm::QuantType::kB, gs};
    if (sm == 90) {
        p.input_dtype = kFloat8_e4m3;
        p.input_quant = gemm::QuantDesc{gemm::QuantType::kK, gs};
    }
    return p;
}
if (format.dtype == kFloat4_e2m1) {
    int gs = format.block_sizes[1];      // FP4 groups on N per today's convention: leave [1]
    p.weight_quant = gemm::QuantDesc{gemm::QuantType::kK, gs};
    return p;
}
if (format.dtype == kUint4 || format.dtype == kUint8) {
    int gs = format.block_sizes[0];      // was [1]; U4/U8 groups on K, new storage is [block_in=gs, block_out=1]
    p.weight_quant = gemm::QuantDesc{gemm::QuantType::kK, gs};
    return p;
}
```

Also update the `scales` shape computation in `prepare()` — if any place reads `format.block_sizes[1]` to compute scales dim, swap based on the tensor being read. For the `input_dim / group_size` expression in the general quantization path (`linear_weight.cc` around line 272), `group_size` is replaced by the block size along the K dim of the weight tensor, which post-swap is `block_sizes[0]` for U4/U8 weights and still `block_sizes[1]` for FP4 weights. Add a local helper at the top of that block:

```cpp
int k_block = (format.dtype == kFloat4_e2m1)
              ? format.block_sizes[1]
              : format.block_sizes[0];
// ... use k_block where today's code used group_size
```

In `moe_weight.cc::LinkLinearExperts`, the only read of block_sizes (none today directly — it uses `e0.weight_format` and `e0.input_dtype()`) stays unchanged.

- [ ] **Step 1.7: Update the Python caller in `kind_map.py`**

In `lmdeploy/turbomind/deploy/kind_map.py`, locate `WeightFormat.to_data_format` (around line 89-105). Replace the body:

```python
def to_data_format(self, cpp_dtype: int, group_size: int = 0):
    """Construct a C++ DataFormat from this WeightFormat. Bridging helper kept
    only through Task 1; Task 3 removes it in favor of make_data_format."""
    if self.block_in is None:
        return None
    gs = group_size if self.block_in == 0 else self.block_in
    if gs == 0:
        return None
    if self.cpp_dtype_name is not None:
        dt = getattr(_tm.DataType, self.cpp_dtype_name, None)
        if dt is not None:
            # Translate legacy gs to (block_in, block_out) per format convention.
            if dt == _tm.DataType.TYPE_FP8_E4M3:
                return _tm.MakeLinearWeightFormat(cpp_dtype, dt, 128, 128)
            if dt == _tm.DataType.TYPE_FP4_E2M1:
                return _tm.MakeLinearWeightFormat(cpp_dtype, dt, 1, gs)
            # TYPE_UINT4 / TYPE_UINT8: grouped along K
            return _tm.MakeLinearWeightFormat(cpp_dtype, dt, gs, 1)
    return None
```

This helper is dead except in historical code paths; it's kept behavior-preserving for C1 and replaced wholesale in Task 3.

- [ ] **Step 1.8: Build and verify**

```
cd /data/lmdeploy-modeling/build && ninja
```

Expected: compiles with no errors. C++ tests that were updated still pass:
```
cd /data/lmdeploy-modeling/build && ./tests/test_data_format
```
Expected: all test cases pass.

Then smoke-test one trivial dense, one AWQ, one FP8 at tp=1 per the template in Preliminaries. Expected: coherent generation for all three; generated tokens ≥ 100.

- [ ] **Step 1.9: Commit**

```bash
cd /data/lmdeploy-modeling && git add -A && git commit -m "refactor(core): MakeLinearWeightFormat takes (compute, weight, block_in, block_out)

Replace the overloaded (compute, weight, group_size) signature where a
single group_size int was secretly populating three different block_sizes
patterns depending on weight dtype, with an explicit (block_in, block_out)
pair. block_sizes is stored in weight-tensor-shape order: block_sizes[i]
corresponds to the described tensor's shape[i] in TM [in, out] layout.
set_weight_spec stays as a wrapper that maps group_size to (block_in,
block_out) per format until Task 3 removes it entirely. Readers that
today hardcode block_sizes[1] as 'the K-dim block' are updated to read
block_sizes[0] for U4/U8 (grouped along K); FP4 stays on [1] (grouped
along N); FP8 is numerically invariant."
```

---

## Task 2: Drop LinearPolicy, carry input/output DataFormats on LinearWeight

**Goal:** Replace the `LinearPolicy { input_dtype, output_dtype, input_quant, weight_quant }` struct with two `DataFormat` fields (`input_format`, `output_format`) on `LinearWeight`. Introduce a pure function `DeriveActivationFormats(weight_format, compute_dtype, sm) -> pair<DataFormat, DataFormat>` replacing `ResolveLinearPolicy`. `LlamaLinear::Forward` synthesizes `gemm::QuantDesc`s on-demand from the three DataFormats at dispatch time.

**Files:**
- Modify: `src/turbomind/models/linear_weight.h:30-44` (struct decl), `45-105` (class fields)
- Modify: `src/turbomind/models/linear_weight.cc:23-57` (ResolveLinearPolicy → DeriveActivationFormats), uses of `policy.input_dtype`, `policy.output_dtype`
- Modify: `src/turbomind/models/moe_weight.cc:65-67` (input_dtype() reader)
- Modify: `src/turbomind/models/llama/LlamaLinear.{h,cu}` (Forward reads DataFormat; QuantDesc synthesized locally)

- [ ] **Step 2.1: Replace `LinearPolicy` with activation-format fields**

In `src/turbomind/models/linear_weight.h`, delete lines 34-40 (the `LinearPolicy` struct). Replace the `ResolveLinearPolicy` declaration at line 43 with:

```cpp
/// Derive (input_format, output_format) for a GEMM whose weight uses `weight_format`.
/// compute_dtype is the model's activation dtype; sm selects hardware path.
std::pair<DataFormat, DataFormat>
DeriveActivationFormats(const DataFormat& weight_format, DataType compute_dtype, int sm);
```

In the same file, replace the "Derived (computed once in do_allocate via ResolveLinearPolicy)" block (lines 80-85) with:

```cpp
// --- Derived (computed once in configure() via DeriveActivationFormats) ---
DataFormat format{};         // weight storage format; alias for cfg.format
DataFormat input_format{};   // activation input format
DataFormat output_format{};  // activation output format

DataType input_dtype()  const { return input_format.dtype;  }
DataType output_dtype() const { return output_format.dtype; }
```

Delete the `LinearPolicy policy{};` line.

- [ ] **Step 2.2: Rewrite `DeriveActivationFormats` body**

In `src/turbomind/models/linear_weight.cc`, replace the `ResolveLinearPolicy` function body (lines 23-57) with:

```cpp
std::pair<DataFormat, DataFormat>
DeriveActivationFormats(const DataFormat& weight_format, DataType compute_dtype, int sm)
{
    DataFormat out;
    out.dtype = compute_dtype;
    out.block_sizes = {1, 1};

    DataFormat in_fmt = out;  // default: trivial activation input

    if (!weight_format.is_quantized()) {
        return {in_fmt, out};
    }

    if (weight_format.dtype == kFloat8_e4m3 && sm == 90) {
        // FP8 native path: activation input is FP8 with same block sizes as weight.
        in_fmt.dtype       = kFloat8_e4m3;
        in_fmt.block_sizes = weight_format.block_sizes;
        in_fmt.scales.dtype = kFloat;
        return {in_fmt, out};
    }

    // All other quantized cases: trivial activation in/out (compute_dtype).
    return {in_fmt, out};
}
```

- [ ] **Step 2.3: Update `configure`, `set_weight_spec` to populate the new fields**

In `src/turbomind/models/linear_weight.cc`, in `configure()` (around line 63), replace the "Default policy for trivial (non-quantized) weights" block with:

```cpp
// Default activation formats for trivial weights; overridden if set_weight_spec
// sees a quantized format.
input_format  = DataFormat{data_type, {1, 1}};
output_format = DataFormat{data_type, {1, 1}};
```

In `set_weight_spec()` (around line 95), after the `format = MakeLinearWeightFormat(...)` call, replace the `policy = ResolveLinearPolicy(...)` line with:

```cpp
std::tie(input_format, output_format) =
    DeriveActivationFormats(format, data_type, getSMVersion());
```

- [ ] **Step 2.4: Update `prepare()` to read from `input_format` / `output_format`**

In `src/turbomind/models/linear_weight.cc` `prepare()`, replace every `policy.input_dtype`, `policy.output_dtype`, `input_dtype()`, `output_dtype()` access — the accessors still work (they now read `input_format.dtype`), but any remaining `policy.` references become compile errors. Global search within the file:

- `policy.input_dtype` → `input_format.dtype`
- `policy.output_dtype` → `output_format.dtype`
- `policy.weight_quant` / `policy.input_quant` — removed; see LlamaLinear step below for where the `QuantDesc` is synthesized at dispatch time instead.

- [ ] **Step 2.5: Update `copy_metadata_to`**

In `src/turbomind/models/linear_weight.cc`, replace the body of `copy_metadata_to` (lines 75-89) with:

```cpp
void LinearWeight::copy_metadata_to(LinearWeight& dst) const
{
    dst.input_dim     = input_dim;
    dst.output_dim    = output_dim;
    dst.group_size    = group_size;
    dst.data_type     = data_type;
    dst.weight_format = weight_format;
    dst.format        = format;
    dst.input_format  = input_format;
    dst.output_format = output_format;
    dst.epilogue      = epilogue;
    dst.has_bias_     = has_bias_;
    dst.is_grouped_   = is_grouped_;
    dst.k_desc        = k_desc;
    dst.q_desc        = q_desc;
}
```

- [ ] **Step 2.6: Update `moe_weight.cc`**

In `src/turbomind/models/moe_weight.cc::LinkLinearExperts` (line 65), the call `d.input_dtype()` already reads the new accessor — no change needed, just confirm it compiles.

- [ ] **Step 2.7: Update `LlamaLinear::Forward` to synthesize QuantDescs locally**

In `src/turbomind/models/llama/LlamaLinear.cu`, locate the GEMM dispatch that today reads `weight.policy.input_quant` and `weight.policy.weight_quant` (grep for `.policy.`). Replace those reads with local synthesis:

```cpp
// Synthesize GEMM QuantDescs from the three DataFormats at dispatch time.
gemm::QuantDesc weight_quant{};
gemm::QuantDesc input_quant{};

if (weight.format.dtype == kFloat8_e4m3) {
    int gs = weight.format.block_sizes[0];  // FP8 is symmetric
    weight_quant = {gemm::QuantType::kB, gs};
    if (weight.input_format.dtype == kFloat8_e4m3) {
        input_quant = {gemm::QuantType::kK, gs};
    }
} else if (weight.format.dtype == kFloat4_e2m1) {
    int gs = weight.format.block_sizes[1];  // FP4 grouped on N
    weight_quant = {gemm::QuantType::kK, gs};
} else if (weight.format.dtype == kUint4 || weight.format.dtype == kUint8) {
    int gs = weight.format.block_sizes[0];  // grouped on K
    weight_quant = {gemm::QuantType::kK, gs};
}
```

Then pass `weight_quant` and `input_quant` where previously `weight.policy.*_quant` were used.

- [ ] **Step 2.8: Build and verify**

```
cd /data/lmdeploy-modeling/build && ninja
```

Expected: compiles. Then smoke-test the same three models from Task 1 at tp=1. Expected: coherent generation unchanged — this is a pure refactor, behavior identical.

- [ ] **Step 2.9: Commit**

```bash
cd /data/lmdeploy-modeling && git add -A && git commit -m "refactor(linear_weight): drop LinearPolicy, carry input/output DataFormats

A linear GEMM is fully described by three DataFormats: weight, input,
and output. LinearPolicy was a bespoke shape that duplicated what
DataFormat already encodes. Replace it with DataFormat input_format and
output_format fields on LinearWeight, derived by a new pure function
DeriveActivationFormats(weight_format, compute_dtype, sm) at configure()
time. LlamaLinear::Forward synthesizes QuantDescs on-demand from the
three DataFormats at GEMM dispatch rather than caching them as separate
state. No behavior change."
```

---

## Task 3: LinearConfig carries DataFormat/epilogue; pre-alloc at construction; kill set_weight_spec / set_grouped / preprocess; derive grouped at prepare() from parent chain; Python WeightFormat refactor; converter.py purge

**Goal:** This is the largest commit — C++ and Python atomic. Drops the two-phase init entirely. `LinearConfig` becomes `{input_dim, output_dim, format, has_bias, epilogue}`. `LinearWeight` constructor derives everything from `cfg` and pre-allocates all param slots. Python `WeightFormat` becomes a frozen factory with `FormatKind` enum and `weight_dtype: _tm.DataType`. `converter.py` purges `_DEFAULT_GROUP_SIZES` / `_SUPPORTED_GROUP_SIZES` / `_validate_quant_group_size` and introduces `_parse_quant_blocks` as the single `group_size` → `BlockSize` translator. `Spec` takes `blocks: BlockSize`. `_commit_linear` builds `LinearConfig` from `linear.data_format` with no intermediate alloc.

Due to breadth, this task is split into sub-tasks 3a–3g. All sub-tasks land as one atomic commit at 3h.

### Sub-task 3a: LinearConfig and LinearWeight header

**Files:**
- Modify: `src/turbomind/models/linear_weight.h`

- [ ] **Step 3a.1: New LinearConfig fields**

In `src/turbomind/models/linear_weight.h`, replace the `LINEAR_FIELDS` macro (lines 14-18) with:

```cpp
#define LINEAR_FIELDS(X) \
    X(int,        input_dim) \
    X(int,        output_dim) \
    X(DataFormat, format) \
    X(bool,       has_bias) \
    X(Epilogue,   epilogue)
```

Ensure the `DataFormat` include is visible; `Epilogue` comes from `src/turbomind/kernels/gemm/types.h` already included via `linear_weight.h` itself.

- [ ] **Step 3a.2: Simplify LinearWeight class body**

In the same file, replace the `LinearWeight` class declaration (lines 45-105) with:

```cpp
class LinearWeight: public core::Module {
public:
    const char* type() const override { return "LinearWeight"; }

    LinearWeight() = default;                           // for MoeWeight block_ (default ctor + copy_metadata_to)
    LinearWeight(const core::LinearConfig& cfg);        // single-phase: derives state, pre-allocs slots

    /// Reconfigure an empty LinearWeight (for testbed_v3.h).
    void configure(int input_dim, int output_dim,
                   const DataFormat& format, bool has_bias);

    void prepare() override;                            // asserts format.dtype != DataType{} at entry

    void copy_metadata_to(LinearWeight& dst) const;

    explicit operator bool() const noexcept { return static_cast<bool>(weight); }

    // --- dimensions ---
    int input_dim  = 0;
    int output_dim = 0;

    // --- three DataFormats fully describe the GEMM ---
    DataFormat format{};          // weight storage
    DataFormat input_format{};    // activation input
    DataFormat output_format{};   // activation output

    DataType input_dtype()  const { return input_format.dtype;  }
    DataType output_dtype() const { return output_format.dtype; }

    DataType data_type() const { return format.dtype; }  // compute dtype shortcut

    // --- GEMM config knobs ---
    Epilogue    epilogue{};
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
    bool has_bias_ = false;
};
```

Key deletions relative to Task 2's state:
- `void set_weight_spec(...)`, `void preprocess()`, `void set_grouped(bool)` — methods removed.
- `int group_size;` public field — removed (read via `format.block_sizes[...]` with the dim-appropriate index).
- `DataType data_type;` public field — removed (read via `data_type()` getter or `format.dtype`).
- `DataType weight_format;` public field — removed (read via `format.dtype`).
- `bool is_grouped_;` private field — removed.

### Sub-task 3b: LinearWeight constructor and prepare() — C++ implementation

**Files:**
- Modify: `src/turbomind/models/linear_weight.cc`

- [ ] **Step 3b.1: Rewrite constructor and configure**

Replace the existing `LinearWeight::LinearWeight(const core::LinearConfig& cfg)` and `LinearWeight::configure(...)` (lines 18-73) with:

```cpp
LinearWeight::LinearWeight(const core::LinearConfig& cfg)
{
    configure(cfg.input_dim, cfg.output_dim, cfg.format, cfg.has_bias);
    epilogue = cfg.epilogue;
}

void LinearWeight::configure(int input_dim, int output_dim,
                             const DataFormat& format, bool has_bias)
{
    this->input_dim  = input_dim;
    this->output_dim = output_dim;
    this->format     = format;
    this->has_bias_  = has_bias;

    // Derive activation formats now so prepare() can read them.
    std::tie(input_format, output_format) =
        DeriveActivationFormats(format, format.dtype, getSMVersion());

    // Pre-allocate param slots based on what the format declares.
    auto alloc = [](Tensor& slot, std::vector<ssize_t> shape, DataType dtype) {
        auto layout = Layout{std::move(shape)};
        slot = Tensor{std::move(layout), dtype, kDEVICE};
        // Zero-initialize so uninitialized reads are deterministic.
        check_cuda_error(cudaMemsetAsync(slot.raw_data(), 0, slot.byte_size(),
                                         core::Context::stream().handle()));
    };

    alloc(weight, {input_dim, output_dim}, format.dtype);

    if (format.scales.present()) {
        // Derive scales shape from weight shape / block_sizes.
        const int bs0 = format.block_sizes.size() > 0 ? format.block_sizes[0] : 1;
        const int bs1 = format.block_sizes.size() > 1 ? format.block_sizes[1] : 1;
        alloc(scales,
              {(input_dim + bs0 - 1) / bs0, (output_dim + bs1 - 1) / bs1},
              format.scales.dtype);
    }
    if (format.zeros.present()) {
        const int bs0 = format.block_sizes.size() > 0 ? format.block_sizes[0] : 1;
        const int bs1 = format.block_sizes.size() > 1 ? format.block_sizes[1] : 1;
        alloc(zeros,
              {(input_dim + bs0 - 1) / bs0, (output_dim + bs1 - 1) / bs1},
              format.zeros.dtype);
    }
    if (has_bias) {
        alloc(bias, {output_dim}, format.dtype);
    }
}
```

- [ ] **Step 3b.2: Delete `set_weight_spec` and `preprocess`**

Remove the entire `LinearWeight::set_weight_spec` function (the one we modified in Tasks 1/2) and the `LinearWeight::preprocess()` definition (around lines 108-114).

- [ ] **Step 3b.3: `prepare()` entry assertion + grouped derivation**

At the top of `LinearWeight::prepare()`, immediately after the early-return on empty weight, insert:

```cpp
TM_CHECK(format.dtype != DataType{})
    << "LinearWeight::prepare: format was never set (pre-Task-3 bypass?)";

// Derive grouped-GEMM selection from the module tree instead of a stored flag.
const bool is_grouped = [this] {
    for (auto* p = this->parent(); p; p = p->parent()) {
        if (auto* moe = dynamic_cast<MoeWeight*>(p)) {
            return moe->method() == MoeMethod::kFused;
        }
    }
    return false;
}();
```

Pass `is_grouped` where `is_grouped_` was previously used, e.g. the `GetConverters(..., is_grouped_, ...)` call becomes `GetConverters(..., is_grouped, ...)`. Add `#include "src/turbomind/models/moe_weight.h"` at the top of `linear_weight.cc`.

- [ ] **Step 3b.4: `prepare()` block_sizes swap on FP8-native transpose**

In the FP8 native branch (the one that transposes weight from `[in, out]` to `[out, in]`), after the `process(weight, k_desc, uint8_t{});` call, insert:

```cpp
// Keep block_sizes in sync with the post-transpose weight shape.
std::swap(format.block_sizes[0], format.block_sizes[1]);
```

- [ ] **Step 3b.5: Rewrite `copy_metadata_to`**

Replace the body to reflect the new field set:

```cpp
void LinearWeight::copy_metadata_to(LinearWeight& dst) const
{
    dst.input_dim     = input_dim;
    dst.output_dim    = output_dim;
    dst.format        = format;
    dst.input_format  = input_format;
    dst.output_format = output_format;
    dst.epilogue      = epilogue;
    dst.has_bias_     = has_bias_;
    dst.k_desc        = k_desc;
    dst.q_desc        = q_desc;
}
```

### Sub-task 3c: Delete post-hoc mutations in FfnWeight::prepare

**Files:**
- Modify: `src/turbomind/models/ffn_weight.cc`

- [ ] **Step 3c.1: Delete `set_grouped` loop and `epilogue = kGatedSilu` assignment**

In `src/turbomind/models/ffn_weight.cc::prepare` (lines 25-47), replace the body with:

```cpp
void FfnWeight::prepare()
{
    Module::prepare();  // recurse into children
}
```

The two removed responsibilities are now owned by:
- Epilogue: Python caller passes `epilogue=kGatedSilu` via `_commit_linear` kwarg on `w1w3` (sub-task 3f).
- Grouped-ness: `LinearWeight::prepare()` walks parent chain to find a kFused MoeWeight (sub-task 3b.3).

### Sub-task 3d: Testbed single-phase configure

**Files:**
- Modify: `src/turbomind/kernels/gemm/test/testbed_v3.h`

- [ ] **Step 3d.1: Collapse three configure/set_weight_spec pairs**

In `src/turbomind/kernels/gemm/test/testbed_v3.h` `GenerateWeight`, replace lines 298-308:

```cpp
auto fmt_trivial = MakeLinearWeightFormat(data_type, data_type, 1, 1);
int block_in, block_out;
if (weight_type == kFloat8_e4m3) {
    block_in = 128; block_out = 128;
} else if (weight_type == kFloat4_e2m1) {
    block_in = 1; block_out = group_size;
} else {
    block_in = group_size; block_out = 1;
}
auto fmt_quant = MakeLinearWeightFormat(data_type, weight_type, block_in, block_out);

original.configure(input_dim, output_dim, fmt_trivial, false);
original.param("weight").alloc({(size_t)input_dim, (size_t)output_dim}, data_type);
rng_.NormalFloat(original.weight(), 1., .1);

quant.configure(input_dim, output_dim, fmt_quant, false);
quant.param("weight").alloc({(size_t)input_dim, (size_t)output_dim}, weight_type);

dequant.configure(input_dim, output_dim, fmt_trivial, false);
dequant.param("weight").alloc({(size_t)input_dim, (size_t)output_dim}, data_type);
```

(The explicit `param("weight").alloc(...)` lines stay because the test uses non-standard shapes in some cases and the testbed's expectations are validated elsewhere; keeping the explicit alloc makes the behavior invariant independent of the constructor's pre-alloc.)

### Sub-task 3e: Pybind LinearConfig binding

**Files:**
- Modify: `src/turbomind/python/bind.cpp`

- [ ] **Step 3e.1: Confirm binding auto-generates the new fields**

`LinearConfig` is bound via `bind_config<LinearConfig>` (line 448). The X-macro `LINEAR_FIELDS` (updated in 3a.1) auto-generates the Python-side settable fields. Run `ninja` after changes and confirm the generated binding exposes `input_dim`, `output_dim`, `format`, `has_bias`, `epilogue`. If `bind_config` doesn't automatically handle `DataFormat` values, add explicit casting support:

Check `bind.cpp` for how `bind_config` handles types. If it only handles primitive types, add an overload or explicit binding for `DataFormat`. The `DataFormat` class is already bound (lines 431-437); its fields are read-only there. For setting on LinearConfig, the field assignment path (`cfg.format = ...`) must accept a `_tm.DataFormat` object.

### Sub-task 3f: Python-side refactor

**Files:**
- Modify: `lmdeploy/turbomind/deploy/kind_map.py`
- Modify: `lmdeploy/turbomind/deploy/linear.py`
- Modify: `lmdeploy/turbomind/deploy/converter.py`
- Modify: `lmdeploy/turbomind/deploy/spec.py`
- Modify: `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py`, `qwen3_spec.py`, `glm4_moe_lite_spec.py`, `gpt_oss_spec.py`
- Modify: `lmdeploy/turbomind/deploy/builder/_base.py`
- Modify: `lmdeploy/turbomind/deploy/builder/ffn.py`

- [ ] **Step 3f.1: Rewrite `WeightFormat` and singletons in `kind_map.py`**

At the top of `lmdeploy/turbomind/deploy/kind_map.py`, after the existing imports, add:

```python
import enum
```

Replace the `WeightFormat` dataclass (lines 30-117) with:

```python
class FormatKind(enum.Enum):
    TRIVIAL            = "trivial"
    AWQ                = "awq"
    GPTQ               = "gptq"
    COMPRESSED_TENSORS = "compressed-tensors"
    FP8                = "fp8"
    MXFP4              = "mxfp4"


@dataclass(frozen=True)
class WeightFormat:
    """Constant per-format-kind descriptor. A factory of DataFormats.

    Owns checkpoint-side concerns (suffix map, normalizer, packer, detection,
    optional dequant / zeros synthesis) plus a factory method that produces a
    concrete ``_tm.DataFormat`` for a particular linear given compute dtype
    and block sizes. Holds no block-size state itself.
    """
    kind:          FormatKind
    suffix_map:    dict[str, str]
    normalizer:    Callable[[Tensor, str], Tensor]
    packer:        Callable[[Tensor, str], Tensor] | None
    weight_dtype:  '_tm.DataType | None'  # None iff kind == TRIVIAL
    accepts:       Callable[[dict[str, Tensor]], bool]
    zeros_factory: Callable[[Tensor], Tensor] | None
    dequant:       Callable[[dict[str, Tensor]], dict[str, Tensor]] | None

    def __hash__(self) -> int:
        return hash(self.kind)

    def make_data_format(self, *,
                         compute_dtype,
                         block_in: int,
                         block_out: int):
        if self.kind is FormatKind.TRIVIAL:
            assert block_in == 1 and block_out == 1, (
                f"TRIVIAL requires block_in == block_out == 1, "
                f"got ({block_in}, {block_out})")
            return _tm.MakeLinearWeightFormat(compute_dtype, compute_dtype,
                                              block_in, block_out)
        assert block_in >= 1 and block_out >= 1, (
            f"{self.kind.value} requires block_in >= 1 and block_out >= 1, "
            f"got ({block_in}, {block_out})")
        return _tm.MakeLinearWeightFormat(compute_dtype, self.weight_dtype,
                                          block_in, block_out)

    def complete_tensors(self, tensors: dict[str, Tensor]) -> None:
        if self.zeros_factory is not None and "scales" in tensors and "zeros" not in tensors:
            tensors["zeros"] = self.zeros_factory(tensors["scales"])
```

Replace the singleton constructors (lines 418-494) with:

```python
TRIVIAL_FORMAT = WeightFormat(
    kind=FormatKind.TRIVIAL,
    suffix_map=TRIVIAL_SUFFIXES,
    normalizer=_normalize_trivial,
    packer=None,
    weight_dtype=None,
    accepts=_accepts_trivial,
    zeros_factory=None,
    dequant=None,
)

AWQ_FORMAT = WeightFormat(
    kind=FormatKind.AWQ,
    suffix_map=AWQ_SUFFIXES,
    normalizer=_normalize_awq,
    packer=_pack_u4_qweight,
    weight_dtype=_tm.DataType.TYPE_UINT4,
    accepts=_accepts_awq,
    zeros_factory=None,
    dequant=_dequant_awq,
)

GPTQ_FORMAT = WeightFormat(
    kind=FormatKind.GPTQ,
    suffix_map=GPTQ_SUFFIXES,
    normalizer=_normalize_gptq,
    packer=_pack_u4_qweight,
    weight_dtype=_tm.DataType.TYPE_UINT4,
    accepts=_accepts_gptq,
    zeros_factory=_zeros_int4_symmetric,
    dequant=None,
)

COMPRESSED_TENSOR_FORMAT = WeightFormat(
    kind=FormatKind.COMPRESSED_TENSORS,
    suffix_map=COMPRESSED_TENSOR_SUFFIXES,
    normalizer=_normalize_compressed_tensor,
    packer=_pack_u4_qweight,
    weight_dtype=_tm.DataType.TYPE_UINT4,
    accepts=_accepts_compressed_tensor,
    zeros_factory=_zeros_int4_symmetric,
    dequant=None,
)

FP8_FORMAT = WeightFormat(
    kind=FormatKind.FP8,
    suffix_map=FP8_SUFFIXES,
    normalizer=_normalize_fp8,
    packer=None,
    weight_dtype=_tm.DataType.TYPE_FP8_E4M3,
    accepts=_accepts_fp8,
    zeros_factory=None,
    dequant=_dequant_fp8,
)

MXFP4_FORMAT = WeightFormat(
    kind=FormatKind.MXFP4,
    suffix_map=MXFP4_SUFFIXES,
    normalizer=_normalize_mxfp4,
    packer=_pack_mxfp4_weight,
    weight_dtype=_tm.DataType.TYPE_FP4_E2M1,
    accepts=_accepts_mxfp4,
    zeros_factory=None,
    dequant=None,
)
```

- [ ] **Step 3f.2: Add `BlockSize` NamedTuple and rewrite `build_linear`**

Near the top of `kind_map.py`, after existing imports, add:

```python
from typing import NamedTuple


class BlockSize(NamedTuple):
    block_in: int
    block_out: int
```

Replace `build_linear` (line 536 onwards) with:

```python
def build_linear(
    params: dict[str, torch.Tensor],
    prefix: str,
    *,
    compute_dtype,
    blocks: BlockSize,
    index: int | None = None,
) -> 'Linear | None':
    """Build a Linear bundle from checkpoint tensors at *prefix*.

    The detected format governs which block sizes are used: TRIVIAL always
    uses (1, 1); quantized formats use the caller-provided *blocks*.
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

    if fmt.kind is FormatKind.TRIVIAL:
        data_format = fmt.make_data_format(
            compute_dtype=compute_dtype, block_in=1, block_out=1)
    else:
        data_format = fmt.make_data_format(
            compute_dtype=compute_dtype,
            block_in=blocks.block_in,
            block_out=blocks.block_out)

    tensors: dict[str, torch.Tensor] = {
        kind: fmt.normalizer(available[s], kind)
        for s, kind in fmt.suffix_map.items()
        if s in available
    }
    if not tensors:
        return None
    fmt.complete_tensors(tensors)

    return Linear(tensors=tensors, weight_format=fmt, data_format=data_format)
```

Remove the `from dataclasses import dataclass, replace` line and replace with `from dataclasses import dataclass` (no more `replace` usage).

- [ ] **Step 3f.3: Rewrite `converter.py` quant-config parsing**

In `lmdeploy/turbomind/deploy/converter.py`, delete `_DEFAULT_GROUP_SIZES` (line 29), `_SUPPORTED_GROUP_SIZES` (line 37), and `_validate_quant_group_size` (line 46). Replace with:

```python
from .kind_map import BlockSize


def _parse_quant_blocks(model_format, quant_config) -> BlockSize:
    """Translate quant-config into (block_in, block_out). Single translation
    point; no defaults.
    """
    if model_format in (None, 'hf'):
        return BlockSize(block_in=1, block_out=1)
    if model_format == 'mxfp4':
        return BlockSize(block_in=32, block_out=1)          # OCP MX standard
    gs = (quant_config or {}).get('group_size')
    if gs is None or gs < 1:
        raise ValueError(
            f"Format {model_format!r} requires group_size >= 1 in quant_config; "
            f"got {gs!r}.")
    if model_format in ('awq', 'gptq', 'compressed-tensors'):
        return BlockSize(block_in=gs, block_out=1)
    if model_format == 'fp8':
        if gs != 128:
            raise ValueError(f"FP8 requires group_size == 128, got {gs}")
        return BlockSize(block_in=128, block_out=128)
    raise ValueError(f"Unsupported model_format: {model_format}")
```

In `get_tm_config` (line 101), replace `_validate_quant_group_size(...)` with a direct call to `_parse_quant_blocks(engine_config.model_format, quant_config)`. Replace the `group_size or 0` passed to `spec_cls` with the resulting `blocks`:

```python
blocks = _parse_quant_blocks(engine_config.model_format, quant_config)
# ... existing flow through dtype / session_len / hf_cfg ...
spec = spec_cls(hf_cfg, engine_config, blocks=blocks)
```

- [ ] **Step 3f.4: Update `Spec.__init__` and `_linear`**

In `lmdeploy/turbomind/deploy/spec.py`, change the `__init__` signature (line 52):

```python
def __init__(self, hf_cfg: dict, engine_cfg: 'TurbomindEngineConfig',
             *, blocks: 'BlockSize'):
    self.hf_cfg = hf_cfg
    self.engine_cfg = engine_cfg
    self._blocks = blocks
    self._parse_base(hf_cfg)
```

Replace `_linear` (line 149):

```python
def _linear(self, pfx: str):
    from .kind_map import build_linear
    return build_linear(self.params, pfx,
                        compute_dtype=self._cpp_dtype(),
                        blocks=self._blocks)
```

- [ ] **Step 3f.5: Update every source_model spec to match**

Each spec that subclasses `TextModelSpec` has its own `__init__` that calls `super().__init__(hf_cfg, engine_cfg, group_size=group_size)`. Change these to pass `blocks=blocks` instead. Files:

- `lmdeploy/turbomind/deploy/source_model/qwen3_spec.py`
- `lmdeploy/turbomind/deploy/source_model/qwen3_5_spec.py`
- `lmdeploy/turbomind/deploy/source_model/glm4_moe_lite_spec.py`
- `lmdeploy/turbomind/deploy/source_model/gpt_oss_spec.py`

In each, change the `__init__` signature from `group_size: int = 0` to `blocks: BlockSize`, and `super().__init__(hf_cfg, engine_cfg, group_size=group_size)` to `super().__init__(hf_cfg, engine_cfg, blocks=blocks)`. Also update any direct `build_linear(..., block_in=self._group_size, block_out=self._group_size)` call site (e.g. `qwen3_5_spec.py:299-302`) to use `blocks=self._blocks`:

```python
gate_up_lin = build_linear(self.params, f'{pfx}.gate_up_proj',
                           index=expert_idx,
                           compute_dtype=self._cpp_dtype(),
                           blocks=self._blocks)
down_lin = build_linear(self.params, f'{pfx}.down_proj',
                        index=expert_idx,
                        compute_dtype=self._cpp_dtype(),
                        blocks=self._blocks)
```

Also replace `_pad_inter_size(raw_inter, self._group_size, ...)` with `_pad_inter_size(raw_inter, self._blocks.block_in, ...)` (or pass `block_out` if that's the active grouping dim — match the pre-existing semantics).

- [ ] **Step 3f.6: Rewrite `Builder._commit_linear` and `_commit_tensor` in `_base.py`**

In `lmdeploy/turbomind/deploy/builder/_base.py`, replace `_commit_linear` (around line 361) with:

```python
def _commit_linear(self, name: str, linear: Linear,
                   split_side: SplitSide | None = None,
                   *, tp=None, ranks=None,
                   epilogue=None):
    """Commit a Linear bundle to a named LinearWeight child on every GPU.

    Param slots are pre-allocated in the C++ LinearWeight constructor from
    cfg.format. This method only creates the child (on first call for name)
    and copies sharded tensor bytes.
    """
    self._ensure_handles()
    w = linear.tensors.get('weight')
    if w is None:
        return

    tp    = tp    if tp    is not None else self._tp
    ranks = ranks if ranks is not None else self._ranks
    if epilogue is None:
        epilogue = _tm.Epilogue.kNone

    lin_cfg = self._make_linear_config_for(linear, split_side, tp, epilogue=epilogue)
    effective_tp = tp if split_side else 1
    split_dim = _SPLIT_SIDE_TO_DIM.get(split_side) if split_side else None

    # Apply packer once, above the per-GPU loop.
    fmt = linear.weight_format
    tensors = ({k: fmt.packer(t, k) for k, t in linear.tensors.items()}
               if fmt.packer else linear.tensors)

    # Uniform TP-divisibility check.
    if effective_tp > 1 and split_dim is not None:
        for kind, tensor in tensors.items():
            kind_split_dim = split_dim
            if kind == 'bias' and split_side is SplitSide.INPUT:
                kind_split_dim = None
            if kind_split_dim is not None:
                d = tensor.shape[kind_split_dim]
                assert d % effective_tp == 0, (
                    f"TP split: {name}.{kind} dim {kind_split_dim} "
                    f"has size {d}, not divisible by tp={effective_tp}.")

    for i, handle in enumerate(self._handles):
        with self._contexts[i]:
            rank = (ranks[i] if effective_tp > 1 and ranks else 0)
            child = handle.child(name) or handle.create_child(name, lin_cfg)
            self._copy_linear_to_handle(child, tensors, split_side, effective_tp, rank)


def _make_linear_config_for(self, linear, split_side, tp, *, epilogue):
    w = linear.tensors['weight']
    in_dim, out_dim = w.shape[0], w.shape[-1]
    if split_side is SplitSide.OUTPUT:
        out_dim //= tp
    elif split_side is SplitSide.INPUT:
        in_dim  //= tp
    cfg = _tm.LinearConfig()
    cfg.input_dim  = in_dim
    cfg.output_dim = out_dim
    cfg.format     = linear.data_format
    cfg.has_bias   = 'bias' in linear.tensors
    cfg.epilogue   = epilogue
    return cfg


def _copy_linear_to_handle(self, handle, tensors, split_side, tp, rank):
    split_dim = _SPLIT_SIDE_TO_DIM.get(split_side) if split_side else None
    for kind, tensor in tensors.items():
        kind_split_dim = split_dim
        if kind == 'bias' and split_side is SplitSide.INPUT:
            kind_split_dim = None
        shard = _shard(tensor, kind_split_dim, tp, rank)
        _copy_shard_to_param(handle, kind, shard)
```

Replace `_commit_tensor` (around line 468) with:

```python
def _commit_tensor(self, name: str, tensor: torch.Tensor | None,
                   split_side: SplitSide | None = None,
                   *, tp=None, ranks=None):
    """Commit a raw tensor to a named parameter slot on the builder's handle.
    Used for non-LinearWeight params (norm, conv1d, scalars, sinks, root
    text-model's tok_embeddings).
    """
    self._ensure_handles()
    if tensor is None:
        return

    tp    = tp    if tp    is not None else self._tp
    ranks = ranks if ranks is not None else self._ranks

    effective_tp = tp if split_side else 1
    split_dim = _SPLIT_SIDE_TO_DIM.get(split_side) if split_side else None

    for i, handle in enumerate(self._handles):
        with self._contexts[i]:
            rank = (ranks[i] if effective_tp > 1 and ranks else 0)
            shard = _shard(tensor, split_dim, effective_tp, rank)
            _copy_shard_to_param(handle, name, shard)
```

Delete `_infer_cpp_linear_dtype` and `_infer_compute_dtype` helpers (they are no longer called — the packer-less trivial path was their consumer).

- [ ] **Step 3f.7: Update `_copy_shard_to_param`**

Since the C++ slot is now pre-allocated, `_copy_shard_to_param` no longer needs to call `alloc`. Replace its body (around line 214):

```python
def _copy_shard_to_param(handle, param_name: str, shard: torch.Tensor) -> None:
    """Move shard to GPU, cast dtype to match slot, copy bytes.

    The slot is pre-allocated by the C++ LinearWeight constructor; we only
    copy. byte_size mismatch raises immediately.
    """
    if not shard.is_cuda:
        shard = shard.cuda(0).contiguous()
    elif not shard.is_contiguous():
        shard = shard.contiguous()

    dst = handle.param(param_name).get()
    assert dst, f"param '{param_name}' not allocated on {handle.type()}"
    shard = _cast_shard_for_tm(shard, dst)
    assert dst.byte_size == shard.nbytes, (
        f"{param_name}: slot byte_size={dst.byte_size} != "
        f"shard.nbytes={shard.nbytes}")
    dst.copy_from(shard)
```

For raw-tensor params on non-LinearWeight handles (norm weight, conv1d, sinks, tok_embeddings — none pre-allocated in C++), keep an alloc path. Add:

```python
def _copy_shard_to_raw_param(handle, param_name, shard):
    """Alloc-then-copy for non-LinearWeight params (norm, conv1d, scalars)."""
    if not shard.is_cuda:
        shard = shard.cuda(0).contiguous()
    elif not shard.is_contiguous():
        shard = shard.contiguous()
    alloc_shape = list(shard.shape)
    alloc_dtype = _torch_dtype_to_cpp(shard.dtype)
    dst = handle.param(param_name).alloc(alloc_shape, alloc_dtype)
    shard = _cast_shard_for_tm(shard, dst)
    assert dst.byte_size == shard.nbytes
    dst.copy_from(shard)
```

And in `_commit_tensor`, call `_copy_shard_to_raw_param` instead of `_copy_shard_to_param`. Similarly in `_add_norm_child`.

- [ ] **Step 3f.8: Update `FfnBuilder.add_ffn`**

In `lmdeploy/turbomind/deploy/builder/ffn.py`, replace the `add_ffn` body (around line 111):

```python
def add_ffn(self, w1, w2, w3):
    """Fuse w1+w3 if possible, then commit. Epilogue is set only on the
    fused w1w3 commit when fused SiLU is active."""
    act_type = getattr(self.config, 'act_type', 0)
    if isinstance(act_type, int):
        act_type = {0: 'silu', 1: 'gpt-oss'}.get(act_type, 'silu')
    is_moe = getattr(self.config, 'fused_moe', False)

    fused = None
    fused_silu = False
    if w1 is not None and w3 is not None:
        fused, fused_silu = fuse_ffn_linears(
            w1, w3, self._tp, act_type, is_moe=is_moe)

    # Keep FfnConfig.fuse_silu in sync for any non-LinearConfig consumers.
    self.config.fuse_silu = fused_silu

    if fused is not None:
        epilogue = _tm.Epilogue.kGatedSilu if fused_silu else _tm.Epilogue.kNone
        self._commit_linear('w1w3', fused, SplitSide.OUTPUT, epilogue=epilogue)
    else:
        if w1 is not None:
            self._commit_linear('w1', w1, SplitSide.OUTPUT)
        if w3 is not None:
            self._commit_linear('w3', w3, SplitSide.OUTPUT)
    if w2 is not None:
        self._commit_linear('w2', w2, SplitSide.INPUT)
```

Remove the `model_dtype=model_dtype` kwargs from the `_commit_linear` calls (the new signature doesn't take `model_dtype` — the dtype comes from `linear.data_format.dtype`).

Similarly in `attention.py`, `mla.py`, `moe.py`, `deltanet.py`: remove every `model_dtype=self.config.data_type` kwarg passed to `_commit_linear`. The compute dtype is now entirely carried by `linear.data_format`.

- [ ] **Step 3f.9: Update `Linear` dataclass to make `data_format` mandatory**

In `lmdeploy/turbomind/deploy/linear.py`, change the `Linear` dataclass (line 122):

```python
@dataclass
class Linear:
    tensors: dict[str, Tensor]
    weight_format: 'WeightFormat'
    data_format: '_tm.DataFormat'

    def __post_init__(self):
        assert 'weight' in self.tensors, "Linear must carry a 'weight' tensor"
        assert self.data_format is not None, "Linear.data_format is mandatory"
```

Remove the `= field(default=None, compare=False, repr=False)` defaults. Every producer must now supply `data_format`.

In the same file, fix every `Linear(...)` constructor call site within `linear.py` (the `split_out_dim`, `split_in_dim`, `concat_out_dim`, `concat_in_dim`, `interleave_linears`, `chunk_linears` functions) to thread `data_format`. For trivial transforms that preserve format:

```python
return Linear(tensors=..., weight_format=self.weight_format,
              data_format=self.data_format)
```

For fusion helpers that combine two Linears, inherit from the first input's `data_format` since both inputs have the same format (already guarded by `_ensure_compatible_formats` upstream):

```python
return Linear(tensors=fused, weight_format=w1.weight_format,
              data_format=w1.data_format)
```

### Sub-task 3g: Thread `data_format` through builder fusion helpers

**Files:**
- Modify: `lmdeploy/turbomind/deploy/builder/attention.py`, `mla.py`, `deltanet.py`, `ffn.py`

- [ ] **Step 3g.1: Update `Linear(...)` constructors in every helper**

Every `Linear(tensors=..., weight_format=..., data_format=...)` call in the builder helpers must supply a real `data_format`. Typical patterns:

- `dequant_mixed`: when dequantizing, the output is trivial format, so `data_format = TRIVIAL_FORMAT.make_data_format(compute_dtype=linear.data_format.dtype, block_in=1, block_out=1)`.
- `fold_kv_b` / `pad_wo_input` (mla.py): output is in same format as input (trivial in practice for MLA); inherit `data_format` from input.
- `split_qkv` / `fuse_gdn` (deltanet.py): inherit from input.
- `split_output_gate` / `repeat_kv_for_tp` / `fuse_qkv` (attention.py): inherit from input via `@transform_tensors` decorator — update the decorator to thread `data_format` when constructing the output `Linear`.

For `@transform_tensors` in `_base.py` (around line 164), update the return-construction:

```python
outputs = tuple(
    Linear(ts, weight_format=first.weight_format,
           data_format=first.data_format)
    for ts in out_buckets)
```

This is already the pattern — just ensure no `Linear(ts, ...)` call omits `data_format`.

### Sub-task 3h: Build, verify full matrix, commit

- [ ] **Step 3h.1: Build**

```
cd /data/lmdeploy-modeling/build && ninja
```

Expected: compiles cleanly. If any symbol is missing (e.g. `set_weight_spec` still referenced somewhere outside updated files), track down and delete.

- [ ] **Step 3h.2: Full verification matrix**

Run `scripts/test_turbomind_model.py` on:

| Model | tp | Expected |
| --- | --- | --- |
| trivial dense | 1 | coherent; ≥100 tokens |
| trivial dense | 2 | coherent; ≥100 tokens |
| AWQ | 1 | coherent; ≥100 tokens |
| AWQ | 2 | coherent; ≥100 tokens |
| FP8 | 1 | coherent; ≥100 tokens |
| FP8 | 2 | coherent; ≥100 tokens |
| MXFP4 (GPT-OSS) | 1 | coherent; ≥100 tokens |
| MLA (GLM-4.7-Flash) | 2 | coherent; ≥100 tokens |
| DeltaNet (Qwen3.5) | 1 | coherent; ≥100 tokens |
| compressed-tensors gs=32, if cached | 1 | coherent; ≥100 tokens (latent-bug regression guard) |

Any gibberish output or exit code ≠ 0 → halt, bisect the failing sub-task against a prior-task commit, and fix before committing.

- [ ] **Step 3h.3: Commit**

```bash
cd /data/lmdeploy-modeling && git add -A && git commit -m "refactor(linear_weight): single-phase LinearConfig; pre-alloc slots; drop set_weight_spec/set_grouped/preprocess; Python WeightFormat factory; purge group_size defaults

Biggest atomic commit of the linear-lifecycle rethink.

C++:
- LinearConfig gains DataFormat format and Epilogue epilogue; drops data_type.
- LinearWeight(cfg) is single-phase: derives input_format/output_format via
  DeriveActivationFormats, pre-allocates weight/scales/zeros/bias param slots
  whose shapes come from cfg.format.block_sizes.
- set_weight_spec, set_grouped, preprocess, the old configure(dtype) overload,
  and public fields weight_format/group_size/data_type are deleted. The
  is_grouped_ private field is removed; prepare() walks the parent chain to
  find a kFused MoeWeight ancestor and derives grouped-GEMM selection at
  runtime.
- prepare() asserts format.dtype != DataType{} on entry; trivial-default
  branch deleted. FP8-native transpose also swaps format.block_sizes to
  preserve the invariant block_sizes[i] == weight.shape[i].
- FfnWeight::prepare's post-hoc set_grouped loop and epilogue=kGatedSilu
  assignment deleted; Python supplies epilogue directly on LinearConfig.

Python:
- WeightFormat becomes a frozen FormatKind-tagged descriptor with a
  make_data_format(compute_dtype, block_in, block_out) factory. No block-size
  fields on the descriptor; truly constant singletons.
- BlockSize NamedTuple replaces the scalar group_size. Spec stores
  self._blocks: BlockSize.
- converter.py: _DEFAULT_GROUP_SIZES / _SUPPORTED_GROUP_SIZES /
  _validate_quant_group_size deleted. _parse_quant_blocks is the sole
  translator from quant_config's group_size into BlockSize; raises on
  missing values; MXFP4 hardcodes (32, 1) per OCP MX standard; FP8 asserts
  gs==128; AWQ/GPTQ/CT map to (gs, 1) grouped-on-K.
- build_linear(params, prefix, *, compute_dtype, blocks, index) constructs
  Linear.data_format via the factory; trivial formats always use (1, 1)
  regardless of caller-declared blocks.
- _commit_linear / _commit_tensor accept tp/ranks kwargs for top-level
  linears; _commit_linear accepts epilogue. _make_linear_config_for is
  name-agnostic. _copy_shard_to_param no longer allocates (slots are
  pre-allocated); raw Tensor params go through _copy_shard_to_raw_param.
- FfnBuilder.add_ffn supplies epilogue=kGatedSilu when committing fused
  w1w3 with fused SiLU; no post-hoc mutation.
- Linear.data_format is mandatory (non-None); __post_init__ guards.

Compressed-tensors with group_size=32 now flows through correctly
(previously silently clobbered to 128)."
```

---

## Task 4: tok_embeddings as Tensor parameter

**Goal:** Move `tok_embeddings` from a `LinearWeight` child to a plain `core::Tensor` parameter on `ModelWeight`. Drop vocab-dim padding in the Python `token_embeds` function. Preserve `vocab_size_padded` derivation for downstream sampling kernels.

**Files:**
- Modify: `src/turbomind/models/model_weight.h:37-44` (X-macro)
- Modify: `src/turbomind/models/model_weight.cc:38-48` (verify / vocab_size)
- Modify: `src/turbomind/models/language_model.cc:194` (drop `->weight`)
- Modify: `lmdeploy/turbomind/deploy/spec.py` (`token_embeds`)
- Modify: `lmdeploy/turbomind/deploy/source_model/*_spec.py` (`model()`)

- [ ] **Step 4.1: ModelWeight X-macro reshape**

In `src/turbomind/models/model_weight.h`, replace:

```cpp
#define MODEL_WEIGHT_CHILDREN(X)         \
    X(LinearWeight,     tok_embeddings)  \
    X(LinearWeight,     output)          \
    X(NormWeight,       norm)            \
    X(core::ModuleList, layers)

#define MODEL_WEIGHT_PARAMS(X)
```

with:

```cpp
#define MODEL_WEIGHT_CHILDREN(X)         \
    X(LinearWeight,     output)          \
    X(NormWeight,       norm)            \
    X(core::ModuleList, layers)

#define MODEL_WEIGHT_PARAMS(X)           \
    X(tok_embeddings)
```

- [ ] **Step 4.2: Update `verify()` and `vocab_size` derivation in `model_weight.cc`**

In `src/turbomind/models/model_weight.cc`, replace the `vocab_size` / `verify` logic (lines 38-82):

```cpp
// tok_embeddings is now a Tensor parameter; it may be unpadded (vocab) or
// padded (historical). vocab_size_padded is always derived from the output
// LinearWeight's sharded output_dim * tp_size so downstream sampling kernels
// see the same value regardless of tok_embeddings padding.
if (tok_embeddings) {
    vocab_size = tok_embeddings.shape(0);
}
if (output) {
    vocab_size_padded = output->output_dim * tp_size;
} else {
    vocab_size_padded = round_up((size_t)vocab_size, (size_t)tp_size);
}
```

In `verify()`, replace the `missing.push_back("missing tok_embeddings")` block so that `!tok_embeddings` checks the Tensor instead of the child-pointer:

```cpp
if (!tok_embeddings) {
    missing.push_back(full_path() + ": missing tok_embeddings");
}
```

(The overloaded `operator bool` on Tensor returns true iff `byte_size > 0`.)

- [ ] **Step 4.3: Update `language_model.cc`**

In `src/turbomind/models/language_model.cc:194`, change:

```cpp
const auto& embedding_table = weights_.tok_embeddings->weight;
```

to:

```cpp
const auto& embedding_table = weights_.tok_embeddings;
```

- [ ] **Step 4.4: Update `spec.py::token_embeds`**

In `lmdeploy/turbomind/deploy/spec.py`, replace `token_embeds` (line 180):

```python
def token_embeds(self, root, key):
    """Commit tok_embeddings as a raw Tensor parameter on the root handle.
    No vocab-dim padding: the lookup never indexes past vocab_size - 1.
    """
    emb = self._get(key)
    if emb is None:
        return
    tp = self.engine_cfg.attn_tp_size * self.engine_cfg.attn_cp_size
    root._commit_tensor('tok_embeddings', emb,
                        split_side=SplitSide.OUTPUT,
                        tp=tp, ranks=self._attn_ranks)
```

Note: `key` is still the full key ending in `.weight` (e.g. `'model.embed_tokens.weight'`) per current convention.

- [ ] **Step 4.5: Update every spec's `model()` method**

In each of `qwen3_spec.py`, `qwen3_5_spec.py`, `glm4_moe_lite_spec.py`, `gpt_oss_spec.py`, change:

```python
root.tok_embeddings = self.token_embeds(self._embed_key)
```

to:

```python
self.token_embeds(root, self._embed_key)
```

(Remove the `LinearBuilder` attachment pattern for tok_embeddings. `lm_head` / `output` stays as `root.output = self.lm_head(lm_key)` for now; that changes in Task 5.)

- [ ] **Step 4.6: Build and verify**

```
cd /data/lmdeploy-modeling/build && ninja
```

Then run trivial dense tp=1 and tp=2. Expected: coherent output. Confirm `vocab_size_padded` still matches the old value by comparing token-id → logits shape (or just rely on the end-to-end test producing coherent results — any mismatch in `vocab_size_padded` usually manifests as garbage logits).

If tied-embeddings variant is available (e.g. Qwen3-0.5B with tied embeddings), run it at tp=1 to confirm the tied path still works (tok_embeddings unpadded; lm_head via LinearBuilder still padded).

- [ ] **Step 4.7: Commit**

```bash
cd /data/lmdeploy-modeling && git add -A && git commit -m "refactor(model_weight): tok_embeddings as Tensor parameter, drop vocab padding

tok_embeddings is a lookup table indexed by token id; it never participates
in a GEMM. Wrapping it in LinearWeight was scaffolding. Move it from
MODEL_WEIGHT_CHILDREN (LinearWeight) to MODEL_WEIGHT_PARAMS (Tensor).
language_model.cc drops the ->weight indirection.

The Python token_embeds function stops padding along vocab. Lookup logic
never exceeds vocab - 1, so padding rows were dead storage. Each rank's
shard still covers hidden / tp along the hidden dim — only the vocab-
padding behavior changes.

vocab_size_padded derivation shifts from tok_embeddings.shape(0) to
output->output_dim * tp_size so downstream sampling / penalty / logprob
kernels see unchanged values regardless of tok_embeddings padding."
```

---

## Task 5: Delete LinearBuilder, unify lm_head via self._linear

**Goal:** Remove `LinearBuilder`, `set_weight`, `make_linear_config`, `builder/linear.py` entirely. `lm_head` becomes a direct `root._commit_linear('output', self._linear(key), ...)` call with the same vocab-padding that today's `lm_head` function applies (preserved as a `pad_out_dim` on the `Linear` bundle).

**Files:**
- Delete: `lmdeploy/turbomind/deploy/builder/linear.py`
- Modify: `lmdeploy/turbomind/deploy/builder/__init__.py` (drop exports)
- Modify: `lmdeploy/turbomind/deploy/spec.py` (`lm_head`)
- Modify: `lmdeploy/turbomind/deploy/source_model/*_spec.py` (`model()`)

- [ ] **Step 5.1: Delete `builder/linear.py`**

```
cd /data/lmdeploy-modeling && git rm lmdeploy/turbomind/deploy/builder/linear.py
```

- [ ] **Step 5.2: Drop exports from `builder/__init__.py`**

In `lmdeploy/turbomind/deploy/builder/__init__.py`, remove the `LinearBuilder` and `make_linear_config` imports (line 11) and their entries in `__all__`.

- [ ] **Step 5.3: Rewrite `spec.py::lm_head`**

In `lmdeploy/turbomind/deploy/spec.py`, replace `lm_head` (line 193):

```python
def lm_head(self, root, key):
    """Commit the LM head as a LinearWeight child of the root text model.

    key is a prefix (e.g. 'lm_head' or 'model.embed_tokens' for tied
    embeddings); self._linear(key) probes .weight / .bias / quant suffixes
    and returns a Linear in TM [hidden, vocab] layout. We then pad along
    the output (vocab) dim so sampling kernels see a TP-divisible
    vocab_size_padded.
    """
    lin = self._linear(key)
    if lin is None:
        return
    tp = self.engine_cfg.attn_tp_size * self.engine_cfg.attn_cp_size
    padded_vocab = ((self._vocab_size + tp - 1) // tp) * tp
    # Guard: lm_head is almost always trivial format in practice.
    # Block-aligned padding for quantized lm_head is a separate concern.
    assert lin.weight_format.kind.value == 'trivial', (
        f"lm_head non-trivial format ({lin.weight_format.kind}) not supported "
        "by the simple vocab-padding path.")
    from .linear import Linear, pad_out_dim
    padded_tensors = {
        kind: pad_out_dim(t, padded_vocab, dim=-1)
        for kind, t in lin.tensors.items()
    }
    # data_format is unchanged (trivial, block_sizes == {1, 1}).
    lin = Linear(tensors=padded_tensors,
                 weight_format=lin.weight_format,
                 data_format=lin.data_format)
    root._commit_linear('output', lin,
                        split_side=SplitSide.OUTPUT,
                        tp=tp, ranks=self._attn_ranks)
```

- [ ] **Step 5.4: Normalize `_embed_key` / `_norm_key` to prefixes**

In `spec.py::_parse_base` (line 118), change:

```python
self._layer_prefix, self._embed_key, self._norm_key = \
    detect_layer_prefix(None, cfg)
```

— inspect `detect_layer_prefix` (in `source_model/utils.py`) and update it to return prefixes without the `.weight` suffix. Every consumer:

- `token_embeds(root, self._embed_key)` → now needs to pass `self._embed_key + '.weight'` or accept a prefix. Pick one convention and stick.

For minimal churn: keep `_embed_key`/`_norm_key` as full keys (ending in `.weight`) but have `token_embeds` strip the suffix when needed and `lm_head` accept full-key input too. Update `lm_head` call sites to pass the full key:

In each spec's `model()`:

```python
lm_key_full = self._embed_key if self._tie_embeddings else 'lm_head.weight'
lm_key = lm_key_full[:-len('.weight')]    # strip suffix for self._linear prefix
self.lm_head(root, lm_key)
```

Or just: `self.lm_head(root, 'model.embed_tokens' if self._tie_embeddings else 'lm_head')` — hard-code the pattern shared across all specs.

- [ ] **Step 5.5: Update every spec's `model()` for lm_head**

In each of `qwen3_spec.py`, `qwen3_5_spec.py`, `glm4_moe_lite_spec.py`, `gpt_oss_spec.py`, replace:

```python
lm_key = self._embed_key if self._tie_embeddings else 'lm_head.weight'
root.output = self.lm_head(lm_key)
```

with:

```python
lm_prefix = 'model.embed_tokens' if self._tie_embeddings else 'lm_head'
self.lm_head(root, lm_prefix)
```

(Assumes `model.embed_tokens` is the tied-embedding prefix; adjust per-spec if a particular arch differs.)

- [ ] **Step 5.6: Build and verify**

```
cd /data/lmdeploy-modeling/build && ninja
```

Run:
- trivial dense, tp=1 and tp=2
- AWQ, tp=1 and tp=2
- one tied-embeddings model, tp=1

Expected: coherent output in all cases. lm_head now flows through `self._linear` → `build_linear` → Linear-bundle with quantized or trivial data_format → `root._commit_linear` — the same machinery every other linear uses.

- [ ] **Step 5.7: Commit**

```bash
cd /data/lmdeploy-modeling && git add -A && git commit -m "refactor(deploy): delete LinearBuilder, unify lm_head via self._linear

LinearBuilder existed only as scaffolding for the _commit_tensor-on-
LinearWeight bypass. In the unified world it serves no purpose.

spec.py's lm_head becomes a direct call: self._linear(key) returns a
Linear bundle (with automatic quant detection, trivial normalization,
transpose to TM layout); we pad along the vocab dim to vocab_size_padded;
root._commit_linear commits it as the 'output' child.

Tied embeddings: caller passes 'model.embed_tokens' prefix; self._linear
reads the same tensor that tok_embeddings already read and commits it
independently (transposed, padded, sharded along vocab)."
```

---

## Task 6: Linear.data_format mandatory (hardened); delete preprocess_linear; thread data_format through fusion helpers

**Goal:** Task 3 made `data_format` mandatory on `Linear`. This task sweeps any remaining call sites that still need updating, deletes `preprocess_linear`, and switches `fmt.block_out` readers to `linear.data_format.block_sizes[...]`.

**Files:**
- Modify: `lmdeploy/turbomind/deploy/linear.py` (delete preprocess_linear)
- Modify: `lmdeploy/turbomind/deploy/builder/ffn.py` (_can_fuse_w1w3 reader)
- Modify: `lmdeploy/turbomind/deploy/builder/deltanet.py` (split_qkv reader)

- [ ] **Step 6.1: Delete `preprocess_linear`**

In `lmdeploy/turbomind/deploy/linear.py`, remove the `preprocess_linear` function (line 199). Also remove any imports of it in other files — search with:

```
cd /data/lmdeploy-modeling && rg -l "preprocess_linear"
```

Expected: only `linear.py` references it (self-mention) after removal. If any other file imports it, delete those imports too.

- [ ] **Step 6.2: `ffn.py::_can_fuse_w1w3` reads `data_format.block_sizes`**

In `lmdeploy/turbomind/deploy/builder/ffn.py`, replace `_can_fuse_w1w3` (line 52):

```python
def _can_fuse_w1w3(w1, tp: int) -> bool:
    """Check whether w1+w3 fusion is safe for the given TP.

    For block-quantized formats along the output (N) dim, the fused scale
    count 2 * cdiv(N/tp, block_out) must equal cdiv(2*N/tp, block_out).
    This holds iff (N/tp) % block_out == 0.
    """
    if tp <= 1:
        return True
    # block_sizes[1] is the block size along the output dim of the weight
    # in TM [in, out] layout.
    block_out = w1.data_format.block_sizes[1]
    if block_out <= 1:
        return True
    w = w1.tensors.get("weight")
    if w is None:
        return True
    return (w.size(-1) // tp) % block_out == 0
```

- [ ] **Step 6.3: `deltanet.py::split_qkv` reads `data_format.block_sizes`**

In `lmdeploy/turbomind/deploy/builder/deltanet.py::split_qkv` (line 23), replace:

```python
wfmt = linear.weight_format
block_out = (wfmt.block_out or 0) if wfmt is not None else 0
```

with:

```python
# block_sizes[1] = block_out for a weight tensor in TM [in, out] layout.
block_out = linear.data_format.block_sizes[1] if linear.data_format else 1
```

Ensure the rest of `split_qkv` continues to use `block_out` as-is.

- [ ] **Step 6.4: Grep for any remaining `fmt.block_in` / `fmt.block_out` readers**

```
cd /data/lmdeploy-modeling && rg "\.block_(in|out)" lmdeploy/turbomind/deploy -n
```

Expected matches after 6.2 / 6.3: only references inside `_base.py`'s `_commit_linear` (if any remain — check) and inside `converter.py`'s `_parse_quant_blocks` / `BlockSize`. Any other hit is a leftover reader; switch to `linear.data_format.block_sizes[...]`.

- [ ] **Step 6.5: Build and verify**

```
cd /data/lmdeploy-modeling/build && ninja   # Python-only changes in this task, but rebuild to be safe
```

Run the full matrix from Task 3h.2 at tp=1:
- trivial dense, AWQ, FP8, MXFP4, DeltaNet, MLA.

Expected: coherent output in all cases.

- [ ] **Step 6.6: Commit**

```bash
cd /data/lmdeploy-modeling && git add -A && git commit -m "refactor(deploy): delete preprocess_linear; thread data_format through fusion helpers

preprocess_linear was unreachable — its only call path (the dead deferred
data_format attach in _commit_linear) was removed by earlier cleanup.

Fusion helpers that today read the mutable WeightFormat.block_out (now
gone) switch to reading linear.data_format.block_sizes[1] — the authoritative
per-linear block size along the output dim, stored in TM weight-shape
order. Covers _can_fuse_w1w3 (ffn) and split_qkv (deltanet)."
```

---

## Self-review checklist (run after all tasks written)

After writing the plan, verify:

1. **Spec coverage:**
   - §1 C++ LinearWeight lifecycle → Tasks 1, 2, 3 (construct, configure, pre-alloc, prepare, grouped derivation, transpose block-swap).
   - §2 Python WeightFormat factory → Task 3 (f.1, f.2).
   - §3 Unified commit path → Tasks 3 (f.6, f.7, f.8) + 4 (tok_embeddings) + 5 (lm_head).
   - §4 Linear dataclass finalization → Tasks 3 (f.9) + 6.
   - §5 Commit plan → each task ends with a commit matching the planned commit message.
   - §6 Verification plan → each task includes the appropriate subset of the model matrix.
   - §7 Summary — cross-checked against what each task deletes/adds.

2. **Placeholder scan:** search the plan for red flags:
   ```
   rg -n "TBD|TODO|fill in|add appropriate|handle edge cases" docs/superpowers/plans/2026-04-21-linear-lifecycle-rethink.md
   ```
   Expected: no matches.

3. **Type consistency:** Method and kwarg names must match across tasks. Key names to double-check:
   - `_commit_linear(..., *, tp, ranks, epilogue)` — same signature in Tasks 3, 4, 5.
   - `_commit_tensor(..., *, tp, ranks)` — same signature in Tasks 3, 4.
   - `_make_linear_config_for(self, linear, split_side, tp, *, epilogue)` — Task 3 only.
   - `BlockSize(block_in, block_out)` — same throughout.
   - `WeightFormat.make_data_format(*, compute_dtype, block_in, block_out)` — Task 3, called from Task 5's `lm_head` indirectly.

If any inconsistency surfaces during implementation, fix in-place and note in commit message.

---

## Execution handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-21-linear-lifecycle-rethink.md`.** Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session via the executing-plans skill, batch execution with checkpoints.

Which approach?
