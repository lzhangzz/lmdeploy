# Linear Dtype Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Centralize TurboMind's linear dtype derivation logic into a single `ResolveDtypes()` function, making the relationship between model default dtype, weight format, and SM version explicit and self-documenting.

**Architecture:** Introduce a `LinearDtypes` struct (derived dtypes + quant descriptors) and a `ResolveDtypes()` function that replaces scattered if-else logic in `do_allocate()`. Rename `weight_type` → `weight_format` and replace mutable `input_type` with a derived `input_dtype()` accessor. All existing behavior is preserved — this is a pure refactoring.

**Tech Stack:** C++17, CUDA, TurboMind build system (ninja)

---

### Task 1: Add `LinearDtypes` struct and `ResolveDtypes()` function

**Files:**
- Modify: `src/turbomind/models/linear_weight.h`
- Modify: `src/turbomind/models/linear_weight.cc`

This is the core of the change — the new dtype derivation function. We add it first without changing the existing fields, so the code compiles but doesn't use it yet.

- [ ] **Step 1: Add `LinearDtypes` struct to `linear_weight.h`**

Add before the `LinearWeight` class definition (after the `using` statements, around line 12):

```cpp
struct LinearDtypes {
    DataType input_dtype{};
    DataType output_dtype{};
    DataType scale_dtype{};

    gemm::QuantDesc input_quant{};
    gemm::QuantDesc weight_quant{};
};
```

Also add the forward declaration of `ResolveDtypes`:

```cpp
LinearDtypes ResolveDtypes(DataType data_type, DataType weight_format, int group_size, int sm);
```

- [ ] **Step 2: Implement `ResolveDtypes()` in `linear_weight.cc`**

Add after the existing `IsDenseFloatType` helper (around line 20). This function encodes the full dtype policy matrix from the spec:

```cpp
LinearDtypes ResolveDtypes(DataType data_type, DataType weight_format, int group_size, int sm)
{
    LinearDtypes r;
    r.output_dtype = data_type;
    r.input_dtype  = data_type;
    r.scale_dtype  = data_type;

    const bool is_qweight = weight_format == kUint4 || weight_format == kUint8;

    if (IsDenseFloatType(weight_format)) {
        // Dense FP16/BF16/FP32 — no quantization descriptors
        return r;
    }

    if (weight_format == kFloat8_e4m3) {
        TM_CHECK_EQ(group_size, 128)
            << "FP8 weight format requires group_size=128, got " << group_size;
        r.weight_quant = QuantDesc{gemm::QuantType::kB, group_size};
        if (sm == 90) {
            r.input_dtype = kFloat8_e4m3;
            r.input_quant = QuantDesc{gemm::QuantType::kK, group_size};
            r.scale_dtype = kFloat;
        }
        return r;
    }

    if (weight_format == kFloat4_e2m1) {
        r.scale_dtype  = kUint8;
        r.weight_quant = QuantDesc{gemm::QuantType::kK, group_size};
        return r;
    }

    if (is_qweight) {
        TM_CHECK(group_size > 0 && group_size <= 256)
            << "Invalid group_size for quantized weight: " << group_size;
        r.weight_quant = QuantDesc{gemm::QuantType::kK, group_size};
        return r;
    }

    TM_CHECK(0) << "Unsupported weight format: " << to_string(weight_format);
    return r;
}
```

- [ ] **Step 3: Build to verify compilation**

Run: `cd /data/lmdeploy-modeling/build && ninja`

Expected: Clean build, no errors. The new function is defined but not yet called.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/linear_weight.h src/turbomind/models/linear_weight.cc
git commit -m "refactor(linear): add LinearDtypes struct and ResolveDtypes() function"
```

---

### Task 2: Restructure `LinearWeight` fields

**Files:**
- Modify: `src/turbomind/models/linear_weight.h`
- Modify: `src/turbomind/models/linear_weight.cc`

Now we restructure the fields: rename `weight_type` → `weight_format`, replace `input_type` with derived state, and add accessors. We update `configure()` and `do_allocate()` to use `ResolveDtypes()`.

- [ ] **Step 1: Update `linear_weight.h` class definition**

Replace the public dtype fields (lines 47-51) with:

```cpp
    // --- Input (immutable after setter) ---
    DataType data_type{};       // model-scope default compute dtype, set in configure()
    DataType weight_format{};   // checkpoint weight storage format, set in do_allocate()

    // --- Derived (computed once in do_allocate via ResolveDtypes) ---
    LinearDtypes resolved_{};

    DataType input_dtype() const  { return resolved_.input_dtype; }
    DataType output_dtype() const { return resolved_.output_dtype; }

    // --- Public fields consumed by execution layers ---
    int  input_dim  = 0;
    int  output_dim = 0;
    int  group_size = 0;

    Epilogue    epilogue{};

    MatrixLayout k_desc{};
    MatrixLayout q_desc{};
```

Remove the old fields: `weight_type`, `input_type`, `weight_quant`, `input_quant`.

Keep `weight`, `bias`, `scales`, `zeros` tensors as-is.

- [ ] **Step 2: Update `configure()` in `linear_weight.cc`**

```cpp
void LinearWeight::configure(int input_dim, int output_dim, DataType data_type, bool has_bias)
{
    this->data_type = data_type;
    this->input_dim = input_dim;
    this->output_dim = output_dim;
    has_bias_        = has_bias;
    // weight_format and resolved_ are set in do_allocate()
}
```

Note: no longer sets `input_type` or `weight_type` here — those are derived later.

- [ ] **Step 3: Update `do_allocate()` in `linear_weight.cc`**

```cpp
void LinearWeight::do_allocate(DataType actual_weight_type, int actual_group_size)
{
    weight_format = actual_weight_type;
    group_size    = actual_group_size;
    resolved_     = ResolveDtypes(data_type, actual_weight_type, actual_group_size, getSMVersion());

    weight = Tensor({input_dim, output_dim}, actual_weight_type, kDEVICE);
    add_param("weight", weight);

    if (has_bias_) {
        bias = Tensor{{output_dim}, data_type, kDEVICE};
        add_param("bias", bias);
    }

    scales = {};
    zeros  = {};

    if (actual_weight_type == kFloat8_e4m3) {
        scales = Tensor{{cdiv(input_dim, actual_group_size), cdiv(output_dim, actual_group_size)},
                        resolved_.scale_dtype, kDEVICE};
        add_param("scales", scales);
    }
    else if (actual_weight_type == kFloat4_e2m1) {
        scales = Tensor{{cdiv(input_dim, actual_group_size), output_dim}, kUint8, kDEVICE};
        add_param("scales", scales);
    }
    else if (actual_weight_type == kUint4 || actual_weight_type == kUint8) {
        TM_CHECK(input_dim % actual_group_size == 0) << input_dim << " " << actual_group_size;
        scales = Tensor{{input_dim / actual_group_size, output_dim}, data_type, kDEVICE};
        zeros  = Tensor{{input_dim / actual_group_size, output_dim}, data_type, kDEVICE};
        add_param("scales", scales);
        add_param("zeros", zeros);
    }

    k_desc = {};
    q_desc = {};

    k_desc.type  = weight.dtype();
    k_desc.order = gemm::kRowMajor;
    k_desc.rows  = input_dim;
    k_desc.cols  = output_dim;
    k_desc.ld    = output_dim;
}
```

- [ ] **Step 4: Update `prepare()` in `linear_weight.cc`**

In `prepare()`, replace all references to old fields with new accessors:

- `weight_type` → `weight_format`
- `input_type` → `input_dtype()` (or `resolved_.input_dtype`)
- `data_type` in contexts that mean output → `output_dtype()` (or just keep `data_type` since it's the same)
- `weight_quant` → `resolved_.weight_quant`
- `input_quant` → `resolved_.input_quant`

The full `prepare()` method body (lines 159-312) needs these substitutions. The logic flow stays identical. Specifically:

Line 167: `if (weight_type == kFloat8_e4m3 && input_type == kFloat8_e4m3)` →
`if (weight_format == kFloat8_e4m3 && input_dtype() == kFloat8_e4m3)`

Line 183: `else if (weight_type == kFloat8_e4m3)` →
`else if (weight_format == kFloat8_e4m3)`

Line 191: `GetConverters(data_type, weight_type, input_type, ...)` →
`GetConverters(data_type, weight_format, input_dtype(), ...)`

Line 198: `byte_size(weight_type, 8)` → `byte_size(weight_format, 8)`

Line 236-241: `kd.type = data_type_v<uint4_t>` etc. stays unchanged (these are about temp tensor dtypes)

Line 248: `kd.type = weight_type` → `kd.type = weight_format`

Line 282: `if (data_type == kHalf && weight_type == kFloat4_e2m1)` →
`if (data_type == kHalf && weight_format == kFloat4_e2m1)`

Line 289: the `scales` and `zeros` references in `prepare()` use local variables, not fields, so no changes needed.

- [ ] **Step 5: Update `alloc()` in `linear_weight.cc`**

In `alloc()` (lines 109-144), the `IsDenseFloatType` check references `data_type` — this stays as-is since we kept the `data_type` name. No changes needed in `alloc()`.

- [ ] **Step 6: Update `LinearWeightRegistrar` at bottom of `linear_weight.cc`**

The registrar (lines 314-331) calls `configure(...)` — no change needed since `configure()` signature is unchanged.

- [ ] **Step 7: Build to verify compilation**

Run: `cd /data/lmdeploy-modeling/build && ninja`

Expected: Build will fail on consumer files (`LlamaLinear.cu`, `moe_weight.cc`, `testbed_v3.h`) because they reference the removed fields. This is expected — we fix them in Task 3.

- [ ] **Step 8: Commit**

```bash
git add src/turbomind/models/linear_weight.h src/turbomind/models/linear_weight.cc
git commit -m "refactor(linear): restructure LinearWeight fields, use ResolveDtypes()"
```

---

### Task 3: Update consumers

**Files:**
- Modify: `src/turbomind/models/llama/LlamaLinear.cu`
- Modify: `src/turbomind/models/moe_weight.cc`
- Modify: `src/turbomind/kernels/gemm/test/testbed_v3.h` (may be broken from prior refactoring — if so, skip)

Update all external consumers of the old field names.

- [ ] **Step 1: Update `LlamaLinear.cu`**

Four substitutions:

Line 76: `dense.input_type` → `dense.input_dtype()`
```cpp
        if (input.dtype() != dense.input_dtype()) {
```

Line 125: `dense.input_quant` → `dense.resolved_.input_quant`
```cpp
        op.quant_a   = dense.resolved_.input_quant;
```

Line 126: `dense.weight_quant` → `dense.resolved_.weight_quant`
```cpp
        op.quant_b   = dense.resolved_.weight_quant;
```

Line 135: `dense.data_type` → `dense.output_dtype()`
```cpp
            D       = Tensor{{desc_A.rows, dim}, dense.output_dtype(), kDEVICE};
```

- [ ] **Step 2: Update `moe_weight.cc` `LinkLinearExperts()`**

Lines 55-59: Copy the new fields instead of old ones:
```cpp
    d.data_type     = e0.data_type;
    d.weight_format = e0.weight_format;
    d.resolved_     = e0.resolved_;
```

Remove the old lines that copied `input_type`, `weight_type`, `input_quant`, `weight_quant`.

Line 86: `d.weight_type == kFloat8_e4m3 && d.input_type == kFloat8_e4m3` →
`d.weight_format == kFloat8_e4m3 && d.input_dtype() == kFloat8_e4m3`

Line 98: `d.weight_type` → `d.weight_format`
```cpp
        d.weight = Tensor{make_strided_ptr(weights), {n}, d.weight_format, kDEVICE};
```

- [ ] **Step 3: Update `testbed_v3.h` `LinkExperts()` (skip if file is broken from prior refactoring)**

Lines 88-92: Same pattern as `LinkLinearExperts()`:
```cpp
    d.data_type     = e0.data_type;
    d.weight_format = e0.weight_format;
    d.resolved_     = e0.resolved_;
```

Remove old lines for `input_type`, `weight_type`, `weight_quant`, `input_quant`.

Line 119: same FP8 check update:
`d.weight_type == kFloat8_e4m3 && d.input_type == kFloat8_e4m3` →
`d.weight_format == kFloat8_e4m3 && d.input_dtype() == kFloat8_e4m3`

Line 131: `d.weight_type` → `d.weight_format`

Also update `GenerateWeight()` (line 316): `weight_type == data_type` → `weight_type == data_type`. Wait — this file has its own local `Parameter` struct with `weight_type` and `input_type` fields. Those are independent of `LinearWeight`'s fields. **Do NOT change the `Parameter` struct.** Only change the `LinkExperts()` function which accesses `LinearWeight` fields.

- [ ] **Step 4: Build to verify compilation**

Run: `cd /data/lmdeploy-modeling/build && ninja`

Expected: Clean build, no errors.

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/models/llama/LlamaLinear.cu src/turbomind/models/moe_weight.cc src/turbomind/kernels/gemm/test/testbed_v3.h
git commit -m "refactor(linear): update consumers to use new LinearWeight accessors"
```

---

### Task 4: Build and verify

**Files:** None (verification only)

- [ ] **Step 1: Full clean build**

Run: `cd /data/lmdeploy-modeling/build && ninja clean && ninja`

Expected: Clean build.

- [ ] **Step 2: Run model inference test via turbomind-tester agent**

Dispatch the `turbomind-tester` agent to test one or more models. The agent will:
- Check GPU availability
- Run model inference
- Verify output is meaningful (not gibberish) and at least 128 tokens

This confirms the refactoring preserved all runtime behavior.

- [ ] **Step 3: Commit any fixups if needed**

If the build or tests revealed issues, fix and commit with:
```bash
git commit -m "fix(linear): address build/test issues from dtype policy refactoring"
```
