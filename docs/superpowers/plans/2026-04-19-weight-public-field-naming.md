# Weight public field naming — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the trailing underscore from every **public** data member on `MoeWeight`, `AttentionWeight`, `DeltaNetWeight`, and `LinearWeight`, and update all C++ references so the tree builds cleanly.

**Architecture:** Pure identifier rename driven by the mapping tables in `docs/superpowers/specs/2026-04-19-weight-public-field-naming-design.md`. Private members (suffix `_` retained) and unrelated types (e.g. `FfnWeight` privates, `ModelRequest::hidden_dim_`) are untouched. One logical component per commit keeps review small.

**Tech stack:** C++ (TurboMind under `src/turbomind/`), CMake build from repo root.

**Spec:** `docs/superpowers/specs/2026-04-19-weight-public-field-naming-design.md` (approved).

---

## File map (all modified, none created)

| File | Role |
|------|------|
| `src/turbomind/models/moe_weight.h` | Public field declarations + `num_experts()` body |
| `src/turbomind/models/moe_weight.cc` | ctor + `prepare()` body uses public + private fields |
| `src/turbomind/models/llama/moe_ffn_layer.cc` | Reads `MoeWeight` public fields from `p.weights` |
| `src/turbomind/models/attention_weight.h` | Public fields + `is_mla()` |
| `src/turbomind/models/attention_weight.cc` | ctor initializer list |
| `src/turbomind/models/llama/unified_attention_layer.cc` | `AttentionWeight& weights` / pointer field access |
| `src/turbomind/models/model_weight.cc` | `attn_layer->attention->…` |
| `src/turbomind/models/delta_net_weight.h` | Public field declarations |
| `src/turbomind/models/delta_net_weight.cc` | ctor + `prepare()` |
| `src/turbomind/engine/engine.cc` | reads `DeltaNetWeight*` fields |
| `src/turbomind/models/llama/GatedDeltaNetLayer.cc` | reads `DeltaNetWeight` fields |
| `src/turbomind/models/linear_weight.h` | `format` / `policy` + `input_dtype()` / `output_dtype()` |
| `src/turbomind/models/linear_weight.cc` | `configure`, `copy_metadata_to`, `set_weight_spec` |
| `src/turbomind/models/llama/LlamaLinear.cu` | `weight.policy_.…` → `weight.policy.…` |

---

### Task 1: `MoeWeight` public fields

**Files:**

- Modify: `src/turbomind/models/moe_weight.h`
- Modify: `src/turbomind/models/moe_weight.cc`
- Modify: `src/turbomind/models/llama/moe_ffn_layer.cc`

- [ ] **Step 1: Update `moe_weight.h`**

In the public section, rename members per spec (example — apply the same pattern to every row in the MoeWeight table in the spec):

```cpp
    int num_experts() const { return expert_num; }

    // --- Config fields (public for runtime access) ---
    int  hidden_dim{};
    int  inter_size{};
    int  experts_per_token{};
    bool norm_topk_prob{};
    // ... etc; expert_num at end without trailing underscore
```

Leave the `private:` block unchanged (`layer_id_`, `method_`, `mlp_bias_`, `data_type_`, `tp_size_`, `tp_rank_`, `act_type_`, `fuse_silu_act_`, `block_`). `expert_num` lives only in the public section (it is not a private field today).

- [ ] **Step 2: Update `moe_weight.cc`**

Replace every assignment and read of the old public names with the new names, for example:

```cpp
    experts_per_token = cfg.experts_per_token;
    norm_topk_prob = cfg.norm_topk_prob;
    // ...
    expert_num = cfg.expert_num;
```

In `prepare()`, use `hidden_dim`, `inter_size`, `expert_num` where the public fields are read. Keep `tp_size_`, `mlp_bias_`, `block_`, etc. as-is.

- [ ] **Step 3: Update `moe_ffn_layer.cc`**

Replace `p.weights->experts_per_token_` with `p.weights->experts_per_token`, and the same for `hidden_dim_`, `inter_size_`, `topk_method_`, `n_group_`, `topk_group_`, `norm_topk_prob_`, `routed_scale_`, `scoring_func_` (compare becomes `p.weights->scoring_func == "sigmoid"`).

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/moe_weight.h src/turbomind/models/moe_weight.cc \
  src/turbomind/models/llama/moe_ffn_layer.cc
git commit -m "refactor(weights): drop underscore from MoeWeight public fields"
```

---

### Task 2: `AttentionWeight` public fields

**Files:**

- Modify: `src/turbomind/models/attention_weight.h`
- Modify: `src/turbomind/models/attention_weight.cc`
- Modify: `src/turbomind/models/llama/unified_attention_layer.cc`
- Modify: `src/turbomind/models/model_weight.cc`

- [ ] **Step 1: Update `attention_weight.h`**

```cpp
    bool is_mla() const { return kv_lora_rank > 0; }

    // --- Config fields (public for runtime access) ---
    int      hidden_dim{};
    int      head_dim{};
    // ... through core::RopeConfig rope{};
```

- [ ] **Step 2: Update `attention_weight.cc` initializer list**

Use `hidden_dim(cfg.hidden_dim)`, `head_dim(cfg.head_dim)`, … `rope(cfg.rope)` — no trailing underscores on these members.

- [ ] **Step 3: Update `unified_attention_layer.cc`**

- `attn_weights[0]->rope_` → `attn_weights[0]->rope`
- Every `weights.qk_norm_`, `weights.attn_output_gate_`, `weights.tp_size_`, `weights.head_num_`, `weights.kv_head_num_`, `weights.head_dim_`, `weights.softmax_scale_`, `weights.window_size_`, `weights.use_logn_attn_`, `weights.rope_.` → same identifier without trailing underscore (and `weights.rope.` for nested access).

- [ ] **Step 4: Update `model_weight.cc`**

```cpp
    data_type_    = attn_layer->attention->data_type;
    hidden_units_ = attn_layer->attention->hidden_dim;
    head_dim_     = attn_layer->attention->head_dim;
    kv_head_num_  = attn_layer->attention->kv_head_num;
```

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/models/attention_weight.h \
  src/turbomind/models/attention_weight.cc \
  src/turbomind/models/llama/unified_attention_layer.cc \
  src/turbomind/models/model_weight.cc
git commit -m "refactor(weights): drop underscore from AttentionWeight public fields"
```

---

### Task 3: `DeltaNetWeight` public fields

**Files:**

- Modify: `src/turbomind/models/delta_net_weight.h`
- Modify: `src/turbomind/models/delta_net_weight.cc`
- Modify: `src/turbomind/engine/engine.cc`
- Modify: `src/turbomind/models/llama/GatedDeltaNetLayer.cc`

- [ ] **Step 1: Update `delta_net_weight.h`** — public block: `hidden_dim`, `num_k_heads`, … `data_type` (no suffix).

- [ ] **Step 2: Update `delta_net_weight.cc`**

Ctor initializer list and `prepare()`:

```cpp
    EnsureFloatDtype(A_log, data_type);
    EnsureFloatDtype(dt_bias, data_type);
    EnsureFloatDtype(conv1d, data_type);
```

- [ ] **Step 3: Update `engine.cc`** — replace `dn->key_head_dim_` with `dn->key_head_dim`, and similarly for `value_head_dim_`, `d_conv_`, `num_k_heads_`, `num_v_heads_`.

- [ ] **Step 4: Update `GatedDeltaNetLayer.cc`** — replace `w.num_k_heads_`, `w.num_v_heads_`, `w.key_head_dim_`, `w.value_head_dim_`, `w.d_conv_` with suffix-free names.

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/models/delta_net_weight.h \
  src/turbomind/models/delta_net_weight.cc \
  src/turbomind/engine/engine.cc \
  src/turbomind/models/llama/GatedDeltaNetLayer.cc
git commit -m "refactor(weights): drop underscore from DeltaNetWeight public fields"
```

---

### Task 4: `LinearWeight` `format` / `policy`

**Files:**

- Modify: `src/turbomind/models/linear_weight.h`
- Modify: `src/turbomind/models/linear_weight.cc`
- Modify: `src/turbomind/models/llama/LlamaLinear.cu`

- [ ] **Step 1: Update `linear_weight.h`**

```cpp
    DataFormat   format{};
    LinearPolicy policy{};

    DataType input_dtype() const  { return policy.input_dtype; }
    DataType output_dtype() const { return policy.output_dtype; }
```

Do **not** rename `has_bias_` / `is_grouped_` (private).

- [ ] **Step 2: Update `linear_weight.cc`**

In `configure()`:

```cpp
    policy.input_dtype  = data_type;
    policy.output_dtype = data_type;
```

In `copy_metadata_to()`:

```cpp
    dst.format = format;
    dst.policy = policy;
```

In `set_weight_spec()`:

```cpp
    format = MakeLinearWeightFormat(data_type, weight_format, group_size);
    policy = ResolveLinearPolicy(format, data_type, getSMVersion());
```

- [ ] **Step 3: Update `LlamaLinear.cu`**

```cpp
        op.quant_a   = weight.policy.input_quant;
        op.quant_b   = weight.policy.weight_quant;
```

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/linear_weight.h \
  src/turbomind/models/linear_weight.cc \
  src/turbomind/models/llama/LlamaLinear.cu
git commit -m "refactor(weights): rename LinearWeight format_/policy_ to format/policy"
```

---

### Task 5: Verify build and stray references

**Files:** none (read-only checks)

- [ ] **Step 1: Search for stale **public** patterns under `src/turbomind`**

Run (from repo root):

```bash
rg 'p\.weights->[a-z_]+_' src/turbomind/models/llama/moe_ffn_layer.cc || true
rg 'attn_weights\[0\]->rope_' src/turbomind || true
rg '\b(policy_|format_)\b' src/turbomind/models/linear_weight.h src/turbomind/models/linear_weight.cc src/turbomind/models/llama/LlamaLinear.cu || true
```

Expected: no matches (empty output for each `rg`).

Note: `rg 'hidden_dim_' src/turbomind` may still match **private** fields on `FfnWeight`, `ModelRequest`, etc.; that is expected. Compiler is the source of truth.

- [ ] **Step 2: Configure and build**

If no build directory yet:

```bash
cmake -S /data/lmdeploy-modeling -B /data/lmdeploy-modeling/build -DCMAKE_BUILD_TYPE=Release
```

Build:

```bash
cmake --build /data/lmdeploy-modeling/build -j"$(nproc)"
```

Expected: exit code `0`, no compile errors in the files above.

- [ ] **Step 3: Optional tests**

If the project routinely runs C++ tests:

```bash
ctest --test-dir /data/lmdeploy-modeling/build --output-on-failure
```

Expected: all tests pass (or skip if no `ctest` targets configured).

- [ ] **Step 4: Final commit** — only if Step 5 produced doc or fixups; otherwise Task 4 is the last commit.

---

## Spec coverage (self-review)

| Spec section | Task |
|--------------|------|
| MoeWeight mapping | Task 1 |
| AttentionWeight mapping | Task 2 |
| DeltaNetWeight mapping | Task 3 |
| LinearWeight `format` / `policy` | Task 4 |
| Validation / out of scope | Task 5; privates untouched in Tasks 1–4 |

## Placeholder scan

No TBD/TODO/similar-to-Task-only steps; each task names concrete files and identifier patterns.

## Type/name consistency

Public names after work: `expert_num`, `kv_lora_rank`, `rope`, `format`, `policy`, etc., match the spec tables and are used consistently in accessors (`num_experts()`, `is_mla()`, `input_dtype()`).

---

**Plan complete and saved to `docs/superpowers/plans/2026-04-19-weight-public-field-naming.md`. Two execution options:**

**1. Subagent-driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline execution** — run tasks in this session with executing-plans-style checkpoints.

Which approach do you want?
