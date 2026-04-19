# Weight public field naming — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the trailing underscore from every **public** data member on `MoeWeight`, `AttentionWeight`, `DeltaNetWeight`, `LinearWeight`, and **`ModelWeight`** (see spec errata), and update all C++ references so the tree builds cleanly.

**Architecture:** Pure identifier rename. Private members keep the `_` suffix. Unrelated types (`FfnWeight` / `NormWeight` privates, `ModelRequest`, attention reference kernels, etc.) are untouched. One logical component per commit keeps review small. **Order:** complete Task 2 before Task 6 — `model_weight.cc` reads `AttentionWeight` on the RHS in `prepare()`; Task 6 then renames `ModelWeight`’s own fields (LHS and all other uses).

**Code review notes (harder pass):**

- `unified_attention_layer.cc` uses **`w.`** as well as **`weights.`** for the same public `AttentionWeight` fields (`tp_size_`, `head_num_`, `head_dim_`, `kv_head_num_`, `data_type_`, `bias_`, etc.). Every occurrence must be updated, not only the `weights.` lines called out in the first draft.
- `engine.cc` `CreateSequenceManager()` uses **`weights_.head_dim_`**, **`weights_.kv_head_num_`**, **`weights_.num_layer_`**, **`weights_.data_type_`** — those are **`ModelWeight`** members, not `AttentionWeight`. They were missing from the original file map.
- `ModelWeight` (`model_weight.h` lines 51–64) exposes many **public** `*_` fields; the original spec incorrectly labeled all of `ModelWeight` as out of scope. The design doc now has an **Errata** section; this plan adds **Task 6**.
- `src/turbomind/python/bind.cpp` exposes `LinearWeight` methods only; no `format_` / `policy_` bindings — no Python change expected.
- C++ tests under `tests/csrc` did not reference these weight types in a quick scan; still run the full build.

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
| `src/turbomind/models/model_weight.h` | **Task 6:** public derived field declarations |
| `src/turbomind/models/model_weight.cc` | **Task 2:** RHS in `prepare()` from `AttentionWeight`; **Task 6:** ctor + `prepare()` LHS and all `ModelWeight` field names |
| `src/turbomind/models/llama/unified_decoder.cc` | **Task 6:** `model_weight.*` public fields |
| `src/turbomind/models/language_model.cc` | **Task 6:** `weights_.*` |
| `src/turbomind/turbomind.cc` | **Task 6:** `weights_[i]->vocab_size_` / `hidden_units_` |
| `src/turbomind/engine/engine.cc` | **Task 3:** `dn->…` (`DeltaNetWeight*`); **Task 6:** `weights_.…` (`ModelWeight`) |
| `src/turbomind/models/delta_net_weight.h` | Public field declarations |
| `src/turbomind/models/delta_net_weight.cc` | ctor + `prepare()` |
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

- [ ] **Step 3: Update `unified_attention_layer.cc` (exhaustive)**

Grep inside this file for `weights.` and `w.` followed by a **public** `AttentionWeight` field and drop the suffix. Confirmed occurrences include:

| Pattern (before) | After |
|------------------|-------|
| `attn_weights[0]->rope_` | `attn_weights[0]->rope` |
| `w.tp_size_`, `weights.tp_size_` | `w.tp_size`, `weights.tp_size` |
| `w.head_num_`, `weights.head_num_` | `w.head_num`, `weights.head_num` |
| `w.head_dim_`, `weights.head_dim_` | `w.head_dim`, `weights.head_dim` |
| `w.kv_head_num_`, `weights.kv_head_num_` | `w.kv_head_num`, `weights.kv_head_num` |
| `w.data_type_` | `w.data_type` |
| `weights.qk_norm_` | `weights.qk_norm` |
| `weights.attn_output_gate_` | `weights.attn_output_gate` |
| `weights.softmax_scale_` | `weights.softmax_scale` |
| `weights.window_size_` | `weights.window_size` |
| `weights.use_logn_attn_` | `weights.use_logn_attn` |
| `weights.rope_.` | `weights.rope.` |
| `weights.bias_` | `weights.bias` |

Do **not** change unrelated `*_` names (e.g. `tmp_attn_`, `engine_param_`, `rope_base_buf_`, child modules like `w.kv_a_proj`).

- [ ] **Step 4: Update `model_weight.cc` (RHS only — AttentionWeight)**

In `prepare()`, only the **right-hand** side uses `AttentionWeight`; keep **`ModelWeight`’s current member names** on the LHS until Task 6:

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

*(There is no Task 5; numbering jumps to 6 so it stays aligned with the amended doc history.)*

---

### Task 6: `ModelWeight` public derived fields

**Depends on:** Task 2 (so `AttentionWeight` field names on the RHS of `prepare()` are already final).

**Files:**

- Modify: `src/turbomind/models/model_weight.h`
- Modify: `src/turbomind/models/model_weight.cc`
- Modify: `src/turbomind/models/llama/unified_decoder.cc`
- Modify: `src/turbomind/engine/engine.cc` (only `weights_.…` uses in `CreateSequenceManager` and any other `ModelWeight` field access in this file — **not** the `dn->…` DeltaNet block from Task 3)
- Modify: `src/turbomind/models/language_model.cc`
- Modify: `src/turbomind/turbomind.cc`

**Mapping**

| Current | New |
|---------|-----|
| `data_type_` | `data_type` |
| `hidden_units_` | `hidden_units` |
| `vocab_size_` | `vocab_size` |
| `vocab_size_padded_` | `vocab_size_padded` |
| `embedding_size_` | `embedding_size` |
| `num_layer_` | `num_layer` |
| `head_dim_` | `head_dim` |
| `kv_head_num_` | `kv_head_num` |
| `layer_types_` | `layer_types` |
| `tp_size_` | `tp_size` |
| `tp_rank_` | `tp_rank` |

**Do not** rename `GatedDeltaNetLayer`'s private `layer_types_` — different class, out of scope.

- [ ] **Step 1: Update `model_weight.h`** — public block uses the new names; **private** `stream_`, `alloca_`, `layers_cache_` unchanged.

- [ ] **Step 2: Update `model_weight.cc`**

Ctor initializer: `tp_size(...)`, `tp_rank(...)`. In `prepare()`, rename **both** sides to the new `ModelWeight` names, e.g.:

```cpp
    data_type    = attn_layer->attention->data_type;
    hidden_units = attn_layer->attention->hidden_dim;
    head_dim     = attn_layer->attention->head_dim;
    kv_head_num  = attn_layer->attention->kv_head_num;

    vocab_size        = tok_embeddings->weight.shape(0);
    embedding_size    = vocab_size;
    num_layer         = layers->size();
    vocab_size_padded = round_up((size_t)vocab_size, (size_t)tp_size);

    layer_types.resize(num_layer);
    for (int i = 0; i < num_layer; ++i) {
        layer_types[i] = layer(i)->linear_attn ? 1 : 0;
    }
```

- [ ] **Step 3: Update call sites** — `unified_decoder.cc` (`model_weight.num_layer_` → `num_layer`, `model_weight.layer_types_` → `model_weight.layer_types`, etc.), `engine.cc` (`weights_.layer_types_` → `weights_.layer_types`, …), `language_model.cc`, `turbomind.cc` (`weights_[i]->vocab_size_` → `vocab_size`, …). Re-grep for stale `model_weight\.\w+_` / `weights_\.\w+_` on **`ModelWeight`** fields until clean (ignore `Engine::Impl::tp_rank_` and other non-`ModelWeight` members).

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/models/model_weight.h src/turbomind/models/model_weight.cc \
  src/turbomind/models/llama/unified_decoder.cc src/turbomind/engine/engine.cc \
  src/turbomind/models/language_model.cc src/turbomind/turbomind.cc
git commit -m "refactor(weights): drop underscore from ModelWeight public fields"
```

---

### Task 7: Verify build and stray references

**Files:** none (read-only checks)

- [ ] **Step 1: Search for stale **public** patterns under `src/turbomind`**

Run (from repo root):

```bash
rg 'p\.weights->[a-z_]+_' src/turbomind/models/llama/moe_ffn_layer.cc || true
rg 'attn_weights\[0\]->rope_' src/turbomind || true
rg '\b(policy_|format_)\b' src/turbomind/models/linear_weight.h src/turbomind/models/linear_weight.cc src/turbomind/models/llama/LlamaLinear.cu || true
rg 'model_weight\.(data_type_|hidden_units_|vocab_size_|vocab_size_padded_|embedding_size_|num_layer_|head_dim_|kv_head_num_|layer_types_|tp_size_|tp_rank_)' src/turbomind || true
rg 'weights_\.(data_type_|hidden_units_|vocab_size_|vocab_size_padded_|embedding_size_|num_layer_|head_dim_|kv_head_num_|layer_types_|tp_size_|tp_rank_)' src/turbomind || true
rg 'weights_\[[0-9]+\]->(vocab_size_|hidden_units_)' src/turbomind/turbomind.cc || true
```

Expected: no matches (empty output for each `rg`).

Note: `rg 'hidden_dim_' src/turbomind` may still match **private** fields on `FfnWeight`, `ModelRequest`, etc.; that is expected. Compiler is the source of truth.

Note: `engine.cc` legitimately contains `tp_rank_` on **`Engine::Impl`’s private member**, not `ModelWeight` — do not “clean” those with a blind replace.

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

- [ ] **Step 4: Final commit** — only if verification surfaced fixups; otherwise Task 6 is the last feature commit.

---

## Spec coverage (self-review)

| Spec section | Task |
|--------------|------|
| MoeWeight mapping | Task 1 |
| AttentionWeight mapping | Task 2 |
| DeltaNetWeight mapping | Task 3 |
| LinearWeight `format` / `policy` | Task 4 |
| `ModelWeight` errata / public derived fields | Task 6 |
| Validation | Task 7; privates untouched in Tasks 1–4 and 6 |

## Placeholder scan

No TBD/TODO/similar-to-Task-only steps; each task names concrete files and identifier patterns.

## Type/name consistency

Public names after work: `expert_num`, `kv_lora_rank`, `rope`, `format`, `policy`, `ModelWeight::num_layer`, etc., match the spec / errata and are used consistently in accessors (`num_experts()`, `is_mla()`, `input_dtype()`).

---

**Plan updated** (deeper pass): `docs/superpowers/plans/2026-04-19-weight-public-field-naming.md`. **Execution options:**

**1. Subagent-driven (recommended)** — one subagent per task (Tasks 1–4, 6, then 7), review between tasks.

**2. Inline execution** — run tasks in this session with checkpoints; **respect task order** (especially Task 2 before Task 6).
