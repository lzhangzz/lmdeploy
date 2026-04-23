# Engine Config Copy Elimination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the field-to-field `EngineConfig` → `EngineParam` copy by making `EngineParam` inherit from `EngineConfig`, and remove the dead `step_length` field.

**Architecture:** `EngineParam` becomes a subclass of `EngineConfig` — all 24 config fields are inherited, and `EngineParam` adds only runtime-derived rank fields and `max_forward_token_num`. The copy in `turbomind.cc` collapses to a single cast-assignment.

**Tech Stack:** C++17, pybind11 (unchanged), x-macros (unchanged)

---

### Task 1: Make EngineParam inherit from EngineConfig

**Files:**
- Modify: `src/turbomind/models/llama/llama_params.h`

- [ ] **Step 1: Add include for EngineConfig and change EngineParam**

Add `#include "src/turbomind/engine/engine_config.h"` at the top (after existing includes).

Change `EngineParam` from a standalone struct to one that inherits `EngineConfig`, keeping only its own fields:

```cpp
#include "src/turbomind/engine/engine_config.h"

namespace turbomind {

struct EngineParam : EngineConfig {
    // Runtime-derived fields (set in CreateContext)
    int outer_dp_rank = 0;
    int attn_dp_rank = 0;
    int attn_tp_rank = 0;
    int attn_cp_rank = 0;
    int mlp_tp_rank = 0;

    // Derived field (set in Impl ctor)
    int max_forward_token_num = 0;
};

}  // namespace turbomind
```

Remove ALL the old duplicate fields from `EngineParam`:
- `max_batch_size`, `session_len`, `step_length`
- `quant_policy`, `tune_layer_num`
- `cache_max_block_count`, `cache_chunk_size`, `cache_block_seq_len`
- `enable_prefix_caching`, `enable_metrics`
- `max_forward_token_num` (moved to own section above with `= 0`)
- `max_context_token_num`, `num_tokens_per_iter`, `max_prefill_iters`
- `outer_dp_size`, `outer_dp_rank` (moved), `attn_dp_size`, `attn_dp_rank` (moved)
- `attn_tp_size`, `attn_tp_rank` (moved), `attn_cp_size`, `attn_cp_rank` (moved)
- `mlp_tp_size`, `mlp_tp_rank` (moved)
- `nnodes`, `node_rank`, `devices`

- [ ] **Step 2: Build to verify compilation**

```bash
cd build && ninja 2>&1 | head -50
```

Expected: Clean build or only unrelated warnings. If there are compilation errors, they will likely be from files that include `llama_params.h` but not `engine_config.h` transitively — fix by ensuring `llama_params.h` includes `engine_config.h` (done in Step 1).

- [ ] **Step 3: Commit**

```bash
git add src/turbomind/models/llama/llama_params.h
git commit -m "refactor: make EngineParam inherit EngineConfig, remove duplicate fields and dead step_length"
```

---

### Task 2: Eliminate the field-to-field copy in turbomind.cc

**Files:**
- Modify: `src/turbomind/turbomind.cc`

- [ ] **Step 1: Replace the copy block in Impl::Impl**

Current code (lines 155-198) — the entire block from `engine_param_.cache_block_seq_len = config.cache_block_seq_len;` through `communicator_type_ = std::move(config.communicator);` — is replaced by:

```cpp
static_cast<EngineConfig&>(engine_param_) = config;
```

The `Impl` constructor initializer list stays the same. The lines before and after the copy block that handle derived values remain:

```cpp
TurboMind::Impl::Impl(string model_dir, EngineConfig config, FFICtxFactory ffi_ctx_factory):
    data_type_{}, engine_param_{}, ffi_ctx_factory_{ffi_ctx_factory}
{
    data_type_ = config.data_type;
    TM_CHECK(data_type_ == kBfloat16 || data_type_ == kHalf);

    static_cast<EngineConfig&>(engine_param_) = config;

    auto max_forward_token_num = config.max_prefill_token_num;
    max_forward_token_num += engine_param_.max_batch_size;

    {
        auto sp = engine_param_.attn_tp_size * engine_param_.attn_cp_size;
        engine_param_.max_forward_token_num = ((size_t)max_forward_token_num + sp - 1) / sp * sp;
    }

    comm_size_ = engine_param_.attn_dp_size * engine_param_.attn_tp_size * engine_param_.attn_cp_size;
    FT_CHECK(engine_param_.mlp_tp_size == comm_size_);

    communicator_type_ = std::move(config.communicator);

    HandleMissingParams();
    // ... rest unchanged ...
}
```

Notes: `engine_param_.max_batch_size`, `engine_param_.attn_tp_size`, etc. are inherited from `EngineConfig` and available after the cast-assignment. `config.data_type` and `config.max_prefill_token_num` read from the local parameter. `phases_` continues to use `config.async_` — no change needed.

- [ ] **Step 2: Build to verify compilation**

```bash
cd build && ninja 2>&1 | head -50
```

Expected: Clean build.

- [ ] **Step 3: Run model test to verify correctness**

```bash
python scripts/test_turbomind_model.py --model <available_model> --max-new-tokens 128
```

Verify the model responds with meaningful human-readable text.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/turbomind.cc
git commit -m "refactor: replace field-by-field EngineConfig copy with single cast-assignment"
```

---

## Self-Review

**1. Spec coverage:**
- ✅ `EngineParam` inherits `EngineConfig` (Task 1)
- ✅ Copy eliminated via `static_cast` assignment (Task 2)
- ✅ `step_length` removed (Task 1)
- ✅ Consumers unchanged (verified by build)
- ✅ pybind unchanged
- ✅ `HandleMissingParams()` unchanged
- ✅ `max_forward_token_num` derivation preserved

**2. Placeholder scan:** No TBDs, TODOs, or vague instructions. All code shown inline.

**3. Type consistency:** `EngineConfig` fields accessed via inheritance on `EngineParam` use the same names — `engine_param_.max_batch_size`, `engine_param_.attn_tp_size`, etc. — identical to before. The `static_cast<EngineConfig&>` is the only new syntax, and it correctly references the existing `EngineConfig` type.

Plan complete and saved to `docs/superpowers/plans/2026-04-22-engine-config-copy-elimination.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
