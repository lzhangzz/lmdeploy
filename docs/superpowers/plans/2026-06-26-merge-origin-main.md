# Merge `origin/main` into `memory-1a` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge `origin/main` (70 commits ahead) into `memory-1a` and fully integrate main's new features (qwen3.5-vit, get_ppl, cp fix) onto the new memory/scheduler/cache architecture, ending in a building, verified tree.

**Architecture:** Two-phase merge. Phase 1 produces a single reviewable **merge commit** that preserves the new architecture, adopts main's refactored *core* infrastructure (rope `MropeMode`, `causal` attention), builds with the qwen3.5-vit-*specific* compile units excluded, and passes the text/SSM verification. Phases 2–5 are isolated follow-up commits that integrate each feature and verify it. The deliberately-removed stateful-session/kill subsystem is never reintroduced.

**Tech Stack:** C++17 / CUDA (TurboMind), CMake + Ninja, Python (lmdeploy bindings), git.

**Spec:** `docs/superpowers/specs/2026-06-26-merge-origin-main-design.md`

---

## Conventions (read once before starting)

- **No worktree.** Work directly on branch `memory-1a` (repo rule: no worktree unless explicitly requested).
- **Build** (per repo rule, in the `build/` folder):
  - Configure (first time only): `cd build && sh ../my_generate.sh`
  - Build: `cd build && ninja`
- **Verify a model** (do **not** modify the script; do **not** wrap it in a bash loop):
  ```bash
  python scripts/test_turbomind_model.py \
    --model-id Qwen/Qwen3.5-27B \
    --cache-dir /mnt_cfs/huggingface_hub/hub/ \
    --tp <N> --gpus <DEVICES> --max-new-tokens 256
  ```
  The script sets `HF_HUB_CACHE` / `HF_HUB_OFFLINE` from `--cache-dir` internally. Hardware is **H200 (141 GB VRAM)** — `Qwen/Qwen3.5-27B` runs safely at **`--tp 1` on a single H200** (27B fp16 ≈ 54 GB + KV cache fits comfortably). Still check `get_gpu_usage` for a free GPU first and set `--gpus <free-id>`. **A pass = ≥128 coherent tokens relevant to the prompt; gibberish = a bug to fix before moving on.**
- **GPU commands run OUTSIDE the sandbox** (no driver inside the sandbox). Run build/verify with unrestricted permissions.
- **Commits:** only the commits this plan specifies. Conventional-commit style messages.
- **Rollback:** `git merge --abort` (during Phase 1 before the merge commit) or `git reset --hard premerge-memory-1a`.

### Global rule — stateful sessions stay removed
`b189745a` removed the stateful-session/kill subsystem (the `Sequence` persisted in `SequenceManager` after a request ended). Our branch has **zero** `Request::session`, `start_flag`/`end_flag`/`kill_flag`, `kill_reqs`, `seq_mgr_`. Never reintroduce them. Where main hangs a *new feature* off `r->session.*`, re-express it for our **stateless** model (every request is start+end in one shot). Concretely this turns main's interactive-only guards into simplifications (drop them).

### How the merge resolves files (verified against a throwaway trial merge)
`origin/main` is at **`4778480b` (77 commits ahead of base `a4025b91`)**. The 7 commits that landed most recently touch **zero `src/turbomind/` files** (Python serve / pytorch / ascend / version bump), so the C++ analysis below is current. A `git merge --no-ff origin/main` produces **exactly 12 conflicts**; the larger risk is the ~90 vit-PR files the merge resolves *silently*:
- **Taken wholesale from main** (`ours:0` since base): the entire `kernels/attention/*` suite (causal threading), `rotary_embedding.h`, `llama_rope.h` (**`RopeType::kMrope` removed → `MropeMode`**), `attention_weight.cc`, `model_root.h`, `input_processor.h`, `model_executor.h`, `core/module.h`, all `qwen3_5vit/*` + `vision_model.*` + `kernels/norm/*` + `layer_norm_weight.*` (new files), get_ppl's `cross_entropy_kernels.*` (new).
- **Auto-merged (both changed, no conflict):** `attention_universal.h` (disjoint regions — safe), `attention_weight.h`, `language_model.cc` (→ uses **4-arg** `PatchEmbedding`), `bind.cpp`, `model_request.cc/.h`. Verified consequences: `Request::mm_inputs` auto-merges cleanly into `Request` (so keep the `multimodal_input.h` include and it's an inert field in Phase 1 — `bind.cpp`'s header-only vit bindings then compile as-is); but main's **`SessionParam::start_flag/end_flag` bindings and the `End(session_id)` kill method** also auto-merge in and must be **stripped** (our branch has `Cancel`, not `End`).

Implication: our code must **compile against main's refactored rope/attention**, so Phase 1 adopts `MropeMode`/`causal` rather than "keeping ours"; the only things excluded are the genuinely vit-*specific* `.cc`/`.cu` compile units (headers + header-only bindings stay); and the auto-merged session/kill leaks are removed.

---

## Phase 1 — Structural merge commit (build green + text/SSM verified)

All conflict resolution happens in one in-progress merge; the merge is finished by a single commit (Task 1.11).

### Task 1.1: Safety tag and start the merge

**Files:** none (git state only)

- [ ] **Step 1: Tag a rollback point and confirm clean tree**
```bash
cd /data/lmdeploy-memory
git status --short          # expect only "?? lmdeploy/lib"
git tag premerge-memory-1a
git rev-parse --short HEAD  # expect b189745a
```

- [ ] **Step 2: Start the merge (it will stop with conflicts)**
```bash
git merge --no-ff origin/main
```
Expected: `Automatic merge failed; fix conflicts and then commit the result.`

- [ ] **Step 3: Confirm the exact conflict set (12 files)**
```bash
git diff --name-only --diff-filter=U | sort
```
Expected:
```
lmdeploy/messages.py
src/turbomind/engine/engine.cc
src/turbomind/engine/engine.h
src/turbomind/engine/model_executor.cc
src/turbomind/engine/request.h
src/turbomind/kernels/attention/attention_params.h
src/turbomind/models/input_processor.cc
src/turbomind/models/llama/SequenceManager.cc
src/turbomind/models/llama/SequenceManager.h
src/turbomind/models/llama/unified_attention_layer.cc
src/turbomind/models/output_processor.cc
src/turbomind/turbomind.cc
```
If the set differs (someone re-fetched `origin/main`), STOP and re-run the trial-merge analysis.

### Task 1.2: Resolve the clean "keep ours" conflicts

These five files have only keep-ours hunks for Phase 1, and their callers/headers on our side stay consistent (engine constructs the executor our way; output uses our `OutputRange`). Vision/get_ppl re-land in W1/W2.

**Files:**
- Modify: `src/turbomind/engine/engine.h` (text-only ctor)
- Modify: `src/turbomind/engine/engine.cc` (all 8 hunks keep ours; drops vision ctor, `seq_mgr_` metrics, and main's `Validate` lambda/`kill_reqs` per §global rule)
- Modify: `src/turbomind/engine/model_executor.cc` (keep our `RunCopies` body)
- Modify: `src/turbomind/turbomind.cc` (text-only Engine construction)
- Modify: `src/turbomind/models/output_processor.cc` (keep our `OutputRange`)

- [ ] **Step 1: Take ours for all five**
```bash
git checkout --ours src/turbomind/engine/engine.h src/turbomind/engine/engine.cc \
  src/turbomind/engine/model_executor.cc src/turbomind/turbomind.cc \
  src/turbomind/models/output_processor.cc
git add src/turbomind/engine/engine.h src/turbomind/engine/engine.cc \
  src/turbomind/engine/model_executor.cc src/turbomind/turbomind.cc \
  src/turbomind/models/output_processor.cc
```
> `checkout --ours` discards main's changes in these files. Intended: those changes are vision wiring (W1), get_ppl CE-loss (W2), or `seq_mgr_`-based metrics (deferred). The health-endpoint engine.cc change *is* the `seq_mgr_` metrics — correctly dropped.

- [ ] **Step 2: Verify no markers remain**
```bash
git grep -n '^<<<<<<<\|^>>>>>>>\|^=======$' -- src/turbomind/engine/engine.cc \
  src/turbomind/engine/engine.h src/turbomind/engine/model_executor.cc \
  src/turbomind/turbomind.cc src/turbomind/models/output_processor.cc
```
Expected: no output.

### Task 1.3: Resolve the union / additive conflicts

**Files:**
- Modify: `src/turbomind/engine/request.h` (3 hunks — union)
- Modify: `src/turbomind/kernels/attention/attention_params.h` (1 hunk — take theirs)
- Modify: `lmdeploy/messages.py` (1 hunk — union docstring)

- [ ] **Step 1: `attention_params.h` — take theirs.** Resolve the single hunk to:
```cpp
    bool  causal{true};
    int   layer_id;  // for debugging
```

- [ ] **Step 2: `request.h` — resolve the three conflict hunks** (verified line layout: hunks at 18–22 includes, 52–59 `GenerationConfig::OutType`, 212–254 `Sequence` tail)
  - Hunk 1 (includes): **keep BOTH** — our `#include "src/turbomind/engine/block.h"` **and** main's `#include "src/turbomind/engine/multimodal_input.h"`. The include is required because `Request::mm_inputs` **auto-merged** into the `Request` struct (verified at `request.h:111`, outside the markers); dropping the include would break it.
  - Hunk 2 (inside `struct GenerationConfig`): keep our `enum OutType {` brace style and add main's flag above it:
```cpp
    bool return_ppl = false;

    enum OutType {
```
  - Hunk 3 (`Sequence` tail): keep our **entire** execution-state block (`block_ids` … `input_embeds`/`input_embeds_offsets` … `is_active`/`is_canceled`) and append main's get_ppl members:
```cpp
    bool is_active   = false;
    bool is_canceled = false;

    // get_ppl / CE-loss (W2)
    Interval       input_ce_loss;
    Buffer_<float> ce_loss;  // device, size 1; rank-0 CE-loss accumulator.
};
```
  Do **not** add `bool end_flag` or any `session`/`start_flag`/`end_flag` member.
  > **No action needed for** `Request::mm_inputs` (auto-merged at line 111) — it stays as an inert field in Phase 1, which is exactly what makes `model_request.cc` (`r->mm_inputs = param.mm_inputs;`) and `bind.cpp` compile without edits. Only the per-sequence `Sequence::multimodal_inputs` vector is deferred to W1 (Task 2.1). `SessionParam` resolves to **ours** (`{id, step}`, only we changed it) — no flag leak in the struct.

- [ ] **Step 3: `messages.py` — union docstring.** Keep our new config docstrings (`linear_prefix_cache_min_interval`, `cache_prompt_boundary`, `cache_generation_boundary`, `cache_boundary_policy`) and adopt main's `quant_policy` wording:
```python
        quant_policy: default to 0. For TurboMind, when k/v is quantized
            into int4 or int8, set it to 4 or 8, respectively
```

- [ ] **Step 4: Stage and check markers**
```bash
git add src/turbomind/engine/request.h \
  src/turbomind/kernels/attention/attention_params.h lmdeploy/messages.py
git grep -n '^<<<<<<<\|^>>>>>>>\|^=======$' -- src/turbomind/engine/request.h \
  src/turbomind/kernels/attention/attention_params.h lmdeploy/messages.py
```
Expected: no output.

### Task 1.4: Resolve the modify/delete conflicts (keep deleted)

**Files:** Delete `src/turbomind/models/llama/SequenceManager.cc` and `.h`.

- [ ] **Step 1:**
```bash
git rm src/turbomind/models/llama/SequenceManager.cc \
       src/turbomind/models/llama/SequenceManager.h
```

### Task 1.5: Fix the rope guard in the attention layer (rope compile blocker)

`RopeType::kMrope` was removed from the enum (main's `llama_rope.h`/`rotary_embedding.h`/`attention_weight.cc` taken wholesale), but main **keeps** the `MropeRopeKernelParam mrope` sub-struct (`section`/`stride`/`position_ids`/`position_delta`/`length`) and adds `MropeMode mrope_mode`. Verified: our kept-ours mrope code (the legacy `r.inputs` loop, the `rope_param_.mrope.stride` at the setup, and the per-layer `params.rope_param.mrope.{position_ids,position_delta,length}` assignments) **compiles unchanged** against main's `RopeKernelParam`. The *only* required Phase-1 change is the guard condition. The vision env-source branch is W1.

**Files:**
- Modify: `src/turbomind/models/llama/unified_attention_layer.cc` (the 1 mrope conflict hunk; rest auto-merged)

- [ ] **Step 1: Resolve the conflict to OUR side, changing only the guard.** Keep our entire legacy mrope block (which already uses our `c.history_len + c.inflight_input_len` interval); change just the condition:
```cpp
    // was: else if (rope_param_.type == RopeType::kMrope) {
    else if (rope_param_.mrope_mode != MropeMode::kNone) {
        const auto stride = d.mrope_position_ids.stride(0);
        for (int i = 0; i < rc.size(); ++i) {
            auto& c = *rc[i];
            auto& r = *c.req;
            if (auto pos_ids = r.inputs.try_("mrope_position_ids")) {
                int length                   = pos_ids->shape(0);
                mrope_length_buf_[i]         = length;
                mrope_position_delta_buf_[i] = *r.inputs.at("mrope_position_delta").data<int>();
                if (auto o = Interval{0, length}
                             & Interval{c.history_len + c.inflight_input_len, Interval::Size{c.input_len}}) {
                    copy(pos_ids->data<int>() + o.begin() * 3,
                         (int)o.size() * 3,
                         d.mrope_position_ids.data() + i * stride + o.begin() * 3);
                }
            }
            else {
                mrope_length_buf_[i] = mrope_position_delta_buf_[i] = 0;
            }
        }
        // ... keep the remainder of our existing block (copy-out to d.mrope_length / d.mrope_position_delta) ...
    }
```
Do **not** add main's `env.try_("mrope_length")` borrow branch here — that is W1 (Task 2.7). For Phase-1 text/SSM models `mrope_mode == kNone`, so this block is inert.

- [ ] **Step 2: Confirm no other `RopeType::kMrope` remains** in our-resolved files:
```bash
git grep -n "RopeType::kMrope" -- 'src/turbomind/*'
```
Expected: no output (the other users — `rotary_embedding.h`, `attention_weight.cc`, `llama_rope.h` — were taken from main and no longer reference it).

- [ ] **Step 3: Stage + marker check**
```bash
git add src/turbomind/models/llama/unified_attention_layer.cc
git grep -n '^<<<<<<<\|^>>>>>>>\|^=======$' -- src/turbomind/models/llama/unified_attention_layer.cc
```
Expected: no output.

### Task 1.6: Reconcile main-changed headers vs our deferred `.cc`

Three headers were changed by main (taken to main's version) while their behavior is wired in our resolved `.cc` — fix the interface so Phase 1 compiles.

**Files:**
- Modify: `src/turbomind/models/input_processor.cc` (conflict — keep ours, then add the `env` param)
- Force-ours: `src/turbomind/engine/model_executor.h`
- Force-ours: `src/turbomind/models/model_root.h`

- [ ] **Step 1: `input_processor` — accept main's 4-arg `PatchEmbedding`.** The merged `language_model.cc` calls `PatchEmbedding(phase, input_embeds, copy, env)` (4-arg, from main) and the taken `input_processor.h` declares the 4-arg form. Resolve the `input_processor.cc` conflict to **ours** (no `SequenceManager.h`/`vision_model.h` include), then change our `PatchEmbedding` definition signature to match the header, leaving `env` unused in Phase 1:
```bash
git checkout --ours src/turbomind/models/input_processor.cc
```
Then update **all three sites** in `input_processor.cc` (verified line numbers on HEAD):
- `Impl::PatchEmbedding` (≈line 196): add `, TensorMap& env` (and `(void)env;` for now).
- `InputProcessor::PatchEmbedding` (≈line 255): add `, TensorMap& env` to match the taken header.
- the forwarding call (≈line 257): `impl_->PatchEmbedding(phase, embeds, copy, env);`
```cpp
void InputProcessor::PatchEmbedding(int phase, Tensor& embeds, BatchCopy& copy, TensorMap& env)
{
    impl_->PatchEmbedding(phase, embeds, copy, env);
}
```
Stage: `git add src/turbomind/models/input_processor.cc`. (W1, Task 2.6, replaces the `(void)env;` placeholders with the real multimodal merge.)

- [ ] **Step 2: `model_executor.h` — force ours.** Main adds a `VisionModel*` ctor param + `#include vision_model.h`; only `engine.cc` (resolved-to-ours) constructs `ModelExecutor`, so keep our header. Our version == base (we didn't change it):
```bash
git checkout HEAD -- src/turbomind/engine/model_executor.h
git add src/turbomind/engine/model_executor.h
```
W1 restores main's version and threads the vision param.

- [ ] **Step 3: `model_root.h` — force ours.** Main adds `vision_model_ptr()` + a `VisionModelWeight` child; not needed until W1 (Phase 1 loads no VLM). Keep our header:
```bash
git checkout HEAD -- src/turbomind/models/model_root.h
git add src/turbomind/models/model_root.h
```
W1 restores main's version.

### Task 1.7: Exclude qwen3.5-vit-specific compile units (headers stay)

Only the vit-specific `.cc`/`.cu` **compile units** are excluded from CMake. **All vit headers stay** (they are header-only-safe). The `bind.cpp` vit bindings are **header-only** (`Qwen3_5VitItem`/`Qwen3_5VitInput` from `qwen3_5vit/qwen3_5vit_input.h`, bound via ctors/fields/`bind_config` — no excluded-`.cc` symbols) and compile as-is once `Request::mm_inputs` exists (Task 1.3) — **do not comment them.**

**Files:** `src/turbomind/models/CMakeLists.txt`, `src/turbomind/kernels/norm/CMakeLists.txt`, `src/turbomind/python/CMakeLists.txt`, and any `add_subdirectory(qwen3_5vit)` location.

- [ ] **Step 1: Find the vit-source CMake wiring**
```bash
git grep -n "qwen3_5vit\|vision_model\|layer_norm_weight\|norm/layer_norm\|add_subdirectory(qwen3_5vit)\|kernels/norm" -- 'src/turbomind/**/CMakeLists.txt'
```

- [ ] **Step 2: Comment out (don't delete) the vit-specific compile units** with a restore marker: `qwen3_5vit/*.cc/.cu`, `vision_model.cc`, `vision_model_weight.cc` (if it exists), `layer_norm_weight.cc`, `kernels/norm/layer_norm.cu`, and any vit `.cu` test targets:
```cmake
# TODO(merge-W1): re-enable qwen3.5-vit sources after vit integration
# add_subdirectory(qwen3_5vit)
# vision_model.cc
# layer_norm_weight.cc
```
Do **not** exclude headers and do **not** touch `bind.cpp` here.

- [ ] **Step 3: Stage**
```bash
git add src/turbomind/models/CMakeLists.txt src/turbomind/python/CMakeLists.txt
# plus kernels/norm/CMakeLists.txt and any other edited CMakeLists.txt
```
The exact exclusion set is finalized by the Task 1.9 build — iterate 1.7 ↔ 1.9.

### Task 1.8: Strip the auto-merged session/kill leaks (`bind.cpp`, `model_request`)

`bind.cpp`, `model_request.cc`, and `model_request.h` auto-merged with main's stateful-session code. Verified leaks to remove (these reference members/methods our branch deleted — `SessionParam` is now `{id, step}`, and we have `Cancel`, not `End`):
- `bind.cpp`: main's `py::class_<SessionParam>` binds `start`/`end` (`&SessionParam::start_flag`/`end_flag`) and its init sets `param.start_flag`/`end_flag`; plus a `model_request->End(cb, session_id)` ("end"/`session_id`) binding. **Reduce the `SessionParam` binding to `id`/`step` only and delete the `End` binding; keep our `Cancel` binding and the (header-only) vit bindings.**
- `model_request.h`/`.cc`: drop any leaked `void End(...)` decl/def and any `param.session.start_flag/end_flag` use. Keep our `Cancel`, and keep the legitimate `param.session.id`/`param.session.step` (our request-id/step carrier) and the auto-merged `r->mm_inputs = param.mm_inputs;` and get_ppl `ce_loss` alloc.

- [ ] **Step 1: Grep for the removed symbols** (these are precise — `SessionParam`/`session_id_`/`session_len_` are *legitimate* and intentionally not matched):
```bash
git grep -n "start_flag\|end_flag\|kill_flag\|kill_reqs\|seq_mgr_\|->End(\|\.End(\|\"end\"\|session\.start_flag\|session\.end_flag" -- 'src/turbomind/*' \
  ':!src/turbomind/models/qwen3_5vit/*' ':!src/turbomind/models/vision_model*'
```
Expected after fixes: no hits.

- [ ] **Step 2: Remove each leak** by comparing `git show origin/main:<file>` vs `HEAD:<file>` and keeping our shape (no `End`, no `start_flag`/`end_flag`). Edit `bind.cpp` (`SessionParam` binding + `End` binding) and `model_request.{h,cc}` accordingly.

- [ ] **Step 3: Re-run the grep until clean.**

### Task 1.9: Configure and build (iterate)

- [ ] **Step 1: Build (outside sandbox)**
```bash
cd /data/lmdeploy-memory/build
sh ../my_generate.sh    # only if build/ not yet configured
ninja
```

- [ ] **Step 2: Fix iteratively** — expected trap categories:
  - `SessionParam::start_flag`/`end_flag` or `ModelRequest::End` referenced (in `bind.cpp`/`model_request`) → a Task 1.8 session/kill leak; strip it.
  - `RopeType::kMrope` not found → a leftover in an our-resolved file; convert to `MropeMode` (Task 1.5 pattern).
  - `PatchEmbedding` arity mismatch → align to the 4-arg form (Task 1.6 Step 1).
  - `ModelExecutor`/`ModelRoot` vision member errors → ensure those headers are ours (Task 1.6 Steps 2–3).
  - Undefined reference to a vit `.cc` symbol (e.g. `CreateVisionModel`, a `VisionModel`/`Qwen3_5Vit*` method/vtable) from a non-excluded unit → either that unit shouldn't be calling it in Phase 1 (it leaked from main — revert to ours), or a needed compile unit was over-excluded (Task 1.7). Header-only vit *bindings* should link fine; a link error here means a real `.cc` symbol is used.
  Re-run `ninja` until green. Do not proceed with a broken build.

### Task 1.10: Verify the text / SSM path

- [ ] **Step 1: Pick a free GPU, then run (outside sandbox):**
```bash
cd /data/lmdeploy-memory
python scripts/test_turbomind_model.py \
  --model-id Qwen/Qwen3.5-27B \
  --cache-dir /mnt_cfs/huggingface_hub/hub/ \
  --tp 1 --gpus 0 --max-new-tokens 256
```
- [ ] **Step 2: Read `--- response 0 ---`.** PASS = ≥128 coherent, on-topic tokens; confirm the gated-deltanet/linear-attention path ran (Qwen3.5 is hybrid). Gibberish/crash → debug + rebuild before committing.

### Task 1.11: Commit the merge

- [ ] **Step 1: Confirm no unmerged paths or stray markers**
```bash
git diff --name-only --diff-filter=U          # expect empty
git grep -n '^<<<<<<<\|^>>>>>>>\|^=======$' -- . || echo "no markers (good)"
```
- [ ] **Step 2: Commit (finishes the merge)**
```bash
git commit -m "$(cat <<'EOF'
merge: integrate origin/main into memory-1a (structural)

Resolve conflicts preserving the new memory/scheduler/cache architecture.
Adopt main's refactored core (MropeMode rope, causal attention). Keep the
stateful-session/kill subsystem removed. qwen3.5-vit-specific sources and
bindings are excluded from the build and integrated in a follow-up; get_ppl
request fields landed (logic follows). Text/SSM verified on Qwen3.5-27B.
EOF
)"
git log --oneline -1 && git status
```

### Phase 1 execution notes (actual outcome vs. predicted)

Done in commit `911c745b`. The trial-merge-based predictions diverged in three
benign ways; recorded here so W1 stays accurate:

1. **`input_processor.cc` (Task 1.6 Step 1):** the body auto-merged cleanly to
   main's 4-arg `PatchEmbedding` *with* the multimodal embed branch
   (`PatchMultimodalEmbedding` + `MultiModalEmbeddingData`). Only the include
   block conflicted. Resolved by keeping `#include vision_model.h` (needed for
   `MultiModalEmbeddingData`) and dropping the deleted `SequenceManager.h`. The
   multimodal branch is inert in Phase 1 (nothing produces env `"multimodal"`),
   so W1 Task 2.6 mostly verifies/extends it rather than writing it from scratch.
2. **Session/kill leaks (Task 1.8):** none existed. `SessionParam` auto-merged to
   ours (`{id, step}`), only `Cancel` is bound (no `End`/kill), and
   `model_request.{cc,h}` carry only the legitimate `session.id/step`,
   `r->mm_inputs`, and get_ppl `ce_loss` alloc. Verified by grep; nothing removed.
3. **Qwen3.5 is a VLM → vision *weights* load in Phase 1 (supersedes Task 1.6
   Step 3 + part of Task 1.7).** Loading any Qwen3.5 checkpoint builds the vision
   weight sub-tree via C++ module handles, which requires `ModelRoot`'s vision
   child; with `model_root.h` force-ours'd the loader hit
   `add_child_raw` on `None`. Fix: restore `model_root.h` (vision child) and
   compile the two **standalone** vision weight units
   (`qwen3_5vit_weight.cc`, `qwen3_5vit_block_weight.cc`; they only pull
   `linear/layer_norm/attention_weight` + `registry`). The vision **encoder +
   CUDA kernels + `vision_model.cc`** stay excluded; `model_executor.h`/engine
   stay vision-unaware, so the encoder is never created/run for text. This let
   the unmodified `test_turbomind_model.py` load Qwen3.5-27B and verify the
   text/gated-deltanet path (256 coherent tokens, tp=1 H200). W1 picks up from
   here (Task 2.2 updated accordingly).

---

## Phase 2 — W1: qwen3.5-vit integration

Re-home the multimodal carrier, re-enable the vit sources + bindings, restore main's vision-aware headers, thread `VisionModel` through the engine, and translate main's interactive-only gating to our stateless model.

### Task 2.1: Add the per-sequence multimodal carrier

**Files:** Modify `src/turbomind/engine/request.h`

> `Request::mm_inputs` and the `multimodal_input.h` include already landed in Phase 1 (auto-merged + Task 1.3). W1 only adds the per-sequence feature vector.

- [ ] **Step 1: Add the forward decl** (near the existing `struct Sequence;`):
```cpp
struct MultiModalData;  // defined in models/vision_model.h
```
- [ ] **Step 2: Add the carrier to `Sequence`**, beside `input_embeds`:
```cpp
    // persistent per-sequence vision features (qwen3.5-vit, W1)
    std::vector<std::shared_ptr<MultiModalData>> multimodal_inputs;
```

### Task 2.2: Restore main's vision-aware headers + CMake vit sources

> Phase 1 already restored `model_root.h` (vision child) and compiled the two
> standalone vision **weight** units so Qwen3.5 VLM checkpoints load their vision
> sub-tree (see "Phase 1 execution notes"). W1 only restores the remaining
> vision-aware header and re-enables the vision **encoder + kernels**.

- [ ] **Step 1: Restore main's `model_executor.h`** (still force-ours'd in Phase 1; `input_processor.h`/`model_root.h` are already main's):
```bash
git checkout origin/main -- src/turbomind/engine/model_executor.h
```
- [ ] **Step 2: Uncomment the remaining `TODO(W1)`** vit sources in `src/turbomind/models/CMakeLists.txt` — the vision encoder + CUDA kernels (`vision_model.cc`, `qwen3_5vit/*.cu`, `qwen3_5vit/qwen3_5vit.cc`) and the `test_mrope_position_ids` test. The two weight `.cc` units are already compiled; the vit `bind.cpp` bindings were never commented (header-only) — nothing to do there.
- [ ] **Step 3: Do not build yet** — engine threading (2.3–2.8) must land first.

### Task 2.3: Thread `VisionModel` through the Engine

**Files:** `src/turbomind/engine/engine.h`, `src/turbomind/engine/engine.cc`

- [ ] **Step 1: `engine.h` — union our Phase-1 ctor with main's vision params:**
```cpp
    Engine(EngineParam                  param,
           ObjectAllocator              alloc,
           CacheRegistry                cache_registry,
           LanguageModel                model,
           std::unique_ptr<VisionModel> vision_model,  // null for text-only
           const ModelWeight&           weights,
           Context&                     ctx,
           Gateway&                     gateway,
           int                          device_id,
           int                          queue_id,
           int                          phases);
```
Add `#include "src/turbomind/models/vision_model.h"` (or forward-declare `class VisionModel;`).
- [ ] **Step 2: `engine.cc` — mirror in `Engine::Engine` and `Engine::Impl::Impl`** (decl + def), keeping our `object_allocator_`/`scheduler_` init, adding members `std::unique_ptr<VisionModel> vision_model_;` and `const ModelWeight& weights_;`, and forwarding through `make_unique<Impl>(...)`.
- [ ] **Step 3: Build the engine to confirm signatures.**

### Task 2.4: Construct the VisionModel + executor wiring

**Files:** `src/turbomind/turbomind.cc`, `src/turbomind/engine/model_executor.cc`

- [ ] **Step 1: In `turbomind.cc`, build the vision model and pass it to Engine** (mirror main `turbomind.cc:282-290`, adapted to our Engine ctor):
```cpp
    std::unique_ptr<VisionModel> vision_model;
    if (auto* vw = weights_[index]->vision_model_ptr()) {
        vision_model = CreateVisionModel(*vw, param, ctx, phases_);
    }
    // ... pass std::move(vision_model) and *weights_[index]->text_model_ptr() into Engine(...)
```
Add includes for `vision_model.h` / `vision_model_weight.h`. Remove any main warm-up `param.session.*` lines if present (we have no session).
- [ ] **Step 2: `model_executor.cc` — run the encoder before restore copies** (now that `model_executor.h` carries `VisionModel*`):
```cpp
        if (vision_model_) {
            vision_model_->Run(BatchOp::kPrepare, d.phase, env);
        }
        RunCopies(d.restore_copies);
```
Thread `VisionModel*` into the executor following main's `model_executor.h` ctor; engine.cc passes `vision_model_.get()`.

### Task 2.5: Translate the vit `Add()` gating to stateless

**Files:** `src/turbomind/models/qwen3_5vit/qwen3_5vit.cc`

- [ ] **Step 1: Map main's `RequestCache` to our `Sequence`, drop the interactive guard.** In our arch `env.at("requests")` yields `Sequence*` (has `->req`):
```cpp
    int Add(Sequence& s)
    {
        auto& r = *s.req;
        if (r.mm_inputs) {
            // stateless model: every request is start+end — no interactive guard
            const auto mm_inputs = std::dynamic_pointer_cast<multimodal::Qwen3_5VitInput>(r.mm_inputs);
            if (!mm_inputs) {
                return Request::kInvalid;
            }
            for (const auto& item : mm_inputs->items) {
                if (item.modality != multimodal::Modality::kImage
                    && item.modality != multimodal::Modality::kVideo) {
                    return Request::kInvalid;
                }
                const int tokens = item.token_end - item.token_begin;
                if (tokens <= 0) {
                    return Request::kInvalid;
                }
                s.multimodal_inputs.push_back(std::make_shared<MultiModalData>(
                    MultiModalData{item.data, Interval{item.token_begin, Interval::Size{tokens}}, item.grid_thw}));
            }
        }
        return Request::kOk;
    }
    void Add(int phase, TensorMap& env)
    {
        const Buffer_<Sequence*> rc = env.at("requests").buffer();
        for (int i = 0; i < rc.size(); ++i) {
            auto& s = *TM_CHECK_NOTNULL(rc[i]);
            if (s.status == 0) {       // confirm our Sequence::status convention (Request::kOk)
                s.status = Add(s);
            }
        }
    }
```
Delete the `if ((not r.session.start_flag) or (not r.session.end_flag))` block. Audit the rest of `qwen3_5vit.cc` for any other `RequestCache`/`c.seq`/`r.session` usage.

### Task 2.6: Multimodal embedding patch in input_processor

**Files:** `src/turbomind/models/input_processor.cc`

- [ ] **Step 1: Port main's multimodal embedding handling into `PatchEmbedding`, using the `env` param** (now declared). Compare `git show origin/main:src/turbomind/models/input_processor.cc`. Re-add the `input_embedding_ranges`/`input_embeddings` → `s.input_embeds` logic and the fused vision-embed merge that reads from `env`, but **drop the `if (!r.session.end_flag) clone` branch** (stateless = no persist, reference directly). Keep our `step0`/`seq_len` range math.

### Task 2.7: mrope vision env-source

**Files:** `src/turbomind/models/llama/unified_attention_layer.cc` (and the vit encoder that produces env mrope tensors)

- [ ] **Step 1: Add main's env-source branch** in front of the Phase-1 legacy block, so the C++ vision encoder's device tensors are borrowed with no copy when present:
```cpp
    else if (rope_param_.mrope_mode != MropeMode::kNone) {
        if (env.try_("mrope_length")) {
            // C++ vision encoder produced device tensors in FastRoPE's exact layout
            d.mrope_length         = env.at("mrope_length").buffer().borrow();
            d.mrope_position_delta = env.at("mrope_position_delta").buffer().borrow();
            d.mrope_position_ids   = env.at("mrope_position_ids").borrow();
        }
        else {
            // ... the Phase-1 legacy r.inputs loop (unchanged) ...
        }
    }
```
Confirm `Tensor_<int>::borrow()` / `Buffer_<int>::borrow()` exist in our core (they are main's APIs on main's core types; if our core renamed them, use the equivalent non-owning view). Build to confirm.

- [ ] **Step 2: Confirm the vit encoder produces the env tensors** (`qwen3_5vit` emits `mrope_length`/`mrope_position_delta`/`mrope_position_ids` into `env`) and the layouts match FastRoPE (3-row position ids; see `git show origin/main:src/turbomind/models/qwen3_5vit/mrope_position_ids.cu`).

### Task 2.8: Build, fix, verify

- [ ] **Step 1: Build** (`cd build && ninja`); iterate. New objects: `qwen3_5vit/*`, `vision_model.cc`, `layer_norm_weight.cc`.
- [ ] **Step 2: Re-run the leak grep** (Task 1.8 Step 1) — still **zero** `session`/`kill`/`seq_mgr_` hits; `mm_inputs`/`VisionModel` now legitimately present in vit/engine wiring.
- [ ] **Step 3: Regression — re-verify text/SSM** (Task 1.10 command). Must PASS.
- [ ] **Step 4: VL check (best-effort).** Fetch the image (may need non-sandbox network):
```bash
curl -L -o /tmp/tiger.jpeg \
  https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg
```
Run a qwen3.5-vl image+prompt inference ("Describe this image.") and confirm the response references a tiger/feline. **No qwen3.5-vl model path is provided yet (`/data/models.json` lists none)** — if unavailable, record a flagged, deferred verification; the path is integrated + compiles. Do not block the commit.

### Task 2.9: Commit W1

- [ ] **Step 1:**
```bash
git add -A
git commit -m "$(cat <<'EOF'
feat(turbomind): integrate qwen3.5-vit onto the new architecture

Re-home the multimodal carrier onto the new per-rank Sequence, restore the
vision-aware headers, thread VisionModel through Engine/executor, and
translate vit input/gating from the removed stateful-session model to the
stateless request model.
EOF
)"
```

---

## Phase 3 — W2: get_ppl / CE-loss

`request.h` fields landed in Phase 1. Re-apply the compute/output logic onto our `OutputRange`.

### Task 3.1: Re-apply CE-loss in the output processor

**Files:** `src/turbomind/models/output_processor.cc`

- [ ] **Step 1: Port main's CE-loss onto our structures.** Compare `git show origin/main:src/turbomind/models/output_processor.cc` vs ours. Re-introduce, adapted to our `vector<OutputRange>` (do **not** revert to main's `tuple<int,int,Interval,Interval>`): the `full_ce_loss`/`ce_loss_segments`/`ce_targets` members, the `ComputeAndOutputLogits(Data&, const Tensor&, BatchCopy&, <our request collection>)` extension (use our `Sequence*`/`b.rc`, not `RequestCache`), the `if (d.full_logits || d.full_ce_loss)` branch, and the `OutputLogitsImpl(..., rs)` overload. The CE kernels (`cross_entropy_kernels.*`) already merged.
- [ ] **Step 2: Build** (`ninja`); fix until green.

### Task 3.2: Re-add the `return_ppl` admission clause

**Files:** `src/turbomind/engine/engine.cc`

- [ ] **Step 1: Extend our `Validate` prefix-caching incompatibility check** (no session, no kill_reqs):
```cpp
            else if (r->gen_cfg.output_logits == GenerationConfig::kAll
                     || r->gen_cfg.output_last_hidden_state == GenerationConfig::kAll
                     || r->gen_cfg.return_ppl) {
                TM_LOG_ERROR("Skip inconsistent infer request for ID {}: prefix caching cannot "
                             "output logits/last_hidden_states for all tokens or ppl",
                             r->id);
                r->ec = Request::kInconsistency;
            }
```
Confirm `GenerationConfig::return_ppl` exists; add it (mirror main's `GenerationConfig`) if missing.

### Task 3.3: Wire the Python `/get_ppl` surface

**Files:** `lmdeploy/messages.py`, `lmdeploy/turbomind/turbomind.py`, `lmdeploy/serve/openai/api_server.py`, `lmdeploy/serve/openai/protocol.py`

- [ ] **Step 1: Port the ppl plumbing** from #4679 into our post-refactor Python layer using `git show c296bebb` as reference, matching our `messages.py`/`turbomind.py` shapes.

### Task 3.4: Build, verify, commit

- [ ] **Step 1: Build** (`ninja`), re-verify text path (Task 1.10) — PASS required.
- [ ] **Step 2: Commit**
```bash
git add -A
git commit -m "feat(turbomind): re-integrate get_ppl / CE-loss onto OutputRange"
```

---

## Phase 4 — W3: cp inference fix

Most of #4619 lands clean (`cp_utils.*`) or was taken in Phase 1 (`causal`).

### Task 4.1: Confirm and finish the cp fix

- [ ] **Step 1: Verify the pieces post-merge:**
```bash
git grep -n "causal" src/turbomind/kernels/attention/attention_params.h
ls src/turbomind/kernels/attention/cp_utils.cu src/turbomind/kernels/attention/cp_utils.h
git grep -n "cp_utils\|causal" src/turbomind/models/llama/unified_attention_layer.cc
```
If `cp_utils.*` weren't auto-added, port them from `git show 6276b3bd`; ensure the 3-line `unified_attention_layer.cc` cp addition survived the Task 1.5 resolution.
- [ ] **Step 2: Build** (`ninja`). cp end-to-end needs a multi-GPU cp run — best-effort; flag deferred if unavailable.
- [ ] **Step 3: Commit only if Task 4.1 produced changes:**
```bash
git add -A && git commit -m "fix(turbomind): carry cp inference fix into merged tree" || echo "nothing to commit"
```

---

## Phase 5 — W4: misc + final verification

### Task 5.1: Health endpoint & misc clean adds

- [ ] **Step 1: Confirm the Python health-endpoint + other clean-add commits merged**; no action unless a build/import breaks. Note: TurboMind `ScheduleMetrics` is stubbed (Phase 1) — the health probe sees empty metrics; reviving metrics onto the new scheduler is deferred (spec §9).

### Task 5.2: Final pass

- [ ] **Step 1: Clean rebuild** to catch stale objects: `cd build && ninja -t clean && ninja`.
- [ ] **Step 2: Final text/SSM verification** (Task 1.10 command) — PASS required.
- [ ] **Step 3: Record deferred items** in spec §9 (metrics revival; VL verification if no model was available; cp multi-GPU verification).
- [ ] **Step 4: Confirm history**: `git log --oneline --graph -8` shows the merge commit + W1/W2/(W3) feature commits.

---

## Self-review (author checklist — completed)

- **Spec coverage:** Phase 1 ↔ §3/§5; core-infra-taken-from-main ↔ §5 ("How the merge resolves files") / Tasks 1.5–1.7; global session rule ↔ §4 (Tasks 1.2/1.8); W1 ↔ §6 / Task 2.x; W2 ↔ §6 / Task 3.x; W3 ↔ §6 / Task 4.x; W4/deferred ↔ §6/§9 / Task 5.x; verification ↔ §7 / Tasks 1.10, 2.8, 3.4, 5.2; risk audit ↔ §8 / Tasks 1.5–1.9.
- **Type consistency:** carrier is `std::vector<std::shared_ptr<MultiModalData>> multimodal_inputs` everywhere; `MultiModalData{data, interval, grid_thw}` matches `vision_model.h`; `Request::mm_inputs` is `std::shared_ptr<multimodal::Input>`; mrope uses `rope_param_.mrope_mode != MropeMode::kNone` + our `c.history_len + c.inflight_input_len`; `PatchEmbedding` is the 4-arg form post-Phase-1; `Validate` keeps the `infer_reqs`-only signature.
- **No placeholders:** every code step shows concrete code or an exact `git show <ref>:<path>` reference; build/verify commands have expected output; the iterative build (Task 1.9) enumerates the specific expected trap categories.
