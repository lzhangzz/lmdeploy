# Configurable Prompt-Boundary Skip Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the reusable prompt-boundary (`fork_to`) node end at a configurable `B = prompt_len - K` (engine knob `cache_prompt_boundary_skip`, default 1 = the legacy `prompt_len-1` position) so thinking models whose chat template appends a multi-token volatile suffix (e.g. `<think>\n`) still get partial-tail prefix reuse on the next turn. The gate is matchability: publish a boundary a later request can match up to `B` with no exclusion — including block-aligned prompts the old code skipped.

**Architecture:** A single engine-level integer flows Python → `EngineConfig` → `Scheduler`. A pure geometry function `PlanPromptBoundary(prompt_len, block_size, skip, miss)` decides where the boundary lands and whether a partial node is needed (`B % block_size != 0`). `SetupForks` consumes it and stores `Sequence::prompt_boundary_pos`; the admission clamp and the auto boundary policy read that field instead of the literal `prompt_len - 1`.

**Tech Stack:** C++17 (TurboMind engine), Catch2 unit tests, pybind11-bound `EngineConfig`, Python dataclass `TurbomindEngineConfig`. Build with Ninja in `build/`. Spec: `docs/superpowers/specs/2026-06-30-cache-prompt-boundary-skip-design.md`.

---

## Task 1: Config knob plumbing (Python → EngineConfig → Scheduler) + `prompt_boundary_pos` field

Pure plumbing. After this task the knob exists everywhere and defaults to `1`, but nothing reads it yet, so behavior is unchanged.

**Files:**
- Modify: `lmdeploy/messages.py` (field + docstring near `:323-326`)
- Modify: `lmdeploy/turbomind/turbomind.py:242` (forward to `ec`)
- Modify: `src/turbomind/engine/engine_config.h:27` (X-macro field)
- Modify: `src/turbomind/engine/scheduler.h` (ctor param `:90-97`, member `:223`)
- Modify: `src/turbomind/engine/scheduler.cc:255-276` (ctor param + init, clamped `>= 1`)
- Modify: `src/turbomind/engine/engine.cc:226-233` (pass `param_.cache_prompt_boundary_skip`)
- Modify: `src/turbomind/engine/request.h` (Sequence field `prompt_boundary_pos`)

- [ ] **Step 1: Add the C++ `EngineConfig` field**

In `src/turbomind/engine/engine_config.h`, add to the `ENGINE_FIELDS(X)` macro list immediately after the `cache_prompt_boundary` line:

```cpp
    X(bool, cache_prompt_boundary, false)                                                                              \
    X(int, cache_prompt_boundary_skip, 1)                                                                              \
    X(bool, cache_generation_boundary, false)                                                                          \
```

(The X-macro auto-binds the field to the Python `_tm.EngineConfig`.)

- [ ] **Step 2: Add the Sequence field**

In `src/turbomind/engine/request.h`, in `struct Sequence`, change the prompt-boundary block (currently `:245-248`) to insert `prompt_boundary_pos`:

```cpp
    bool prompt_boundary_node = false;  // a reusable prompt-boundary partial node exists; when the boundary policy
                                        // admits the publish, the producer clamps its forward to prompt_boundary_pos to
                                        // populate the node's KV (and publish a checkpoint when the model is recurrent)
    int  prompt_boundary_pos = 0;       // resolved boundary B = prompt_len - cache_prompt_boundary_skip; 0 = none
    int  prompt_boundary_publish = -1;  // cached CacheBoundaryPolicy decision: -1 undecided, 0 no, 1 yes
```

- [ ] **Step 3: Add the `Scheduler` ctor param + member**

In `src/turbomind/engine/scheduler.h`, add a param to the constructor (after `cache_prompt_boundary`):

```cpp
    Scheduler(ObjectAllocator&                     alloc,
              CacheRegistry                        registry,
              int                                  cache_block_seq_len,
              bool                                 enable_prefix_caching,
              bool                                 cache_prompt_boundary,
              int                                  cache_prompt_boundary_skip,
              bool                                 cache_generation_boundary,
              std::unique_ptr<CacheBoundaryPolicy> boundary_policy,
              const int&                           is_warm_up);
```

And add the member after `cache_prompt_boundary_{false};` (`:223`):

```cpp
    bool                                 cache_prompt_boundary_{false};
    int                                  cache_prompt_boundary_skip_{1};
    bool                                 cache_generation_boundary_{false};
```

- [ ] **Step 4: Initialize the member (clamped) in `scheduler.cc`**

In `src/turbomind/engine/scheduler.cc:255-266`, add the param and the clamped initializer:

```cpp
Scheduler::Scheduler(ObjectAllocator&                     alloc,
                     CacheRegistry                        registry,
                     int                                  cache_block_seq_len,
                     bool                                 enable_prefix_caching,
                     bool                                 cache_prompt_boundary,
                     int                                  cache_prompt_boundary_skip,
                     bool                                 cache_generation_boundary,
                     std::unique_ptr<CacheBoundaryPolicy> boundary_policy,
                     const int&                           is_warm_up):
    enable_prefix_caching_{enable_prefix_caching},
    cache_prompt_boundary_{cache_prompt_boundary},
    cache_prompt_boundary_skip_{cache_prompt_boundary_skip < 1 ? 1 : cache_prompt_boundary_skip},
    cache_generation_boundary_{cache_generation_boundary},
```

- [ ] **Step 5: Pass the config through in `engine.cc`**

In `src/turbomind/engine/engine.cc:226-233`, add the argument between `cache_prompt_boundary` and `cache_generation_boundary`:

```cpp
    scheduler_{object_allocator_,
               std::move(cache_registry),
               param_.cache_block_seq_len,
               param_.enable_prefix_caching,
               param_.cache_prompt_boundary,
               param_.cache_prompt_boundary_skip,
               param_.cache_generation_boundary,
               CreateCacheBoundaryPolicy(param_),
               is_warm_up_},
```

- [ ] **Step 6: Add the Python dataclass field**

In `lmdeploy/messages.py`, add the field after `cache_prompt_boundary: bool = False` (`:324`):

```cpp
    cache_prompt_boundary: bool = False
    cache_prompt_boundary_skip: int = 1
    cache_generation_boundary: bool = False
```

And add to the docstring (after the `cache_prompt_boundary:` entry, before `cache_generation_boundary:`):

```
        cache_prompt_boundary_skip: number of trailing prompt tokens treated as
            the volatile generation-prompt suffix (e.g. a chat template's
            `<think>\n`) and excluded from the reusable prompt-boundary node, so
            the node ends at `prompt_len - cache_prompt_boundary_skip`. Requires
            `cache_prompt_boundary`. Default 1 (exclude only the last token).
```

- [ ] **Step 7: Forward the field to `ec` in `turbomind.py`**

In `lmdeploy/turbomind/turbomind.py`, after line 242 add:

```python
        ec.cache_prompt_boundary = engine_config.cache_prompt_boundary
        ec.cache_prompt_boundary_skip = engine_config.cache_prompt_boundary_skip
        ec.cache_generation_boundary = engine_config.cache_generation_boundary
```

- [ ] **Step 8: Build**

Run (from `build/`): `ninja`
Expected: links cleanly. (If `build/` is not configured yet: `sh ../my_generate.sh` first.)

- [ ] **Step 9: Commit**

```bash
git add lmdeploy/messages.py lmdeploy/turbomind/turbomind.py src/turbomind/engine/engine_config.h src/turbomind/engine/scheduler.h src/turbomind/engine/scheduler.cc src/turbomind/engine/engine.cc src/turbomind/engine/request.h
git commit -m "feat(turbomind): add cache_prompt_boundary_skip config knob (default 1, legacy boundary)"
```

---

## Task 2: Pure `PlanPromptBoundary` geometry function + unit tests (TDD)

A header-only pure function so the existing Catch2 target can test it with no CMake changes (`test_prefix_trie` already links `engine` and the header is include-only).

**Files:**
- Create: `src/turbomind/engine/prompt_boundary.h`
- Test: `src/turbomind/engine/test_prefix_trie.cc` (append cases)

- [ ] **Step 1: Write the failing test**

Append to `src/turbomind/engine/test_prefix_trie.cc` (also add `#include "src/turbomind/engine/prompt_boundary.h"` near the existing includes at the top):

```cpp
TEST_CASE("PlanPromptBoundary: geometry and guards", "[prompt_boundary]")
{
    const int bs = 8;

    // K=1, last block has >1 token (prompt%bs==3): partial node at prompt_len-1, j==last.
    {
        const auto p = PlanPromptBoundary(/*prompt_len=*/19, bs, /*skip=*/1, /*miss=*/0);
        REQUIRE(p.valid);
        REQUIRE(p.partial);
        REQUIRE(p.pos == 18);       // 19 - 1
        REQUIRE(p.block == 2);      // (18-1)/8
        REQUIRE(p.node_size == 2);  // 18 - 16
    }
    // K=1, last block has exactly 1 token (prompt%bs==1): block-aligned, no partial node.
    {
        const auto p = PlanPromptBoundary(/*prompt_len=*/17, bs, /*skip=*/1, /*miss=*/0);
        REQUIRE(p.valid);
        REQUIRE_FALSE(p.partial);
        REQUIRE(p.pos == 16);       // 17 - 1, block-aligned
        REQUIRE(p.block == 1);      // (16-1)/8
    }
    // K=2, last block has >2 tokens (prompt%bs==3): partial node at prompt_len-2.
    {
        const auto p = PlanPromptBoundary(/*prompt_len=*/19, bs, /*skip=*/2, /*miss=*/0);
        REQUIRE(p.valid);
        REQUIRE(p.partial);
        REQUIRE(p.pos == 17);       // 19 - 2
        REQUIRE(p.block == 2);      // (17-1)/8
        REQUIRE(p.node_size == 1);  // 17 - 16
    }
    // K pushes B into the prior block (prompt%bs==2, K=3): B=18 in block 2, partial.
    {
        const auto p = PlanPromptBoundary(/*prompt_len=*/21, bs, /*skip=*/3, /*miss=*/0);
        REQUIRE(p.valid);
        REQUIRE(p.partial);
        REQUIRE(p.pos == 18);       // 21 - 3
        REQUIRE(p.block == 2);      // (18-1)/8
        REQUIRE(p.node_size == 2);  // 18 - 16
    }
    // Partial node needs st.miss < j: miss at j blocks the node.
    {
        const auto p = PlanPromptBoundary(/*prompt_len=*/19, bs, /*skip=*/1, /*miss=*/2);  // j==2
        REQUIRE_FALSE(p.valid);
    }
    // Block-aligned allows st.miss <= j: miss at j still publishes the clamp target.
    {
        const auto p = PlanPromptBoundary(/*prompt_len=*/17, bs, /*skip=*/1, /*miss=*/1);  // j==1
        REQUIRE(p.valid);
        REQUIRE_FALSE(p.partial);
        REQUIRE(p.pos == 16);
    }
    // Block-aligned PROMPT at K=1 (prompt%bs==0): NOT suppressed -- a matchable
    // boundary is published (option B; old code skipped this).
    {
        const auto p = PlanPromptBoundary(/*prompt_len=*/16, bs, /*skip=*/1, /*miss=*/0);
        REQUIRE(p.valid);
        REQUIRE(p.partial);
        REQUIRE(p.pos == 15);       // 16 - 1
        REQUIRE(p.block == 1);      // (15-1)/8
        REQUIRE(p.node_size == 7);  // 15 - 8
    }
    // Think + full-block case: block-aligned prompt, K=2 -> partial node before
    // the volatile suffix that lives in the last full block.
    {
        const auto p = PlanPromptBoundary(/*prompt_len=*/16, bs, /*skip=*/2, /*miss=*/0);
        REQUIRE(p.valid);
        REQUIRE(p.partial);
        REQUIRE(p.pos == 14);       // 16 - 2
        REQUIRE(p.block == 1);      // (14-1)/8
        REQUIRE(p.node_size == 6);  // 14 - 8
    }
    // B < 1 -> no boundary.
    {
        const auto p = PlanPromptBoundary(/*prompt_len=*/1, bs, /*skip=*/1, /*miss=*/0);
        REQUIRE_FALSE(p.valid);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `build/`): `ninja test_prefix_trie`
Expected: FAIL — compile error, `prompt_boundary.h` / `PlanPromptBoundary` not found.

- [ ] **Step 3: Write the implementation**

Create `src/turbomind/engine/prompt_boundary.h`:

```cpp
// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind {

// Pure geometry of the prompt-boundary publish point. Decides where the reusable
// boundary B = prompt_len - skip lands and whether a partial fork_to node is
// needed (B strictly inside a block) or B is block-aligned (whole blocks already
// tile [0, B); only the clamp/checkpoint applies). `miss` is the first prompt
// block index not matched in the trie (AcceptState::miss). No scheduler state.
struct PromptBoundaryPlan {
    bool valid     = false;  // a boundary exists (set prompt_boundary_node)
    bool partial   = false;  // needs a fork_to partial node; else block-aligned
    int  pos       = 0;      // B
    int  block     = 0;      // j = block holding the last token before B
    int  node_size = 0;      // partial node length (B - j*block_size) when partial
};

inline PromptBoundaryPlan PlanPromptBoundary(int prompt_len, int block_size, int skip, int miss)
{
    PromptBoundaryPlan p{};
    if (skip < 1) {
        skip = 1;  // defensive; the scheduler also clamps at construction
    }
    const int B = prompt_len - skip;
    if (B < 1) {
        return p;  // boundary before the first token: nothing to publish
    }
    const int j = (B - 1) / block_size;  // block holding the last token before B
    if (B % block_size != 0) {
        // B strictly inside block j: a partial fork_to node is required so [0, B)
        // is fully matchable (whole blocks [0, j*bs) + this node). miss < j keeps
        // the parent chain [0..j-1] indexed and the node off the fork_from miss
        // block. miss < j (with miss >= 0) implies j >= 1, so block j-1 exists.
        if (miss < j) {
            p.valid     = true;
            p.partial   = true;
            p.pos       = B;
            p.block     = j;
            p.node_size = B - j * block_size;
        }
    }
    else {
        // B block-aligned: block j ends exactly at B and already tiles [0, B);
        // no partial node, only the clamp target. miss <= j: block j is matched
        // or created-and-indexed.
        if (miss <= j) {
            p.valid   = true;
            p.partial = false;
            p.pos     = B;
            p.block   = j;
        }
    }
    return p;
}

}  // namespace turbomind
```

- [ ] **Step 4: Run test to verify it passes**

Run (from `build/`): `ninja test_prefix_trie && ./bin/test_prefix_trie "[prompt_boundary]"`
Expected: PASS (all `PlanPromptBoundary` assertions). Also run `./bin/test_prefix_trie` (no filter) to confirm the existing `[fingerprint]`/`[prefix_trie]` cases still pass.

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/engine/prompt_boundary.h src/turbomind/engine/test_prefix_trie.cc
git commit -m "test(turbomind): add pure PlanPromptBoundary geometry + unit tests"
```

---

## Task 3: Wire `PlanPromptBoundary` into `SetupForks` and read `prompt_boundary_pos` downstream

This is the behavior change. At `K = 1` the boundary keeps the legacy position `prompt_len-1`: a no-op for partial-tail prompts, and a deliberate improvement for block-aligned prompts (which the old code skipped — see spec §4). `K > 1` moves the boundary back to drop the volatile suffix.

**Files:**
- Modify: `src/turbomind/engine/scheduler.cc` (`SetupForks` `:454-489`; clamp `:1123`; comments `:452,1117,1125,1132,949,960`)
- Modify: `src/turbomind/engine/cache_boundary_policy.cc:17`

- [ ] **Step 1: Replace the `SetupForks` prompt-boundary block**

In `src/turbomind/engine/scheduler.cc`, add the include near the other engine includes at the top of the file:

```cpp
#include "src/turbomind/engine/prompt_boundary.h"
```

Then replace the entire current `if (prompt_boundary) { if (const int last = ...) { ... } }` block (`:454-489`) with:

```cpp
    if (prompt_boundary) {
        const auto plan = PlanPromptBoundary(prompt, bs, cache_prompt_boundary_skip_, st.miss);
        if (plan.valid) {
            bool have_target = true;

            if (plan.partial) {
                const int           j      = plan.block;            // j >= 1 (guaranteed by the planner)
                const LogicalBlock* parent = s.block_ids[j - 1].get();
                const PrefixKey     base   = s.block_ids[j - 1]->key;
                const auto          tokens = TokenSegment(s, j * bs, plan.node_size);

                std::vector<Fingerprint> fps;
                CollectStartFps(s, j * bs, plan.pos, fps);

                const auto    next = ExtendPrefixKey(base, tokens, fps);
                BlockHandle   vh   = logical_.Create(j);
                LogicalBlock& y    = *vh;
                y.parent           = parent;
                y.key              = next;
                y.size             = plan.node_size;
                y.tokens.assign(tokens.begin(), tokens.end());
                y.image_fps        = fps;
                y.prefix_id        = cache_.Create(registry_.prefix().object_id(), vh.get());
                if (trie_.Insert(y)) {
                    s.block_ids[j]->fork_to = std::move(vh);  // edge holds the only ref
                }
                else {
                    LogCollision(s, CollisionSite::kPromptBoundary, j * bs, plan.pos);
                    have_target = false;  // undiscoverable: vh drops at scope end -> recycle
                }
            }

            if (have_target) {
                s.prompt_boundary_node = true;
                s.prompt_boundary_pos  = plan.pos;  // clamp the producer's prefill to B
            }
        }
    }
```

Also update the comment block above it (`:449-453`) to:

```cpp
    // Prompt-boundary publish point (fork_to): only when cache_prompt_boundary is
    // enabled. The node ends at the reusable boundary B = prompt_len - K (K =
    // cache_prompt_boundary_skip, default 1). A partial node is published when B
    // is inside a block (B % bs != 0); a block-aligned B only arms the clamp.
```

(The `const bool prompt_boundary = cache_prompt_boundary_;` line at `:430` is retained and still drives this block. The `full_size` / `node_size` / `last` locals are gone with the old code.)

- [ ] **Step 2: Read `prompt_boundary_pos` in the admission clamp**

In `src/turbomind/engine/scheduler.cc:1123`, change:

```cpp
        const int prompt_boundary_pos = s.prompt_boundary_pos;
```

Update the nearby comments (`:1116-1117`, `:1125`, `:1132`) to refer to `B` / `prompt_boundary_pos` instead of `prompt_len-1`, e.g. `:1132` becomes `desired = prompt_boundary_pos;  // land exactly on B`.

- [ ] **Step 3: Read `prompt_boundary_pos` in the auto boundary policy**

In `src/turbomind/engine/cache_boundary_policy.cc:17`, change the return to:

```cpp
    return prompt_boundary_ && threshold_ > 0 && s.prompt_boundary_pos - s.last_ckpt_pos >= threshold_;
```

- [ ] **Step 4: Update the two publication-site comments**

In `src/turbomind/engine/scheduler.cc`, update comment `:949` from `(caller guarantees end == prompt_len-1)` to `(caller guarantees end == B == prompt_boundary_pos)`, and comment `:960` `single-tail-token prompt (prompt_len-1 is a block boundary)` to `block-aligned B is a block boundary`. (Comment-only; the `(end-1)/bs` / `offset+size` logic is already geometry-driven and unchanged.)

- [ ] **Step 4b: Fix `LogAccept` to find `fork_to` on the boundary block**

`LogAccept` (`scheduler.cc:1442`) logs the published node by reading `s.block_ids.back()->fork_to`. With generalized placement the node can live on block `j = (B-1)/bs` which is **not** the last block (e.g. `K > 1`, or a partial node followed by volatile-suffix blocks). The current code would silently drop the `fork_to@...` log line in those cases. Locate the block from `prompt_boundary_pos`:

```cpp
    if (s.prompt_boundary_pos > 0) {
        const int j = (s.prompt_boundary_pos - 1) / bs;  // block holding B (matches PlanPromptBoundary)
        if (j >= 0 && j < (int)s.block_ids.size() && s.block_ids[j]->fork_to) {
            const LogicalBlock& ft = *s.block_ids[j]->fork_to;
            ctail = fmt::format(", fork_to@{}", ft.offset + ft.size);  // created-side publish node end
        }
    }
```

This is logging-only (no scheduling effect), but keeps the diagnostic correct for `K > 1` and block-aligned-prompt boundaries. Note block-aligned `B` publishes no partial node (only a checkpoint), so `ctail` stays empty there — expected.

- [ ] **Step 5: Build**

Run (from `build/`): `ninja`
Expected: compiles and links cleanly.

- [ ] **Step 6: Model-level regression (default K=1 unchanged)**

First check for an empty GPU (`get_gpu_usage` MCP tool, or `nvidia-smi`), and run OUTSIDE the sandbox. Pick a recurrent/hybrid prefix-caching model (e.g. `Qwen/Qwen3.5-27B`, cache dir `/mnt_cfs/huggingface_hub/hub/`; confirm availability via the model-server MCP `list_models` or `/data/models.json`). Run `scripts/test_turbomind_model.py` AS IS:

```bash
python scripts/test_turbomind_model.py \
  --model-id Qwen/Qwen3.5-27B \
  --cache-dir /mnt_cfs/huggingface_hub/hub/ \
  --tp 1 --gpus 0 \
  --enable-prefix-caching \
  --cache-prompt-boundary \
  --max-new-tokens 128
```

Expected: a coherent, on-topic response of ≥128 tokens (gibberish = bug). Verify the response is meaningful human text. At the default `cache_prompt_boundary_skip = 1` the boundary stays at `prompt_len-1`; output must remain correct/coherent. Note this is a *correctness* check, not bit-for-bit log equality: block-aligned prompts now publish an extra `fork_to` boundary (intended, see spec §4), so a `fork_to@...` line may appear where the prior commit had none — the generated text must still be coherent and on-topic.

- [ ] **Step 7: Commit**

```bash
git add src/turbomind/engine/scheduler.cc src/turbomind/engine/cache_boundary_policy.cc
git commit -m "feat(turbomind): publish prompt-boundary node at prompt_len - cache_prompt_boundary_skip"
```

---

## Task 4: Contract sync (`src/turbomind/engine/README.md`)

Required by `checklist.contract-sync`. Content-only edits; do not re-wrap prose.

**Files:**
- Modify: `src/turbomind/engine/README.md` (`contracts.prefix-prepare`, `contracts.scheduler-commit`, `resume-selection`, `concepts.boundary-policy`, `concepts`)

- [ ] **Step 1: Edit `contracts.prefix-prepare`**

Find the sentence: "The `fork_to` node excludes the last prompt token and ends at `prompt_len-1` (the reusable position under the `seq_len-1` resume cap)." Replace with:

```
The `fork_to` node ends at the reusable boundary `B = prompt_len - cache_prompt_boundary_skip` (the configured count of trailing volatile generation-prompt tokens, default 1, so the default excludes only the last prompt token; `B` is capped by the `seq_len-1` resume cap). A partial `fork_to` node is published only when `B` falls inside a block (`B % block_size != 0`); when `B` is block-aligned the whole-block prefix already tiles `[0, B)` and only the boundary clamp/checkpoint applies. Over-excluding (a larger skip) is safe: segment tokens are exact-compared, so a too-long suffix only shortens reuse and never causes a false hit.
```

- [ ] **Step 2: Edit `contracts.scheduler-commit`**

In the clamp clause, replace `or exactly prompt_len-1 when prompt_boundary_node is set` with `or exactly B = prompt_len - cache_prompt_boundary_skip when prompt_boundary_node is set`. Apply the same `prompt_len-1` → `B = prompt_len - cache_prompt_boundary_skip` substitution in the inline code comment that follows (`r.input_len = admitted;  // clamped ...`).

- [ ] **Step 3: Edit `resume-selection`**

Replace "a duplicate prompt resumes via fork-extension at `prompt_len-1` (the producer's prompt-boundary node end)" with "a duplicate (or history-extending) prompt resumes via fork-extension at the producer's prompt-boundary node end `B = prompt_len - cache_prompt_boundary_skip`".

- [ ] **Step 4: Edit `concepts.boundary-policy`**

Append one sentence: "The prompt-boundary predicate measures the interval to `B` (`Sequence::prompt_boundary_pos`), not `prompt_len-1`."

- [ ] **Step 5: Edit `concepts`**

Add a one-line note (next to `multimodal-spans` or after `boundary-policy`): "`cache_prompt_boundary_skip` is the engine knob for the trailing volatile-suffix length; `Sequence::prompt_boundary_pos` is its per-sequence resolved boundary `B = prompt_len - cache_prompt_boundary_skip`."

- [ ] **Step 6: Commit**

```bash
git add src/turbomind/engine/README.md
git commit -m "docs(turbomind): sync engine contract for configurable prompt-boundary skip"
```

---

## Task 5: Multi-turn reuse verification (K>1 actually recovers the partial tail)

End-to-end confirmation that `K` matching the think-suffix length recovers the partial-tail reuse the next turn. GPU work — run outside the sandbox after checking for a free GPU.

**Files:**
- Reference: `scripts/vlm_prefix_cache_check.py` (already drives `cache_prompt_boundary=True`)
- Create: `scripts/prompt_boundary_skip_check.py` (text-only adaptation)

- [ ] **Step 1: Write the check harness**

Create `scripts/prompt_boundary_skip_check.py`, modeled on `scripts/vlm_prefix_cache_check.py`'s engine setup (`cache_prompt_boundary=True`, `enable_prefix_caching=True`, `log_level='INFO'`, `do_sample=False`). It must:

1. Set `cache_prompt_boundary_skip` to the target thinking template's think-suffix token count (compute once: tokenize the chat-template output of `[{'role':'user','content':'hi'}]` with `add_generation_prompt=True` and the same with a sentinel assistant turn; the token-count difference is the suffix length — used ONLY to choose the test config value, not in production code).
2. Send turn 1 (`[user]`), capture the assistant answer, then build turn 2 history with the thinking block stripped (`[user, assistant(answer), user2]`) and send it **sequentially** through the same engine.
3. Parse the engine INFO logs (`LogAccept` `matched [0,M) ... fork_to@<pos>`, `LogResume` `... source=...`) for turn 2.

- [ ] **Step 2: Run with `skip = 1` (baseline) and `skip = suffix_len`**

Run the harness twice (free GPU, outside sandbox), once with `cache_prompt_boundary_skip=1` and once with `cache_prompt_boundary_skip=<suffix_len>`.
Expected: with `skip=<suffix_len>`, turn 2's `fork_to@<pos>` / matched span reaches one partial block further (the recovered `<think>` tokens) than with `skip=1`, and `LogResume` reports `source=prefix`/`fork`. With `skip=1`, turn 2 falls back to the last whole-block boundary (the current-bug behavior).

- [ ] **Step 3: Oracle — output identity**

Confirm turn 2's greedy (`do_sample=False`) output token-ids are identical between the `skip=1` and `skip=<suffix_len>` runs (reuse must not change results — cached KV are the exact tensors). Any divergence is a bug; iterate on the implementation until it holds.

- [ ] **Step 4: Commit**

```bash
git add scripts/prompt_boundary_skip_check.py
git commit -m "test(turbomind): add multi-turn prompt-boundary-skip reuse check"
```

---

## Self-Review

**Spec coverage:**
- Spec §2 (config knob) → Task 1 (steps 1, 4, 6, 7).
- Spec §3 (`prompt_boundary_pos` field) → Task 1 step 2; read in Task 3 steps 2-3.
- Spec §4 (`SetupForks` generalization, `PlanPromptBoundary`, `full_size` removal) → Task 2 (function) + Task 3 step 1.
- Spec §5 (downstream sites) → Task 3 steps 2-4.
- Spec §6 (contract sync) → Task 4.
- Spec §7(a) (unit-testable arithmetic) → Task 2; §7(b) (model regression) → Task 3 step 6; §7(c) (multi-turn check) → Task 5.

**Placeholder scan:** none — every code step shows full code; the only "compute the suffix length" instruction (Task 5 step 1) is test-config selection, explicitly excluded from production code per the spec.

**Type consistency:** `PromptBoundaryPlan` fields (`valid`, `partial`, `pos`, `block`, `node_size`) and `PlanPromptBoundary(prompt_len, block_size, skip, miss)` are identical across Task 2 (definition + tests) and Task 3 (call site). The `Scheduler` ctor param order (`cache_prompt_boundary, cache_prompt_boundary_skip, cache_generation_boundary`) matches across `scheduler.h`, `scheduler.cc`, and `engine.cc`. `cache_prompt_boundary_skip` name is identical across Python, `EngineConfig`, ctor, and member `cache_prompt_boundary_skip_`.
