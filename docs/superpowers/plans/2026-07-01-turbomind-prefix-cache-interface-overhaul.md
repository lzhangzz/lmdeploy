# TurboMind Prefix-Cache Interface Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace TurboMind's `cache_prompt_boundary`/`cache_generation_boundary`/`cache_boundary_policy` booleans + `linear_prefix_cache_min_interval` with two string modes (`cache_prompt` `'all'|'auto'`, `cache_generation` `'all'|'auto'|'none'`) and a renamed `cache_checkpoint_interval` (default 4096), adding image-aware `'auto'` prompt caching and deleting the `CacheBoundaryPolicy` machinery.

**Architecture:** Strings flow from `TurbomindEngineConfig` through the reflected `EngineConfig` struct (auto-bound to Python via `bind_struct`) into the `Scheduler`, which parses each once into a `CacheMode` enum. Publication is decided entirely in `SetupForks`/`PublishGeneration` from the modes (no runtime policy object). `'auto'` prompt caching publishes the partial `fork_to` node only when its token range overlaps a multimodal span.

**Tech Stack:** C++17 (TurboMind engine), CMake+Ninja build, Catch2 unit tests (`test_prefix_trie`), pybind11 (`_turbomind` extension), Python dataclasses (`lmdeploy`).

**Design spec:** `docs/superpowers/specs/2026-07-01-turbomind-prefix-cache-interface-overhaul-design.md`

**Build reminder (from AGENTS.md):** configure once with `sh ../my_generate.sh` from `build/`, then `ninja` (or a target) from `build/`. GPU/model runs must happen outside the sandbox on an empty GPU (`get_gpu_usage`). Never install lmdeploy or run `setup.py`.

---

## Task 1: `CacheMode` enum, parser, and pure publish-gate helper

Introduces the header that replaces `CacheBoundaryPolicy`, plus a pure, unit-testable publish-gate function. TDD: the Catch2 test is written first against the not-yet-existing header.

**Files:**
- Create: `src/turbomind/engine/cache_mode.h`
- Modify (test): `src/turbomind/engine/test_prefix_trie.cc` (append a new `TEST_CASE`)

- [ ] **Step 1: Write the failing test**

Append to `src/turbomind/engine/test_prefix_trie.cc` (add `#include "src/turbomind/engine/cache_mode.h"` near the other engine includes at the top of the file):

```cpp
TEST_CASE("ParseCacheMode maps strings to CacheMode", "[cache_mode]")
{
    using turbomind::CacheMode;
    using turbomind::ParseCacheMode;
    CHECK(ParseCacheMode("none") == CacheMode::kNone);
    CHECK(ParseCacheMode("auto") == CacheMode::kAuto);
    CHECK(ParseCacheMode("all") == CacheMode::kAll);
}

TEST_CASE("DecidePromptBoundaryPublish gates by mode/partial/image", "[cache_mode]")
{
    using turbomind::CacheMode;
    using turbomind::DecidePromptBoundaryPublish;

    // Partial node (B mid-block): 'all' always publishes; 'auto' only with image.
    CHECK(DecidePromptBoundaryPublish(CacheMode::kAll, /*partial=*/true, /*has_image=*/false));
    CHECK(DecidePromptBoundaryPublish(CacheMode::kAll, true, true));
    CHECK_FALSE(DecidePromptBoundaryPublish(CacheMode::kAuto, true, false));
    CHECK(DecidePromptBoundaryPublish(CacheMode::kAuto, true, true));

    // Block-aligned B (no partial node): only 'all' arms the checkpoint clamp.
    CHECK(DecidePromptBoundaryPublish(CacheMode::kAll, /*partial=*/false, /*has_image=*/false));
    CHECK_FALSE(DecidePromptBoundaryPublish(CacheMode::kAuto, false, false));
    CHECK_FALSE(DecidePromptBoundaryPublish(CacheMode::kAuto, false, true));
}
```

- [ ] **Step 2: Build the test target to verify it fails**

Run (from `build/`): `ninja test_prefix_trie`
Expected: FAIL — compile error, `cache_mode.h: No such file or directory` / `ParseCacheMode` not declared.

- [ ] **Step 3: Create `src/turbomind/engine/cache_mode.h`**

```cpp
// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <string>

#include "src/turbomind/core/logger.h"  // TM_LOG_FATAL

namespace turbomind {

// Per-side prefix-cache publication mode. cache_prompt uses {kAuto, kAll}
// (no kNone); cache_generation uses all three.
enum class CacheMode
{
    kNone,
    kAuto,
    kAll
};

// String -> CacheMode. cache_prompt never receives "none" (rejected by the
// Python TurbomindEngineConfig.__post_init__ assert); the shared parser still
// accepts it for cache_generation. TM_LOG_FATAL is [[noreturn]] (std::abort),
// so no trailing return is needed.
inline CacheMode ParseCacheMode(const std::string& s)
{
    if (s == "none") {
        return CacheMode::kNone;
    }
    if (s == "auto") {
        return CacheMode::kAuto;
    }
    if (s == "all") {
        return CacheMode::kAll;
    }
    TM_LOG_FATAL("invalid cache mode: {}", s);
}

// Pure prompt-boundary publish decision, given the mode, whether the geometric
// plan wants a partial fork_to node (else block-aligned checkpoint clamp), and
// whether that partial node's token range holds image tokens. 'all' publishes a
// partial node whenever the plan is partial and arms the clamp when
// block-aligned; 'auto' publishes only an image-bearing partial node and never
// arms the block-aligned clamp.
inline bool DecidePromptBoundaryPublish(CacheMode prompt_mode, bool plan_partial, bool has_image_in_node)
{
    if (plan_partial) {
        return prompt_mode == CacheMode::kAll
               || (prompt_mode == CacheMode::kAuto && has_image_in_node);
    }
    return prompt_mode == CacheMode::kAll;
}

}  // namespace turbomind
```

- [ ] **Step 4: Build the test target and run it to verify it passes**

Run (from `build/`): `ninja test_prefix_trie && ./bin/test_prefix_trie "[cache_mode]"`
Expected: PASS (both `[cache_mode]` cases). (If the binary path differs, use the path `ninja` reports; the target name is `test_prefix_trie`.)

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/engine/cache_mode.h src/turbomind/engine/test_prefix_trie.cc
git commit -m "feat(turbomind): add CacheMode enum, parser, and publish-gate helper"
```

---

## Task 2: `EngineConfig` fields + checkpoint-interval wiring (C++)

Swap the four removed reflected fields for the three new ones and wire `cache_checkpoint_interval` straight into the registry (dropping the `0 => block_seq_len` fallback). `cache_prompt_boundary_skip` stays.

**Files:**
- Modify: `src/turbomind/engine/engine_config.h:26-30`
- Modify: `src/turbomind/turbomind.cc:299-301`

- [ ] **Step 1: Replace the reflected fields in `engine_config.h`**

Replace lines 26-30 (`linear_prefix_cache_min_interval`, `cache_prompt_boundary`, `cache_prompt_boundary_skip`, `cache_generation_boundary`, `cache_boundary_policy`) with:

```cpp
    X(int, cache_checkpoint_interval, 4096)                                                                            \
    X(std::string, cache_prompt, "auto")                                                                              \
    X(int, cache_prompt_boundary_skip, 1)                                                                             \
    X(std::string, cache_generation, "auto")                                                                          \
```

(Net: `linear_prefix_cache_min_interval` → `cache_checkpoint_interval`, `cache_prompt_boundary`(bool) → `cache_prompt`(str), `cache_generation_boundary`(bool) → `cache_generation`(str); `cache_boundary_policy` deleted; `cache_prompt_boundary_skip` unchanged. Keep the trailing backslashes aligned with the surrounding `ENGINE_FIELDS` macro.)

- [ ] **Step 2: Wire `cache_checkpoint_interval` in `turbomind.cc`**

Replace lines 299-301:

```cpp
    cache_registry.set_checkpoint_min_interval(param.linear_prefix_cache_min_interval > 0 ?
                                                   param.linear_prefix_cache_min_interval :
                                                   param.cache_block_seq_len);
```

with:

```cpp
    cache_registry.set_checkpoint_min_interval(param.cache_checkpoint_interval);
```

- [ ] **Step 3: Verify (deferred build)**

This task does not build cleanly on its own because `scheduler`/`engine` still reference the removed fields. Do **not** build yet; the build check happens at the end of Task 3. Just re-read the two edits to confirm no stray references to `linear_prefix_cache_min_interval` / `cache_prompt_boundary` / `cache_generation_boundary` / `cache_boundary_policy` remain in these two files:

Run: `rg -n "linear_prefix_cache_min_interval|cache_prompt_boundary\b|cache_generation_boundary|cache_boundary_policy" src/turbomind/engine/engine_config.h src/turbomind/turbomind.cc`
Expected: no matches.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/engine/engine_config.h src/turbomind/turbomind.cc
git commit -m "refactor(turbomind): EngineConfig cache_prompt/cache_generation/cache_checkpoint_interval"
```

---

## Task 3: Scheduler modes, image-gated publish, and CacheBoundaryPolicy removal (C++)

The core engine change. All edits land together so the `engine` library compiles. This deletes `cache_boundary_policy.{h,cc}`, the `boundary_policy_` member, `ResolvePublishPromptBoundary`, and `Sequence::prompt_boundary_publish`.

**Files:**
- Modify: `src/turbomind/engine/scheduler.h:94-98` (ctor params), `:205` (decl), `:224-227` (members), and the `#include` list
- Modify: `src/turbomind/engine/scheduler.cc:254-268` (ctor), `:418-493` (`SetupForks`), `:935-946` (`ResolvePublishPromptBoundary`), `:1122-1128` (admission clamp), the `CollectStartFps` neighborhood (add `HasMultimodalOverlap`), and the `#include` list
- Modify: `src/turbomind/engine/engine.cc:17` (`#include`), `:230-233` (ctor args)
- Modify: `src/turbomind/engine/request.h:245-250` (remove `prompt_boundary_publish`)
- Modify: `src/turbomind/engine/CMakeLists.txt:8` (drop `cache_boundary_policy.cc`)
- Delete: `src/turbomind/engine/cache_boundary_policy.h`, `src/turbomind/engine/cache_boundary_policy.cc`

- [ ] **Step 1: Delete the policy files**

```bash
git rm src/turbomind/engine/cache_boundary_policy.h src/turbomind/engine/cache_boundary_policy.cc
```

- [ ] **Step 2: Drop `cache_boundary_policy.cc` from `CMakeLists.txt`**

In `src/turbomind/engine/CMakeLists.txt`, delete line 8 (`    cache_boundary_policy.cc`) from the `add_library(engine STATIC ...)` source list.

- [ ] **Step 3: Update `scheduler.h` includes + ctor + members + decl**

(a) Replace the include (line 11) `#include "src/turbomind/engine/cache_boundary_policy.h"` with:

```cpp
#include "src/turbomind/engine/cache_mode.h"
```

(b) Replace the ctor params (lines 94-98):

```cpp
              bool                                 cache_prompt_boundary,
              int                                  cache_prompt_boundary_skip,
              bool                                 cache_generation_boundary,
              std::unique_ptr<CacheBoundaryPolicy> boundary_policy,
              const int&                           is_warm_up);
```

with:

```cpp
              const std::string&                   cache_prompt,
              int                                  cache_prompt_boundary_skip,
              const std::string&                   cache_generation,
              const int&                           is_warm_up);
```

(c) Delete the `ResolvePublishPromptBoundary` declaration (line 205):

```cpp
    bool          ResolvePublishPromptBoundary(Sequence& s);
```

(d) Add a declaration for the new pure predicate next to `PrefixEligible`/`TokenSegment` (near line 218):

```cpp
    // True if any multimodal span overlaps [lo, hi). Pure; used by SetupForks
    // to gate the 'auto' prompt-boundary publish.
    static bool HasMultimodalOverlap(const Sequence& s, int lo, int hi);
```

(e) Replace the boundary members (lines 224-227):

```cpp
    bool                                 cache_prompt_boundary_{false};
    int                                  cache_prompt_boundary_skip_{1};
    bool                                 cache_generation_boundary_{false};
    std::unique_ptr<CacheBoundaryPolicy> boundary_policy_;
```

with:

```cpp
    CacheMode                            prompt_cache_mode_{CacheMode::kAuto};
    int                                  cache_prompt_boundary_skip_{1};
    CacheMode                            generation_cache_mode_{CacheMode::kAuto};
```

- [ ] **Step 4: Update the `#include` in `scheduler.cc`**

Replace (line 13) `#include "src/turbomind/engine/prompt_boundary.h"`'s neighborhood include of the policy header. Specifically remove any `#include "src/turbomind/engine/cache_boundary_policy.h"` line and ensure this include is present near the other engine includes:

```cpp
#include "src/turbomind/engine/cache_mode.h"
```

(`prompt_boundary.h` — which declares `PlanPromptBoundary`/`PromptBoundaryPlan` — stays.)

- [ ] **Step 5: Update the `Scheduler` constructor in `scheduler.cc` (lines 254-277)**

Replace the ctor signature params + initializers so the two strings are parsed and the policy member is gone:

```cpp
Scheduler::Scheduler(ObjectAllocator&                     alloc,
                     CacheRegistry                        registry,
                     int                                  cache_block_seq_len,
                     bool                                 enable_prefix_caching,
                     const std::string&                   cache_prompt,
                     int                                  cache_prompt_boundary_skip,
                     const std::string&                   cache_generation,
                     const int&                           is_warm_up):
    enable_prefix_caching_{enable_prefix_caching},
    prompt_cache_mode_{ParseCacheMode(cache_prompt)},
    cache_prompt_boundary_skip_{cache_prompt_boundary_skip < 1 ? 1 : cache_prompt_boundary_skip},
    generation_cache_mode_{ParseCacheMode(cache_generation)},
    is_warm_up_{is_warm_up},
    alloc_{alloc},
    registry_{std::move(registry)},
    logical_{cache_, cache_block_seq_len},
    trie_{cache_block_seq_len},
    accum_{make_perf_counter()},
    interv_{make_perf_counter()}
{
    logical_.set_recycle_hook([this](LogicalBlock& b) { trie_.Erase(b); });
}
```

- [ ] **Step 6: Add `HasMultimodalOverlap` definition in `scheduler.cc`**

Add this free-standing member definition directly after the `CollectStartFps` definition (just before `enum class CollisionSite` at line ~185):

```cpp
// True if any multimodal span overlaps [lo, hi). Interval is the absolute token
// span [begin, end); a partial prompt block "contains image tokens" when a span
// intersects it, even one that started in an earlier (full) block and extends
// in. multimodal_spans is prompt-ordered ascending by interval.begin().
bool Scheduler::HasMultimodalOverlap(const Sequence& s, int lo, int hi)
{
    for (const auto& sp : s.multimodal_spans) {
        if (sp.interval.begin() >= hi) {
            break;  // ascending; no later span can overlap
        }
        if (sp.interval.end() > lo) {
            return true;
        }
    }
    return false;
}
```

- [ ] **Step 7: Rewrite `SetupForks` (scheduler.cc:418-493)**

Replace the whole `SetupForks` body. `prompt_cache_mode_` is always `kAuto`/`kAll` (no `kNone`), so `fork_from` and the prompt-boundary attempt are always armed; the publish is gated by `DecidePromptBoundaryPublish`:

```cpp
void Scheduler::SetupForks(Sequence& s, AcceptState& st)
{
    const int bs     = logical_.block_size();
    const int prompt = s.prompt_len;

    const int all_blocks = (prompt + bs - 1) / bs;

    // fork_from (read side) is always armed: any prior request may have published
    // a prompt partial node (cache_prompt in {all, auto}) or a generation
    // terminal partial ('all'), so the read edge must always try to match.
    if (st.miss < all_blocks) {
        LogicalBlock& x      = *s.block_ids[st.miss];
        const int     offset = st.miss * bs;
        const int     size   = std::min(prompt - offset, bs);
        PrefixKey     k      = st.miss_key;

        std::vector<Fingerprint> fps;
        std::vector<int>         fp_pos;
        CollectStartFps(s, offset, offset + size, fps, &fp_pos);

        if (LogicalBlock* v = trie_.Search(st.miss_parent, k, TokenSegment(s, offset, size), fps, fp_pos)) {
            x.fork_from = BlockHandle{v};  // edge ref
        }
    }

    // Prompt-boundary publish point (fork_to). B = prompt_len - K (K =
    // cache_prompt_boundary_skip). 'all' publishes a partial node whenever B is
    // mid-block and arms the checkpoint clamp when B is block-aligned. 'auto'
    // publishes the partial node only when its own token range [j*bs, B) holds
    // image tokens, and never arms the block-aligned clamp.
    const auto plan = PlanPromptBoundary(prompt, bs, cache_prompt_boundary_skip_, st.miss);
    if (plan.valid) {
        const bool need_image = plan.partial && prompt_cache_mode_ == CacheMode::kAuto;
        const bool has_image  = need_image && HasMultimodalOverlap(s, plan.block * bs, plan.pos);

        if (DecidePromptBoundaryPublish(prompt_cache_mode_, plan.partial, has_image)) {
            bool have_target = true;

            if (plan.partial) {
                const int     j      = plan.block;  // j >= 1 (guaranteed by the planner)
                LogicalBlock& x      = *s.block_ids[j];
                const auto    tokens = TokenSegment(s, j * bs, plan.node_size);

                std::vector<Fingerprint> fps;
                CollectStartFps(s, j * bs, j * bs + plan.node_size, fps);

                const auto    next = ExtendPrefixKey(s.block_ids[j - 1]->key, tokens, fps);
                BlockHandle   vh   = logical_.Create(j);
                LogicalBlock& y    = *vh;
                y.parent           = s.block_ids[j - 1].get();
                y.key              = next;
                y.size             = plan.node_size;
                y.tokens.assign(tokens.begin(), tokens.end());
                y.image_fps        = fps;
                y.prefix_id        = cache_.Create(registry_.prefix().object_id(), vh.get());
                if (trie_.Insert(y)) {
                    x.fork_to = std::move(vh);  // edge holds the only ref
                }
                else {
                    LogCollision(s, CollisionSite::kPromptBoundary, j * bs, j * bs + plan.node_size);
                    have_target = false;  // undiscoverable: vh drops at scope end -> recycle
                }
            }

            if (have_target) {
                s.prompt_boundary_node = true;
                s.prompt_boundary_pos  = plan.pos;  // clamp the producer's prefill to B
            }
        }
    }
}
```

- [ ] **Step 8: Delete `ResolvePublishPromptBoundary` (scheduler.cc:935-946)**

Delete the whole function (the comment block at 935-936 plus the function body 937-946):

```cpp
// Runtime prompt-boundary publish veto, resolved once and cached (so a retried
// admission pass never re-scans). Sole caller: the admission clamp.
bool Scheduler::ResolvePublishPromptBoundary(Sequence& s)
{
    if (!s.prompt_boundary_node) {
        return false;  // machinery off
    }
    if (s.prompt_boundary_publish < 0) {
        s.prompt_boundary_publish = boundary_policy_->PublishPromptBoundary(s) ? 1 : 0;
    }
    return s.prompt_boundary_publish == 1;
}
```

- [ ] **Step 9: Simplify the admission clamp (scheduler.cc:1122-1135)**

Replace lines 1122-1135:

```cpp
        const int prompt_boundary_pos = s.prompt_boundary_pos;

        // Consult the policy only on the pass that can reach B; >= so an
        // exact landing isn't truncated away. On a veto, skip the clamp.
        const bool is_prompt_boundary =
            s.prompt_boundary_node && begin < prompt_boundary_pos && desired >= prompt_boundary_pos;
        const bool publish_prompt = is_prompt_boundary && ResolvePublishPromptBoundary(s);

        if (publish_prompt) {
            desired = prompt_boundary_pos;  // land exactly on B
        }
        else if (desired < ctx_end) {  // partial chunk: truncate to a block boundary
            desired = desired / bs * bs;
        }
```

with (the publish decision is now final at SetupForks, so no policy re-check):

```cpp
        const int prompt_boundary_pos = s.prompt_boundary_pos;

        // The publish decision is finalized in SetupForks (prompt_boundary_node);
        // the clamp fires on the pass that can reach B (>= so an exact landing
        // isn't truncated away).
        const bool publish_prompt =
            s.prompt_boundary_node && begin < prompt_boundary_pos && desired >= prompt_boundary_pos;

        if (publish_prompt) {
            desired = prompt_boundary_pos;  // land exactly on B
        }
        else if (desired < ctx_end) {  // partial chunk: truncate to a block boundary
            desired = desired / bs * bs;
        }
```

- [ ] **Step 10: Gate `PublishGeneration` by `generation_cache_mode_` (scheduler.cc:773-780)**

Insert the `kNone` early-return after the existing `PrefixEligible`/`filled_len` guard, and replace the `publish_generation_boundary` computation:

```cpp
void Scheduler::PublishGeneration(Sequence& s)
{
    if (!PrefixEligible(s) || s.filled_len <= 0) {
        return;
    }
    if (generation_cache_mode_ == CacheMode::kNone) {
        return;  // index no generated blocks at all
    }

    // 'all' indexes the terminal partial block + adopts the terminal recurrent
    // frontier checkpoint; 'auto' indexes full generated blocks only.
    const bool publish_generation_boundary = (generation_cache_mode_ == CacheMode::kAll);
    ...
```

(Replace only the old `const bool publish_generation_boundary = cache_generation_boundary_ && boundary_policy_->PublishGenerationBoundary(s);` line; leave the rest of the function unchanged.)

- [ ] **Step 11: Remove `Sequence::prompt_boundary_publish` (request.h:250)**

Delete line 250 and fold its meaning away; the two surviving fields keep their comments. Replace lines 245-250:

```cpp
    bool prompt_boundary_node = false;  // a reusable prompt-boundary exists (partial fork_to node when B is mid-block;
                                        // checkpoint-only when B is block-aligned); when the boundary policy admits the
                                        // publish, the producer clamps its forward to prompt_boundary_pos to populate the
                                        // node's KV (and publish a checkpoint when the model is recurrent)
    int  prompt_boundary_pos = 0;       // resolved boundary B = prompt_len - cache_prompt_boundary_skip; 0 = none
    int prompt_boundary_publish = -1;   // cached CacheBoundaryPolicy decision: -1 undecided, 0 no, 1 yes
```

with:

```cpp
    bool prompt_boundary_node = false;  // a reusable prompt-boundary exists and WILL be published: a partial fork_to
                                        // node when B is mid-block, else a block-aligned checkpoint clamp target. The
                                        // producer clamps its forward to prompt_boundary_pos to populate the node's KV
                                        // (and publish a checkpoint when the model is recurrent). Decided in SetupForks.
    int  prompt_boundary_pos = 0;       // resolved boundary B = prompt_len - cache_prompt_boundary_skip; 0 = none
```

- [ ] **Step 12: Update `engine.cc` (line 17 include + lines 230-233 ctor args)**

(a) Remove line 17 `#include "src/turbomind/engine/cache_boundary_policy.h"`.

(b) Replace the scheduler ctor args (lines 230-233):

```cpp
               param_.cache_prompt_boundary,
               param_.cache_prompt_boundary_skip,
               param_.cache_generation_boundary,
               CreateCacheBoundaryPolicy(param_),
```

with:

```cpp
               param_.cache_prompt,
               param_.cache_prompt_boundary_skip,
               param_.cache_generation,
```

- [ ] **Step 13: Build the engine + extension and run the unit test**

Run (from `build/`): `ninja _turbomind test_prefix_trie`
Expected: PASS (clean compile + link). Then run: `./bin/test_prefix_trie "[cache_mode]"` → PASS.

If compilation surfaces any lingering reference to the removed symbols, fix it. Sanity grep:

Run: `rg -n "cache_boundary_policy|CacheBoundaryPolicy|boundary_policy_|ResolvePublishPromptBoundary|prompt_boundary_publish|cache_prompt_boundary_\b|cache_generation_boundary_|linear_prefix_cache_min_interval" src/turbomind`
Expected: no matches.

- [ ] **Step 14: Commit**

```bash
git add -A src/turbomind
git commit -m "refactor(turbomind): scheduler CacheMode modes + image-gated auto prompt boundary; drop CacheBoundaryPolicy"
```

---

## Task 4: Python `TurbomindEngineConfig` + glue

Expose the new interface in the Python dataclass and the pybind glue. The `EngineConfig` fields are auto-bound via `bind_struct`, so no pybind edits are needed — but the `_turbomind` extension from Task 3 must be built for `ec.cache_prompt`/`ec.cache_generation`/`ec.cache_checkpoint_interval` to exist.

**Files:**
- Modify: `lmdeploy/messages.py:248-279` (docstrings), `:329-334` (fields), `:368-369` (validation)
- Modify: `lmdeploy/turbomind/turbomind.py:241-245`

- [ ] **Step 1: Replace the dataclass fields (`messages.py:329-334`)**

Replace:

```python
    enable_prefix_caching: bool = False
    linear_prefix_cache_min_interval: int = 0
    cache_prompt_boundary: bool = False
    cache_prompt_boundary_skip: int = 1
    cache_generation_boundary: bool = False
    cache_boundary_policy: str = ''
```

with:

```python
    enable_prefix_caching: bool = False
    cache_checkpoint_interval: int = 4096
    cache_prompt: str = 'auto'
    cache_prompt_boundary_skip: int = 1
    cache_generation: str = 'auto'
```

- [ ] **Step 2: Replace the docstrings (`messages.py:250-279`)**

Replace the docstring block for `linear_prefix_cache_min_interval` … `cache_boundary_policy` (lines 250-279) with:

```
        cache_checkpoint_interval: minimum token gap between reusable
            recurrent-state checkpoints (CacheRegistry checkpoint_min_interval).
            Must be > 0. Default 4096.
        cache_prompt: partial prompt-boundary publication mode, one of
            'all' | 'auto'. 'all' publishes the reusable partial fork_to node at
            B = prompt_len - cache_prompt_boundary_skip whenever B is mid-block
            (and arms a recurrent-state checkpoint clamp when B is block-aligned),
            so a duplicate prompt skips prefill (costs one extra prefill forward +
            a partial block). 'auto' (default) does that only when the partial
            block holds image tokens (reusing vision-encoded KV) and is inert for
            text-only prompts. Requires enable_prefix_caching.
        cache_prompt_boundary_skip: number of trailing prompt tokens treated as
            the volatile generation-prompt suffix (e.g. a chat template's
            `<think>\n`) and excluded from the reusable prompt-boundary node, so
            the node ends at prompt_len - cache_prompt_boundary_skip. Default 1
            (exclude only the last token). Applies when cache_prompt is 'all' or
            'auto'.
        cache_generation: generated-block caching mode, one of
            'all' | 'auto' | 'none'. 'all' indexes full generated blocks and the
            terminal partial block, and adopts the terminal recurrent frontier
            checkpoint (exact multi-turn resume, costs a partial block). 'auto'
            (default) indexes full generated blocks only. 'none' indexes no
            generated blocks at all. Requires enable_prefix_caching.
```

- [ ] **Step 3: Replace the validation (`messages.py:368-369`)**

Replace:

```python
        assert self.linear_prefix_cache_min_interval >= 0, \
            'invalid linear_prefix_cache_min_interval'
```

with:

```python
        assert self.cache_prompt in ('all', 'auto'), 'invalid cache_prompt'
        assert self.cache_generation in ('all', 'auto', 'none'), 'invalid cache_generation'
        assert self.cache_checkpoint_interval > 0, 'invalid cache_checkpoint_interval'
        assert self.cache_prompt_boundary_skip >= 1, 'invalid cache_prompt_boundary_skip'
```

- [ ] **Step 4: Update the pybind glue (`turbomind.py:241-245`)**

Replace:

```python
        ec.linear_prefix_cache_min_interval = engine_config.linear_prefix_cache_min_interval
        ec.cache_prompt_boundary = engine_config.cache_prompt_boundary
        ec.cache_prompt_boundary_skip = engine_config.cache_prompt_boundary_skip
        ec.cache_generation_boundary = engine_config.cache_generation_boundary
        ec.cache_boundary_policy = engine_config.cache_boundary_policy
```

with:

```python
        ec.cache_checkpoint_interval = engine_config.cache_checkpoint_interval
        ec.cache_prompt = engine_config.cache_prompt
        ec.cache_prompt_boundary_skip = engine_config.cache_prompt_boundary_skip
        ec.cache_generation = engine_config.cache_generation
```

- [ ] **Step 5: Verify import + config construction (no GPU)**

Run (from repo root, in the normal env; this only imports Python and touches the built extension's attributes):

```bash
python -c "from lmdeploy import TurbomindEngineConfig as C; \
c=C(enable_prefix_caching=True); print(c.cache_prompt, c.cache_generation, c.cache_checkpoint_interval); \
import pytest\ntry:\n c2=C(cache_prompt='none')\nexcept AssertionError as e:\n print('rejected none:', e)"
```

Expected: prints `auto auto 4096` then `rejected none: invalid cache_prompt`. (If a one-liner with `\n` is awkward in the shell, put the same code in a scratch file and run it; do not commit the scratch file.)

- [ ] **Step 6: Commit**

```bash
git add lmdeploy/messages.py lmdeploy/turbomind/turbomind.py
git commit -m "feat(lmdeploy): TurbomindEngineConfig cache_prompt/cache_generation/cache_checkpoint_interval"
```

---

## Task 5: Migrate the verification scripts

Update the one script that owns the boundary CLI (`test_turbomind_model.py`, approved migration) and the two check scripts that pass the old kwargs.

**Files:**
- Modify: `scripts/test_turbomind_model.py` (constants, docstring, validator, argparse, three call sites, print block)
- Modify: `scripts/vlm_prefix_cache_check.py:147-148`
- Modify: `scripts/prompt_boundary_skip_check.py:139-141`

- [ ] **Step 1: Constants (`test_turbomind_model.py:139-140`)**

Replace:

```python
DEFAULT_LINEAR_PREFIX_CACHE_MIN_INTERVAL = 0
DEFAULT_CACHE_BOUNDARY_POLICY = ''
```

with:

```python
DEFAULT_CACHE_CHECKPOINT_INTERVAL = 4096
DEFAULT_CACHE_PROMPT = 'auto'
DEFAULT_CACHE_GENERATION = 'auto'
```

- [ ] **Step 2: Usage docstring (`test_turbomind_model.py:56-59`)**

Replace the four usage lines:

```
      [--linear-prefix-cache-min-interval N] \\
      [--cache-prompt-boundary] \\
      [--cache-generation-boundary] \\
      [--cache-boundary-policy NAME] \\
```

with:

```
      [--cache-checkpoint-interval N] \\
      [--cache-prompt {all,auto}] \\
      [--cache-generation {all,auto,none}] \\
```

- [ ] **Step 3: Validator (`test_turbomind_model.py:189-210`)**

Replace the `linear_prefix_cache_min_interval` param + check. In `_validate_engine_params`, change the keyword param at line 196 from `linear_prefix_cache_min_interval: int,` to `cache_checkpoint_interval: int,` and replace lines 208-210:

```python
    if linear_prefix_cache_min_interval < 0:
        raise ValueError(
            f'linear_prefix_cache_min_interval must be >= 0, got {linear_prefix_cache_min_interval}')
```

with:

```python
    if cache_checkpoint_interval < 1:
        raise ValueError(
            f'cache_checkpoint_interval must be >= 1, got {cache_checkpoint_interval}')
```

- [ ] **Step 4: Argparse (`test_turbomind_model.py:342-367`)**

Replace the three argument definitions (`--linear-prefix-cache-min-interval`, `--cache-prompt-boundary`, `--cache-generation-boundary`, `--cache-boundary-policy` — lines 342-367) with:

```python
    parser.add_argument(
        '--cache-checkpoint-interval',
        type=_positive_int,
        default=DEFAULT_CACHE_CHECKPOINT_INTERVAL,
        help=('TurbomindEngineConfig.cache_checkpoint_interval '
              f'(default: {DEFAULT_CACHE_CHECKPOINT_INTERVAL})'),
    )
    parser.add_argument(
        '--cache-prompt',
        choices=['all', 'auto'],
        default=DEFAULT_CACHE_PROMPT,
        help=('Prompt-boundary caching mode '
              f'(TurbomindEngineConfig.cache_prompt; default: {DEFAULT_CACHE_PROMPT!r})'),
    )
    parser.add_argument(
        '--cache-generation',
        choices=['all', 'auto', 'none'],
        default=DEFAULT_CACHE_GENERATION,
        help=('Generation caching mode '
              f'(TurbomindEngineConfig.cache_generation; default: {DEFAULT_CACHE_GENERATION!r})'),
    )
```

- [ ] **Step 5: `run_smoke_infer` signature + validate + config (`test_turbomind_model.py:415-451`)**

(a) Replace the three params at lines 415-418:

```python
    linear_prefix_cache_min_interval: int = DEFAULT_LINEAR_PREFIX_CACHE_MIN_INTERVAL,
    cache_prompt_boundary: bool = False,
    cache_generation_boundary: bool = False,
    cache_boundary_policy: str = DEFAULT_CACHE_BOUNDARY_POLICY,
```

with:

```python
    cache_checkpoint_interval: int = DEFAULT_CACHE_CHECKPOINT_INTERVAL,
    cache_prompt: str = DEFAULT_CACHE_PROMPT,
    cache_generation: str = DEFAULT_CACHE_GENERATION,
```

(b) In the `_validate_engine_params(...)` call (line 427), replace `linear_prefix_cache_min_interval=linear_prefix_cache_min_interval,` with `cache_checkpoint_interval=cache_checkpoint_interval,`.

(c) In the `TurbomindEngineConfig(...)` construction (lines 447-450), replace:

```python
        linear_prefix_cache_min_interval=linear_prefix_cache_min_interval,
        cache_prompt_boundary=cache_prompt_boundary,
        cache_generation_boundary=cache_generation_boundary,
        cache_boundary_policy=cache_boundary_policy,
```

with:

```python
        cache_checkpoint_interval=cache_checkpoint_interval,
        cache_prompt=cache_prompt,
        cache_generation=cache_generation,
```

- [ ] **Step 6: `print_report` signature + print block (`test_turbomind_model.py:496-514`)**

(a) Replace params at lines 496-499 the same way as Step 5(a).

(b) Replace the print lines 511-514:

```python
    print(f'linear_prefix_cache_min_interval: {linear_prefix_cache_min_interval}')
    print(f'cache_prompt_boundary: {1 if cache_prompt_boundary else 0}')
    print(f'cache_generation_boundary: {1 if cache_generation_boundary else 0}')
    print(f'cache_boundary_policy: {cache_boundary_policy!r}')
```

with:

```python
    print(f'cache_checkpoint_interval: {cache_checkpoint_interval}')
    print(f'cache_prompt: {cache_prompt!r}')
    print(f'cache_generation: {cache_generation!r}')
```

- [ ] **Step 7: `run_smoke_test` signature + forwarding (`test_turbomind_model.py:557-604`)**

(a) Replace params at lines 557-560 the same way as Step 5(a).

(b) In the `run_smoke_infer(...)` call (lines 581-584), replace:

```python
        linear_prefix_cache_min_interval=linear_prefix_cache_min_interval,
        cache_prompt_boundary=cache_prompt_boundary,
        cache_generation_boundary=cache_generation_boundary,
        cache_boundary_policy=cache_boundary_policy,
```

with:

```python
        cache_checkpoint_interval=cache_checkpoint_interval,
        cache_prompt=cache_prompt,
        cache_generation=cache_generation,
```

(c) In the `print_report(...)` call (lines 600-603), make the identical replacement as (b).

- [ ] **Step 8: `main()` arg forwarding (`test_turbomind_model.py:625-628`)**

Replace:

```python
        linear_prefix_cache_min_interval=args.linear_prefix_cache_min_interval,
        cache_prompt_boundary=args.cache_prompt_boundary,
        cache_generation_boundary=args.cache_generation_boundary,
        cache_boundary_policy=args.cache_boundary_policy,
```

with:

```python
        cache_checkpoint_interval=args.cache_checkpoint_interval,
        cache_prompt=args.cache_prompt,
        cache_generation=args.cache_generation,
```

- [ ] **Step 9: Update the two check scripts**

(a) `scripts/vlm_prefix_cache_check.py:147-148` — replace:

```python
        cache_prompt_boundary=True,
        cache_generation_boundary=True,
```

with:

```python
        cache_prompt='all',
        cache_generation='all',
```

(b) `scripts/prompt_boundary_skip_check.py:139-141` — replace:

```python
        cache_prompt_boundary=True,
        cache_generation_boundary=True,
        cache_prompt_boundary_skip=skip,
```

with:

```python
        cache_prompt='all',
        cache_generation='all',
        cache_prompt_boundary_skip=skip,
```

- [ ] **Step 10: Verify the scripts parse (no GPU)**

Run (from repo root):

```bash
python scripts/test_turbomind_model.py --help
rg -n "linear_prefix_cache_min_interval|cache_prompt_boundary\b|cache_generation_boundary|cache_boundary_policy" scripts/
```

Expected: `--help` prints and shows `--cache-checkpoint-interval`, `--cache-prompt {all,auto}`, `--cache-generation {all,auto,none}`; the `rg` prints only the surviving `cache_prompt_boundary_skip` references (in `prompt_boundary_skip_check.py` and the test-script's skip plumbing if present), and no `linear_prefix_cache_min_interval` / bare `cache_prompt_boundary` / `cache_generation_boundary` / `cache_boundary_policy`.

- [ ] **Step 11: Commit**

```bash
git add scripts/test_turbomind_model.py scripts/vlm_prefix_cache_check.py scripts/prompt_boundary_skip_check.py
git commit -m "chore(scripts): migrate to cache_prompt/cache_generation/cache_checkpoint_interval"
```

---

## Task 6: Contract (`README.md`) + user docs sync

Same-commit normative sync (`checklist.contract-sync`) plus the user-facing config docs. Content-only edits; do not re-wrap prose (AGENTS.md).

**Files:**
- Modify: `src/turbomind/engine/README.md` (`concepts` line ~35, `boundary-policy` ~123-125, `scheduler-owned` ~147, `prefix-prepare` ~255, `scheduler-commit` ~265/271, `resume-selection` ~367, `generation frontier` ~403, `boundary-policy` checklist ~461-463)
- Modify: `docs/en/inference/turbomind_config.md:112,119-127`
- Modify: `docs/zh_cn/inference/turbomind_config.md:114,121-129`

- [ ] **Step 1: README — `concepts.boundary-policy` (section at ~123-125)**

Rewrite the `### boundary-policy` concept (line 123 heading + line 125 paragraph) to describe modes instead of a policy object:

```
### boundary-policy

Partial-block boundary publication is decided entirely at Accept-time in
`SetupForks` (prompt) and at finalization in `PublishGeneration` (generation),
from two `CacheMode` knobs — `EngineConfig::cache_prompt` (`all`|`auto`) and
`EngineConfig::cache_generation` (`all`|`auto`|`none`) — parsed once into
`Scheduler::prompt_cache_mode_` / `generation_cache_mode_`. There is no runtime
veto object. `cache_prompt=all` publishes the partial prompt `fork_to` node
whenever `B` is mid-block and arms the block-aligned checkpoint clamp otherwise;
`cache_prompt=auto` publishes the partial node only when its token range
`[j*bs, B)` overlaps a multimodal span (`Scheduler::HasMultimodalOverlap`) and
never arms the block-aligned clamp. `cache_generation=all` indexes the terminal
partial generated block and adopts the terminal recurrent frontier checkpoint;
`auto` indexes full generated blocks only; `none` indexes no generated blocks.
The decision is a pure function of cross-rank-identical sequence attributes
(prompt geometry, `cache_prompt_boundary_skip`, `multimodal_spans`), so it is
consistent across ranks. `Sequence::prompt_boundary_node` now means the boundary
will be published (no deferred re-check).
```

- [ ] **Step 2: README — remaining references**

Make these content-only edits (keep existing line breaks):

- Line ~147 (`scheduler-owned`): delete the sentence "The scheduler also owns a `CacheBoundaryPolicy`, selected by `EngineConfig::cache_boundary_policy` and built by `CreateCacheBoundaryPolicy`, used for the runtime partial-block boundary publish decisions (`concepts.boundary-policy`)." Replace with: "The scheduler parses `EngineConfig::cache_prompt` / `cache_generation` into `CacheMode` values used for partial-block boundary publish decisions (`concepts.boundary-policy`)."
- Line ~255 (`prefix-prepare`): replace "gated on the boundary knobs alone … `fork_from` is bound when `cache_prompt_boundary` or `cache_generation_boundary` is enabled, and a partial `fork_to` node is created when `cache_prompt_boundary` is enabled and `B` falls inside a block." with: "`fork_from` is always bound (any prior request may have published a prompt or generation partial node). A partial `fork_to` node is created when `B` falls inside a block and `cache_prompt` admits it: `all` always, `auto` only when the node's token range overlaps a multimodal span."
- Line ~265 and ~271 (`scheduler-commit`): replace the parenthetical "…exactly B = prompt_len - cache_prompt_boundary_skip when `prompt_boundary_node` is set *and the boundary policy admits the prompt-boundary publish*; the policy is consulted only on a pass that can reach `B`, resolved once per request and cached, and a vetoed publish skips the clamp…" with "…exactly B = prompt_len - cache_prompt_boundary_skip when `prompt_boundary_node` is set (the publish decision is finalized in `SetupForks`; the clamp fires on the pass that can reach `B`)…". Update the line 271 code comment similarly (drop "and the boundary policy admits the publish").
- Line ~367 (`resume-selection`): drop any "`cache_prompt_boundary`" wording; reference the published prompt-boundary node without the removed knob name.
- Line ~403 (generation frontier adoption): replace "gated by `cache_generation_boundary` *and the boundary policy's generation-boundary predicate*" with "gated by `cache_generation == all`".
- Line ~461-463 (`checklist.boundary-policy`): rewrite the checklist question to: "Are partial-block boundary publishes decided at Accept/finalization from `cache_prompt` / `cache_generation` (`CacheMode`), with no runtime veto object? Is `cache_prompt=auto` gated on multimodal overlap of the partial node's range? Is the decision a pure function of cross-rank-identical attributes? Is full-block publication kept mode-free (coverage-driven)? Is the recurrent-checkpoint spacing knob `cache_checkpoint_interval` (> 0, no block_seq_len fallback)?"

Also update the `concepts` line ~35 to add: "`cache_prompt` / `cache_generation` are the two `CacheMode` publication knobs; `cache_checkpoint_interval` is the recurrent-checkpoint spacing (`CacheRegistry::checkpoint_min_interval`, > 0)." (Keep the existing `cache_prompt_boundary_skip` / `prompt_boundary_pos` sentence.)

- [ ] **Step 3: `docs/en/inference/turbomind_config.md` (lines 112, 119-127)**

- Line 112: replace the "Enable `cache_prompt_boundary` … / `cache_generation_boundary` …" guidance paragraph with mode-based guidance:

```
Set `cache_prompt='all'` when the same prompt is processed repeatedly (multi-sample decoding, shared/system prompts) so a duplicate prompt skips prefill; the default `'auto'` does this only for image-bearing partial prompt blocks (reusing vision-encoded KV) and is inert for text-only prompts. Set `cache_generation='all'` when you need to resume from the exact generation end (e.g. multi-turn chat); `'auto'` (default) caches full generated blocks only; `'none'` caches no generated blocks. The partial-block node costs extra VRAM and copy bandwidth, so prefer `'auto'` unless the reuse pays off.
```

- Lines 119-120 (the example snippet): replace `cache_prompt_boundary=True,` / `cache_generation_boundary=True,` with `cache_prompt='all',` / `cache_generation='all',`.
- Lines 125-127 (bullets): replace the `cache_prompt_boundary` / `cache_generation_boundary` bullets with `cache_prompt` / `cache_generation` bullets that state the `all|auto[|none]` semantics from Section 1 of the spec; update the `cache_prompt_boundary_skip` bullet's "Requires `cache_prompt_boundary`" → "Applies when `cache_prompt` is `'all'` or `'auto'`". Add a `cache_checkpoint_interval` bullet (min recurrent-checkpoint gap, > 0, default 4096) if the interval knob is documented in this file; otherwise leave interval docs as-is.

- [ ] **Step 4: `docs/zh_cn/inference/turbomind_config.md` (lines 114, 121-129)**

Mirror Step 3 in Chinese: line 114 guidance paragraph, lines 121-122 example snippet (`cache_prompt='all'` / `cache_generation='all'`), lines 127-129 bullets (`cache_prompt` / `cache_generation` mode semantics; `cache_prompt_boundary_skip` "需要开启 `cache_prompt_boundary`" → "当 `cache_prompt` 为 'all' 或 'auto' 时生效").

- [ ] **Step 5: Verify no stale references in docs/README**

Run: `rg -n "cache_prompt_boundary\b|cache_generation_boundary|cache_boundary_policy|linear_prefix_cache_min_interval|CacheBoundaryPolicy" src/turbomind/engine/README.md docs/en/inference/turbomind_config.md docs/zh_cn/inference/turbomind_config.md`
Expected: no matches (only `cache_prompt_boundary_skip` may remain, which is fine).

- [ ] **Step 6: Commit**

```bash
git add src/turbomind/engine/README.md docs/en/inference/turbomind_config.md docs/zh_cn/inference/turbomind_config.md
git commit -m "docs(turbomind): sync contract + config docs to cache_prompt/cache_generation modes"
```

---

## Task 7: Model-level verification (GPU, outside sandbox)

Verify the built engine end-to-end. **Run outside the sandbox on an empty GPU** (check `get_gpu_usage`; use the model-server MCP `list_models` / `get_model_cache_path` for a locally cached model). Use `scripts/test_turbomind_model.py` **AS IS** (it now carries the new flags). Response must be coherent, on-topic human text of ≥128 tokens — verify it, do not just check exit code.

**Files:** none (verification only).

- [ ] **Step 1: Rebuild the extension (if not already current)**

Run (from `build/`): `ninja _turbomind`
Expected: up to date / clean build.

- [ ] **Step 2: Default-mode smoke test on a text model**

Pick an empty GPU `G` and a locally cached text model `M` with cache dir `D`. Run (outside sandbox):

```bash
python scripts/test_turbomind_model.py --model-id M --cache-dir D --tp 1 --gpus G \
    --enable-prefix-caching --max-new-tokens 128
```

Expected: report prints `cache_prompt: 'auto'`, `cache_generation: 'auto'`, `cache_checkpoint_interval: 4096`; the response is coherent, on-topic, ≥128 tokens.

- [ ] **Step 3: `cache_generation` mode sweep on the same model**

Run twice (outside sandbox), verifying a coherent ≥128-token response each time:

```bash
python scripts/test_turbomind_model.py --model-id M --cache-dir D --tp 1 --gpus G \
    --enable-prefix-caching --max-new-tokens 128 --cache-generation none
python scripts/test_turbomind_model.py --model-id M --cache-dir D --tp 1 --gpus G \
    --enable-prefix-caching --max-new-tokens 128 --cache-generation all
```

Expected: both runs produce coherent on-topic ≥128-token text (no gibberish, no crash). This exercises the `kNone` early-return and the `kAll` terminal-partial path.

- [ ] **Step 4: VLM `auto` reuse check (if a native-vision model is available locally)**

Run (outside sandbox), on a TurboMind native-vision model per `scripts/vlm_prefix_cache_check.py`:

```bash
python scripts/vlm_prefix_cache_check.py --tp 1 --max-new-tokens 128 <script's image/prompt args>
```

Expected: the script's turn-2 reuse assertions pass (image-bearing partial prompt block published on turn 1 and reused on turn 2) and the greedy output matches the no-cache oracle. If no native-vision model is cached, skip this step and note it.

- [ ] **Step 5: Final sanity grep (repo-wide)**

Run: `rg -n "linear_prefix_cache_min_interval|cache_boundary_policy|CacheBoundaryPolicy|\bcache_prompt_boundary\b|\bcache_generation_boundary\b|ResolvePublishPromptBoundary|prompt_boundary_publish" --glob '!docs/superpowers/**'`
Expected: no matches anywhere (spec/plan docs under `docs/superpowers/**` are excluded and may still reference old names historically).

- [ ] **Step 6: Commit (only if any fixups were needed)**

If Steps 1-5 forced a code fix, commit it with a descriptive message. Otherwise there is nothing to commit for this task.

---

## Self-Review notes

- **Spec coverage:** §2 mode representation → Task 1; §3 config surface → Tasks 2 (C++) + 4 (Python); §4 scheduler behavior (SetupForks gate, HasMultimodalOverlap, final-at-SetupForks, PublishGeneration) → Task 3; §5 checkpoint interval → Task 2; §6 removals → Task 3; §7 scripts/docs/contract → Tasks 5 + 6; §8 testing → Task 1 (unit) + Task 7 (model); §9 behavior callouts → covered by docs edits in Task 6.
- **Type consistency:** `CacheMode`, `ParseCacheMode`, `DecidePromptBoundaryPublish`, `HasMultimodalOverlap`, `prompt_cache_mode_`, `generation_cache_mode_`, `cache_checkpoint_interval`, `cache_prompt`, `cache_generation` are used identically across all tasks.
- **Build ordering:** Tasks 2 and 3 are interdependent; the first clean build is at Task 3 Step 13 (intentional — noted in Task 2 Step 3).
