# Design: overhaul of TurboMind's prefix-cache interface

- Date: 2026-07-01
- Status: Approved design, pre-implementation
- Scope: TurboMind C++ engine boundary-publication knobs + the Python
  `TurbomindEngineConfig` glue + the three scheduler/verification scripts and
  contract/docs sync. No PyTorch backend, no per-request chat-template logic, no
  new main-`lmdeploy`-CLI flags.

## 1. Goal & motivation

Today the prefix-cache boundary behavior is spread across four coupled knobs
with non-obvious interactions:

- `cache_prompt_boundary: bool` — publish the partial prompt-boundary `fork_to`
  node.
- `cache_generation_boundary: bool` — index the terminal partial generated
  block + adopt the terminal recurrent frontier.
- `cache_boundary_policy: str` (`''`/`'default'`/`'auto'`) — an extra runtime
  gate: `'auto'` (`AutoCacheBoundaryPolicy`) only publishes a partial node when
  its distance from the last checkpoint is `>= linear_prefix_cache_min_interval/2`.
- `linear_prefix_cache_min_interval: int` — double duty: (a) the `'auto'`-policy
  threshold, (b) `CacheRegistry::checkpoint_min_interval` (recurrent-checkpoint
  spacing), with `0` special-cased to mean `cache_block_seq_len`.

This overhaul collapses the two booleans + `cache_boundary_policy` into two
self-describing string modes, deletes the `CacheBoundaryPolicy` machinery, adds
a new "cache the image-bearing partial prompt block" behavior as the default,
and renames the interval knob to reflect its single remaining job.

New interface (`TurbomindEngineConfig`):

- `cache_prompt: str = 'auto'` — `'all' | 'auto'` (no `'none'`).
- `cache_generation: str = 'auto'` — `'all' | 'auto' | 'none'`.
- `cache_checkpoint_interval: int = 4096` — replaces
  `linear_prefix_cache_min_interval`; `> 0` required.
- `cache_prompt_boundary_skip: int = 1` — unchanged (the `K` in
  `B = prompt_len - K`; orthogonal to the mode).

Mode semantics:

| knob | value | meaning |
| --- | --- | --- |
| `cache_prompt` | `all` | today's `cache_prompt_boundary=True` & `cache_boundary_policy=''`: publish the partial `fork_to` node whenever `B` is mid-block; arm the checkpoint clamp when `B` is block-aligned. |
| `cache_prompt` | `auto` (default) | like today's `cache_prompt_boundary=False`, **except** publish the partial `fork_to` node when its own token range holds image tokens. No block-aligned clamp. |
| `cache_generation` | `all` | today's `cache_generation_boundary=True`: index full generated blocks + the terminal partial block, adopt the terminal recurrent frontier checkpoint. |
| `cache_generation` | `auto` (default) | today's `cache_generation_boundary=False`: index full generated blocks only. |
| `cache_generation` | `none` (new) | do not cache generation at all — index no generated blocks, full or partial. |

## 2. Mode representation (strings end-to-end, parsed once to a C++ enum)

Chosen over "convert to int in the Python glue" and "Python `Enum` + C++ int":
strings match the existing `cache_boundary_policy` string-field precedent, keep
`EngineConfig` a plain reflected struct, need no pybind mapping table, and stay
JSON/config-dict friendly. Validation lives in Python (`__post_init__`); the C++
side parses the string once into an `enum class` and `TM_LOG_FATAL`s on an
unknown value (defense in depth).

New lightweight header `src/turbomind/engine/cache_mode.h`:

```cpp
#pragma once

#include <string>

#include "src/turbomind/core/logger.h"  // TM_LOG_FATAL

namespace turbomind {

enum class CacheMode
{
    kNone,
    kAuto,
    kAll
};

// String -> CacheMode. cache_prompt never receives "none" (rejected by the
// Python __post_init__ assert); the shared parser still accepts it for
// cache_generation. TM_LOG_FATAL is [[noreturn]] (std::abort), so no trailing
// return is needed.
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

}  // namespace turbomind
```

## 3. Config surface

### 3.1 Python — `TurbomindEngineConfig` (`lmdeploy/messages.py`)

Remove `cache_prompt_boundary`, `cache_generation_boundary`,
`cache_boundary_policy`, `linear_prefix_cache_min_interval`. Keep
`cache_prompt_boundary_skip`. Add:

```python
cache_prompt: str = 'auto'            # 'all' | 'auto'
cache_generation: str = 'auto'        # 'all' | 'auto' | 'none'
cache_checkpoint_interval: int = 4096
cache_prompt_boundary_skip: int = 1   # unchanged
```

Docstrings:

- `cache_prompt`: partial prompt-boundary publication mode. `'all'` publishes the
  reusable partial `fork_to` node at `B = prompt_len - cache_prompt_boundary_skip`
  whenever `B` is mid-block (and arms a recurrent checkpoint clamp when `B` is
  block-aligned), so a duplicate prompt skips prefill (costs one extra prefill
  forward + a partial block). `'auto'` (default) does that only when the partial
  block holds image tokens — reusing expensive vision-encoded KV — and is inert
  for text-only prompts. Requires `enable_prefix_caching`.
- `cache_generation`: generated-block caching mode. `'all'` indexes full
  generated blocks and the terminal partial block, and adopts the terminal
  recurrent frontier checkpoint (exact multi-turn resume, costs a partial
  block). `'auto'` (default) indexes full generated blocks only. `'none'`
  indexes no generated blocks at all. Requires `enable_prefix_caching`.
- `cache_checkpoint_interval`: minimum token gap between reusable recurrent-state
  checkpoints (`CacheRegistry::checkpoint_min_interval`). Must be `> 0`. Default
  4096.
- `cache_prompt_boundary_skip`: number of trailing prompt tokens treated as the
  volatile generation-prompt suffix (e.g. a chat template's `<think>\n`) and
  excluded from the reusable prompt-boundary node, so it ends at
  `prompt_len - cache_prompt_boundary_skip`. Default 1. Applies when
  `cache_prompt` is `'all'` or `'auto'`.

`__post_init__` validation (replace the old
`linear_prefix_cache_min_interval >= 0` assert):

```python
assert self.cache_prompt in ('all', 'auto'), 'invalid cache_prompt'
assert self.cache_generation in ('all', 'auto', 'none'), 'invalid cache_generation'
assert self.cache_checkpoint_interval > 0, 'invalid cache_checkpoint_interval'
assert self.cache_prompt_boundary_skip >= 1, 'invalid cache_prompt_boundary_skip'
```

### 3.2 C++ — `EngineConfig` (`engine_config.h`)

Replace the four removed `ENGINE_FIELDS` entries
(`linear_prefix_cache_min_interval`, `cache_prompt_boundary`,
`cache_generation_boundary`, `cache_boundary_policy`) with:

```cpp
    X(int, cache_checkpoint_interval, 4096)                                                                            \
    X(std::string, cache_prompt, "auto")                                                                              \
    X(int, cache_prompt_boundary_skip, 1)                                                                             \
    X(std::string, cache_generation, "auto")                                                                          \
```

(`cache_prompt_boundary_skip` is unchanged; the block above just shows its
neighbors. `cache_boundary_policy` is deleted outright.)

### 3.3 Python glue — `lmdeploy/turbomind/turbomind.py` (~240–245)

```python
ec.enable_prefix_caching      = engine_config.enable_prefix_caching
ec.cache_checkpoint_interval  = engine_config.cache_checkpoint_interval
ec.cache_prompt               = engine_config.cache_prompt
ec.cache_prompt_boundary_skip = engine_config.cache_prompt_boundary_skip
ec.cache_generation           = engine_config.cache_generation
```

Remove the old `ec.linear_prefix_cache_min_interval`,
`ec.cache_prompt_boundary`, `ec.cache_generation_boundary`,
`ec.cache_boundary_policy` assignments.

## 4. Scheduler behavior

`Scheduler` stores two parsed modes and drops the boolean/policy members:

```cpp
// scheduler.h
CacheMode prompt_mode_{CacheMode::kAuto};       // replaces cache_prompt_boundary_
int       cache_prompt_boundary_skip_{1};       // unchanged
CacheMode generation_mode_{CacheMode::kAuto};   // replaces cache_generation_boundary_
// removed: bool cache_prompt_boundary_, bool cache_generation_boundary_,
//          std::unique_ptr<CacheBoundaryPolicy> boundary_policy_
```

Constructor signature changes the two `bool` boundary params to two
`const std::string&` (parsed via `ParseCacheMode`) and drops the
`std::unique_ptr<CacheBoundaryPolicy>` param:

```cpp
Scheduler::Scheduler(..., const std::string& cache_prompt,
                          int                 cache_prompt_boundary_skip,
                          const std::string&  cache_generation, ...):
    prompt_mode_{ParseCacheMode(cache_prompt)},
    cache_prompt_boundary_skip_{cache_prompt_boundary_skip < 1 ? 1 : cache_prompt_boundary_skip},
    generation_mode_{ParseCacheMode(cache_generation)},
    ...
```

`engine.cc` (~230): pass `param_.cache_prompt`, `param_.cache_prompt_boundary_skip`,
`param_.cache_generation`; drop `CreateCacheBoundaryPolicy(param_)` and the
`cache_boundary_policy.h` include.

### 4.1 `SetupForks` read/publish sides (`scheduler.cc:418-493`)

`prompt_mode_` is always `kAuto` or `kAll` (no `'none'`), so the prompt-boundary
attempt and the `fork_from` read side are **always armed** — the guard bools
`prompt_boundary` / `fork_match` are removed:

```cpp
void Scheduler::SetupForks(Sequence& s, AcceptState& st)
{
    const int bs         = logical_.block_size();
    const int prompt     = s.prompt_len;
    const int all_blocks = (prompt + bs - 1) / bs;

    // fork_from: always armed. Any prior request may have published a prompt
    // partial node (cache_prompt in {all, auto}) or a generation terminal partial
    // ('all'), so the read edge must always try to match.
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
        bool have_target = false;

        if (plan.partial) {
            const int  lo   = plan.block * bs;   // j*bs
            const int  hi   = plan.pos;          // B
            const bool gate = (prompt_mode_ == CacheMode::kAll)
                              || (prompt_mode_ == CacheMode::kAuto && HasMultimodalOverlap(s, lo, hi));
            if (gate) {
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
                    x.fork_to    = std::move(vh);  // edge holds the only ref
                    have_target  = true;
                }
                else {
                    LogCollision(s, CollisionSite::kPromptBoundary, j * bs, j * bs + plan.node_size);
                }
            }
        }
        else {
            // Block-aligned B: only 'all' arms the recurrent-checkpoint clamp;
            // 'auto' has no partial block to hold image KV, so it publishes nothing.
            have_target = (prompt_mode_ == CacheMode::kAll);
        }

        if (have_target) {
            s.prompt_boundary_node = true;
            s.prompt_boundary_pos  = plan.pos;  // clamp the producer's prefill to B
        }
    }
}
```

New pure predicate (determinism-safe: reads only `multimodal_spans` geometry,
which is cross-rank identical; `multimodal_spans` is prompt-ordered ascending by
`interval.begin()`):

```cpp
// True if any multimodal span overlaps [lo, hi). Interval is the absolute token
// span [begin, end); a partial prompt block "contains image tokens" when a span
// intersects it, even one that started in an earlier (full) block and extends in.
bool HasMultimodalOverlap(const Sequence& s, int lo, int hi)
{
    for (const auto& sp : s.multimodal_spans) {
        const int b = sp.interval.begin();
        if (b >= hi) {
            break;  // spans are ascending; no later span can overlap
        }
        if (sp.interval.end() > lo) {
            return true;
        }
    }
    return false;
}
```

### 4.2 Publish decision is final at `SetupForks`

With the `CacheBoundaryPolicy` object gone and the `auto`/`all` decision made
here from cross-rank-identical attributes, `Sequence::prompt_boundary_node` now
directly means "publish" (no deferred policy re-check). Therefore:

- Delete `Sequence::prompt_boundary_publish` (`request.h:250`) and its
  serialization, if any.
- Delete `Scheduler::ResolvePublishPromptBoundary` (`scheduler.cc:941-950`) and
  its declaration (`scheduler.h:205`).
- Admission clamp (`scheduler.cc:1126-1136`): the publish decision reduces to the
  geometry test.

```cpp
        const int prompt_boundary_pos = s.prompt_boundary_pos;

        const bool publish_prompt =
            s.prompt_boundary_node && begin < prompt_boundary_pos && desired >= prompt_boundary_pos;
        // was: is_prompt_boundary && ResolvePublishPromptBoundary(s)

        if (publish_prompt) {
            desired = prompt_boundary_pos;  // land exactly on B
        }
        else if (desired < ctx_end) {
            desired = desired / bs * bs;
        }
```

### 4.3 `PublishGeneration` (`scheduler.cc:773-780`)

```cpp
void Scheduler::PublishGeneration(Sequence& s)
{
    if (!PrefixEligible(s) || s.filled_len <= 0) {
        return;
    }
    if (generation_mode_ == CacheMode::kNone) {
        return;  // NEW: index no generated blocks at all
    }

    // 'all' indexes the terminal partial block + adopts the terminal frontier
    // checkpoint; 'auto' indexes full generated blocks only.
    const bool publish_generation_boundary = (generation_mode_ == CacheMode::kAll);
    ...
}
```

The rest of `PublishGeneration` is unchanged: the `size < x.capacity &&
!publish_generation_boundary` early-break keeps a terminal partial block private
under `'auto'`, and the frontier-adoption block stays gated on
`publish_generation_boundary` (so only `'all'`).

## 5. Checkpoint interval

`turbomind.cc:299-301` becomes a direct assignment; the `> 0 ? … :
cache_block_seq_len` fallback is dropped (positivity enforced in Python
`__post_init__`):

```cpp
    cache_registry.set_checkpoint_min_interval(param.cache_checkpoint_interval);
```

## 6. Removals / cleanup

- Delete `src/turbomind/engine/cache_boundary_policy.h` and
  `src/turbomind/engine/cache_boundary_policy.cc`.
- Remove `cache_boundary_policy.cc` from `src/turbomind/engine/CMakeLists.txt`.
- Remove the `#include "src/turbomind/engine/cache_boundary_policy.h"` from
  `scheduler.h`, `scheduler.cc`, and `engine.cc`.
- Remove `boundary_policy_` member + its ctor param from `scheduler.{h,cc}`;
  replace the two `bool` ctor params with two `std::string` params.
- Remove `Sequence::prompt_boundary_publish` (`request.h`) and
  `Scheduler::ResolvePublishPromptBoundary`.
- Add `src/turbomind/engine/cache_mode.h` (Section 2). No new `.cc` (header-only
  `inline`), so no CMake source addition.

## 7. Scripts, docs, contract sync

### 7.1 `scripts/test_turbomind_model.py` (approved migration)

Migrate the flags/params in lock-step (the script currently exposes the full
boundary interface):

- `--linear-prefix-cache-min-interval` → `--cache-checkpoint-interval`
  (`type=int`, default 4096); rename the `linear_prefix_cache_min_interval`
  params/locals to `cache_checkpoint_interval`; drop the old `>= 0` check, add a
  `> 0` check.
- `--cache-prompt-boundary` (store_true) + `--cache-boundary-policy` (str) →
  `--cache-prompt` (`choices=['all', 'auto']`, default `'auto'`).
- `--cache-generation-boundary` (store_true) → `--cache-generation`
  (`choices=['all', 'auto', 'none']`, default `'auto'`).
- Thread the three renamed values through all call sites
  (`TurbomindEngineConfig(...)`, the print block, and the arg-plumbing
  functions) as `cache_prompt=`, `cache_generation=`,
  `cache_checkpoint_interval=`.

### 7.2 `scripts/prompt_boundary_skip_check.py`, `scripts/vlm_prefix_cache_check.py`

Replace the removed `TurbomindEngineConfig(...)` kwargs
(`cache_prompt_boundary=True`, `cache_generation_boundary=True`) with
`cache_prompt='all'`, `cache_generation='all'`. `cache_prompt_boundary_skip`
stays as-is in `prompt_boundary_skip_check.py`.

### 7.3 `src/turbomind/engine/README.md` (`checklist.contract-sync`)

Same-commit, content-only edits (no prose re-wrapping):

- `concepts.boundary-policy` — the `CacheBoundaryPolicy` object is removed;
  publication is now decided at `SetupForks` from the `cache_prompt` /
  `cache_generation` modes and is final (no deferred re-check). Note the new
  `'auto'` prompt rule: publish the partial `fork_to` node only when its token
  range overlaps a multimodal span.
- `contracts.prefix-prepare` — the prompt `fork_to` node is published per
  `cache_prompt`: `'all'` whenever `B` is mid-block; `'auto'` only when the
  partial block holds image tokens. Block-aligned `B` arms the checkpoint clamp
  under `'all'` only.
- `contracts.scheduler-commit` — the clamp fires on `prompt_boundary_node`
  alone (no policy re-check).
- `resume-selection` — unchanged mechanics; wording references the modes.
- `checklist.cache-memory` — the recurrent-checkpoint spacing knob is
  `cache_checkpoint_interval` (was `linear_prefix_cache_min_interval`; the
  `0 ⇒ cache_block_seq_len` fallback is gone, `> 0` required).
- `concepts` — one line: `cache_prompt`/`cache_generation` are the two
  `CacheMode` knobs; `cache_prompt_boundary_skip` and
  `Sequence::prompt_boundary_pos` unchanged.

### 7.4 `docs/en|zh_cn/inference/turbomind_config.md`

Update the knob table/prose (content-only) to the new names and the mode
semantics.

## 8. Testing

- **Unit (`test_prefix_trie.cc`, CPU, `BUILD_TEST=ON`).** `PlanPromptBoundary`
  itself is unchanged, but add cases for the new mode gate wired around it:
  `'auto'` + image overlap in `[j*bs, B)` → publish; `'auto'` + no image → no
  publish; `'all'` mid-block → publish (regression); block-aligned `B` under
  `'auto'` → nothing, under `'all'` → clamp target. If the gate is factored into
  a small pure helper (mode + `HasMultimodalOverlap` result → publish bool), test
  that directly.
- **Model regression (`scripts/test_turbomind_model.py`, used AS IS after the
  §7.1 migration; `--max-new-tokens 128`, verify coherent on-topic text of
  ≥128 tokens; pick an empty GPU via `get_gpu_usage`, run outside the sandbox).**
  A text model with defaults (`cache_prompt=auto` ⇒ no prompt partials for
  text; `cache_generation=auto`), plus one run with `--cache-generation none`
  and one with `--cache-generation all`, each verified for coherent output and a
  repeated-prefix cache hit without output change.
- **VLM multi-turn (`scripts/vlm_prefix_cache_check.py`).** On a TurboMind
  native-vision model with `cache_prompt='auto'`: confirm the image-bearing
  partial prompt block publishes a `fork_to` node and a second turn reuses it
  (INFO logs: `fork_to@B` on turn 1, `LogResume source=fork` on turn 2), with a
  greedy-token oracle unchanged vs. a no-cache run. A text-only prompt under
  `'auto'` publishes nothing.

**Out of scope:** PyTorch backend; per-request/chat-template computation of the
skip length; new main-`lmdeploy`-CLI flags.

## 9. Behavior-change callouts

1. `cache_generation='auto'` (default) == today's `cache_generation_boundary=False`
   — unchanged.
2. `cache_prompt='auto'` (default) differs from today's default
   `cache_prompt_boundary=False`: text-only prompts are still unaffected, but a
   TurboMind native-vision VLM prompt now publishes/clamps a partial
   prompt-boundary node when the partial block holds image tokens (the new
   feature; costs one extra prefill forward + a partial block on those requests).
3. `cache_checkpoint_interval` default is a fixed `4096`, replacing
   "`0` ⇒ `cache_block_seq_len` (e.g. 64)". This widens default
   recurrent-checkpoint spacing; `0` is now invalid.
4. The `cache_boundary_policy='auto'` min-interval/2 gating of partial-boundary
   publication is gone entirely; there is no per-checkpoint-distance gate on
   boundary publication anymore.

## Data flow summary

```
TurbomindEngineConfig{cache_prompt, cache_generation, cache_checkpoint_interval, cache_prompt_boundary_skip}
  -> EngineConfig{cache_prompt(str), cache_generation(str), cache_checkpoint_interval(int), cache_prompt_boundary_skip(int)}
  -> Scheduler ctor: prompt_mode_ = ParseCacheMode(cache_prompt); generation_mode_ = ParseCacheMode(cache_generation)
  -> turbomind.cc: cache_registry.set_checkpoint_min_interval(cache_checkpoint_interval)
  -> [Accept/SetupForks] fork_from always armed; PlanPromptBoundary -> mode gate
       ('all': mid-block partial / block-aligned clamp; 'auto': partial only if HasMultimodalOverlap)
       -> set Sequence.prompt_boundary_{node,pos}
  -> [Schedule clamp] publish_prompt = prompt_boundary_node && crossing B  (no policy re-check)
  -> [PublishGeneration] kNone: return; kAuto: full blocks only; kAll: + terminal partial + frontier checkpoint
```
