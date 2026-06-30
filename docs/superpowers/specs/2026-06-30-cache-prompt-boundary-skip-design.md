# Design: configurable prompt-boundary skip length

- Date: 2026-06-30
- Status: Approved design, pre-implementation
- Scope: TurboMind C++ engine prompt-boundary (`fork_to`) publication + the
  Python `TurbomindEngineConfig` glue. No chat-template / per-request logic.

## 1. Goal & core problem

**Goal.** Let the reusable prompt-boundary node end at a configurable position
`B = prompt_len - K`, instead of the hard-coded `prompt_len - 1` (exclude exactly
one token).

**Core problem.** The `cache_prompt_boundary` feature publishes a partial
`fork_to` node so the *next* conversation turn can reuse the partial tail of the
shared history. `SetupForks` ends that node at `prompt_len - 1`
(`src/turbomind/engine/scheduler.cc:454-488`), assuming only the single last
prompt token is volatile and everything before it reappears verbatim as the
prefix of the next turn.

That assumption breaks for thinking models. Their chat template appends a
multi-token think trigger via `add_generation_prompt` (e.g.
`<|im_start|>assistant\n<think>\n`). The next turn strips the whole thinking
block from history, so the next prompt diverges *before* `<think>`:

```
turn 1 prompt : [common history] <|im_start|>assistant\n <think>\n
turn 2 prompt : [common history] <|im_start|>assistant\n [answer] <|im_end|> ...
                                                        ^ divergence (think stripped)
```

The stable, reusable prefix `[H]` ends right after the assistant header; the
volatile suffix `[G] = <think>\n` (~2 tokens) does not reappear. But the
published node covers `[H] + [G]` minus one token, so it still contains
`<think>`. The next turn's `fork_from` match misses it and falls back to the last
whole-block boundary, losing the partial-tail reuse.

**Solution.** Make the excluded-suffix length a single engine-level integer `K`
(`cache_prompt_boundary_skip`, default 1 = today's behavior), and publish the
boundary node at `B = prompt_len - K`, placed in whatever block contains `B`.

**Why a global config (not per-request).** The volatile suffix is a property of
the model's chat template, constant per deployment. A global value is safe even
when it over-estimates: segment tokens are exact-compared in the trie, so a
too-long skip only shortens reuse (the node ends earlier than optimal) and never
produces a false hit. Under-excluding is the current bug. We therefore avoid all
per-request chat-template/tokenization machinery.

## 2. The config knob

A single engine-level integer, default `1`.

**Python — `lmdeploy/messages.py` (`TurbomindEngineConfig`):**

```python
cache_prompt_boundary_skip: int = 1
```

Docstring: number of trailing prompt tokens treated as the volatile
generation-prompt suffix (e.g. a chat template's `<think>\n`) and excluded from
the reusable prompt-boundary node. `1` excludes only the last token (the resume
cap). Requires `cache_prompt_boundary`.

**Python — `lmdeploy/turbomind/turbomind.py` (~line 242, beside the existing
boundary fields):**

```python
ec.cache_prompt_boundary_skip = engine_config.cache_prompt_boundary_skip
```

**C++ — `EngineConfig` (`engine_config.h`):**

```cpp
int cache_prompt_boundary_skip = 1;
```

**C++ — `Scheduler`:** store as `cache_prompt_boundary_skip_`, clamped `>= 1` at
construction (a `0`/negative value would violate the `seq_len-1` resume cap),
threaded through the same constructor path as `cache_prompt_boundary_`. The
design assumes `skip < block_size` (a handful of tokens); a pathological
`skip >= block_size` still behaves safely (boundary moves further back or is
skipped) but is not a design target.

## 3. One stored boundary position on `Sequence`

The boundary position is computed once in `Accept`/`SetupForks` and stored on the
`Sequence`, so every downstream site reads one field instead of re-deriving
`prompt_len - 1`. This also keeps the boundary-policy determinism contract intact
(predicates stay pure functions of cross-rank-identical *sequence* attributes).

**C++ — `Sequence` (`request.h`), beside the existing boundary fields:**

```cpp
bool prompt_boundary_node = false;  // unchanged: a reusable prompt-boundary node exists
int  prompt_boundary_pos  = 0;      // NEW: that node's end position B = prompt_len - K; 0 = none
int  prompt_boundary_publish = -1;  // unchanged: cached policy decision
```

`prompt_boundary_node` keeps its exact current meaning (a publish target exists;
clamp/publish machinery is armed). `prompt_boundary_pos` carries *where* `B` is,
set only when `prompt_boundary_node` is set. It is a pure function of `prompt_len`
(cross-rank identical) and the global `K`, so it is itself cross-rank identical.

## 4. Generalizing `SetupForks` to place the node at `B = prompt_len - K`

Replaces the last-block-only, exclude-one-token logic
(`scheduler.cc:454-488`). The boundary `B = prompt_len - K` lands in block
`j = (B-1)/bs` — the last partial block in the common case (small `K`), or an
earlier block when `K >= tail_len`. The gate is `B`'s own alignment: a `fork_to`
partial node is needed only when `B` falls *inside* a block (`B % bs != 0`); when
`B` is block-aligned the prefix `[0, B)` is already fully tiled by whole blocks
and needs no partial node to be fully matched.

```cpp
if (cache_prompt_boundary_) {
    const int skip = cache_prompt_boundary_skip_;   // >= 1 (assumes skip < bs)
    const int B    = prompt - skip;                 // reusable boundary position

    if (B >= 1) {
        const int j = (B - 1) / bs;                 // block holding the last token before B
        bool      have_target = false;

        if (B % bs != 0) {
            // B is strictly inside block j. Publish a partial fork_to node so a
            // future turn whose stable history is [0, B) matches the whole blocks
            // [0, j*bs) and then fork-extends through this node to B.
            // st.miss < j: blocks [0..j-1] are matched/created-and-indexed (valid
            // parent chain) and block j is not the fork_from miss block.
            if (st.miss < j) {
                const LogicalBlock* parent    = s.block_ids[j - 1].get();   // j >= 1 here
                const PrefixKey     base      = s.block_ids[j - 1]->key;
                const int           node_size = B - j * bs;                  // 1..bs-1
                const auto          tokens    = TokenSegment(s, j * bs, node_size);

                std::vector<Fingerprint> fps;
                CollectStartFps(s, j * bs, B, fps);
                const auto next = ExtendPrefixKey(base, tokens, fps);

                BlockHandle   vh = logical_.Create(j);
                LogicalBlock& y  = *vh;
                y.parent = parent; y.key = next; y.size = node_size;
                y.tokens.assign(tokens.begin(), tokens.end());
                y.image_fps = fps;
                y.prefix_id = cache_.Create(registry_.prefix().object_id(), vh.get());
                if (trie_.Insert(y)) {
                    s.block_ids[j]->fork_to = std::move(vh);  // edge holds the only ref
                    have_target = true;
                }
                else {
                    LogCollision(s, CollisionSite::kPromptBoundary, j * bs, B);
                }
            }
        }
        else {
            // B is block-aligned: [0, B) is fully matched by whole blocks, so no
            // partial node. Block j (= last-1 when skip < bs) ends exactly at B; we
            // still arm the clamp so the producer lands a forward end at B and
            // publishes a recurrent checkpoint there (the natural forward end is
            // prompt = B + K, not block-aligned, so always-on full-block
            // publication would not checkpoint B). st.miss <= j: block j is
            // matched or created-and-indexed.
            have_target = (st.miss <= j);
        }

        if (have_target) {
            s.prompt_boundary_node = true;
            s.prompt_boundary_pos  = B;
        }
    }
}
```

**Naming cleanup.** The rewrite removes the old `full_size` variable
(`scheduler.cc:456`), which was the last block's *occupancy* / partial-tail
length (`prompt_len - last*bs`), not a full block — a misleading name. If a
comment still needs that quantity it uses `tail_len`.

**Equivalence at `K = 1`:**

- `prompt_len % bs > 1`: `B = prompt_len-1`, `B % bs != 0`, `j = last`,
  `node_size = tail_len-1` → partial branch, identical node to today.
- `prompt_len % bs == 1`: `B = last*bs`, `B % bs == 0`, `j = last-1` →
  block-aligned branch with `st.miss <= j` (= today's `st.miss < last`),
  `have_target = true`, no partial node — exactly today's `full_size == 1` case.
- block-aligned prompt (`prompt_len % bs == 0`), `K = 1`: `B = prompt_len-1`,
  `B % bs != 0`, `j = last`, `node_size = bs-1` → partial branch. Today this was
  skipped by the old `prompt % bs != 0` guard. **Decision: keep the new behavior**
  (block-aligned prompts also gain a partial boundary node at `K=1`); the prior
  guard was the prompt's alignment, but the correct quantity is `B`'s alignment.
  This is a small extra reuse win, not a correctness change.

**Why keep the block-aligned `else` branch / the `< j` vs `<= j` asymmetry.** The
block-aligned branch's only job is to force a recurrent **checkpoint** at `B` via
the clamp; a pure-KV model gains nothing from it, a recurrent model gains a
resumable checkpoint. It preserves today's `full_size == 1` behavior exactly. The
partial branch's stricter `st.miss < j` keeps the published node off the
`fork_from` miss block (the read edge already handles that block); the
block-aligned branch allows `st.miss == j` (block `j` is the miss block, full,
created-and-indexed, used as the boundary).

## 5. Downstream sites read `prompt_boundary_pos`

Two literal sites change; the rest is already geometry-driven.

**(a) Admission clamp (`scheduler.cc` ~1123):**

```cpp
        // was: const int prompt_boundary_pos = s.prompt_len - 1;
        const int prompt_boundary_pos = s.prompt_boundary_pos;

        const bool is_prompt_boundary =
            s.prompt_boundary_node && begin < prompt_boundary_pos && desired >= prompt_boundary_pos;
        const bool publish_prompt = is_prompt_boundary && ResolvePublishPromptBoundary(s);

        if (publish_prompt) {
            desired = prompt_boundary_pos;  // land exactly on B
        }
        else if (desired < ctx_end) {
            desired = desired / bs * bs;
        }
```

`prompt_boundary_node` is only ever true when `prompt_boundary_pos` was set, so
the field is always valid here. `begin < prompt_boundary_pos` still skips the
clamp once the producer's first forward has passed `B`.

**(b) `AutoCacheBoundaryPolicy::PublishPromptBoundary` (`cache_boundary_policy.cc:17`):**

```cpp
    // was: (s.prompt_len - 1) - s.last_ckpt_pos >= threshold_
    return prompt_boundary_ && threshold_ > 0 && s.prompt_boundary_pos - s.last_ckpt_pos >= threshold_;
```

Reached only via `ResolvePublishPromptBoundary`, which returns early when
`!prompt_boundary_node`, so `prompt_boundary_pos > 0` whenever it runs. Stays a
pure function of cross-rank-identical attributes.

**(c) Already geometry-driven — no functional change, comments updated:**

- `PlanForkToPopulation` / `PlanPromptBoundaryPublication` locate the block via
  `(end - 1) / bs` and target the `fork_to` node / block by its own
  `offset + size` / `offset + capacity`. With `end == B` and the node placed in
  block `j` (Section 4), these resolve to block `j` automatically.
- The consumer side (`Scheduler::Resume` fork-extension) resumes at the matched
  `fork_from` node's `offset + size`, which is now `B`. It reads node geometry,
  never `prompt_len - 1`.

Comment-only updates: `scheduler.cc:452, 949, 960, 1117, 1125, 1132` change the
wording from `prompt_len-1` to `B (prompt_len - K)`.

At `K = 1`, `prompt_boundary_pos == prompt_len - 1` for every sequence that sets
it, so (a) and (b) are bit-identical to today.

## 6. Contract sync (`src/turbomind/engine/README.md`)

`checklist.contract-sync` requires the normative doc change in the same commit.
Content-only edits (no prose re-wrapping):

- **`contracts.prefix-prepare`** — replace "The `fork_to` node excludes the last
  prompt token and ends at `prompt_len-1` ..." with: the node ends at the
  reusable boundary `B = prompt_len - cache_prompt_boundary_skip` (default 1, so
  the default excludes only the last token; `B` is capped by the `seq_len-1`
  resume cap). A partial `fork_to` node is published only when `B % block_size
  != 0`; a block-aligned `B` needs only the boundary clamp/checkpoint.
  Over-excluding is safe — segment tokens are exact-compared, so a too-long skip
  only shortens reuse and never causes a false hit.
- **`contracts.scheduler-commit`** — the clamp clause gains `B = prompt_len -
  cache_prompt_boundary_skip` in place of `prompt_len-1`.
- **`resume-selection`** — duplicate/history-extending prompt resumes via
  fork-extension at the producer's prompt-boundary node end `B = prompt_len -
  cache_prompt_boundary_skip`.
- **`concepts.boundary-policy`** — note the prompt-boundary predicate measures the
  interval to `B` (`prompt_boundary_pos`), not `prompt_len-1`.
- **`concepts`** — one line: `cache_prompt_boundary_skip` is the engine knob and
  `Sequence::prompt_boundary_pos` is its per-sequence resolved boundary `B`.

Semantics are unchanged at the default (`skip == 1` => `B == prompt_len-1`); the
contract simply gains the configurable-`B` vocabulary.

## 7. Testing

**(a) Make the new arithmetic unit-testable (small refactor).** Factor the pure
boundary decision out of `SetupForks` into a free function with no scheduler
state:

```cpp
struct PromptBoundaryPlan {
    bool valid     = false;  // a boundary exists (set prompt_boundary_node)
    bool partial   = false;  // needs a fork_to partial node (else block-aligned)
    int  pos       = 0;      // B
    int  block     = 0;      // j
    int  node_size = 0;      // partial node length when partial
};

// Pure function of geometry; mirrors Section 4's branch/guard logic.
PromptBoundaryPlan PlanPromptBoundary(int prompt_len, int block_size, int skip, int miss);
```

`SetupForks` calls it and then does only the trie/cache mutation. Add cases to
`src/turbomind/engine/test_prefix_trie.cc` (CPU, `BUILD_TEST=ON`):

- `K=1, prompt_len % bs > 1` → partial, `pos == prompt_len-1`, `block == last`
  (regression).
- `K=1, prompt_len % bs == 1` → block-aligned, `pos == last*bs`, not partial
  (regression).
- `K=2, prompt_len % bs > 2` → partial, `pos == prompt_len-2`.
- `K` pushing `B` into the prior block / block-aligned `B` / `miss` gating
  (`st.miss < j` vs `<= j`) → expected `valid`/`partial`.

**(b) Model-level regression — `scripts/test_turbomind_model.py` used AS IS.** On
a recurrent/hybrid model with `--cache-prompt-boundary` and default
`cache_prompt_boundary_skip=1`, `--max-new-tokens 128`: verify a coherent,
on-topic >=128-token response, and re-run a repeated prefix to confirm a cache
hit without output change. Proves the default path is unchanged. Pick an empty
GPU (`get_gpu_usage`/`nvidia-smi`), run outside the sandbox.

**(c) Multi-turn reuse check (new small harness, modeled on
`scripts/vlm_prefix_cache_check.py`, which already drives
`cache_prompt_boundary=True`).** On a thinking chat template with
`cache_prompt_boundary_skip` set to the template's think-suffix token count, send
two sequential turns where the think block is stripped from history, and assert
on existing INFO logs:

- turn 2 `LogAccept` shows `matched`/`fork_to@B` extending one partial block
  further than with `skip=1` (the recovered partial-tail tokens); `LogResume`
  shows `source=prefix`/`fork`.
- Oracle: greedy token-id output unchanged vs. a no-cache run (cached KV are the
  exact tensors).

**Out of scope:** any per-request/chat-template computation of the suffix length
(we use the global config); the PyTorch backend.

## Data flow summary

```
TurbomindEngineConfig.cache_prompt_boundary_skip (K, default 1)
  -> EngineConfig.cache_prompt_boundary_skip -> Scheduler.cache_prompt_boundary_skip_ (>=1)
  -> [Accept/SetupForks] B = prompt_len - K; PlanPromptBoundary -> fork_to node in block j
     (partial when B % bs != 0) or block-aligned clamp target; store Sequence.prompt_boundary_pos = B
  -> [Schedule clamp] forward lands at B; PlanPromptBoundaryPublication copies KV + checkpoint at B
  -> [next turn Accept/Resume] full blocks [0, j*bs) match; fork_from extends through the node to B
```
