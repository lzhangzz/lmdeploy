# Scheduler Unification Design

Date: 2026-07-02
Scope: `src/turbomind/engine/scheduler.{h,cc}`, `src/turbomind/engine/request.h` (one enum rename),
`src/turbomind/engine/block.h` (one new pool primitive), `src/turbomind/engine/engine.cc` (renamed call sites),
`src/turbomind/engine/README.md` (contract updates in the same change).

## Problem

After the recent partial-sibling and checkpoint-interval work, `scheduler.cc` (~1650 lines) has
accumulated complex branching. The clutter is concentrated in three places that all suffer from the
same root cause — the block vs. partial-sibling duality and position-race logic are re-derived
inline at each site instead of being expressed once:

1. `Resume` runs four sequential special-cased searches (contiguous prefix scan, frontier fast
   path, backward checkpoint walk with an interior-sibling sub-case, fork extension) that all
   answer one question: what is the highest position restorable, and what copies does it need?
2. The forward-end clamp in `RunRequiredAdmission` is a three-way precedence tangle
   (prompt-boundary clamp vs. checkpoint-due truncation vs. partial-chunk alignment) built inline
   with the `publish_prompt` flag.
3. Publication planning is routed by that same flag into two parallel planners
   (`PlanPromptBoundaryPublication` / `PlanFullBlockPublication`, plus `PlanForkToPopulation`)
   that both ask: which node ends exactly at the forward end, and does it need a checkpoint
   and/or a KV copy?

Additionally, `PublishGeneration`'s terminal checkpoint adoption carries two backward window scans
(droppable-vs-blocker classification) that are the subtlest logic in the file, and several function
names have drifted from what the functions do.

## Goals and non-goals

Goals:

- Unify the three decision tangles into declarative candidate/precedence forms with fewer branches.
- Preserve observable behavior, with one negotiated exception (terminal adoption, below).
- Rename functions whose names no longer match their behavior.
- Update `README.md` contracts in the same change.

Non-goals:

- The two-phase admission machinery (`ScratchAllocator` planning, `EvictingIterator` /
  `AllocatingIterator`, replay log, required vs. optional tiers) is kept as-is.
- No new headers; helpers stay private to `Scheduler` or file-local in `scheduler.cc`.
- No feature drops beyond the terminal-adoption change below.

## Behavior change (the one negotiated drop)

Terminal checkpoint adoption in `PublishGeneration` currently classifies in-window checkpoints
below `filled_len` as droppable (private block: deallocate + drop ref) or blocking (shared block:
skip adoption to preserve `checkpoint_min_interval` spacing). Both scans are removed:

- Adopt the terminal frontier checkpoint **unconditionally** whenever the terminal block is being
  indexed, the frontier allocation is valid, and the block's checkpoint slot is empty.
- If another valid checkpoint lies within `checkpoint_min_interval` below `filled_len`, the
  adopted checkpoint is **demoted to evict-first priority** (timestamp 0) instead of being
  suppressed. The interval constraint is relaxed at sequence finalization only; the redundant
  checkpoint never displaces other cache state because any eviction pass reclaims it first.

Consequence: checkpoints near a finished sequence's tail may sit closer than the interval,
retaining up to one extra (evict-first, unprotected) checkpoint slot per finished sequence.

## Design

### 1. Resume: candidate enumeration

One value type replaces the four sequential searches:

```cpp
struct ResumeCandidate {
    int           pos{};                            // resume position (token)
    ResumeSource  source{ResumeSource::kNone};
    int           ckpt_id{};                        // checkpoint to restore into the frontier; 0 = none
    LogicalBlock* fork_src{};                       // sibling KV to copy from; nullptr = none
    LogicalBlock* fork_dst{};                       // block receiving the KV copy
};
```

`PlanResume` (renamed from `Resume`) keeps its existing structure: cache-id reservation, the
contiguous-prefix scan producing `prefix_end` / `readonly_block_num` (step 1, unchanged), then
candidate generation and selection replace steps 2–3:

```cpp
ResumeCandidate best{};  // pos 0, kNone

// Fork extension first (highest precedence, short-circuits everything else):
// copy an indexed, valid sibling's KV into the block at the prefix boundary.
// A feasible extension ends strictly past prefix_end while every other
// candidate is capped at prefix_end, so it always wins; applies with or
// without checkpointing.
if (prefix_end % bs == 0 && prefix_end / bs < (int)s.block_ids.size()) {
    LogicalBlock& x = *s.block_ids[prefix_end / bs];
    if (LogicalBlock* y = x.partial.get()) {
        const int e = y->offset + y->size;
        if (y->is_valid && e <= upper && e > prefix_end && ValidAlloc(y->prefix_id)
            && (!ckpt || ValidAlloc(y->checkpoint_id))) {
            best = {e, ResumeSource::kFork, ckpt ? y->checkpoint_id : 0, y, &x};
        }
    }
}

if (best.pos == 0) {
    if (!ckpt) {
        // Without checkpointing, KV grants per-token resume anywhere in the prefix.
        best = {prefix_end, prefix_end > 0 ? ResumeSource::kPrefix : ResumeSource::kNone};
    }
    else {
        // Frontier: live state, no copy.
        if (const int fpos = s.frontier_pos - s.inflight_input_len;
            ValidAlloc(s.frontier_cache_id) && 0 < fpos && fpos <= prefix_end) {
            best = {fpos, ResumeSource::kFrontier};
        }
        // Published checkpoints covered by the valid prefix, scanned backward.
        // A block yields its own (block-end) checkpoint and its interior
        // partial sibling's checkpoint as the same kind of candidate; both are
        // checkpoint-only restores (KV is covered by the valid prefix, so no
        // KV copy and no is_valid requirement). A sibling-sourced resume
        // reports kFork. Block ends strictly decrease going backward and a
        // sibling is strictly shorter than its block, so once a block cannot
        // beat best (e <= best.pos) nothing earlier can either; the first hit
        // ends the walk via the same check on the next iteration.
        for (int i = std::min<int>(s.block_ids.size(), (prefix_end + bs - 1) / bs); i > 0; --i) {
            const LogicalBlock& x = *s.block_ids[i - 1];
            const int           e = x.key ? x.offset + x.size : x.offset + x.capacity;
            if (e <= best.pos) {
                break;
            }
            if (e <= prefix_end && ValidAlloc(x.checkpoint_id)) {
                best = {e, ResumeSource::kCheckpoint, x.checkpoint_id};
                break;
            }
            if (const LogicalBlock* y = x.partial.get()) {
                const int ye = y->offset + y->size;
                if (ye <= prefix_end && ye > best.pos && ValidAlloc(y->checkpoint_id)) {
                    best = {ye, ResumeSource::kFork, y->checkpoint_id};
                    break;
                }
            }
        }
    }
}
```

Selection is strict `>` on `pos`. Fork extension short-circuits: when feasible it ends strictly
past `prefix_end`, which no other candidate can reach, so evaluating it first is exactly
equivalent to the old last-with-strict-`>` placement and skips the candidate loop entirely. The
checkpoint walk stays backward with the old early exits (`e <= best.pos` prunes everything
earlier; a hit breaks immediately, since all remaining candidates are strictly smaller), seeded
by the frontier candidate — which is how the frontier beats an equal-position checkpoint (no
copy needed) and a block's own checkpoint dominates its same-block sibling. The interior-sibling
case, previously a special sub-branch of the walk, is now just another candidate — reported as
`kFork` since the resume point comes from a partial sibling node (previously logged as
`checkpoint`; every sibling-sourced resume now uniformly reports `fork`).

Steps 4–5 (restore copy plans, allocation and protection sets) consume `best`:

```cpp
s.resume_len    = best.pos;
s.resume_source = best.source;

if (best.fork_dst) {
    s.restore_copies.push_back({best.fork_src->prefix_id, best.fork_dst->prefix_id});
}
if (ckpt && best.pos > 0 && best.ckpt_id) {
    s.restore_copies.push_back({best.ckpt_id, s.frontier_cache_id});
    s.last_ckpt_pos = std::max(s.last_ckpt_pos, best.pos);  // spacing measured from the restore
}
```

The allocation/protection loop (prefix blocks + frontier) is unchanged.

### 2. Forward-end clamp: one pure function

The inline branching in `RunRequiredAdmission` moves into a private const member function with an
explicit precedence list:

```cpp
// Land the forward end on a boundary candidate. Precedence:
//   1. Prompt-boundary clamp: the boundary node is armed and this pass reaches
//      B -> land exactly on B (>= so an exact landing is not truncated away).
//   2. Checkpoint-due alignment: when checkpoint bytes are registered and a
//      prompt-region pass would run past the due position
//      (last_ckpt_pos + checkpoint_min_interval), end on the last block
//      boundary in the admitted range — at or past the due position and
//      strictly past begin (progress guarantee) — so the full-block checkpoint
//      can be taken there; the remainder runs next pass.
//   3. Partial-chunk alignment: a pass that does not reach the context end
//      lands on a block boundary.
//   4. Otherwise: run to desired.
// A result <= begin means nothing runs this pass.
int Scheduler::ClampForwardEnd(const Sequence& s, int begin, int desired, int ctx_end) const
{
    if (s.prompt_boundary_node && begin < s.prompt_boundary_pos && desired >= s.prompt_boundary_pos) {
        return s.prompt_boundary_pos;
    }
    const int bs      = logical_.block_size();
    const int aligned = desired / bs * bs;
    const int due     = s.last_ckpt_pos + registry_.checkpoint_min_interval();
    if (CheckpointPublicationEligible() && registry_.has_checkpoint() && desired <= s.prompt_len
        && desired > due && aligned >= due && aligned > begin) {
        return aligned;
    }
    if (desired < ctx_end) {
        return aligned;
    }
    return desired;
}
```

The admission loop keeps only:

```cpp
const int end = ClampForwardEnd(s, begin, begin + admitted, ctx_end);
const int len = end - begin;
if (len <= 0) {
    continue;  // nothing admitted this pass; CommitResults leaves it inactive
}
s.input_len = len;

const bool at_prompt_boundary = s.prompt_boundary_node && end == s.prompt_boundary_pos;
```

(This reproduces the old `publish_prompt`, which required `begin < B && desired >= B` before
setting `end = B`. After clamping: `end == B` implies `desired >= B` (the clamp never raises the
end), and `len > 0` guarantees `begin < end == B`; conversely a pass with `begin >= B` has
`end > B`. So `end == B` alone is equivalent.)

### 3. Publication: one planner keyed on "the node ending at end"

`PlanPromptBoundaryPublication`, `PlanFullBlockPublication`, and `PlanForkToPopulation` merge into
one function. The mutually-exclusive routing survives as the `at_prompt_boundary` argument:

```cpp
void Scheduler::PlanPublication(ScheduleState& pass, int i, Sequence& s, int end, bool at_prompt_boundary)
{
    LogicalBlock& x        = *s.block_ids[(end - 1) / logical_.block_size()];
    const bool    at_block = x.offset + x.capacity == end;
    LogicalBlock* sibling =
        (!at_block && x.partial && x.partial->offset + x.partial->size == end) ? x.partial.get() : nullptr;
    LogicalBlock* node = at_block ? &x : sibling;

    // (a) Population: an indexed, not-yet-populated partial sibling at the
    // prompt boundary receives this request's partial KV via a device copy.
    // pass.planned dedups intent across requests sharing the node this pass;
    // the slot itself is allocated in the optional phase from inactive memory.
    if (at_prompt_boundary && sibling && !sibling->is_valid && !ValidAlloc(sibling->prefix_id)
        && !pass.planned.count(sibling->prefix_id)) {
        pass.planned.insert(sibling->prefix_id);
        pass.pending_populate[i] = sibling;
        pass.has_optionals       = true;
    }

    // (b) Checkpoint onto the node. The prompt-boundary pass bypasses the min
    // interval; the full-block path requires a block-aligned end and is subject
    // to the interval and to cache_generation=none suppression of
    // generation-region checkpoints.
    if (!CheckpointPublicationEligible() || s.publish_cache_id == 0 || node == nullptr) {
        return;
    }
    if (!at_prompt_boundary) {
        if (!at_block) {
            return;  // full-block group: no full block ends here
        }
        if (generation_cache_mode_ == CacheMode::kNone && end > s.prompt_len) {
            return;  // generated blocks are never indexed under 'none'
        }
        if (end - s.last_ckpt_pos < registry_.checkpoint_min_interval()) {
            return;
        }
    }
    if (!ValidAlloc(node->checkpoint_id)) {
        pass.pending_publish[i] = {node, end, s.publish_cache_id};
        pass.has_optionals      = true;
    }
}
```

Call site in the admission loop (replacing the two-way routing):

```cpp
PlanPublication(pass, i, s, end, at_prompt_boundary);
```

Equivalence notes:

- Old prompt-boundary path targeted `&x` when B is block-aligned, else the matching sibling —
  identical to `node` above.
- Old fork-to population had the same geometry guard (`y.offset + y.size == end`), `is_valid`,
  `ValidAlloc`, and `planned` checks, and only ran on the prompt-boundary route.
- Old full-block path required `x.offset + x.capacity == end`, applied the `none` suppression and
  the min interval — identical to the `!at_prompt_boundary` block above.

### 4. Terminal adoption: adopt always, demote on interval violation

In `Finalize` (renamed from `PublishGeneration`), the adoption block becomes:

```cpp
if (publish_generation_boundary && x.offset + size == s.filled_len && ValidAlloc(s.frontier_cache_id)
    && !ValidAlloc(x.checkpoint_id)) {
    if (const int stale = x.checkpoint_id) {
        cache_.Invalidate(stale);  // evicted leftover slot
    }
    const int f     = std::exchange(s.frontier_cache_id, 0);
    x.checkpoint_id = f;
    cache_[f].owner = up;
    logical_.Retain(up);  // ref held by the live allocation
    gen.terminal_ckpt = true;

    // If another valid checkpoint lies within checkpoint_min_interval
    // below filled_len, this adoption undercuts the interval. Keep it
    // (terminal state is the best resume point) but demote it to
    // evict-first priority so the redundancy is reclaimed first while
    // it remains demoted.
    const int interval = registry_.checkpoint_min_interval();
    for (int j = static_cast<int>(i); j >= 0; --j) {
        const LogicalBlock& p         = *s.block_ids[j];
        const int           block_pos = p.offset + p.size;
        if (block_pos < s.filled_len) {
            if (s.filled_len - block_pos >= interval) {
                break;  // outside the window; earlier block/partial positions are older
            }
            if (ValidAlloc(p.checkpoint_id)) {
                cache_.Demote(f);
                gen.demoted = true;  // observability (LogFinalized)
                break;
            }
        }

        if (const LogicalBlock* y = p.partial.get()) {
            const int pos = y->offset + y->size;
            if (pos < s.filled_len && s.filled_len - pos < interval && ValidAlloc(y->checkpoint_id)) {
                cache_.Demote(f);
                gen.demoted = true;  // observability (LogFinalized)
                break;
            }
        }
    }
}
```

`GenStat` drops `dropped` and gains `bool demoted`; `LogFinalized` reports `", terminal ckpt (demoted)"` in place of the dropped count.

New pool primitive:

```cpp
// CacheBlockPool: demote a slot to evict-first priority. Timestamp 0 sorts
// first in SortedIndices() and is below every eviction cutoff and pass floor,
// so both admission phases reclaim it before any other slot.
void Demote(int cache_id)
{
    blocks_[cache_id].timestamp = 0;
}
```

(`block.h` already documents `CacheBlock::timestamp` as "eviction priority; zero means highest",
and `next_timestamp_` starts at 1, so 0 is never handed out by `Stamp` and is safe as the
demotion value.)

### 5. Collision rollback helper

Two sites (Accept full-block insert and generation indexing) roll back an un-inserted block
identically; the prompt-boundary insert discards its node instead (no rollback). Fold the two
into a file-local helper:

```cpp
// Roll a block back to private (un-indexed) state after a failed trie insert.
void UnindexBlock(LogicalBlock& x)
{
    x.parent = nullptr;
    x.key    = {};
    x.size   = 0;
    x.tokens.clear();
    x.image_fps.clear();
}
```

### 6. Renames

| Old                                                                                 | New                    | Why                                                                                                             |
| ----------------------------------------------------------------------------------- | ---------------------- | --------------------------------------------------------------------------------------------------------------- |
| `Accept`                                                                            | `AdmitPrompt`          | It binds the prompt to the trie at admission; "Accept" reads as request admission.                              |
| `Resume`                                                                            | `PlanResume`           | Pure planning (cache-prepare phase); groups with `PlanRequests` / `PlanPublication`.                            |
| `Continue`                                                                          | `PlanContinue`         | Same.                                                                                                           |
| `PublishGeneration`                                                                 | `Finalize`             | Indexes generated blocks and adopts the terminal checkpoint on normal finish; the log already says "finalized". |
| `Publish(s, t0, end)`                                                               | `MarkProduced`         | Clears producer marks, flips `is_valid`; frees "publish" for checkpoint publication, a different mechanism.     |
| `SetupForks`                                                                        | `SetupPartialSiblings` | Finishes the fork → partial-sibling terminology migration (`0c311642`).                                         |
| `CreateMissingBlocks`                                                               | `IndexMissingBlocks`   | It creates and indexes; indexing is the part that matters.                                                      |
| `ScheduleState::pending_fork`                                                       | `pending_populate`     | Holds the sibling node awaiting KV population; reads next to `pending_publish`.                                 |
| `PlanPromptBoundaryPublication`, `PlanFullBlockPublication`, `PlanForkToPopulation` | `PlanPublication`      | Merged (Section 3).                                                                                             |

Kept: `Schedule`, `PlanRequests`, `RunRequiredAdmission`, `RunOptionalAdmission`, `ReplayMemory`,
`CommitResults`, `EnsureBlocks`, `MatchPrompt`, `SetProducers` / `CheckProducers`, `Release`,
`ResumeSource::kFork` and the candidate's `fork_src` / `fork_dst` ("fork" accurately names the
resume action; only the edge storage migrated to "partial sibling").

Renamed call sites: `engine.cc` (`Accept`, `Resume`/`Continue` are invoked via the scheduler from
`Schedule` only — external call sites are `Accept`, `PublishGeneration`, `Release`).
`Scheduler::PublishStat` doc comments referring to `Publish()` follow the rename.

### 7. README contract updates (same change)

- `contracts.checkpoint-adoption`: rewritten — adoption is unconditional (frontier valid, slot
  empty, terminal block indexed); the droppable/blocker classification is removed; an in-window
  redundant checkpoint demotes the adopted checkpoint to evict-first priority instead.
- `contracts.checkpoint-publish`: note the terminal-adoption interval exception (spacing enforced
  at publication time; adoption may undercut it with a demoted slot).
- `contracts.cache-eviction`: document the demotion primitive (timestamp-0 slots are reclaimed
  first by both phases).
- Function-name references across `concepts.scheduler-transaction`, `principles.scheduler-boundary`,
  `contracts.prefix-prepare`, `contracts.cache-prepare`, `contracts.scheduler-commit`,
  `contracts.resume-selection`, `contracts.prefix-publish`, `checklist.cache-prepare`: follow the
  renames (`Accept` → `AdmitPrompt`, `Resume` → `PlanResume`, `Continue` → `PlanContinue`,
  `PublishGeneration` → `Finalize`). Behavior text in `scheduler-commit` stays (behavior
  unchanged); wording may be touched only to name the candidate model.
- `contracts.resume-selection`: two wording updates — fork extension is the highest-precedence
  resume source (a feasible extension always ends past the valid prefix, so it dominates), and
  every sibling-sourced resume reports `source=fork`, including the interior partial-sibling
  checkpoint-only restore that previously reported `source=checkpoint` (`kCheckpoint` is now
  exclusively a block's own checkpoint).

## Testing

- Build: `ninja` in `build/`.
- `scripts/test_turbomind_model.py` (as is), per repo rules: at least 128 response tokens,
  response verified meaningful. Two model classes:
  - a recurrent (GDN) model — exercises frontier/checkpoint/fork candidates, checkpoint-due
    truncation, terminal adoption + demotion;
  - a plain-attention model — exercises the `!ckpt` prefix candidate path.
- Repeated-prompt runs on the same server process to exercise prefix reuse, partial-sibling
  population, and resume-from-checkpoint (verified via the cache log lines: `resume ... source=`,
  `published ...`, `finalized ...`).
- `test_prefix_trie` unit test still builds and passes (trie untouched, but `UnindexBlock` and
  rename fallout must compile).

## Expected outcome

- `scheduler.cc` shrinks by roughly 150 lines; the three decision sites become declarative
  (candidate list + max selection; ordered precedence list; single node-at-end planner).
- One behavior change, bounded and documented: terminal adoption never skips, and an
  interval-violating terminal checkpoint is evict-first.

## Follow-up: cache-id ownership model

Date: 2026-07-02 (post-implementation audit)

The implemented publication path carried two "stale slot" `Invalidate` sites (`CommitResults`
attach, `Finalize` adoption) whose safety rests on an unchecked single-holder convention: an
owner-attached checkpoint id is only ever invalidated by code that overwrites the holding field in
the same statement group. The convention is correct today but fragile (a violation corrupts the
free list silently). Replace it with a strict ownership model:

1. A cache id is owned by exactly one entity: a `LogicalBlock` (its `prefix_id` / `checkpoint_id`
   slots, `CacheBlock::owner` set) or a `Sequence` (`frontier_cache_id`, `owner == nullptr`).
2. An id is invalidated (returned to the pool free list) only when its owner is destroyed:
   `LogicalBlockPool::Recycle` for block-owned ids, `Scheduler::Release` for sequence-owned ids.
   Ownership may be transferred between entities (a single `std::exchange`-shaped move that
   updates both the field and `CacheBlock::owner`), which moves the invalidation duty with it.
3. Nothing else invalidates a cache id. Eviction deallocates backing memory but the id stays
   bound to its owner as an unallocated ("zombie") slot; consumers always test `ValidAlloc`.

### Checkpoint slots become block-owned, allocated in place (the prefix-slot model)

`Sequence::publish_cache_id` is removed. A block's `checkpoint_id` is created lazily at first
publication planning — owner set to the node at `Create` — and re-allocated in place forever
after, exactly like `prefix_id`:

```cpp
// PlanPublication, checkpoint group (replaces the s.publish_cache_id gate):
if (!CheckpointPublicationEligible() || !registry_.has_checkpoint() || node == nullptr) {
    return;
}
...gates...
if (!ValidAlloc(node->checkpoint_id)) {
    if (node->checkpoint_id == 0) {
        node->checkpoint_id = cache_.Create(registry_.checkpoint().object_id(), node);
    }
    TM_CHECK(pass.planned.insert(node->checkpoint_id).second);  // one publisher per node per pass
    pass.pending_publish[i] = {node, end, node->checkpoint_id};
    pass.has_optionals      = true;
}
```

At most one request can plan a given node per pass: a block target is producer-excluded (the
committed forward writes `end-1` inside it), and a sibling target is reachable only by the
request whose trie insert created the boundary node (first-wins arming of
`prompt_boundary_node`). The checked `pass.planned` reservation turns any violation of that
argument into a crash instead of a silent double-allocation in the optional phase.

Because the slot is owner-attached at creation, `ReplayMemory` takes/drops the allocation
reference uniformly (`logical_.Retain(c.owner)` on alloc, `Drop` on evict) — the manual
`Retain` at attach disappears. The `CommitResults` attach shrinks to metadata + copy:

```cpp
if (s.publish_target) {
    // The slot (publish_target->checkpoint_id) was allocated by the optional
    // phase. Single-publisher-per-node-per-pass is enforced at plan time,
    // and planning required !ValidAlloc, so no dedup branch is needed.
    s.last_ckpt_pos = s.publish_end;
    ckpt_published  = true;
    s.publish_copies.push_back({s.frontier_cache_id, s.publish_target->checkpoint_id});
    cache_.Stamp(s.publish_target->checkpoint_id);
    s.publish_target = nullptr;
    s.publish_end    = 0;
}
```

Deleted outright: the `ValidAlloc(t.checkpoint_id)` dedup + `ReleaseCacheId(publish_cache_id)`
branch, the stale `Invalidate`, the ownership transfer, the manual `Retain`, the publish-id
creation in `PlanResume` / `PlanContinue` (the `PlanContinue` checkpoint block empties out and is
removed), and the publish-id release in `Release`.

### Terminal adoption transfers ownership both ways

Adoption in `Finalize` moves the sequence-owned frontier id into the block (sequence -> block
transfer). If the block already holds a zombie checkpoint id (created but unallocated), that id
moves the other way and rides the dying sequence's frontier field to `Release` (block -> sequence
transfer), where it is invalidated with its new owner:

```cpp
if (publish_generation_boundary && x.offset + size == s.filled_len && ValidAlloc(s.frontier_cache_id)
    && !ValidAlloc(x.checkpoint_id)) {
    const int f = std::exchange(s.frontier_cache_id, 0);
    if (const int zombie = std::exchange(x.checkpoint_id, 0)) {
        cache_[zombie].owner = nullptr;  // now sequence-owned; invalidated at Release
        s.frontier_cache_id  = zombie;
    }
    x.checkpoint_id = f;
    cache_[f].owner = up;
    logical_.Retain(up);  // the live allocation was taken owner-less; ref moves with ownership
    gen.terminal_ckpt = true;
    ...demotion scan unchanged...
}
```

The adopted frontier keeps its manual `Retain`: its allocation was committed while the slot was
sequence-owned (`owner == nullptr`, no ref), so the block->allocation ref is established at
transfer. `ReleaseCacheId` keeps its `TM_CHECK(c.owner == nullptr)` and now serves the frontier
and swapped-in zombies only.

### Invariants after the change

- `Invalidate` call sites: `Recycle` (block owner dies), `ReleaseCacheId` from `Release`
  (sequence owner dies). No others.
- Every persistent id field is single-holder; transfers are exchanges that update `owner` in the
  same statement group. No id can be observed after passing through the free list.

### README updates (same change)

- `ownership.prefix`: state the three ownership rules; drop the `CommitResults` clause from the
  allocation-ref enumeration (publication refs now flow through `ReplayMemory` uniformly).
- `contracts.checkpoint-publish`: planning targets the block-owned checkpoint slot, created
  lazily and re-allocated in place; no request-owned publication id exists.
- `contracts.checkpoint-adoption`: document the two-way transfer (frontier in, zombie out via the
  finished sequence's frontier field).

### Testing

Same as the main plan: `ninja`, `test_prefix_trie`, and a GDN model scenario via
`scripts/test_turbomind_model.py` with repeated prompts (resume/publish/finalize log lines
verified, responses meaningful, >= 128 tokens).
