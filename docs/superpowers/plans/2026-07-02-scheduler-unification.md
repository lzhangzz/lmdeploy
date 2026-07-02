# Scheduler Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify the TurboMind scheduler by unifying its three branching tangles (resume step selection, forward-end clamping, publication planning) into declarative forms, rename drifted function names, and replace terminal-adoption window scans with adopt-always + evict-first demotion.

**Architecture:** Per spec `docs/superpowers/specs/2026-07-02-scheduler-unification-design.md`. Six code tasks, each behavior-preserving except Task 6 (the one negotiated behavior change): (1) mechanical renames, (2) `UnindexBlock` helper, (3) `ResumeCandidate` model in `PlanResume`, (4) `ClampForwardEnd` extraction, (5) `PlanPublication` merge, (6) `CacheBlockPool::Demote` + unconditional terminal adoption. Task 7 is GPU verification. README contract updates land in the same task as the code they describe.

**Tech Stack:** C++ (TurboMind engine), ninja build in `build/`, GPU verification via `scripts/test_turbomind_model.py` (script must be used AS IS).

**Constraints (from AGENTS.md):** never install lmdeploy or run `setup.py`; GPU commands run outside the sandbox; check `nvidia-smi` for idle GPUs before any GPU run; every model test must produce meaningful text (≥ 128 requested tokens); do not re-wrap README lines — edit content only.

**Models (from `/data/models.json`, HF cache dir `/mnt_cfs/huggingface_hub/hub/`):** `Qwen/Qwen3.5-27B` (hybrid GDN — exercises checkpoint/frontier/fork paths), `Qwen/Qwen3-8B` (plain attention — exercises the no-checkpoint prefix path).

There is no C++ unit-test harness for the scheduler; verification is behavioral via the cache WARN logs (`LogResume`/`LogPublished`/`LogFinalized` in `scheduler.cc`) plus response quality. Tasks 1–6 are gated by clean builds; Task 7 is the behavioral check for the whole stack.

---

### Task 1: Mechanical renames (behavior-preserving)

**Files:**
- Modify: `src/turbomind/engine/scheduler.h`
- Modify: `src/turbomind/engine/scheduler.cc`
- Modify: `src/turbomind/engine/engine.cc` (call sites at lines 408, 707)
- Modify: `src/turbomind/engine/request.h` (comments only)
- Modify: `src/turbomind/engine/README.md` (code-reference names only)

- [ ] **Step 1: Apply the rename mapping to `scheduler.h` and `scheduler.cc`**

Rename declarations, definitions, call sites, and comment references. Mapping (old → new):

| Old | New |
| --- | --- |
| `Scheduler::Accept` | `Scheduler::AdmitPrompt` |
| `Scheduler::Resume` | `Scheduler::PlanResume` |
| `Scheduler::Continue` | `Scheduler::PlanContinue` |
| `Scheduler::PublishGeneration` | `Scheduler::Finalize` |
| `Scheduler::Publish` (the `(Sequence&, int t0, int end)` overload) | `Scheduler::MarkProduced` |
| `Scheduler::SetupForks` | `Scheduler::SetupPartialSiblings` |
| `Scheduler::CreateMissingBlocks` | `Scheduler::IndexMissingBlocks` |
| `ScheduleState::pending_fork` | `ScheduleState::pending_populate` |

Notes:
- In `scheduler.cc`, internal calls live in `AdmitPrompt` (calls `MatchPrompt`, `IndexMissingBlocks`, `SetupPartialSiblings`), `PlanRequests` (calls `PlanContinue`/`PlanResume`), `CommitResults` (calls `MarkProduced`), and `RunOptionalAdmission`/`CommitResults` (use `pending_populate`).
- Update comments that name the old functions, e.g. the `PublishStat` field comments in `scheduler.h` (`Publish()` → `MarkProduced()`; `CommitResults()` stays), the header comments above `Accept`/`Schedule`/`PublishGeneration`/`Resume`/`Continue` declarations, and `scheduler.cc` comments such as "Publish flips is_valid", "publication, Publish", "PlanFullBlockPublication checkpoints there" (leave `PlanFullBlockPublication` alone — it is removed in Task 5; rename only the eight identifiers above).
- Do NOT rename: `ResumeSource::kFork`, `fork_src`/`fork_dst` locals, `PlanForkToPopulation` / `PlanPromptBoundaryPublication` / `PlanFullBlockPublication` (deleted in Task 5), `Publish` in the identifiers `publish_cache_id` / `publish_target` / `publish_end` / `publish_copies` / `PublishPlan` / `pending_publish` (checkpoint publication keeps the word).

- [ ] **Step 2: Update `engine.cc` call sites**

Line 408: `scheduler_.Accept(*x);` → `scheduler_.AdmitPrompt(*x);`
Line 707: `scheduler_.PublishGeneration(c);` → `scheduler_.Finalize(c);`

Then check for stale comment references:

Run: `rg -n "Accept\(|SetupForks|PublishGeneration|CreateMissingBlocks|Scheduler::Resume|Scheduler::Continue" src/turbomind/engine/engine.cc src/turbomind/engine/engine.h`
Expected: no matches (fix any comment hits with the mapping).

- [ ] **Step 3: Update `request.h` comments**

Line 235: `// set at Accept: leading prompt blocks found in trie` → `// set at AdmitPrompt: leading prompt blocks found in trie`
Line 236: `// transient: planned by Resume() this pass` → `// transient: planned by PlanResume() this pass`
Lines 159–160 (`ResumeSource` doc comment): `Scheduler::Resume()` → `Scheduler::PlanResume()`
Line 249 (in the `prompt_boundary_node` comment): `Decided in SetupForks.` → `Decided in SetupPartialSiblings.`

- [ ] **Step 4: Update README code references (content only, no re-wrapping)**

In `src/turbomind/engine/README.md`, apply the mapping to code references only — backticked names and `Scheduler::`-qualified names. Affected spellings:

- `` `Scheduler::Accept()` `` / `` `Accept` `` / `Accept-time` (in `principles.boundary-policy`) → `AdmitPrompt` forms. Do not touch prose uses of "accept/accepted" (gateway/admission text).
- `` `Resume` ``, `` `Resume()` ``, `` `Scheduler::Resume()` `` → `PlanResume` forms; `` `Continue` `` → `` `PlanContinue` `` (only where it names the scheduler method).
- `` `PublishGeneration` `` / `` `Scheduler::PublishGeneration()` `` → `Finalize` forms.
- `` `SetupForks` `` → `` `SetupPartialSiblings` ``.

Run: `rg -n "SetupForks|PublishGeneration|Scheduler::Resume|Scheduler::Accept|\`Accept\`|\`Resume\`|\`Continue\`" src/turbomind/engine/README.md`
Expected: no matches.

- [ ] **Step 5: Build**

From `build/`: `ninja`. Expected: exit 0.

- [ ] **Step 6: Commit**

```bash
git add src/turbomind/engine/scheduler.h src/turbomind/engine/scheduler.cc src/turbomind/engine/engine.cc src/turbomind/engine/request.h src/turbomind/engine/README.md
git commit -m "refactor(scheduler): rename drifted function names (AdmitPrompt/PlanResume/PlanContinue/Finalize/MarkProduced)"
```

### Task 2: `UnindexBlock` rollback helper

**Files:**
- Modify: `src/turbomind/engine/scheduler.cc` (anonymous namespace; two rollback sites in `IndexMissingBlocks` and `Finalize`)

- [ ] **Step 1: Add the helper to the anonymous namespace** (next to `CollectStartFps`)

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

- [ ] **Step 2: Replace the rollback in `IndexMissingBlocks`**

```cpp
            if (!trie_.Insert(x)) {
                LogCollision(s, CollisionSite::kAccept, offset, offset + size);
                // Stays un-indexed; treated as a private block from here on.
                x.parent = nullptr;
                x.key    = {};
                x.size   = 0;
                x.tokens.clear();
                x.image_fps.clear();
            }
```

becomes

```cpp
            if (!trie_.Insert(x)) {
                LogCollision(s, CollisionSite::kAccept, offset, offset + size);
                // Stays un-indexed; treated as a private block from here on.
                UnindexBlock(x);
            }
```

- [ ] **Step 3: Replace the rollback in `Finalize`**

```cpp
        if (!trie_.Insert(x)) {
            LogCollision(s, CollisionSite::kPublish, x.offset, x.offset + size);
            x.parent = nullptr;
            x.key    = {};
            x.size   = 0;
            x.tokens.clear();
            x.image_fps.clear();
            break;
        }
```

becomes

```cpp
        if (!trie_.Insert(x)) {
            LogCollision(s, CollisionSite::kPublish, x.offset, x.offset + size);
            UnindexBlock(x);
            break;
        }
```

(The third insert site, the prompt-boundary node in `SetupPartialSiblings`, discards its freshly created node on collision instead of rolling back — leave it as is.)

- [ ] **Step 4: Build and commit**

From `build/`: `ninja`. Expected: exit 0.

```bash
git add src/turbomind/engine/scheduler.cc
git commit -m "refactor(scheduler): fold duplicated trie-insert rollback into UnindexBlock"
```

### Task 3: `ResumeCandidate` model in `PlanResume`

**Files:**
- Modify: `src/turbomind/engine/scheduler.cc` (anonymous namespace + `PlanResume` sections 2–4)
- Modify: `src/turbomind/engine/README.md` (`contracts.resume-selection`)

- [ ] **Step 1: Add the candidate type to the anonymous namespace** (next to `UnindexBlock`)

```cpp
// One feasible resume position with the copies it needs. Selection is strict
// > on pos; kNone/pos 0 is the empty candidate.
struct ResumeCandidate {
    int           pos{};                            // resume position (token)
    ResumeSource  source{ResumeSource::kNone};
    int           ckpt_id{};                        // checkpoint to restore into the frontier; 0 = none
    LogicalBlock* fork_src{};                       // sibling KV to copy from; nullptr = none
    LogicalBlock* fork_dst{};                       // block receiving the KV copy
};
```

- [ ] **Step 2: Replace sections 2–4 of `PlanResume`**

Delete everything from the comment `// 2. Resume step selection` through the end of the block that seeds `s.last_ckpt_pos` (i.e. the `step`/`source`/`fork_dst`/`fork_src`/`restore_ckpt` declarations, the `if (ckpt) { ... }` step-selection block, the `// 3. Fork extension ...` block, the `s.resume_len` / `s.resume_source` assignments, and the `// 4. Restore copy plans` block ending with the `// step == 0 with checkpointing: ...` comment). Replace with:

```cpp
    // 2. Resume candidate selection: strict > on pos.
    ResumeCandidate best{};

    // Fork extension first (highest precedence): copy an indexed, valid
    // sibling's KV into the block at the prefix boundary. A feasible extension
    // ends strictly past prefix_end while every other candidate is capped at
    // prefix_end, so it always wins; short-circuit. Applies with or without
    // checkpointing. The target may be a shared indexed node whose KV was
    // evicted (is_valid == false); the restore copy re-populates it and
    // MarkProduced flips is_valid after the forward proves content.
    if (prefix_end % bs == 0 && prefix_end / bs < static_cast<int>(s.block_ids.size())) {
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
            if (prefix_end > 0) {
                best = {prefix_end, ResumeSource::kPrefix};
            }
        }
        else {
            // Frontier: live state, no copy; seeding best makes it beat an
            // equal-position checkpoint.
            if (const int fpos = s.frontier_pos - s.inflight_input_len;
                ValidAlloc(s.frontier_cache_id) && 0 < fpos && fpos <= prefix_end) {
                best = {fpos, ResumeSource::kFrontier};
            }
            // Published checkpoints covered by the valid prefix, scanned
            // backward. A block yields its own (block-end) checkpoint and its
            // interior partial sibling's checkpoint as the same kind of
            // candidate; both are checkpoint-only restores (KV is covered by
            // the valid prefix, so no KV copy and no is_valid requirement). A
            // sibling-sourced resume reports kFork. Block ends strictly
            // decrease going backward and a sibling is strictly shorter than
            // its block, so once a block cannot beat best (e <= best.pos)
            // nothing earlier can either, and any hit ends the walk.
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

    s.resume_len    = best.pos;
    // source is kNone exactly when pos == 0, so no extra guard is needed.
    s.resume_source = best.source;

    // 3. Restore copy plans (cache ids; resolved to pointers at setup)
    if (best.fork_dst) {
        s.restore_copies.push_back({best.fork_src->prefix_id, best.fork_dst->prefix_id});
    }
    if (ckpt && best.pos > 0 && best.ckpt_id) {
        s.restore_copies.push_back({best.ckpt_id, s.frontier_cache_id});
        // Measure recurrent-checkpoint spacing from the restored position, not
        // from 0: without this a fresh request resuming deep into a shared
        // prefix believes a checkpoint is immediately due.
        s.last_ckpt_pos = std::max(s.last_ckpt_pos, best.pos);
    }
    // best.pos == 0 with checkpointing: GDN recognizes a forward starting at
    // position 0 (history_len + inflight_input_len == 0) and resets.
```

Renumber the following section comment `// 5. Allocation set and eviction-protection set ...` to `// 4. ...`.

- [ ] **Step 3: Update `contracts.resume-selection` in README.md** (content only, no re-wrapping)

Replace the sentence:

> When a matched indexed block carries a `partial` sibling whose end `ye` lies within the contiguous valid prefix (`ye <= prefix_end`), `Resume()` may select that sibling's published checkpoint as `resume_len` (`source=checkpoint`, checkpoint restore only, no KV copy). Fork-extension at `B` applies when the sibling extends past `prefix_end` or KV must be repopulated (`source=fork`).

with:

> When a matched indexed block carries a `partial` sibling whose end `ye` lies within the contiguous valid prefix (`ye <= prefix_end`), `PlanResume()` may select that sibling's published checkpoint as `resume_len` (`source=fork`, checkpoint restore only, no KV copy); every sibling-sourced resume reports `source=fork`, and `source=checkpoint` is exclusively a block's own checkpoint. Fork-extension at `B` — when the sibling extends past `prefix_end` or KV must be repopulated — is the highest-precedence resume source: a feasible extension always ends past the valid prefix, so it dominates every frontier and checkpoint candidate.

(If Task 1 already turned `Resume()` into `PlanResume()` in this sentence, match on the renamed text.)

- [ ] **Step 4: Build and commit**

From `build/`: `ninja`. Expected: exit 0.

```bash
git add src/turbomind/engine/scheduler.cc src/turbomind/engine/README.md
git commit -m "refactor(scheduler): unify resume step selection into ResumeCandidate model"
```

### Task 4: Extract `ClampForwardEnd`

**Files:**
- Modify: `src/turbomind/engine/scheduler.h` (private method declaration)
- Modify: `src/turbomind/engine/scheduler.cc` (new method + call site in `RunRequiredAdmission`)

- [ ] **Step 1: Declare in `scheduler.h`** (next to `PlanPromptBoundaryPublication` / `PlanFullBlockPublication`)

```cpp
    // Land the forward end on a boundary candidate; a result <= begin means
    // nothing runs this pass. Precedence documented at the definition.
    int ClampForwardEnd(const Sequence& s, int begin, int desired, int ctx_end) const;
```

- [ ] **Step 2: Define in `scheduler.cc`** (above `Schedule`)

```cpp
// Land the forward end on a boundary candidate. Precedence:
//   1. Prompt-boundary clamp: the boundary node is armed and this pass reaches
//      B -> land exactly on B (>= so an exact landing is not truncated away).
//   2. Checkpoint-due alignment: when checkpoint bytes are registered and a
//      prompt-region pass would run past the due position
//      (last_ckpt_pos + checkpoint_min_interval), end on the last block
//      boundary in the admitted range — at or past the due position and
//      strictly past begin (progress guarantee) — so the full-block checkpoint
//      can be taken there; the remainder runs in the next pass.
//   3. Partial-chunk alignment: a pass that does not reach the context end
//      lands on a block boundary.
//   4. Otherwise: run to desired.
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

- [ ] **Step 3: Replace the inline clamp in `RunRequiredAdmission`**

Delete from `int desired = begin + admitted;` through the closing brace of the `else { ... }` clamp block (the `prompt_boundary_pos` local, the `publish_prompt` computation, and both truncation branches), plus the subsequent `const int len = desired - begin;`. Replace with:

```cpp
        const int end = ClampForwardEnd(s, begin, begin + admitted, ctx_end);
        const int len = end - begin;
        if (len <= 0) {
            continue;  // nothing admitted this pass; CommitResults leaves it inactive
        }
        s.input_len = len;

        // The publish decision is finalized in SetupPartialSiblings
        // (prompt_boundary_node); the clamp lands a pass exactly on B iff it
        // fired (an end past B implies begin >= B), so end == B identifies the
        // prompt-boundary pass.
        const bool at_prompt_boundary = s.prompt_boundary_node && end == s.prompt_boundary_pos;
```

Then delete the now-duplicated lines below (`if (len <= 0) ...`, `s.input_len = len;`, `const int end = begin + s.input_len;`) — the producer-conflict check and everything after keep using `begin` / `end` unchanged. At the publication routing, replace `if (publish_prompt)` with `if (at_prompt_boundary)` (the two-planner routing itself is merged in Task 5).

- [ ] **Step 4: Build and commit**

From `build/`: `ninja`. Expected: exit 0.

```bash
git add src/turbomind/engine/scheduler.h src/turbomind/engine/scheduler.cc
git commit -m "refactor(scheduler): extract forward-end clamp into ClampForwardEnd"
```

### Task 5: Merge publication planning into `PlanPublication`

**Files:**
- Modify: `src/turbomind/engine/scheduler.h` (replace three declarations with one)
- Modify: `src/turbomind/engine/scheduler.cc` (replace three definitions with one; update call site)

- [ ] **Step 1: Replace declarations in `scheduler.h`**

Delete the declarations of `PlanForkToPopulation`, `PlanPromptBoundaryPublication`, and `PlanFullBlockPublication` (and their comment block). Add:

```cpp
    // Publication planning for a committed forward ending at `end`, routed by
    // whether this is the prompt-boundary pass (end == B). Finds the node
    // ending exactly at `end` (the block itself when block-aligned, else its
    // partial sibling), then decides partial sibling KV population and the
    // checkpoint. Only records intent; slots are allocated in the optional
    // admission phase. Reserves the sibling's prefix_id in pass.planned to
    // dedup intent across requests.
    void PlanPublication(ScheduleState& pass, int i, Sequence& s, int end, bool at_prompt_boundary);
```

- [ ] **Step 2: Replace definitions in `scheduler.cc`**

Delete the three functions `PlanForkToPopulation`, `PlanPromptBoundaryPublication`, `PlanFullBlockPublication` (including their leading comments). Add in their place:

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
    // A partial sibling is a distinct logical block from any request's
    // required prefix blocks, so it never collides with a required id.
    if (at_prompt_boundary && sibling && !sibling->is_valid && !ValidAlloc(sibling->prefix_id)
        && !pass.planned.count(sibling->prefix_id)) {
        pass.planned.insert(sibling->prefix_id);
        pass.pending_populate[i] = sibling;
        pass.has_optionals       = true;
    }

    // (b) Checkpoint onto the node. The prompt-boundary pass bypasses the min
    // interval; the full-block path requires a block-aligned end and is
    // subject to the interval and to cache_generation=none suppression of
    // generation-region checkpoints (a block whose coverage extends past the
    // prompt holds generated tokens and is never indexed under 'none', so its
    // checkpoint would only serve this request's own resume).
    if (!CheckpointPublicationEligible() || s.publish_cache_id == 0 || node == nullptr) {
        return;
    }
    if (!at_prompt_boundary) {
        if (!at_block) {
            return;  // full-block group: no full block ends here
        }
        if (generation_cache_mode_ == CacheMode::kNone && end > s.prompt_len) {
            return;
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

- [ ] **Step 3: Replace the call site in `RunRequiredAdmission`**

```cpp
        if (at_prompt_boundary) {
            PlanPromptBoundaryPublication(pass, i, s, end);  // partial sibling KV + prompt-boundary checkpoint
        }
        else {
            PlanFullBlockPublication(pass, i, s, end);  // full-block checkpoint (coverage only)
        }
```

becomes

```cpp
        // Optional optimizations (allocated later, from inactive memory). One
        // checkpoint per forward, routed by its end.
        PlanPublication(pass, i, s, end, at_prompt_boundary);
```

(Also delete the stale multi-line comment above the old routing that explains `publish_prompt` routing, keeping the replacement comment above.)

- [ ] **Step 4: Behavior-equivalence checklist (read-only, no code)**

Confirm against the deleted code: (1) old prompt-boundary checkpoint target was `at_block ? &x : (at_partial ? x.partial.get() : nullptr)` — identical to `node`; (2) old `PlanForkToPopulation` guards were `y.offset + y.size == end`, `!y.is_valid`, `!ValidAlloc(y_cache)`, `!planned.count(y_cache)` — all present; (3) old full-block guards `x.offset + x.capacity == end`, `none` suppression, min interval, `!ValidAlloc(x.checkpoint_id)` — all present.

- [ ] **Step 5: Build and commit**

From `build/`: `ninja`. Expected: exit 0.

```bash
git add src/turbomind/engine/scheduler.h src/turbomind/engine/scheduler.cc
git commit -m "refactor(scheduler): merge publication planners into PlanPublication"
```

### Task 6: `Demote` + unconditional terminal adoption (behavior change)

**Files:**
- Modify: `src/turbomind/engine/block.h` (new `CacheBlockPool::Demote`)
- Modify: `src/turbomind/engine/scheduler.cc` (`GenStat`, adoption block in `Finalize`, `LogFinalized`)
- Modify: `src/turbomind/engine/README.md` (`contracts.checkpoint-adoption`, `contracts.checkpoint-publish`, `contracts.cache-eviction`)

- [ ] **Step 1: Add `Demote` to `CacheBlockPool` in `block.h`** (after the `Stamp` declarations)

```cpp
    // Demote a slot to evict-first priority: timestamp 0 sorts first in
    // SortedIndices() and is below every eviction cutoff and pass floor.
    // Stamp never hands out 0 (next_timestamp_ starts at 1).
    void Demote(int cache_id)
    {
        blocks_[cache_id].timestamp = 0;
    }
```

- [ ] **Step 2: Update `GenStat` in `scheduler.cc`**

```cpp
    bool terminal_ckpt = false;
    int  dropped       = 0;  // redundant full-block checkpoints dropped on terminal adoption
```

becomes

```cpp
    bool terminal_ckpt = false;
    bool demoted       = false;  // adopted checkpoint undercuts the interval -> evict-first
```

- [ ] **Step 3: Replace the adoption block in `Finalize`**

Replace the whole `if (publish_generation_boundary && x.offset + size == s.filled_len && ...)` block (from the `const int interval = ...` line through the closing brace of the drop loop; keep the long comment above the `if` about frontier correspondence and `frontier_pos` — it still applies) with:

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
            // evict-first priority so the redundancy never displaces other
            // cache state.
            const int interval = registry_.checkpoint_min_interval();
            for (int j = static_cast<int>(i); j-- > 0;) {
                const LogicalBlock& p   = *s.block_ids[j];
                const int           pos = p.offset + p.size;
                if (s.filled_len - pos >= interval) {
                    break;  // outside the window
                }
                if (ValidAlloc(p.checkpoint_id)) {
                    cache_.Demote(f);
                    gen.demoted = true;  // observability (LogFinalized)
                    break;
                }
            }
        }
```

- [ ] **Step 4: Update `LogFinalized`**

```cpp
        std::string ckpt =
            g.terminal_ckpt ? (g.dropped ? fmt::format(", terminal ckpt (dropped {})", g.dropped) : ", terminal ckpt") :
                              "";
```

becomes

```cpp
        std::string ckpt = g.terminal_ckpt ? (g.demoted ? ", terminal ckpt (demoted)" : ", terminal ckpt") : "";
```

- [ ] **Step 5: Update README contracts** (content only, no re-wrapping)

In `contracts.checkpoint-adoption`, replace the two sentences:

> On adoption, redundant full-block checkpoints within `checkpoint_min_interval` below `filled_len` that sit on blocks being indexed this pass (`pos > prompt_len`, still private — no consumer reference) are dropped, mirroring eviction. If the only in-window checkpoint sits on an already-shared block (`pos <= prompt_len`), adoption is skipped instead, preserving the interval without touching shared state.

with:

> Adoption is unconditional (frontier valid, slot empty, terminal block indexed). When another valid checkpoint lies within `checkpoint_min_interval` below `filled_len`, the adopted checkpoint is demoted to evict-first priority (timestamp 0) instead of being suppressed: the interval is relaxed at finalization only, and the redundant checkpoint never displaces other cache state because every eviction pass reclaims it first.

In `contracts.checkpoint-publish`, after the sentence "The prompt-boundary checkpoint bypasses the minimum interval.", insert:

> Terminal adoption (`contracts.checkpoint-adoption`) may also undercut the interval; the adopted checkpoint is demoted to evict-first priority instead of suppressed.

In `contracts.cache-eviction`, after the sentence ending "they age and are reclaimed before live working-set blocks under pressure.", insert:

> A demoted slot (timestamp 0, set by terminal adoption) is always the first eviction candidate in both admission phases.

- [ ] **Step 6: Build and commit**

From `build/`: `ninja`. Expected: exit 0.

```bash
git add src/turbomind/engine/block.h src/turbomind/engine/scheduler.cc src/turbomind/engine/README.md
git commit -m "feat(scheduler): adopt terminal checkpoint unconditionally, demote on interval violation"
```

### Task 7: GPU verification

**Files:** plan checkboxes and verification notes only.

- [x] **Step 1: Pick GPUs**

Run `nvidia-smi` (outside the sandbox); pick idle GPUs (no processes, ~0 MiB). Qwen3.5-27B bf16 needs ~54 GB of weights plus cache: use `--tp 2 --gpus <a>,<b>` on two idle GPUs. Qwen3-8B fits one GPU (`--tp 1`).

- [x] **Step 2: Create the shared-prefix prompt file**

```python
import json
para = ("The quick brown fox jumps over the lazy dog near the quiet river bank "
        "while the morning sun rises over the distant hills. ") * 1
base  = para * 400   # ~10k tokens
extra = para * 200   # ~5k tokens
json.dump([base + "\nSummarize the text above.",
           base + extra + "\nList three facts from the text above."],
          open("/tmp/sched_prompts.json", "w"))
```

- [x] **Step 3: Recurrent-model scenario (outside the sandbox)**

```bash
python scripts/test_turbomind_model.py \
    --model-id Qwen/Qwen3.5-27B --cache-dir /mnt_cfs/huggingface_hub/hub/ \
    --tp 2 --gpus <a>,<b> \
    --enable-prefix-caching --cache-prompt all --cache-generation all \
    --cache-prompt-boundary-skip 2 --max-prefill-token-num 8192 \
    --session-len 32768 --max-new-tokens 256 \
    --prompt-file /tmp/sched_prompts.json --prompt-ids 0 0 1 1 2>&1 | tee /tmp/sched_gdn.log
```

(`--prompt-ids 0 0 1 1`: the repeated prompt 0 exercises prompt-boundary resume via the published partial sibling / boundary checkpoint; prompt 1 extends the shared prefix, exercising checkpoint resume and checkpoint-due truncation; the repeated prompt 1 resumes from the newly published checkpoints. `--cache-generation all` makes terminal adoption fire; with the prompt-boundary checkpoint a short 256-token generation ends well inside the 4096-token default interval, so the demotion path fires too.)

- [x] **Step 4: Verify the recurrent log**

In `/tmp/sched_gdn.log` `[TM][WARN]` lines, check all of:

1. Prompt 0 first run: `published ... boundary [...)` and/or `ckpt@` at B (prompt-boundary pass landed on B).
2. Prompt 0 repeat: `resume ... source=fork` (or `frontier`/`checkpoint` if the boundary block is whole-block-covered) at or near B, with a small `computed` span (tens of tokens, not thousands).
3. Prompt 1 first run: resumes from the shared prefix (`source=checkpoint` or `fork`), and the > 4096-token recompute region is split with a block-aligned `ckpt@` (checkpoint-due truncation still works).
4. Prompt 1 repeat: resumes at the higher checkpoint with a small computed span.
5. At least one `finalized gen [...] terminal ckpt (demoted)` line (demotion fired), or `terminal ckpt` when no in-window checkpoint existed — both prove adoption; `(demoted)` proves the new path.
6. All four responses are meaningful English relevant to the prompts (256 generated tokens requested). Gibberish = a broken restore/publication path — stop and debug (per AGENTS.md, iterate until fixed; do not proceed with active bugs).

- [x] **Step 5: Attention-model scenario (outside the sandbox)**

```bash
python scripts/test_turbomind_model.py \
    --model-id Qwen/Qwen3-8B --cache-dir /mnt_cfs/huggingface_hub/hub/ \
    --tp 1 --gpus <c> \
    --enable-prefix-caching --cache-prompt all --cache-generation all \
    --max-prefill-token-num 8192 --session-len 32768 --max-new-tokens 256 \
    --prompt-file /tmp/sched_prompts.json --prompt-ids 0 0 2>&1 | tee /tmp/sched_attn.log
```

Check: the repeated prompt 0 logs `resume [0,...) ... source=prefix` or `source=fork` (no checkpoint category exists; the `!ckpt` candidate paths), computed span far smaller than the prompt, and both responses meaningful.

- [x] **Step 6: Commit plan checkboxes / verification notes**

Append a `## Verification results` section to this plan (GPUs used, log paths, observed resume/publish/finalize lines), then:

```bash
git add docs/superpowers/plans/2026-07-02-scheduler-unification.md
git commit -m "docs: record scheduler-unification verification results"
```

## Verification results

Run date: 2026-07-02 UTC.

Pre-GPU verification after Task 6: `ninja` from `build/` exited 0; `./bin/test_prefix_trie` exited 0 with 73 assertions in 10 cases; `git diff --check HEAD~1 HEAD` was clean.

GPU selection: GPU 4 was occupied by pre-existing PID 518828 and was not used. Final `get_gpu_usage` snapshot after verification showed GPUs 1,2,3,5,6,7 at 4 MiB, GPU 0 at 122 MiB, and GPU 4 still occupied at 140605 MiB.

Shared prompt file: `/tmp/sched_prompts.json`, two shared-prefix prompts, 122074 bytes.

Recurrent model, authoritative run: `/tmp/sched_gdn_tp1.log`. The original plan listed `--tp 2`, but the H200 has 141 GiB VRAM and the final run used the user's requested single-GPU shape:

```bash
TM_CACHE_LOG_INTERVAL=1 python scripts/test_turbomind_model.py --model-id Qwen/Qwen3.5-27B --cache-dir /mnt_cfs/huggingface_hub/hub --tp 1 --gpus 0 --enable-prefix-caching --cache-prompt all --cache-generation all --cache-prompt-boundary-skip 2 --max-prefill-token-num 8192 --session-len 32768 --max-new-tokens 256 --prompt-file /tmp/sched_prompts.json --prompt-ids 0 0 1 1 2>&1 | tee /tmp/sched_gdn_tp1.log
```

Recurrent setup/tokens: lines 2168-2182 show `model: Qwen/Qwen3.5-27B`, `tp: 1`, `gpus: 0`, `max_new_tokens: 256`, `session_len: 32768`, `cache_checkpoint_interval: 4096`, `cache_prompt: 'all'`, `cache_generation: 'all'`, `cache_prompt_boundary_skip: 2`, `prompt_count: 4`. Lines 2185-2205 show pipeline load 19.15 s, inference 6.59 s, and generated 256 tokens for each of four prompts.

Recurrent cache evidence:

1. Prompt 0 first run published the shared prefix and prompt boundary with checkpoint: lines 313-322 show `published prefix [0,8192) ... ckpt@8192` and `published prefix [8192,9600) ... boundary [9600,9616) ... ckpt@9616`.
2. Prompt 0 repeat resumed at the boundary with tiny recompute: line 333 shows `resume [0,9616) ... source=fork | computed [9616,9618) 2 tok`.
3. Prompt 1 first run resumed from the shared checkpoint and then published the extended boundary checkpoint: lines 341-343 show `resume [0,8192) ... source=checkpoint | computed [8192,14417) 6225 tok` and `published prefix [9600,14400) ... boundary [14400,14417) ... ckpt@14417`.
4. Prompt 1 repeat resumed at the higher checkpoint with tiny recompute: line 351 shows `resume [0,14417) ... source=fork | computed [14417,14419) 2 tok`.
5. Terminal adoption and demotion fired: lines 2137 and 2153 show `finalized gen ... terminal ckpt (demoted)` for the generated ranges.

Recurrent response quality: lines 2207-2282 show all four 256-token responses were meaningful English relevant to the prompts. The summarize prompts describe the repeated fox/dog/river/sun sentence, and the fact prompts identify facts about the fox, dog, river, and sun.

Checkpoint-due supplement: `/tmp/sched_gdn_checkpoint.log`. This run used `cache_prompt: 'auto'` because the main `cache_prompt: 'all'` run correctly clamps to the prompt boundary first, which masks a clean checkpoint-due split. Command:

```bash
TM_CACHE_LOG_INTERVAL=1 python scripts/test_turbomind_model.py --model-id Qwen/Qwen3.5-27B --cache-dir /mnt_cfs/huggingface_hub/hub --tp 1 --gpus 0 --enable-prefix-caching --cache-prompt auto --cache-generation all --cache-prompt-boundary-skip 2 --max-prefill-token-num 8192 --session-len 32768 --max-new-tokens 256 --prompt-file /tmp/sched_prompts.json --prompt-ids 0 1 2>&1 | tee /tmp/sched_gdn_checkpoint.log
```

Checkpoint-due setup/tokens: lines 2154-2168 show `model: Qwen/Qwen3.5-27B`, `tp: 1`, `gpus: 0`, `cache_prompt: 'auto'`, `cache_generation: 'all'`, `cache_prompt_boundary_skip: 2`, `prompt_count: 2`; lines 2171-2183 show pipeline load 19.02 s, inference 6.36 s, and generated 256 tokens for both prompts.

Checkpoint-due cache evidence: lines 313-322 publish `[0,8192)` with `ckpt@8192`; lines 331-332 show prompt 1 resumed from checkpoint at 8192 and split the >4096-token recompute span on a block boundary: `computed [8192,14400) 6208 tok` and `published prefix [9600,14400) ... ckpt@14400`. Lines 2125 and 2140 show terminal checkpoints were adopted and demoted. Lines 2185-2222 show both responses were meaningful and prompt-relevant.

Attention model: `/tmp/sched_attn.log`. Command:

```bash
TM_CACHE_LOG_INTERVAL=1 python scripts/test_turbomind_model.py --model-id Qwen/Qwen3-8B --cache-dir /mnt_cfs/huggingface_hub/hub --tp 1 --gpus 2 --enable-prefix-caching --cache-prompt all --cache-generation all --max-prefill-token-num 8192 --session-len 32768 --max-new-tokens 256 --prompt-file /tmp/sched_prompts.json --prompt-ids 0 0 2>&1 | tee /tmp/sched_attn.log
```

Attention setup/tokens: lines 1539-1553 show `model: Qwen/Qwen3-8B`, `tp: 1`, `gpus: 2`, `cache_prompt: 'all'`, `cache_generation: 'all'`, `cache_prompt_boundary_skip: 1`, `prompt_count: 2`; lines 1556-1568 show pipeline load 13.28 s, inference 1.99 s, and generated 256 tokens for both prompts.

Attention cache evidence: lines 228-235 show the first prompt published `[0,8192)` and boundary `[9600,9615)`; line 242 shows the repeat resumed from the forked boundary with `computed [9615,9616) 1 tok`. Line 1528 shows finalization of the generated range. Lines 1570-1593 show both responses were meaningful English summaries of the repeated fox/dog/river/sun text.
