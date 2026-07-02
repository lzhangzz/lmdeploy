# Partial Sibling Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify `LogicalBlock::fork_from`/`fork_to` into one `partial` sibling edge and make interior partial-node checkpoints resume candidates, so a request whose prompt fully matches past a published boundary (e.g. ckpt@10586 inside matched `[0,10880)`) resumes at the boundary instead of falling back to an earlier block-aligned checkpoint.

**Architecture:** The two per-request fork edges become one structural, first-wins edge on the shared trie node (full/containing block → strictly shorter identity-verified partial sibling; partials carry no outgoing edge, so the graph is acyclic). `Scheduler::Resume` step 2 gains an interior candidate: the sibling's checkpoint restores without any KV copy when its end lies inside the valid prefix. Publication code derives "should I populate this node" from geometry instead of the edge's field name. Spec: `docs/superpowers/specs/2026-07-02-partial-sibling-resume-design.md`.

**Tech Stack:** C++ (TurboMind engine), ninja build in `build/`, GPU smoke test via `scripts/test_turbomind_model.py` (AS IS), model index at `/data/models.json` (no MCP in this environment).

**Contract note:** All changes must preserve `src/turbomind/engine/README.md` contracts; the doc itself is updated in Task 4. Relevant items: `contracts.prefix-prepare`, `contracts.cache-prepare`, `contracts.boundary-policy`, `contracts.checkpoint-publish`.

**Testing note:** There is no C++ unit-test harness for the scheduler; each task is verified by a full `ninja` build, and end-to-end behavior is verified once on GPU in Task 5. Do not modify `scripts/test_turbomind_model.py`.

---

### Task 1: Unify the edge — mechanical rename to `partial`

**Files:**
- Modify: `src/turbomind/engine/block.h:146-148`
- Modify: `src/turbomind/engine/scheduler.cc` (all `fork_from` / `fork_to` uses; exact sites below)
- Modify: `src/turbomind/engine/scheduler.h:130-131,151,203-206`
- Modify: `src/turbomind/engine/request.h:245-249` (comment only)
- Modify: `src/turbomind/engine/cache_mode.h:38-42` (comment only)

- [ ] **Step 1: Replace the two fields in `block.h`**

Replace (currently lines 146-148):

```cpp
    // Fork edges (strong, RAII)
    BlockHandle fork_from;  // partial-match source (read side)
    BlockHandle fork_to;    // prompt-boundary publish target (write side)
```

with:

```cpp
    // First-known indexed partial sibling at this block index: an identity-
    // verified node with the same parent and a strict token-prefix of this
    // block's content. Every edge points to a sibling with strictly smaller
    // `size` (a carrier indexed later by PublishGeneration only grows), so
    // size strictly decreases along edge paths and the graph is acyclic.
    // First-wins: bound at most once, at Accept, on a block created in the
    // same pass (mirrors trie first-wins insertion). Strong, RAII.
    BlockHandle partial;
```

- [ ] **Step 2: Rename every use in `scheduler.cc`**

All current `fork_from`/`fork_to` member accesses collapse onto `.partial`. This is safe within one request because the matcher bind site (miss block) and the creator bind site (boundary block `j`, with `miss < j`) always target different, freshly created blocks. Exact sites (line numbers pre-edit):

1. Line 344 comment: `SetupForks(s, st);           // partial sibling bind (matcher side) + boundary node creation (creator side)`
2. Lines 433-435 comment block: replace with
```cpp
    // Matcher-side sibling bind: any prior request may have published a
    // prompt partial node (cache_prompt in {all, auto}) or a generation
    // terminal partial ('all'), so the miss block must always try to match.
```
3. Line 447: `x.partial = BlockHandle{v};  // edge ref (fresh block; first-wins trivially holds)`
4. Line 451 comment: `// Prompt-boundary publish point (creator-side partial sibling). B = prompt_len - K ...` (keep the rest of the sentence about `all`/`auto` unchanged).
5. Line 483: `x.partial = std::move(vh);  // edge holds the only ref`
6. Lines 586-596 (Resume step 3) — rewritten fully in Task 3; for this task just rename `x.fork_from` → `x.partial` (3 places) so it compiles.
7. Lines 926-948 (`PlanForkToPopulation`): rename `x.fork_to` → `x.partial` (3 places) and update the function comment to:
```cpp
// When this pass reaches the prompt boundary, plan the device copy that
// populates the indexed prompt-end partial sibling. Returns the node when a
// copy is planned, nullptr otherwise. The geometry guard (y.offset + y.size
// == end) rejects a sibling belonging to a different boundary.
```
8. Lines 961-967 (`PlanPromptBoundaryPublication`): comments say "partial sibling" instead of "fork_to node"; code becomes
```cpp
        const bool    at_partial = x.partial && x.partial->offset + x.partial->size == end;
        LogicalBlock* target     = at_block ? &x : (at_partial ? x.partial.get() : nullptr);
```
9. Line 1450-1452 (`LogAccept` mtail): field rename plus log token:
```cpp
        if (matched < (int)s.block_ids.size() && s.block_ids[matched]->partial) {
            const LogicalBlock& y = *s.block_ids[matched]->partial;
            mtail                 = fmt::format(", partial@{}", y.offset + y.size);  // matched-side partial reuse
        }
```
10. Lines 1459-1461 (`LogAccept` ctail): field rename, log token, and a new end guard. With the unified field, block `j` could in principle expose a matcher-bound sibling that does not end at `B` (block-aligned boundary case); the guard keeps the created-side tail truthful:
```cpp
            if (j >= 0 && j < (int)s.block_ids.size() && s.block_ids[j]->partial) {
                const LogicalBlock& ft = *s.block_ids[j]->partial;
                if (ft.offset + ft.size == s.prompt_boundary_pos) {
                    ctail = fmt::format(", partial_to@{}", ft.offset + ft.size);  // created-side publish node end
                }
            }
```
11. Line 1577 (`LogCollision`): `note  = " (no partial node)";`
12. Comment-only touches: lines 242, 328, 433, 620, 951, 955, 1033, 1108, 1198, 1201, 1243, 1270, 1322 — replace `fork_to`/`fork_from`/`fork-to` wording with "partial sibling" / "partial node". Do NOT rename the identifiers `pending_fork`, `PlanForkToPopulation`, `PublishStat::forked`, `ResumeSource::kFork`, or the `source=fork` log string — the fork *event* (a request seeding from / populating a sibling) keeps its name; only the stored edge is renamed.

- [ ] **Step 3: Update comments in `scheduler.h`, `request.h`, `cache_mode.h`**

`scheduler.h:130-131`: `// fork_from (partial match) and fork_to (prompt-boundary publish point).` → `// the partial sibling edge (matcher bind + prompt-boundary node creation).`
`scheduler.h:151`: `// a fork_to boundary populated this pass` → `// a partial sibling populated this pass`
`scheduler.h:203-205`: replace `fork-to node's prefix_id` with `partial sibling's prefix_id`.
`request.h:246`: `a partial fork_to` → `a partial sibling` (keep line breaks; do not re-wrap).
`cache_mode.h:39`: `plan wants a partial fork_to node` → `plan wants a partial sibling node`.

- [ ] **Step 4: Grep for leftovers and build**

Run: `rg -n 'fork_from|fork_to' src/` — expect matches only in `src/turbomind/engine/README.md` (updated in Task 4).
Run: `ninja` from `build/` (configure first with `sh ../my_generate.sh` if `build/` is not configured).
Expected: build succeeds.

- [ ] **Step 5: Commit**

```bash
git add src/turbomind/engine/block.h src/turbomind/engine/scheduler.cc src/turbomind/engine/scheduler.h src/turbomind/engine/request.h src/turbomind/engine/cache_mode.h
git commit -m "refactor: unify fork_from/fork_to into one partial sibling edge"
```

---

### Task 2: Assert the edge invariants at both bind sites

**Files:**
- Modify: `src/turbomind/engine/scheduler.cc` (`SetupForks`, the two bind sites from Task 1)

Important: do NOT assert on the bound-to node's own `partial` edge. An indexed
partial node may legitimately carry one — `PublishGeneration`
(`cache_generation=all`) indexes a former miss block in place as the terminal
generation partial, keeping its matcher-bound edge. Acyclicity holds because
`size` strictly decreases along every edge, not because partials are sinks.
Only first-wins on the *binding* block is assertable (both bind sites target a
block created in the same pass, so the slot is provably empty).

- [ ] **Step 1: Matcher-side assert**

Around the (post-Task-1) miss-block bind:

```cpp
        if (LogicalBlock* v = trie_.Search(st.miss_parent, k, TokenSegment(s, offset, size), fps, fp_pos)) {
            TM_CHECK(!x.partial);  // first-wins: x created this pass, slot empty
            TM_CHECK_LT(v->size, size);  // strictly shorter sibling (acyclicity)
            x.partial = BlockHandle{v};  // edge ref
        }
```

(`size` here is the local `std::min(prompt - offset, bs)` — the binding block's
Accept-time content extent; `trie_.Search` never returns a full-length match,
so the check documents rather than changes behavior.)

- [ ] **Step 2: Creator-side assert**

Around the (post-Task-1) boundary-node bind:

```cpp
                if (trie_.Insert(y)) {
                    TM_CHECK(!x.partial);  // first-wins: x created this pass (miss < j), slot empty
                    x.partial = std::move(vh);  // edge holds the only ref
                }
```

(`y.size == plan.node_size < bs <= x.capacity` by `PlanPromptBoundary`
geometry; no additional size assert needed.)

- [ ] **Step 3: Build**

Run: `ninja` from `build/`. Expected: success.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/engine/scheduler.cc
git commit -m "feat: assert partial-sibling edge invariants (first-wins, acyclic)"
```

---

### Task 3: Resume selects interior partial checkpoints

**Files:**
- Modify: `src/turbomind/engine/scheduler.cc:553-598` (`Scheduler::Resume`, steps 2 and 3)

- [ ] **Step 1: Extend the step-2 checkpoint walk**

Replace the existing `if (step < prefix_end) { ... }` loop (post-Task-1 line numbers near 565-579) with:

```cpp
        // Latest block checkpoint within the reusable prefix. A block's own
        // (block-end) checkpoint dominates any partial sibling in the same
        // block; either hit ends the backward walk (earlier candidates are
        // strictly smaller).
        if (step < prefix_end) {
            for (int i = std::min<int>(s.block_ids.size(), (prefix_end + bs - 1) / bs); i > 0; --i) {
                const LogicalBlock& x = *s.block_ids[i - 1];
                const int           e = x.key ? x.offset + x.size : x.offset + x.capacity;
                if (e <= step) {
                    break;
                }
                if (e <= prefix_end && ValidAlloc(x.checkpoint_id)) {
                    step         = e;
                    source       = ResumeSource::kCheckpoint;
                    restore_ckpt = x.checkpoint_id;
                    break;
                }
                // Interior partial sibling: mid-block checkpoint inside the
                // valid prefix. Its KV range is covered by the valid full
                // blocks, so this is a checkpoint-only restore (no KV copy,
                // y.is_valid not required — same trust as the block case).
                if (const LogicalBlock* y = x.partial.get()) {
                    const int ye = y->offset + y->size;
                    if (ye <= prefix_end && ye > step && ValidAlloc(y->checkpoint_id)) {
                        step         = ye;
                        source       = ResumeSource::kCheckpoint;
                        restore_ckpt = y->checkpoint_id;
                        break;
                    }
                }
            }
        }
```

- [ ] **Step 2: Step-3 fork extension reads the unified edge**

Replace the existing step-3 block (post-Task-1, formerly `x.fork_from`) with:

```cpp
    // 3. Fork extension: an indexed partial sibling can beat the current step
    //    by copying its content into the block at the boundary. The target may
    //    be a shared indexed node whose KV was evicted (is_valid == false);
    //    the restore copy re-populates it and Publish flips is_valid after
    //    the forward proves content.
    if (prefix_end % bs == 0 && prefix_end / bs < static_cast<int>(s.block_ids.size())) {
        LogicalBlock& x = *s.block_ids[prefix_end / bs];
        if (LogicalBlock* y = x.partial.get()) {
            const int e = y->offset + y->size;
            if (y->is_valid && e <= upper && e > step && ValidAlloc(y->prefix_id)
                && (!ckpt || ValidAlloc(y->checkpoint_id))) {
                step         = e;
                source       = ResumeSource::kFork;
                fork_dst     = &x;
                fork_src     = y;
                restore_ckpt = ckpt ? y->checkpoint_id : 0;
            }
        }
    }
```

(`fork_dst`/`fork_src` locals and the `restore_copies` emission at step 4 are unchanged.)

- [ ] **Step 3: Build**

Run: `ninja` from `build/`. Expected: success.

- [ ] **Step 4: Commit**

```bash
git add src/turbomind/engine/scheduler.cc
git commit -m "feat: resume from interior partial-sibling checkpoints inside the valid prefix"
```

---

### Task 4: README and terminology updates

**Files:**
- Modify: `src/turbomind/engine/README.md` (sections: `boundary-policy` ~line 126, `prefix-prepare` ~line 256, `cache-prepare` ~line 260, `checkpoint-publish` ~line 384)

Preserve the file's existing line breaks; edit content, not wrapping. Reference items by `<section>.<leaf>`.

- [ ] **Step 1: Terminology — rename edge mentions**

In `contracts.boundary-policy` and `contracts.checkpoint-publish`: replace `fork_to` with "partial sibling node (`LogicalBlock::partial`)" on first mention in each section and plain "partial sibling" after; replace `fork_from` likewise in `contracts.prefix-prepare` ("the first miss may bind a partial-match source (`fork_from`)" → "the first miss may bind the partial sibling edge (`LogicalBlock::partial`)"; "`fork_from` is always bound" → "the matcher-side bind is always attempted").

- [ ] **Step 2: `contracts.prefix-prepare` — add the structural-edge sentence**

Append to the section (as its own sentence, keeping surrounding wrapping intact):

```
The partial sibling edge is structural, not per-request intent: it lives on the indexed block, points to an identity-verified sibling with strictly smaller size (size strictly decreases along edge paths, keeping the graph acyclic even when a generation-indexed partial carries an edge), and is first-wins — bound at most once, at Accept, on a block created in the same pass (matcher side at the miss block, creator side at the boundary block).
```

- [ ] **Step 3: `contracts.cache-prepare` — add the interior-resume sentence**

Append:

```
Resume may select an interior partial sibling's checkpoint as the resume point: when the sibling's end lies inside the contiguous valid prefix, its KV range is already covered by the valid full blocks, so planning emits a checkpoint restore copy only (no KV copy); a sibling extending past the prefix end keeps the fork-extension semantics (KV copy plus checkpoint restore when the model checkpoints).
```

- [ ] **Step 4: Build (README-only change; build to keep the loop uniform) and commit**

Run: `ninja` from `build/`. Expected: no-op / success.

```bash
git add src/turbomind/engine/README.md
git commit -m "docs: partial sibling edge contract (structural, first-wins, interior resume)"
```

---

### Task 5: GPU end-to-end verification

**Files:** none modified. Uses `scripts/test_turbomind_model.py` AS IS.

The motivating scenario used images with `--cache-prompt auto`; the test script takes text prompts only, so exercise the identical scheduler path with `--cache-prompt all` (publishes the partial boundary node for any mid-block `B`; the code path from `SetupForks` through `Resume` is the same). Three nested prompts emulate the three rounds: P2 extends P1, P3 extends P2, so round 3 fully matches the block containing round 1's boundary.

- [ ] **Step 1: Model and GPU selection**

MCP model-server tools are NOT available in this environment. The model index is at `/data/models.json`; use:
- Primary model: `Qwen/Qwen3.5-27B`, cache dir `/mnt_cfs/huggingface_hub/hub/` (hybrid model with recurrent checkpointing — exercises the interior-checkpoint path of Task 3, which only activates when `registry_.has_checkpoint()`).
- Attention regression model (step 6): `Qwen/Qwen3-8B` from the same cache dir.

Check for a free GPU with `nvidia-smi --query-gpu=index,memory.used,memory.total --query-compute-apps=pid --format=csv` (pick a GPU with no compute processes) instead of the `get_gpu_usage` tool. GPU commands must run outside the sandbox.

- [ ] **Step 2: Write the nested prompts file**

Create `/tmp/nested_prompts.json` — a JSON array `[P1, P2, P3]` where P1 is a long passage (aim for several thousand tokens, e.g. a repeated essay-like text — long enough to span many 64-token blocks and cross `--cache-checkpoint-interval`), `P2 = P1 + " Now summarize the above in detail."`, `P3 = P2 + " Then list three key takeaways."`. Exact string content is free; the nesting (each prompt a strict prefix-extension of the previous) is what matters.

- [ ] **Step 3: Run the smoke test (sequential rounds via prompt order)**

```bash
TM_LOG_LEVEL=WARNING python scripts/test_turbomind_model.py \
    --model-id Qwen/Qwen3.5-27B --cache-dir /mnt_cfs/huggingface_hub/hub/ \
    --tp 1 --gpus <FREE_GPU> \
    --enable-prefix-caching --cache-prompt all --cache-generation none \
    --max-batch-size 1 \
    --prompt-file /tmp/nested_prompts.json \
    --max-new-tokens 256 2>&1 | tee /tmp/partial_sibling_test.log
```

`--max-batch-size 1` forces the three prompts to run as consecutive rounds so each later round matches the earlier round's published blocks. Requested response length is 256 (>= 128 as required).

- [ ] **Step 4: Verify scheduler logs**

Inspect `/tmp/partial_sibling_test.log` for the `[scheduler.cc]` WARN lines:
- Round 1 (`req 0`): `published prefix ... boundary [b0,B) ... ckpt@B` with mid-block `B` — the partial sibling exists.
- Round 2 (`req 1`): `matched ..., partial@B` and `resume [0,B) ... source=fork` (or `source=checkpoint`) — regression: the old miss-block path still works.
- Round 3 (`req 2`): matched extent strictly greater than `B`, and **`resume [0,B') ... source=checkpoint` where `B'` is round 2's boundary (mid-block), not a fallback to an earlier block-aligned checkpoint**. This is the fixed behavior; pre-fix it resumed at the last `cache_checkpoint_interval`-aligned position.

If round 3 still falls back, debug and iterate (per the debugging loop in AGENTS.md) — do not stop with the bug active.

- [ ] **Step 5: Verify response quality**

Read all three `--- response N begin/end ---` sections: each must be meaningful human text relevant to the prompt (a summary / takeaways for rounds 2-3), not gibberish, with generated token counts reported. Gibberish indicates a KV/checkpoint restore bug — debug and iterate.

- [ ] **Step 6: Attention-model regression**

Repeat steps 3-5 with `Qwen/Qwen3-8B` (plain attention, no checkpoint category), same cache dir. Expect round 2/3 resumes with `source=prefix` or `source=fork` and meaningful responses; the interior-checkpoint branch must simply never fire (no behavior change).

- [ ] **Step 7: Commit nothing; report**

No source changes in this task. Summarize the observed log lines (match/resume/publish for each round) as the verification record.
