# Partial sibling edges: interior partial nodes as resume candidates

Date: 2026-07-02
Status: approved design, pending implementation plan

## Problem

With `--enable-prefix-caching --cache-prompt-boundary-skip 2 --cache-prompt auto
--cache-generation none`, a 3-round conversation (round 1 carries 3 images,
boundary checkpoint published at 10586 on a partial fork node):

- Round 2 misses at block 165, binds `fork_from` to the partial node
  `[10560,10586)`, and resumes at 10586 (`source=fork`).
- Round 3 matches *more* blocks (`[0,10880)`, 170 blk) — including the full
  block `[10560,10624)` — and therefore never binds `fork_from`. Resume falls
  back to the block-aligned checkpoint at 9216 (`source=checkpoint`) and
  recomputes 1915 tokens instead of ~545.

Root cause: the partial node's checkpoint at 10586 is *discoverable* (the
shared full block at index 165 carries a fork edge to it), but `Resume`
ignores fork edges except at the single block sitting exactly at `prefix_end`.
A partial node strictly inside the matched prefix is invisible as a resume
candidate. The same blind spot applies to attention-only models when a matched
full block's KV allocation was evicted: the prefix walk stops there, and the
fork extension only works if that exact block happens to carry `fork_from`.

## Reframing: one `partial` sibling edge

The original edges encode per-request intent: `fork_from` = "my private miss
block can seed itself from this node" (read), `fork_to` = "I will populate
this node at my boundary" (write). Once the edges live on shared trie nodes
and any later request may consume them, they are a structural property of the
index, not request intent. We unify them:

```cpp
// In struct LogicalBlock, replacing `BlockHandle fork_from; BlockHandle fork_to;`:

// First-known indexed partial sibling at this block index: an identity-
// verified node with the same parent and a strict token-prefix of this
// block's content. Every edge points to a sibling with strictly smaller
// `size`, so the edge graph is acyclic. First-wins: bound at most once, at
// Accept, on a block created in the same pass (mirrors trie first-wins
// insertion).
BlockHandle partial;
```

The two old roles become uses of the one edge:

- Read side (`Resume`): the sibling is a resume candidate — checkpoint-only
  restore when its end is inside the valid prefix (new), KV copy + checkpoint
  restore when it extends past `prefix_end` (existing fork extension).
- Write side (publication): whether *this* request populates the sibling at
  its boundary is derived where needed, from `prompt_boundary_node` plus the
  edge's geometry (`partial->offset + partial->size == end`), instead of being
  encoded by which field the edge lives in.

### First-wins binding rule

`x.partial` is bound at most once and never overwritten. Both bind sites in
`SetupForks` target a block created in the same pass, so the slot is provably
empty and the rule reduces to a `TM_CHECK(!x.partial)` assertion:

- Matcher path (miss-block `trie_.Search`): `x` is the miss block, created
  this pass by `CreateMissingBlocks`.
- Creator path (boundary-node creation): `x` is block `j` with `miss < j`
  (guaranteed by `PlanPromptBoundary`), so it is also created this pass and
  distinct from the miss block. The edge is the only ref keeping the new node
  alive; a genuine occupied-slot conflict is unreachable.

Rationale for first-wins over keep-longest: the payoff of keep-longest is
bounded by one block of recompute (< block_size tokens); keep-longest requires
rebind logic whose edge-drop can recycle a node (and its checkpoint) other
sequences still want. First-wins never releases anything and matches the
trie's conflict rule.

### Edge-carrying partial nodes (generation boundary)

`PublishGeneration` (`cache_generation=all`) indexes the request's private
blocks in place, including the terminal partial block. A former miss block
indexed this way keeps its matcher-bound `partial` edge, so an indexed
partial node *can* carry an outgoing edge — this exists today with
`fork_from` and remains legal. Acyclicity does not rely on partials being
edge sinks: every edge points to a sibling with strictly smaller `size`
(the matcher binds a strict prefix of the block's Accept-time content, and
generation indexing only grows the carrier's `size`), so size strictly
decreases along any edge path and no reference cycle can form. Consumers
never traverse more than one edge hop, so edge-carrying partials need no
special handling.

### Why discovery needs no new trie searches

Every indexed full block `x` with an existing partial sibling `y` acquires the
edge through an existing path:

1. Same request creates both (boundary path): `x.partial = y` at creation;
   `x` is then inserted and shared.
2. Later request whose miss block is `j`: the miss-block `trie_.Search` finds
   `y` and binds `x.partial = y` on the block it creates and inserts.
3. Later request with miss `< j` cannot exist: `y.parent` is block `j-1`'s
   node; if that node was just created by this request, the older `y` cannot
   have it as parent.

So Accept keeps its current structure; only the field unification and the
first-wins guards change.

## Resume changes (`Scheduler::Resume`, scheduler.cc)

### Step 2 — interior checkpoint candidates (checkpoint models)

While walking blocks backward for the latest checkpoint, also consider the
partial sibling's checkpoint. `ye <= prefix_end` means the sibling's KV range
is already covered by the valid full-block prefix, so this is a
checkpoint-only restore — no KV copy, and `y.is_valid` (KV validity) is
deliberately not required, matching how the loop trusts
`ValidAlloc(x.checkpoint_id)` alone:

```cpp
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
        // Interior partial sibling: mid-block checkpoint inside the valid
        // prefix (KV already covered; checkpoint restore only).
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

The block's own (block-end) checkpoint is checked first since it dominates any
partial in the same block; either hit terminates the backward walk because
candidates in earlier blocks are strictly smaller. For the motivating log,
round 3 restores ckpt@10586 and recomputes 545 tokens instead of 1915.

### Step 3 — fork extension keeps its semantics, reads the unified edge

The beyond-`prefix_end` case (matched full block with evicted/invalid KV, or
private miss block) keeps KV-copy + checkpoint semantics, reading `x.partial`:

```cpp
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

Behavioral note: `fork_dst` can now be a shared indexed node whose KV was
evicted (`is_valid == false`); the restore copy re-populates it and `Publish`
flips `is_valid` back after the forward proves content (existing flow).

## Publication changes

- `PlanForkToPopulation`: read `x.partial` instead of `x.fork_to`; the
  existing geometry/validity guards (`y.offset + y.size == end`, `!y.is_valid`,
  `!ValidAlloc(y.prefix_id)`, pass-level dedup) already reject a sibling that
  belongs to someone else's shorter boundary.
- `PlanPromptBoundaryPublication`: `at_fork_to` becomes
  `x.partial && x.partial->offset + x.partial->size == end` — same length
  check, unified field.
- `CommitResults` (`pending_fork` population copy) is field-agnostic and
  unchanged.

## Invariants

Asserted at both bind sites in `SetupForks` and stated in
`src/turbomind/engine/README.md`:

- `partial` edges point to an identity-verified sibling at the same block
  index with strictly smaller `size`; size strictly decreases along edge
  paths, so the graph is a DAG and no circular refcount is possible. (An
  indexed partial node may itself carry an edge — see the generation-boundary
  section — but only to a strictly shorter sibling.)
- `partial` is first-wins and both bind sites target same-pass-created
  blocks: `TM_CHECK(!x.partial)` on the binding block. No assert on the
  bound-to node's own edge (it may legitimately carry one).

## Logging and docs

- `LogAccept`: the matched-side tail prints `partial@<end>` (was
  `fork_from@`); the created-side tail keeps reporting the boundary node end
  (was `fork_to@`), derived from `x.partial` at the boundary block and
  guarded by `end == prompt_boundary_pos` so a matcher-bound sibling that
  does not end at `B` is never misreported as the publish node.
- `LogResume` unchanged (`source=checkpoint` with a mid-block resume position
  is self-explanatory).
- `README.md` updates, referencing by section: `contracts.prefix-prepare`
  (edge unification, first-wins, discovery-by-construction),
  `contracts.cache-prepare` (Resume may select an interior partial's
  checkpoint, restore-only), `contracts.boundary-policy` /
  `contracts.checkpoint-publish` (rename `fork_to`-node references to the
  partial sibling edge).

## Out of scope

- Multiple partial siblings per block index (first-wins keeps one; others
  remain reachable only via their creator's live sequence, as today).
- Binding edges after Accept (a partial published after a request's Accept is
  not visible to that request's later Resumes, matching current `fork_from`
  behavior).
- Eviction policy changes: the sibling's prefix/checkpoint allocations remain
  independently evictable; only the small logical node is pinned by the edge.

## Testing

- Build in `build/` with `ninja`.
- Reproduce the motivating scenario with `scripts/test_turbomind_model.py`
  (multi-round conversation, round 1 with images, server flags
  `--enable-prefix-caching --cache-prompt-boundary-skip 2 --cache-prompt auto
  --cache-generation none`), verifying:
  - round 3 logs `resume [0,10586)`-style mid-block resume with
    `source=checkpoint` (not a fallback to the earlier block-aligned ckpt);
  - responses remain meaningful (>= 128 tokens, relevant to the prompt) in
    all rounds.
- Regression: single-round and round-2 behavior unchanged (`source=fork`
  path still selected when the sibling extends past `prefix_end`).
