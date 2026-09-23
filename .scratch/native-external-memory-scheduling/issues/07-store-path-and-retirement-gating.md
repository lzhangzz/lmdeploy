# Store path, publication, and retirement gating

Type: grilling
Status: resolved
Blocked by: 04

## Question

When are stores planned and committed, and how does retirement gating become transaction-expressed?

Today stores are staged per-phase after batches (`StageStores(phase, ...)` / `OnBatchComplete`) and retirement is gated by `CanRetire`/`OnRetire` from the engine loop — cache work must finish before a retiring sequence is released. Decide:

- The store-side analog of `contracts.checkpoint-publish`: is store intent planned at commit as coverage-driven publication (which coverage, which categories, at which granularity), at finalization, or both? Does the existing publication machinery (frontier-to-slot copies after `kUnprep`) generalize to outbound external moves, or does outbound need its own intent kind?
- Whether stored coverage is recorded on publish or on store completion, and what a failed store means for coverage bookkeeping.
- The transaction-expressed replacement for retirement gating: what in the transaction's own state says "this sequence cannot be released yet" (replacing `CanRetire`), how it composes with `retiring && inflight == 0` (`invariants.cleanup`), and what happens to in-flight stores at shutdown (replacing `Drain`).
- Warm-up interplay: `is_warm_up` suppresses recurrent checkpoint publication today; what it suppresses on the store path natively.
- Input from the async-transaction decision: store intents are submitted at commit and folded at `Reconcile()` (`pending_stores` on the sequence, ticket 04); decide where the ready-event fencing lives in that shape — today stores are staged at `Setup` and submitted one loop turn later so `Update`/`Sync` fence the producing batch first (E6/E8, F29/G4's client-side gating), and the service needs the recorded ready event (F22).
- Input from the resume decision: S5c's guarantee — the armed chunk-boundary snapshot becomes a *required* allocation when the admitted forward lands exactly on the chunk boundary (today expressed only in a cross-file comment, X8) — must become a stated contract leaf in this design, since the store path depends on snapshots existing when staged.

## Answer

Resolved 2026-09-22 through grilling (Q2/Q3/Q5 confirmed directly; Q1/Q4 confirmed as relocation-with-reason after "how is this different?" challenges). Two deliberate non-redesigns, three real closures.

**The decision.**

1. **Store intents are planned at the publication stage (relocation with reason).** When a committed forward publishes valid coverage above `store_cursor`, the same publication stage that plans checkpoint publication plans the outbound `StoreIntent`: chunk-aligned ranges, sources = the just-published slots, G5 splitting around missing checkpoint slots. Committed with the pass; the intent carries the producing batch's completion event as its ready token, and the seam submits when it fires. The staging arithmetic is unchanged from today's `StageStores` — what changes is ownership (transaction, not an engine hook), the fencing mechanism (event token, not a one-loop-turn deferral), and the deaths: E6 (`StageStores` at `Setup`), E8's `SubmitStores` half, E7's truncation hook, and — added by the 2026-09-22 admission re-grilling — S5b's planning-stage arming (`PlanStoreCheckpoint` and `Sequence::store_checkpoint`; see the resolution comment). Without this move the patch survives as a loop hook; this is acceptance item 3's movement.
2. **The cursor advances at commit; failures leave gaps (same-as-today, made explicit).** A failed or skipped store never retries; the tokens are permanently absent remotely and later stores start above the gap (G4). X9 closes as accepted behavior with reason: a store is cache-warmth optimization; retry machinery isn't justified, and gaps cannot corrupt token-addressed remote content.
3. **Intents are clamped to produced coverage at the fold.** Before the seam submits (submission already waits on the completion event), a pending intent's `end` is clamped to `Align(filled_len)`; an intent clamped to or below its start is dropped. E7's semantics ("store only what was actually produced") preserved with no post-batch hook.
4. **Releasability is transaction state; `Quiesce()` replaces `Drain` (deliberate non-redesign).** Releasable iff `retiring && inflight == 0 && pending_stores.empty() && !pending_retrieve && record terminal` — semantically today's `CanRetire` predicate, testing sequence fields instead of the coordinator's session map, so the loop's `Retire` consumes transaction output and `CanRetire`/`OnRetire` have nothing left to guard. `Quiesce()` (called from `Join` after the loop stops) does exactly R2: drop unsubmitted intents, blocking-drain submitted ones through the seam, force records terminal, no collectives, daemon TTL covers stragglers; the destructor's checks verify quiescence. Retirement gating *semantics* are correct today and survive intact. Warm-up needs no store-path rule: warm-up sequences are ineligible (`kNone`), so no record, no cursor, no intents — exclusion rides eligibility (ticket 09).
5. **S5c becomes a stated contract leaf (X8 closed; amended 2026-09-22 by the admission re-grilling).** When checkpoint-category bytes move and an admitted forward ends exactly on an external-coverage chunk boundary, the chunk-boundary snapshot slot is **decided at the publication decision point** — an additional trigger for the same `pending_publish` plan in `PlanPublication`, with `end` in hand, independent of the publication min-interval and generation-cache gates — **and allocated in optional admission** from inactive memory, via the existing optional allocator. A failed allocation defers the store staging (a permanent cursor gap per X9), never the forward. Originally this leaf allocated the slot in required admission via the PR's in-loop push; the re-grilling killed the push (ticket 06's amendment) and the leaf was re-expressed on the publication machinery, which is pre-PR native. See the resolution comment for S5b's death.

**Code sketch** (design-level):

```cpp
// Publication stage of Schedule(), after MarkProduced:
if (s.external.state != CoverageState::kNone && Align(produced_end) > s.external.store_cursor) {
    StoreIntent intent{
        .start        = s.external.store_cursor,
        .end          = Align(produced_end),           // chunk-aligned; clamped at fold if the forward finishes early
        .kv_sources   = PrefixSlotsFor(s, s.external.store_cursor, Align(produced_end)),
        .ckpt_sources = CheckpointSlotsAtBoundaries(s, s.external.store_cursor, Align(produced_end)),  // G5-split
    };
    intent.ready = pass.batch_done_event;              // completion token; the seam submits on it
    // (no pin here — the seam's Submit acquires, per the `transfer_refs` verdict)
    s.external.store_cursor = Align(produced_end);     // advances at commit, chunk-aligned; gaps permanent
    s.pending_stores.push_back(std::move(intent));
}

// Completion fold (Reconcile), before the seam submits (ticket 04's lifecycle: not yet in flight; no pins held):
for (auto it = s.pending_stores.begin(); it != s.pending_stores.end();) {
    if (it->in_flight) { ++it; continue; }
    it->end = std::min(it->end, Align(s.filled_len));
    it      = it->end <= it->start ? s.pending_stores.erase(it) : std::next(it);
}

bool Scheduler::Releasable(const Sequence& s) const {
    return s.retiring && s.inflight == 0 && s.pending_stores.empty() && !s.pending_retrieve
           && (s.external.state == CoverageState::kDone || s.external.state == CoverageState::kNone);
}
```

**What dies or closes** (coverage-audit hooks): E6, E7, E8's `SubmitStores` half (→ publication-stage planning + event-token submission + fold clamping); S5b (→ dropped: the publication-decision-point trigger and the store intent own the snapshot, `Sequence::store_checkpoint` dies); G4/X9 (→ accepted-with-reason); X8 (→ contract leaf); `CanRetire`/`OnRetire`/`Drain` as coordinator methods (→ `Releasable`/`Quiesce` under scheduler ownership); R1's cancel-skips-stores behavior folds into intent drop at fold; R2/R3 map to `Quiesce` + destructor checks.

**Handed downstream:** today's T5 rank-0 store-range broadcast may die outright — native planning is deterministic from cross-rank-identical state (cursor, published coverage), leaving only the checkpoint-availability AllReduce as a real collective → ticket 09 decides. Cancel-with-pending-store composition with `contracts.cancel-release` → ticket 09 (intent drop semantics are set here; the cancel edges are 09's).

Check against `contracts.prefix-publish`, `contracts.checkpoint-adoption`, `invariants.cleanup`, and `checklist.delayed-release`.

---

## Resolution (amendment, 2026-09-22 — admission re-grilling)

Two changes landed from ticket 06's re-grilling of `RunRequiredAdmission`'s shape:

1. **S5b dies whole.** `PlanStoreCheckpoint` (planning-stage arming of `Sequence::store_checkpoint`) has no owner left: eligibility moved to the branch predicate at the publication decision point, node resolution to `PlanPublication` (`end` in hand) and `CheckpointSlotsAtBoundaries`, the remembered pointer's only consumers were the dead S5c push and its `TM_CHECK_EQ` assert, and the arming's pre-landing protection covered a window that no longer exists — allocate → publish → hand to the seam happens inside one pass, and the seam's `AcquireTransfer` at submission is the only pin (the pre-submit window needs none: the sources are stamped involved slots above `pass.floor`, unreachable by the optional phase's eviction). The audit gains an S5b row (Dropped); V1's "armed-snapshot state kept" is corrected — the state dies.
2. **S5c re-expressed on the publication machinery** (point 5 above). The in-loop push dies; the store becomes an additional trigger for the same `pending_publish` plan at `PlanPublication`, allocated by the existing optional allocator (`try_optional`, inactive memory only, defer-on-skip); a skipped snapshot is a permanent cursor gap per X9. The audit row moves to Changed.

**Code picture** (store side after both amendments; anchors are PR-tree line numbers):

Deleted outright: `PlanStoreCheckpoint` (`scheduler.cc:713-736`), its call site (`:1140`), `Sequence::store_checkpoint` and its clearing site (`request.h`, `scheduler.cc:800`), and the S5c in-loop push (`:1265-1267`). The store leaves no trace in the planning stage and none in required admission.

The store trigger sits in `PlanPublication`, before the publication gates (same head and node resolution as pre-PR):

```cpp
// (b) Checkpoint onto the node. [pre-PR comment stands]
if (!CheckpointPublicationEligible() || !registry_.has_checkpoint() || node == nullptr) {
    return;
}
// Store snapshot on an exact chunk landing: the node's checkpoint slot,
// planned for the optional phase exactly like a publication checkpoint.
// Independent of the publication gates below (min interval and
// generation-cache suppression are publication policy, not store policy).
// Participation rides the coverage record (spec §5's clamp condition,
// third term) — input-embeds sequences are ineligible (unprojectable),
// while fingerprinted multimodal spans are content-addressed remotely
// via the token-id projection (ticket 08).
if (external_chunk_size_ > 0 && end % external_chunk_size_ == 0
    && s.external.state != CoverageState::kNone && !is_valid(node->checkpoint)) {
    if (!node->checkpoint) {
        node->checkpoint = cache_.Create(registry_.checkpoint().object_id(), node);
    }
    TM_CHECK(pass.planned.insert(node->checkpoint.get()).second);
    pass.pending_publish[i] = {node, end, node->checkpoint.get()};
    pass.has_optionals      = true;
    return;
}
// ... publication gates and planning below: pre-PR verbatim ...
```

`RunOptionalAdmission` consumes it pre-PR verbatim — `try_optional` allocates from inactive slots only (the sweep bounded by `pass.floor`) and on skip leaves the node's slot unallocated, retried only on a later exact landing. One-pass timeline: plan (`PlanPublication`) → allocate (optional) → run the forward → publish/attach → enqueue → the seam submits on the batch event, acquiring the transfer pins (`AcquireTransfer`) → fold. A skipped snapshot G5-splits around the missing slot: the chunk ships partial or gaps — never a stall, never a retry.

**Audit (same day, buy-nothing sweep across the record).** Killed: `PinSources` — the pin is the seam's at `Submit` (the `transfer_refs` verdict; this ticket's original sketch line amended above); `local_prefix_end` — `FindResume` gains the `prefix_end` field it already computes internally, no second walker (ticket 05's sketch amended); `pass.planned_retrieve` — the planned intent is Sequence-owned `planned_retrieve` (the PR's `restore_plan` placement, its targets already staged in the Sequence's `alloc_blocks`; declared in spec §4); `CeilDiv` — the existing `ceil_div`; `AlignDown` — G1's `Align`; `UnsubmittedStores`/`DropIntent` — inlined in the fold sketch. Verified earned and left alone: `FindResume`, `PlanRequests`, `AcquireTransfer`, `PlanPublication`/`ClampForwardEnd` machinery, `ExternalEligible`, `Releasable`/`Quiesce`, the intent and record types, `ExternalMemoryService`, `external_chunk_size_`, `AbandonCoverage`/`Query`/`healthy`. Not existing vocabulary, kept as design names (corrected 2026-09-23): `batch_done_event`/`BatchDoneEvent` (the intent's ready token — the PR passes a `shared_ptr<CudaEvent>` recorded via `RecordReady`, `engine/lmcache.cc:603-616`) and `PrefixSlotsFor`/`CheckpointSlotsAtBoundaries`.

**Correction (same day):** `FreshSlotsFor` was listed above as earned — wrongly. It renamed the PR's `PrepareRestore` destination walk and hid its structure; killed by relocating that walk into planning (inlined in the §5 sketches, `PlanResume`). `RetrieveIntent` keeps a single `targets` vector — the intent's own record of its slots, needed by submit (destination pointers, transfer pins) and both fold outcomes (install-swap, abandon-deallocate); staging itself pushes straight to `involved_blocks`/`alloc_blocks`. The two destination kinds (prefix slot per logical block, checkpoint slot per chunk boundary) live in the walk, not the type — unlike `StoreIntent`'s split, which is earned by G5's differential semantics.

**Correction (2026-09-23 audit):** the store-trigger sketch above originally used the retired `transfer_chunk_size_` name, omitted the participation term (`s.external.state != kNone`), and carried a backwards comment (token-id replacement is what *makes* multimodal KV remotely addressable, per ticket 08); as written it would arm store snapshots for external-ineligible sequences — the exact X2 coupling spec §5 fixes. Rewritten in place to the locked naming and the three-part scoping; the dropped `input_embeds` terms are subsumed by participation.
