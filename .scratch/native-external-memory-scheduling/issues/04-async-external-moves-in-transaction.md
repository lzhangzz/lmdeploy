# Async external moves inside the scheduler transaction

Type: grilling
Status: resolved
Blocked by: 03

## Question

How does the synchronous scheduler transaction express asynchronous External Moves?

The transaction is plan-then-commit (`contracts.cache-prepare`, `contracts.scheduler-commit`); retrievals and stores complete out-of-band on the service's schedule. Today this tension is resolved by the external `RequestSession` phase machine gating `Schedulable()` from the engine loop — the patch being replaced.

Decide:

- What an External Move looks like as transaction output: is a retrieve a planned copy like `restore_copies` (with a remote source), a distinct intent kind, or a new transaction concept?
- What state a sequence holds while an External Move is in flight, and which existing deferral family it belongs to (producer-conflict skip, allocation-failure defer-and-stop) — or whether a new transaction-visible wait state is needed.
- How completion re-enters the transaction (the `Poll`/batch-complete path replaced by what?), including partial completion and failure.
- How the transaction expresses readiness as output so the engine loop needs no external-memory-specific gating (the interface ownership locked in charting Q1).
- Concurrency bounds: in-flight External Moves per sequence and per pass (may graduate to a separate ticket if this stalls).

Check against `principles.delayed-cleanup`, `invariants.cleanup`, `checklist.forward-progress` (a request waiting on a retrieve must not livelock the empty-pass OOM path).

## Answer

Resolved 2026-09-21 through grilling (Q1–Q5 confirmed, Q3 sharpened through a vocabulary challenge). The transaction absorbs async by owning **two loop entries** and **in-flight intent state on the `Sequence`**; the coordinator's phase machine and its three-hook handshake dissolve into the normal plan/commit/reconcile cycle.

**The decision.**

1. **Two scheduler entries, no external gating.** The loop calls `Reconcile()` then `Schedule(...)` every iteration. `Reconcile()` polls the seam and folds terminal outcomes into coverage records and installs; it runs even on passes with nothing schedulable, because finished-but-unretired sequences still need stores to drain (replaces `Poll()`, E8). `Schedule()` receives **all non-retiring sequences** — the external-memory eligible-set filter (`Schedulable`, E3) moves inside the transaction. The four hooks that existed only to run the retrieve handshake (`PrepareSchedule`, `Schedulable`, `OnScheduled`, `Poll`) are gone by design. `Reconcile()` precedes `Schedule()` so terminal outcomes inform the same pass's planning (today's loop order, E2/E8, already has this shape).
2. **External-wait is a skip, not a stop.** A sequence with `CoverageState::kLookup` or an in-flight retrieve is not planned into the pass at all — the producer-conflict family (`contracts.prefix-conflict`: skip and continue), never defer-and-stop: no forward, no targets, nothing to admit; it holds `is_active = false` per `contracts.scheduler-inactive` and consumes no resources (no per-pass arm/clear churn — the P5–P7 handshake is dead). A sequence with a *planned* retrieve is planned into the pass **forward-less** — no resume or forward derivation — so required admission can allocate its staged targets (amended 2026-09-22 alongside ticket 06's re-grilling). No forward overlaps a planned or in-flight retrieve (same as today, per the capability scope lock).
3. **Intent state on the `Sequence`.** Scheduler-owned, replacing the coordinator's `PendingRetrieve`, `stores_`, and `pending_stores`:
   - `ExternalCoverage external` — knowledge (state/extent/store_cursor, ticket 03);
   - `std::unique_ptr<RetrieveIntent> planned_retrieve` — planned in the pass, targets staged into `alloc_blocks`; null when none, at most one (added by the 2026-09-22 amendments: the PR's `restore_plan` placement — a deferred intent persists and is retried, not re-derived);
   - `std::unique_ptr<RetrieveIntent> pending_retrieve` — null when none, at most one per sequence;
   - `std::vector<StoreIntent> pending_stores` — outbound moves draining per pass.
   Intent lifecycle: planned (on the sequence, in the pass) → submitted (`Schedule()` hands it to the seam; a store's submission gates on its ready event) → **in flight** (transfer running, not terminal; lasts arbitrarily many iterations — the seam reports terminal outcomes only, G2) → folded (`Reconcile()` consumes the outcome and clears it). The coverage **query is not a move and not an intent**: `kLookup` is record state; a Query goes through the seam whose terminal outcome sets `extent` and initializes `store_cursor` — success folds the record to `kKnown`, failure to `kDone` with extent 0 and cursor 0 (stores still run; local recompute proceeds; corrected 2026-09-23). Bounds are same-as-today by construction (one retrieve per sequence, stores batched per pass); tunable bounds are policy, out of scope — the map's concurrency fog line is cleared.
4. **Readiness is transaction liveness state, health-gated.** The scheduler exposes "external work in flight or sequences waiting on it" for the loop's head-of-line failure check (replacing `HasPendingReleases`, E9). The suppression stays **health-gated** exactly as today (F5/X6): while external memory is unhealthy, waiting does not suppress `kOutOfMemory` head-of-line failure — unhealthy external memory may never unblock, and hanging forever is worse than failing.
5. **Install-safety is a producer-mark check.** At `Reconcile()`, a successful retrieve installs only if no foreign `LogicalBlock::producer` mark sits on the target nodes (`contracts.prefix-ownership`) — the transaction's own exclusion state replaces the coordinator's `CanInstall` peek (P9/F37). Install refused or transfer failed → record goes to `kDone` (abandoned), the fresh slots are dropped, and the next pass plans local recompute. Uncertain transfers simply stay in flight (buffers pinned service-side) until terminal.

**Vocabulary ruling (surfaced by this ticket's grilling).** The word "tier" (a placement stratum — the local pool or External Memory) is retired from the working vocabulary: the design has exactly two placements, both already named. The contract's "required tier / optional tier" (`contracts.scheduler-admission`) is a different sense (admission phase) that ticket 06 would collide with; the spec renames that wording when it amends those leaves ("required admission / optional admission"), and code names avoid `tier_` prefixes entirely. (The retirement was completed globally by the resume decision, ticket 05.)

**Code sketch** (design-level; builds on ticket 03's types):

```cpp
// Engine loop, per iteration — external-memory-specific loop logic is gone:
//   scheduler_.Reconcile();                       // poll seam, fold outcomes
//   scheduler_.Schedule(non_retiring_sequences);  // transaction decides skips

void Scheduler::Reconcile() {
    ExternalEvents events;
    service_.Poll(events);  // terminal outcomes only: success/failure per intent
    for (auto& q : events.queries) {      // coverage queries
        auto& r = Find(q.sequence).external;
        r.extent       = q.matched;       // 0 on failure — stores still run
        r.store_cursor = q.matched;
        r.state        = q.matched > 0 ? CoverageState::kKnown : CoverageState::kDone;
    }
    for (auto& t : events.retrieves) {
        auto& s = Find(t.sequence);
        if (t.success && !HasForeignProducer(s, *s.pending_retrieve)) {
            Install(*s.pending_retrieve);        // ticket 03 install semantics
        } else {
            s.external.state = CoverageState::kDone;  // abandon; recompute next pass
        }
        s.pending_retrieve.reset();              // fresh slots dropped on abandon
    }
    for (auto& st : events.stores) {
        EraseInFlight(st);                       // sources unpinned service-side
    }                                             // retirement gating is ticket 07's
}

// Inside PlanRequests — two skips, not stops:
//   if (s.external.state == CoverageState::kLookup || s.pending_retrieve) {
//       continue;  // external-wait: not planned into the pass at all (producer-conflict family)
//   }
//   if (s.planned_retrieve) {
//       // in the pass, forward-less: no resume/forward derivation; the staged targets
//       // are its required set (ticket 06's loop allocates them)
//   }
```

**What dies by this decision** (coverage-audit hooks): the `RequestSession` phase machine and its 13 transitions (P1–P13), the eligible-set hiding (E3/X4), the arm/admit/retry handshake (P5–P7, X5), engine hooks `PrepareSchedule`/`OnScheduled`/`Poll` as external-memory call sites (E2/E4/E8), `HasPendingReleases` as coordinator state (E9), and `CanInstall` as a coordinator peek (P9).

**Handed downstream:** store-intent submission timing and the ready-event fencing detail (client-side destination/source exposure control, G4/F29 — the one-loop-turn deferral today) → ticket 07; rank-symmetric participation in `Reconcile()` (the T7/X7 early-return-condition hazard) and TP-wide health aggregation → ticket 09; admission of retrieve intents (required-but-async) → ticket 06; where the retrieve's `[preserve_end, end)` calculus and `RetrieveEnd` clamp land → ticket 05.

**Amendment (from the sharing decision, ticket 08):** install is no longer unconditional on terminal success: a completed retrieve whose target nodes are already valid (another sequence installed first) drops its fresh slots instead of installing — fold-time subsumption. Install requires terminal success ∧ install-safety (producer marks) ∧ targets-not-already-valid.

**Correction (2026-09-23 audit):** the query fold in the sketch above originally set `kDone` on success ("knowledge acquired") — that starves the retrieve-decision rule (ticket 05/spec §5 fires on `kKnown`), so no retrieve would ever be planned. Fixed: query success → `kKnown` (extent, `store_cursor = extent`), failure → `kDone` (0/0). Same-as-today mapping: the PR's post-lookup transition moves a session to `kRetrieve`-waiting (≈ `kKnown`) on success and straight to `kReady` (≈ `kDone`, matched 0 / cursor 0) on failure (`lmcache.cc:585-594`).
