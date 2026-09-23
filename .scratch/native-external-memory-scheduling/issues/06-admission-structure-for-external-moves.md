# Admission structure for external moves

Type: grilling
Status: resolved
Blocked by: 04

## Question

Where do External Moves sit in the admission model?

`contracts.scheduler-admission` fixes two phases: required (prefix blocks + frontier, evicts up to cutoff, defers on failure) and optional (publication, fork-to; never evicts active state, never defers a forward). Decide:

- Is a retrieve needed for a chosen resume point *required-admission material* (the forward cannot run without it) even though it completes asynchronously — and if so, what "required but async" means for the two-phase replay, or whether admission gains a third phase for external moves.
- How admission accounts local memory for inbound retrieved content (it lands in local slots: when are those reserved, protected, and released if the request is canceled mid-retrieve).
- How stores sit in admission: optional like publication, or unadmitted background intent.
- Whether the mechanism admits evict-to-external (demotion under local pressure) at all, or only store-on-publication; if demotion is in, what replaces the timestamp/LRU decision input (mechanism only — the policy itself is out of scope).
- Interaction with the eviction-protection set (`invariants.protection-set`): are retrieved-but-not-yet-consumed blocks protected, and by whom.

Check against `contracts.allocation` (atomicity at the transaction boundary), `contracts.eviction`, and `checklist.scheduler-commit` (Schedule() remains the only commit point).

## Answer

Resolved 2026-09-22 through grilling (Q1–Q4 confirmed; Q5 settled as (B) after the retrieve-to-recompute conversion was challenged and not defended). The two-phase admission structure stands — no third phase — with inbound moves riding required admission. **Amended later the same day** by a re-grilling of the loop shape (resolution comment below): defer-not-stop reversed to stop-the-pass, the first draft's invented per-kind allocators dropped, and the loop returned to the pre-PR shape plus the shared lambda and the planned-retrieve branch.

**The decision.**

1. **Move targets ride required admission through the shared allocation path** (amended by re-grilling; originally "defer-not-stop"). Planning stages a planned retrieve's target slots into the sequence's `alloc_blocks`; the unique_id-ordered required admission loop allocates them through the same `try_allocate_required` lambda as forwards (atomic per intent via the lambda's rollback, all targets or none, `contracts.allocation`; eviction up to the sequence's cutoff, same discipline as forwards). Allocation failure stops the pass, exactly as any required-admission failure — the deferred intent persists and is retried next pass (planning re-derives nothing: the sequence ran nothing, so the plan's inputs are unchanged). Vocabulary per ticket 04's ruling: required admission / optional admission.
2. **Inbound memory lifecycle.** Planning creates fresh intent-owned target slots → required admission allocates → commit submits (service transfer-pins destinations) → reconcile folds: install (slots become node slots, stamped) or abandon (deallocated with the intent). Protection is continuous: in-flight targets are the sequence's `involved_blocks`; install happens in `Reconcile()` and the same iteration's `Schedule()` picks the installed nodes into its required set — no gap. Cancel-mid-retrieve release follows the intent's fold (edges are ticket 09's).
3. **Outbound moves are unadmitted.** Stores allocate nothing and sit outside admission entirely; their only capacity side-effect is the source transfer-pin (structurally un-evictable for the move's duration, H4), whose pressure risk is carried by liveness.
4. **Demotion ruled out of scope.** No evict-to-external under pressure; stores stay production-driven. Reasons: new capability beyond the same-as-today scope lock, and a deadlock-shaped hazard — uncertain transfers pin more local memory exactly when memory is scarce, so pressure-triggered demotion can deepen the pressure it meant to relieve (map's Out of scope updated).
5. **No retrieve-to-recompute conversion (choice B).** A non-suppressed stalled victim fails with `kOutOfMemory` directly; liveness suppression carries *all* legitimate waiting. Today's F4 mercy rule is dropped with reason, audit-noted: the conversion's rescue case is narrow and pathological (nothing admitting anywhere, yet a clamped slice fits), and its other effects (terminal routing, lease release) are achieved more simply by failing. The atomicity argument for conversion does not survive scrutiny: both paths need the full context resident to complete, so recompute banks nothing the wait wouldn't.

**Stall-handling contract** (as settled here plus tickets 04/05):

| Stall cause | Handling |
|---|---|
| Healthy in-flight moves (own or others') | Liveness suppression: wait, pins self-release |
| Unhealthy/uncertain pins | Suppression lifts; front-most non-suppressed victim fails (`kOutOfMemory`) — F5 semantics kept deliberately |
| Genuinely too-large request | `kOutOfMemory` per `checklist.forward-progress` |

**Vocabulary rulings from this ticket's grilling:** "stall handling" and "victim" are the engine's existing concepts (the front-most, non-retiring, zero-inflight request a deadlocked pass terminates); "relief" was coined and dropped before adoption — the mechanism, had it survived, would have been named by its parts (stall-triggered abandonment); "move-carrying"/"forward-carrying" (first draft) and "move-carrier" (the re-grilling) were dropped the same way — a sequence is named by its state ("a sequence with a planned retrieve"), never by a coined carrier noun. None earned a glossary entry. **Style rulings from the re-grilling:** exclusive branches are written `if`/`else`, never one arm ending in `continue` to reach the other; the forward path is the positive (`if`) branch, the loop's normal case; success goes in the `if`, `break` in the `else`.

**Code sketch** (admission loop shape; pre-PR flat loop plus the PR's lambda and the planned-retrieve branch — everything else pre-PR verbatim):

```cpp
auto try_allocate_required = [&](int i) -> bool {
    // PR body verbatim: alloc/evict loop over pass.requests[i]->alloc_blocks
    // with eviction up to pass.cutoff[i]; rollback of planned blocks on
    // failure; committed_replay_size / max_evict_ts / evict_pos commits on
    // success.
};

for (int i = 0; i < static_cast<int>(pass.requests.size()); ++i) {
    auto& s = *pass.requests[i];

    if (max_evict_ts >= pass.cutoff[i]) {
        break;
    }

    if (!s.planned_retrieve) {                     // forward: the normal case
        // resource.Test, clamp, producer check (pre-PR verbatim), then:
        if (try_allocate_required(i)) {
            resource.Commit(s);
            s.is_active       = true;
            pass.committed[i] = true;
            PlanPublication(pass, i, s, end, at_prompt_boundary);
            SetProducers(s, begin, end);
        } else {
            break;
        }
    } else {                                       // planned retrieve: targets only
        if (try_allocate_required(i)) {
            s.planned_retrieve->admitted = true;   // this pass's Schedule() commits and submits
        } else {
            break;                                 // as any required failure; intent retried next pass
        }
    }
}
```

**What dies by this decision** (coverage-audit hooks): S3's no-Test/Commit restore admission special case (→ the planned-retrieve branch of the same loop, explicit `else` in place of the PR's `continue` fall-through), P5–P7's retryable arm/clear cycles (→ the intent persisted and retried each pass), F4's `Fallback` conversion (→ dropped, choice B), and the demotion idea (→ out of scope). The re-grilling additionally killed this ticket's first-draft defer-not-stop semantics and its invented per-kind allocator helpers (`TryAllocateMoveTargets`/`TryAllocateRequired`/`DeferAndStop`): the shared `try_allocate_required` lambda is the only allocation path.

---

## Resolution (re-grilling, 2026-09-22)

Re-opened by challenge: *`RunRequiredAdmission` should be kept as close as possible to the pre-PR shape.* Established against `git show dc83a0193~1:src/turbomind/engine/scheduler.cc`: pre-PR is a flat single loop (cutoff guard → `resource.Test` → clamp → producer check → inline alloc slice → `Commit`/`is_active`); the PR added exactly four things to the function — the `try_allocate_required` lambda extraction, the restore branch (which **breaks** on failure, `scheduler.cc:1230` current), the `store_checkpoint` push (S5c), and the `SortedEvictableBlocks` listing.

1. **Loop shape:** pre-PR verbatim, plus two keeps — the lambda and the branch. The branch predicate is `s.planned_retrieve` (a retrieve in flight is external-waiting and skipped upstream, so `pending_retrieve` never reaches this loop). The move arm calls the *same* lambda: planning stages the intent's target slots into `alloc_blocks`, exactly as the PR's restore branch staged restore slots. Explicit `else` (no `continue` fall-through), forward path as the positive `if` branch, success in the `if` / `break` in the `else`.
2. **Failure semantics reversed:** defer-not-stop → `break`. Keeping the lambda verbatim forces it: the lambda's failure path only returns false, and uncommitted replay (including the failed attempt's evictions) is discarded at the function tail *because the caller breaks*; a `continue` would let a later sequence's success commit the failed attempt's eviction entries into the append-only replay. Defer would require an inline `pass.replay.resize(pass.committed_replay_size)` in the lambda — a third departure from pre-PR. A genuinely-never-fitting intent reaches the stall table's too-large row.
3. **`SortedEvictableBlocks` reverts** textually to `SortedBlocks()`; pin exclusion stays a contract leaf (`contracts.external-move-pins`), not a function shape — where the filter lives is the impl ticket's call.
4. **Consequences on ticket 07:** S5c dies with the push (its only allocation site); the chunk-boundary snapshot moved to the publication decision point + optional admission, and S5b's planning-stage arming died with it — recorded in ticket 07's amendment.

Audit corrections: S3 → Expressed (same branch, explicit `else` for the PR's `continue` fall-through; failure breaks as today). The first draft's claim that defer-not-stop "matches today's S3" was false — today's restore branch breaks — and is moot under ruling 2.

**Correction (same day, correctness pass).** A deferred intent *persists* on the sequence and is retried — planning does not re-derive it (the sequence ran nothing, so `FindResume` inputs and the extent are unchanged); the retrieve-decision rule guards `!s.planned_retrieve` accordingly, preventing an overwrite that would re-create targets while the old ones sit staged in `alloc_blocks`. Consequent wording: "replanned next pass" became "persists and is retried" throughout. Related fixes in the same pass: the §4 skip now covers planned retrieves (no forward planned, but the sequence stays in the pass for target allocation), the intent lifecycle's in-flight state is "transfer running" (not "submitted"), and the intent structs declare their lifecycle fields (`admitted`, `ready`, `in_flight`).
