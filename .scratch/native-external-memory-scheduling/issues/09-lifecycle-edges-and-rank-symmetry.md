# Lifecycle edges: cancel, warm-up, rank symmetry, eligibility

Type: grilling
Status: resolved
Blocked by: 05, 06, 07

## Question

How do the lifecycle edges compose with the native model? Stress-test the decided paths against:

- **Cancel**: a request canceled with a retrieval or store in flight — who aborts the move, what happens to reserved inbound slots and partial coverage, and how the abort composes with `contracts.cancel-release` (no cross-pass ownership state to clean up). Does the service's cancellation semantics (from ticket 02) suffice?
- **Warm-up**: `is_warm_up` excludes sequences from the integration today (`OnAccepted` early return); where does that exclusion live natively — an eligibility predicate in the transaction, analogous to `PrefixEligible`?
- **Rank symmetry**: lookup runs on rank 0 only and results must agree across ranks. `principles.boundary-policy` demands decisions be pure functions of cross-rank-identical attributes. Decide how external facts (coverage, completion) enter every rank's transaction identically — and what happens if the service completes a retrieval on some ranks but not others. Input from the async-transaction decision: folding happens in `Reconcile()`, so rank-symmetric *participation* in `Reconcile()` is now the load-bearing shape (the T7/X7 hazard — the early-return condition itself must be collective-participation state), and the health-gated liveness output (ticket 04's Q4) needs its TP-wide aggregation decided here (today's AllReduce-MIN over `[healthy, per-store, per-retrieve]`, T6). Input from the store-path decision: today's T5 rank-0 store-range broadcast may die outright (native planning is deterministic from cross-rank-identical state), leaving only the checkpoint-availability AllReduce — decide which collectives remain and where they live in the transaction.
- **Static eligibility exclusions**: today `OnAccepted` excludes dynamic-rope (`rope_base`), ppl/logits/hidden-state outputs, input embeds, and empty multimodal fingerprints. Decide where each lives in the native model and whether any should become expressible rather than excluded. Input from the resume decision: the chunk-clamp condition's third term (`state != kNone`) relies on eligibility being carried by the coverage record — settling the predicate's home settles that term. Input from the sharing decision: the empty-fingerprint exclusion is *unprojectability* — spans without fingerprints cannot be turned into content-true token ids (`_replace_multimodal_token_ids` skips them), so the exclusion is a correctness requirement, not a tuning choice.

Check against `invariants.cleanup`, `checklist.async-progress`, and the boundary-policy purity rule.

## Answer

Resolved 2026-09-22 through grilling (Q3/Q4 confirmed directly; Q1 confirmed with the exact event order drawn out; Q2 confirmed as relocation-with-reason after "how is this better?"). One relocation, one stated invariant, one event order, one predicate.

**The decision.**

1. **Cancel follows the retiring flag through the fold.** Exact order: (1) Cancel observation sets `retiring = true` — excluded from all future scheduling immediately; (2) the same iteration's `Reconcile()` drops every *unsubmitted* intent (staged stores before the seam submits, planned-but-unadmitted retrieves with their targets, source pins released) and abandons the coverage record, releasing the lease service-side (a submitted retrieve's locks already belong to the daemon, F32, and ride the transfer out); (3) the same iteration's `Schedule()` does not plan it; (4) the drain window folds TP-reduced outcomes — the retrieve's result is discarded (retiring refuses install), stores fold and unpin; an in-flight forward at cancel time completes first per the unchanged `invariants.cleanup`; (5) `Retire` releases once the releasability conjunction holds. No step touches another sequence's state; `contracts.cancel-release` holds (intents are per-sequence state whose fold handles retiring — no cross-pass ownership cleanup). The service's semantics (pre-submission cancel, mandatory drain) suffice.
2. **One eligibility truth in the transaction (relocation with reason).** `ExternalEligible(s)` is a single scheduler-owned predicate consulted at the admission analog; the coverage record exists iff it holds, so `state != kNone` *is* the eligibility fact (closing the clamp condition's third term from ticket 05). Warm-up is dynamic, read through the `is_warm_up` reference the Scheduler already holds. X3's triplication is not merged wholesale: `PrefixEligible` and `Validate` govern different domains and stay; what dies is the external-eligibility copy living outside the transaction. The value is ownership and the death of a cross-file agreement invariant — not capability; no behavior changes.
3. **Rank symmetry: reduce before folding.** All rank divergence enters through collectives inside `Reconcile()`, never through divergent transaction state. Query results: rank-0 lookup + broadcast (T2, kept — the seam runs the query on rank 0, every rank folds the broadcast). Transfer outcomes: **TP-reduced before folding** — a retrieve succeeding on rank 0 but failing on rank 1 folds as failure on both (everyone abandons, recompute — today's F36/F38), and no rank folds before the reduction reports all-finished. Store planning is deterministic from cross-rank-identical state, so today's T5 range broadcast **dies**; the checkpoint-availability AllReduce-MIN remains at planning. Fold-time subsumption (ticket 08) agrees because everything feeding it does. The spec states the invariant that kills T7/X7: *every collective's participation set is a pure function of cross-rank-identical transaction state.*
4. **Static exclusions stay excluded, reasons documented in the predicate.** `rope_base` (dynamic NTK: KV depends on it, tokens don't encode it), ppl/all-logits/hidden-state outputs (the request needs full prompt forwards — resume-skipping invalid for it), input embeds (unprojectable: embeddings determine KV), empty fingerprints (unprojectable, per ticket 08). None made expressible: each could be (e.g. `rope_base` via `cache_salt`), but the same-as-today bar and remote-content fragmentation say no; expressibility is recorded as deliberately not taken.

**Collectives inventory after this decision:**

| Fact | Mechanism | Verdict |
|---|---|---|
| Query result | rank-0 lookup + broadcast (T2) | kept — inside the seam/fold |
| Transfer outcomes | AllReduce-MIN before fold (T6) | kept — reduce-then-fold |
| Checkpoint availability | AllReduce-MIN at planning (T5's availability half) | kept |
| Store ranges | rank-0 broadcast (T5's range half) | **dies** — deterministic planning |
| Health/liveness | MIN aggregation (T6 entry 0) | kept — feeds the liveness output |

**Code sketch:**

```cpp
// scheduler.cc — the single external-eligibility truth. The record exists iff true.
bool Scheduler::ExternalEligible(const Sequence& s) const {
    return !is_warm_up_                                                    // dynamic, time-based
        && s.token_ids                                                     // non-empty prompt
        && s.rope_base == 0.f                                              // dynamic NTK changes KV
        && !s.gen_cfg.return_ppl                                           // needs full prompt forwards
        && s.gen_cfg.output_logits != GenerationConfig::kAll
        && s.gen_cfg.output_last_hidden_state != GenerationConfig::kAll
        && s.input_embeds.empty() && s.input_embeds_offsets.empty()        // unprojectable
        && std::all_of(s.multimodal_spans.begin(), s.multimodal_spans.end(),
                       [](const auto& span) { return !span.fingerprint.empty(); });
}
// Admission analog: if (external_chunk_size_ > 0 && ExternalEligible(s)) { create record; service_.Query(s); }
```

**What dies** (coverage-audit hooks): the coordinator's `OnAccepted` eligibility copy (L1) and its dynamic warm-up early-return (→ `ExternalEligible`), P10's retiring-retrieve special case (→ fold discards on retiring), R1's cancel-skips-stores (→ unsubmitted-intent drop at fold), T5's range broadcast (→ determinism), and X3's three-way agreement discipline for the external path (→ one predicate). The five-step cancel order and the rank-symmetry invariant are inputs to the spec's scenario walkthroughs and contract leaves respectively.
