# Native External-Memory Scheduling — Design Spec

Status: locked design (2026-09-22). Implementation is a separate follow-up effort.
Sources: decision tickets `issues/03`–`issues/09` (this directory), research assets `assets/01-pr4983-inventory.md` (item ids `C/P/E/S/G/L/F/R/M/T/H/V/X`) and `assets/02-service-boundary.md` (facts `F1–F51`, gaps `G1–G10`), the normative contract `src/turbomind/engine/README.md` (cited `<section>.<leaf>`), and the post-ticket `transfer_refs` verdict. Each section cites the ticket that lands it.

## 1. Framing

PR 4983 integrates LMCache as an engine-loop coordinator (`LmCache`) that gates scheduling from outside (`Schedulable` eligibility hiding, E3), patches the Scheduler's output through probes (`ProbeResume`/`PrepareRestore`/`CompleteRestore`, S1–S4), and reshapes global scheduler behavior through a constructor argument (`transfer_chunk_size`, C6/S5, X2). That framing is rejected. The native model puts external memory **inside the scheduler transaction**: the Scheduler's own planning, admission, resume selection, publication, and retirement machinery treat External Memory as the second place sequence cache state can live, readiness is expressed by the transaction itself, and the engine loop has no external-memory-specific logic. The LMCache client (`src/turbomind/lmcache/`) survives unchanged as external memory's I/O service behind a seam.

Scope locks (charting): spec-only artifact; the transaction owns the transaction–loop interface; clean-slate target with a coverage audit (§12); mechanism only — policies and tuning out; prior-art surveys forbidden; capability same-as-today, all cache categories; `engine/README.md` amendments are *specified* here (§11) and land with the implementation, per `checklist.contract-sync`.

## 2. Domain model

Glossary (`/CONTEXT.md`): **External Memory** — sequence cache state held outside the engine's local pool, backed by LMCache; **External Coverage** — the per-sequence extent of cache state, across all cache categories, known to exist in external memory; **External Move** — a scheduler-planned relocation of sequence cache state between the local pool and external memory, planned as an intent inside the scheduler transaction, executed asynchronously by external memory's I/O service (retrieve = inbound, store = outbound). The word "tier" is eliminated from the vocabulary: the design has two placements, both named, and external placement is *not* structural (§3).

## 3. Representation — planning state, not structural state (ticket 03; amended by 05, 08)

The structural model (`LogicalBlock`, `CacheBlock`, `PrefixTrie`) stays purely local and gains no external concepts. The unified-index alternative (remote slots on `LogicalBlock`, trie nodes kept alive by external backing) is rejected: the service cannot be enumerated or introspected (G2), remote validity is opaque (daemon LRU, lease TTL — a local "externally valid" mark goes silently stale), and holding trie metadata for all stored content is unbounded growth duplicating an index the daemon keeps.

State lives on the `Sequence`, scheduler-owned:

```cpp
enum class CoverageState {
    kNone,    // ineligible, external memory absent, or nothing to look up
    kLookup,  // coverage query in flight, extent unknown
    kKnown,   // extent known, content not yet realized locally
    kDone     // extent consumed or abandoned; cursor stays meaningful for stores
};

struct ExternalCoverage {
    CoverageState state        = CoverageState::kNone;
    int           extent       = 0;  // chunk-aligned prefix length known remotely
    int           store_cursor = 0;  // first token not yet staged for store (monotonic)
};

struct RetrieveIntent {
    int    start;         // chunk-aligned; may sit below preserve_end, service skips the overlap
    int    preserve_end;  // block-aligned local prefix kept as-is
    int    end;           // chunk-aligned retrieve end (§5 cap)
    std::vector<CacheBlock*> targets;  // fresh destinations: one prefix slot per logical block in
                                       // [preserve_end, end) + one checkpoint slot per chunk boundary in
                                       // (start, end]; each slot's category fixes its install destination
    bool   admitted = false;  // set by required admission; Schedule()'s commit submits
};

struct StoreIntent {
    int    start;  // chunk-aligned, >= external.store_cursor
    int    end;    // chunk-aligned produced end
    std::vector<CacheBlock*> kv_sources;    // node prefix slots covering [start, end)
    std::vector<CacheBlock*> ckpt_sources;  // node checkpoint slots at chunk boundaries in (start, end]
    BatchDoneEvent ready{};           // the producing batch's completion token; the seam submits when it fires
    bool           in_flight = false; // the seam's transfer has started (lifecycle, §4)
};
```

- External Moves are **first-class planned intents**, not `(src, dst)` cache-block copy plans: a remote endpoint is not a `CacheBlock` (no allocation handle, timestamp, or evictability); forcing it would lie to engine-thread address resolution (`principles.device-content`).
- **Install erases provenance**: terminal success (and install-safety, §4) swaps allocations into `node->prefix`/`node->checkpoint` slots, marks `is_valid`, stamps — PR 4983's S4 semantics survive — and the content is native local content thereafter. No provenance bit; the consumed extent feeding the store cursor is the only residue.
- The service reports a single chunk-granular matched length (F19): the record holds one extent even though realization installs two categories; per-category availability is learnable only at transfer time.
- External-memory geometry: a single fact, `external_chunk_size_` (0 = external memory absent), replaces the `transfer_chunk_size` constructor argument (C6). A descriptor struct (`ExternalDesc`) was sketched and dropped — `enabled` is redundant with nonzero granularity, `moves_checkpoints` derives from the scheduler's own `registry_`.
- The seam exchanges **intents down, terminal outcomes up, poll-driven** (the service is callback-free, F14/F15). Leases, `BlockId` addressing, CUDA ready events, and the client-side destination-fencing hazard (G4) stay below it. The coverage **query is not a move and not an intent**: `kLookup` is record state; a Query goes through the seam whose terminal outcome sets `extent` and initializes `store_cursor` — success folds to `kKnown` (extent known, `store_cursor = extent`), failure folds to `kDone` with extent 0 / cursor 0 (stores still run; local recompute proceeds; corrected 2026-09-23).

```cpp
class ExternalMemoryService {  // the seam: intents down, terminal outcomes up, poll-driven
public:
    void Submit(const RetrieveIntent&, Sequence&);  // lease delegation is service-side
    void Submit(const StoreIntent&, Sequence&);     // sources/destinations pinned here (AcquireTransfer)
    void AbandonCoverage(Sequence&);                // releases lease; cancels if pre-submission
    void Query(Sequence&);                           // rank-0 lookup; results broadcast (§9)
    // Poll() hands up terminal results only (success/failure per intent); no
    // partial progress exists to report (G2).
};
```

## 4. Transaction semantics for async moves (ticket 04; amended by 08)

**Two scheduler entries; no external gating.** The loop calls `Reconcile()` then `Schedule(...)` every iteration. `Reconcile()` polls the seam and folds terminal outcomes; it runs even on passes with nothing schedulable (finished-but-unretired sequences still need stores to drain). `Schedule()` receives **all non-retiring sequences** — the external-memory eligible-set filter (E3) moves inside the transaction. The four hooks that existed only to run the retrieve handshake (`PrepareSchedule`, `Schedulable`, `OnScheduled`, `Poll`) are gone by design. `Reconcile()` precedes `Schedule()` so terminal outcomes inform the same pass's planning.

**External-wait is a skip, not a stop.** A sequence with `CoverageState::kLookup` or an in-flight retrieve is not planned into the pass at all — the producer-conflict family (`contracts.prefix-conflict`), never defer-and-stop: no forward, no targets, nothing to admit; it holds `is_active = false` per `contracts.scheduler-inactive` and consumes no resources (the P5–P7 arm/clear handshake is dead). A sequence with a *planned* retrieve is planned into the pass **forward-less** — no resume or forward derivation — so required admission can allocate its staged targets (§6); it too is inactive until install. No forward overlaps a planned or in-flight retrieve (same as today).

**Intent state.** `Sequence` carries the planned and in-flight intents: `std::unique_ptr<RetrieveIntent> planned_retrieve` (planned in the pass, its target slots staged into `alloc_blocks`; null when none, at most one — the PR's `restore_plan` placement; a deferred intent persists and is retried, never re-derived), `std::unique_ptr<RetrieveIntent> pending_retrieve` (submitted, in flight; null when none), and `std::vector<StoreIntent> pending_stores`. Intent lifecycle: planned (on the sequence, in the pass) → submitted (`Schedule()` hands it to the seam; a store's submission gates on its ready event) → **in flight** (transfer running, not terminal) → folded (`Reconcile()` consumes the outcome and clears it). Bounds same-as-today by construction; tunable bounds are policy, out of scope.

**Fold semantics.** Terminal retrieve outcome installs only if: terminal success ∧ install-safety (no foreign `LogicalBlock::producer` mark on target nodes — the transaction's own exclusion state replaces the coordinator's `CanInstall` peek, P9/F37) ∧ targets-not-already-valid (fold-time subsumption, ticket 08: another sequence installed first → drop the fresh slots instead of installing). Refused or failed → record `kDone` (abandoned), fresh slots dropped, next pass plans local recompute. Uncertain transfers stay in flight until terminal (buffers pinned service-side). Query outcomes set `extent`/`store_cursor`: success folds to `kKnown`, failure to `kDone` with 0/0 (stores still run). Store outcomes release source pins; retirement gating is §7.

**Readiness is health-gated transaction liveness.** The scheduler exposes "external work in flight or sequences waiting on it" for the loop's head-of-line failure check (replacing `HasPendingReleases`, E9). Suppression is health-gated exactly as today (F5/X6): while external memory is unhealthy, waiting does *not* suppress `kOutOfMemory` head-of-line failure — unhealthy external memory may never unblock, and hanging forever is worse than failing. Pins are unbounded; while external memory is unhealthy, a fitting request can be failed for a pin-related cause — accepted with reason (same as today).

## 5. Resume selection and clamping (ticket 05)

`resume_len` stays **purely local** — external coverage never enters resume precedence. Proof travels with the store; install is the proof event; the extent is advisory knowledge (acquired once, lease-bounded, never per-pass revalidatable — G2). `principles.resume-proof` holds: neither generic validity nor external knowledge raises `resume_len`, only its installed result does.

**The retrieve-decision rule** (inside `PlanResume`, after local candidate selection) replaces P4/P5 and `ProbeResume`:

```cpp
// local: the FindResume result PlanResume already holds — no second call; it gains
// prefix_end, a value FindResume already computes internally
const int L = local.candidate.pos;  // frontier/checkpoint/fork, as today
auto&    ext = s.external;
if (ext.state == CoverageState::kKnown && !s.planned_retrieve && !s.pending_retrieve) {
    // a planned intent persists until admitted — the guard prevents overwrite, no re-derive
    if (L >= ext.extent) {
        service_.AbandonCoverage(s);          // lease released service-side
        ext.state = CoverageState::kDone;  // local subsumes external
    } else if (service_.healthy()) {        // single-rank view; TP aggregation is §9
        const int end      = std::min(ext.extent, Align(s.prompt_len - 1));
        const int preserve = std::min(local.prefix_end,
                                      registry_.has_checkpoint() ? end - external_chunk_size_ : end);
        auto intent = std::make_unique<RetrieveIntent>();
        intent->start = Align(preserve); intent->preserve_end = preserve; intent->end = end;
        // The PR's PrepareRestore walk, relocated into planning. Its two remaining parts are
        // owned elsewhere and not re-derived here: out-of-range nodes' existing prefixes (the
        // ordinary involved set) and the frontier (PlanResume).
        for (auto& node : s.block_ids) {
            if (preserve <= node->offset && node->offset < end) {
                intent->targets.push_back(cache_.Create(registry_.prefix().object_id()));
            }
        }
        for (int pos = intent->start + external_chunk_size_; pos <= end; pos += external_chunk_size_) {
            auto& node = *s.block_ids[pos / logical_.block_size() - 1];
            if (!node.checkpoint) {
                node.checkpoint = cache_.Create(registry_.checkpoint().object_id(), &node);
            }
            intent->targets.push_back(cache_.Create(registry_.checkpoint().object_id()));
        }
        for (auto* b : intent->targets) {  // fresh slots are never valid: unconditional staging
            s.involved_blocks.push_back(b);
            s.alloc_blocks.push_back(b);   // the required set §6's lambda allocates
        }
        s.planned_retrieve = std::move(intent);
    }
}
// resume_len itself is computed exactly as today — from local validity only.
```

Local wins, external gap-fills above it; the G3 trailing-chunk refresh rule and the `prompt_len − 1` cap (`invariants.executable-context`) are kept; in-flight subsumption applies only pre-submission (F31) — post-submission moves drain and install-or-drop (§4).

**Chunk-boundary clamping is scoped per sequence** (fixes X2, a deliberate behavior change from PR 4983):

```cpp
external_chunk_size_ > 0                        // external memory present (0 = absent)
&& registry_.has_checkpoint()                   // chunk-positioned categories move
&& s.external.state != CoverageState::kNone     // this sequence participates in external coverage
```

Under this condition the chunk boundary joins block boundaries and `B` as a `ClampForwardEnd` boundary candidate (S5a splitting and the S5d interval-bypass follow it; the §7 chunk-boundary snapshot trigger rides the same participation); ineligible sequences get pre-external behavior. Installed nodes are ordinary `is_valid` nodes, counted into `readonly_block_num` by the next `PlanResume` (`invariants.readonly-block-num` untouched); an external-waiting sequence is inactive with `input_len == 0`, so executable-context and async-progress bookkeeping are untouched.

## 6. Admission (ticket 06)

The two-phase structure stands (renamed per §11: required admission / optional admission). No third phase.

- **Move targets ride required admission through the shared allocation path** (amended by re-grilling; see the ticket's resolution comment). Planning stages a planned retrieve's target slots into `alloc_blocks`; the unique_id-ordered required loop allocates them through the same `try_allocate_required` lambda as forwards — atomic per intent via the lambda's rollback (`contracts.allocation`), eviction up to the sequence's cutoff, same discipline as forwards. Allocation failure stops the pass, exactly as any required-admission failure; the deferred intent persists and is retried next pass (planning re-derives nothing: the sequence ran nothing, so the plan's inputs are unchanged). A genuinely-never-fitting intent reaches the stall table's too-large row.
- **Inbound lifecycle:** reserve (plan) → allocate (admit) → pin (submit) → install-or-drop (fold). Protection is continuous: in-flight targets are the sequence's `involved_blocks`; install happens in `Reconcile()` and the same iteration's `Schedule()` picks the installed nodes into its required set.
- **Outbound moves are unadmitted.** Stores allocate nothing; their only capacity side-effect is the source transfer-pin (§10).
- **No retrieve-to-recompute conversion** (choice B): the atomicity argument does not survive scrutiny — both paths need the full context resident, so recompute banks nothing the wait wouldn't. F4 is dropped with reason: its rescue case is narrow and pathological, and terminal routing is achieved more simply by failing.

**Stall-handling contract:**

| Stall cause | Handling |
|---|---|
| Healthy in-flight moves (own or others') | Liveness suppression: wait, pins self-release |
| Unhealthy/uncertain pins | Suppression lifts; front-most non-suppressed victim fails (`kOutOfMemory`) — F5 semantics kept |
| Genuinely too-large request | `kOutOfMemory` per `checklist.forward-progress` |

```cpp
// Required admission: pre-PR shape plus the PR's try_allocate_required lambda
// and the planned-retrieve branch; everything else pre-PR verbatim
// (SortedBlocks listing, cutoff guard, Test, clamp, producer check).
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

## 7. Store path, publication, retirement (ticket 07)

- **Store intents are planned at the publication stage** (relocation with reason: the staging arithmetic is unchanged from `StageStores`; what changes is ownership, fencing, and the deaths of E6/E7/E8's `SubmitStores` half). Sources are the just-published valid slots; ranges chunk-aligned; G5 splitting around missing checkpoint slots; the intent carries the producing batch's completion event as its ready token, and the seam submits when it fires.

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
    // (no pin here — the seam's Submit acquires, AcquireTransfer, per §3/§10)
    s.external.store_cursor = Align(produced_end);     // advances at commit, chunk-aligned; gaps permanent
    s.pending_stores.push_back(std::move(intent));
}
// Completion fold, before the seam submits (§4 lifecycle: not yet in flight; no pins held):
for (auto it = s.pending_stores.begin(); it != s.pending_stores.end();) {
    if (it->in_flight) { ++it; continue; }
    it->end = std::min(it->end, Align(s.filled_len));
    it      = it->end <= it->start ? s.pending_stores.erase(it) : std::next(it);
}
```

- **The cursor advances at commit; failures leave gaps** (G4/X9 accepted with reason: a store is cache-warmth optimization; retry machinery isn't justified; gaps cannot corrupt token-addressed remote content).
- **Releasability is transaction state** (deliberate non-redesign — the predicate is today's `CanRetire` semantics testing sequence fields):
  `retiring && inflight == 0 && pending_stores.empty() && !pending_retrieve && record terminal (kDone|kNone)`.
- **`Quiesce()` replaces `Drain`** (from `Join`, after the loop stops): drop unsubmitted intents, blocking-drain submitted ones through the seam, force records terminal; no collectives, no session-END RPC, daemon TTL covers stragglers; the destructor's checks verify quiescence.
- **S5c becomes a stated contract leaf** (X8 closed; amended by the admission re-grilling — see ticket 07's resolution comment): when checkpoint-category bytes move and an admitted forward ends exactly on an external-coverage chunk boundary, the chunk-boundary snapshot slot is **decided at the publication decision point** — an additional trigger for the same `pending_publish` plan in `PlanPublication`, independent of the publication gates — **and allocated in optional admission** from inactive memory via the existing optional allocator. A failed allocation defers the store staging (a permanent cursor gap, X9), never the forward. The PR's planning-stage arming (S5b, `PlanStoreCheckpoint`, `Sequence::store_checkpoint`) and its in-loop push die. See §11.
- Warm-up needs no store-path rule: warm-up sequences are ineligible (§9), so no record, no cursor, no intents.

## 8. Sharing and identity (ticket 08)

- **No cross-sequence dedup.** Each sequence retrieves its own copy; sharing is emergent — the first install lands in trie-matched nodes (`is_valid`), the second sequence's next planning finds `L >= E` and subsumes. Fold-time subsumption (§4) cheapens the in-flight window's tail. Cross-request reuse, refcounting (`ownership.prefix`), and eviction behave exactly as for locally produced content; there is no separate indexing step and no parallel external index.
- **Remote identity is token-level and content-true.** `_replace_multimodal_token_ids` (lmdeploy/turbomind/turbomind.py:70, applied whenever `lmcache_addr` is set) rewrites multimodal spans to per-image cache-key ids (`int(fingerprint.hex(), 16) & 0xffff`, the vLLM/LMCache projection) before the engine sees the tokens. `cache_salt` stays unset — G7 closes as by-design, redundant. Recorded caveat: the projection is 16 bits (birthday-bound collisions across many distinct images); the local trie retains full 256-bit fingerprint identity on top; ecosystem-standard, service-side, out of scope.
- **Leases stay per-sequence, service-side, implied by intent lifecycle** — held while coverage is unconsumed, delegated on retrieve, released on abandon; concurrent leases on the same content are ordinary read locks.

## 9. Lifecycle edges (ticket 09)

**Cancel follows the retiring flag through the fold.** Exact order: (1) cancel observation sets `retiring = true` — excluded from scheduling immediately; (2) the same iteration's `Reconcile()` drops every *unsubmitted* intent and abandons the coverage record (lease released; a submitted retrieve's locks already belong to the daemon, F32); (3) the same iteration's `Schedule()` does not plan it; (4) the drain window folds TP-reduced outcomes — the retrieve's result is discarded (retiring refuses install), stores fold and unpin; an in-flight forward completes first per unchanged `invariants.cleanup`; (5) `Retire` releases once releasability holds. `contracts.cancel-release` holds; the service's pre-submission-cancel + mandatory-drain semantics suffice.

**One eligibility truth.** `ExternalEligible(s)` is a single scheduler-owned predicate consulted at the admission analog; the record exists iff it holds (`state != kNone` *is* the eligibility fact — the clamp condition's third term):

```cpp
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
// if (external_chunk_size_ > 0 && ExternalEligible(s)) { create record; service_.Query(s); }
```

All static exclusions kept with documented reasons (rope_base: tokens don't encode it; ppl/logits/hidden-state: resume-skipping invalid for them; embeds and empty fingerprints: unprojectable). None made expressible — same-as-today bar; expressibility recorded as deliberately not taken. `PrefixEligible` and `Validate` stay (different domains).

**Rank symmetry: reduce before folding.** All rank divergence enters through collectives inside `Reconcile()`. Query results: rank-0 lookup + broadcast (T2). Transfer outcomes: TP-reduced before folding (a retrieve succeeding on one rank but failing on another folds as failure on both — everyone abandons; no rank folds before all-finished). Store planning is deterministic from cross-rank-identical state. Fold-time subsumption agrees because everything feeding it does. Collectives after this decision:

| Fact | Mechanism | Verdict |
|---|---|---|
| Query result | rank-0 lookup + broadcast (T2) | kept — inside the seam/fold |
| Transfer outcomes | AllReduce-MIN before fold (T6) | kept — reduce-then-fold |
| Checkpoint availability | AllReduce-MIN at planning | kept |
| Store ranges | rank-0 broadcast (T5's range half) | **dies** — deterministic planning |
| Health/liveness | MIN aggregation | kept — feeds the liveness output |

The spec states the invariant that kills T7/X7: *every collective's participation set is a pure function of cross-rank-identical transaction state.*

## 10. Transfer pinning (`transfer_refs` verdict)

`CacheBlock::transfer_refs` **survives and is necessary**: the daemon accesses device memory by raw IPC address (no stale-detection can protect it), so hard exclusion is the only correctness mechanism for in-flight move sources/destinations; soft stamping cannot reach waiting sequences' targets (skipped by planning); releasability covers only request-owned slots; and the fatal checks in `Deallocate`/`Invalidate` guard every free path, current and future. What changes is the driver: the seam acquires at submit and releases at fold (intent lifecycle), replacing coordinator choreography.

Pins are unbounded — no budget, no enforcement point, same as today. While external memory is healthy, liveness suppression waits in-flight pins out; while it is unhealthy, a request whose footprint fits within the configured maximum session length can be failed for a pin-related cause — accepted with reason (§4).

## 11. Specified `engine/README.md` amendments

Landed with the implementation, per `checklist.contract-sync` (the contract is updated in the same change as the code). Exact wording of new and changed leaves:

**`concepts` — new leaves:**

- `external-memory`: Sequence cache state held outside the engine's local pool, backed by an I/O service behind a seam. External placement is planning state, not structural state: the logical-block/cache-block/trie model stays purely local, and external content becomes structural only at install, landing in ordinary slots as native local content.
- `external-coverage`: The per-sequence extent of cache state, across all cache categories, known to exist in external memory; advisory knowledge acquired once at query, lease-bounded, never per-pass revalidatable.
- `external-move`: A scheduler-planned relocation of sequence cache state between the local pool and external memory, planned as an intent (retrieve: range + fresh target slots; store: range + source slots) inside the transaction and executed asynchronously by the service. A remote endpoint is not a `CacheBlock`.
- `reconcile`: The scheduler entry that polls the seam and folds terminal outcomes, run every engine-loop iteration even when nothing is schedulable; precedes `Schedule()`.

**`contracts.cache-prepare` — amended:** planning may additionally plan external-move intents under the retrieve-decision rule (local subsumes at `L >= extent`; otherwise one gap-filling inbound intent with the preserve rule and the `prompt_len − 1` cap) and plan store intents at the publication stage. External coverage never raises `resume_len`.

**`contracts.scheduler-commit` — amended:** `Schedule()` is also the commit point for external-move intents (submission to the seam) and the store cursor. Boundary candidates include the external-coverage chunk boundary only under: `external_chunk_size_ > 0 && registry_.has_checkpoint() && external.state != kNone`.

**`contracts.scheduler-admission` — amended (admission wording):** "required admission" / "optional admission" replace the current "required tier" / "optional tier" wording (the old leaves still carry it until this amendment lands). A sequence with a planned retrieve stages the intent's target slots into its required set; failure stops the pass as any required-admission failure does. New sentence: *when checkpoint-category bytes move and an admitted forward ends exactly on an external-coverage chunk boundary, the chunk-boundary snapshot slot is decided at the publication decision point and allocated in optional admission from inactive memory; a failed allocation is a store cursor gap, never a forward stall* (the S5c leaf; store-intent checkpoint sources exist when staged).

**`contracts.external-fold` — new:** `Reconcile()` folds terminal outcomes: a retrieve installs only on success ∧ no foreign producer mark on targets ∧ targets-not-already-valid; otherwise the record is abandoned and the next pass recomputes locally. Store outcomes release source pins. Query outcomes set extent and store cursor. All rank divergence enters through collectives inside `Reconcile()`; *every collective's participation set is a pure function of cross-rank-identical transaction state* (outcomes are TP-reduced before folding; query results are broadcast).

**`contracts.external-move-pins` — new:** In-flight move sources/destinations are transfer-pinned (`CacheBlock::transfer_refs`, acquired at submission, released at fold) and structurally un-evictable; `Deallocate`/`Invalidate` fatal-check the count. The seam owns acquire/release; nothing address-like or lease-like crosses it.

**`contracts.releasability` — new:** A sequence is releasable iff `retiring && inflight == 0 && pending_stores.empty() && !pending_retrieve && record terminal`. Cancellation follows the retiring flag: unsubmitted intents die in the same iteration's fold; submitted moves drain with results discarded. `Quiesce()` (from `Join`) drops unsubmitted intents, blocking-drains submitted ones, forces records terminal — no collectives.

**`contracts.external-eligibility` — new:** One scheduler-owned predicate decides participation (record existence = eligibility): non-warm-up, non-empty tokens, `rope_base == 0`, no ppl/all-logits/hidden-state outputs, no input embeds, all spans fingerprinted — each exclusion documented (unprojectable or forward-requiring).

**`checklist` — new leaves:** `external-fold` (do folds consume only reduced outcomes, with participation rank-symmetric?).

## 12. PR 4983 coverage audit

Dispositions: **Expressed** (native home), **Changed** (with reason), **Dropped** (with reason), **Unchanged** (service/implementation side, out of this spec's reach). Item ids per `assets/01-pr4983-inventory.md`.

| Items | Disposition | Where / reason |
|---|---|---|
| C1 enable gate | Expressed | no-op seam + `external_chunk_size_ = 0` |
| C2 allocator/layout side effect | Unchanged | engine-shape concern, lands with implementation |
| C3 session-id broadcast | Unchanged | seam/service wiring |
| C4 chunk divisibility check | Expressed | kept at wiring (`TM_CHECK`) |
| C5 pool registration | Unchanged | seam (`Register`) |
| C6 `transfer_chunk_size` ctor arg | Dropped | replaced by `external_chunk_size_` (§3) |
| P1–P13 phase machine | Dropped | replaced by `CoverageState` + intents + fold (§4); all 13 transitions mapped to plan/commit/fold/retiring edges |
| E1 `OnAccepted` | Expressed | `ExternalEligible` + record creation + `Query` (§9) |
| E2 `PrepareSchedule` | Dropped | planning rule (§5) |
| E3 `Schedulable` | Dropped | external-wait skip (§4) |
| E4 `OnScheduled` | Dropped | commit submits intents (§4) |
| E5 `Fallback` | Dropped | choice B — conversion unjustified (§6) |
| E6 `StageStores` | Dropped | publication-stage planning (§7) |
| E7 `OnBatchComplete` truncation | Dropped | fold clamp (§7) |
| E8 `Poll` | Dropped | `Reconcile()` (§4) |
| E9 `HasPendingReleases` | Dropped | health-gated liveness output (§4) |
| E10 `CanRetire` | Dropped | `Releasable` (§7) |
| E11 `OnRetire` | Dropped | fold discards on retiring (§9) |
| E12 `Drain` | Dropped | `Quiesce()` (§7) |
| E13 dead `LmCache::Ready` | Dropped | dead public API, no consumer |
| S1 `ProbeResume` | Dropped | planning runs `FindResume` in-process (§5) |
| S2 `PrepareRestore` | Dropped | intents replace wholesale plan-swap (§3); destination walk relocated into planning (§5) |
| S3 restore admission special case | Expressed | planned-retrieve branch of the required loop, shared lambda, explicit `else` for the PR's `continue` fall-through; failure breaks as today (§6) |
| S4 `CompleteRestore` | Expressed | install semantics survive as fold (§3/§4; amended by fold-time subsumption) |
| S5a chunk splitting | Changed | scoped per sequence — X2 fix (§5) |
| S5b planning-stage arming | Dropped | publication-decision-point trigger + store intent own the snapshot; `Sequence::store_checkpoint` dies (§7) |
| S5c required allocation on landing | Changed | in-loop push dies; decided at the publication decision point, allocated in optional admission; failure is an X9 gap (§7, §11) |
| S5d interval bypass | Expressed | follows the clamp condition (§5) |
| G1 `Align` | Expressed | intent planning (§5) |
| G2 `RetrieveEnd` | Expressed | intent `end` cap (§5) |
| G3 preserve rule | Expressed | kept (§5) |
| G4 store cursor | Expressed | kept; gaps accepted with reason — X9 closed (§7) |
| G5 store splitting | Expressed | kept in planning (§7) |
| G6 pin mapping | Expressed | seam-driven `AcquireTransfer` (§3, §10) |
| L1 eligibility exclusions | Expressed | `ExternalEligible` with reasons (§9) |
| L2 rank gating | Expressed | rank-0 query + broadcast (§9) |
| F1–F3 stall guard/victim | Expressed | liveness + stall table (§4, §6) |
| F4 LMCache relief valve | Dropped | choice B (§6) |
| F5 health-gate caveat | Changed | kept deliberately — health-gated suppression, pins unbounded, accepted (§4) |
| R1 cancel/retirement | Expressed | five-step cancel order + releasability (§9, §7) |
| R2 `Drain` | Expressed | `Quiesce()` (§7) |
| R3 destructor checks | Expressed | quiesce verification (§7) |
| M1/M2/M3 categories | Expressed | all categories, both directions; no generation-time offload (unchanged capability) |
| T1 session broadcast | Unchanged | seam wiring |
| T2 lookup broadcast | Expressed | kept (§9) |
| T3 proof AllReduce | Dropped | deterministic planning comparison (§5, §9) |
| T4 admission-status AllReduce | Dropped | native required admission is rank-symmetric by determinism (§6, §9) |
| T5 range broadcast / availability | Changed/Dropped | range broadcast dies (determinism); availability MIN kept (§9) |
| T6 outcomes MIN | Expressed | reduce-then-fold (§9) |
| T7 early-return condition | Expressed | killed by the participation invariant (§9) |
| H1 per-op fallback | Expressed | fold abandon → local recompute (§4) |
| H2 uncertain pinning | Expressed | stays in flight until terminal (§4) |
| H3 health flag | Expressed | liveness aggregation (§4, §9) |
| H4 pin invariants | Expressed | `transfer_refs` (§10) |
| V1 `lmcache_matched_end`/`store_checkpoint`/`restore_plan` | Expressed/Dropped | extent on the record; `store_checkpoint` arming dies (S5b); `restore_plan` replaced by intents |
| V2/V3 reads/indirect writes | Expressed | transaction-owned planning/fold state |
| X1 dead API | Dropped | with E13 |
| X2 global reshape | Changed | scoped clamp condition — the fix itself (§5) |
| X3 triplication | Expressed | one predicate for the external path (§9) |
| X4 hiding | Dropped | the skip, inside (§4) |
| X5 three-pass handshake | Dropped | plan/commit/fold (§4) |
| X6 health gate | Expressed | kept (§4) |
| X7 loop-shape agreement | Expressed | participation invariant (§9) |
| X8 cross-file comment | Expressed | contract leaf (§7, §11) |
| X9 cursor gaps | Expressed | accepted with reason (§7) |

## 13. Scenario walkthroughs

**S-A. Retrieve-in-flight under admission pressure.** Requests A (uid 1, external hit, extent ≫ local) and B..F (local, memory-hungry). Pass 1: A is `kKnown`, no pending move → the retrieve-decision rule plans one intent (§5); required admission allocates its staged targets through the shared lambda against B..F's forwards (§6) — suppose it allocates: `Schedule()` commits and submits; the seam pins destinations; A is now external-waiting — skipped, inactive, no resources re-derived (§4). B..F proceed; A's targets sit in its `involved_blocks`, protected. The transfer completes; `Reconcile()` reduces outcomes across ranks (§9), installs (targets valid, no foreign producer, not already valid), and the same iteration's `Schedule()` plans A's resume from installed validity. Variant: admission fails under pressure — the pass stops at A and the intent is retried next pass (§6); an in-flight transfer keeps the stall path suppressed (§4); otherwise, if nothing at all can admit, the front-most non-suppressed victim fails per the stall table (§6).

**S-B. Cancel with pending store.** Request R finishes generating; its final store intents are committed and submitted (source pins held); the user cancels a *later* turn — `retiring` flips at cancel observation; the same iteration's `Reconcile()` drops R's unsubmitted intents and abandons coverage (lease released); R's submitted stores drain — outcomes fold, pins release; `Retire` fires when `inflight == 0 && pending_stores.empty() && record terminal` (§9's five steps, §7's releasability). No other sequence's state is touched; no cross-pass cleanup state exists.

**S-C. Shared-prefix double-retrieve.** Requests P and Q share a 40-chunk prompt; local trie is cold; both queries return extent 40; both plan retrieves (no dedup, §8). P's completes first: reduced outcome, installs into the trie-matched nodes — both hold request refs on the same nodes; `is_valid`. Q's next planning computes `L >= E` → subsumes, abandons its coverage (lease released). If Q's retrieve was already in flight, its fold drops the fresh slots (fold-time subsumption) — duplicate bandwidth was paid only inside the window; identity holds (token-keyed, content-true per §8). Third arrival R finds a trie hit and never touches the external path.

**S-D. Warm-up.** Warm-up requests are `ExternalEligible == false` (dynamic `is_warm_up_`): no record, no query, no cursor, no intents — and the clamp condition's third term is false, so no chunk splitting (matching today's S5 "not warm-up" term). Recurrent checkpoint publication suppression during warm-up is the pre-existing contract leaf, untouched. After warm-up ends, subsequently admitted requests participate normally.

## 14. Out of scope (recap)

LMCache protocol/client internals; other engines; prior-art surveys; scheduling policies and tuning (thresholds, workloads); implementation, migration execution, rollout ordering; generation-time KV offload beyond PR 4983; evict-to-external demotion. See the map's Out of scope for the full statements with reasons.
