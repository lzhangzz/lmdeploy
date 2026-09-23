# Native External-Memory Scheduling

Label: wayfinder:map

## Destination

A locked design spec for scheduler-transaction-native External Memory support in TurboMind: the Scheduler's own planning/admission/resume/eviction/publication machinery treats External Memory as a first-class placement for all cache categories, readiness is expressed by the transaction itself, and PR 4983's patch surfaces are replaced by design. Deliverable: design doc plus specified `src/turbomind/engine/README.md` contract amendments. Implementation is a separate follow-up effort.

## Notes

- Domain: TurboMind C++ engine (`src/turbomind/engine/`), scheduler transaction and its interface with the engine loop. The LMCache client (`src/turbomind/lmcache/`) survives as external memory's I/O service, unchanged in scope.
- Scope locks (charting session, 2026-09-21):
  - Terminal artifact is the spec, not code (Q1a). Spec assembly is carried into this map as its terminal ticket.
  - "Native" = not patches over the existing Scheduler's output (PR 4983's `ProbeResume`/`PrepareRestore`/`CompleteRestore` coordinator pattern is the rejected framing). The scheduling model lives in the scheduler transaction (Q2, Q4).
  - External Coverage scope = same as today's capability, all cache categories (Q3, Q4).
  - The spec owns the transaction–loop interface: readiness becomes a transaction-level concept and the engine loop's external-memory-specific hooks are designed away (Q1). Loop mechanics beyond consuming transaction output stay out.
  - Posture toward PR 4983: clean-slate target with a coverage audit; migration ordering is the implementation effort's concern (Q2a).
  - Mechanism only, no policies (Q5): the spec defines the structural model; store/retrieve timing policies and tuning are out.
  - Prior-art surveys of other engines are forbidden. There is nothing close to our Scheduler. Research tickets stay codebase-internal.
- Skills every session should consult: `grilling` + `domain-modeling` for decision tickets; `research` for research tickets. Update `CONTEXT.md` inline when a term resolves.
- Standing context: `src/turbomind/engine/README.md` is the normative contract — cite it as `<section>.<leaf>` and check every proposal against its checklist. Repo `AGENTS.md` planning rule applies: design tickets express proposals as concrete code sketches against the real headers, no placeholder ellipses.
- Acceptance bar for the spec (all four, checkable without writing code):
  1. A behavioral inventory of PR 4983's scheduling-relevant behavior exists, and every item is either natively expressed or explicitly dropped with reason.
  2. `engine/README.md` gains External Memory contract leaves such that the design would satisfy `checklist.contract-sync`.
  3. The ad-hoc surfaces (engine-loop hooks and scheduler probes) are gone by design.
  4. The mechanism is stress-tested on paper against concrete scenarios (retrieve-in-flight under admission pressure, cancel with pending store, shared-prefix double-retrieve, warm-up).

## Decisions so far

<!-- one line per closed ticket; zoom the link for detail -->

- [External-memory service boundary facts](issues/02-external-memory-service-boundary.md): the seam is pull-based (poll-to-completion, no callbacks), uncancelable after submission (stores must drain; uncertainty pins buffers with no abort), TP agreement is entirely engine-side (rank-0 lookup + broadcast + AllReduce-MIN), failure reaction is uniformly local-recompute fallback, granularity is fixed and mixed (block-granular KV, chunk-granular checkpoints); 10 explicit gaps (G1–G10) bound what the native model may not assume.
- [PR 4983 behavioral inventory](issues/01-pr4983-behavioral-inventory.md): ~90 cited behaviors in 13 sections (phase machine with all 13 transitions, 12 hook call sites, probe semantics, transfer_chunk_size reshaping of clamping/checkpoint cadence, eligibility triplication, OOM/fallback liveness, retirement gating, TP collectives, health model); 9 surprises flagged X1–X9, notably: readiness expressed by hiding sequences from the eligible set, retrieve admission as a three-hook arm/admit/retry handshake, next_store_token never rewinds, and store correctness pinned by a cross-file comment.
- [External-memory representation in the scheduler data model](issues/03-external-memory-representation.md): external placement is planning state, not structural state — the structural model (LogicalBlock/CacheBlock/PrefixTrie) stays purely local; External Coverage is a scheduler-owned record on Sequence (state/extent/store_cursor); external moves are first-class planned intents (not copy plans with fake remote endpoints); install lands retrieved content in ordinary slots as native; the seam exchanges intents for terminal outcomes, poll-driven, no leases or addresses crossing up. (Amended by the resume decision: a single `external_chunk_size_` fact replaces the ctor arg, not a descriptor struct.)
- [Async external moves inside the scheduler transaction](issues/04-async-external-moves-in-transaction.md): two scheduler entries (Reconcile folds terminal outcomes every iteration; Schedule takes all non-retiring sequences — the eligible-set filter and four engine hooks die); external-wait is a producer-conflict-family skip, never a stop, no overlap; in-flight state is pending_retrieve/pending_stores on Sequence (query is not a move); readiness is health-gated transaction liveness (F5/X6 kept); install-safety is a producer-mark check.
- [Resume selection and clamping with external memory](issues/05-resume-selection-with-external-memory.md): resume_len stays purely local — external coverage never enters resume precedence; the retrieve-decision rule (local subsumes at L >= E, otherwise plan one gap-filling inbound intent with the G3 preserve rule and prompt_len−1 cap) replaces the P4/P5 proof logic and ProbeResume; chunk-boundary clamping is scoped per sequence (`external_chunk_size_ > 0 && registry_.has_checkpoint() && state != kNone`), fixing X2 as an audit-noted behavior change. Vocabulary: "tier" eliminated — Tier Move renamed External Move.
- [Admission structure for external moves](issues/06-admission-structure-for-external-moves.md): the two-phase structure stands — inbound move targets ride required admission through the same `try_allocate_required` lambda as forwards (targets staged into `alloc_blocks` by planning, atomic per intent, cutoff discipline); allocation failure stops the pass as any required failure and the intent persists, retried next pass (amended by re-grilling: defer-not-stop reversed, loop returned to pre-PR shape + lambda + branch); inbound lifecycle is reserve→allocate→pin→install-or-drop with continuous protection; outbound moves are unadmitted (source pins only); demotion ruled out of scope; F4's retrieve-to-recompute conversion dropped (choice B) — non-suppressed stalled victims fail kOutOfMemory directly, liveness carries all waiting.
- [Store path, publication, and retirement gating](issues/07-store-path-and-retirement-gating.md): store intents planned at the publication stage and committed with the pass (relocation with reason — staging arithmetic unchanged, three engine hooks die; fencing = completion-event token below the seam); cursor advances at commit, failures leave permanent gaps (X9 accepted-with-reason); intents clamped to produced coverage at the fold; releasability is transaction state and Quiesce() replaces Drain (deliberate non-redesign); S5c becomes a stated contract leaf — decided at the publication decision point, allocated in optional admission, failure a permanent X9 gap — and S5b's planning-stage arming dies (both amended by the admission re-grilling; X8 closed).
- [Cross-request sharing and retrieval dedup](issues/08-cross-request-sharing-and-dedup.md): sharing is emergent, not built — no cross-sequence dedup (each sequence retrieves its own copy; the first install makes the second subsume at planning), with fold-time subsumption added (targets already valid at fold → drop fresh slots, amending ticket 04's install); no cache_salt folding — remote identity is already content-true because multimodal spans are projected to fingerprint-derived cache-key ids before the engine sees them (G7 closes by-design; 16-bit birthday caveat recorded); leases stay per-sequence, service-side.
- [Lifecycle edges: cancel, warm-up, rank symmetry, eligibility](issues/09-lifecycle-edges-and-rank-symmetry.md): cancel follows the retiring flag through the fold (five-step order recorded; unsubmitted intents die same-iteration, submitted drain with results discarded); one scheduler-owned ExternalEligible predicate — record existence is the eligibility fact (relocation with reason, X3's external copy dies); rank symmetry is reduce-then-fold with the stated invariant "every collective's participation set is a pure function of cross-rank-identical transaction state" (T5's range broadcast dies; query broadcast, outcome MIN, availability MIN, health MIN remain); all static exclusions kept with documented reasons, none made expressible.
- [Assemble and lock the design spec](issues/10-assemble-and-lock-the-spec.md): the spec is locked at `spec.md` — fourteen sections assembled from all resolved decisions: representation, transaction semantics, resume/admission/store, sharing, lifecycle, transfer pinning; eight specified contract amendments; the full PR 4983 coverage audit; four surviving scenario walkthroughs. The map's destination is reached.

## Not yet specified

<!-- in-scope fog; graduates as the frontier advances -->

(none — the scenario suite landed as spec §13; the contract-leaf list landed as spec §11; all fog has graduated or been ruled out of scope)

## Out of scope

<!-- work ruled beyond the destination; closed, never graduates -->

- LMCache protocol and client internals (`src/turbomind/lmcache/` redesign) — kept as the I/O service.
- Other engines: the PyTorch engine's integration and any non-TurboMind engine work.
- Prior-art surveys of vLLM/SGLang/TensorRT-LLM external-KV designs — forbidden by the dev driving this map.
- Scheduling policies and tuning (store/retrieve timing policy defaults beyond mechanism, thresholds, workload studies).
- Implementation, migration execution, and rollout ordering — follow-up effort.
- Generation-time KV offload beyond what PR 4983 does today (mid-decode external moves as a new capability).
- Evict-to-external demotion under local pressure (ruled out while resolving the admission ticket): a new capability beyond the same-as-today scope, with a deadlock-shaped hazard — uncertain transfers pin more memory exactly when memory is scarce.
