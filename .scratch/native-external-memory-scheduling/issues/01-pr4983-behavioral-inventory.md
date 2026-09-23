# PR 4983 scheduling behavioral inventory

Type: research
Status: resolved

## Question

What is the complete scheduling-relevant behavior of the current LMCache integration (PR 4983, this tree), as a baseline for the coverage audit the native-model spec must pass (acceptance item 1)?

Inventory at minimum:

- The `RequestSession` phase machine inside `src/turbomind/engine/lmcache.cc` (lookup / retrieve / ready / store phases, and every transition trigger).
- Every engine-loop hook call site in `src/turbomind/engine/engine.cc` (`OnAccepted`, `PrepareSchedule`, `Schedulable`, `OnScheduled`, `Fallback`, `StageStores`, `OnBatchComplete`, `Poll`, `Drain`, `HasPendingReleases`, `CanRetire`, `OnRetire`) — what scheduling decision each one participates in.
- The scheduler-side probes (`Scheduler::ProbeResume`, `PrepareRestore`, `CompleteRestore`, and the `transfer_chunk_size` constructor arg in `src/turbomind/engine/scheduler.h`): what scheduler state each reads or mutates, and when.
- Eligibility exclusions in `OnAccepted` (warm-up, `rope_base`, ppl/logits/hidden-state outputs, input embeds, empty multimodal fingerprints) and rank gating (rank-0-only lookup).
- Granularity and alignment rules (`chunk_size % logical_block_size` check, `Align`, `RetrieveEnd`, preserve-end logic around `pools_.size() > 1`).
- The idle/OOM fallback path: how `HasPendingReleases` keeps the loop alive and how `Fallback` picks and handles a victim.
- Retirement gating (`CanRetire`/`OnRetire`) and pending-store draining at shutdown.
- Which cache categories move today (prefix KV at block granularity; checkpoint-category parts at chunk granularity — see `Register()` in lmcache.cc).

Codebase-internal only; no web or prior-art research. Deliver findings as an asset under `.scratch/native-external-memory-scheduling/assets/`, link it from the resolution, and resolve this ticket per the tracker's wayfinding operations.

## Answer

Asset: [assets/01-pr4983-inventory.md](../assets/01-pr4983-inventory.md) — 13 sections, ~90 stable-id items (C/P/E/S/G/L/F/R/M/T/H/V/X), each a behavior + `file:line` citation + the scheduling decision it affects, usable directly as the acceptance-item-1 coverage checklist.

Summary of what it covers: construction/registration/geometry (enable gate, allocator side effects, chunk÷block check, pool registration at block/chunk granularity); the full `RequestSession` phase machine with all 13 transition triggers (lookup → retrieve-waiting → in-flight → ready, including unhealthy/fallback/retire/exhausted-local-coverage exits); all 12 engine-loop hook call sites with their positions in the loop and the decision each participates in; the scheduler probes (`ProbeResume` pure read; `PrepareRestore` wholesale-replaces normal planning and rides required admission without resource consumption; `CompleteRestore` install semantics) plus the `transfer_chunk_size` machinery (chunk-boundary forward splitting, checkpoint arming, required allocation on exact landing, publication bypassing the min interval); granularity/alignment rules (`Align`, `RetrieveEnd`, preserve-end `end − chunk` clamp with checkpoints, monotonic store cursor, store splitting on missing checkpoints); eligibility exclusions and rank gating; the idle/OOM fallback path and `HasPendingReleases` liveness; retirement gating and the collective-free shutdown `Drain`; which categories move (prefix KV at block granularity both directions, checkpoint category at chunk granularity, frontier populated only by the next PlanResume); the six TP collectives and the rank-symmetric state they presuppose; the failure/health model (per-op fallback, `uncertain` pinning, self-healing `transfers_healthy_`); and the `Sequence` fields touched.

Most audit-relevant surprises: `LmCache::Ready` is dead public API; enabling LMCache globally reshapes scheduler forward-splitting/checkpoint cadence via the `transfer_chunk_size` ctor arg even for external-ineligible sequences; eligibility is triplicated across `OnAccepted`/`PrefixEligible`/`Validate`; readiness is expressed by hiding sequences from the scheduler rather than in the transaction; `HasPendingReleases` is gated on `transfers_healthy_` (OOM head-of-line failure is not suppressed while draining); the store-correctness dependency on required admission of chunk snapshots exists only as a cross-file comment; and `next_store_token` never rewinds, so skipped chunks are permanently absent remotely.

(map.md decision-line append deferred to the parent session per its ownership.)

