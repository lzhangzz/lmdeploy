# Asset 01 — PR 4983 scheduling behavioral inventory

Baseline for map acceptance item 1 ("every item is either natively expressed or explicitly dropped with reason").
All citations are `file:line` in this tree (branch `lmcache`). Engine contract leaves cited as
`src/turbomind/engine/README.md` `<section>.<leaf>`. Item ids (`Lnn`, `Enn`, `Snn`, …) are stable handles for the
coverage audit; the audit answers per item: *natively expressed where / dropped why*.

Primary sources: `src/turbomind/engine/lmcache.h`, `lmcache.cc`, `engine.cc`, `scheduler.h`, `scheduler.cc`,
`request.h`, `request.cc`, `block.h`, `block.cc`, `src/turbomind/turbomind.cc`, and the I/O-service headers
`src/turbomind/lmcache/{connector,lookup,retrieve,store}.h/.cc` (interface semantics only; client redesign is out of
scope per the map).

---

## 1. Construction, registration, geometry

- **C1 — enable gate is a non-empty server address.** `LmCache::Create` returns a no-op (null impl) when
  `addr.empty()` (`lmcache.cc:712-723`); every public method null-checks `impl_` (`lmcache.cc:729-812`). Scheduling
  sees a present-but-inert coordinator when disabled. Address comes from `param.lmcache_addr`
  (`turbomind.cc:326,352-357`).
- **C2 — enabling LMCache changes the cache allocator and layout.** With `use_lmcache`, the object cache uses a plain
  device `Allocator` (cudaMallocAsync cannot be IPC-exported) and `TuneObjectCacheLayout` instead of the default plan
  (`turbomind.cc:326-337`). Affects what the daemon can address, not scheduling directly — but it is an engine-shape
  side effect of external memory existing.
- **C3 — session id is random on rank 0, broadcast to all TP ranks** (`lmcache.cc:125-129`). All ranks share one
  remote session namespace.
- **C4 — chunk/block divisibility is a hard check.** `TM_CHECK(chunk_size_ % block_size_ == 0)`
  (`lmcache.cc:132-133`); `block_size_ = param.cache_block_seq_len * param.attn_cp_size` (`turbomind.cc:356`).
  Chunk granularity is a multiple of the logical block granularity, always.
- **C5 — pool registration: prefix category part 0 at block granularity; every checkpoint-category part at chunk
  granularity** (`lmcache.cc:143-153`). Guards: prefix must have exactly one part (`:146`); every pool's part bytes
  must divide the allocator page size (`:154-158`); the registered region must contain the allocator storage
  (`:159-163`). The daemon is told `base/size/storage_offset/pools` so transfers address device memory directly
  (`:164-169`).
- **C6 — scheduler learns the chunk size at construction.** `Engine::Impl` passes `lmcache.chunk_size()` as the
  `Scheduler` ctor's `transfer_chunk_size` (`engine.cc:246-254`, `scheduler.cc:286-300`), then `lmcache_.Bind(scheduler_)`
  gives the coordinator the back-pointer (`engine.cc:259`, `lmcache.cc:203-206`). The scheduler never sees the
  coordinator, only the chunk size.

## 2. `RequestSession` phase machine (`lmcache.cc:49-69`)

One session per accepted request, keyed `req->unique_id` (`lmcache.cc:185-187`). Fields: `phase`, optional
`lookup` (rank 0 only), optional `lease`, optional `retrieve` (`PendingRetrieve`), `next_store_token` (store
reservation cursor, initialized to the remote hit length), `pending_stores` counter. Helper
`WaitingForRetrieve() == (phase == kRetrieve && !retrieve)` — "waiting" means *armed for restore admission*.

Phases and every transition trigger:

- **P1 `kLookup` (entry).** Set in `OnAccepted` when `Align(prompt_len) > 0` (`lmcache.cc:188-189`). Rank 0 submits
  the LOOKUP RPC for tokens `[0, Align(prompt_len))` — deliberately querying the *whole* prefix including local hits,
  because local coverage cannot establish the remote STORE starting point (`lmcache.cc:193-200`). Other ranks enter
  `kLookup` with no lookup object and wait for the broadcast.
- **P2 `kLookup → kReady` (entry shortcut).** `Align(prompt_len) == 0` (prompt shorter than one chunk) — set at
  `OnAccepted` (`lmcache.cc:189`).
- **P3 `kLookup → kRetrieve` / `→ kReady`, in `PollLookups`.** Rank 0 polls each lookup; on success takes the
  `RequestLease` (`lookup.h:21-44`) and reports `{uid, matched}`; completions are broadcast to all ranks
  (`lmcache.cc:550-584`). All ranks then set `lmcache_matched_end = matched` (chunk-aligned: the client converts a
  chunk count to `chunks * chunk_size`, `lookup.cc:191-200`), `next_store_token = matched`, and phase =
  `RetrieveEnd(seq) > 0 ? kRetrieve : kReady` (`lmcache.cc:585-594`). A failed lookup logs a warning and recomputes
  locally (matched = 0 → `kReady`, stores start at 0) (`lmcache.cc:576-579`).
- **P4 `kRetrieve(waiting) → kReady`, in `PrepareSchedule` (skip restore).** (a) transfers unhealthy → `MakeReady`
  (`lmcache.cc:217-219`); (b) TP-reduced proof `local.resume_end >= RetrieveEnd` — local cache already covers the
  remote hit (`lmcache.cc:222-224,233-235`).
- **P5 `kRetrieve(waiting) → restore plan armed, still `kRetrieve(waiting)`, in `PrepareSchedule`.** Otherwise
  `scheduler_->PrepareRestore(seq, Align(preserve), RetrieveEnd(seq), preserve)` (`lmcache.cc:237-238`). The session
  stays waiting; the *scheduler* now holds a `Sequence::restore_plan`.
- **P6 `kRetrieve(waiting) → kRetrieve(in flight)`, in `OnScheduled`.** If `restore_plan->admitted` (scheduler's
  required admission succeeded this pass) and `PrepareRetrieve` succeeded on every rank (AllReduce MIN of
  {1 ok, −1 exception, 0 not admitted}), the lease delegates `[plan.start, plan.end)` and releases the rest, and the
  transfer is activated (`lmcache.cc:242-280`). Activation only stages; the RETRIEVE RPC is submitted on the first
  `Poll` after the construction-time CUDA ready event completes (`retrieve.cc:56-62,88-107,133-137`) — the engine's
  Update/Sync must first fence earlier users of the recycled destination storage (`lmcache.cc:277-279`).
- **P7 `kRetrieve(waiting), not admitted → stays waiting.** `session.retrieve.reset()` plus
  `CompleteRestore(seq, false)` — the plan is cleared and re-armed by the next `PrepareSchedule`
  (`lmcache.cc:284-287`). Retrieve admission is retryable across passes.
- **P8 `kRetrieve(waiting), preparation exception on any rank → kReady`** (`lmcache.cc:258-261,281-283`).
- **P9 `kRetrieve(in flight) → kReady`, in `PollTransfers`.** Terminal result on all ranks → `MakeReady(session,
  install)` where `install = success && !retiring && CanInstall` (`lmcache.cc:661-695`). Install-safety re-check:
  no native forward may already be writing an allocated-but-incomplete node in `[preserve_end, end)`
  (`CanInstall`, `lmcache.cc:454-466`).
- **P10 `kRetrieve → kReady` on retirement, in `PollTransfers`.** Waiting+retiring → `MakeReady` (no install);
  in-flight+retiring → `transfer.Cancel()` then normal completion handling (`lmcache.cc:661-671`; pre-submission
  cancellation is immediate, submitted writes must drain — `retrieve.h:36-38`).
- **P11 `kRetrieve(waiting) → kReady`, in `Fallback`** (OOM pressure relief, §7) (`lmcache.cc:298-306`).
- **P12 `kReady` is terminal** until the session is erased by `OnRetire` (`lmcache.cc:433-441`).
- **P13 `MakeReady` is the single funnel into `kReady`.** It drops the retrieve, closes any restore plan via
  `CompleteRestore(seq, install)`, and releases leftover lease locks (`lmcache.cc:468-478`).

Store-side state is *not* a phase: staged/submitted stores live outside the machine (`batches_[phase]`,
`stores_`), tracked per session only by the `pending_stores` counter (`lmcache.cc:71-95`).

## 3. Engine-loop hook call sites (`engine.cc`)

Loop order per iteration (`InternalThreadEntry`, `engine.cc:803-898`): pop/validate/broadcast → `Accept`
(→ `OnAccepted`) → `Cancel` → **`lmcache_.Poll()`** → notify → `Retire` → if active: `Schedule` (→
`PrepareSchedule`, `scheduler_.Schedule`, `OnScheduled`) → `FailStalledHeadOfLine` → `Setup` (→ `StageStores`) →
executor handoff → `Update` (→ `OnBatchComplete`) → `Retire`. Plus shutdown: `Join` → `Drain`; destructor →
`OnRetire` per sequence.

- **E1 `OnAccepted(seq)` — `engine.cc:433`, called inside `Accept` immediately after `scheduler_.AdmitPrompt`.**
  Creates the session unless ineligible (§6). Scheduling decision affected: whether the sequence can ever enter
  lookup/retrieve/store flows; ineligible sequences behave exactly as pre-PR.
- **E2 `PrepareSchedule()` — `engine.cc:450`, first statement of `Engine::Impl::Schedule`, before the eligible list
  is built.** Runs P4/P5 above: per waiting session, either skip (local suffices / unhealthy) or arm a restore plan.
  Scheduling decision affected: which sequences carry a `restore_plan` into this scheduler pass; the TP AllReduce
  MIN proof (`lmcache.cc:230`) makes skip/preserve identical on all ranks.
- **E3 `Schedulable(seq)` — `engine.cc:465`, the eligible-set filter (`!c.retiring && lmcache_.Schedulable(c)`).**
  True iff no session, `kReady`, or waiting-for-retrieve (`lmcache.cc:291-296`). `kLookup` and in-flight-retrieve
  sessions are **excluded from the scheduler entirely** — the coordinator hides them rather than expressing
  not-readiness inside the transaction. Also consulted in `FailStalledHeadOfLine` victim selection (`engine.cc:610`).
- **E4 `OnScheduled()` — `engine.cc:480`, immediately after `scheduler_.Schedule` returns, before active/inactive
  partitioning.** Runs P6–P8: prepares/activates retrieves for restore-admitted sessions, TP-reduces preparation
  status. Scheduling decision affected: whether the restore reservations made by the scheduler this pass are
  committed to a transfer or rolled back (`CompleteRestore(false)` frees the reservation for retry).
- **E5 `Fallback(seq)` — `engine.cc:615`, in `FailStalledHeadOfLine`.** Converts a waiting-for-retrieve victim to
  `kReady` (local recompute); returns true iff it acted (`lmcache.cc:298-306`). Scheduling decision affected: whether
  the OOM head-of-line is failed or given one more chance (§7).
- **E6 `StageStores(phase, seqs, count)` — `engine.cc:648`, in `Setup` over the active batch.** Stages STORE
  descriptors for newly produced token ranges (§5, §9). Scheduling decision affected: pins source cache blocks
  (`AcquireTransfer`) which are then excluded from eviction (`block.cc:60-73`), and advances
  `next_store_token` — but only on the shared TP-agreed ranges.
- **E7 `OnBatchComplete(phase)` — `engine.cc:785`, last statement of `Update`** (after `filled_len` is synced from
  batch output). Truncates each staged store's `end` to `min(end, Align(filled_len))` — a partial forward or early
  finish stores only what was actually produced — and marks the batch produced (`lmcache.cc:373-381`).
- **E8 `Poll()` — `engine.cc:842`, once per loop iteration after `Accept`/`Cancel`, before `Retire`.**
  `PollLookups` → `PollTransfers` → `SubmitStores` (`lmcache.cc:383-388`). This is the only place retrieves/stores
  advance and the only place stores are submitted (submission is deferred one loop turn past staging so Update/Sync
  fences recycled storage first).
- **E9 `HasPendingReleases()` — `engine.cc:593`, guard in `FailStalledHeadOfLine`.** True iff transfers healthy and
  any session has `pending_stores != 0` or an in-flight retrieve (`lmcache.cc:426-431`). Keeps the loop from
  declaring OOM while transfers hold pinned memory (see §7, §11 for the healthy-gate caveat).
- **E10 `CanRetire(seq)` — `engine.cc:347`, in `Retire`.** Requires `kReady && pending_stores == 0`
  (`lmcache.cc:419-424`); combined with the loop's own `retiring && inflight == 0` test. Scheduling decision
  affected: a finished sequence cannot release its cache until its stores complete (`invariants.retiring` +
  `checklist.delayed-release`).
- **E11 `OnRetire(seq)` — `engine.cc:351` (and `engine.cc:215` in `~Impl` for unwinding).** Erases the session,
  TM_CHECKing `pending_stores == 0 && !retrieve` (`lmcache.cc:433-441`). After this the sequence is interrupt/
  release as usual (`engine.cc:351-355`).
- **E12 `Drain()` — `engine.cc:123`, in `Join` after the engine loop thread has joined.** Local-only shutdown
  drain (§8).
- **E13 `LmCache::Ready(seq)` — `lmcache.h:50`, `lmcache.cc:413-417,791-794` — **no call sites in the tree**.
  A public readiness predicate that nothing consumes (engine uses `Schedulable` everywhere). Dead surface; the audit
  can drop it with that reason.

## 4. Scheduler-side probes (`scheduler.h:133-141`)

- **S1 `ProbeResume(seq)` — const, pure.** Calls `FindResume` (the same routine `PlanResume` uses) and returns
  `{resume_end = best candidate pos, prefix_end = readonly_blocks * block_size}` (`scheduler.cc:1472-1476`,
  `FindResume` at `scheduler.cc:524-614`). Reads: block validity/extent, frontier, checkpoints, partial siblings.
  Mutates nothing. Consumed only by `PrepareSchedule`'s proof. This is the coordinator peeking at scheduler state to
  decide skip/preserve — the rejected "probe" framing in the map's scope lock.
- **S2 `PrepareRestore(seq, start, end, preserve_end)` — `scheduler.cc:1478-1517`.** Preconditions
  `!is_active && inflight == 0`. Mutates scheduler/sequence state: `EnsureBlocks`, `ResetPlanBuffers` (clears
  alloc/restore/publish buffers **and** `involved_blocks`), then builds a `CacheRestorePlan` (`request.h:161-176`)
  with fresh prefix slots for every node whose offset lies in `[preserve_end, end)`, a fresh checkpoint slot per
  chunk boundary in `(start, end]` (lazily creating the owning `node->checkpoint`), and requires the frontier; every
  slot lands in `involved_blocks` and, when unallocated, `alloc_blocks`. So the restore plan replaces normal
  `PlanResume` output wholesale for that sequence this pass.
- **S3 restore-plan-aware planning.** `PlanRequests` skips `PlanResume`/`PlanContinue`/`PlanStoreCheckpoint` for any
  sequence carrying a `restore_plan` (`scheduler.cc:1130-1133`). `RunRequiredAdmission` admits restore-carrying
  sequences through the *same* required-admission/eviction machinery (`try_allocate_required`) but **without**
  `resource.Test`/`Commit` and without making them active: on success it just sets `restore_plan->admitted = true`
  (`scheduler.cc:1228-1233`). Restore reservations therefore compete with forwards for memory under eviction
  cutoffs, but consume no forward-token/context budget and never run a forward themselves.
- **S4 `CompleteRestore(seq, success)` — `scheduler.cc:1519-1548`.** Moves the plan out and resets plan buffers. On
  failure: nothing else (unused fresh slots are deallocated by `CacheRestorePlan::~CacheRestorePlan`,
  `request.cc:47-56`). On success: swaps `allocation/alloc_key` from each private plan slot into the target
  `node->prefix` / `node->checkpoint` slots (keeping slot identity for raw-pointer holders), marks restored prefix
  nodes `is_valid = true`, stamps installed blocks (not top eviction candidates), and sets
  `filled_len = plan->end`; the *next* normal `PlanResume` establishes the exact frontier via its ordinary restore
  copy (`scheduler.cc:1546-1548`). `Release` also clears `restore_plan` (`scheduler.cc:799`).
- **S5 `transfer_chunk_size` constructor arg — scheduler behavior changes for *everyone* when LMCache is enabled**
  (`scheduler.cc:294,300`), not just eligible sessions:
  - **S5a chunk-boundary forward splitting.** `ClampForwardEnd` clamps a prefill/replay pass's desired end to the
    next chunk boundary whenever checkpoints are registered, not warm-up, no input embeds, and the pass has not
    reached the context end (`scheduler.cc:1057-1064` with the precedence comment at `:1045-1056`). GDN prefill
    (including replay) must visit every chunk boundary.
  - **S5b chunk checkpoint arming.** `PlanStoreCheckpoint` arms `Sequence::store_checkpoint` = the node checkpoint
    at the next chunk boundary within the planned forward, and adds it to the protected involved set
    (`scheduler.cc:713-738`); it is excluded when input embeds are present (`:721`).
  - **S5c required allocation on exact landing.** If the admitted forward ends exactly on a chunk boundary and the
    armed snapshot is unallocated, it becomes a **required** allocation (`scheduler.cc:1265-1267`). This is the
    guarantee StageStores relies on ("New GDN boundary snapshots are guaranteed by required admission",
    `lmcache.cc:366-368`).
  - **S5d chunk-boundary publication bypass.** `PlanPublication` publishes the armed store checkpoint when
    `end % transfer_chunk_size == 0`, bypassing `checkpoint_min_interval` and the usual full-block routing
    (`scheduler.cc:1007-1015`); CommitResults then copies frontier→slot and stamps it (`scheduler.cc:1442-1457`).
  - All S5 paths are inert when the registry has no checkpoint category (plain transformer: `pools_.size() == 1`).

## 5. Granularity and alignment rules

- **G1 `Align(tokens) = tokens / chunk_size * chunk_size`** — floor to chunk (`lmcache.cc:545-548`). Used for:
  lookup query end (`OnAccepted`), store range end (`StageStores`), store truncation (`OnBatchComplete`), restore
  `start` (`PrepareSchedule`).
- **G2 `RetrieveEnd(seq) = min(lmcache_matched_end, Align(max(0, prompt_len - 1)))`** (`lmcache.cc:449-452`): the
  remote restore never covers the final prompt token (a forward must still run to produce output), and is
  chunk-aligned on both terms (matched is chunk-aligned per `lookup.cc:191-200`).
- **G3 preserve-end rule.** `preserve = min(local.prefix_end, pools_.size() > 1 ? end - chunk_size : end)`
  (`lmcache.cc:225`). With checkpoint pools the trailing chunk of the retrieve range is always re-fetched rather
  than preserved (rationale is not stated at the site; the effect is that the boundary checkpoint at `end` is
  refreshed together with its KV). `preserve_end` is block-aligned by construction; `start = Align(preserve)` may
  sit one chunk below it, and the retrieve request skips `preserve_end - start` tokens (`lmcache.cc:238,487`) so
  remote writes land exactly in the fresh slots `[preserve_end, end)`.
- **G4 store cursor.** `next_store_token` starts at the remote hit length (post-lookup) and advances to each
  staged range's aligned end (`lmcache.cc:588,369`); stores are only staged when the aligned produced end exceeds
  the cursor (`lmcache.cc:327-330`). Canceled/failed stores do not rewind it — the tokens are simply never stored
  (ranges below the cursor are skipped forever).
- **G5 store splitting on missing checkpoints.** With checkpoint pools, a chunk boundary whose owning node lacks a
  *valid* (live) checkpoint slot — validated per rank, AllReduce MIN — splits the store range: contiguous sub-ranges
  are staged, the unusable chunk is skipped (`lmcache.cc:338-366`).
- **G6 block-level pin mapping.** STORE pins prefix blocks at block granularity (pool group 0) and checkpoint
  sources at chunk granularity (groups 1..n), one shared source pin for all checkpoint parts
  (`lmcache.cc:519-541`); RETRIEVE pins the preserved node prefixes plus every fresh plan slot
  (`lmcache.cc:489-505`).

## 6. Eligibility exclusions and rank gating

- **L1 exclusions in `OnAccepted`** (`lmcache.cc:172-184`): `is_warm_up_` (held by reference — warm-up phase is
  excluded dynamically, `lmcache.h:23`), empty `token_ids`, `rope_base != 0` (dynamic NTK changes KV for the same
  tokens), `return_ppl`, `output_logits == kAll`, `output_last_hidden_state == kAll` (these need prompt forwards),
  non-empty `input_embeds`/`input_embeds_offsets` (legacy Python-embedding path), any multimodal span with an empty
  fingerprint. Excluded ⇒ no session ⇒ the sequence is invisible to the coordinator (but still subject to S5's
  chunk machinery). Note the near-duplicate exclusion logic in `Scheduler::PrefixEligible`
  (`scheduler.cc:262-269`) and `Validate` (`engine.cc:296-311`) — three parallel eligibility predicates that must
  stay in agreement.
- **L2 rank gating.** LOOKUP submitted on rank 0 only; other ranks enter `kLookup` empty and learn results by
  broadcast (`lmcache.cc:190-200,562-584`). The lease lives only on rank 0 and is delegated/released only there
  (`lmcache.cc:273-276,611-613`). All other decisions are made symmetrically then fixed up by collectives (§10).

## 7. Idle/OOM fallback path (`FailStalledHeadOfLine`, `engine.cc:589-626`)

- **F1 guard.** Do nothing if `s.active != 0` (work was admitted), `is_warm_up_`, or
  `lmcache_.HasPendingReleases()` — in-flight transfers keep the loop alive because their pinned memory will
  release (`engine.cc:593-595`, `checklist.forward-progress`).
- **F2 transient drain.** Any sequence with `inflight > 0` aborts the check — the in-flight batch will release
  memory at Update (`engine.cc:607-609`).
- **F3 victim = smallest `unique_id`** among non-retiring, `Schedulable`, zero-inflight sequences — the
  highest-priority head-of-line, "root of the OOM" (`engine.cc:610-613`).
- **F4 LMCache relief valve.** If the victim is waiting-for-retrieve, `Fallback` makes it `kReady` (abandon the
  restore, recompute locally) and the pass ends without failing anyone (`engine.cc:615-617`). Only a victim that is
  *not* recoverable (kLookup, retrieve in flight, or no session) proceeds to `kOutOfMemory` failure.
- **F5 caveat.** `HasPendingReleases` is gated on `transfers_healthy_` (`lmcache.cc:426-431`): in the
  unhealthy-but-draining state the OOM failure is no longer suppressed even though transfers still pin memory (the
  failed sequence then cannot retire until they drain, per E10 — memory is not leaked, but head-of-line failure can
  occur under a condition the healthy path would have waited out). Flagged for the audit.

## 8. Retirement gating and shutdown drain

- **R1 normal retirement.** Finish/cancel sets `retiring` (`engine.cc:739-749,358-372`); `Retire` releases a
  sequence only when `retiring && inflight == 0 && CanRetire` (`engine.cc:344-356`). Pending stores delay
  retirement; the loop's `Poll` keeps advancing them. Cancel interacts via `SubmitStores` skipping `is_canceled`
  stores (`lmcache.cc:608-610`) and `PollTransfers` cancelling retrieves of retiring sequences (P10).
- **R2 `Drain()` — after the loop stops, before executor destruction** (`engine.cc:119-125`): drop all unsubmitted
  staged descriptors (only pins are removed), **blocking**-drain every submitted store and in-flight retrieve
  (`StoreContext::Drain`/`RetrieveContext::Drain` poll until terminal), and close any remaining restore plans with
  `CompleteRestore(seq, false)` (`lmcache.cc:390-411`). Explicitly no TP collectives and no remote session END RPC
  — daemon-side cleanup is left to TTL. Sequences keep their allocations until the executor stops.
- **R3 destructor path.** `~Impl` joins, drains (R2), then for every remaining sequence `lmcache_.OnRetire` +
  `scheduler_.Release` (`engine.cc:212-220`) — the TM_CHECKs in `OnRetire` verify the drain really quiesced every
  session.

## 9. Cache categories that move

- **M1 prefix KV (group 0):** block granularity, both directions. Retrieve fills fresh plan slots then installs
  into `node->prefix` (`lmcache.cc:489-497`, `scheduler.cc:1536-1540`). Store sources are `node->prefix` blocks
  (`lmcache.cc:524-528`).
- **M2 checkpoint category (groups 1..PartCount):** chunk granularity, both directions, all parts of a checkpoint
  object move together sharing one allocation/source pin (`lmcache.cc:499-504,529-541`). Retrieve installs into
  `node->checkpoint` slots at boundaries `(start, end]` (`scheduler.cc:1541-1545`); store availability depends on
  those same slots being valid (G5). The frontier itself is required/allocated by `PrepareRestore`
  (`scheduler.cc:1502-1506`) but populated by the *next* PlanResume's normal restore copy, not by the transfer.
- **M3 nothing else moves:** generation-time mid-decode offload does not exist (map out-of-scope confirms);
  stores are prompt/production-driven via `StageStores` on active batches only.

## 10. TP symmetry mechanics (collectives inventory)

All ranks must execute these together, in the same loop shape:

- **T1** session-id broadcast at construction (`lmcache.cc:129`).
- **T2** lookup completion broadcast `{uid, matched}` from rank 0 (`lmcache.cc:584`).
- **T3** `PrepareSchedule` proof AllReduce MIN (2 ints per waiting session: skip flag, preserve end)
  (`lmcache.cc:230`).
- **T4** `OnScheduled` admission status AllReduce MIN (1 int per waiting session) (`lmcache.cc:268`).
- **T5** `StageStores` range broadcast from rank 0 with strict equality TM_CHECK, plus checkpoint-availability
  AllReduce MIN (`lmcache.cc:335-353`).
- **T6** `PollTransfers` status AllReduce MIN — vector `[connector healthy, per-store status, per-retrieve
  status]`; MIN semantics: everyone-finished, install requires everyone-installable (`lmcache.cc:638-695`).
  Every rank keeps identical `stores_`/session sets so the vector sizes line up (local submission failures keep
  their slot, `lmcache.cc:630-631`).
- **T7** the early-return condition in `PollTransfers` is itself rank-symmetric state (`transfers_healthy_`,
  `stores_`, batch-produced flags, kRetrieve sessions) (`lmcache.cc:640-645`) — divergence would deadlock the
  collective.

## 11. Failure and health model

- **H1 per-operation fallback.** Lookup failure → local recompute (P3). Retrieve preparation failure → local
  recompute (P8). Retrieve terminal failure or install-unsafe → local recompute of the affected range (P9 — the
  transfer's private slots are simply discarded). Store failure → tokens never stored (cursor already advanced).
  No request ever fails *because of* external memory, except via the F5 interaction.
- **H2 `uncertain` transfers.** A timed-out/errored transfer whose CUDA writes may still land keeps its source (or
  destination) buffers pinned and keeps returning no-result until terminal (`store.cc:145-189`,
  `retrieve.cc:170-220`, headers `store.h:20-21`, `retrieve.h:30-35`). Uncertain ⇒ zeroes the TP health entry.
- **H3 `transfers_healthy_` is recomputed each `PollTransfers`** as MIN over ranks of `connector_->healthy()`,
  zeroed by currently-outstanding uncertain transfers (`lmcache.cc:649-682`). The connector's own `healthy_` is
  heartbeat-monitored and flips both ways — false on ping failure, true again on recovery with registration replay
  (`MonitorServer`, `connector.cc:290, :297`; full statement in asset 02 F5) — so the engine-level flag means
  "connector healthy on every rank and no outstanding uncertain transfer": it can drop from server unhealthiness
  as well as uncertainty, and self-heals via heartbeat recovery or once uncertain transfers drain. *(Corrected
  2026-09-23: this item originally claimed `healthy_` is "only ever set true at registration", which is false.)* While false: waiting sessions are
  made ready (P4), staging is skipped (`lmcache.cc:310-311`), submission is skipped and staged descriptors dropped
  (`lmcache.cc:608-609`), and `HasPendingReleases` reports false (F5).
- **H4 transfer pins vs. allocator invariants.** Pinned blocks are excluded from `SortedEvictableBlocks`
  (`block.cc:60-73`) and `Deallocate`/`Invalidate` TM_CHECK `transfer_refs == 0` (`block.cc:29,37`) — in-flight external
  moves are structurally un-evictable, which is exactly the memory the OOM path cannot reclaim (E9/F1).

## 12. `Sequence` fields the integration touches (`request.h`)

- **V1 added/owned by the integration:** `lmcache_matched_end` (confirmed remote hit; STORE reservations never
  advance it, `request.h:252`), `store_checkpoint` (armed chunk snapshot in the required working set,
  `request.h:253`), `restore_plan` (`std::unique_ptr<CacheRestorePlan>`, `request.h:254`; plan type at
  `request.h:161-176`).
- **V2 read by the integration:** `token_ids`, `prompt_len`, `seq_len`, `history_len`, `input_len`,
  `inflight_input_len`, `filled_len`, `rope_base`, `gen_cfg.{return_ppl,output_logits,output_last_hidden_state}`,
  `input_embeds(_offsets)`, `multimodal_spans[].fingerprint`, `block_ids` (`->prefix`, `->checkpoint`, validity),
  `retiring`, `is_canceled`, `is_active`/`inflight` (via PrepareRestore precondition), `req->{unique_id,id}`.
- **V3 written indirectly through scheduler probes:** `involved_blocks`, `alloc_blocks`, `frontier`,
  `frontier_pos`(implicit), node `is_valid`/checkpoint slots, `filled_len` (S2/S4).

## 13. Surprising findings / audit flags

- **X1** `LmCache::Ready` is dead public API (E13).
- **X2** Enabling LMCache reshapes *global* scheduler behavior via `transfer_chunk_size` (S5) — forward splitting
  and checkpoint cadence change for sequences that never touch external memory (any non-warm-up, non-input-embeds request
  on a checkpoint-category model). A native design must own this coupling explicitly rather than as a constructor
  side effect.
- **X3** Eligibility is triplicated (L1: `OnAccepted`, `PrefixEligible`, `Validate`) with slightly different
  predicates — the audit should treat "one eligibility truth" as an item.
- **X4** Readiness is expressed by *hiding sequences from the scheduler* (E3) rather than by a scheduler-visible
  state; restore admission is retryable across passes via plan arm/clear cycles (P5–P7) — three engine-loop hooks
  (`PrepareSchedule`, `OnScheduled`, plus `Schedulable`) exist only to run this handshake.
- **X5** The retrieve handshake spans at least three scheduler passes: arm+admit (pass N), transfer+install (async),
  then a normal `PlanResume` over the restored state (pass ≥ N+1). Install can be refused at completion time
  (`CanInstall`, P9) because *other* requests' native forwards may be mid-write on shared nodes.
- **X6** `HasPendingReleases`' health gate (F5/H3) and the self-healing health flag are easy to lose in a redesign;
  both are load-bearing for `checklist.forward-progress` under external-memory pressure.
- **X7** Rank symmetry depends on ranked-shape agreement of loop-level state (T7) — every collective is entered
  from per-iteration hooks, so the native design must derive the same participation from transaction state or the
  loop deadlocks.
- **X8** Store correctness depends on a scheduler guarantee expressed only in a comment — chunk-boundary snapshots
  exist when needed because required admission allocates them on exact landings (S5c ↔ `lmcache.cc:366-368`). The
  spec should turn this into a stated contract leaf rather than an inter-file comment dependency.
- **X9** `next_store_token` never rewinds on failed/skipped stores (G4): skipped chunks are permanently absent
  remotely, and later stores start above them. Coverage-audit item: is a gap-tracking policy needed natively, or is
  "monotonic cursor, gaps allowed" the accepted behavior?
