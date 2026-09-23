# Asset 02 — External-memory service boundary facts

Ticket: `issues/02-external-memory-service-boundary.md` (research)
Date: 2026-09-21
Method: codebase-internal reading only. Sources: `src/turbomind/lmcache/{connector,lookup,store,retrieve,mq,event,protocol,transfer_geometry}.{h,cc}` and the consumer `src/turbomind/engine/lmcache.{h,cc}` with call sites in `src/turbomind/engine/engine.cc`. All line references are to these files. No web or prior-art sources were consulted.

Purpose: the provider side of the seam. The native scheduling model may rely on the facts below; anything not established here (see "Gaps") must not be assumed.

---

## 1. Connector: configuration, identity, lifecycle

- **F1 — Configuration surface.** `ConnectorConfig` (`connector.h:17-24`): `server_addr`, `model_name`, `session_id` ("shared by the engine's TP ranks"), `world_size`, `worker_id` ("LOOKUP/QUERY/FREE/END keys still use worker_id=None"), `request_timeout_ms` default `300'000`.
- **F2 — Construction is connecting and validating.** `Impl::Impl` (`connector.cc:64-73`) creates the `MessageQueueClient`, performs a **blocking** `kGetChunkSize` RPC (timeout `kHeartbeatTimeoutMs` = 10 s, `connector.cc:33-35`, `FetchChunkSize` at `connector.cc:227-235`; non-positive or >int-max chunk size throws), and starts a heartbeat thread. A server absent at engine start makes the `Connector` constructor throw; there is no retry (`LmCache::Create` propagates, `engine/lmcache.cc:712-723`).
- **F3 — Instance identity.** `instance_id()` is a random `int64` per Connector (`MakeInstanceId`, `connector.cc:53-58`). It appears in registration, unregister, and STORE/RETRIEVE payloads (`store.cc:102`, `retrieve.cc:101`) so the daemon can pair transfers with the registered IPC memory.
- **F4 — Session identity.** `MakeScopedRequestId` = `(session_id.empty() ? instance_id : session_id) + ":" + request_id` (`connector.cc:106-109`). The engine generates `session_id` on rank 0 and broadcasts it (`engine/lmcache.cc:125-129`), so all ranks scope request ids identically; without a session id, per-rank random instance ids would scope differently.
- **F5 — Health monitoring and auto re-registration.** `HeartbeatLoop` (`connector.cc:303-314`) runs `MonitorServer` every `kHeartbeatIntervalMs` = 10 s: `kPing` (10 s timeout) carrying `instance_id` when registered. Failure flips `healthy_` to false with a warning; recovery flips it back **and replays the saved `kRegisterKvCache` payloads** (`registration_payloads_` retained at `connector.cc:194-198`, replay at `connector.cc:284-286`). In-flight transfers at server-crash time are not retried by the service.
- **F6 — Teardown.** `~Impl` (`connector.cc:75-84`): stop heartbeat; if registered, send `kUnregisterKvCache` with `instance_id` and wait up to `kShutdownRequestTimeout` = 1 s for the reply "until the unregister response confirms that the daemon dropped IPC" (`UnregisterForShutdown`, `connector.cc:237-261`), then `MessageQueueClient::Stop()`. Failures are logged, not thrown.
- **F7 — Per-rank model.** One Connector per TP rank ("Per-rank MP DEALER", `connector.h:48`). Scheduler-class RPCs (LOOKUP, QUERY, FREE, END) key on `worker_id=None`; STORE/RETRIEVE key on `worker_id=rank` (`lookup.cc:27`, `store.cc:25`, `retrieve.cc:25`).

## 2. Registration model

- **F8 — RegistrationConfig.** (`connector.h:31-36`) `base` (allocation base pointer), `size` (region bytes), `storage_offset_bytes` (allocator page base − allocation base), `pools`. The engine derives them from the `ObjectAllocator` and checks the storage lies within the region (`engine/lmcache.cc:143-169`).
- **F9 — Pool granularity is per-pool, engine-chosen.** The engine registers the **prefix pool as block-granular** (`tokens_per_block = block_size_`, one part) and, when the registry has checkpoints, **each checkpoint part as chunk-granular** (`tokens_per_block = chunk_size_`) (`engine/lmcache.cc:145-153`).
- **F10 — Divisibility contract.** Service side: `chunk_size_ % pool.tokens_per_block == 0` per pool (`connector.cc:150`). Engine side additionally: `chunk_size_ % block_size_ == 0` ("LMCache chunk_size must be divisible by the logical block size", `engine/lmcache.cc:133`) and `page_size % part_bytes == 0` per pool (`engine/lmcache.cc:155-158`).
- **F11 — Memory export and transfer views.** `Register` exports the allocation with `cudaIpcGetMemHandle(base)` (`connector.cc:134-135`): the daemon **reads and writes engine device memory directly over CUDA IPC**. Per-pool "transport views" come from `SelectTransferGeometry` (`transfer_geometry.h:20-37`): strided slice view when `part_bytes >= 256 KiB` and `% 4096 == 0` (8 KiB tiles when `blocks_per_chunk > 1` and `part_bytes*blocks_per_chunk >= 24 MiB` and `% 8192 == 0`, else 4 KiB), else a heads view (double heads while `part_bytes % (64*heads) == 0` and `part_bytes/(4*heads) >= 2048`, capped at 32 heads / max_threads). Hard floors: `part_bytes % 4 == 0`, `max_threads >= 32`, else throw. The comment fixes intent: "This changes only the transport view, never the cache allocation or model head count."
- **F12 — Wire payload.** `kRegisterKvCache` carries instance_id, IPC wrappers (shape/stride/storage_offset/device_uuid via msgpack ext, `protocol.h:91-135`), model_name, world_size, the literal string `"vllm"` (`connector.cc:189`), a **default-constructed `LayoutHints`** (so `kv_layout="HND", num_kv_heads=1, tokens_per_block=1, head_dim=1` placeholders, `protocol.h:73-80`), and per-pool `EngineGroupInfo{pool_index, layer_indices, tokens_per_block, sw_size_tokens=-1}` (`connector.cc:159`).
- **F13 — Flat block addressing.** `BlockId(pool_index, part_address)` = `(addr − pool_base) / part_bytes` with `pool_base = base + storage_offset_bytes` (`connector.cc:111-120`). An address off the registered grid is a **fatal** `TM_CHECK` ("LMCache source address is not on the registered block grid"). These ids are the addressing shared with the daemon in STORE/RETRIEVE `block_ids`.

## 3. Transport (message queue) semantics

- **F14 — One I/O thread, FIFO submit, unordered replies.** `MessageQueueClient` (`mq.h:41-61`) owns a dedicated thread with a ZMQ DEALER socket (linger 0, immediate 1, sndtimeo 1 s; `mq.cc:177-185`). Submits queue under a mutex and are sent in submission order; replies are correlated by UID and may arrive in any order (`mq.cc:121-141`, `ProcessInbound` `mq.cc:241-272`). `ResponseMode::kTrack` returns a future-backed handle; `kIgnore` is fire-and-forget (used only by `Connector::Notify` for FREE/END, which swallows exceptions, `connector.h:78-86`).
- **F15 — Poll-friendly completion check; abandonment is detach-only.** `RequestHandle::Ready()` reads an atomic `resolved` flag — deliberately not `std::future::wait_for(0)` because that "can block the engine thread" (`mq.cc:80-84`). `WaitUntil` blocks; `Get()` rethrows transport errors. Destroying a handle resolves its promise with an error ("LMCache request handle was abandoned", `mq.cc:96-101`); the I/O thread then skips sending it and prunes the pending entry (`mq.cc:197-199`, `274-284`). **Nothing is sent to the daemon** — abandonment is invisible server-side.
- **F16 — Transport failure propagation.** A send failure resolves the failing task and every still-queued task with errors ("LMCache transport unavailable before send", `mq.cc:218-239`); a poll-loop failure `RejectAll`s everything pending (`mq.cc:294-336`); UID/type mismatch resolves with an error (`mq.cc:259-262`).
- **F17 — Protocol facts.** msgpack frames; `RequestType` values pinned to "LMCache 0.5.5's RequestType enum" (`protocol.h:40-54`). `TransferResponse` is strictly `[event_handle(bytes), success(bool)]` (`protocol.h:101-104`, validation `156-176`). `IPCCacheServerKey` = model_name, world_size, optional worker_id, token_ids, start, end, request_id, cache_salt, `num_kv_readers` default 1 ("One reader per TP shard's object", `protocol.h:58-71`).

## 4. Operation contexts and async semantics

### Lookup (`lookup.h/.cc`)

- **F18 — Two-stage, rank-0-only.** `LookupContext` submits `kLookup` (worker_id=None, plus world_size arg) then, after the reply, polls `kQueryPrefetchStatus` with a 5 ms backoff until it yields `Some(chunks)` (`lookup.cc:20`, `131-132`, `152-157`, `186-195`).
- **F19 — Terminal outcomes and granularity.** `Poll()` is non-blocking and sticky-returns `optional<LookupResult>`. Success: `matched_tokens = chunks * chunk_size` — **matched length is always a chunk multiple** (`lookup.cc:196-200`). Failure: unhealthy at construction or during poll (`lookup.cc:126-129`, `165-168`), deadline = construction + `request_timeout_ms` (`lookup.cc:124`, `169-171`), invalid chunk count (throws → caught → failure, `lookup.cc:197-204`). A failed lookup is a miss, never retried.
- **F20 — Lease handoff.** On success the context holds a `RequestLease`; `TakeLease()` moves it out once (`lookup.cc:139-150`, `208-214`). "Unknown outcomes acquire no lease and rely on the daemon's existing TTL" (`lookup.cc:148`).
- **F21 — Lease semantics.** The lease owns the matched read locks and the remote request session, independent of the RPC (`lookup.h:19-20`). `ReleaseLocks()` sends `kFreeLookupLocks` (with world_size); the destructor sends `kEndSession` (`lookup.cc:44-64`). `DelegateLocks(start, end)` marks the range workers took over; subsequent `ReleaseLocks()` frees only the unused prefix and suffix (`lookup.cc:50-64`). Both notifications are fire-and-forget (`kIgnore`); a failed notification "leaves cleanup to the daemon's existing TTL" (`connector.h:76-77`).

### Store (`store.h/.cc`)

- **F22 — Ready event, shareable per batch.** Constructed with a `shared_ptr<CudaEvent>` recorded on an engine stream; "All STOREs produced by a batch can share this event" (`RecordReady`, `store.cc:67-72`; engine usage `engine/lmcache.cc:603-616`). `CudaEvent` is interprocess + disable-timing (`event.h:21`).
- **F23 — Immediate submission.** If healthy, the `kStore` RPC (key with worker_id, instance_id, block_ids, ready-event IPC handle) is submitted as the final construction step (`store.cc:92-104`). Unhealthy → terminal `StoreResult{false, "LMCache server is unhealthy; Store skipped"}`.
- **F24 — Completion = RPC + daemon CUDA event.** After the reply, an **empty `event_handle` is the daemon's guarantee that no device work ran** → terminal; otherwise the client opens the daemon's done event (`cudaIpcOpenEventHandle`) and polls `cudaEventQuery` to `cudaSuccess` (`store.cc:121-155`; same for retrieve). Header contract: "A terminal result means the daemon can no longer read the source buffers" (`store.h:20-21`).
- **F25 — No cancellation; destruction requires a terminal result.** `~State` `TM_CHECK`s `result` — "submitted STORE must be drained before destruction" (`store.cc:48-51`). `Drain()` blocks in a 1 ms sleep loop (`store.cc:174-179`).
- **F26 — Uncertainty.** `uncertain()` is set on timeout / unhealthy / post-submission exceptions: "retaining source buffers until CUDA completion" (`store.cc:160-172`, `181-189`). The engine must keep the pinned source blocks alive until the terminal result eventually arrives.
- **F27 — Engine clamps staged ends.** At batch completion the engine rewrites `request.end = min(end, Align(filled_len))` — staged store ranges were optimistic about forward progress (`engine/lmcache.cc:373-381`).

### Retrieve (`retrieve.h/.cc`)

- **F28 — Three stages; preparation is free.** `kPrepared` (event recorded on the engine stream, **no locks, no RPC** — free to destroy), `Activate()` → `kWaitingReady` (deadline starts here), `TrySubmit` → `kSubmitted` (`retrieve.h:31-33`, `retrieve.cc:46-68`, `133-137`). Destruction of an activated-but-undrained context is a fatal `TM_CHECK` (`retrieve.cc:65-68`).
- **F29 — Client-side ready gating (fixed hazard).** Submission waits for the engine-stream ready event via `cudaEventQuery` because "The MP Retrieve handler does not wait on the ready handle, so destinations must not be exposed until that event completes" (`retrieve.h:18-20`, `retrieve.cc:85-108`). The engine correspondingly defers `Activate()` until "the next Poll, after the engine's Update/Sync has fenced earlier users of recycled storage" (`engine/lmcache.cc:276-279`).
- **F30 — Payload and completion.** `kRetrieve` carries key, instance_id, block_ids, ready handle, and `skip_first_n_tokens` (engine sets it to `plan.preserve_end − plan.start`, so the daemon does not overwrite the preserved prefix; `retrieve.cc:100-105`, `engine/lmcache.cc:480-488`). Completion chain mirrors STORE (F24; `retrieve.cc:149-174`).
- **F31 — Cancellation is pre-submission only.** `Cancel()` is effective only when `stage != kSubmitted && !result` → immediate terminal failure; after submission nothing can cancel — `Drain()` = Cancel + blocking poll (`retrieve.cc:197-210`).
- **F32 — Failure character changes at submission.** Pre-submission unavailability fails cleanly: "unavailable before submission; recomputing locally", and (stage `kWaitingReady`) locks are freed via `kFreeLookupLocks` (`retrieve.cc:70-90`). Post-submission exceptions/timeouts go to `MarkUncertain` — destinations retained until CUDA completion (`retrieve.cc:176-189`, `212-220`). "After submission the daemon owns these locks; unconsumed groups from a failed transfer expire by TTL" (`retrieve.cc:78-80`).

## 5. Failure modes observable by the engine, and the engine's shown reaction

- **F33 — Server absent at startup.** Connector constructor throws → engine construction fails (`engine/lmcache.cc:712-723`). No retry path.
- **F34 — Server unhealthy mid-run.** `connector->healthy()` goes false (heartbeat, F5). Engine reactions in code:
  - Lookups fail fast ("Lookup skipped") → `PollLookups` logs "recomputing locally" with `matched=0` (`engine/lmcache.cc:577-579`).
  - New stores are not staged/submitted: `StageStores` and `SubmitStores` no-op under `transfers_healthy_ == false` (`engine/lmcache.cc:310`, `608-610`).
  - Waiting retrieves fall back to local compute: `MakeReady` → `CompleteRestore(c, false)` (`engine/lmcache.cc:217-219`, `468-478`).
  - `transfers_healthy_` is recomputed every `PollTransfers` as the TP-wide AllReduce-MIN of entry 0 = `connector->healthy()` AND no uncertain store/retrieve on any rank (`engine/lmcache.cc:649-682`): **one rank's uncertainty suspends new transfers on all ranks** until it clears.
- **F35 — Miss / failed lookup.** `lmcache_matched_end = 0` → phase `kReady` (or zero-length retrieve) → local recompute; the session continues (`engine/lmcache.cc:585-594`).
- **F36 — Retrieve failure (post-submission, non-success).** Status 1 → `MakeReady(install=false)` → `CompleteRestore(sequence, false)` — no retrieve retry (`engine/lmcache.cc:673-695`).
- **F37 — Install gating.** Success installs only if `!retiring && CanInstall(...)`: "A native forward may already be writing an allocated, incomplete node. Never replace its allocation, even though our private copy is complete." (`engine/lmcache.cc:454-466`). The install decision is TP-wide: status 2 (installable) is MIN-reduced, so any rank degrading it yields plain-ready (`engine/lmcache.cc:674-695`).
- **F38 — Retrieve preparation failure on any rank.** Local exception → status −1 → MIN < 1 → every rank falls back (`engine/lmcache.cc:252-288`). Non-admitted (`status 0`) prepared contexts are discarded — "prepared contexts own no remote locks" (`engine/lmcache.cc:285-287`).
- **F39 — Retiring sequences.** `PollTransfers` cancels retrieves of retiring sequences (effective pre-submission; post-submission they still poll to terminal) and releases waiting retrieves (`engine/lmcache.cc:661-671`). `CanRetire` requires `phase == kReady && pending_stores == 0` (`engine/lmcache.cc:419-424`); the engine's `Retire` honors it (`engine.cc:344-352`).
- **F40 — Shutdown drain.** `Engine::Join` → `lmcache_.Drain()` after the internal thread joined (`engine.cc:119-125`): discard unsubmitted descriptors (only pins removed), drain submitted stores and retrieves to terminal, `CompleteRestore(false)` for planned sessions. "No TP collective or session END runs here." (`engine/lmcache.cc:390-411`).
- **F51 — Failed stores do not roll back the store cursor.** `next_store_token = r.end` is set unconditionally when staging (`engine/lmcache.cc:355-370`); a failed or skipped store's range is simply never stored, not retried. Store failure is a warning only (`store.cc:111-119`).

## 6. Granularity facts

- **F41 — chunk_size provenance.** The daemon's, fetched once at Connector construction via `kGetChunkSize` (F2); constant for the process lifetime; server-defined, not negotiable.
- **F42 — Per-pool granularity.** Prefix pool: logical-block-granular. Checkpoint pools: chunk-granular (F9). Registered addressing requires part-grid alignment (F13).
- **F43 — Engine-side chunk alignment.** `Align(tokens)` floors to chunk multiples (`engine/lmcache.cc:545-548`); matched length is a chunk multiple (F19); `RetrieveEnd = min(lmcache_matched_end, Align(prompt_len − 1))` — **the last prompt token is always recomputed locally** (`engine/lmcache.cc:449-452`).
- **F44 — Transfer-alignment floors.** See F11's numeric conditions (`transfer_geometry.h:22-36`).

## 7. TP coordination

- **F45 — Rank-local calls.** Connector construction/registration, `BlockId`, STORE and RETRIEVE submission/completion, heartbeats, unregister. The service provides **no** cross-rank mechanism beyond the shared session id (F4) and daemon-side world_size accounting.
- **F46 — Rank-0-only lookup with broadcast of results.** Only rank 0 owns `LookupContext`s (`engine/lmcache.cc:190-200`). `PollLookups` polls them on rank 0, then distributes the completion list `vector<{uid, matched}>` via `comm::Broadcast(tp_group_, done, 0)`; every rank then sets `lmcache_matched_end` and `next_store_token` identically (`engine/lmcache.cc:550-595`). This broadcast **is** the mechanism that keeps ranks consistent for lookup.
- **F47 — Collective decisions are all engine-side** (`engine/lmcache.cc`):
  - `PrepareSchedule` proof vector `[resume_end >= end, min(prefix_end, checkpoints ? end − chunk_size : end)]` → AllReduce MIN → all-ranks-local-resume else `PrepareRestore` with the MIN preserve (`engine/lmcache.cc:208-240`).
  - `OnScheduled` admission status (1 prepared / 0 not admitted / −1 failed) → MIN → lease delegation + `Activate()` only when every rank prepared (`engine/lmcache.cc:242-289`).
  - `StageStores`: rank-0 range list broadcast with a fatal `TM_CHECK` on divergence ("STORE batch ranges differ across TP ranks"); checkpoint chunk availability MIN — a chunk is storable only if its state is valid on **all** ranks (`engine/lmcache.cc:335-353`).
  - `PollTransfers` status MIN: health gate, per-store completion (erased only when all ranks finished), per-retrieve 0/1/2 outcome (`engine/lmcache.cc:638-695`).
  - `session_id` broadcast at construction (F4).
- **F48 — Daemon-side TP accounting.** Keys carry `world_size`; `kFreeLookupLocks` is notified with `world_size`; `num_kv_readers` defaults to 1 per shard object (F7, F17, `lookup.cc:54`, `62`).

## 8. What the service needs from the engine vs decides itself

- **F49 — Engine obligations (service needs):**
  - One IPC-exportable allocation plus pool layout, before any transfer use (F8-F13).
  - Engine stream + CUDA event integration: ready events for store (shareable per batch) and retrieve (per-context); engine-side fencing of recycled storage before retrieve activation (F22, F28-F29).
  - A single-threaded, explicitly-polled driving loop: `lmcache_.Poll()` is called once per internal-loop iteration (`engine.cc:842`, inside `InternalThreadEntry` `engine.cc:788`); the service never calls back into the engine. It adds exactly two threads of its own (MQ I/O, heartbeat) (F5, F14).
  - Block pinning for the transfer duration (`AcquireTransfer`/`ReleaseTransfer`, `engine/lmcache.cc:28-47`, `71-95`) and buffer retention until terminal results (destructors CHECK; F25, F28).
  - All TP collectives and cross-rank agreement (F46-F47).
  - Eligibility gating of what enters the service at all — `OnAccepted` excludes warm-up, `rope_base != 0`, PPL, all-logits, last-hidden-state, input embeds/offsets, and multimodal spans without fingerprints (`engine/lmcache.cc:172-184`).
- **F50 — Service-internal decisions (opaque to the engine):** chunk_size (server), transfer geometry/views, wire format, MQ threading, heartbeat + automatic re-registration, request scoping/ids, lease/lock semantics with TTL fallback, per-context timeout behavior, CUDA event lifecycle details.

## 9. Gaps — not established by this seam (the scheduler must not assume them)

- **G1 — No daemon-visible cancellation.** Abandoned handles and unknown outcomes rely on the daemon's TTL (`connector.h:76-77`, `lookup.cc:148`, F15, F21). Submitted stores and activated retrieves cannot be cancelled at all (F25, F31).
- **G2 — No partial-progress or bandwidth feedback.** Completion is all-or-nothing per context; the only outcome signals are the success bool and the done event (F24, F30). No progress fraction, no queue depth.
- **G3 — Unbounded uncertainty.** `uncertain()` transfers must be retained until the daemon's done event fires or the context otherwise finishes; there is no forced-abort API and no upper bound besides eventual completion (F26, F32).
- **G4 — Ready-handle gating is client-side only.** The protocol carries the engine ready event, but the daemon's MP retrieve handler ignores it; destination exposure control is entirely the engine's responsibility (F29). Any redesign must preserve this fencing.
- **G5 — Lookup precision is chunk-granular.** Sub-chunk coverage cannot be expressed by LOOKUP (F19); `RetrieveEnd`'s `prompt_len − 1` clamp is the engine's compensation for forward-end safety (F43).
- **G6 — LayoutHints are placeholders.** Registration sends default-constructed hints; the layout truth travels in the wrappers' shape/stride and `EngineGroupInfo.tokens_per_block` (F12).
- **G7 — `cache_salt` is never set by the engine.** `Request::cache_salt` defaults to empty and no engine call site populates it (`connector.h:45`; no assignment in `engine/lmcache.cc`).
- **G8 — `LmCache::Ready()` has no in-repo consumer.** Exported at `engine/lmcache.h:50` (wrapper `engine/lmcache.cc:791-794`) but no call site exists; the consumed predicates are `Schedulable`, `CanRetire`, `HasPendingReleases`, `Fallback` (`engine.cc:347`, `465`, `593`, `610-615`, `842`).
- **G9 — One global timeout knob.** `request_timeout_ms` (default 300 s) applies uniformly to lookup, store, and each retrieve context; there is no per-operation deadline override in the API (F1).
- **G10 — Best-effort cleanup notifications.** FREE/END are fire-and-forget (`kIgnore`); their loss is covered only by daemon TTL (F21).

## 10. Threading summary (fixed facts)

- Engine side: all service calls happen on the engine's single internal thread (`engine.cc:788`, `842`, and the schedule/batch call sites F49); `Poll`/`Ready` are non-blocking; `WaitUntil`/`Call`/`Drain` block.
- Service side: two owned threads — the MQ I/O thread (F14) and the heartbeat thread (F5). `RequestHandle::Ready()` is explicitly engineered to never block the engine thread (F15).
