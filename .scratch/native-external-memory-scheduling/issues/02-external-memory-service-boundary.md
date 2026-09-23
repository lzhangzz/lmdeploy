# External-memory service boundary facts

Type: research
Status: resolved

## Question

What does the external-memory I/O service (`src/turbomind/lmcache/`: connector, lookup, store, retrieve, mq, event, protocol, transfer_geometry) actually expose and guarantee today — the fixed seam facts the native scheduling model may rely on?

Document as-is, without redesigning anything:

- The `Connector` API surface and lifecycle: configuration, registration model (`RegistrationConfig`, pools, storage offsets), session identity.
- Async semantics of lookup/retrieve/store: how completion is signaled (`event.h`, `mq.h`), what ordering guarantees exist, whether operations are cancelable and what cancellation means.
- Failure modes observable by the engine side: timeouts, partial retrieves, missed lookups, server absence — and what the engine is expected to do in each.
- Granularity facts: `chunk_size()` provenance, per-pool granularity (block-granular KV pool vs chunk-granular checkpoint pools), any alignment requirements.
- TP coordination: which calls are rank-local, which are collective, what keeps ranks consistent (e.g. rank-0 lookup with broadcast results — find the actual mechanism).
- What the service needs from the engine (memory registration, stream/event integration, threading expectations) versus what it decides itself.

The output defines the provider side of the seam: the native model consumes these facts; it must not assume anything not established here. Codebase-internal only; no web or prior-art research. Deliver findings as an asset under `.scratch/native-external-memory-scheduling/assets/`, link it from the resolution, and resolve this ticket per the tracker's wayfinding operations.

## Answer

Findings: [assets/02-service-boundary.md](../assets/02-service-boundary.md) — 51 itemized seam facts (F1–F51) and 10 explicit gaps (G1–G10), each traceable to file and symbol in `src/turbomind/lmcache/` and `src/turbomind/engine/lmcache.cc`.

Summary of the seam as it exists: the service is a per-rank `Connector` (ZMQ DEALER + one MQ I/O thread + one heartbeat thread) over the LMCache 0.5.5 protocol, registering one CUDA-IPC-exported allocation (base/size/storage_offset plus per-pool `{part_bytes, tokens_per_block}`) and addressing transfers by flat `BlockId`s on the registered grid. Completion of every operation is **pull-based and non-blocking**: engine-thread `Poll()` chains an RPC reply with an unpacked `[event_handle, success]` and a daemon-side IPC CUDA event polled to `cudaSuccess`; an empty event handle is the daemon's guarantee that no device work ran. Lookup is rank-0-only and reaches other ranks via an engine-side broadcast of `{uid, matched}`; all other rank agreement (admission, store staging, transfer outcomes, health) is engine-side AllReduce-MIN — the service itself provides no cross-rank mechanism beyond the shared `session_id` and daemon-side `world_size` accounting. Cancellation exists only pre-submission for RETRIEVE; STORE and submitted RETRIEVE must drain to a terminal result (destructors `TM_CHECK` this), and uncertain outcomes (timeout, server loss) pin the involved buffers until eventual CUDA completion with no abort API. Failure handling shown by the engine is uniformly "fall back to local compute and keep the session alive" (miss, failed lookup, retrieve failure, preparation failure on any rank), with `transfers_healthy_` as a TP-wide gate that one rank's uncertainty zeroes. Granularity: `chunk_size` is server-defined and immutable; the prefix pool is block-granular, checkpoint pools chunk-granular; matched lengths are chunk multiples and `RetrieveEnd` always recomputes the last prompt token. The gaps section (no daemon-visible cancellation, no partial-progress signal, unbounded uncertainty, client-side-only ready gating for retrieve, chunk-granular lookup precision, placeholder LayoutHints, unset `cache_salt`, single global timeout) lists everything the native scheduling model must **not** assume the service provides.
