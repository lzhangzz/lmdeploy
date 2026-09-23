# External-memory representation in the scheduler data model

Type: grilling
Status: resolved
Blocked by: 01, 02

## Question

How does the Scheduler's planning data model represent External Memory placement natively?

Today the transaction knows: `LogicalBlock` (token interval, refs, producer, prefix/checkpoint slots), `CacheBlock` (local object slot), `PrefixTrie` (identity index), and per-sequence `External Coverage` facts live outside the transaction in `LmCache::Impl` (extent-based, chunk-granular). PR 4983 patches the transaction's output via `ProbeResume`/`PrepareRestore`/`CompleteRestore`.

Candidate framings to decide among (or reject):

- External placement as a slot kind on `LogicalBlock` alongside prefix/checkpoint slots (a remote slot), making external placement structural.
- External Coverage as a per-sequence extent consulted by planning, with a chunk-granular external index parallel to the `PrefixTrie`.
- External memory as a third placement *stratum* in the existing pool/allocator vocabulary (`CacheBlockPool`, `ObjectAllocator`), with logical blocks gaining a placement field.

Decide: where external placement lives, how external identity relates to `contracts.prefix-identity` (token + fingerprint + parent chains), how chunk granularity composes with logical-block granularity, where the provider seam sits (the boundary against the service facts from ticket 02), and whether `External Coverage` is derived from structure or maintained as a fact. Express the answer as concrete code sketches against the real headers. Check against `principles.cache-semantics` (scheduler must not know byte meanings) and `ownership.scheduler`.

## Answer

Resolved 2026-09-21 through grilling (Q1–Q8, all confirmed with the recommended answers). External placement is **planning state, not structural state**.

**The decision.**

1. **Planning state, not structural state.** The structural model (`LogicalBlock`, `CacheBlock`, `PrefixTrie`) stays purely local; it gains no external concepts. External Coverage is a per-sequence planning fact the scheduler's planning consumes. Retrieved content becomes structural only at install — landing in ordinary slots, indistinguishable from locally produced content thereafter. The unified-index alternative (remote slots on `LogicalBlock`, trie nodes kept alive by external backing) is rejected on the service facts: external memory cannot be enumerated or introspected (gap G2), remote validity is opaque (daemon LRU, lease TTL — a local "externally valid" mark goes silently stale), and holding trie metadata for all stored content is unbounded growth duplicating an index the daemon already keeps.
2. **Coverage record on `Sequence`, scheduler-owned.** Replaces `lmcache_matched_end` (V1) and the coordinator's uid-keyed session map. Pure scheduling state; all service-side bookkeeping (lease, transfer pins, RPC contexts) stays behind the seam keyed by request, driven by intent lifecycle — no opaque lease token on the record.
3. **External Moves are first-class planned intents**, not `(src, dst)` cache-block copy plans with remote endpoints: a remote endpoint is not a `CacheBlock` (no allocation handle, timestamp, or evictability), and pretending otherwise lies to engine-thread address resolution (`principles.device-content`). Intents are planned by planning/publication stages and committed by `Schedule()`.
4. **Install erases provenance.** Terminal success (and install-safety) swaps allocations into `node->prefix`/`node->checkpoint` slots, marks `is_valid`, stamps, and the content is native — PR 4983's S4 install semantics survive. No provenance bit; the consumed extent feeding the store cursor is the only residue.
5. **Provider seam: intents down, terminal outcomes up, poll-driven** (the service is callback-free by construction, F14/F15). Leases, `BlockId` addressing, CUDA ready events, and the client-side destination fencing hazard (G4) stay below the seam. The exact seam operation list is settled by tickets 04/07; this fixes only its shape.
6. **External-memory geometry is a first-class scheduler concept** (`ExternalDesc` below), replacing the `transfer_chunk_size` constructor argument (C6). Fixes X2's invisible global coupling: whether clamping/cadence behavior attached to the descriptor applies to external-ineligible sequences remains ticket 05's.

**Derived facts recorded.** The service reports a single chunk-granular matched length (F19) — no per-category coverage at lookup — so the record holds one extent even though realization installs two categories; per-category availability is learnable only at transfer time. Coverage is not trie-indexed; comparison against local hits happens at planning (ticket 05).

**Code sketches** (design-level, naming provisional):

```cpp
// request.h — scheduler-owned external-coverage state on Sequence. Lifecycle edges are
// ticket 04's; this enumeration is the representation.
enum class CoverageState {
    kNone,    // ineligible, external memory absent, or nothing to look up
    kLookup,  // coverage query in flight, extent unknown
    kKnown,   // extent known, content not yet realized locally
    kDone     // extent consumed or abandoned; cursor stays meaningful for stores
};

struct ExternalCoverage {
    CoverageState state         = CoverageState::kNone;
    int           extent        = 0;  // chunk-aligned prefix length known remotely
    int           store_cursor  = 0;  // first token not yet staged for store (monotonic)
};

struct RetrieveIntent {
    int    start;         // chunk-aligned; may sit below preserve_end, service skips the overlap
    int    preserve_end;  // block-aligned local prefix kept as-is
    int    end;           // chunk-aligned retrieve end (RetrieveEnd rule — ticket 05)
    std::vector<CacheBlock*> targets;  // fresh destinations: one prefix slot per logical block in
                                       // [preserve_end, end) + one checkpoint slot per chunk boundary in
                                       // (start, end]; each slot's category fixes its install destination
    bool   admitted = false;  // set by required admission; Schedule()'s commit submits (2026-09-22 amendment)
};

struct StoreIntent {
    int    start;  // chunk-aligned, >= external.store_cursor
    int    end;    // chunk-aligned produced end
    std::vector<CacheBlock*> kv_sources;    // node prefix slots covering [start, end)
    std::vector<CacheBlock*> ckpt_sources;  // node checkpoint slots at chunk boundaries in (start, end]
    // (2026-09-22 amendment, matching spec §3: the lifecycle fields live here too)
    BatchDoneEvent ready{};           // the producing batch's completion token; the seam submits when it fires
    bool           in_flight = false; // the seam's transfer has started (ticket 04's lifecycle)
};
```

```cpp
// scheduler.h — first-class external-memory geometry; replaces the transfer_chunk_size
// constructor argument so the coupling is explicit and owned (X2).
struct ExternalDesc {
    bool enabled           = false;
    int  chunk_size        = 0;  // TM_CHECK(chunk_size % logical_block_size == 0) at wiring
    bool moves_checkpoints;      // registry has checkpoint category and pools registered
};
```

```cpp
// The provider seam — shape only; the operation list is fixed by tickets 04/07.
// Intents down, terminal outcomes up, poll-driven. Nothing address-like or
// lease-like crosses upward.
class ExternalMemoryService {  // the seam: intents down, terminal outcomes up, poll-driven
public:
    void Submit(const RetrieveIntent&, Sequence&);  // lease delegation is service-side
    void Submit(const StoreIntent&, Sequence&);
    void AbandonCoverage(Sequence&);                // releases lease; cancels if pre-submission
    // Poll() hands up terminal results only (success/failure per intent);
    // no partial progress exists to report (G2).
};
```

**What dies by this decision** (coverage-audit hooks): `lmcache_matched_end` and the coordinator session map (→ `ExternalCoverage`), the `transfer_chunk_size` ctor arg (→ `ExternalDesc`, later collapsed to a single granularity fact), `PrepareRestore`'s wholesale plan-replacement (S2 — intents replace it), the dead `LmCache::Ready` surface (E13), and the two rejected representation alternatives recorded above with reasons.

**Handed downstream:** lifecycle edges of `CoverageState` and in-flight intents → ticket 04; chunk/block clamping calculus and S5 behavior under `ExternalDesc` → ticket 05; admission of retrieve intents (required-but-async) → ticket 06; store-cursor semantics incl. X9 never-rewinds → ticket 07; identity (external coverage is token-keyed, `IPCCacheServerKey` carries token ids only — same-tokens/different-image collision hazard, `cache_salt` unset G7) → flagged into ticket 08; eligibility truth (X3 triplication) home → ticket 09.

**Amendment (from the resume decision, ticket 05):** `ExternalDesc` is dropped — `enabled` is redundant with nonzero granularity and `moves_checkpoints` derives from the scheduler's own `registry_`. The surviving fact is `external_chunk_size_` (0 = external memory absent) plus the seam reference from ticket 04. The gist — external-memory geometry becomes an explicit scheduler-owned concept replacing the constructor arg — stands; the struct does not.
