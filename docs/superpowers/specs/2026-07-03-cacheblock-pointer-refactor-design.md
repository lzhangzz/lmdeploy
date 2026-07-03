# CacheBlock pointer refactor design

Date: 2026-07-03
Status: approved

## Goal

Refer to `CacheBlock` by real pointers (`CacheBlock*`) instead of `int` indices
into `CacheBlockPool`, everywhere: `LogicalBlock` slots, `Sequence` fields,
copy plans, admission/replay records, engine-thread setup resolution, and
modules. No `int` cache id remains anywhere.

Motivation: type-safety and clarity. Eliminate the index indirection, the
`cache_[id]` lookups, the index-0 sentinel, and the bounds checks that only
approximate what the type system can carry directly.

This is a pure refactor: behavior, ownership rules, admission order, eviction
order, and the memory replay commit point are unchanged.

## Storage: deque-backed pool

`CacheBlockPool::blocks_` changes from `std::vector<CacheBlock>` to
`std::deque<CacheBlock>` so node addresses are stable across growth. Freed
slots are recycled through a pointer free list. This matches today's memory
behavior exactly (high-water mark, slot recycling) and today's ABA story
(a recycled slot's identity can be reused, guarded by the ownership rule that
nothing holds a cache reference past its owner's destruction, plus the
`alloc_key` snapshot for allocation staleness).

The pmr-node alternative (mirroring `LogicalBlockPool`) was rejected: its
live-node registry needs removal bookkeeping on `Invalidate`, while the deque
itself already enumerates every slot for the eviction sweep — the same
scan-and-filter `SortedIndices()` does today.

## New pool and block API

```cpp
struct CacheBlock {
    uint64_t       timestamp{};    // eviction priority; zero means highest
    int            object_id{-1};  // ObjectAllocator registration id
    object_alloc_t allocation{};   // {const Allocation*}; .a == nullptr => no live allocation
    uint64_t       alloc_key{};    // snapshot of allocation->key at replay (ABA stale check)
    LogicalBlock*  owner{};        // weak identity back-reference; nullptr = sequence-owned

    char* base(int part) const { return allocation->base(part); }
    int   part_count() const { return allocation->part_count(); }
    bool  valid() const noexcept { return allocation.a != nullptr; }

    // Moved off the pool: they touch only this block's state.
    void Deallocate(ObjectAllocator& alloc);  // was CacheBlockPool::Deallocate
    void Demote() noexcept                    // was CacheBlockPool::Demote
    {
        TM_CHECK(valid());
        timestamp = 0;
    }
};

// nullptr replaces the index-0 sentinel ("no slot").
inline bool is_valid(const CacheBlock* b)
{
    return b && b->valid();
}

class CacheBlockPool {
public:
    CacheBlock* Create(int object_id, LogicalBlock* owner = nullptr)
    {
        TM_CHECK_GE(object_id, 0);
        CacheBlock* b;
        if (free_.empty()) {
            b = &blocks_.emplace_back();
        }
        else {
            b = free_.back();
            free_.pop_back();
        }
        *b           = {};
        b->object_id = object_id;
        b->owner     = owner;
        return b;
    }

    void Invalidate(CacheBlock* b)  // owner destroyed; return the slot for reuse
    {
        TM_CHECK_GE(b->object_id, 0);  // double-invalidate check
        *b = {};
        free_.push_back(b);
    }

    // Eviction candidates: exactly the currently allocated blocks. Scans the
    // whole deque (freed slots fail valid()), same cost shape as today.
    std::vector<CacheBlock*> SortedBlocks()
    {
        std::vector<CacheBlock*> v;
        for (auto& b : blocks_) {
            if (b.valid()) {
                v.push_back(&b);
            }
        }
        std::sort(v.begin(), v.end(), [](auto* a, auto* b) { return a->timestamp < b->timestamp; });
        return v;
    }

    uint64_t Stamp(const std::vector<CacheBlock*>& blocks);  // reverse order, skips nullptr
    uint64_t Stamp(CacheBlock* b);                           // needs next_timestamp_, stays here

    size_t size() const noexcept
    {
        return blocks_.size() - free_.size();
    }

private:
    uint64_t                 next_timestamp_{1};
    std::deque<CacheBlock>   blocks_;
    std::vector<CacheBlock*> free_;
};
```

Removed from the pool: the constructor's dummy slot at index 0, `operator[]`,
`Deallocate`, `Demote`, and every `TM_CHECK_GT(id, 0)` /
`TM_CHECK_LT(id, size)` bounds guard. `Scheduler::ValidAlloc(int)` is replaced
by the free `is_valid(const CacheBlock*)` above — it needed the scheduler only
for `cache_` member access.

`SortedIndices()` is renamed `SortedBlocks()` and returns
`std::vector<CacheBlock*>`.

## Field and type renames

The `_id` suffix would be a lie on a pointer; renames follow the data:

- `LogicalBlock::prefix_id` / `checkpoint_id` (int) →
  `CacheBlock* prefix` / `CacheBlock* checkpoint`
- `Sequence::frontier_cache_id` (int) → `CacheBlock* frontier`
- `Sequence::alloc_cache_ids` / `involved_cache_ids` (`vector<int>`) →
  `std::vector<CacheBlock*> alloc_blocks` / `involved_blocks`
- `CacheCopy{int src, dst}` → `CacheCopy{CacheBlock* src, dst}`
  (`ResolvedCopy` is untouched; it already holds raw device pointers)
- scheduler.cc file-local `ScheduleState`:
  - `evict_ids` → `std::vector<CacheBlock*> evict_blocks`
  - `planned` → `std::unordered_set<CacheBlock*>` (and `planned_now` →
    `std::vector<CacheBlock*>`)
  - `AllocReplay` / `EvictReplay` carry `CacheBlock* block`
  - `PublishPlan::cache_id` → `CacheBlock* slot`
- `Scheduler::ReleaseCacheId(int)` → `ReleaseFrontier(CacheBlock*)`;
  the `cache_id == 0` early-out becomes `b == nullptr`, no lookup, and the
  `object_id >= 0` liveness test is kept as-is (a released zombie slot was
  never re-invalidated; the check shape is unchanged).

## Call-site simplifications

Enabled by not needing the pool object to reach a block:

- `EvictingIterator` / `AllocatingIterator` drop their
  `const CacheBlockPool&` member and iterate `std::vector<CacheBlock*>`
  directly. Their deleted rvalue-container constructors stay.
- `ReplayMemory` operates on `item.block` directly:
  `item.block->Deallocate(alloc_)` for evictions,
  `b->allocation = alloc_.Allocate(b->object_id)` for allocations.
- `Finalize`'s frontier-adoption zombie swap becomes plain pointer exchanges:
  `CacheBlock* f = std::exchange(s.frontier, nullptr); ... f->owner = up;`.
- `engine.cc` setup: the copy-resolve lambda dereferences
  `CacheCopy::src/dst` directly; no `scheduler_.cache()[id]` lookup.
- The `cache_block_pool` entry is dropped from the `kSetup` `TensorMap`.
  `UnifiedAttentionLayer` reads `h->prefix->base(0) + prefix_cache_offset_`;
  `GatedDeltaNetLayer` reads `s.frontier->base(...)`. Both keep their
  `TM_CHECK_NOTNULL(...->allocation.a)` validity checks.
- With no remaining external callers, the `Scheduler::cache()` accessor is
  deleted. The teardown drain in `~Scheduler` uses `cache_.SortedBlocks()`:
  `b->Deallocate(alloc_); if (LogicalBlock* o = b->owner) logical_.Drop(o);`.
- All `cache_id != 0` / `if (*it)` sentinel tests become `if (b)` /
  `if (b == nullptr)`.

`LogicalBlockPool::Recycle` invalidates by pointer:

```cpp
if (CacheBlock* c = p->prefix) {
    cache_.Invalidate(c);
}
if (CacheBlock* c = p->checkpoint) {
    cache_.Invalidate(c);
}
```

## Contract document updates

Per `checklist.contract-sync`, `src/turbomind/engine/README.md` is updated in
the same change. Terminology only — "cache id" becomes "cache block (slot)"
where it names the handle — in `concepts.scheduler-transaction`,
`concepts.cache-object`, `ownership.object-allocator`, `ownership.prefix`,
`principles.scheduler-boundary`, `contracts.cache-prepare`,
`contracts.cache-metadata`, `contracts.scheduler-commit`,
`invariants.protection-set`, and `checklist.cache-prepare`. No normative rule
changes: one owner per slot, invalidation only at owner destruction, eviction
frees backing memory but never the slot, two-phase admission, and
`ReplayMemory` as the sole allocation/deallocation point are preserved
verbatim; only the handle representation changes from index to pointer.

## Safety argument

Pointer lifetime discipline is inherited unchanged from the id-recycling
discipline: `Invalidate` is called only from the two owner-destruction sites
(`LogicalBlockPool::Recycle` for block-owned slots, `Scheduler::Release` via
`ReleaseFrontier` for sequence-owned slots), and per `ownership.prefix` nothing
holds a cache reference past its owner's destruction. Slot reuse after
`Invalidate` is therefore invisible to correct code, exactly as index reuse is
today. Allocation-level staleness keeps the existing `alloc_key` ABA check.

## Testing

- `src/turbomind/memory/test_memory.cc` gets the mechanical pointer
  conversion; no new tests are added.
- Build with `ninja` in `build/`.
- Run the existing memory test.
- Verify with `scripts/test_turbomind_model.py` on a locally cached model
  (response length ≥ 128 tokens, response verified meaningful) — a pure
  refactor must produce identical behavior.

## Out of scope

- Any change to eviction policy, admission phases, timestamps, or ownership.
- Compaction, pool shrinking, or freeing dead slots (deque never shrinks,
  same as the current vector).
- The PyTorch engine.
