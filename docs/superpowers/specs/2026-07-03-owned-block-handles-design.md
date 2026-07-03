# Owned block handles design

Date: 2026-07-03
Status: approved

## Goal

Manage `LogicalBlock` and `CacheBlock` lifetimes through owned RAII handles
instead of raw pointers at the ownership edges:

- Rename the existing `BlockHandle` to `LogicalBlockPtr` (semantics unchanged).
- Introduce a move-only unique `CacheBlockPtr` for cache slots; slot
  invalidation becomes automatic at owner destruction.
- Fold the manually paired allocation→owner strong reference
  (`LogicalBlockPool::Retain`/`Drop` keyed on `CacheBlock::owner`) into a
  `LogicalBlockPtr pin` member held by the slot while its allocation is valid.

Scope is the ownership edges only. Borrowed references stay raw pointers:
`Sequence::alloc_blocks` / `involved_blocks`, `CacheCopy`, replay records, the
per-pass `planned` set, `PublishPlan::slot`, `ResumeCandidate::ckpt`,
`Sequence::publish_target`, `LogicalBlock::parent`, `CacheBlock::owner`, and
the `PrefixTrie` index (strong handles in the trie would make indexed nodes
immortal and break eviction reclaim; the recycle hook stays).

This is a pure refactor: behavior, ownership rules, admission order, eviction
order, and the memory replay commit point are unchanged. Lifetime discipline
that today lives in documented protocol moves into types.

## Handle types (block.h)

`BlockHandle` is renamed `LogicalBlockPtr` everywhere (`Sequence::block_ids`,
`LogicalBlock::partial`, `LogicalBlockPool::Create`, tests). No semantic
change: intrusive, non-atomic, copyable strong handle.

The new unique handle:

```cpp
// Unique, non-atomic owning handle to a cache slot (engine-thread only).
// Destruction invalidates the slot. Precondition at destruction: the slot's
// allocation is already gone (memory release is a separate concern; the
// normal path is ReplayMemory, the release paths call Deallocate first).
class CacheBlockPtr {
    CacheBlock* p_{};

public:
    CacheBlockPtr() = default;
    explicit CacheBlockPtr(CacheBlock* p) noexcept: p_{p} {}
    CacheBlockPtr(const CacheBlockPtr&) = delete;
    CacheBlockPtr(CacheBlockPtr&& o) noexcept: p_{std::exchange(o.p_, nullptr)} {}
    CacheBlockPtr& operator=(CacheBlockPtr o) noexcept
    {
        std::swap(p_, o.p_);
        return *this;
    }
    ~CacheBlockPtr()
    {
        if (p_) {
            TM_CHECK(!p_->valid());
            p_->mgr->Invalidate(p_);
        }
    }

    CacheBlock* get() const noexcept { return p_; }
    CacheBlock* operator->() const noexcept { return p_; }
    explicit operator bool() const noexcept { return p_ != nullptr; }
};
```

(Definition order in the header follows the `BlockHandle` pattern: class
first, `valid()`-touching member bodies out of line after `CacheBlock` and
`CacheBlockPool` are complete.)

`CacheBlock` gains a `CacheBlockPool* mgr{}` back-pointer, set at `Create`,
mirroring `LogicalBlock::mgr`. `CacheBlockPool::Create` returns
`CacheBlockPtr`. `CacheBlockPool::Invalidate` becomes private; only the
handle destructor calls it (`friend class CacheBlockPtr`).

## Owner sites become handles

```cpp
struct LogicalBlock {
    // ...
    CacheBlockPtr prefix;
    CacheBlockPtr checkpoint;
};

struct Sequence {
    // ...
    CacheBlockPtr frontier;
};
```

- `LogicalBlockPool::Recycle` drops its two manual `Invalidate` calls;
  `*p = LogicalBlock{}` destroys the handles.
- `Scheduler::ReleaseFrontier` is deleted. `Scheduler::Release` does:

```cpp
if (s.frontier && s.frontier->valid()) {
    s.frontier->Deallocate(alloc_);
}
s.frontier = {};
```

- Borrow sites take raw pointers via `.get()`. The free
  `is_valid(const CacheBlock*)` predicate stays; call sites holding a handle
  use `is_valid(h.get())` (or an overload taking `const CacheBlockPtr&`).

## The pin: allocation→owner ref folded in

```cpp
struct CacheBlock {
    // ...
    LogicalBlock*   owner{};  // weak identity; survives eviction
    LogicalBlockPtr pin;      // held iff allocation valid && owner != nullptr
};
```

`Deallocate` drops the pin as its last action. Dropping the pin may recycle
the owner, which invalidates this very slot — the same hazard shape as
today's post-`Drop` comment, so callers must not touch the slot after
`Deallocate` returns:

```cpp
void CacheBlock::Deallocate(ObjectAllocator& alloc)
{
    TM_CHECK(valid());
    alloc.Deallocate(object_id, allocation);
    allocation = {};
    alloc_key  = 0;
    timestamp  = 0;
    pin        = {};  // may recycle owner and free this slot; do last
}
```

The five manual `Retain`/`Drop` sites collapse:

1. **Replay alloc** (`ReplayMemory`): `logical_.Retain(c.owner)` becomes
   `c.pin = LogicalBlockPtr{c.owner}` (the constructor no-ops on nullptr, so
   sequence-owned slots keep an empty pin).
2. **Replay evict** (`ReplayMemory`): the prefix-category
   `owner->is_valid = false` flip moves *before* `Deallocate` (it reads
   `c.owner` while the slot is still safe to touch), then
   `c.Deallocate(alloc_)` alone; the explicit `Drop` is gone.
3. **`Release` private-block path**: just `c->Deallocate(alloc_)`; the
   explicit `Drop` is gone.
4. **Teardown drain in `~Scheduler`**: just `b->Deallocate(alloc_)` per
   sorted block. Safe: a block with two valid slots holds two pins, so
   recycle fires only after its last slot deallocates — same order as today.
5. **`Finalize` frontier adoption**: handle moves; the ref transfer is
   setting the pin on the now-block-owned slot:

```cpp
CacheBlockPtr f = std::move(s.frontier);
if (CacheBlockPtr zombie = std::move(x.checkpoint)) {
    TM_CHECK(!zombie->valid());
    zombie->owner = nullptr;
    s.frontier = std::move(zombie);  // invalidated with the sequence at Release
}
f->owner     = up;
f->pin       = LogicalBlockPtr{up};  // was logical_.Retain(up)
x.checkpoint = std::move(f);
```

No ownership cycle: a non-empty pin implies `refs > 0`, so `Recycle` only
ever destroys slot handles whose pins are already empty.

## Contract document updates

Per `checklist.contract-sync`, `src/turbomind/engine/README.md` is updated in
the same change: `concepts.logical-block`, `ownership.prefix`,
`contracts.cache-metadata`, `contracts.eviction`, `contracts.cancel-release`,
`contracts.checkpoint-adoption`. The "explicit `Retain`/`Drop` keyed on
`owner`" wording becomes "a valid allocation holds a `LogicalBlockPtr` pin on
its owner; slot handles are unique `CacheBlockPtr`s that invalidate the slot
at owner destruction." No normative rule changes: one owner per slot,
invalidation only at owner destruction, eviction frees backing memory but
never the slot, and `ReplayMemory` as the sole normal allocation point are
preserved; only the enforcement mechanism moves from documented discipline
into types.

## Testing

- `src/turbomind/memory/test_memory.cc` gets the mechanical handle
  conversion; no new tests are added.
- Build with `ninja` in `build/`.
- Run the existing memory test.
- Verify with `scripts/test_turbomind_model.py` on a locally cached model
  (response length ≥ 128 tokens, response verified meaningful) — a pure
  refactor must produce identical behavior.

## Out of scope

- Converting cross-pass borrows (`involved_blocks`, restore-copy sources,
  `publish_target`) into strong handles; they keep today's discipline.
- Any change to eviction policy, admission phases, timestamps, or ownership
  semantics.
- The PyTorch engine.
