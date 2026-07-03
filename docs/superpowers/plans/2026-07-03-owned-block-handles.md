# Owned Block Handles (LogicalBlockPtr / CacheBlockPtr) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace raw-pointer ownership of cache slots with RAII handles: rename `BlockHandle` → `LogicalBlockPtr`, introduce unique `CacheBlockPtr` for the three slot-owner sites, and fold the manual allocation→owner `Retain`/`Drop` protocol into a `LogicalBlockPtr pin` member on `CacheBlock`.

**Architecture:** Pure refactor per the approved spec `docs/superpowers/specs/2026-07-03-owned-block-handles-design.md`. Behavior is unchanged; lifetime discipline moves into types. Owner sites (`LogicalBlock::prefix`/`checkpoint`, `Sequence::frontier`) hold `CacheBlockPtr`; all borrow sites stay raw `CacheBlock*` taken via `.get()`. The change does not compile piecewise — Tasks 1–3 are one compile unit of work with a single build/test/commit at the end of Task 3.

**Tech Stack:** C++17, TurboMind engine (`src/turbomind/engine`), ninja build in `build/`, Catch2 tests, `scripts/test_turbomind_model.py` for end-to-end verification.

**Contract note:** This touches scheduler/cache management, so `src/turbomind/engine/README.md` must be updated in the same change (Task 4) per `checklist.contract-sync`. No normative rule changes — only the enforcement-mechanism wording.

______________________________________________________________________

### Task 1: Handle types and pools (`block.h`, `block.cc`)

**Files:**

- Modify: `src/turbomind/engine/block.h` (full rewrite below)

- Modify: `src/turbomind/engine/block.cc` (full rewrite below)

- [ ] **Step 1: Replace `src/turbomind/engine/block.h` with:**

```cpp
#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <utility>
#include <vector>

#include "src/turbomind/core/check.h"
#include "src/turbomind/engine/fingerprint.h"
#include "src/turbomind/engine/prefix_key.h"
#include "src/turbomind/memory/common.h"
#include "src/turbomind/memory/object.h"

namespace turbomind {

struct LogicalBlock;
class LogicalBlockPool;
struct CacheBlock;
class CacheBlockPool;

// Intrusive, non-atomic strong handle to a logical block (engine-thread only).
// Holds one ref for its lifetime; copy retains, destruction drops.
class LogicalBlockPtr {
    LogicalBlock* p_{};

public:
    LogicalBlockPtr() = default;
    explicit LogicalBlockPtr(LogicalBlock* p);
    LogicalBlockPtr(const LogicalBlockPtr& o);
    LogicalBlockPtr(LogicalBlockPtr&& o) noexcept: p_{std::exchange(o.p_, nullptr)} {}
    LogicalBlockPtr& operator=(LogicalBlockPtr o) noexcept
    {
        std::swap(p_, o.p_);
        return *this;
    }
    ~LogicalBlockPtr();

    LogicalBlock& operator*() const noexcept;  // defined after LogicalBlock is complete
    LogicalBlock* operator->() const noexcept
    {
        return p_;
    }
    LogicalBlock* get() const noexcept
    {
        return p_;
    }
    explicit operator bool() const noexcept
    {
        return p_ != nullptr;
    }
    friend bool operator==(const LogicalBlockPtr& a, const LogicalBlockPtr& b) noexcept
    {
        return a.p_ == b.p_;
    }
};

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
    ~CacheBlockPtr();  // defined after CacheBlockPool is complete

    CacheBlock& operator*() const noexcept;  // defined after CacheBlock is complete
    CacheBlock* operator->() const noexcept
    {
        return p_;
    }
    CacheBlock* get() const noexcept
    {
        return p_;
    }
    explicit operator bool() const noexcept
    {
        return p_ != nullptr;
    }
};

struct CacheBlock {
    uint64_t       timestamp{};    // eviction priority; zero means highest
    int            object_id{-1};  // ObjectAllocator registration id
    object_alloc_t allocation{};   // {const Allocation*}; .a == nullptr => no live allocation
    uint64_t       alloc_key{};    // snapshot of allocation->key at replay (ABA stale check)

    // Slot -> owning logical block (weak identity). Set at Create; persists
    // across evict/realloc. nullptr = sequence-owned (frontier).
    LogicalBlock* owner{};

    // Allocation-held strong ref on `owner`: non-empty iff the allocation is
    // valid and owner != nullptr. Taken at replay-commit, dropped by
    // Deallocate. Replaces the manual LogicalBlockPool::Retain/Drop pairing.
    LogicalBlockPtr pin;

    CacheBlockPool* mgr{};  // set at Create; used by ~CacheBlockPtr to invalidate

    // Base of part `p`; `part` indexes the resolved Allocation.
    char* base(int part) const
    {
        return allocation->base(part);
    }
    int part_count() const
    {
        return allocation->part_count();
    }
    bool valid() const noexcept
    {
        return allocation.a != nullptr;
    }

    // Deallocates the backing object and clears the slot back to "no
    // allocation" state (the owner identity persists). Dropping the pin may
    // recycle the owner, which invalidates this very slot — callers must not
    // touch the slot after this returns. Pre-condition: a live allocation.
    void Deallocate(ObjectAllocator& alloc);

    // Demote to evict-first priority: timestamp 0 sorts first in
    // SortedBlocks() and is below every eviction cutoff and pass floor.
    // Stamp never hands out 0 (next_timestamp_ starts at 1).
    void Demote() noexcept
    {
        TM_CHECK(valid());
        timestamp = 0;
    }
};

// nullptr means "no slot".
inline bool is_valid(const CacheBlock* b) noexcept
{
    return b != nullptr && b->valid();
}

inline bool is_valid(const CacheBlockPtr& b) noexcept
{
    return is_valid(b.get());
}

class CacheBlockPool {
public:
    CacheBlockPtr Create(int object_id, LogicalBlock* owner = nullptr);

    // Eviction candidates: exactly the currently allocated blocks. The cached
    // allocation handle is the validity flag; the timestamp only orders the candidates.
    std::vector<CacheBlock*> SortedBlocks();

    uint64_t Stamp(const std::vector<CacheBlock*>& blocks);
    uint64_t Stamp(CacheBlock* b);

    size_t size() const noexcept
    {
        return blocks_.size() - free_.size();
    }

private:
    friend class CacheBlockPtr;

    // Owner handle destroyed; reset the slot and return it for reuse.
    void Invalidate(CacheBlock* b);

    uint64_t next_timestamp_{1};

    std::deque<CacheBlock>   blocks_;  // stable addresses; never shrinks
    std::vector<CacheBlock*> free_;
};

inline CacheBlockPtr::~CacheBlockPtr()
{
    if (p_) {
        TM_CHECK(!p_->valid());
        p_->mgr->Invalidate(p_);
    }
}

inline CacheBlock& CacheBlockPtr::operator*() const noexcept
{
    return *p_;
}

struct LogicalBlock {
    // Position and content extent within the sequence
    int offset{-1};
    int capacity{0};
    int size{0};  // filled tokens of an indexed node; 0 for private blocks

    // Intrusive strong refcount (requests + partial sibling edges + valid allocations)
    int               refs{0};
    LogicalBlockPool* mgr{};  // set at Create; used by handle / Retain / Drop

    // Cache slots, one per category; owned (empty handle = not created)
    CacheBlockPtr prefix;
    CacheBlockPtr checkpoint;

    // Prefix trie node state (mutated only via the trie methods)
    const LogicalBlock*      parent{};  // nullptr = root; non-owning identity
    PrefixKey                key{};     // empty => not (yet) a prefix node
    std::vector<int>         tokens;
    std::vector<Fingerprint> image_fps;       // start-fingerprints of images beginning in this block (usually empty)
    bool                     indexed{false};  // present in the trie index

    // First-known indexed partial sibling at this block index: an identity-
    // verified node with the same parent and a strict token-prefix of this
    // block's content. Every edge points to a sibling with strictly smaller
    // `size` (a carrier indexed later by Finalize only grows), so
    // size strictly decreases along edge paths and the graph is acyclic.
    // First-wins: bound at most once, at AdmitPrompt, on a block created in the
    // same pass (mirrors trie first-wins insertion). Strong, RAII.
    LogicalBlockPtr partial;

    bool     is_valid{false};  // content proven produced; cleared on prefix evict
    uint64_t producer{0};      // request currently writing this range; 0 = none
};

// Owns logical block lifetime via an intrusive refcount. Nodes live in a
// deque with a free list (stable addresses, never shrinks), so a
// LogicalBlock* is a stable identity. When refs reaches 0 the node is
// recycled: a recycle hook removes it from the PrefixTrie index, every
// attached cache slot's allocation is already invalid (a valid allocation
// holds a ref via CacheBlock::pin), so destroying the slot handles only
// returns slot metadata.
class LogicalBlockPool {
public:
    explicit LogicalBlockPool(int block_size = 0): block_size_{block_size} {}

    ~LogicalBlockPool();

    void ResetBlockSize(int block_size)
    {
        TM_CHECK_GT(block_size, 0);
        TM_CHECK_EQ(live_, 0);
        block_size_ = block_size;
    }

    int block_size() const noexcept
    {
        return block_size_;
    }

    void set_recycle_hook(std::function<void(LogicalBlock&)> h)
    {
        on_recycle_ = std::move(h);
    }

    LogicalBlockPtr Create(int logical_index);

    void Retain(LogicalBlock* p) noexcept
    {
        if (p) {
            ++p->refs;
        }
    }

    void Drop(LogicalBlock* p)  // sole decrement funnel
    {
        if (p) {
            TM_CHECK_GT(p->refs, 0);
            if (--p->refs == 0) {
                Recycle(p);
            }
        }
    }

    size_t size() const noexcept
    {
        return live_;
    }

private:
    void Recycle(LogicalBlock* p);  // sole place that frees a node

    int block_size_{};
    int live_{};

    std::deque<LogicalBlock>   nodes_;  // stable addresses; never shrinks
    std::vector<LogicalBlock*> free_;

    std::function<void(LogicalBlock&)> on_recycle_;
};

inline LogicalBlockPtr::LogicalBlockPtr(LogicalBlock* p): p_{p}
{
    if (p_) {
        p_->mgr->Retain(p_);
    }
}

inline LogicalBlockPtr::LogicalBlockPtr(const LogicalBlockPtr& o): p_{o.p_}
{
    if (p_) {
        p_->mgr->Retain(p_);
    }
}

inline LogicalBlockPtr::~LogicalBlockPtr()
{
    if (p_) {
        p_->mgr->Drop(p_);
    }
}

inline LogicalBlock& LogicalBlockPtr::operator*() const noexcept
{
    return *p_;
}

}  // namespace turbomind
```

Notes on what changed vs the old file:

- `BlockHandle` renamed `LogicalBlockPtr`; new `CacheBlockPtr` inserted before `CacheBlock` (it only needs forward declarations; its `valid()`-touching bodies are out of line after `CacheBlockPool`).

- `CacheBlock` gained `pin` and `mgr`; `LogicalBlock::prefix`/`checkpoint` are now `CacheBlockPtr` (this makes both structs move-only — the pools' `emplace_back` and `*p = T{}` move-assign resets still compile).

- `CacheBlockPool::Invalidate` is private (`friend class CacheBlockPtr`); `Create` returns `CacheBlockPtr`.

- `LogicalBlockPool` no longer needs `CacheBlockPool& cache_` (its only use was `Recycle`'s manual `Invalidate` calls, now gone), so the constructor loses that parameter.

- Added `is_valid(const CacheBlockPtr&)` overload so most call sites compile unchanged.

- [ ] **Step 2: Replace `src/turbomind/engine/block.cc` with:**

```cpp
#include "src/turbomind/engine/block.h"

#include <algorithm>
#include <utility>

namespace turbomind {

CacheBlockPtr CacheBlockPool::Create(int object_id, LogicalBlock* owner)
{
    TM_CHECK_GE(object_id, 0);
    CacheBlock* b;
    if (TM_UNLIKELY(free_.empty())) {
        b = &blocks_.emplace_back();
    }
    else {
        b = free_.back();
        free_.pop_back();
    }
    *b           = CacheBlock{};
    b->object_id = object_id;
    b->owner     = owner;
    b->mgr       = this;
    return CacheBlockPtr{b};
}

void CacheBlockPool::Invalidate(CacheBlock* b)
{
    TM_CHECK_GE(b->object_id, 0);  // double-invalidate check
    *b = CacheBlock{};
    free_.push_back(b);
}

void CacheBlock::Deallocate(ObjectAllocator& alloc)
{
    TM_CHECK(valid());
    alloc.Deallocate(object_id, allocation);
    allocation = {};
    alloc_key  = 0;
    timestamp  = 0;
    pin        = {};  // may recycle the owner and free this slot; do last
}

std::vector<CacheBlock*> CacheBlockPool::SortedBlocks()
{
    std::vector<CacheBlock*> v;
    v.reserve(blocks_.size());
    for (auto& b : blocks_) {
        if (b.valid()) {
            v.push_back(&b);
        }
    }
    std::sort(v.begin(), v.end(), [](const CacheBlock* a, const CacheBlock* b) {  //
        return a->timestamp < b->timestamp;
    });
    return v;
}

uint64_t CacheBlockPool::Stamp(const std::vector<CacheBlock*>& blocks)
{
    const auto ret = next_timestamp_;
    for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) {
        if (*it) {
            Stamp(*it);
        }
    }
    return ret;
}

uint64_t CacheBlockPool::Stamp(CacheBlock* b)
{
    TM_CHECK_GE(TM_CHECK_NOTNULL(b)->object_id, 0);
    b->timestamp = next_timestamp_++;
    return b->timestamp;
}

LogicalBlockPool::~LogicalBlockPool()
{
    if (live_ != 0) {
        TM_LOG_ERROR("leaked {} logical blocks", live_);
    }
}

LogicalBlockPtr LogicalBlockPool::Create(int logical_index)
{
    TM_CHECK_GT(block_size_, 0);
    TM_CHECK_GE(logical_index, 0);

    LogicalBlock* p;
    if (TM_UNLIKELY(free_.empty())) {
        p = &nodes_.emplace_back();
    }
    else {
        p = free_.back();
        free_.pop_back();
    }
    p->mgr      = this;
    p->offset   = logical_index * block_size_;
    p->capacity = block_size_;
    ++live_;
    return LogicalBlockPtr{p};  // refs 0 -> 1
}

void LogicalBlockPool::Recycle(LogicalBlock* p)
{
    if (on_recycle_) {
        on_recycle_(*p);  // PrefixTrie::Erase (pool stays prefix-agnostic)
    }
    // Destroying prefix/checkpoint handles invalidates the slots; their
    // allocations are already gone (a valid allocation's pin holds a ref).
    *p = LogicalBlock{};  // drops fork edge + slot handles, frees tokens
    free_.push_back(p);
    --live_;
}

}  // namespace turbomind
```

- [ ] **Step 3: Do NOT build yet** — `request.h` and `scheduler.cc` still use the old names; continue to Task 2.

______________________________________________________________________

### Task 2: Owner fields in `request.h`

**Files:**

- Modify: `src/turbomind/engine/request.h:220` and `:239`

- [ ] **Step 1: Rename the block-handle vector**

Change:

```cpp
    std::vector<BlockHandle> block_ids;  // logical (each holds one request ref)
```

to:

```cpp
    std::vector<LogicalBlockPtr> block_ids;  // logical (each holds one request ref)
```

- [ ] **Step 2: Make the frontier owned**

Change:

```cpp
    CacheBlock*   frontier       = nullptr;  // checkpoint working state for the next forward
```

to:

```cpp
    CacheBlockPtr frontier;                  // checkpoint working state for the next forward (owned slot)
```

(`CacheCopy` and the surrounding raw-pointer fields — `publish_target`, `alloc_blocks`, `involved_blocks` — are borrows and stay unchanged.)

______________________________________________________________________

### Task 3: Scheduler and module call sites

**Files:**

- Modify: `src/turbomind/engine/scheduler.h` (remove `ReleaseFrontier` declaration, line 208)

- Modify: `src/turbomind/engine/scheduler.cc` (sites listed below; line numbers are pre-change)

- Modify: `src/turbomind/models/llama/GatedDeltaNetLayer.cc:162`

- (No change needed: `src/turbomind/engine/engine.cc` and `src/turbomind/models/llama/unified_attention_layer.cc:302` — `CacheCopy` already carries raw pointers, and `*h->prefix` compiles via `CacheBlockPtr::operator*`.)

- [ ] **Step 1: `scheduler.h` — delete the declaration**

Remove:

```cpp
    void ReleaseFrontier(CacheBlock* b);
```

- [ ] **Step 2: `scheduler.cc` — constructor init and rename sweep**

Line 300: `logical_{cache_, cache_block_seq_len}` → `logical_{cache_block_seq_len}` (the pool no longer takes the cache pool).

Rename every `BlockHandle` to `LogicalBlockPtr` (lines 334, 382, 414, 462, 489, 539, 647, 768). Line 462's comment stays: `x.partial = LogicalBlockPtr{v};  // edge ref`.

- [ ] **Step 3: `scheduler.cc` — teardown drain (~Scheduler, lines 316–324)**

Replace the loop body with:

```cpp
    // Drain all live allocations so allocation-held pins are released and the
    // remaining trie nodes recycle before the pools are destroyed.
    // SortedBlocks() returns exactly the allocated blocks (alloc set).
    // A block with two valid slots holds two pins, so it recycles only after
    // its last slot deallocates — same order as the old explicit Drop.
    for (CacheBlock* b : cache_.SortedBlocks()) {
        b->Deallocate(alloc_);
    }
```

- [ ] **Step 4: `scheduler.cc` — slot creation sites compile as-is**

Lines 335, 416, 496 (`h->prefix = cache_.Create(registry_.prefix().object_id(), h.get());` and the `y.prefix` variant) need no text change — `Create` now returns `CacheBlockPtr` and the assignment moves. Verify only.

- [ ] **Step 5: `scheduler.cc` — `PlanResume` (lines 529–657)**

Line 529: `if (ckpt && s.frontier == nullptr)` → `if (ckpt && !s.frontier)`.

Line 573: `best = {e, ResumeSource::kFork, ckpt ? y->checkpoint : nullptr, y, &x};` → `best = {e, ResumeSource::kFork, ckpt ? y->checkpoint.get() : nullptr, y, &x};`

Line 607: `best = {e, ResumeSource::kCheckpoint, x.checkpoint};` → `best = {e, ResumeSource::kCheckpoint, x.checkpoint.get()};`

Line 613: `best = {ye, ResumeSource::kFork, y->checkpoint};` → `best = {ye, ResumeSource::kFork, y->checkpoint.get()};`

Line 627: `s.restore_copies.push_back({best.fork_src->prefix, best.fork_dst->prefix});` → `s.restore_copies.push_back({best.fork_src->prefix.get(), best.fork_dst->prefix.get()});`

Line 630: `s.restore_copies.push_back({best.ckpt, s.frontier});` → `s.restore_copies.push_back({best.ckpt, s.frontier.get()});`

Lines 649–657: append `.get()` on the pushes:

```cpp
        s.involved_blocks.push_back(x.prefix.get());
        if (!is_valid(x.prefix)) {
            s.alloc_blocks.push_back(x.prefix.get());
        }
    }
    if (ckpt) {
        s.involved_blocks.push_back(s.frontier.get());
        if (!is_valid(s.frontier)) {
            s.alloc_blocks.push_back(s.frontier.get());
        }
    }
```

(`is_valid(x.prefix)` / `is_valid(s.frontier)` resolve to the new `CacheBlockPtr` overload — no change.)

- [ ] **Step 6: `scheduler.cc` — `PlanContinue` line 683**

`CacheBlock* p = s.block_ids[i]->prefix;` → `CacheBlock* p = s.block_ids[i]->prefix.get();`

- [ ] **Step 7: `scheduler.cc` — delete `ReleaseFrontier`, rewrite `Release` (lines 752–783)**

Delete the whole `Scheduler::ReleaseFrontier` function. In `Release`, change the private-block loop and the frontier teardown:

```cpp
void Scheduler::Release(Sequence& s)
{
    for (const LogicalBlockPtr& h : s.block_ids) {
        LogicalBlock& x = *h;
        if (!x.indexed) {
            // Private blocks are undiscoverable: drop their allocations now so
            // the allocation-held pins go away and the block can recycle.
            for (CacheBlock* c : {x.prefix.get(), x.checkpoint.get()}) {
                if (is_valid(c)) {
                    c->Deallocate(alloc_);  // drops the pin (request ref still pins x)
                }
            }
        }
    }
    s.block_ids.clear();  // request refs -> recycles unreferenced blocks

    // Sequence-owned slot (frontier / adopted zombie): release the memory,
    // then drop the handle (its destructor invalidates the slot).
    if (s.frontier) {
        TM_CHECK(s.frontier->owner == nullptr);
        if (s.frontier->valid()) {
            s.frontier->Deallocate(alloc_);
        }
        s.frontier = {};
    }
```

(The trailing field resets from line 785 on are unchanged; delete the old `ReleaseFrontier(std::exchange(s.frontier, nullptr));` line.)

- [ ] **Step 8: `scheduler.cc` — `Finalize` frontier adoption (lines 885–897)**

Replace:

```cpp
            CacheBlock* f = std::exchange(s.frontier, nullptr);
            if (CacheBlock* zombie = std::exchange(x.checkpoint, nullptr)) {
                // Created-but-unallocated slot: transfer it to the dying
                // sequence so it is invalidated with its owner at Release.
                TM_CHECK(!zombie->valid());
                zombie->owner = nullptr;
                s.frontier    = zombie;
            }
            x.checkpoint = f;
            f->owner     = up;
            // The frontier's allocation was committed while the slot was
            // sequence-owned (no ref); the ref moves with the ownership.
            logical_.Retain(up);
```

with:

```cpp
            CacheBlockPtr f = std::move(s.frontier);
            if (CacheBlockPtr zombie = std::move(x.checkpoint)) {
                // Created-but-unallocated slot: transfer it to the dying
                // sequence so it is invalidated with its owner at Release.
                TM_CHECK(!zombie->valid());
                zombie->owner = nullptr;
                s.frontier    = std::move(zombie);
            }
            f->owner = up;
            // The frontier's allocation was committed while the slot was
            // sequence-owned (no pin); the pin is taken as ownership moves.
            f->pin       = LogicalBlockPtr{up};
            x.checkpoint = std::move(f);
```

- [ ] **Step 9: `scheduler.cc` — `PlanPublication` (lines 951–991)**

Lines 951–953 (`pass.planned` holds raw pointers):

```cpp
    if (at_prompt_boundary && sibling && !sibling->is_valid && !is_valid(sibling->prefix)
        && !pass.planned.count(sibling->prefix.get())) {
        pass.planned.insert(sibling->prefix.get());
```

Lines 986–991:

```cpp
    if (!is_valid(node->checkpoint)) {
        if (!node->checkpoint) {
            node->checkpoint = cache_.Create(registry_.checkpoint().object_id(), node);
        }
        TM_CHECK(pass.planned.insert(node->checkpoint.get()).second);
        pass.pending_publish[i] = {node, end, node->checkpoint.get()};
    }
```

- [ ] **Step 10: `scheduler.cc` — `RunOptionalAdmission` line 1273**

`if (!try_optional(node->prefix))` → `if (!try_optional(node->prefix.get()))`

- [ ] **Step 11: `scheduler.cc` — `ReplayMemory` (lines 1300–1315)**

Replace the visitor body's two branches with:

```cpp
                if constexpr (std::is_same_v<T, EvictReplay>) {
                    const bool is_prefix = c.object_id == registry_.prefix().object_id_or_negative();
                    if (LogicalBlock* o = c.owner; o && is_prefix) {
                        o->is_valid = false;  // read owner before the slot may die
                    }
                    c.Deallocate(alloc_);  // drops the pin; may recycle the owner and free this slot
                }
                else {
                    c.allocation = alloc_.Allocate(c.object_id);  // single-object; {nullptr} on OOM
                    TM_CHECK(c.allocation.a);                     // admission guarantees capacity
                    c.alloc_key = c.allocation->key;              // snapshot for stale detection
                    c.pin       = LogicalBlockPtr{c.owner};       // empty when owner == nullptr (sequence-owned)
                }
```

- [ ] **Step 12: `scheduler.cc` — `CommitResults` (lines 1365–1380)**

Line 1365: `s.publish_copies.push_back({s.block_ids[(end - 1) / bs]->prefix, y.prefix});` → `s.publish_copies.push_back({s.block_ids[(end - 1) / bs]->prefix.get(), y.prefix.get()});`

Line 1368: `cache_.Stamp(y.prefix);` → `cache_.Stamp(y.prefix.get());`

Line 1376: `CacheBlock* slot = s.publish_target->checkpoint;` → `CacheBlock* slot = s.publish_target->checkpoint.get();`

Line 1380: `s.publish_copies.push_back({s.frontier, slot});` → `s.publish_copies.push_back({s.frontier.get(), slot});`

- [ ] **Step 13: `GatedDeltaNetLayer.cc:162`**

`const CacheBlock& cb = *TM_CHECK_NOTNULL(s.frontier);` → `const CacheBlock& cb = *TM_CHECK_NOTNULL(s.frontier.get());`

- [ ] **Step 14: Build**

Run from `build/`: `ninja`
Expected: clean build. Fix any residual compile errors the sweep missed (they will be old-name uses or raw/handle mismatches of exactly the kinds converted above — apply the same `.get()` / rename transformations; do not change semantics).

- [ ] **Step 15: Run the existing unit tests**

From `build/`, locate and run the memory and prefix-trie test binaries (e.g. `ls bin | grep -i -e memory -e trie`, then run them, e.g. `./bin/test_memory && ./bin/test_prefix_trie`).
Expected: all assertions pass, exit code 0 for both.

- [ ] **Step 16: Commit**

```bash
git add src/turbomind/engine/block.h src/turbomind/engine/block.cc \
        src/turbomind/engine/request.h src/turbomind/engine/scheduler.h \
        src/turbomind/engine/scheduler.cc \
        src/turbomind/models/llama/GatedDeltaNetLayer.cc
git commit -m "refactor: owned handles for cache slots (LogicalBlockPtr/CacheBlockPtr)"
```

(Include any additional files fixed in Step 14.)

______________________________________________________________________

### Task 4: Contract document update (`src/turbomind/engine/README.md`)

**Files:**

- Modify: `src/turbomind/engine/README.md` — leaves `concepts.logical-block`, `ownership.prefix`, `contracts.cache-metadata`, `contracts.eviction`, `contracts.cancel-release`, `contracts.checkpoint-adoption`

Rules: terminology/mechanism updates only, no normative rule changes. Preserve existing line wrapping; edit content, not wrapping. The recurring transformation: "strong references held through RAII `BlockHandle`s" → "`LogicalBlockPtr`s"; "the allocation reference taken and dropped explicitly via `LogicalBlockPool::Retain`/`Drop` keyed on `CacheBlock::owner`" → "the allocation reference held as the slot's `LogicalBlockPtr pin`, taken at replay-commit and dropped by `Deallocate`"; slot invalidation "called at owner destruction" → "performed by the owning `CacheBlockPtr`'s destruction".

- [ ] **Step 1: `concepts.logical-block`**

In the sentence describing the refcount, replace the parenthetical `(request and fork references held through `BlockHandle`s; the cache-allocation reference taken via `Retain`/`Drop`keyed on the slot's`CacheBlock::owner` identity)` with `(request and fork references held through `LogicalBlockPtr`s; the cache-allocation reference held as the slot's `LogicalBlockPtr pin`)`.

- [ ] **Step 2: `ownership.prefix`**

In the long lifetime paragraph, replace the mechanism description: `BlockHandle`s → `LogicalBlockPtr`s, and rewrite the clause "the strong allocation reference is taken and dropped explicitly via `LogicalBlockPool::Retain`/`Drop` keyed on that `owner` (sequence-owned slots leave `owner == nullptr` and take no allocation ref)" as "the strong allocation reference is the slot's `LogicalBlockPtr pin`, set when the memory replay commits the allocation and cleared by `CacheBlock::Deallocate` (sequence-owned slots leave `owner == nullptr` and keep an empty pin)". Keep every rule (one owner per slot, invalidation only at owner destruction) verbatim; add that slot handles are unique `CacheBlockPtr`s whose destruction invalidates the slot.

- [ ] **Step 3: `contracts.cache-metadata`**

Update "a valid allocation holds one strong ref on its owner" to name the pin: "a valid allocation holds one strong ref on its owner via the slot's `LogicalBlockPtr pin`". Mention the slot's `mgr` back-pointer only if the section already enumerates fields; otherwise leave field lists as-is.

- [ ] **Step 4: `contracts.eviction`**

"Evicting an allocation releases the allocation reference it held on its logical block when the slot's `owner` is set" → "... releases the allocation reference (the slot's pin) it held on its logical block ...". No rule change.

- [ ] **Step 5: `contracts.cancel-release`**

"each allocation holds an allocation ref in `LogicalBlock::refs` via `Retain`/`Drop` on the slot's `owner`" → "each allocation holds an allocation ref in `LogicalBlock::refs` via the slot's pin". No rule change.

- [ ] **Step 6: `contracts.checkpoint-adoption`**

"transfers the allocation's reference to that block" → "transfers the allocation's reference to that block (the pin is set on the now-block-owned slot)". No rule change.

- [ ] **Step 7: Commit**

```bash
git add src/turbomind/engine/README.md
git commit -m "docs: update engine contract wording for owned block handles"
```

______________________________________________________________________

### Task 5: End-to-end model verification

Rules (workspace): check GPU availability first; GPU commands must run outside the sandbox; use the test script AS IS; response must be ≥128 tokens and meaningful; never install lmdeploy.

- [ ] **Step 1: Pick a model and check the GPU**

Read `/data/models.json` for a locally cached model and its cache directory. Check `nvidia-smi` for a free GPU (run unsandboxed).

- [ ] **Step 2: Run the model test (unsandboxed)**

Run `scripts/test_turbomind_model.py` per its usage (read the script header for arguments), targeting the chosen model, with requested response length ≥128 tokens.
Expected: the model responds with coherent human language relevant to the test prompt. Gibberish = bug; if OOM, check whether the GPU is occupied by another process before debugging.

- [ ] **Step 3: Verify the response and finish**

Read the actual output text and confirm it is meaningful. This is a pure refactor: any behavior difference is a bug — if found, debug (start from the pin/Deallocate ordering and the Finalize adoption move sequence) and iterate until fixed.
