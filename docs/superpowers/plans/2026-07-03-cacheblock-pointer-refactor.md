# CacheBlock Pointer Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every `int` cache id with a real `CacheBlock*` — in the pool, `LogicalBlock`, `Sequence`, copy plans, replay records, engine setup, and modules.

**Architecture:** `CacheBlockPool` switches from `std::vector<CacheBlock>` + index free list to `std::deque<CacheBlock>` (stable addresses) + pointer free list. The index-0 sentinel becomes `nullptr`. `Deallocate`/`Demote` move onto `CacheBlock`; `Scheduler::ValidAlloc(int)` becomes a free `is_valid(const CacheBlock*)` in `block.h`. Modules dereference blocks directly, so the `cache_block_pool` env entry and `Scheduler::cache()` accessor are deleted.

**Tech Stack:** C++17, ninja build in `build/`, verification via `scripts/test_turbomind_model.py`.

**Spec:** `docs/superpowers/specs/2026-07-03-cacheblock-pointer-refactor-design.md`

**Important:** This is one atomic type change across headers; the code does not compile until Tasks 1–7 are all done. Do NOT run `ninja` between them expecting success. There is exactly one code commit (after the build passes), then the README commit, then verification.

**Contract:** `src/turbomind/engine/README.md` is normative. This refactor changes handle representation only; every rule in `ownership.prefix`, `contracts.scheduler-admission`, `contracts.allocation` etc. must hold unchanged. Task 8 updates the document's terminology in the same change (`checklist.contract-sync`).

---

### Task 1: `block.h` / `block.cc` — pool and block API

**Files:**
- Modify: `src/turbomind/engine/block.h`
- Modify: `src/turbomind/engine/block.cc`

- [ ] **Step 1.1: Rewrite `CacheBlock` and `CacheBlockPool` in `block.h`**

Replace the includes line `#include <vector>` section to also have `#include <deque>` (keep `<vector>`). Replace the `CacheBlock` struct and `CacheBlockPool` class (currently lines 56–134) with:

```cpp
struct CacheBlock {
    uint64_t       timestamp{};    // eviction priority; zero means highest
    int            object_id{-1};  // ObjectAllocator registration id
    object_alloc_t allocation{};   // {const Allocation*}; .a == nullptr => no live allocation
    uint64_t       alloc_key{};    // snapshot of allocation->key at replay (ABA stale check)

    // Slot -> owning logical block (weak identity). Set at Create; persists
    // across evict/realloc. nullptr = sequence-owned (frontier).
    LogicalBlock* owner{};

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
    // allocation" state (the owner identity persists). Pre-condition: the
    // slot has a live allocation.
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

// nullptr replaces the old index-0 sentinel ("no slot").
inline bool is_valid(const CacheBlock* b) noexcept
{
    return b != nullptr && b->valid();
}

class CacheBlockPool {
public:
    CacheBlock* Create(int object_id, LogicalBlock* owner = nullptr);

    // Owner destroyed; reset the slot and return it for reuse.
    void Invalidate(CacheBlock* b);

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
    uint64_t next_timestamp_{1};

    std::deque<CacheBlock>   blocks_;  // stable addresses; never shrinks
    std::vector<CacheBlock*> free_;
};
```

Notes: the default constructor with the dummy index-0 slot, `operator[]`, and the pool-level `Deallocate`/`Demote`/`SortedIndices` are gone.

- [ ] **Step 1.2: Update `LogicalBlock` fields in `block.h`**

In `LogicalBlock` (currently lines 136–168), replace

```cpp
    // Cache slots, one per category
    int prefix_id{0};
    int checkpoint_id{0};
```

with

```cpp
    // Cache slots, one per category; nullptr = not created
    CacheBlock* prefix{};
    CacheBlock* checkpoint{};
```

- [ ] **Step 1.3: Rewrite `block.cc`**

Replace the `CacheBlockPool` member definitions and `LogicalBlockPool::Recycle` cache-slot handling. Full new content of the pool functions:

```cpp
CacheBlock* CacheBlockPool::Create(int object_id, LogicalBlock* owner)
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
    *b           = {};
    b->object_id = object_id;
    b->owner     = owner;
    return b;
}

void CacheBlockPool::Invalidate(CacheBlock* b)
{
    TM_CHECK_GE(b->object_id, 0);  // double-invalidate check
    *b = {};
    free_.push_back(b);
}

void CacheBlock::Deallocate(ObjectAllocator& alloc)
{
    TM_CHECK(valid());
    alloc.Deallocate(object_id, allocation);
    allocation = {};
    alloc_key  = 0;
    timestamp  = 0;
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
```

In `LogicalBlockPool::Recycle`, replace

```cpp
    if (const int c = p->prefix_id) {
        cache_.Invalidate(c);  // allocation already gone (see class comment)
    }
    if (const int c = p->checkpoint_id) {
        cache_.Invalidate(c);
    }
```

with

```cpp
    if (CacheBlock* c = p->prefix) {
        cache_.Invalidate(c);  // allocation already gone (see class comment)
    }
    if (CacheBlock* c = p->checkpoint) {
        cache_.Invalidate(c);
    }
```

Delete the old `CacheBlockPool::Invalidate(int)`, `Create(int, LogicalBlock*) -> int`, `Deallocate(ObjectAllocator&, int)`, `SortedIndices`, and `Stamp(int)` / `Stamp(const std::vector<int>&)` definitions. Add `#include "src/turbomind/memory/object.h"` stays as-is (already included via block.h).

### Task 2: `request.h` — Sequence fields and CacheCopy

**Files:**
- Modify: `src/turbomind/engine/request.h`

- [ ] **Step 2.1: `CacheCopy` holds pointers**

Replace (currently lines 154–157):

```cpp
struct CacheCopy {
    CacheBlock* src{};
    CacheBlock* dst{};
};
```

(keep the existing comment above it). Add `struct CacheBlock;` forward declaration near the top if `block.h` is not already included by `request.h`; check the includes — `request.h` uses `BlockHandle` (`std::vector<BlockHandle> block_ids`), so it already includes `block.h` and no forward declaration is needed.

- [ ] **Step 2.2: Rename Sequence fields**

Replace (currently lines 222–227):

```cpp
    std::vector<CacheBlock*> alloc_blocks;     // cache blocks needing allocation this schedule pass
    std::vector<CacheBlock*> involved_blocks;  // cache blocks stamped for eviction protection (= required alloc set);
                                               // persistent across PlanContinue, rebuilt by PlanResume
```

and (currently line 239):

```cpp
    CacheBlock*   frontier          = nullptr;  // checkpoint working state for the next forward
```

(keep the surrounding fields — `frontier_pos`, `publish_target`, etc. — unchanged, preserving alignment).

### Task 3: `scheduler.h`

**Files:**
- Modify: `src/turbomind/engine/scheduler.h`

- [ ] **Step 3.1: Update the Scheduler class surface**

1. Delete the `cache()` accessor (lines 100–103).
2. In `PublishPlan` (lines 166–170), replace `int cache_id{};` with `CacheBlock* slot{};` and update its comment: `slot is the target's own (block-owned) checkpoint slot; nullptr => nothing.`
3. Replace `void ReleaseCacheId(int cache_id);` with `void ReleaseFrontier(CacheBlock* b);`
4. Delete the `ValidAlloc` member function (lines 216–222) including its comment block — the free `is_valid()` in `block.h` replaces it (the comment about the cached allocation being the validity flag lives on `CacheBlock::valid()` / spec).

### Task 4: `scheduler.cc` — the bulk conversion

**Files:**
- Modify: `src/turbomind/engine/scheduler.cc`

This is a mechanical sweep. Global substitutions (apply everywhere in this file):

| old | new |
|---|---|
| `x.prefix_id` / `->prefix_id` | `x.prefix` / `->prefix` |
| `x.checkpoint_id` / `->checkpoint_id` | `x.checkpoint` / `->checkpoint` |
| `s.frontier_cache_id` | `s.frontier` |
| `s.alloc_cache_ids` | `s.alloc_blocks` |
| `s.involved_cache_ids` | `s.involved_blocks` |
| `ValidAlloc(e)` | `is_valid(e)` |
| `cache_.Deallocate(alloc_, e)` | `e->Deallocate(alloc_)` |
| `cache_.Demote(e)` | `e->Demote()` |
| `cache_.SortedIndices()` | `cache_.SortedBlocks()` |

Then the structural edits, in file order:

- [ ] **Step 4.1: File-local helpers (lines 27–52)**

`ResetPassBuffers`/`ResetPlanBuffers`: field renames only (comment: `involved_cache_ids` → `involved_blocks`). Replay records become:

```cpp
struct AllocReplay {
    CacheBlock* block;
};

struct EvictReplay {
    CacheBlock* block;
};
```

- [ ] **Step 4.2: `EvictingIterator` (lines 54–100) — drop the pool member**

```cpp
class EvictingIterator {
public:
    explicit EvictingIterator(const std::vector<CacheBlock*>& blocks): blocks_{&blocks} {}

    EvictingIterator(std::vector<CacheBlock*>&&) = delete;

    EvictingIterator(const EvictingIterator& base, uint64_t cutoff):
        blocks_{base.blocks_}, pos_{base.pos_}, cutoff_{cutoff}
    {
    }

    EvictingIterator(const EvictingIterator&) noexcept = default;
    EvictingIterator& operator=(const EvictingIterator&) noexcept = default;

    explicit operator bool() const noexcept
    {
        return pos_ < blocks_->size() && (*blocks_)[pos_]->timestamp < cutoff_;
    }

    uint64_t Evict(ScratchAllocator& scratch, Replay& replay)
    {
        CacheBlock* b = (*blocks_)[pos_++];
        scratch.Evict(b->object_id, b->allocation.a);
        replay.push_back(EvictReplay{b});
        return b->timestamp;
    }

    size_t pos() const noexcept
    {
        return pos_;
    }

    void SeekTo(size_t pos) noexcept
    {
        pos_ = pos;
    }

private:
    const std::vector<CacheBlock*>* blocks_;
    size_t                          pos_{};
    uint64_t                        cutoff_{std::numeric_limits<uint64_t>::max()};
};
```

- [ ] **Step 4.3: `AllocatingIterator` (lines 102–144) — drop the pool member**

```cpp
class AllocatingIterator {
public:
    explicit AllocatingIterator(const std::vector<CacheBlock*>& blocks): iter_{blocks.begin()}, end_{blocks.end()} {}

    AllocatingIterator(std::vector<CacheBlock*>&&) = delete;

    AllocatingIterator(const AllocatingIterator&) = delete;
    AllocatingIterator& operator=(const AllocatingIterator&) = delete;

    explicit operator bool() const noexcept
    {
        return iter_ != end_;
    }

    // Idempotent: blocks already allocated for real (cached alloc set), or
    // planned by an earlier request in this pass, are skipped.
    bool Allocate(ScratchAllocator&               scratch,
                  std::unordered_set<CacheBlock*>& planned,
                  std::vector<CacheBlock*>&        planned_now,
                  Replay&                          replay)
    {
        CacheBlock* b = *iter_;
        if (b->valid() || planned.count(b)) {
            ++iter_;
            return true;
        }
        if (scratch.Allocate(b->object_id)) {
            ++iter_;
            planned.insert(b);
            planned_now.push_back(b);
            replay.push_back(AllocReplay{b});
            return true;
        }
        return false;
    }

private:
    std::vector<CacheBlock*>::const_iterator iter_;
    std::vector<CacheBlock*>::const_iterator end_;
};
```

- [ ] **Step 4.4: `ResumeCandidate` (lines 194–200)**

`int ckpt_id{};` → `CacheBlock* ckpt{};` with comment `// checkpoint to restore into the frontier; nullptr = none`. Update its two writers (Step 4.8) and reader (Step 4.8 item 3).

- [ ] **Step 4.5: `ScheduleState` (lines 254–268)**

```cpp
    std::vector<CacheBlock*>        evict_blocks;  // SortedBlocks() snapshot, shared by both phases
    size_t                          evict_pos{0};  // oldest-first eviction cursor shared by both phases
    std::unordered_set<CacheBlock*> planned;       // cache blocks planned/reserved for allocation
```

(other members unchanged).

- [ ] **Step 4.6: `~Scheduler` drain (lines 327–332)**

```cpp
    for (CacheBlock* b : cache_.SortedBlocks()) {
        b->Deallocate(alloc_);
        if (LogicalBlock* o = b->owner) {
            logical_.Drop(o);
        }
    }
```

Update the comment above: `SortedIndices()` → `SortedBlocks()`.

- [ ] **Step 4.7: Creation sites**

`EnsureBlocks` (line 343): `h->prefix = cache_.Create(registry_.prefix().object_id(), h.get());  // owner = node`
`IndexMissingBlocks` (line 424) and `SetupPartialSiblings` (line 504): same shape, `x.prefix = ...` / `y.prefix = ...`.

- [ ] **Step 4.8: `PlanResume` (lines 523–668)**

1. Frontier creation (line 537): `if (ckpt && s.frontier == nullptr) { s.frontier = cache_.Create(registry_.checkpoint().object_id()); ... }`
2. Candidate writers (lines 581, 615, 621): `best = {e, ResumeSource::kFork, ckpt ? y->checkpoint : nullptr, y, &x};`, `best = {e, ResumeSource::kCheckpoint, x.checkpoint};`, `best = {ye, ResumeSource::kFork, y->checkpoint};`
3. Restore plans (lines 633–643): comment becomes `// 3. Restore copy plans (cache blocks; resolved to addresses at setup)`; `s.restore_copies.push_back({best.fork_src->prefix, best.fork_dst->prefix});` and `if (ckpt && best.pos > 0 && best.ckpt) { s.restore_copies.push_back({best.ckpt, s.frontier}); ... }`
4. Section 4 (lines 655–667): push `x.prefix` / `s.frontier` into `s.involved_blocks` / `s.alloc_blocks` guarded by `is_valid(...)`.

- [ ] **Step 4.9: `PlanContinue` (lines 690–700)**

```cpp
    for (int i = first_new; i < static_cast<int>(s.block_ids.size()); ++i) {
        CacheBlock* p = s.block_ids[i]->prefix;
        s.involved_blocks.push_back(p);
        s.alloc_blocks.push_back(p);
    }
```

and the frontier check `TM_CHECK(is_valid(s.frontier));`. Update the two comments mentioning `involved_cache_ids`.

- [ ] **Step 4.10: `ReleaseCacheId` → `ReleaseFrontier` (lines 760–773)**

```cpp
void Scheduler::ReleaseFrontier(CacheBlock* b)
{
    if (b == nullptr) {
        return;
    }
    if (b->object_id >= 0) {
        TM_CHECK(b->owner == nullptr);  // sequence-owned slots only (frontier / adopted zombie)
        if (b->valid()) {
            b->Deallocate(alloc_);
        }
        cache_.Invalidate(b);
    }
}
```

- [ ] **Step 4.11: `Release` (lines 775–807)**

Private-block loop: `for (CacheBlock* c : {x.prefix, x.checkpoint}) { if (is_valid(c)) { c->Deallocate(alloc_); logical_.Drop(&x); } }`
Frontier: `ReleaseFrontier(std::exchange(s.frontier, nullptr));`
Buffer clears: `s.alloc_blocks.clear(); s.involved_blocks.clear();`

- [ ] **Step 4.12: `Finalize` zombie swap (lines 892–937)**

```cpp
        if (publish_generation_boundary && x.offset + size == s.filled_len && is_valid(s.frontier)
            && !is_valid(x.checkpoint)) {
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
```

The demote loop below: `is_valid(p.checkpoint)` / `is_valid(y->checkpoint)` and `f->Demote();`.

- [ ] **Step 4.13: `PlanPublication` (lines 946–1003)**

Population branch: `sibling->prefix` in the `is_valid` / `pass.planned` tests and inserts. Checkpoint branch:

```cpp
    if (!is_valid(node->checkpoint)) {
        if (node->checkpoint == nullptr) {
            node->checkpoint = cache_.Create(registry_.checkpoint().object_id(), node);
        }
        TM_CHECK(pass.planned.insert(node->checkpoint).second);
        pass.pending_publish[i] = {node, end, node->checkpoint};
        pass.has_optionals      = true;
    }
```

Update the comment `exactly like prefix_id` → `exactly like the prefix slot`, and `required id` → `required slot` in the (a) comment.

- [ ] **Step 4.14: `PlanRequests` (lines 1099–1115)**

`cache_.Stamp(s.involved_blocks)` and the restore-source loop stays `cache_.Stamp(c.src);` (now a pointer).

- [ ] **Step 4.15: `RunRequiredAdmission` (lines 1122–1234)**

`pass.evict_blocks = cache_.SortedBlocks();`, `EvictingIterator evict_pos{pass.evict_blocks};`, `AllocatingIterator allocating{s.alloc_blocks};`, `std::vector<CacheBlock*> planned_now;`, rollback loop `for (CacheBlock* b : planned_now) { pass.planned.erase(b); }`.

- [ ] **Step 4.16: `RunOptionalAdmission` (lines 1236–1299)**

`EvictingIterator base{pass.evict_blocks};` and:

```cpp
    auto try_optional = [&](CacheBlock* b) -> bool {
        if (b->valid()) {
            return true;
        }
        bool ok = opt.Allocate(b->object_id);
        while (!ok && evicting) {
            evicting.Evict(opt, pass.replay);
            ok = opt.Allocate(b->object_id);
        }
        if (!ok) {
            return false;
        }
        pass.planned.insert(b);
        pass.replay.push_back(AllocReplay{b});
        return true;
    };
```

Call sites: `try_optional(node->prefix)`, and `if (const PublishPlan& pub = pass.pending_publish[i]; pub.slot) { if (try_optional(pub.slot)) { ... } }`. Update the comment `A partial sibling id reserved` → `A partial sibling slot reserved`.

- [ ] **Step 4.17: `ReplayMemory` (lines 1301–1330)**

```cpp
    for (const auto& op : pass.replay) {
        std::visit(
            [&](const auto& item) {
                using T       = std::decay_t<decltype(item)>;
                CacheBlock& c = *item.block;
                if constexpr (std::is_same_v<T, EvictReplay>) {
                    const bool is_prefix = c.object_id == registry_.prefix().object_id_or_negative();
                    c.Deallocate(alloc_);  // clears allocation; owner persists
                    if (LogicalBlock* o = c.owner) {
                        if (is_prefix) {
                            o->is_valid = false;
                        }
                        logical_.Drop(o);  // may recycle the block and free this slot
                    }
                }
                else {
                    c.allocation = alloc_.Allocate(c.object_id);  // single-object; {nullptr} on OOM
                    TM_CHECK(c.allocation.a);                     // admission guarantees capacity
                    c.alloc_key = c.allocation->key;              // snapshot for stale detection
                    logical_.Retain(c.owner);                     // no-op when owner == nullptr (sequence-owned)
                }
            },
            op);
    }
```

- [ ] **Step 4.18: `CommitResults` (lines 1332–1408)**

Uncommitted reset: `s.alloc_blocks.clear();`. Populate branch: `s.publish_copies.push_back({s.block_ids[(end - 1) / bs]->prefix, y.prefix}); cache_.Stamp(y.prefix);`. Publish branch:

```cpp
            CacheBlock* slot = s.publish_target->checkpoint;
            TM_CHECK(is_valid(slot));
            s.last_ckpt_pos = s.publish_end;
            ckpt_published  = true;
            s.publish_copies.push_back({s.frontier, slot});
            // Allocated outside the stamped involved sets: stamp now so
            // the fresh checkpoint is not the top eviction candidate.
            cache_.Stamp(slot);
```

### Task 5: `engine.cc` — setup resolution, drop the pool env entry

**Files:**
- Modify: `src/turbomind/engine/engine.cc:613-642`

- [ ] **Step 5.1: Resolve copies through the pointers**

```cpp
        const ObjectAllocator& alloc   = scheduler_.allocator();
        auto                   resolve = [&](std::vector<CacheCopy>& in, std::vector<ResolvedCopy>& out) {
            for (const auto& [src, dst] : in) {
                TM_CHECK_NOTNULL(src->allocation.a);  // validity (resolved allocation) on both ends
                TM_CHECK_NOTNULL(dst->allocation.a);
                TM_CHECK_EQ(src->object_id, dst->object_id);        // same object => same part layout
                TM_CHECK_EQ(src->part_count(), dst->part_count());  // both replay-populated to the same layout
                TM_CHECK_EQ(src->part_count(), alloc.PartCount(src->object_id));
                for (int p = 0; p < src->part_count(); ++p) {
                    out.push_back({src->base(p), dst->base(p), alloc.PartBytes(src->object_id, p)});
                }
            }
            in.clear();
        };
```

- [ ] **Step 5.2: Drop the env entry**

Delete lines 637 (`const CacheBlockPool* cache_block_pool = ...`) and the `{"cache_block_pool", ...}` entry from the `TensorMap` (line 642), leaving:

```cpp
    TensorMap env{{"requests", rs}, {"batch", d.buf()}, {"copy", copy.buf()}};
```

### Task 6: `unified_attention_layer.cc`

**Files:**
- Modify: `src/turbomind/models/llama/unified_attention_layer.cc:294-310`

- [ ] **Step 6.1: Dereference blocks directly**

Delete the `c_pool` line and rewrite the loop body:

```cpp
    {  /// Upload KV cache ptrs
        auto blocks  = block_ptrs_buf_.data();
        auto offsets = block_ptrs_offsets_buf_.data();

        offsets[0] = 0;
        for (int i = 0; i < rc.size(); ++i) {
            const auto& r = *rc[i];
            for (const auto& h : r.block_ids) {
                const CacheBlock& cb = *h->prefix;
                TM_CHECK_NOTNULL(cb.allocation.a);
                *blocks++ = cb.base(0) + prefix_cache_offset_;
            }
            offsets[i + 1] = offsets[i] + r.block_ids.size();
        }
```

Also remove the now-unused `CacheBlockPool` include if this file includes `block.h` only for the pool type (it still needs `block.h` for `CacheBlock`/`BlockHandle` — keep the include).

### Task 7: `GatedDeltaNetLayer.cc`

**Files:**
- Modify: `src/turbomind/models/llama/GatedDeltaNetLayer.cc:158-167`

- [ ] **Step 7.1: Dereference the frontier directly**

Delete the `c_pool` line (158) and replace lines 164–165:

```cpp
        const CacheBlock& cb = *TM_CHECK_NOTNULL(s.frontier);
        TM_CHECK_NOTNULL(cb.allocation.a);
```

(rest of the loop reads `cb.base(...)` unchanged).

- [ ] **Step 7.2: Build**

Run from `build/`: `ninja`
Expected: full build success, zero warnings introduced. If configure is missing, run `sh ../my_generate.sh` first. Fix any residual compile errors (they will be missed rename sites; apply the Task 4 substitution table).

- [ ] **Step 7.3: Sweep for stragglers**

Run: `rg -n "prefix_id|checkpoint_id|frontier_cache_id|alloc_cache_ids|involved_cache_ids|ValidAlloc|SortedIndices|ReleaseCacheId|cache_block_pool" src/`
Expected: no matches (comments included — update any comment stragglers).

- [ ] **Step 7.4: Commit**

```bash
git add src/turbomind/engine/block.h src/turbomind/engine/block.cc src/turbomind/engine/request.h \
        src/turbomind/engine/scheduler.h src/turbomind/engine/scheduler.cc src/turbomind/engine/engine.cc \
        src/turbomind/models/llama/unified_attention_layer.cc src/turbomind/models/llama/GatedDeltaNetLayer.cc
git commit -m "refactor: refer to CacheBlock by pointer instead of pool index"
```

### Task 8: README contract terminology

**Files:**
- Modify: `src/turbomind/engine/README.md`

- [ ] **Step 8.1: Update handle terminology**

Terminology-only edits — do NOT re-wrap lines, do NOT change any normative rule. In each of these leaves, replace wording that names the handle as an id/index with the pointer/slot wording, and update renamed symbols:

- `concepts.scheduler-transaction`: "reserves cache ids" → "reserves cache block slots".
- `ownership.object-allocator`: "Logical blocks point to cache ids, not raw memory." → "Logical blocks point to `CacheBlock` slots, not raw memory."
- `ownership.prefix`: "Cache ids follow a strict ownership model: every id is owned by …" → "Cache block slots follow a strict ownership model: every slot is owned by …"; `prefix_id`/`checkpoint_id` → `prefix`/`checkpoint`; `frontier_cache_id` → `frontier`; "an id is invalidated (returned to the pool free list) only when its owner is destroyed" → "a slot is invalidated (returned to the pool free list) only when its owner is destroyed"; "nothing else invalidates a cache id" → "nothing else invalidates a slot"; "never frees the id" → "never frees the slot".
- `principles.scheduler-boundary` and `contracts.cache-prepare`: "reserve cache ids" → "reserve cache block slots"; "emit restore copy intent as cache-id pairs" → "as `CacheBlock*` pairs"; `involved_cache_ids` → `involved_blocks` (both occurrences in `cache-prepare` and in `invariants.protection-set`, `checklist.cache-prepare`).
- `principles.device-content` and `contracts.scheduler-commit`: "`(src, dst)` cache-id pairs" → "`(src, dst)` cache-block pairs"; "same `(src_id, dst_id)` cache-id plan" → "same `(src, dst)` cache-block plan"; "stamps each request's `involved_cache_ids`" → "`involved_blocks`".
- `contracts.cache-metadata`: "CacheBlockPool records cache ids, object ids, …" → "`CacheBlockPool` owns `CacheBlock` slot storage (stable addresses); each slot records its object id, allocation handle, timestamp, and a weak `owner` back-reference …"; "LogicalBlock records which cache ids are attached" → "which cache slots are attached".
- `contracts.checkpoint-publish`: "no request-owned publication id exists" → "no request-owned publication slot exists"; "reservation of the slot id at plan time" → "reservation of the slot at plan time"; "checkpoint cache ids exist" → "checkpoint cache slots exist".
- `contracts.resume-selection` / `contracts.checkpoint-adoption` / `contracts.cache-eviction`: "frontier id" → "frontier slot"; "transfers the frontier cache id" → "transfers the frontier cache slot"; "ids are invalidated only by owner destruction" → "slots are invalidated only by owner destruction".

Search check afterwards: `rg -n "cache id|cache-id|cache ids|prefix_id|checkpoint_id|frontier_cache_id|involved_cache_ids" src/turbomind/engine/README.md`
Expected: no matches.

- [ ] **Step 8.2: Commit**

```bash
git add src/turbomind/engine/README.md
git commit -m "docs: cache handle terminology follows CacheBlock pointer refactor"
```

### Task 9: Verification

- [ ] **Step 9.1: Check GPU availability**

Use `get_gpu_usage` to find an empty GPU before running anything on GPU. GPU commands must run outside the sandbox.

- [ ] **Step 9.2: Model test**

Pick a model from `/data/models.json` (set `hf_constants.HF_HUB_OFFLINE`/`HF_HUB_CACHE` per workspace rules — the test script may already handle this; use the script AS IS, do not modify it). Run `scripts/test_turbomind_model.py` with a normal prompt and a requested response length of at least 128 tokens.

Expected: the model responds with meaningful human words relevant to the prompt. Verify the response text — gibberish means the refactor broke pointer resolution (most likely a missed rename in a module setup path). If OOM occurs, check whether the GPU is occupied by another process before debugging.

- [ ] **Step 9.3: Prefix-reuse smoke (same behavior as before)**

Run the same test script a second time with the same prompt (prefix caching path) and verify the response is still meaningful. This exercises `SortedBlocks`, eviction stamps, and restore copies with the pointer representation.

---

## Self-Review Notes

- Spec coverage: pool rewrite (T1), field renames (T2), scheduler surface (T3), scheduler body incl. all simplifications (T4), engine resolution + env-entry removal (T5), modules (T6, T7), README (T8), verification (T9). `test_memory.cc` mentions `CacheBlock` only in a comment — no change needed (spec's "mechanical conversion" note is vacuous; confirmed by inspection).
- Type consistency: `is_valid(const CacheBlock*)` (T1) used in T4; `CacheBlock::Deallocate(ObjectAllocator&)`/`Demote()` (T1) used in T4; `PublishPlan::slot` (T3) used in T4.16; `SortedBlocks()` (T1) used in T4.6/T4.15; `evict_blocks`/`planned` types (T4.5) match iterator signatures (T4.2/T4.3).
