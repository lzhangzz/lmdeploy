#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <memory_resource>
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

// Intrusive, non-atomic strong handle to a logical block (engine-thread only).
// Holds one ref for its lifetime; copy retains, destruction drops.
class BlockHandle {
    LogicalBlock* p_{};

public:
    BlockHandle() = default;
    explicit BlockHandle(LogicalBlock* p);
    BlockHandle(const BlockHandle& o);
    BlockHandle(BlockHandle&& o) noexcept: p_{std::exchange(o.p_, nullptr)} {}
    BlockHandle& operator=(BlockHandle o) noexcept
    {
        std::swap(p_, o.p_);
        return *this;
    }
    ~BlockHandle();

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
    friend bool operator==(const BlockHandle& a, const BlockHandle& b) noexcept
    {
        return a.p_ == b.p_;
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

struct LogicalBlock {
    // Position and content extent within the sequence
    int offset{-1};
    int capacity{0};
    int size{0};  // filled tokens of an indexed node; 0 for private blocks

    // Intrusive strong refcount (requests + partial sibling edges + valid allocations)
    int               refs{0};
    LogicalBlockPool* mgr{};  // set at Create; used by handle / Retain / Drop

    // Cache slots, one per category; nullptr = not created
    CacheBlock* prefix{};
    CacheBlock* checkpoint{};

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
    BlockHandle partial;

    bool     is_valid{false};  // content proven produced; cleared on prefix evict
    uint64_t producer{0};      // request currently writing this range; 0 = none
};

// Owns logical block lifetime via an intrusive refcount. Nodes are allocated
// discretely from a pooled memory resource, so a LogicalBlock* is a stable
// identity. When refs reaches 0 the node is recycled: a recycle hook removes
// it from the PrefixTrie index, every attached cache slot's allocation is
// already invalid (a valid allocation holds a ref via CacheBlock::owner), so
// Invalidate only returns slot metadata.
class LogicalBlockPool {
    using NodeAlloc = std::pmr::polymorphic_allocator<LogicalBlock>;

public:
    LogicalBlockPool(CacheBlockPool& cache, int block_size = 0): block_size_{block_size}, cache_{cache} {}

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

    BlockHandle Create(int logical_index);

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

    CacheBlockPool& cache_;

    std::pmr::unsynchronized_pool_resource res_;
    NodeAlloc                              alloc_{&res_};
    std::function<void(LogicalBlock&)>     on_recycle_;
};

inline BlockHandle::BlockHandle(LogicalBlock* p): p_{p}
{
    if (p_) {
        p_->mgr->Retain(p_);
    }
}

inline BlockHandle::BlockHandle(const BlockHandle& o): p_{o.p_}
{
    if (p_) {
        p_->mgr->Retain(p_);
    }
}

inline BlockHandle::~BlockHandle()
{
    if (p_) {
        p_->mgr->Drop(p_);
    }
}

inline LogicalBlock& BlockHandle::operator*() const noexcept
{
    return *p_;
}

}  // namespace turbomind
