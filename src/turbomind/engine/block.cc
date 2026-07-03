#include "src/turbomind/engine/block.h"

#include <algorithm>
#include <utility>

namespace turbomind {

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

LogicalBlockPool::~LogicalBlockPool()
{
    if (live_ != 0) {
        TM_LOG_ERROR("leaked {} logical blocks", live_);
    }
}

BlockHandle LogicalBlockPool::Create(int logical_index)
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
    return BlockHandle{p};  // refs 0 -> 1
}

void LogicalBlockPool::Recycle(LogicalBlock* p)
{
    if (on_recycle_) {
        on_recycle_(*p);  // PrefixTrie::Erase (pool stays prefix-agnostic)
    }
    if (CacheBlock* c = p->prefix) {
        cache_.Invalidate(c);  // allocation already gone (see class comment)
    }
    if (CacheBlock* c = p->checkpoint) {
        cache_.Invalidate(c);
    }
    *p = LogicalBlock{};  // drops fork edge, frees tokens (was destroy+deallocate)
    free_.push_back(p);
    --live_;
}

}  // namespace turbomind
