// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/core/buffer.h"

namespace turbomind::core {

class BatchCopyV2 {
public:
    ~BatchCopyV2();

    BatchCopyV2();

    BatchCopyV2(const BatchCopyV2&)                = delete;
    BatchCopyV2& operator=(const BatchCopyV2&)     = delete;
    BatchCopyV2(BatchCopyV2&&) noexcept            = delete;
    BatchCopyV2& operator=(BatchCopyV2&&) noexcept = delete;

    template<class T>
    T* operator()(const T* src, size_t size, T* dst)
    {
        // return core::Copy(src, size, dst);

        if (TM_LIKELY((T*)prev_dst_last_ == dst && (const T*)prev_src_last_ == src)) {
            size_.back() += sizeof(T) * size;
        }
        else {
            src_.push_back((const char*)src);
            dst_.push_back((char*)dst);
            size_.push_back(sizeof(T) * size);
        }
        count_ += 1;
        prev_src_last_ = reinterpret_cast<const char*>(src + size);
        prev_dst_last_ = reinterpret_cast<char*>(dst + size);
        return dst + size;
    }

    void operator()(const Buffer& src, size_t size, Ref<Buffer> dst_)
    {
        auto& dst = dst_.get();
        TM_CHECK_EQ(src.dtype(), dst.dtype());
        TM_CHECK_LE(size, src.size());
        TM_CHECK_LE(size, dst.size());
        (*this)((const char*)src.raw_data(), byte_size(src.dtype(), size), (char*)dst.raw_data());
    }

    void Run();

    Buffer_<BatchCopyV2*> buf()
    {
        return {&self_, 1, kCPU};
    }

    friend std::ostream& operator<<(std::ostream& os, const BatchCopyV2& a)
    {
        os << "(" << a.count_ << ", " << a.src_.size() << ")";
        return os;
    }

private:
    void Reset()
    {
        src_.clear();
        dst_.clear();
        size_.clear();
        prev_src_last_ = {};
        prev_dst_last_ = {};
        count_         = 0;
    }

private:
    std::vector<const char*> src_;
    std::vector<char*>       dst_;
    std::vector<size_t>      size_;
    const char*              prev_src_last_;
    char*                    prev_dst_last_;
    size_t                   count_;
    BatchCopyV2*             self_;
};

}  // namespace turbomind::core
