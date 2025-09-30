// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

class BatchedCopy {
public:
    template<class T>
    T* Add(const T* src, int size, T* dst)
    {
        src_.push_back((void*)src);
        dst_.push_back((void*)dst);
        size_.push_back(sizeof(T) * size);
        return dst + size;
    }

    template<class T>
    T* operator()(const T* src, int size, T* dst)
    {
        return Add(src, size, dst);
    }

    void Submit(cudaStream_t stream)
    {
        if (size_.empty()) {
            return;
        }

        invokeBatchedCopy(src_.data(), dst_.data(), size_.data(), size_.size(), stream);
        sync_check_cuda_error();

        src_.clear();
        dst_.clear();
        size_.clear();
    }

    void Launch(cudaStream_t stream) {
        Submit(stream);
    }

private:
    std::vector<void*> src_;
    std::vector<void*> dst_;
    std::vector<int>   size_;
};

}  // namespace turbomind
