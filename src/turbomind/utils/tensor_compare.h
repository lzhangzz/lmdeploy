// Copyright (c) OpenMMLab. All rights reserved.

#pragma once
#include <cuda_runtime.h>
#include <sstream>
#include <string>
#include <vector>

namespace turbomind {

enum CmpMode
{
    kCmpNone,
    kCmpRead,
    kCmpWrite,
};

extern CmpMode compare_mode;

template<typename T>
void Compare(T* ptr, size_t size, std::string key, CmpMode mode, cudaStream_t stream);

template<typename T>
void CheckNan(const T* ptr, size_t size, std::string key, cudaStream_t stream);

}  // namespace turbomind
