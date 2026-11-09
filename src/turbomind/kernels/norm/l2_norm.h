// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cuda_runtime.h>

#include "src/turbomind/core/core.h"

namespace turbomind {

void invokeL2NormalizeQK(core::Tensor& q, core::Tensor& k, float epsilon, cudaStream_t stream);

}  // namespace turbomind
