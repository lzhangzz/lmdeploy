// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cuda_runtime.h>

#include "src/turbomind/core/core.h"

namespace turbomind {

void ComputeBetaG(core::Tensor&       beta,
                  core::Tensor&       g,
                  const core::Tensor& b,
                  const core::Tensor& a,
                  const core::Tensor& A_log,
                  const core::Tensor& dt_bias,
                  cudaStream_t        stream);

}  // namespace turbomind
