// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#include "src/turbomind/core/core.h"
#include "src/turbomind/kernels/core/data_type.h"

namespace turbomind {

void extend_to_u16(uint16_t* dst, const uint4_t* src, size_t n, cudaStream_t st = {});

void extend_to_u16(uint16_t* dst, const uint8_t* src, size_t n, cudaStream_t st);

void fuse_scales_and_zeros(half* fused, const half* scales, half* zeros, size_t n, cudaStream_t st = {});

void AdjustUe8m0ScaleForHalf(uint8_t* data, int n, cudaStream_t st);

}  // namespace turbomind
