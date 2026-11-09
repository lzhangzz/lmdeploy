// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

TM_HOST_DEVICE constexpr int get_log_tile(int size, int tile_size)
{
    if (tile_size >= 32 && size >= 24)
        return 5;
    if (tile_size >= 16 && size >= 12)
        return 4;
    if (tile_size >= 8 && size >= 6)
        return 3;
    if (tile_size >= 4 && size >= 3)
        return 2;
    if (tile_size >= 2 && size >= 2)
        return 1;
    return 0;
}

TM_HOST_DEVICE constexpr int2 get_tiled_shape(int m, int n, int cta_m, int cta_n)
{
    return {ceil_div(m, cta_m), ceil_div(n, cta_n)};
}

struct CtaMap_ {

    TM_HOST_DEVICE static int3 get_tiled_shape(int m, int n, int k, int cta_m, int cta_n, int split_cnt)
    {
        return {(m + cta_m - 1) / cta_m, (n + cta_n - 1) / cta_n, split_cnt};
    }

    TM_HOST_DEVICE static int get_log_tile(int2 tiled_mn, int N)
    {
        return gemm::get_log_tile(tiled_mn.y, N);
    }

    TM_HOST_DEVICE static dim3 get_grid_shape(int3 tiled_shape, int log_tile)
    {
        int tile = 1 << log_tile;
        return {static_cast<unsigned>(tiled_shape.x * tile),
                static_cast<unsigned>((tiled_shape.y + tile - 1) / tile),
                static_cast<unsigned>(tiled_shape.z)};
    }
};
}  // namespace turbomind::gemm
