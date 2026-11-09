// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/kernels/gemm/thread_map.h"

#include <iostream>

namespace turbomind::gemm {

template<int M_, int N_, int K_, int tM_, int tN_, int tK_, class ArrangementMN, int gK, bool rK = 0>
struct MMA_Map {
    static constexpr int M = M_;
    static constexpr int N = N_;
    static constexpr int K = K_;

    static constexpr int TileM = tM_;
    static constexpr int TileN = tN_;
    static constexpr int TileK = tK_;

    static constexpr int kGroupM = ArrangementMN::gM;
    static constexpr int kGroupN = ArrangementMN::gN;
    static constexpr int kGroupK = gK;

    static constexpr int kGroupCount = kGroupM * kGroupN * kGroupK;

    static constexpr int kIterM = M / tM_ / kGroupM;
    static constexpr int kIterN = N / tN_ / kGroupN;
    static constexpr int kIterK = K / tK_ / kGroupK;

    static constexpr int kFootprintM = kIterM * tM_;
    static constexpr int kFootprintN = kIterN * tN_;
    static constexpr int kFootprintK = kIterK * tK_;

    static constexpr int kDeltaM = tM_ * ArrangementMN::dM;
    static constexpr int kDeltaN = tN_ * ArrangementMN::dN;
    static constexpr int kDeltaK = tK_ * (rK ? gK : 1);

    static constexpr auto kPartitionM = ArrangementMN::pM;
    static constexpr auto kPartitionN = ArrangementMN::pN;
    static constexpr auto kPartitionK = rK ? Partition::kRaked : Partition::kBlocked;

    __device__ static int3 get_offset(int group_id)
    {
        constexpr int kGroupMN = kGroupM * kGroupN;

        const auto mn = ArrangementMN::get_offset(group_id % kGroupMN, pair<M / TileM, N / TileN>{});
        const int  k  = group_id / kGroupMN;

        return {mn.x * tM_, mn.y * tN_, k * tK_ * (rK ? 1 : kIterK)};
    }
};

namespace {

template<class TMap>
void Print_(TMap)
{
    std::cout << "M, N, K = " << TMap::M << " " << TMap::N << " " << TMap::K << "\n";
    std::cout << "TM, TN, TK = " << TMap::TileM << " " << TMap::TileN << " " << TMap::TileK << "\n";
    std::cout << "group count = " << TMap::kGroupCount << "\n";
    // std::cout << "M1, N1, K1 = " << TMap::M1 << " " << TMap::N1 << " " << TMap::K1 << "\n";
    std::cout << "itM, itN, itK = " << TMap::kIterM << " " << TMap::kIterN << " " << TMap::kIterK << "\n";
    std::cout << "fpM, fpN, fpK = " << TMap::kFootprintM << " " << TMap::kFootprintN << " " << TMap::kFootprintK
              << "\n";
    std::cout << "dM, dN, dK = " << TMap::kDeltaM << " " << TMap::kDeltaN << " " << TMap::kDeltaK << "\n";
}

}  // namespace

/// TODO: Striped partition?

}  // namespace turbomind::gemm
