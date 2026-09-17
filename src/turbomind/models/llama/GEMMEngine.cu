// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/GEMMEngine.h"

#include "src/turbomind/core/context.h"

#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

GEMMEngine::GEMMEngine(): workspace_{core::Context::stream().handle()} {}

GEMMEngine::~GEMMEngine()
{
    workspace_.Release(core::Context::stream().handle());
}

void GEMMEngine::set_measure(bool measure)
{
    dispatch_policy_ = measure ? gemm::DispatchPolicy::kAppend : gemm::DispatchPolicy::kReuse;
}

int GEMMEngine::Export(std::ostream& os)
{
    if (os) {
        return gemm_.Export(os);
    }
    return 0;
}

int GEMMEngine::Import(std::istream& is)
{
    auto n_records = 0;
    if (is) {
        n_records = gemm_.Import(is);
    }
    if (n_records) {
        dispatch_policy_ = gemm::DispatchPolicy::kReuse;
    }
    return n_records;
}

std::vector<int> GEMMEngine::GetTuningSeq() const
{
    return gemm_.GetTuningSeq();
}

}  // namespace turbomind
