// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <istream>
#include <ostream>
#include <vector>

#include "src/turbomind/kernels/gemm/gemm.h"

namespace turbomind {

// Per-rank GEMM engine: engine object, dispatch cache, workspace and measure
// policy. Owned by the engine-root Context; delegates and modules capture it
// by reference and never copy it.
class GEMMEngine {
public:
    GEMMEngine();

    ~GEMMEngine();

    GEMMEngine(const GEMMEngine&) = delete;
    GEMMEngine& operator=(const GEMMEngine&) = delete;

    gemm::Gemm& gemm() noexcept
    {
        return gemm_;
    }

    gemm::Workspace* workspace() noexcept
    {
        return &workspace_;
    }

    gemm::DispatchPolicy dispatch_policy() const noexcept
    {
        return dispatch_policy_;
    }

    // Warm-up / tuning surface. Dispatch-cache mutation stays executor-thread-only;
    // measure mode is only used during warm-up.
    void set_measure(bool measure);

    int Export(std::ostream& os);

    int Import(std::istream& is);

    [[nodiscard]] std::vector<int> GetTuningSeq() const;

private:
    gemm::Gemm           gemm_;
    gemm::DispatchPolicy dispatch_policy_{gemm::DispatchPolicy::kDefault};

    gemm::Workspace workspace_;
};

}  // namespace turbomind
