// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <istream>
#include <optional>
#include <ostream>

#include "src/turbomind/core/core.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/models/linear.h"

namespace turbomind {

/// Standalone executor for the Python torch-linear path (linear.py). Owns a
/// private GEMMEngine and runs plan query / replay / tuning over data-only
/// Linear modules. Retires with the stage-1 Python delta, when linear.py
/// constructs a Context and binds the modules instead.
class LlamaLinear {
public:
    LlamaLinear();

    /// Plan-replay forward over a data-only module.
    void Forward(const gemm::ExecPlan& plan,
                 const Tensor&         input,
                 const Tensor&         input_scales,
                 const Linear&         weight,
                 const Buffer_<int>&   indices,
                 const Buffer_<int>&   offsets,
                 Ref<Tensor>           output,
                 Ref<Tensor>           output_scales);

    gemm::OutputSpec GetOutputSpec(const Tensor& input, const Linear& weight, const Buffer_<int>& indices = {}) const;

    std::optional<gemm::ExecPlan> GetExecPlan(const Tensor&       input,
                                              const Linear&       weight,
                                              const Buffer_<int>& indices = {},
                                              const Buffer_<int>& offsets = {});

    std::optional<gemm::ExecPlan> Tune(const Tensor&       input,
                                       const Tensor&       input_scales,
                                       const Linear&       weight,
                                       const Buffer_<int>& indices,
                                       const Buffer_<int>& offsets,
                                       Ref<Tensor>         output,
                                       Ref<Tensor>         output_scales);

    gemm::Gemm& gemm() noexcept;

    [[maybe_unused]] int Export(std::ostream& os);

    [[maybe_unused]] int Import(std::istream& is);

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace turbomind
