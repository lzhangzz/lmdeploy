// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/core/module_config.h"

namespace turbomind {

class NormWeight: public core::Module {
public:
    const char* type() const override { return "NormWeight"; }

    NormWeight() = default;

    explicit NormWeight(const core::NormConfig& cfg);

    NormWeight(int dim, DataType dtype, DeviceType device = kDEVICE);

    /// Construct with arbitrary shape.
    NormWeight(std::vector<ssize_t> shape, DataType dtype, DeviceType device = kDEVICE);

    /// Configure dimensions (deferred — no allocation).
    void configure(int dim, DataType dtype);

    /// Configure with arbitrary shape (deferred — no allocation).
    void configure(std::vector<ssize_t> shape, DataType dtype);

    /// Post-load: cast weight to configured dtype if needed.
    void prepare() override;

#define NORM_WEIGHT_CHILDREN(X)

#define NORM_WEIGHT_PARAMS(X) \
    X(weight)

    TM_MODULE_DECLARE(NormWeight, NORM_WEIGHT_CHILDREN, NORM_WEIGHT_PARAMS)

private:
    std::vector<ssize_t> shape_;
    DataType              dtype_{};
};

}  // namespace turbomind
