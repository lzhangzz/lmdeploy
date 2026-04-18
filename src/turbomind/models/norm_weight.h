// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"

namespace turbomind::core {

struct NormConfig: ModuleConfig {
    NormConfig(): ModuleConfig{"NormWeight"} {}

    #define NORM_FIELDS(X) \
        X(int,      dim) \
        X(DataType, data_type) \
        X(float,    norm_eps, 0.f)

    NORM_FIELDS(TM_MEMBER)
    TM_FOR_EACH(NormConfig, NORM_FIELDS)

    #undef NORM_FIELDS
};

}  // namespace turbomind::core

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

    float norm_eps_{};

private:
    std::vector<ssize_t> shape_;
    DataType              dtype_{};
};

}  // namespace turbomind
