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

    /// Allocate the weight tensor on first call, then return it.
    Tensor alloc(const std::string& param_name, const core::WeightSpec& spec) override;

    Tensor&       weight()       { return weight_; }
    const Tensor& weight() const { return weight_; }

private:
#define NORM_WEIGHT_PARAMS(X) \
    X(weight_)

    NORM_WEIGHT_PARAMS(TM_PARAM_MEMBER)

    Tensor* param(const std::string& name_str) const override;
    void    for_each_param(std::function<void(const char*, Tensor&)>) const override;

    std::vector<ssize_t> shape_;
    DataType              dtype_{};
};

}  // namespace turbomind
