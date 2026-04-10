// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/core/module_config.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/norm_weight.h"

namespace turbomind {

/// Weight module for Gated DeltaNet (linear attention) layers.
class DeltaNetWeight: public core::Module {
public:
    const char* type() const override { return "DeltaNetWeight"; }

    DeltaNetWeight() = default;

    DeltaNetWeight(const core::DeltaNetConfig& cfg);

    void prepare() override;

    Tensor alloc(const std::string& param_name, const core::WeightSpec& spec) override;

    // --- X-macro field lists ---
#define DELTA_NET_WEIGHT_CHILDREN(X) \
    X(LinearWeight, in_proj_all) \
    X(LinearWeight, out_proj)    \
    X(NormWeight,   norm)

#define DELTA_NET_WEIGHT_PARAMS(X) \
    X(conv1d_) \
    X(A_log_)  \
    X(dt_bias_)

    DELTA_NET_WEIGHT_CHILDREN(TM_CHILD_MEMBER)
    DELTA_NET_WEIGHT_PARAMS(TM_PARAM_MEMBER)

    // Generated overrides
    Module* add_child(std::string name, std::unique_ptr<Module> child) override;
    Module* child(const std::string& name) const override;
    Tensor* param(const std::string& name) const override;
    void    for_each_child(std::function<void(const char*, Module*)> visitor) const override;
    void    for_each_param(std::function<void(const char*, Tensor&)> visitor) const override;

private:
    int      hidden_dim_{};
    int      num_k_heads_{};
    int      num_v_heads_{};
    int      key_head_dim_{};
    int      value_head_dim_{};
    int      d_conv_{};
    bool     bias_{};
    int      tp_size_{};
    int      tp_rank_{};
    DataType data_type_{};
};

}  // namespace turbomind
