// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/delta_net_weight.h"

#include "src/turbomind/core/registry.h"

namespace turbomind {

DeltaNetWeight::DeltaNetWeight(const core::DeltaNetConfig& cfg)
    : hidden_dim_{cfg.hidden_dim}
    , num_k_heads_{cfg.num_k_heads}
    , num_v_heads_{cfg.num_v_heads}
    , key_head_dim_{cfg.key_head_dim}
    , value_head_dim_{cfg.value_head_dim}
    , d_conv_{cfg.d_conv}
    , bias_{cfg.has_bias}
    , tp_size_{cfg.tp_size}
    , tp_rank_{cfg.tp_rank}
    , data_type_{cfg.data_type}
{
}

void DeltaNetWeight::prepare()
{
    Module::prepare();
}

Tensor DeltaNetWeight::alloc(const std::string& param_name, const core::WeightSpec& spec)
{
    if (param_name == "A_log" && !A_log) {
        A_log = Tensor{{num_v_heads_ / tp_size_}, data_type_, kDEVICE};
    }
    if (param_name == "A_log") return A_log;

    if (param_name == "dt_bias" && !dt_bias) {
        dt_bias = Tensor{{num_v_heads_ / tp_size_}, data_type_, kDEVICE};
    }
    if (param_name == "dt_bias") return dt_bias;

    if (param_name == "conv1d" && !conv1d) {
        int conv_dim = (num_k_heads_ * key_head_dim_ * 2 + num_v_heads_ * value_head_dim_) / tp_size_;
        conv1d = Tensor{{d_conv_, conv_dim}, data_type_, kDEVICE};
    }
    if (param_name == "conv1d") return conv1d;

    return Module::alloc(param_name, spec);
}

namespace {
struct DeltaNetWeightRegistrar {
    DeltaNetWeightRegistrar() {
        core::ModuleRegistry::instance().register_type(
            "DeltaNetWeight",
            [](const core::ModuleConfig& base_cfg) -> std::unique_ptr<core::Module> {
                return std::make_unique<DeltaNetWeight>(
                    static_cast<const core::DeltaNetConfig&>(base_cfg));
            });
    }
};
static DeltaNetWeightRegistrar _delta_net_weight_reg;
}  // anonymous namespace

TM_MODULE_METHODS(DeltaNetWeight, DELTA_NET_WEIGHT_CHILDREN, DELTA_NET_WEIGHT_PARAMS)

}  // namespace turbomind
