// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/delta_net_weight.h"

#include "src/turbomind/core/registry.h"
#include "src/turbomind/utils/memory_utils.h"

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

    EnsureFloatDtype(A_log, data_type_);
    EnsureFloatDtype(dt_bias, data_type_);
    EnsureFloatDtype(conv1d, data_type_);
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
