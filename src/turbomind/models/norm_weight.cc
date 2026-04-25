// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/norm_weight.h"

#include "src/turbomind/core/registry.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

NormWeight::NormWeight(const core::NormConfig& cfg)
    : shape_{cfg.dim}, dtype_{cfg.data_type}, norm_eps_{cfg.norm_eps}
{
}

void NormWeight::prepare()
{
    EnsureFloatDtype(weight, dtype_);
}

namespace {
struct NormWeightRegistrar {
    NormWeightRegistrar() {
        core::ModuleRegistry::instance().register_type(
            "NormWeight",
            [](const core::ModuleConfig& base_cfg) -> std::unique_ptr<core::Module> {
                return std::make_unique<NormWeight>(
                    static_cast<const core::NormConfig&>(base_cfg));
            });
    }
};
static NormWeightRegistrar _norm_weight_reg;
}  // anonymous namespace

TM_MODULE_METHODS(NormWeight, NORM_WEIGHT_CHILDREN, NORM_WEIGHT_PARAMS)

}  // namespace turbomind
