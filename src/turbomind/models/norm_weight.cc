// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/norm_weight.h"

#include "src/turbomind/core/module_config.h"
#include "src/turbomind/core/registry.h"

namespace turbomind {

NormWeight::NormWeight(int dim, DataType dtype, DeviceType device)
    : shape_{dim}, dtype_{dtype}
{
    if (device != kDEVICE) {
        *weight_ = Tensor{shape_, dtype_, device};
    }
}

NormWeight::NormWeight(std::vector<ssize_t> shape, DataType dtype, DeviceType device)
    : shape_{std::move(shape)}, dtype_{dtype}
{
    if (device != kDEVICE) {
        *weight_ = Tensor{shape_, dtype_, device};
    }
}

NormWeight::NormWeight(const core::NormConfig& cfg)
{
    configure(cfg.dim, cfg.data_type);
}

void NormWeight::configure(int dim, DataType dtype)
{
    shape_ = {dim};
    dtype_ = dtype;
}

void NormWeight::configure(std::vector<ssize_t> shape, DataType dtype)
{
    shape_ = std::move(shape);
    dtype_ = dtype;
}

Tensor NormWeight::alloc(const std::string& param_name, const core::WeightSpec& spec)
{
    // Allocate on first access
    if (!weight_ && param_name == "weight") {
        // Always use the model's compute dtype (dtype_) for allocation.
        // The rms_norm kernel requires w.dtype() == x.dtype().
        // The Python side (_cast_shard_for_tm) handles casting from source
        // dtype to the model's compute dtype during weight loading.
        *weight_ = Tensor{shape_, dtype_, kDEVICE};
    }
    if (param_name == "weight") {
        return *weight_;
    }
    return Module::alloc(param_name, spec);
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

}  // namespace turbomind
