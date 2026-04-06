// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/norm_weight.h"

#include "src/turbomind/core/registry.h"

namespace turbomind {

NormWeight::NormWeight(int dim, DataType dtype, DeviceType device)
    : shape_{dim}, dtype_{dtype}
{
    if (device == kDEVICE) {
        add_param("weight", weight_);
    }
    else {
        weight_ = Tensor{shape_, dtype, device};
        add_param("weight", weight_);
    }
}

NormWeight::NormWeight(std::vector<ssize_t> shape, DataType dtype, DeviceType device)
    : shape_{std::move(shape)}, dtype_{dtype}
{
    if (device == kDEVICE) {
        add_param("weight", weight_);
    }
    else {
        weight_ = Tensor{shape_, dtype, device};
        add_param("weight", weight_);
    }
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
        weight_ = Tensor{shape_, dtype_, kDEVICE};
        add_param("weight", weight_);
    }
    if (param_name == "weight") {
        return weight_;
    }
    return ModuleBase::alloc(param_name, spec);
}

namespace {
struct NormWeightRegistrar {
    NormWeightRegistrar() {
        core::ModuleRegistry::instance().register_type(
            "NormWeight",
            [](const core::ModuleConfig& cfg) -> std::unique_ptr<core::ModuleBase> {
                auto dtype = static_cast<DataType>(std::get<int64_t>(cfg.at("data_type")));
                // Check for multi-dimensional "dims" string (e.g. "4 10240")
                if (auto it = cfg.find("dims"); it != cfg.end()) {
                    const auto& dims_str = std::get<std::string>(it->second);
                    std::vector<ssize_t> shape;
                    std::istringstream iss(dims_str);
                    ssize_t d;
                    while (iss >> d) {
                        shape.push_back(d);
                    }
                    return std::make_unique<NormWeight>(std::move(shape), dtype);
                }
                return std::make_unique<NormWeight>(
                    std::get<int64_t>(cfg.at("dim")),
                    dtype);
            });
    }
};
static NormWeightRegistrar _norm_weight_reg;
}  // anonymous namespace

}  // namespace turbomind
