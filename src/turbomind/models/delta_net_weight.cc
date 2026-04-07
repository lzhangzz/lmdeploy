// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/delta_net_weight.h"
#include "src/turbomind/models/norm_weight.h"

#include "src/turbomind/core/registry.h"

namespace turbomind {

DeltaNetWeight::DeltaNetWeight(int      hidden_dim,
                               int      num_k_heads,
                               int      num_v_heads,
                               int      key_head_dim_,
                               int      value_head_dim_,
                               int      d_conv,
                               bool     bias,
                               int      tp_size,
                               int      tp_rank,
                               DataType data_type)
    : hidden_dim_(hidden_dim)
    , num_k_heads_(num_k_heads)
    , num_v_heads_(num_v_heads)
    , key_head_dim_(key_head_dim_)
    , value_head_dim_(value_head_dim_)
    , d_conv_(d_conv)
    , bias_(bias)
    , tp_size_(tp_size)
    , tp_rank_(tp_rank)
    , data_type_(data_type)
{
}

void DeltaNetWeight::prepare()
{
    for (auto& [name, child] : children_) {
        child->prepare();
    }
}

Tensor DeltaNetWeight::alloc(const std::string& param_name, const core::WeightSpec& spec)
{
    if (param_name == "A_log" && !A_log_) {
        *A_log_ = Tensor{{num_v_heads_ / tp_size_}, spec.dtype, kDEVICE};
    }
    if (param_name == "A_log") return *A_log_;

    if (param_name == "dt_bias" && !dt_bias_) {
        *dt_bias_ = Tensor{{num_v_heads_ / tp_size_}, spec.dtype, kDEVICE};
    }
    if (param_name == "dt_bias") return *dt_bias_;

    if (param_name == "conv1d" && !conv1d_) {
        int conv_dim = (num_k_heads_ * key_head_dim_ + num_v_heads_ * (value_head_dim_ + key_head_dim_)) / tp_size_;
        *conv1d_ = Tensor{{d_conv_, conv_dim}, spec.dtype, kDEVICE};
    }
    if (param_name == "conv1d") return *conv1d_;

    return Module::alloc(param_name, spec);
}

namespace {
struct DeltaNetWeightRegistrar {
    DeltaNetWeightRegistrar() {
        core::ModuleRegistry::instance().register_type(
            "DeltaNetWeight",
            [](const core::ModuleConfig& cfg) -> std::unique_ptr<core::Module> {
                return std::make_unique<DeltaNetWeight>(
                    std::get<int64_t>(cfg.at("hidden_dim")),
                    std::get<int64_t>(cfg.at("num_k_heads")),
                    std::get<int64_t>(cfg.at("num_v_heads")),
                    std::get<int64_t>(cfg.at("key_head_dim")),
                    std::get<int64_t>(cfg.at("value_head_dim")),
                    std::get<int64_t>(cfg.at("d_conv")),
                    cfg.count("bias") && std::get<int64_t>(cfg.at("bias")),
                    std::get<int64_t>(cfg.at("tp_size")),
                    std::get<int64_t>(cfg.at("tp_rank")),
                    static_cast<DataType>(std::get<int64_t>(cfg.at("data_type"))));
            });
    }
};
static DeltaNetWeightRegistrar _delta_net_weight_reg;
}  // anonymous namespace

}  // namespace turbomind
