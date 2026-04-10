// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/attention_weight.h"

#include "src/turbomind/core/registry.h"
#include "src/turbomind/kernels/core/math.h"

namespace turbomind {

AttentionWeight::AttentionWeight(const core::AttentionConfig& cfg)
    : hidden_dim_(cfg.hidden_dim)
    , head_dim_(cfg.head_dim)
    , head_num_(cfg.head_num)
    , kv_head_num_(cfg.kv_head_num)
    , mla_(MLAParam{cfg.q_lora_rank, cfg.kv_lora_rank, cfg.qk_rope_dim, cfg.v_head_dim})
    , bias_(cfg.has_bias)
    , qk_norm_(cfg.qk_norm)
    , tp_size_(cfg.tp_size)
    , tp_rank_(cfg.tp_rank)
    , data_type_(cfg.data_type)
    , window_size_(cfg.window_size)
    , sink_(cfg.attn_sink)
    , attn_output_gate_(cfg.attn_output_gate)
{
}

void AttentionWeight::prepare()
{
    Module::prepare();
}

Tensor AttentionWeight::alloc(const std::string& param_name, const core::WeightSpec& spec)
{
    if (param_name == "sinks" && !sinks) {
        sinks = Tensor{{head_num_ / tp_size_}, spec.dtype, kDEVICE};
    }
    if (param_name == "sinks") {
        return sinks;
    }
    return Module::alloc(param_name, spec);
}

namespace {
struct AttentionWeightRegistrar {
    AttentionWeightRegistrar() {
        core::ModuleRegistry::instance().register_type(
            "AttentionWeight",
            [](const core::ModuleConfig& base_cfg) -> std::unique_ptr<core::Module> {
                return std::make_unique<AttentionWeight>(
                    static_cast<const core::AttentionConfig&>(base_cfg));
            });
    }
};
static AttentionWeightRegistrar _attention_weight_reg;
}  // anonymous namespace

TM_MODULE_METHODS(AttentionWeight, ATTENTION_WEIGHT_CHILDREN, ATTENTION_WEIGHT_PARAMS)

}  // namespace turbomind
