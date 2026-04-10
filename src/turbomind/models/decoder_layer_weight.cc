// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/decoder_layer_weight.h"
#include "src/turbomind/models/attention_weight.h"
#include "src/turbomind/models/ffn_weight.h"
#include "src/turbomind/models/moe_weight.h"
#include "src/turbomind/models/delta_net_weight.h"
#include "src/turbomind/models/norm_weight.h"

#include "src/turbomind/core/registry.h"

namespace turbomind {

DecoderLayerWeight::~DecoderLayerWeight() = default;

DecoderLayerWeight::DecoderLayerWeight(const core::DecoderLayerConfig&) {}

bool DecoderLayerWeight::verify(std::vector<std::string>& missing)
{
    Module::verify(missing);
    // At least one of attention or linear_attn must exist
    if (!attention && !linear_attn) {
        missing.push_back(full_path() + ": missing attention or linear_attn");
    }
    // At least one of feed_forward or moe_ffn must exist
    if (!feed_forward && !moe_ffn) {
        missing.push_back(full_path() + ": missing feed_forward or moe_ffn");
    }
    // attention_norm must exist
    if (!attention_norm) {
        missing.push_back(full_path() + ": missing attention_norm");
    }
    return missing.empty();
}

// --- X-macro generated method bodies ---

core::Module* DecoderLayerWeight::add_child(std::string name, std::unique_ptr<Module> child)
{
    std::string name_str = std::move(name);
    DECODER_LAYER_WEIGHT_CHILDREN(TM_ADD_CHILD_CASE)
    return nullptr;
}

core::Module* DecoderLayerWeight::child(const std::string& name_str) const
{
    DECODER_LAYER_WEIGHT_CHILDREN(TM_CHILD_CASE)
    return nullptr;
}

void DecoderLayerWeight::for_each_child(
    std::function<void(const char*, Module*)> visitor) const
{
    DECODER_LAYER_WEIGHT_CHILDREN(TM_VISIT_CHILD)
}

namespace {
struct DecoderLayerWeightRegistrar {
    DecoderLayerWeightRegistrar() {
        core::ModuleRegistry::instance().register_type(
            "DecoderLayerWeight",
            [](const core::ModuleConfig&) -> std::unique_ptr<core::Module> {
                return std::make_unique<DecoderLayerWeight>();
            });
    }
};
static DecoderLayerWeightRegistrar _decoder_layer_weight_reg;
}  // anonymous namespace

}  // namespace turbomind
