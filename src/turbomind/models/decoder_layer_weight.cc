// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/decoder_layer_weight.h"
#include "src/turbomind/models/attention_weight.h"
#include "src/turbomind/models/ffn_weight.h"
#include "src/turbomind/models/moe_weight.h"
#include "src/turbomind/models/delta_net_weight.h"
#include "src/turbomind/models/norm_weight.h"

#include "src/turbomind/core/registry.h"

namespace turbomind {

bool DecoderLayerWeight::verify(std::vector<std::string>& missing)
{
    Module::verify(missing);
    // At least one of attention or linear_attn must exist
    if (!child("attention") && !child("linear_attn")) {
        missing.push_back(full_path() + ": missing attention or linear_attn");
    }
    // At least one of feed_forward or moe_ffn must exist
    if (!child("feed_forward") && !child("moe_ffn")) {
        missing.push_back(full_path() + ": missing feed_forward or moe_ffn");
    }
    // attention_norm must exist
    if (!child("attention_norm")) {
        missing.push_back(full_path() + ": missing attention_norm");
    }
    return missing.empty();
}

// Typed child accessors — full types are visible here
AttentionWeight* DecoderLayerWeight::attention() const { return static_cast<AttentionWeight*>(child("attention")); }
DeltaNetWeight*  DecoderLayerWeight::linear_attn() const { return static_cast<DeltaNetWeight*>(child("linear_attn")); }
FfnWeight*       DecoderLayerWeight::ffn() const { return static_cast<FfnWeight*>(child("feed_forward")); }
MoeWeight*       DecoderLayerWeight::moe() const { return static_cast<MoeWeight*>(child("moe_ffn")); }
NormWeight*      DecoderLayerWeight::attn_norm() const { return static_cast<NormWeight*>(child("attention_norm")); }
NormWeight*      DecoderLayerWeight::ffn_norm() const { return static_cast<NormWeight*>(child("ffn_norm")); }

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
