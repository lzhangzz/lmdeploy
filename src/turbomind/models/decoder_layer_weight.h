// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/module.h"
#include "src/turbomind/core/module_config.h"

namespace turbomind {

class AttentionWeight;
class DeltaNetWeight;
class FfnWeight;
class MoeWeight;
class NormWeight;

/// Architecture-independent decoder layer weight composite.
class DecoderLayerWeight: public core::Module {
public:
    const char* type() const override { return "DecoderLayerWeight"; }

    DecoderLayerWeight() = default;
    explicit DecoderLayerWeight(const core::DecoderLayerConfig&);  // defined in .cc
    ~DecoderLayerWeight() override;  // defined in .cc where child types are complete

    bool verify(std::vector<std::string>& missing) override;

    // --- X-macro field lists ---
#define DECODER_LAYER_WEIGHT_CHILDREN(X) \
    X(AttentionWeight, attention)     \
    X(DeltaNetWeight,  linear_attn)   \
    X(FfnWeight,       feed_forward)  \
    X(MoeWeight,       moe_ffn)       \
    X(NormWeight,      attention_norm) \
    X(NormWeight,      ffn_norm)

    DECODER_LAYER_WEIGHT_CHILDREN(TM_CHILD_MEMBER)

    // Generated overrides
    Module* add_child(std::string name, std::unique_ptr<Module> child) override;
    Module* child(const std::string& name) const override;
    void    for_each_child(std::function<void(const char*, Module*)> visitor) const override;
};

}  // namespace turbomind
