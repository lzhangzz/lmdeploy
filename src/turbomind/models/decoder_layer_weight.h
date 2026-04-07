// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/module.h"

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

    bool verify(std::vector<std::string>& missing) override;

    // --- Typed child members ---
    core::Submodule<AttentionWeight> attention    {*this, "attention"};
    core::Submodule<DeltaNetWeight>  linear_attn  {*this, "linear_attn"};
    core::Submodule<FfnWeight>       feed_forward {*this, "feed_forward"};
    core::Submodule<MoeWeight>       moe_ffn      {*this, "moe_ffn"};
    core::Submodule<NormWeight>      attn_norm    {*this, "attention_norm"};
    core::Submodule<NormWeight>      ffn_norm     {*this, "ffn_norm"};
};

}  // namespace turbomind
