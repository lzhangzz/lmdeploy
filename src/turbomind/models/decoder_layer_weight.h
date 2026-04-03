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

    // Typed child accessors — defined in .cc where full types are visible
    AttentionWeight* attention() const;
    DeltaNetWeight*  linear_attn() const;
    FfnWeight*       ffn() const;
    MoeWeight*       moe() const;
    NormWeight*      attn_norm() const;
    NormWeight*      ffn_norm() const;
};

}  // namespace turbomind
