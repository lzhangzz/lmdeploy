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
    AttentionWeight* attention_    = nullptr;
    DeltaNetWeight*  linear_attn_  = nullptr;
    FfnWeight*       feed_forward_ = nullptr;
    MoeWeight*       moe_ffn_      = nullptr;
    NormWeight*      attn_norm_    = nullptr;
    NormWeight*      ffn_norm_     = nullptr;


    // --- Typed accessors ---
    AttentionWeight* attention() const { return attention_; }
    DeltaNetWeight*  linear_attn() const { return linear_attn_; }
    FfnWeight*       ffn() const { return feed_forward_; }
    MoeWeight*       moe() const { return moe_ffn_; }
    NormWeight*      attn_norm() const { return attn_norm_; }
    NormWeight*      ffn_norm() const { return ffn_norm_; }
};

}  // namespace turbomind
