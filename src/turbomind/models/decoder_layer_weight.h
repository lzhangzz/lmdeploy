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
class DecoderLayerWeight: public core::Module<DecoderLayerWeight> {
public:
    static constexpr const char* kTypeName = "DecoderLayerWeight";
    const char* type() const override { return kTypeName; }

    DecoderLayerWeight() = default;

    bool verify(std::vector<std::string>& missing) override;

    // --- Typed child members ---
    AttentionWeight* attention_    = nullptr;
    DeltaNetWeight*  linear_attn_  = nullptr;
    FfnWeight*       feed_forward_ = nullptr;
    MoeWeight*       moe_ffn_      = nullptr;
    NormWeight*      attn_norm_    = nullptr;
    NormWeight*      ffn_norm_     = nullptr;

    static constexpr auto kChildren = std::make_tuple(
        std::pair{"attention",      &DecoderLayerWeight::attention_},
        std::pair{"linear_attn",    &DecoderLayerWeight::linear_attn_},
        std::pair{"feed_forward",   &DecoderLayerWeight::feed_forward_},
        std::pair{"moe_ffn",        &DecoderLayerWeight::moe_ffn_},
        std::pair{"attention_norm", &DecoderLayerWeight::attn_norm_},
        std::pair{"ffn_norm",       &DecoderLayerWeight::ffn_norm_}
    );
    friend class core::Module<DecoderLayerWeight>;

    // --- Typed accessors ---
    AttentionWeight* attention() const { return attention_; }
    DeltaNetWeight*  linear_attn() const { return linear_attn_; }
    FfnWeight*       ffn() const { return feed_forward_; }
    MoeWeight*       moe() const { return moe_ffn_; }
    NormWeight*      attn_norm() const { return attn_norm_; }
    NormWeight*      ffn_norm() const { return ffn_norm_; }
};

}  // namespace turbomind
