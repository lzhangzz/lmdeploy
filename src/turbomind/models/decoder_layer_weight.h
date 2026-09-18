// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/module.h"

namespace turbomind::core {

struct DecoderLayerConfig: ModuleConfig {
    DecoderLayerConfig(): ModuleConfig{"DecoderLayerWeight"} {}
    template<typename Visitor>
    static void for_each(Visitor&&)
    {
    }
};

}  // namespace turbomind::core

namespace turbomind {

class Attention;
class DeltaNet;
class Ffn;
class Moe;
class NormWeight;

/// Architecture-independent decoder layer weight composite.
class DecoderLayerWeight: public core::Module {
public:
    const char* type() const override
    {
        return "DecoderLayerWeight";
    }

    DecoderLayerWeight() = default;
    DecoderLayerWeight(const core::ModuleConfig&);

    ~DecoderLayerWeight() override;  // defined in .cc where child types are complete

    bool verify(std::vector<std::string>& missing) override;

    // --- X-macro field lists ---
#define DECODER_LAYER_WEIGHT_CHILDREN(X)                                                                               \
    X(Attention, attention)                                                                                            \
    X(DeltaNet, linear_attn)                                                                                           \
    X(Ffn, feed_forward)                                                                                               \
    X(Moe, moe_ffn)                                                                                                    \
    X(NormWeight, attention_norm)                                                                                      \
    X(NormWeight, ffn_norm)

#define DECODER_LAYER_WEIGHT_PARAMS(X)

    TM_MODULE_DECLARE(DecoderLayerWeight, DECODER_LAYER_WEIGHT_CHILDREN, DECODER_LAYER_WEIGHT_PARAMS)
};

}  // namespace turbomind
