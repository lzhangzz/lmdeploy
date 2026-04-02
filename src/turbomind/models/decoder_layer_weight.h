// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/module.h"
#include "src/turbomind/models/llama/llama_params.h"

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

    DecoderLayerWeight(int           layer_id,
                       ModelParam    model_param,
                       EngineParam   engine_param,
                       MoeParam      moe_param);

    core::Module* ensure_child(const std::string& segment) override;
    bool verify(std::vector<std::string>& missing) override;

    // Typed child accessors — defined in .cc where full types are visible
    AttentionWeight* attention() const;
    DeltaNetWeight*  linear_attn() const;
    FfnWeight*       ffn() const;
    MoeWeight*       moe() const;
    NormWeight*      attn_norm() const;
    NormWeight*      ffn_norm() const;

private:
    int         layer_id_{};
    ModelParam  model_param_{};
    EngineParam engine_param_{};
    MoeParam    moe_param_{};
};

}  // namespace turbomind
