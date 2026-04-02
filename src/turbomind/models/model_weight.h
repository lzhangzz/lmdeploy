// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/norm_weight.h"
#include "src/turbomind/models/llama/llama_params.h"

#include <vector>

namespace turbomind {

class DecoderLayerWeight;

/// Root weight module for a model. Owns the full weight tree.
class ModelWeight: public core::Module {
public:
    const char* type() const override { return "ModelWeight"; }

    ModelWeight() = default;

    ModelWeight(DataType          data_type,
                const ModelParam&  model_param,
                const EngineParam& engine_param,
                const MoeParam&    moe_param);

    core::Module* ensure_child(const std::string& segment) override;
    void                    prepare() override;
    bool                    verify(std::vector<std::string>& missing) override;

    core::ContextGuard context() const
    {
        return core::ContextGuard{stream_, alloca_};
    }

    // --- Typed child accessors ---
    LinearWeight*        tok_embeddings() const { return static_cast<LinearWeight*>(child("tok_embeddings")); }
    LinearWeight*        output() const { return static_cast<LinearWeight*>(child("output")); }
    NormWeight*          norm() const { return static_cast<NormWeight*>(child("norm")); }
    DecoderLayerWeight*  layer(int i) const;
    std::vector<DecoderLayerWeight*> layers() const;
    int                  num_layers() const { return num_layer_; }

    // --- Lifecycle (same as old LlamaWeight) ---
    bool is_initialized() const { return initialized_; }
    // release() and to_device() inherited from Module base class

    // --- Model config accessors for LanguageModel ---
    int   hidden_units() const { return hidden_units_; }
    int   vocab_size_padded() const { return vocab_size_padded_; }
    int   tp_size() const { return tp_size_; }

private:
    DataType    data_type_{};
    ModelParam  model_param_{};
    EngineParam engine_param_{};
    MoeParam    moe_param_{};

    size_t hidden_units_{};
    size_t vocab_size_{};
    size_t vocab_size_padded_{};
    size_t embedding_size_{};
    size_t num_layer_{};

    int  tp_size_{};
    int  tp_rank_{};

    bool initialized_{false};

    core::Stream    stream_{};
    core::Allocator alloca_{};

    mutable std::vector<DecoderLayerWeight*> layers_cache_;
};

}  // namespace turbomind
