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
class ModelWeight: public core::Module<ModelWeight> {
public:
    static constexpr const char* kTypeName = "ModelWeight";

    const char* type() const override { return kTypeName; }

    ModelWeight() = default;

    ModelWeight(DataType          data_type,
                const ModelParam&  model_param,
                const EngineParam& engine_param,
                const MoeParam&    moe_param);

    void                    prepare() override;
    bool                    verify(std::vector<std::string>& missing) override;

    core::ContextGuard context() const
    {
        return core::ContextGuard{stream_, alloca_};
    }

    // --- Typed child members ---
    LinearWeight*        tok_embeddings_ = nullptr;
    LinearWeight*        output_         = nullptr;
    NormWeight*          norm_           = nullptr;
    core::ModuleList*    layers_         = nullptr;

    static constexpr auto kChildren = std::make_tuple(
        std::pair{"tok_embeddings", &ModelWeight::tok_embeddings_},
        std::pair{"output",         &ModelWeight::output_},
        std::pair{"norm",           &ModelWeight::norm_},
        std::pair{"layers",         &ModelWeight::layers_}
    );
    friend class core::Module<ModelWeight>;

    // --- Typed child accessors ---
    LinearWeight*        tok_embeddings() const { return tok_embeddings_; }
    LinearWeight*        output() const { return output_; }
    NormWeight*          norm() const { return norm_; }
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
