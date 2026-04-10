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

    void                    prepare() override;
    bool                    verify(std::vector<std::string>& missing) override;

    core::ContextGuard context() const
    {
        return core::ContextGuard{stream_, alloca_};
    }

    // --- X-macro field lists ---
#define MODEL_WEIGHT_CHILDREN(X)         \
    X(LinearWeight,     tok_embeddings)  \
    X(LinearWeight,     output)          \
    X(NormWeight,       norm)            \
    X(core::ModuleList, layers)

    MODEL_WEIGHT_CHILDREN(TM_CHILD_MEMBER)

    Module* add_child(std::string name, std::unique_ptr<Module> child) override;
    Module* child(const std::string& name) const override;
    void    for_each_child(std::function<void(const char*, Module*)> visitor) const override;

    // --- Accessors ---
    DecoderLayerWeight*               layer(int i) const;
    std::vector<DecoderLayerWeight*>  layers_list() const;
    int                               num_layers() const { return num_layer_; }

    // --- Lifecycle (same as old LlamaWeight) ---
    bool is_initialized() const { return initialized_; }

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
