// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/model_weight.h"
#include "src/turbomind/models/attention_weight.h"
#include "src/turbomind/models/decoder_layer_weight.h"
#include "src/turbomind/kernels/core/math.h"

namespace turbomind {

ModelWeight::ModelWeight(const EngineParam& engine_param)
    : tp_size_(engine_param.attn_tp_size * engine_param.attn_cp_size)
    , tp_rank_(engine_param.attn_tp_rank)
{
    // Initialize GPU stream and allocator for tensor allocation during weight loading.
    // The CUDA device is already set by CudaDeviceGuard in TurboMind::CreateWeights.
    stream_ = core::Stream::create();
    alloca_ = core::Allocator{stream_, /*use_default_pool=*/true};
}

void ModelWeight::prepare()
{
    for_each_child([](const char* /*name*/, Module* child) {
        if (child) child->prepare();
    });

    auto* layer0 = layer(0);
    TM_CHECK(layer0 && layer0->attention);
    data_type_    = layer0->attention->data_type_;
    hidden_units_ = layer0->attention->hidden_dim_;
    head_dim_     = layer0->attention->head_dim_;
    kv_head_num_  = layer0->attention->kv_head_num_;

    vocab_size_        = tok_embeddings->weight.shape(0);
    embedding_size_    = vocab_size_;
    num_layer_         = layers->size();
    vocab_size_padded_ = round_up((size_t)vocab_size_, (size_t)tp_size_);

    layer_types_.resize(num_layer_);
    for (int i = 0; i < num_layer_; ++i) {
        layer_types_[i] = layer(i)->linear_attn ? 1 : 0;
    }
}

DecoderLayerWeight* ModelWeight::layer(int i) const
{
    if (!layers) {
        return nullptr;
    }
    return static_cast<DecoderLayerWeight*>(layers->child(std::to_string(i)));
}

std::vector<DecoderLayerWeight*> ModelWeight::layers_list() const
{
    if (!layers_cache_.empty()) {
        return layers_cache_;
    }
    if (!layers) {
        return {};
    }
    layers_cache_.resize(layers->size());
    for (int i = 0; i < layers->size(); ++i) {
        layers_cache_[i] = static_cast<DecoderLayerWeight*>(layers->child(std::to_string(i)));
    }
    return layers_cache_;
}

bool ModelWeight::verify(std::vector<std::string>& missing)
{
    Module::verify(missing);
    if (!tok_embeddings) {
        missing.push_back(full_path() + ": missing tok_embeddings");
    }
    if (!norm) {
        missing.push_back(full_path() + ": missing norm");
    }
    return missing.empty();
}

TM_MODULE_METHODS(ModelWeight, MODEL_WEIGHT_CHILDREN, MODEL_WEIGHT_PARAMS)

}  // namespace turbomind
