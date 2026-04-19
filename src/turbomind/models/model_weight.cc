// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/model_weight.h"
#include "src/turbomind/models/attention_weight.h"
#include "src/turbomind/models/decoder_layer_weight.h"
#include "src/turbomind/kernels/core/math.h"

namespace turbomind {

ModelWeight::ModelWeight(const EngineParam& engine_param)
    : tp_size(engine_param.attn_tp_size * engine_param.attn_cp_size)
    , tp_rank(engine_param.attn_tp_rank)
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

    auto* l0 = layer(0);
    TM_CHECK(l0);
    // Find first full-attention layer (linear-attn layers have no attention child)
    DecoderLayerWeight* attn_layer = nullptr;
    for (int i = 0; i < (int)layers->size(); ++i) {
        if (layer(i)->attention) {
            attn_layer = layer(i);
            break;
        }
    }
    TM_CHECK(attn_layer) << "No full-attention layer found";
    data_type    = attn_layer->attention->data_type;
    hidden_units = attn_layer->attention->hidden_dim;
    head_dim     = attn_layer->attention->head_dim;
    kv_head_num  = attn_layer->attention->kv_head_num;

    vocab_size        = tok_embeddings->weight.shape(0);
    embedding_size    = vocab_size;
    num_layer         = layers->size();
    vocab_size_padded = round_up((size_t)vocab_size, (size_t)tp_size);

    layer_types.resize(num_layer);
    for (int i = 0; i < num_layer; ++i) {
        layer_types[i] = layer(i)->linear_attn ? 1 : 0;
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
