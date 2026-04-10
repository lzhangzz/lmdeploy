// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/model_weight.h"
#include "src/turbomind/models/decoder_layer_weight.h"
#include "src/turbomind/kernels/core/math.h"

namespace turbomind {

ModelWeight::ModelWeight(DataType       data_type,
                         const ModelParam&  model_param,
                         const EngineParam& engine_param,
                         const MoeParam&    moe_param)
    : data_type_(data_type)
    , model_param_(model_param)
    , engine_param_(engine_param)
    , moe_param_(moe_param)
    , hidden_units_(model_param.hidden_units)
    , vocab_size_(model_param.vocab_size)
    , embedding_size_(model_param.embedding_size)
    , num_layer_(model_param.layer_num)
{
    // Pad vocab size to be divisible by attn_tp_size
    int tp = engine_param.attn_tp_size * engine_param.attn_cp_size;
    vocab_size_padded_ = round_up(vocab_size_, (size_t)tp);

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

core::Module* ModelWeight::add_child(std::string name, std::unique_ptr<core::Module> child)
{
    std::string name_str = std::move(name);
    MODEL_WEIGHT_CHILDREN(TM_ADD_CHILD_CASE)
    return nullptr;
}

core::Module* ModelWeight::child(const std::string& name_str) const
{
    MODEL_WEIGHT_CHILDREN(TM_CHILD_CASE)
    return nullptr;
}

void ModelWeight::for_each_child(
    std::function<void(const char*, core::Module*)> visitor) const
{
    MODEL_WEIGHT_CHILDREN(TM_VISIT_CHILD)
}

}  // namespace turbomind
