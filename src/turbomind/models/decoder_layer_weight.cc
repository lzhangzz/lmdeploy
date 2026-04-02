// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/decoder_layer_weight.h"
#include "src/turbomind/models/attention_weight.h"
#include "src/turbomind/models/ffn_weight.h"
#include "src/turbomind/models/moe_weight.h"
#include "src/turbomind/models/delta_net_weight.h"
#include "src/turbomind/models/norm_weight.h"

namespace turbomind {

DecoderLayerWeight::DecoderLayerWeight(int          layer_id,
                                       ModelParam   model_param,
                                       EngineParam  engine_param,
                                       MoeParam     moe_param)
    : layer_id_(layer_id)
    , model_param_(model_param)
    , engine_param_(engine_param)
    , moe_param_(moe_param)
{
}

core::Module* DecoderLayerWeight::ensure_child(const std::string& segment)
{
    if (segment == "attention_norm") {
        auto child = std::make_unique<NormWeight>(model_param_.hidden_units, model_param_.data_type);
        return add_child("attention_norm", std::move(child));
    }
    if (segment == "ffn_norm") {
        auto child = std::make_unique<NormWeight>(model_param_.hidden_units, model_param_.data_type);
        return add_child("ffn_norm", std::move(child));
    }
    if (segment == "attention") {
        int window_size = 0;
        if (layer_id_ < (int)model_param_.window_size.size()) {
            window_size = model_param_.window_size[layer_id_];
        }
        auto child = std::make_unique<AttentionWeight>(
            model_param_.hidden_units,
            model_param_.head_dim,
            model_param_.head_num,
            model_param_.kv_head_num,
            model_param_.mla,
            model_param_.attn_bias,
            model_param_.qk_norm,
            engine_param_.attn_tp_size,
            engine_param_.attn_tp_rank,
            model_param_.data_type,
            window_size,
            model_param_.attn_sink,
            model_param_.attn_output_gate);
        return add_child("attention", std::move(child));
    }
    if (segment == "feed_forward") {
        int inter_size = 0;
        if (layer_id_ < (int)model_param_.inter_size.size()) {
            inter_size = model_param_.inter_size[layer_id_];
        }
        auto child = std::make_unique<FfnWeight>(
            model_param_.hidden_units,
            inter_size,
            model_param_.mlp_bias,
            engine_param_.mlp_tp_size,
            engine_param_.mlp_tp_rank,
            model_param_.data_type,
            model_param_.act_type,
            true);  // fuse_silu_act
        return add_child("feed_forward", std::move(child));
    }
    if (segment == "moe_ffn") {
        auto child = std::make_unique<MoeWeight>(
            layer_id_,
            moe_param_,
            model_param_.hidden_units,
            model_param_.mlp_bias,
            model_param_.data_type,
            engine_param_.mlp_tp_size,
            engine_param_.mlp_tp_rank,
            model_param_.act_type,
            true);  // fuse_silu_act
        return add_child("moe_ffn", std::move(child));
    }
    if (segment == "linear_attn") {
        auto child = std::make_unique<DeltaNetWeight>(
            model_param_.hidden_units,
            model_param_.linear_num_key_heads,
            model_param_.linear_num_value_heads,
            model_param_.linear_key_head_dim,
            model_param_.linear_value_head_dim,
            model_param_.linear_conv_kernel_dim,
            false,  // bias
            engine_param_.attn_tp_size,
            engine_param_.attn_tp_rank,
            model_param_.data_type);
        return add_child("linear_attn", std::move(child));
    }
    return nullptr;
}

bool DecoderLayerWeight::verify(std::vector<std::string>& missing)
{
    Module::verify(missing);
    // At least one of attention or linear_attn must exist
    if (!child("attention") && !child("linear_attn")) {
        missing.push_back(full_path() + ": missing attention or linear_attn");
    }
    // At least one of feed_forward or moe_ffn must exist
    if (!child("feed_forward") && !child("moe_ffn")) {
        missing.push_back(full_path() + ": missing feed_forward or moe_ffn");
    }
    // attention_norm must exist
    if (!child("attention_norm")) {
        missing.push_back(full_path() + ": missing attention_norm");
    }
    return missing.empty();
}

// Typed child accessors — full types are visible here
AttentionWeight* DecoderLayerWeight::attention() const { return static_cast<AttentionWeight*>(child("attention")); }
DeltaNetWeight*  DecoderLayerWeight::linear_attn() const { return static_cast<DeltaNetWeight*>(child("linear_attn")); }
FfnWeight*       DecoderLayerWeight::ffn() const { return static_cast<FfnWeight*>(child("feed_forward")); }
MoeWeight*       DecoderLayerWeight::moe() const { return static_cast<MoeWeight*>(child("moe_ffn")); }
NormWeight*      DecoderLayerWeight::attn_norm() const { return static_cast<NormWeight*>(child("attention_norm")); }
NormWeight*      DecoderLayerWeight::ffn_norm() const { return static_cast<NormWeight*>(child("ffn_norm")); }

}  // namespace turbomind
