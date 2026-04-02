// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/attention_weight.h"

#include "src/turbomind/kernels/core/math.h"

namespace turbomind {

AttentionWeight::AttentionWeight(int          hidden_dim,
                                 int          head_dim,
                                 int          head_num,
                                 int          kv_head_num,
                                 MLAParam     mla,
                                 bool         bias,
                                 bool         qk_norm,
                                 int          tp_size,
                                 int          tp_rank,
                                 DataType     data_type,
                                 int          window_size,
                                 bool         sink,
                                 bool         attn_output_gate)
    : hidden_dim_(hidden_dim)
    , head_dim_(head_dim)
    , head_num_(head_num)
    , kv_head_num_(kv_head_num)
    , mla_(mla)
    , bias_(bias)
    , qk_norm_(qk_norm)
    , tp_size_(tp_size)
    , tp_rank_(tp_rank)
    , data_type_(data_type)
    , window_size_(window_size)
    , sink_(sink)
    , attn_output_gate_(attn_output_gate)
{
}

core::Module* AttentionWeight::ensure_child(const std::string& segment)
{
    const int q_factor = attn_output_gate_ ? 2 : 1;

    if (mla_.kv_lora_rank == 0) {
        // Standard (non-MLA) attention
        if (segment == "w_qkv") {
            auto child = std::make_unique<LinearWeight>();
            child->configure(hidden_dim_,
                             (head_num_ * q_factor + 2 * kv_head_num_) * head_dim_ / tp_size_,
                             data_type_,
                             bias_);
            return add_child("w_qkv", std::move(child));
        }
    }
    else {
        // MLA attention
        if (mla_.q_lora_rank) {
            if (segment == "q_a_proj") {
                auto child = std::make_unique<LinearWeight>();
                child->configure(hidden_dim_, mla_.q_lora_rank, data_type_, false);
                return add_child("q_a_proj", std::move(child));
            }
            if (segment == "q_b_proj") {
                auto child = std::make_unique<LinearWeight>();
                child->configure(mla_.q_lora_rank, head_num_ * head_dim_ / tp_size_, data_type_, false);
                return add_child("q_b_proj", std::move(child));
            }
        }
        else {
            if (segment == "q_proj") {
                auto child = std::make_unique<LinearWeight>();
                child->configure(hidden_dim_, head_num_ * head_dim_ / tp_size_, data_type_, false);
                return add_child("q_proj", std::move(child));
            }
        }
        if (segment == "kv_a_proj") {
            auto child = std::make_unique<LinearWeight>();
            child->configure(hidden_dim_, mla_.kv_lora_rank + mla_.qk_rope_dim, data_type_, false);
            return add_child("kv_a_proj", std::move(child));
        }
    }

    // Output projection (common to both standard and MLA)
    if (segment == "wo") {
        auto child = std::make_unique<LinearWeight>();
        child->configure(head_num_ * head_dim_ / tp_size_, hidden_dim_, data_type_, bias_);
        return add_child("wo", std::move(child));
    }

    // Q/K norms (used when qk_norm is enabled)
    if (segment == "q_norm" && qk_norm_) {
        auto child = std::make_unique<NormWeight>(head_dim_, data_type_);
        return add_child("q_norm", std::move(child));
    }
    if (segment == "k_norm" && qk_norm_) {
        auto child = std::make_unique<NormWeight>(head_dim_, data_type_);
        return add_child("k_norm", std::move(child));
    }

    // Q_A layernorm (MLA with q_lora_rank)
    if (segment == "q_a_layernorm" && mla_.kv_lora_rank > 0 && mla_.q_lora_rank) {
        auto child = std::make_unique<NormWeight>(mla_.q_lora_rank, data_type_);
        return add_child("q_a_layernorm", std::move(child));
    }
    // KV_A layernorm (MLA)
    if (segment == "kv_a_layernorm" && mla_.kv_lora_rank > 0) {
        auto child = std::make_unique<NormWeight>(mla_.kv_lora_rank, data_type_);
        return add_child("kv_a_layernorm", std::move(child));
    }

    // Attention sinks (optional)
    if (segment == "sinks" && sink_) {
        auto child = std::make_unique<NormWeight>(head_num_ / tp_size_, data_type_);
        return add_child("sinks", std::move(child));
    }

    return nullptr;
}

void AttentionWeight::prepare()
{
    for (auto& [name, child] : children_) {
        child->prepare();
    }
}

// Convenience tensor accessors
Tensor* AttentionWeight::q_norm() const
{
    auto* m = q_norm_mod();
    return m ? &m->weight() : nullptr;
}
Tensor* AttentionWeight::k_norm() const
{
    auto* m = k_norm_mod();
    return m ? &m->weight() : nullptr;
}
Tensor* AttentionWeight::q_a_layernorm() const
{
    auto* m = q_a_layernorm_mod();
    return m ? &m->weight() : nullptr;
}
Tensor* AttentionWeight::kv_a_layernorm() const
{
    auto* m = kv_a_layernorm_mod();
    return m ? &m->weight() : nullptr;
}
Tensor* AttentionWeight::sinks() const
{
    auto* m = static_cast<NormWeight*>(child("sinks"));
    return m ? &m->weight() : nullptr;
}

}  // namespace turbomind
