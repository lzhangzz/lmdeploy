// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/delta_net_weight.h"
#include "src/turbomind/models/norm_weight.h"

namespace turbomind {

DeltaNetWeight::DeltaNetWeight(int      hidden_dim,
                               int      num_k_heads,
                               int      num_v_heads,
                               int      key_head_dim_,
                               int      value_head_dim_,
                               int      d_conv,
                               bool     bias,
                               int      tp_size,
                               int      tp_rank,
                               DataType data_type)
    : hidden_dim_(hidden_dim)
    , num_k_heads_(num_k_heads)
    , num_v_heads_(num_v_heads)
    , key_head_dim_(key_head_dim_)
    , value_head_dim_(value_head_dim_)
    , d_conv_(d_conv)
    , bias_(bias)
    , tp_size_(tp_size)
    , tp_rank_(tp_rank)
    , data_type_(data_type)
{
}

core::Module* DeltaNetWeight::ensure_child(const std::string& segment)
{
    // in_proj_all fuses: qkv(q,k,v) + z + a + b
    int tp          = tp_size_;
    int q_dim       = num_k_heads_ * key_head_dim_ / tp;
    int k_dim       = num_k_heads_ * key_head_dim_ / tp;
    int v_dim       = num_v_heads_ * value_head_dim_ / tp;
    int z_dim       = num_v_heads_ * value_head_dim_ / tp;
    int ab_dim      = 2 * (num_v_heads_ / tp);
    int in_proj_dim = q_dim + k_dim + v_dim + z_dim + ab_dim;
    // conv1d operates on the full qkv: shape [q_dim + k_dim + v_dim, d_conv]
    int conv_dim    = q_dim + k_dim + v_dim;
    int v_heads_tp  = num_v_heads_ / tp;

    if (segment == "in_proj_all") {
        auto child = std::make_unique<LinearWeight>();
        child->configure(hidden_dim_, in_proj_dim, data_type_, bias_);
        return add_child("in_proj_all", std::move(child));
    }
    if (segment == "out_proj") {
        auto child = std::make_unique<LinearWeight>();
        child->configure(num_v_heads_ * value_head_dim_ / tp_size_, hidden_dim_, data_type_, bias_);
        return add_child("out_proj", std::move(child));
    }
    if (segment == "conv1d") {
        // conv1d weight: shape [d_conv, conv_dim] (2D parameter)
        auto norm = std::make_unique<NormWeight>(std::vector<ssize_t>{d_conv_, conv_dim}, data_type_);
        return add_child("conv1d", std::move(norm));
    }
    if (segment == "A_log") {
        auto norm = std::make_unique<NormWeight>(v_heads_tp, data_type_);
        return add_child("A_log", std::move(norm));
    }
    if (segment == "dt_bias") {
        auto norm = std::make_unique<NormWeight>(v_heads_tp, data_type_);
        return add_child("dt_bias", std::move(norm));
    }
    if (segment == "norm") {
        // norm is per-head: shape [value_head_dim_]
        auto norm = std::make_unique<NormWeight>(value_head_dim_, data_type_);
        return add_child("norm", std::move(norm));
    }
    return nullptr;
}

void DeltaNetWeight::prepare()
{
    for (auto& [name, child] : children_) {
        child->prepare();
    }
}

Tensor* DeltaNetWeight::conv1d() const
{
    auto* m = static_cast<NormWeight*>(child("conv1d"));
    return m ? &m->weight() : nullptr;
}

Tensor* DeltaNetWeight::A_log() const
{
    auto* m = static_cast<NormWeight*>(child("A_log"));
    return m ? &m->weight() : nullptr;
}

Tensor* DeltaNetWeight::dt_bias() const
{
    auto* m = static_cast<NormWeight*>(child("dt_bias"));
    return m ? &m->weight() : nullptr;
}

Tensor* DeltaNetWeight::norm() const
{
    auto* m = static_cast<NormWeight*>(child("norm"));
    return m ? &m->weight() : nullptr;
}

}  // namespace turbomind
