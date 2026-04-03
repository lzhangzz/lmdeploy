// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/ffn_weight.h"

#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind {

FfnWeight::FfnWeight(int hidden_dim, int inter_size, bool bias, int tp_size, int tp_rank,
                     DataType data_type, ActivationType act_type, bool fuse_silu_act)
    : hidden_dim_(hidden_dim)
    , inter_size_(inter_size)
    , bias_(bias)
    , tp_size_(tp_size)
    , tp_rank_(tp_rank)
    , data_type_(data_type)
    , act_type_(act_type)
    , is_fused_silu_(fuse_silu_act && act_type == ActivationType::kSilu)
{
    TM_CHECK(inter_size_ % tp_size_ == 0) << inter_size_ << " " << tp_size_;
    inter_size_ /= tp_size_;
}

core::Module* FfnWeight::ensure_child(const std::string& segment)
{
    if (segment == "w1") {
        auto child = std::make_unique<LinearWeight>();
        child->configure(hidden_dim_, inter_size_, data_type_, bias_);
        return add_child("w1", std::move(child));
    }
    if (segment == "w3") {
        auto child = std::make_unique<LinearWeight>();
        child->configure(hidden_dim_, inter_size_, data_type_, bias_);
        return add_child("w3", std::move(child));
    }
    if (segment == "w2") {
        auto child = std::make_unique<LinearWeight>();
        child->configure(inter_size_, hidden_dim_, data_type_, bias_);
        return add_child("w2", std::move(child));
    }
    if (segment == "w1w3") {
        // Fused gate+up projection
        auto child = std::make_unique<LinearWeight>();
        child->configure(hidden_dim_, inter_size_ * 2, data_type_, bias_);
        return add_child("w1w3", std::move(child));
    }
    return nullptr;
}

void FfnWeight::prepare()
{
    // Set epilogue on existing w1w3 child if fused silu is active.
    // The w1/w3 fusion (interleave/chunk) is now done on the Python side.
    if (auto* fused = w1w3()) {
        if (is_fused_silu_) {
            fused->epilogue = gemm::Epilogue::kGatedSilu;
        }
    }

    // Prepare (format conversion) for all children
    for (auto& [name, child] : children_) {
        // Propagate grouped-GEMM flag for MoE expert weights
        if (is_fused_moe_) {
            if (auto* lw = dynamic_cast<LinearWeight*>(child.get())) {
                lw->set_grouped(true);
            }
        }
        child->prepare();
    }
}

}  // namespace turbomind
