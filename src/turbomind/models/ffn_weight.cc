// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/ffn_weight.h"

#include "src/turbomind/core/registry.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind {

FfnWeight::FfnWeight(const core::FfnConfig& cfg)
    : hidden_dim_{cfg.hidden_dim}
    , inter_size_{cfg.inter_size}
    , bias_{cfg.has_bias}
    , tp_size_{cfg.tp_size}
    , tp_rank_{cfg.tp_rank}
    , data_type_{cfg.data_type}
    , act_type_{static_cast<ActivationType>(cfg.act_type)}
    , is_fused_silu_{cfg.fuse_silu && static_cast<ActivationType>(cfg.act_type) == ActivationType::kSilu}
    , is_fused_moe_{cfg.fused_moe}
{
    TM_CHECK(inter_size_ % tp_size_ == 0) << inter_size_ << " " << tp_size_;
    inter_size_ /= tp_size_;
}

void FfnWeight::prepare()
{
    // Set epilogue on existing w1w3 child if fused silu is active.
    // The w1/w3 fusion (interleave/chunk) is now done on the Python side.
    if (w1w3) {
        auto* fused = static_cast<LinearWeight*>(w1w3);
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

namespace {
struct FfnWeightRegistrar {
    FfnWeightRegistrar() {
        core::ModuleRegistry::instance().register_type(
            "FfnWeight",
            [](const core::ModuleConfig& base_cfg) -> std::unique_ptr<core::Module> {
                return std::make_unique<FfnWeight>(
                    static_cast<const core::FfnConfig&>(base_cfg));
            });
    }
};
static FfnWeightRegistrar _ffn_weight_reg;
}  // anonymous namespace

}  // namespace turbomind
