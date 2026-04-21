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
}

void FfnWeight::prepare()
{
    // Derive per-rank inter_size from actual weight dimensions.
    // Weight tensors are already TP-sharded by the Python builder,
    // so w1 output_dim equals per-rank inter_size.  For fused w1w3,
    // output_dim = 2 * inter_size (gate + up).
    if (w1w3) {
        inter_size_ = w1w3->output_dim / 2;
    } else if (w1) {
        inter_size_ = w1->output_dim;
    }

    // Set epilogue on existing w1w3 child if fused silu is active.
    if (w1w3) {
        auto* fused = static_cast<LinearWeight*>(w1w3.get());
        if (is_fused_silu_) {
            fused->epilogue = gemm::Epilogue::kGatedSilu;
        }
    }

    // Propagate grouped-GEMM flag for MoE expert weights
    if (is_fused_moe_) {
        auto set_grouped = [](const char*, Module* m) {
            if (auto* lw = dynamic_cast<LinearWeight*>(m)) {
                lw->set_grouped(true);
            }
        };
        for_each_child(set_grouped);
    }

    Module::prepare();  // recurse into children
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

TM_MODULE_METHODS(FfnWeight, FFN_WEIGHT_CHILDREN, FFN_WEIGHT_PARAMS)

}  // namespace turbomind
