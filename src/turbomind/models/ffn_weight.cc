// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/ffn_weight.h"

#include "src/turbomind/core/registry.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind {

FfnWeight::FfnWeight(const core::FfnConfig& cfg)
    : FfnWeight(cfg.hidden_dim, cfg.inter_size, cfg.has_bias, cfg.tp_size, cfg.tp_rank,
                cfg.data_type, static_cast<ActivationType>(cfg.act_type), cfg.fuse_silu)
{
}

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
static int64_t cfg_get(const core::ModuleConfig& cfg, const std::string& key, int64_t def = 0)
{
    auto it = cfg.find(key);
    return it != cfg.end() ? std::get<int64_t>(it->second) : def;
}

static bool cfg_bool(const core::ModuleConfig& cfg, const std::string& key)
{
    auto it = cfg.find(key);
    return it != cfg.end() && std::get<int64_t>(it->second);
}

struct FfnWeightRegistrar {
    FfnWeightRegistrar() {
        core::ModuleRegistry::instance().register_type(
            "FfnWeight",
            [](const core::ModuleConfig& cfg) -> std::unique_ptr<core::Module> {
                auto ffn = std::make_unique<FfnWeight>(
                    cfg_get(cfg, "hidden_dim"),
                    cfg_get(cfg, "inter_size"),
                    cfg_bool(cfg, "has_bias"),
                    cfg_get(cfg, "tp_size"),
                    cfg_get(cfg, "tp_rank"),
                    static_cast<DataType>(cfg_get(cfg, "data_type")),
                    static_cast<ActivationType>(cfg_get(cfg, "act_type")),
                    cfg_bool(cfg, "fuse_silu_act"));
                if (cfg_bool(cfg, "fused_moe")) {
                    ffn->set_fused_moe(true);
                }
                return ffn;
            });
    }
};
static FfnWeightRegistrar _ffn_weight_reg;
}  // anonymous namespace

}  // namespace turbomind
