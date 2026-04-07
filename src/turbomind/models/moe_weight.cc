// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/moe_weight.h"

#include "src/turbomind/core/registry.h"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

MoeWeight::MoeWeight(int              layer_id,
                     const MoeParam&  param,
                     int              hidden_dim,
                     bool             mlp_bias,
                     DataType         data_type,
                     int              tp_size,
                     int              tp_rank,
                     ActivationType   act_type,
                     bool             fuse_silu_act)
    : layer_id_(layer_id)
    , moe_param_(param)
    , hidden_dim_(hidden_dim)
    , mlp_bias_(mlp_bias)
    , data_type_(data_type)
    , tp_size_(tp_size)
    , tp_rank_(tp_rank)
    , act_type_(act_type)
    , fuse_silu_act_(fuse_silu_act)
{
    if ((int)moe_param_.expert_num.size() > layer_id_) {
        expert_num_ = moe_param_.expert_num[layer_id_];
    }
}

Tensor MoeWeight::alloc(const std::string& param_name, const core::WeightSpec& spec)
{
    if (param_name == "score_correction_bias" && expert_num_ > 0) {
        if (!score_correction_bias_) {
            score_correction_bias_ = Tensor{{expert_num_}, spec.dtype, kDEVICE};
            add_param("score_correction_bias", score_correction_bias_);
        }
        return score_correction_bias_;
    }
    return Module::alloc(param_name, spec);
}

// Adapted from LinkExperts in LlamaDenseWeight.cc for LinearWeight
static void LinkLinearExperts(std::function<LinearWeight*(int)> experts, int n, LinearWeight& d)
{
    const auto& e0 = *experts(0);

    d.input_dim    = e0.input_dim;
    d.output_dim   = e0.output_dim;
    d.group_size   = e0.group_size;
    d.data_type    = e0.data_type;
    d.weight_format = e0.weight_format;
    d.format_       = e0.format_;
    d.policy_       = e0.policy_;
    d.k_desc       = e0.k_desc;
    d.q_desc       = e0.q_desc;
    d.epilogue     = e0.epilogue;

    d.k_desc.num = d.q_desc.num = n;

    if (e0.bias) {
        d.bias = Tensor{{n, e0.output_dim}, e0.bias.dtype(), kDEVICE};
    }

    std::vector<std::pair<void*, int>> weights;
    std::vector<std::pair<void*, int>> scales;

    for (int i = 0; i < n; ++i) {
        auto& e = *experts(i);
        weights.emplace_back(e.weight.raw_data(), e.k_desc.ld);
        if (e.scales) {
            scales.emplace_back(e.scales.raw_data(), e.q_desc.ld);
        }
        if (e.bias) {
            Copy(e.bias, d.bias.slice(i, 1).squeeze(0));
        }
    }

    auto stream = core::Context::stream().handle();

    if (d.weight_format == kFloat8_e4m3 && d.input_dtype() == kFloat8_e4m3) {
        auto make_blocked_ptr = [&](const auto& ptrs) {
            return std::shared_ptr<void>{gemm::MakeBlockedPtrs(ptrs, stream), [](auto p) { cudaFree(p); }};
        };
        d.weight = Tensor{make_blocked_ptr(weights), {n}, e0.weight.dtype(), kDEVICE};
        d.scales = Tensor{make_blocked_ptr(scales), {n}, e0.scales.dtype(), kDEVICE};
        d.k_desc.offsets = d.q_desc.offsets = (int*)1;
    }
    else {
        auto make_strided_ptr = [&](const auto& ptrs) {
            return std::shared_ptr<void>{gemm::MakeStridedPtrs(ptrs, stream), [](auto p) { cudaFree(p); }};
        };
        d.weight = Tensor{make_strided_ptr(weights), {n}, d.weight_format, kDEVICE};
        if (e0.scales) {
            d.scales = Tensor{make_strided_ptr(scales), {n}, e0.scales.dtype(), kDEVICE};
        }
        d.k_desc.ld = d.q_desc.ld = 0;
    }
}

FfnWeight* MoeWeight::expert(int i) const
{
    if (!experts) {
        return nullptr;
    }
    return static_cast<FfnWeight*>(experts->child(std::to_string(i)));
}

void MoeWeight::prepare()
{
    // First prepare all children (experts, gate, etc.)
    for (auto& [name, child] : children()) {
        child->prepare();
    }

    // Create batched block view for fused MoE path
    if (expert_num_ > 0 && method() == MoeParam::kFused) {
        block_ = std::make_unique<FfnWeight>(hidden_dim_,
                                              moe_param_.inter_size,
                                              mlp_bias_,
                                              tp_size_,
                                              tp_rank_,
                                              data_type_,
                                              act_type_,
                                              fuse_silu_act_);

        // Link each linear in the block to the corresponding expert linears
        auto get_expert_w1w3 = [this](int i) -> LinearWeight* {
            auto* exp = expert(i);
            return exp ? exp->w1w3.get() : nullptr;
        };
        auto get_expert_w1 = [this](int i) -> LinearWeight* {
            auto* exp = expert(i);
            return exp ? exp->w1.get() : nullptr;
        };
        auto get_expert_w3 = [this](int i) -> LinearWeight* {
            auto* exp = expert(i);
            return exp ? exp->w3.get() : nullptr;
        };
        auto get_expert_w2 = [this](int i) -> LinearWeight* {
            auto* exp = expert(i);
            return exp ? exp->w2.get() : nullptr;
        };

        if (get_expert_w1w3(0)) {
            // Fused w1w3 path: experts have a single fused gate+up projection
            block_->add_child("w1w3", std::make_unique<LinearWeight>());
            LinkLinearExperts(get_expert_w1w3, expert_num_, *block_->w1w3);
        }
        else {
            // Separate w1/w3 path: link individually
            block_->add_child("w1", std::make_unique<LinearWeight>());
            block_->add_child("w3", std::make_unique<LinearWeight>());
            if (get_expert_w1(0)) {
                LinkLinearExperts(get_expert_w1, expert_num_, *block_->w1);
            }
            if (get_expert_w3(0)) {
                LinkLinearExperts(get_expert_w3, expert_num_, *block_->w3);
            }
        }

        block_->add_child("w2", std::make_unique<LinearWeight>());
        if (get_expert_w2(0)) {
            LinkLinearExperts(get_expert_w2, expert_num_, *block_->w2);
        }

        // Propagate the actual fused-silu state from the first expert to
        // the block.  Each expert's prepare() has already run above, so
        // is_fused_silu() now reflects whether the GEMM epilogue applies
        // SiLU (true for quantized formats, false for dense bf16/fp16).
        if (auto* e0 = expert(0)) {
            block_->set_fused_silu(e0->is_fused_silu());
        }
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

struct MoeWeightRegistrar {
    MoeWeightRegistrar() {
        core::ModuleRegistry::instance().register_type(
            "MoeWeight",
            [](const core::ModuleConfig& cfg) -> std::unique_ptr<core::Module> {
                MoeParam moe_param;
                moe_param.method           = static_cast<MoeParam::Method>(cfg_get(cfg, "method"));
                moe_param.experts_per_token = cfg_get(cfg, "experts_per_token");
                moe_param.inter_size       = cfg_get(cfg, "inter_size");
                moe_param.norm_topk_prob   = cfg_bool(cfg, "norm_topk_prob");
                moe_param.shared_gate      = cfg_bool(cfg, "shared_gate");
                moe_param.routed_scale     = cfg.count("routed_scale")
                    ? static_cast<float>(std::get<double>(cfg.at("routed_scale"))) : 1.0f;
                moe_param.router_bias      = cfg_bool(cfg, "router_bias");
                moe_param.topk_group      = cfg_get(cfg, "topk_group");
                moe_param.topk_method     = cfg.count("topk_method")
                    ? std::get<std::string>(cfg.at("topk_method")) : "greedy";
                moe_param.n_group         = cfg_get(cfg, "n_group");
                moe_param.scoring_func    = cfg.count("scoring_func")
                    ? std::get<std::string>(cfg.at("scoring_func")) : "softmax";
                moe_param.router_n_groups = cfg_get(cfg, "router_n_groups");
                int expert_num = cfg_get(cfg, "expert_num");
                moe_param.expert_num.assign(1, expert_num);
                // Pass layer_id=0 so the constructor indexes into expert_num[0]
                // (the vector has a single element for this per-layer instance).
                return std::make_unique<MoeWeight>(
                    0,
                    moe_param,
                    cfg_get(cfg, "hidden_dim"),
                    cfg_bool(cfg, "mlp_bias"),
                    static_cast<DataType>(cfg_get(cfg, "data_type")),
                    cfg_get(cfg, "tp_size"),
                    cfg_get(cfg, "tp_rank"),
                    static_cast<ActivationType>(cfg_get(cfg, "act_type")),
                    cfg_bool(cfg, "fuse_silu_act"));
            });
    }
};
static MoeWeightRegistrar _moe_weight_reg;
}  // anonymous namespace

}  // namespace turbomind
