// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/models/ffn_weight.h"

namespace turbomind {

enum class MoeMethod {
    kNaive,
    kFused,
};

}  // namespace turbomind

namespace turbomind::core {

struct MoeConfig: ModuleConfig {
    MoeConfig(): ModuleConfig{"MoeWeight"} {}

    #define MOE_FIELDS(X) \
        X(int,         layer_id) \
        X(int,         method) \
        X(int,         experts_per_token) \
        X(int,         inter_size) \
        X(bool,        norm_topk_prob) \
        X(bool,        shared_gate) \
        X(double,      routed_scale) \
        X(bool,        router_bias) \
        X(int,         topk_group) \
        X(std::string, topk_method) \
        X(int,         n_group) \
        X(std::string, scoring_func) \
        X(int,         router_n_groups) \
        X(int,         expert_num) \
        X(int,         hidden_dim) \
        X(bool,        mlp_bias) \
        X(DataType,    data_type) \
        X(int,         tp_size) \
        X(int,         tp_rank) \
        X(int,         act_type) \
        X(bool,        fuse_silu)

    MOE_FIELDS(TM_MEMBER)
    TM_FOR_EACH(MoeConfig, MOE_FIELDS)

    #undef MOE_FIELDS
};

}  // namespace turbomind::core

namespace turbomind {

class MoeWeight: public core::Module {
public:
    const char* type() const override { return "MoeWeight"; }

    MoeWeight() = default;

    MoeWeight(const core::MoeConfig& cfg);

    void prepare() override;
    int num_experts() const { return expert_num_; }

    // --- X-macro child members ---
#define MOE_WEIGHT_CHILDREN(X)           \
    X(LinearWeight, gate)                \
    X(LinearWeight, shared_gate)         \
    X(core::ModuleList, experts)

#define MOE_WEIGHT_PARAMS(X) \
    X(score_correction_bias)

    TM_MODULE_DECLARE(MoeWeight, MOE_WEIGHT_CHILDREN, MOE_WEIGHT_PARAMS)

    // --- Typed accessors ---
    FfnWeight*    expert(int i) const;
    FfnWeight*    block() const { return block_.get(); }
    MoeMethod method() const { return method_; }

    // --- Config fields (public for runtime access) ---
    int  hidden_dim_{};
    int  inter_size_{};
    int  experts_per_token_{};
    bool norm_topk_prob_{};
    bool shared_gate_{};
    float routed_scale_{};
    bool router_bias_{};
    int  topk_group_{};
    std::string topk_method_;
    int  n_group_{};
    std::string scoring_func_;
    int  router_n_groups_{};
    int  expert_num_{};

private:
    int            layer_id_{};
    MoeMethod      method_{MoeMethod::kFused};
    bool           mlp_bias_{};
    DataType       data_type_{};
    int            tp_size_{};
    int            tp_rank_{};
    ActivationType act_type_{};
    bool           fuse_silu_act_{};

    mutable std::unique_ptr<FfnWeight> block_;
};

}  // namespace turbomind
