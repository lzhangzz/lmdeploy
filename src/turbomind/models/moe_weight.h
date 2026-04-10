// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/core/module_config.h"
#include "src/turbomind/models/ffn_weight.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

class MoeWeight: public core::Module {
public:
    const char* type() const override { return "MoeWeight"; }

    MoeWeight() = default;

    MoeWeight(const core::MoeConfig& cfg);

    Tensor  alloc(const std::string& param_name, const core::WeightSpec& spec) override;
    void prepare() override;
    int num_experts() const { return expert_num_; }

    // --- X-macro child members ---
#define MOE_WEIGHT_CHILDREN(X)           \
    X(LinearWeight, gate)                \
    X(LinearWeight, shared_gate)         \
    X(core::ModuleList, experts)

#define MOE_WEIGHT_PARAMS(X) \
    X(score_correction_bias_)

    MOE_WEIGHT_CHILDREN(TM_CHILD_MEMBER)
    MOE_WEIGHT_PARAMS(TM_PARAM_MEMBER)

    // Generated overrides
    Module* add_child(std::string name, std::unique_ptr<Module> child) override;
    Module* child(const std::string& name) const override;
    Tensor* param(const std::string& name) const override;
    void    for_each_child(std::function<void(const char*, Module*)> visitor) const override;
    void    for_each_param(std::function<void(const char*, Tensor&)> visitor) const override;

    // --- Typed accessors ---
    FfnWeight*    expert(int i) const;
    FfnWeight*    block() const { return block_.get(); }
    Tensor*       score_correction_bias() const { return score_correction_bias_ ? &score_correction_bias_ : nullptr; }
    MoeParam::Method method() const { return moe_param_.method; }
    const MoeParam& moe_param() const { return moe_param_; }

private:
    int            layer_id_{};
    MoeParam       moe_param_{};
    int            hidden_dim_{};
    bool           mlp_bias_{};
    DataType       data_type_{};
    int            tp_size_{};
    int            tp_rank_{};
    ActivationType act_type_{};
    bool           fuse_silu_act_{};
    int            expert_num_{};

    mutable std::unique_ptr<FfnWeight> block_;
};

}  // namespace turbomind
