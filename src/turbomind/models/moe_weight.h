// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/models/ffn_weight.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

class MoeWeight: public core::Module {
public:
    const char* type() const override { return "MoeWeight"; }

    MoeWeight() = default;

    MoeWeight(int              layer_id,
              const MoeParam&  param,
              int              hidden_dim,
              bool             mlp_bias,
              DataType         data_type,
              int              tp_size,
              int              tp_rank,
              ActivationType   act_type,
              bool             fuse_silu_act);

    Tensor  alloc(const std::string& param_name, const core::WeightSpec& spec) override;
    void prepare() override;
    int num_experts() const { return expert_num_; }

    // --- Typed child members (Submodule) ---
    core::Submodule<LinearWeight>     gate        {*this, "gate"};
    core::Submodule<LinearWeight>     shared_gate {*this, "shared_gate"};
    core::Submodule<core::ModuleList> experts     {*this, "experts"};

    // --- Typed accessors ---
    FfnWeight*    expert(int i) const;
    FfnWeight*    block() const { return block_.get(); }
    Tensor*       score_correction_bias() const { return const_cast<Tensor*>(param("score_correction_bias")); }
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
    Tensor score_correction_bias_;
};

}  // namespace turbomind
