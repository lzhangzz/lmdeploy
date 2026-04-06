// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/models/ffn_weight.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

class MoeWeight: public core::Module<MoeWeight> {
public:
    static constexpr const char* kTypeName = "MoeWeight";
    const char* type() const override { return kTypeName; }

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

    // --- Typed child members ---
    LinearWeight*     gate_        = nullptr;
    LinearWeight*     shared_gate_ = nullptr;
    core::ModuleList* experts_     = nullptr;

    static constexpr auto kChildren = std::make_tuple(
        std::pair{"gate",        &MoeWeight::gate_},
        std::pair{"shared_gate", &MoeWeight::shared_gate_},
        std::pair{"experts",     &MoeWeight::experts_}
    );
    friend class core::Module<MoeWeight>;

    // --- Typed accessors ---
    LinearWeight* gate() const { return gate_; }
    LinearWeight* shared_gate() const { return shared_gate_; }
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
