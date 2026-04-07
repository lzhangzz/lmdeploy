// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/norm_weight.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

class AttentionWeight: public core::Module {
public:
    const char* type() const override { return "AttentionWeight"; }

    AttentionWeight() = default;

    AttentionWeight(int          hidden_dim,
                    int          head_dim,
                    int          head_num,
                    int          kv_head_num,
                    MLAParam     mla,
                    bool         bias,
                    bool         qk_norm,
                    int          tp_size,
                    int          tp_rank,
                    DataType     data_type,
                    int          window_size,
                    bool         sink,
                    bool         attn_output_gate);

    void prepare() override;
    Tensor alloc(const std::string& param_name, const core::WeightSpec& spec) override;

    // --- Typed child members ---
    core::Submodule<LinearWeight> w_qkv             {*this, "w_qkv"};
    core::Submodule<LinearWeight> wo                {*this, "wo"};
    core::Submodule<LinearWeight> q_proj            {*this, "q_proj"};
    core::Submodule<LinearWeight> q_a_proj          {*this, "q_a_proj"};
    core::Submodule<LinearWeight> q_b_proj          {*this, "q_b_proj"};
    core::Submodule<LinearWeight> kv_a_proj         {*this, "kv_a_proj"};
    core::Submodule<NormWeight>   q_norm_mod        {*this, "q_norm"};
    core::Submodule<NormWeight>   k_norm_mod        {*this, "k_norm"};
    core::Submodule<NormWeight>   q_a_layernorm_mod {*this, "q_a_layernorm"};
    core::Submodule<NormWeight>   kv_a_layernorm_mod{*this, "kv_a_layernorm"};
    core::Parameter              sinks_             {*this, "sinks"};

    // Convenience tensor accessors
    Tensor* q_norm() const;
    Tensor* k_norm() const;
    Tensor* q_a_layernorm() const;
    Tensor* kv_a_layernorm() const;
    Tensor* sinks() const;

    int  window_size() const { return window_size_; }
    bool is_mla() const { return mla_.kv_lora_rank > 0; }

private:
    int      hidden_dim_{};
    int      head_dim_{};
    int      head_num_{};
    int      kv_head_num_{};
    MLAParam mla_{};
    bool     bias_{};
    bool     qk_norm_{};
    int      tp_size_{};
    int      tp_rank_{};
    DataType data_type_{};
    int      window_size_{};
    bool     sink_{};
    bool     attn_output_gate_{};
};

}  // namespace turbomind
