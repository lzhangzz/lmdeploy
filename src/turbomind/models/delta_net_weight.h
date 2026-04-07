// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/norm_weight.h"

namespace turbomind {

/// Weight module for Gated DeltaNet (linear attention) layers.
class DeltaNetWeight: public core::Module {
public:
    const char* type() const override { return "DeltaNetWeight"; }

    DeltaNetWeight() = default;

    DeltaNetWeight(int      hidden_dim,
                   int      num_k_heads,
                   int      num_v_heads,
                   int      key_head_dim,
                   int      value_head_dim,
                   int      d_conv,
                   bool     bias,
                   int      tp_size,
                   int      tp_rank,
                   DataType data_type);

    void prepare() override;

    const Tensor* conv1d() const { return conv1d_ ? conv1d_.ptr() : nullptr; }
    const Tensor* A_log() const { return A_log_ ? A_log_.ptr() : nullptr; }
    const Tensor* dt_bias() const { return dt_bias_ ? dt_bias_.ptr() : nullptr; }

    Tensor alloc(const std::string& param_name, const core::WeightSpec& spec) override;

    // --- Typed child members ---
    core::Submodule<LinearWeight> in_proj_all{*this, "in_proj_all"};
    core::Submodule<LinearWeight> out_proj{*this, "out_proj"};
    mutable core::Parameter      conv1d_{*this, "conv1d"};
    mutable core::Parameter      A_log_{*this, "A_log"};
    mutable core::Parameter      dt_bias_{*this, "dt_bias"};
    core::Submodule<NormWeight>   norm{*this, "norm"};

private:
    int      hidden_dim_{};
    int      num_k_heads_{};
    int      num_v_heads_{};
    int      key_head_dim_{};
    int      value_head_dim_{};
    int      d_conv_{};
    bool     bias_{};
    int      tp_size_{};
    int      tp_rank_{};
    DataType data_type_{};
};

}  // namespace turbomind
