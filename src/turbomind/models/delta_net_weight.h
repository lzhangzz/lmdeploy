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

    Module* ensure_child(const std::string& segment) override;
    void prepare() override;

    // --- Typed child accessors ---
    LinearWeight* in_proj_all() const { return static_cast<LinearWeight*>(child("in_proj_all")); }
    LinearWeight* out_proj() const { return static_cast<LinearWeight*>(child("out_proj")); }
    Tensor*       conv1d() const;
    Tensor*       A_log() const;
    Tensor*       dt_bias() const;
    Tensor*       norm() const;

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
