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

    // --- Typed child members ---
    LinearWeight* in_proj_all_ = nullptr;
    LinearWeight* out_proj_    = nullptr;
    NormWeight*   conv1d_      = nullptr;
    NormWeight*   A_log_       = nullptr;
    NormWeight*   dt_bias_     = nullptr;
    NormWeight*   norm_        = nullptr;


    // --- Typed accessors ---
    LinearWeight* in_proj_all() const { return in_proj_all_; }
    LinearWeight* out_proj() const { return out_proj_; }
    NormWeight*   conv1d() const { return conv1d_; }
    NormWeight*   A_log() const { return A_log_; }
    NormWeight*   dt_bias() const { return dt_bias_; }
    NormWeight*   norm() const { return norm_; }

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
