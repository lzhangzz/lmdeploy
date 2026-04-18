// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/norm_weight.h"

namespace turbomind::core {

struct DeltaNetConfig: ModuleConfig {
    DeltaNetConfig(): ModuleConfig{"DeltaNetWeight"} {}

    #define DELTANET_FIELDS(X) \
        X(int,      hidden_dim) \
        X(int,      num_k_heads) \
        X(int,      num_v_heads) \
        X(int,      key_head_dim) \
        X(int,      value_head_dim) \
        X(int,      d_conv, 4) \
        X(bool,     has_bias) \
        X(int,      tp_size) \
        X(int,      tp_rank) \
        X(DataType, data_type)

    DELTANET_FIELDS(TM_MEMBER)
    TM_FOR_EACH(DeltaNetConfig, DELTANET_FIELDS)

    #undef DELTANET_FIELDS
};

}  // namespace turbomind::core

namespace turbomind {

/// Weight module for Gated DeltaNet (linear attention) layers.
class DeltaNetWeight: public core::Module {
public:
    const char* type() const override { return "DeltaNetWeight"; }

    DeltaNetWeight() = default;

    DeltaNetWeight(const core::DeltaNetConfig& cfg);

    void prepare() override;

    // --- X-macro field lists ---
#define DELTA_NET_WEIGHT_CHILDREN(X) \
    X(LinearWeight, in_proj_all) \
    X(LinearWeight, out_proj)    \
    X(NormWeight,   norm)

#define DELTA_NET_WEIGHT_PARAMS(X) \
    X(conv1d) \
    X(A_log)  \
    X(dt_bias)

    TM_MODULE_DECLARE(DeltaNetWeight, DELTA_NET_WEIGHT_CHILDREN, DELTA_NET_WEIGHT_PARAMS)

    // --- Config fields (public for runtime access) ---
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
