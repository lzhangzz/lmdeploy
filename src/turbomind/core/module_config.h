// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

// Transitional header — config structs pending migration to their _weight.h files.
// Infrastructure (TM_MEMBER, TM_PTR, TM_FOR_EACH, ModuleConfig, ModuleListConfig) lives in module.h.

#include "src/turbomind/core/module.h"

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

struct NormConfig: ModuleConfig {
    NormConfig(): ModuleConfig{"NormWeight"} {}

    #define NORM_FIELDS(X) \
        X(int,      dim) \
        X(DataType, data_type)

    NORM_FIELDS(TM_MEMBER)
    TM_FOR_EACH(NormConfig, NORM_FIELDS)

    #undef NORM_FIELDS
};

struct DecoderLayerConfig: ModuleConfig {
    DecoderLayerConfig(): ModuleConfig{"DecoderLayerWeight"} {}
    template<typename Visitor>
    static void for_each(Visitor&&) {}
};

}  // namespace turbomind::core
