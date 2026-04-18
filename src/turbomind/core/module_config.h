// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

// Transitional header — config structs pending migration to their _weight.h files.
// Infrastructure (TM_MEMBER, TM_PTR, TM_FOR_EACH, ModuleConfig, ModuleListConfig) lives in module.h.

#include "src/turbomind/core/module.h"

namespace turbomind::core {

struct AttentionConfig: ModuleConfig {
    AttentionConfig(): ModuleConfig{"AttentionWeight"} {}

    #define ATTENTION_FIELDS(X) \
        X(int,      hidden_dim) \
        X(int,      head_dim) \
        X(int,      head_num) \
        X(int,      kv_head_num) \
        X(int,      kv_lora_rank) \
        X(int,      q_lora_rank) \
        X(int,      qk_rope_dim) \
        X(int,      v_head_dim) \
        X(bool,     has_bias) \
        X(bool,     qk_norm) \
        X(int,      tp_size) \
        X(int,      tp_rank) \
        X(DataType, data_type) \
        X(int,      window_size, -1) \
        X(bool,     attn_sink) \
        X(bool,     attn_output_gate) \
        X(int,      rope_dim) \
        X(int,      repeat_kv) \
        X(int,      qk_nope_dim)

    ATTENTION_FIELDS(TM_MEMBER)
    TM_FOR_EACH(AttentionConfig, ATTENTION_FIELDS)

    #undef ATTENTION_FIELDS
};

struct FfnConfig: ModuleConfig {
    FfnConfig(): ModuleConfig{"FfnWeight"} {}

    #define FFN_FIELDS(X) \
        X(int,      hidden_dim) \
        X(int,      inter_size) \
        X(bool,     has_bias) \
        X(int,      tp_size) \
        X(int,      tp_rank) \
        X(DataType, data_type) \
        X(int,      act_type) \
        X(bool,     fuse_silu) \
        X(bool,     fused_moe)

    FFN_FIELDS(TM_MEMBER)
    TM_FOR_EACH(FfnConfig, FFN_FIELDS)

    #undef FFN_FIELDS
};

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
