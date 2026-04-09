// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <string>

#include "src/turbomind/core/data_type.h"

namespace turbomind::core {

/// Base class for all module config structs. Carries the module type name
/// used by the registry for dispatch.
struct ModuleConfig {
    std::string module_type;
};
struct LinearConfig: ModuleConfig {
    LinearConfig(): ModuleConfig{"LinearWeight"} {}
    int      input_dim{};
    int      output_dim{};
    DataType data_type{};
    bool     has_bias{};
};

struct AttentionConfig: ModuleConfig {
    AttentionConfig(): ModuleConfig{"AttentionWeight"} {}
    int      hidden_dim{};
    int      head_dim{};
    int      head_num{};
    int      kv_head_num{};
    int      kv_lora_rank{};
    int      q_lora_rank{};
    int      qk_rope_dim{};
    int      v_head_dim{};
    bool     has_bias{};
    bool     qk_norm{};
    int      tp_size{};
    int      tp_rank{};
    DataType data_type{};
    int      window_size{-1};
    bool     attn_sink{};
    bool     attn_output_gate{};
};

struct FfnConfig: ModuleConfig {
    FfnConfig(): ModuleConfig{"FfnWeight"} {}
    int      hidden_dim{};
    int      inter_size{};
    bool     has_bias{};
    int      tp_size{};
    int      tp_rank{};
    DataType data_type{};
    int      act_type{};
    bool     fuse_silu{};
    bool     fused_moe{};
};

struct MoeConfig: ModuleConfig {
    MoeConfig(): ModuleConfig{"MoeWeight"} {}
    int            layer_id{};
    int            method{};
    int            experts_per_token{};
    int            inter_size{};
    bool           norm_topk_prob{};
    bool           shared_gate{};
    double         routed_scale{};
    bool           router_bias{};
    int            topk_group{};
    std::string    topk_method{};
    int            n_group{};
    std::string    scoring_func{};
    int            router_n_groups{};
    int            expert_num{};
    int            hidden_dim{};
    bool           mlp_bias{};
    DataType       data_type{};
    int            tp_size{};
    int            tp_rank{};
    int            act_type{};
    bool           fuse_silu{};
};

struct DeltaNetConfig: ModuleConfig {
    DeltaNetConfig(): ModuleConfig{"DeltaNetWeight"} {}
    int      hidden_dim{};
    int      num_k_heads{};
    int      num_v_heads{};
    int      key_head_dim{};
    int      value_head_dim{};
    int      d_conv{4};
    bool     has_bias{};
    int      tp_size{};
    int      tp_rank{};
    DataType data_type{};
};

struct ModuleListConfig: ModuleConfig {
    ModuleListConfig(): ModuleConfig{"ModuleList"} {}
};

struct NormConfig: ModuleConfig {
    NormConfig(): ModuleConfig{"NormWeight"} {}
    int      dim{};
    DataType data_type{};
};

struct DecoderLayerConfig: ModuleConfig {
    DecoderLayerConfig(): ModuleConfig{"DecoderLayerWeight"} {}
};

}  // namespace turbomind::core
