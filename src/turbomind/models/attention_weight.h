// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/norm_weight.h"
#include "src/turbomind/models/llama/llama_params.h"

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
        X(int,      qk_nope_dim) \
        X(float,    softmax_scale, 0.f) \
        X(bool,     use_logn_attn, false) \
        X(int,      max_position_embeddings, 0)

    ATTENTION_FIELDS(TM_MEMBER)
    TM_FOR_EACH(AttentionConfig, ATTENTION_FIELDS)

    #undef ATTENTION_FIELDS
};

}  // namespace turbomind::core

namespace turbomind {

class AttentionWeight: public core::Module {
public:
    const char* type() const override { return "AttentionWeight"; }

    AttentionWeight() = default;

    AttentionWeight(const core::AttentionConfig& cfg);

    void prepare() override;

    // --- X-macro field lists ---
#define ATTENTION_WEIGHT_CHILDREN(X) \
    X(LinearWeight, w_qkv)          \
    X(LinearWeight, wo)             \
    X(LinearWeight, q_proj)         \
    X(LinearWeight, q_a_proj)       \
    X(LinearWeight, q_b_proj)       \
    X(LinearWeight, kv_a_proj)      \
    X(NormWeight,   q_norm)         \
    X(NormWeight,   k_norm)         \
    X(NormWeight,   q_a_layernorm)  \
    X(NormWeight,   kv_a_layernorm)

#define ATTENTION_WEIGHT_PARAMS(X) \
    X(sinks)

    TM_MODULE_DECLARE(AttentionWeight, ATTENTION_WEIGHT_CHILDREN, ATTENTION_WEIGHT_PARAMS)

    bool is_mla() const { return mla_.kv_lora_rank > 0; }

    // --- Config fields (public for runtime access) ---
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
    float    softmax_scale_{};
    bool     use_logn_attn_{};
    int      max_position_embeddings_{};
};

}  // namespace turbomind
