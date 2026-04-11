// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/core/module_config.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/norm_weight.h"
#include "src/turbomind/models/llama/llama_params.h"

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
