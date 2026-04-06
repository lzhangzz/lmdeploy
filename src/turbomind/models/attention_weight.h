// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/norm_weight.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

class AttentionWeight: public core::Module<AttentionWeight> {
public:
    static constexpr const char* kTypeName = "AttentionWeight";
    const char* type() const override { return kTypeName; }

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

    // --- Typed child members ---
    LinearWeight* w_qkv_              = nullptr;
    LinearWeight* wo_                 = nullptr;
    LinearWeight* q_proj_             = nullptr;
    LinearWeight* q_a_proj_           = nullptr;
    LinearWeight* q_b_proj_           = nullptr;
    LinearWeight* kv_a_proj_          = nullptr;
    NormWeight*   q_norm_mod_         = nullptr;
    NormWeight*   k_norm_mod_         = nullptr;
    NormWeight*   q_a_layernorm_mod_  = nullptr;
    NormWeight*   kv_a_layernorm_mod_ = nullptr;
    NormWeight*   sinks_mod_          = nullptr;

    static constexpr auto kChildren = std::make_tuple(
        std::pair{"w_qkv",          &AttentionWeight::w_qkv_},
        std::pair{"wo",             &AttentionWeight::wo_},
        std::pair{"q_proj",         &AttentionWeight::q_proj_},
        std::pair{"q_a_proj",       &AttentionWeight::q_a_proj_},
        std::pair{"q_b_proj",       &AttentionWeight::q_b_proj_},
        std::pair{"kv_a_proj",      &AttentionWeight::kv_a_proj_},
        std::pair{"q_norm",         &AttentionWeight::q_norm_mod_},
        std::pair{"k_norm",         &AttentionWeight::k_norm_mod_},
        std::pair{"q_a_layernorm",  &AttentionWeight::q_a_layernorm_mod_},
        std::pair{"kv_a_layernorm", &AttentionWeight::kv_a_layernorm_mod_},
        std::pair{"sinks",          &AttentionWeight::sinks_mod_}
    );
    friend class core::Module<AttentionWeight>;

    // --- Typed accessors ---
    LinearWeight* w_qkv() const { return w_qkv_; }
    LinearWeight* wo() const { return wo_; }
    LinearWeight* q_proj() const { return q_proj_; }
    LinearWeight* q_a_proj() const { return q_a_proj_; }
    LinearWeight* q_b_proj() const { return q_b_proj_; }
    LinearWeight* kv_a_proj() const { return kv_a_proj_; }
    NormWeight*   q_norm_mod() const { return q_norm_mod_; }
    NormWeight*   k_norm_mod() const { return k_norm_mod_; }
    NormWeight*   q_a_layernorm_mod() const { return q_a_layernorm_mod_; }
    NormWeight*   kv_a_layernorm_mod() const { return kv_a_layernorm_mod_; }
    NormWeight*   sinks_mod() const { return sinks_mod_; }

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
