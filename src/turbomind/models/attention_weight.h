// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/norm_weight.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

class AttentionWeight: public core::Module {
public:
    const char* type() const override { return "AttentionWeight"; }

    AttentionWeight() = default;

    /// Construct with config for lazy child creation.
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

    core::Module* ensure_child(const std::string& segment) override;

    void prepare() override;

    // --- Typed child accessors ---
    LinearWeight* w_qkv() const { return static_cast<LinearWeight*>(child("w_qkv")); }
    LinearWeight* wo() const { return static_cast<LinearWeight*>(child("wo")); }
    LinearWeight* q_proj() const { return static_cast<LinearWeight*>(child("q_proj")); }
    LinearWeight* q_a_proj() const { return static_cast<LinearWeight*>(child("q_a_proj")); }
    LinearWeight* q_b_proj() const { return static_cast<LinearWeight*>(child("q_b_proj")); }
    LinearWeight* kv_a_proj() const { return static_cast<LinearWeight*>(child("kv_a_proj")); }
    NormWeight*   q_norm_mod() const { return static_cast<NormWeight*>(child("q_norm")); }
    NormWeight*   k_norm_mod() const { return static_cast<NormWeight*>(child("k_norm")); }
    NormWeight*   q_a_layernorm_mod() const { return static_cast<NormWeight*>(child("q_a_layernorm")); }
    NormWeight*   kv_a_layernorm_mod() const { return static_cast<NormWeight*>(child("kv_a_layernorm")); }

    // Convenience: return the underlying tensor directly
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
