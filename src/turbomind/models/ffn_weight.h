// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/core/module_config.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

class FfnWeight: public core::Module {
public:
    const char* type() const override { return "FfnWeight"; }

    FfnWeight() = default;

    FfnWeight(const core::FfnConfig& cfg);

    void prepare() override;

    // --- X-macro child members ---
#define FFN_WEIGHT_CHILDREN(X) \
    X(LinearWeight, w1)        \
    X(LinearWeight, w3)        \
    X(LinearWeight, w2)        \
    X(LinearWeight, w1w3)

    FFN_WEIGHT_CHILDREN(TM_CHILD_MEMBER)

    // Generated overrides
    Module* add_child(std::string name, std::unique_ptr<Module> child) override;
    Module* child(const std::string& name) const override;
    void    for_each_child(std::function<void(const char*, Module*)> visitor) const override;

    int            inter_size() const { return inter_size_; }
    ActivationType act_type() const { return act_type_; }
    bool           is_fused_silu() const { return is_fused_silu_; }

    /// Set grouped-GEMM mode for MoE (affects weight conversion layout).
    void set_fused_moe(bool fused_moe) { is_fused_moe_ = fused_moe; }

    /// Override is_fused_silu_ (used by MoE block view after linking experts).
    void set_fused_silu(bool val) { is_fused_silu_ = val; }

private:
    int           hidden_dim_{};
    int           inter_size_{};
    bool          bias_{};
    int           tp_size_{};
    int           tp_rank_{};
    DataType      data_type_{};
    ActivationType act_type_{};
    bool          is_fused_silu_{};
    bool          is_fused_moe_{};
};

}  // namespace turbomind
