// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

class FfnWeight: public core::Module<FfnWeight> {
public:
    static constexpr const char* kTypeName = "FfnWeight";
    const char* type() const override { return kTypeName; }

    FfnWeight() = default;

    FfnWeight(int hidden_dim, int inter_size, bool bias, int tp_size, int tp_rank,
              DataType data_type, ActivationType act_type, bool fuse_silu_act);

    void prepare() override;

    // --- Typed child members ---
    mutable LinearWeight* w1_   = nullptr;
    mutable LinearWeight* w3_   = nullptr;
    mutable LinearWeight* w2_   = nullptr;
    mutable LinearWeight* w1w3_ = nullptr;

    static constexpr auto kChildren = std::make_tuple(
        std::pair{"w1",   &FfnWeight::w1_},
        std::pair{"w3",   &FfnWeight::w3_},
        std::pair{"w2",   &FfnWeight::w2_},
        std::pair{"w1w3", &FfnWeight::w1w3_}
    );
    friend class core::Module<FfnWeight>;

    // --- Typed accessors (now return cached pointers) ---
    LinearWeight* w1()   const { return w1_; }
    LinearWeight* w3()   const { return w3_; }
    LinearWeight* w2()   const { return w2_; }
    LinearWeight* w1w3() const { return w1w3_; }

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
