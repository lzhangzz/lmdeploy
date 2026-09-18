// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <memory>
#include <vector>

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/models/ffn.h"

namespace turbomind {

class Moe;

void LinkLinearExperts(const std::vector<Linear*>& experts, Linear& destination);

}  // namespace turbomind

namespace turbomind::core {

struct MoeConfig: ModuleConfig {
    MoeConfig(): ModuleConfig{"MoeWeight"} {}

#define MOE_FIELDS(X)                                                                                                  \
    X(int, expert_num)                                                                                                 \
    X(int, experts_per_token)                                                                                          \
    X(int, act_type)                                                                                                   \
    X(bool, fuse_silu)                                                                                                 \
    X(bool, norm_topk_prob)                                                                                            \
    X(std::string, topk_method)                                                                                        \
    X(std::string, scoring_func)                                                                                       \
    X(int, topk_group)                                                                                                 \
    X(int, n_group)                                                                                                    \
    X(int, router_n_groups)                                                                                            \
    X(double, routed_scale)                                                                                            \
    X(int, ep_size, 1)                                                                                                 \
    X(int, ep_rank, 0)                                                                                                 \
    X(DataType, data_type)

    MOE_FIELDS(TM_MEMBER)
    TM_FOR_EACH(MoeConfig, MOE_FIELDS)

#undef MOE_FIELDS
};

}  // namespace turbomind::core

namespace turbomind {

/// Executable MoE module base: the model's routing parameters, the gate,
/// the shared-expert fold, and grouped expert execution through the linked
/// block. The transport is the Implementation axis: the AG/RS default
/// (MoeDefaultImpl) and the DeepEP all-to-all (MoeDeepEPImpl) are
/// executable subclasses of this base, selected at construction by the
/// registry factory under the frozen "MoeWeight" string from the ambient
/// EngineParam's moe_a2a_backend (see ADR 0001).
class Moe: public core::Module {
public:
    const char* type() const override
    {
        return "MoeWeight";
    }

    Moe(const core::MoeConfig& cfg);

    void prepare() override;
    void link_block();

    /// The routed-expert block is built inside prepare() and is not an
    /// X-macro child, so the bind walk must descend into it explicitly.
    void bind(BindEnv& env) override
    {
        Module::bind(env);
        if (block_) {
            block_->bind(env);
        }
    }

    /// Full MoE forward: gate, route, transport, grouped experts, combine,
    /// shared-expert fold. token_mask: [tokens]; invalid tokens route
    /// nowhere. local_token_nums: per-attn-dp-rank token counts (DeepEP
    /// input partitioning; unused by the default transport).
    virtual void
    Forward(const Tensor& input, Ref<Tensor> output, const bool* token_mask, const std::vector<int>& local_token_nums) = 0;

    int num_experts() const
    {
        return expert_num;
    }
    int num_local_experts() const
    {
        return expert_num / ep_size;
    }
    int local_expert_offset() const
    {
        return ep_rank * num_local_experts();
    }

    // --- X-macro child members ---
#define MOE_CHILDREN(X)                                                                                                \
    X(Linear, gate)                                                                                                    \
    X(Linear, shared_gate)                                                                                             \
    X(Ffn, shared)                                                                                                     \
    X(core::ModuleList, experts)

#define MOE_PARAMS(X) X(score_correction_bias)

    TM_MODULE_DECLARE(Moe, MOE_CHILDREN, MOE_PARAMS)

    // --- Typed accessors ---
    Ffn* expert(int i) const;
    Ffn* block() const
    {
        return block_.get();
    }

    // --- Config fields (public for runtime access) ---
    int         expert_num{};
    int         experts_per_token{};
    bool        norm_topk_prob{};
    float       routed_scale{};
    int         topk_group{};
    std::string topk_method;
    int         n_group{};
    std::string scoring_func;
    int         router_n_groups{};
    int         ep_size{1};
    int         ep_rank{0};

    // Wire the shared meta-MoE pack (a ModelWeight::meta_experts entry) whose
    // routed gate/experts this layer aliases in prepare(). Called by the
    // Python loader; never set for a MoE that owns its routed weights.
    void set_meta_pack(const Moe* meta_pack)
    {
        meta_pack_ = meta_pack;
    }

protected:
    // Create gate/experts children that share the meta pack's tensors.
    void AliasRouted(const Moe& meta_pack);

    /// Router gate projection (logits, with bias applied).
    static Tensor_<float> Gate(const Tensor& input, Linear& gate);

    const Moe* meta_pack_{};  // non-owning; nullptr = own routed weights

    ActivationType act_type_{};
    bool           fuse_silu_act_{};

    DataType data_type_{};

    std::unique_ptr<Ffn> block_;
};

}  // namespace turbomind
