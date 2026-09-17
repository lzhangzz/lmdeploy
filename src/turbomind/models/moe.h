// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <memory>
#include <vector>

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/models/ffn.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind::comm {
class TokenDispatcher;
}

namespace turbomind {

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

class Moe;

/// Shared per-rank MoE transport/routing state — the retired executors'
/// state, held once by the Context (kind state) and bound to every Moe
/// module via bind(). The default and DeepEP transports are mutually
/// exclusive per rank; only the selected branch allocates.
struct MoeState {
    MoeState(const EngineParam& engine, const Context& ctx, const Moe& weights);
    ~MoeState();

    bool use_a2a{};  // DeepEP all-to-all transport (else table compaction)

    // Engine geometry the transports were sized from
    int  mlp_tp_size{};
    int  ep_size{};
    int  ep_rank{};
    int  max_token_num{};           // routing-buffer row bound
    int  max_token_per_rank_num{};  // DeepEP per-rank token bound
    int& is_warm_up;

    // Default-transport routing buffers
    Buffer_<int>   h_offsets_;
    Buffer_<int>   masks_;
    Buffer_<int>   f2n_;
    Buffer_<int>   f2E_;
    Buffer_<int>   en2f_;
    Buffer_<float> scales_;
    Buffer_<int>   accum_;
    Buffer_<int>   offsets_;

    // DeepEP transport
    Buffer_<float> topk_weights_;
    Buffer_<int>   topk_indices_;
    Stream         scales_copy_stream_;
    Event          scales_copy_ready_event_;
    Event          scales_copy_done_event_;
#ifdef USE_NCCL
    std::unique_ptr<comm::TokenDispatcher> dispatcher_;
#endif

private:
    void InitDefault(const Moe& weights);
    void InitA2A(const Context& ctx, const Moe& weights);
    void CompileKernels(const Moe& weights);
};

/// Executable MoE module: gate, routing (including the warm-up random
/// path), transport gather/scatter with default-vs-DeepEP as an internal
/// branch over the shared MoeState, grouped expert execution through the
/// linked block, combine, and shared-expert fold.
class Moe: public core::Module {
public:
    const char* type() const override
    {
        return "MoeWeight";
    }

    Moe() = default;

    Moe(const core::MoeConfig& cfg);

    void prepare() override;
    void link_block();

    /// The routed-expert block is built inside prepare() and is not an
    /// X-macro child, so the bind walk must descend into it explicitly.
    /// Also captures the shared per-rank MoeState. Defined in moe.cc
    /// (Context is incomplete here).
    void bind(Context& ctx) override;

    /// Full MoE forward: gate, route, transport, grouped experts, combine,
    /// shared-expert fold. token_mask: [tokens]; invalid tokens route
    /// nowhere. local_token_nums: per-attn-dp-rank token counts (DeepEP
    /// input partitioning; unused by the default transport). In-place
    /// contract: the DeepEP path writes its result through the input
    /// tensor's memory, so callers must pass the same tensor as input
    /// and output (the decoder does).
    void
    Forward(const Tensor& input, Ref<Tensor> output, const bool* token_mask, const std::vector<int>& local_token_nums);

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

private:
    // Create gate/experts children that share the meta pack's tensors.
    void AliasRouted(const Moe& meta_pack);

    Tensor_<float> Gate(const Tensor& input, Linear& gate);

    void ForwardDefault(const Tensor& input, Ref<Tensor> output, const bool* token_mask);
    void ForwardA2A(const Tensor&           input,
                    Ref<Tensor>             output,
                    const bool*             token_mask,
                    const std::vector<int>& local_token_nums);

    const Moe* meta_pack_{};  // non-owning; nullptr = own routed weights

    ActivationType act_type_{};
    bool           fuse_silu_act_{};

    DataType data_type_{};

    std::unique_ptr<Ffn> block_;

    MoeState* state_{};  // shared per-rank kind state, captured in bind
};

}  // namespace turbomind
