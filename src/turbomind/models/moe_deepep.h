// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <memory>

#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/moe.h"

namespace turbomind::comm {
class TokenDispatcher;
}

namespace turbomind {

struct Context;

/// Per-rank state of the DeepEP MoE implementation: the token dispatcher,
/// its top-k buffers, and the scales-copy stream/events. Created by the
/// first MoeDeepEPImpl in the bind walk and rendezvoused through the
/// BindEnv shared map under "moe.deepep".
struct MoeDeepEPState {
    MoeDeepEPState(const EngineParam& engine, const Context& ctx, const Moe& weights);
    ~MoeDeepEPState();

    // Engine geometry the transport was sized from
    int  mlp_tp_size{};
    int  ep_size{};
    int  ep_rank{};
    int  max_token_num{};           // routing-buffer row bound
    int  max_token_per_rank_num{};  // DeepEP per-rank token bound
    int& is_warm_up;

    // DeepEP transport
    Buffer_<float> topk_weights_;
    Buffer_<int>   topk_indices_;
    Buffer_<int>   f2n_;
    Buffer_<int>   f2E_;
    Buffer_<int>   en2f_;
    Buffer_<float> scales_;
    Buffer_<int>   offsets_;
    Stream         scales_copy_stream_;
    Event          scales_copy_ready_event_;
    Event          scales_copy_done_event_;
#ifdef USE_NCCL
    std::unique_ptr<comm::TokenDispatcher> dispatcher_;
#endif

private:
    void InitA2A(const Context& ctx, const Moe& weights);
    void CompileKernels(const Moe& weights);
};

/// The DeepEP all-to-all MoE implementation (`moe_a2a_backend = "deepep"`):
/// gate, top-k routing, token dispatch through the DeepEP dispatcher,
/// grouped expert execution on the dispatched tokens, combine, and
/// shared-expert fold. In-place output contract: writes its result through
/// the input tensor's memory, so callers must pass the same tensor as input
/// and output (the decoder does).
class MoeDeepEPImpl: public Moe {
public:
    MoeDeepEPImpl(const core::MoeConfig& cfg):
        Moe(cfg)
    {
    }

    void bind(BindEnv& env) override;

    void Forward(const Tensor&           input,
                 Ref<Tensor>             output,
                 const bool*             token_mask,
                 const std::vector<int>& local_token_nums) override;

private:
    std::shared_ptr<MoeDeepEPState> state_;  // shared per-rank state, captured in bind
};

}  // namespace turbomind
