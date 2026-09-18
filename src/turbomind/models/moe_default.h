// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <memory>

#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/moe.h"

namespace turbomind {

struct Context;

/// Per-rank state of the AG/RS default MoE implementation: routing scratch
/// for table-compaction routing. Created by the first MoeDefaultImpl in the
/// bind walk and rendezvoused through the BindEnv shared map under
/// "moe.default".
struct MoeDefaultState {
    MoeDefaultState(const EngineParam& engine, const Context& ctx, const Moe& weights);
    ~MoeDefaultState();

    // Engine geometry the transport was sized from
    int  mlp_tp_size{};
    int  ep_size{};
    int  ep_rank{};
    int  max_token_num{};  // routing-buffer row bound
    int& is_warm_up;

    // Routing buffers
    Buffer_<int>   h_offsets_;
    Buffer_<int>   masks_;
    Buffer_<int>   f2n_;
    Buffer_<int>   f2E_;
    Buffer_<int>   en2f_;
    Buffer_<float> scales_;
    Buffer_<int>   accum_;
    Buffer_<int>   offsets_;
};

/// The AG/RS default MoE implementation (`moe_a2a_backend = "default"`):
/// gate, table-compaction routing (including the warm-up random path),
/// grouped expert execution through the linked block, combine with
/// shared-expert fold.
class MoeDefaultImpl: public Moe {
public:
    MoeDefaultImpl(const core::MoeConfig& cfg):
        Moe(cfg)
    {
    }

    void bind(BindEnv& env) override;

    void Forward(const Tensor&           input,
                 Ref<Tensor>             output,
                 const bool*             token_mask,
                 const std::vector<int>& local_token_nums) override;

private:
    std::shared_ptr<MoeDefaultState> state_;  // shared per-rank state, captured in bind
};

}  // namespace turbomind
