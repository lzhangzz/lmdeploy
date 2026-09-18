// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/context.h"

#include "src/turbomind/models/attention.h"
#include "src/turbomind/models/moe.h"

namespace turbomind {

// Out-of-line: the kind-state members are incomplete in context.h, so both
// the constructor's exception-cleanup path and destruction need this TU.
Context::Context(int device_id):
    core_stream{core::Stream::create()},
    allocator{core::Allocator(core_stream, false)},
    stream{core_stream.handle()},
    comm{},  // value initialize
    is_warm_up{std::make_unique<int>()}
{
    core::ContextGuard guard{core_stream};
    gemm_engine = std::make_unique<GEMMEngine>();
    TM_CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
}

Context::~Context() = default;

void Context::set_moe_state(std::unique_ptr<MoeState> state) noexcept
{
    moe_state_ = std::move(state);
}

void Context::set_attention_state(std::unique_ptr<AttentionState> state) noexcept
{
    attention_state_ = std::move(state);
}

}  // namespace turbomind
