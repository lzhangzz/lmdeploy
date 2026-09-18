// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/context.h"

#include <utility>

#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

namespace {
// Ambient EngineParam of the module-build thread; parked/unparked by
// EngineParamGuard (entered by the Python weight-context guard).
thread_local const EngineParam* g_ambient_engine_param = nullptr;
}  // namespace

EngineParamGuard::EngineParamGuard(const EngineParam& engine_param):
    prev_{std::exchange(g_ambient_engine_param, &engine_param)}
{
}

EngineParamGuard::~EngineParamGuard()
{
    g_ambient_engine_param = prev_;
}

const EngineParam* EngineParamGuard::current()
{
    return g_ambient_engine_param;
}

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

}  // namespace turbomind
