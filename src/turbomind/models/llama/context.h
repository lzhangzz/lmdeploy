// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include <cuda_runtime.h>

#include <cublasLt.h>
#include <cublas_v2.h>

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/core.h"
#include "src/turbomind/models/llama/GEMMEngine.h"

namespace turbomind {

struct EngineParam;
class CacheRegistry;

struct Communicators {
    comm::HostComm h_global;
    comm::HostComm h_comm;
    comm::HostComm h_tp_group;
    comm::HostComm h_dp_group;
    comm::HostComm h_ep_group;

    comm::DeviceComm d_comm;
    int              d_tp_group;
    int              d_cp_group;
    int              d_dp_group;
    int              d_mlp_group;
    int              d_node_group;
};

// Execution context for the model
struct Context {
    core::Stream                core_stream;
    core::Allocator             allocator;
    cudaStream_t                stream;
    std::unique_ptr<GEMMEngine> gemm_engine;  // per-rank GEMM engine
    cudaDeviceProp              device_prop;
    Communicators               comm;  // initialize later
    std::unique_ptr<int>        is_warm_up;

    Context(int device_id);
    ~Context();

    GEMMEngine& linear_engine() noexcept
    {
        return *gemm_engine;
    }
};

/// Everything a binding module needs, carried through the bind walk.
/// `shared` is the transient kind-state rendezvous: the first module of a
/// kind creates the kind's per-rank state and parks it under the kind's
/// key; the rest fetch and cast. The map lives only for the walk (owned by
/// the engine's bind step); modules co-own their state via shared_ptr
/// afterwards.
struct BindEnv {
    const EngineParam&                                      engine;
    Context&                                                context;
    CacheRegistry&                                          registry;
    int&                                                    is_warm_up;
    std::unordered_map<std::string, std::shared_ptr<void>>& shared;
    int&                                                    phases;
};

/// Parks the rank's EngineParam ambiently on the module-build thread for
/// the guard's lifetime — entered by the weight-context guard beside the
/// stream/allocator it pushes, so registry factories can read engine
/// policy and sizing at module construction. Nullptr outside a guard
/// (standalone paths).
class EngineParamGuard {
public:
    explicit EngineParamGuard(const EngineParam& engine_param);
    ~EngineParamGuard();

    EngineParamGuard(const EngineParamGuard&)            = delete;
    EngineParamGuard& operator=(const EngineParamGuard&) = delete;

    /// The ambient EngineParam of the current thread; nullptr when no
    /// engine-backed guard is active.
    static const EngineParam* current();

private:
    const EngineParam* prev_;
};

inline Allocator GetSymmAllocator(const comm::DeviceComm& comm)
{
    TM_CHECK(comm);
    return core::SimpleAllocator::Create(
        [&comm](auto size) {
            auto p = comm->Allocate(size);
            comm->Register(p, size);
            return p;
        },
        [&comm](void* p, auto size) {
            comm->Deregister(p);
            comm->Free(p);
        },
        kDEVICE);
}

}  // namespace turbomind
