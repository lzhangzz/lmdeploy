// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <memory>

#include <cuda_runtime.h>

#include <cublasLt.h>
#include <cublas_v2.h>

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/core.h"
#include "src/turbomind/models/llama/GEMMEngine.h"

namespace turbomind {

struct MoeState;
struct AttentionState;
struct DeltaNetState;

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
    core::Stream                    core_stream;
    core::Allocator                 allocator;
    cudaStream_t                    stream;
    std::unique_ptr<GEMMEngine>     gemm_engine;  // per-rank GEMM engine
    cudaDeviceProp                  device_prop;
    Communicators                   comm;  // initialize later
    std::unique_ptr<int>            is_warm_up;
    std::unique_ptr<MoeState>       moe_state_;        // per-rank MoE kind state
    std::unique_ptr<AttentionState> attention_state_;  // per-rank attention kind state
    std::unique_ptr<DeltaNetState>  deltanet_state_;   // per-rank GDN kind state

    Context(int device_id);
    ~Context();

    GEMMEngine& linear_engine() noexcept
    {
        return *gemm_engine;
    }

    MoeState* moe_state() const noexcept
    {
        return moe_state_.get();
    }
    // Defined in context.cc: the kind-state type is incomplete here.
    void set_moe_state(std::unique_ptr<MoeState> state) noexcept;

    AttentionState* attention_state() const noexcept
    {
        return attention_state_.get();
    }
    void set_attention_state(std::unique_ptr<AttentionState> state) noexcept;

    DeltaNetState* deltanet_state() const noexcept
    {
        return deltanet_state_.get();
    }
    void set_deltanet_state(std::unique_ptr<DeltaNetState> state) noexcept;
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
