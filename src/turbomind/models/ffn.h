// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/kernels/activation.h"
#include "src/turbomind/models/linear.h"

namespace turbomind::core {

struct FfnConfig: ModuleConfig {
    FfnConfig(): ModuleConfig{"FfnWeight"} {}

#define FFN_FIELDS(X)                                                                                                  \
    X(int, hidden_dim)                                                                                                 \
    X(int, inter_size)                                                                                                 \
    X(int, act_type)                                                                                                   \
    X(bool, fuse_silu)                                                                                                 \
    X(bool, is_expert)                                                                                                 \
    X(DataType, data_type)                                                                                             \
    X(int, tp_size)                                                                                                    \
    X(int, tp_rank)

    FFN_FIELDS(TM_MEMBER)
    TM_FOR_EACH(FfnConfig, FFN_FIELDS)

#undef FFN_FIELDS
};

}  // namespace turbomind::core

namespace turbomind {

/// Routing tables shared by the grouped (MoE expert-block) Ffn::Forward.
struct GroupedRoute {
    Buffer_<int> indices;           // grouped row -> source token slot
    Buffer_<int> offsets;           // per-local-expert row offsets [E+1]
    Buffer_<int> f2E;               // grouped row -> expert id (activation grouping)
    const int*   num_valid_tokens;  // device pointer: routed token count
};

/// Executable FFN module: owns the whole feed-forward — fused (w1w3) and
/// separate (w1/w3) projections, the FP8 fused-activation path, activation,
/// and the down projection — for dense and grouped inputs, with the branching
/// internal. The GEMMs ride the Linear children; bind() is the default
/// recursive walk capturing their engines.
class Ffn: public core::Module {
public:
    const char* type() const override
    {
        return "FfnWeight";
    }

    Ffn() = default;

    Ffn(const core::FfnConfig& cfg);

    /// Dense: y = W2·act(W1x, W3x). Activation internal (fused-silu/FP8 and
    /// w1/w3-separate branches). Output filled when provided.
    void Forward(const Tensor& input, Ref<Tensor> output);

    /// Grouped (the MoE expert block): routing tables drive expert-grouped
    /// GEMMs and the grouped activation. Same internal branches.
    void Forward(const Tensor& input, const GroupedRoute& route, Ref<Tensor> output);

    // --- X-macro child members ---
#define FFN_CHILDREN(X)                                                                                                \
    X(Linear, w1)                                                                                                      \
    X(Linear, w3)                                                                                                      \
    X(Linear, w2)                                                                                                      \
    X(Linear, w1w3)

#define FFN_PARAMS(X)

    TM_MODULE_DECLARE(Ffn, FFN_CHILDREN, FFN_PARAMS)

    int            hidden_dim{};
    int            inter_size{};
    ActivationType act_type{};
    bool           is_fused_silu{};
    int            tp_size{};
    int            tp_rank{};

private:
    bool     is_expert_{};
    DataType data_type_{};
};

}  // namespace turbomind
