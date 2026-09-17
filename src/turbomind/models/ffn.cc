// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/ffn.h"

#include "src/turbomind/core/context.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

Ffn::Ffn(const core::FfnConfig& cfg):
    hidden_dim{cfg.hidden_dim},
    inter_size{cfg.inter_size / cfg.tp_size},
    act_type{static_cast<ActivationType>(cfg.act_type)},
    is_fused_silu{cfg.fuse_silu && act_type == ActivationType::kSilu},
    is_expert_{cfg.is_expert},
    data_type_{cfg.data_type},
    tp_size{cfg.tp_size},
    tp_rank{cfg.tp_rank}
{
}

void Ffn::prepare()
{
    Module::prepare();
}

void Ffn::Forward(const Tensor& input, Ref<Tensor> output)
{
    TM_FUNCTION_SCOPE();

    NvtxScope scope("ffn");

    const int token_num = input.shape(0);

    const auto stream = core::Context::stream().handle();

    Tensor gating;
    Tensor inter;

    auto* fused     = w1w3.get();
    bool  use_fused = fused && fused->weight;

    Tensor inter_scales;
    Tensor unused_scales;

    if (use_fused) {
        Tensor mix;
        if (is_fused_silu && fused->output_dtype() == kFloat8_e4m3) {
            TM_SCOPE_CALL(fused->Forward(input, unused_scales, mix, inter_scales));
            gating = mix;  // FP8 fused output is already half-N (inter_size)
        }
        else {
            TM_SCOPE_CALL(fused->Forward(input, mix));
            gating = mix.slice({0, 0}, {(int)token_num, inter_size});
            if (!is_fused_silu) {
                inter = mix.slice({0, inter_size}, {(ssize_t)token_num, inter_size});
            }
        }
    }
    else {
        TM_SCOPE_CALL(w1->Forward(input, gating));
        TM_DEBUG_TENSOR(gating, "w1", 3);

        TM_SCOPE_CALL(w3->Forward(input, inter));
        TM_DEBUG_TENSOR(inter, "w3", 3);
    }

    // When using the fused kernel (w1w3 + fused silu), activation is already applied.
    // Otherwise (separate w1/w3 or non-fused), apply activation explicitly.
    if (!use_fused || !is_fused_silu) {
        // gate' = silu(gate) * up
        Activation(gating, inter, act_type, stream);
        TM_CUDA_CHECK(cudaGetLastError());
        TM_DEBUG_TENSOR(gating, "act", 3);
    }

    {  // w2(x)
        NvtxScope scope("w2");
        if (inter_scales) {
            TM_SCOPE_CALL(w2->Forward(gating, inter_scales, output, unused_scales));
        }
        else {
            TM_SCOPE_CALL(w2->Forward(gating, output));
        }
    }
}

void Ffn::Forward(const Tensor& input, const GroupedRoute& route, Ref<Tensor> output)
{
    TM_FUNCTION_SCOPE();

    const auto st = core::Context::stream().handle();

    Tensor unused_scales;

    if (w1w3) {
        Tensor inter;
        Tensor inter_scales;
        if (is_fused_silu && w1w3->output_dtype() == kFloat8_e4m3) {
            TM_SCOPE_CALL(w1w3->Forward(input, unused_scales, route.indices, route.offsets, inter, inter_scales));
            TM_SCOPE_CALL(w2->Forward(
                inter.slice({0, 0}, {-1, inter_size}), inter_scales, {}, route.offsets, output, unused_scales));
        }
        else {
            TM_SCOPE_CALL(w1w3->Forward(input, route.indices, route.offsets, inter));

            if (!is_fused_silu) {
                TM_SCOPE_CALL(Activation(inter, w1w3->bias, route.f2E, act_type, route.num_valid_tokens, st));
            }

            TM_SCOPE_CALL(w2->Forward(inter.slice({0, 0}, {-1, inter_size}), {}, route.offsets, output));
        }
    }
    else {
        // Separate w1/w3 path
        Tensor gating;
        TM_SCOPE_CALL(w1->Forward(input, route.indices, route.offsets, gating));

        Tensor up;
        TM_SCOPE_CALL(w3->Forward(input, route.indices, route.offsets, up));

        TM_SCOPE_CALL(Activation(gating, up, act_type, route.num_valid_tokens, st));

        TM_SCOPE_CALL(w2->Forward(gating, {}, route.offsets, output));
    }
}

TM_MODULE_REGISTER_AS("FfnWeight", Ffn, core::FfnConfig);

TM_MODULE_METHODS(Ffn, FFN_CHILDREN, FFN_PARAMS)

}  // namespace turbomind
