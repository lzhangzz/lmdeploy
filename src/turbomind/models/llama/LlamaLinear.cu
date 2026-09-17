// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/core/context.h"
#include "src/turbomind/core/core.h"
#include "src/turbomind/core/logger.h"
#include "src/turbomind/core/scope.h"

#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/types.h"

#include "src/turbomind/models/llama/GEMMEngine.h"
#include "src/turbomind/models/llama/LlamaLinear.h"

#include <memory>
#include <optional>

namespace turbomind {

using namespace gemm;

struct LlamaLinear::Impl {

    Impl(): owned_engine_{std::make_unique<GEMMEngine>()}, engine_{owned_engine_.get()} {}

    std::unique_ptr<GEMMEngine> owned_engine_;
    GEMMEngine*                 engine_;

    std::tuple<Tensor, MatrixLayout, Tensor, MatrixLayout> GetOperandB(const Linear& weight)
    {
        const Tensor& B      = weight.weight;
        const Tensor& V      = weight.scales;
        MatrixLayout  desc_B = weight.k_desc;
        MatrixLayout  desc_V = weight.q_desc;
        return {B, desc_B, V, desc_V};
    }

    std::tuple<Tensor, MatrixLayout, Tensor, MatrixLayout> GetOperandA(const Linear&       weight,
                                                                       const Tensor&       input,
                                                                       const Tensor&       input_scales,
                                                                       Buffer_<int>        indices,
                                                                       const Buffer_<int>& offsets)
    {
        auto st = core::Context::stream().handle();

        Tensor A;
        Tensor U;

        // Size-0 / null Buffer must not count as indexed MoE (w2 / down path).
        const bool has_indices = static_cast<bool>(indices) && indices.size() > 0;
        const int  m           = has_indices ? indices.size() : input.shape(0);

        TM_CHECK(weight.family);
        weight.family->ConvertInput(A, U, weight, input, input_scales, st);

        MatrixLayout desc_A{A.dtype(), gemm::Order::kRowMajor, m, (int)A.shape(1), (int)A.stride(0)};
        MatrixLayout desc_U{};
        if (U) {
            desc_U = {U.dtype(), kColMajor, (int)U.shape(1), (int)U.shape(0), (int)U.stride(0)};
        }
        if (offsets) {
            desc_A.num = desc_U.num = weight.k_desc.num;
            desc_A.offsets = desc_U.offsets = const_cast<int*>(offsets.data());
        }
        if (has_indices) {
            desc_A.idxs = desc_U.idxs = const_cast<int*>(indices.data());
        }

        return {A, desc_A, U, desc_U};
    }

    Operation GetOperation(const Linear& weight) const
    {
        Operation operation{};
        operation.dispatch  = engine_->dispatch_policy();
        operation.epilogue  = weight.epilogue;
        operation.quant_a   = MakeQuantDesc(weight.input_format);
        operation.quant_b   = MakeQuantDesc(weight.weight_format);
        operation.batch_dim = 0;
        operation.family    = weight.family->id;
        return operation;
    }

    OutputSpec GetOutputSpec(const Tensor& input, const Linear& weight, const Buffer_<int>& indices) const
    {
        const bool grouped = weight.k_desc.ld == 0;
        const bool indexed = static_cast<bool>(indices) && indices.size() > 0;
        const int  k       = input.shape(-1);
        const int  m       = indexed ? indices.size() : input.size() / k;

        auto output_shape = input.shape();
        if (grouped) {
            output_shape = {m, weight.output_dim};
        }
        else {
            output_shape.back() = weight.output_dim;
        }
        return weight.family->output_spec(core::Layout{std::move(output_shape)}, weight.epilogue);
    }

    Gemm::Arguments GetArguments(Tensor&             A,
                                 Tensor&             U,
                                 const OutputSpec&   output_spec,
                                 const Tensor&       input,
                                 const Tensor&       input_scales,
                                 const Linear&       weight,
                                 const Buffer_<int>& indices,
                                 const Buffer_<int>& offsets,
                                 Tensor&             output,
                                 Tensor&             output_scales)
    {
        Tensor       in = input.view({-1, input.shape(-1)});
        MatrixLayout desc_A;
        MatrixLayout desc_U;
        std::tie(A, desc_A, U, desc_U) = GetOperandA(weight, in, input_scales, indices, offsets);
        auto&& [B, desc_B, V, desc_V]  = GetOperandB(weight);

        const Tensor& global_scale = weight.global_scale;
        MatrixLayout  global_scale_desc{};
        if (global_scale) {
            global_scale_desc     = {global_scale.dtype(), kRowMajor, 1, 1, weight.k_desc.ld == 0 ? 0 : 1};
            global_scale_desc.num = weight.k_desc.num;
        }

        Tensor& D = output;
        if (!D) {
            D = Tensor{output_spec.layout, output_spec.dtype, kDEVICE};
        }
        Tensor D_gemm = D.view({-1, D.shape(-1)});

        MatrixLayout desc_D{
            D_gemm.dtype(),
            kRowMajor,
            static_cast<int>(D_gemm.shape(0)),
            weight.output_dim,
            static_cast<int>(D_gemm.stride(0)),
        };

        if (offsets) {
            desc_D.num     = desc_B.num;
            desc_D.offsets = const_cast<int*>(offsets.data());
        }

        Tensor&      W = output_scales;
        MatrixLayout desc_W{};
        void*        W_ptr = nullptr;
        if (output_spec.scales_dtype != kNull) {
            if (!W) {
                W = Tensor{output_spec.scales_layout, output_spec.scales_dtype, kDEVICE};
            }
            desc_W = {W.dtype(),
                      kColMajor,
                      static_cast<int>(W.shape(1)),
                      static_cast<int>(W.shape(0)),
                      static_cast<int>(W.stride(0))};
            W_ptr  = W.raw_data();
        }

        Gemm::Arguments args{};
        args.operation         = GetOperation(weight);
        args.A                 = A.raw_data();
        args.Adesc             = desc_A;
        args.U                 = U.data_or((void*)nullptr);
        args.Udesc             = desc_U;
        args.B                 = B.raw_data();
        args.Bdesc             = desc_B;
        args.V                 = V.data_or((void*)nullptr);
        args.Vdesc             = desc_V;
        args.global_scale      = global_scale.data_or((void*)nullptr);
        args.global_scale_desc = global_scale_desc;
        args.C                 = D_gemm.raw_data();
        args.Cdesc             = desc_D;
        args.D                 = D_gemm.raw_data();
        args.Ddesc             = desc_D;
        args.W                 = W_ptr;
        args.Wdesc             = desc_W;
        args.workspace         = engine_->workspace();
        args.stream            = core::Context::stream().handle();
        return args;
    }

    void Forward(const ExecPlan&     plan,
                 const Tensor&       input,
                 const Tensor&       input_scales,
                 const Linear&       weight,
                 const Buffer_<int>& indices,
                 const Buffer_<int>& offsets,
                 Tensor&             output,
                 Tensor&             output_scales)
    {
        TM_FUNCTION_SCOPE();
        Tensor          A;
        Tensor          U;
        Gemm::Arguments args = GetArguments(
            A, U, plan.output_spec(), input, input_scales, weight, indices, offsets, output, output_scales);
        const int ec = engine_->gemm().Run(plan, args);
        if (ec) {
            TM_LOG_ERROR("{}: {}", __PRETTY_FUNCTION__, ec);
        }
    }
};

LlamaLinear::LlamaLinear(): impl_{std::make_shared<Impl>()} {}

gemm::Gemm& LlamaLinear::gemm() noexcept
{
    return impl_->engine_->gemm();
}

void LlamaLinear::Forward(const gemm::ExecPlan& plan,
                          const Tensor&         input,
                          const Tensor&         input_scales,
                          const Linear&         weight,
                          const Buffer_<int>&   indices,
                          const Buffer_<int>&   offsets,
                          Ref<Tensor>           output,
                          Ref<Tensor>           output_scales)
{
    impl_->Forward(plan, input, input_scales, weight, indices, offsets, output.get(), output_scales.get());
}

OutputSpec LlamaLinear::GetOutputSpec(const Tensor& input, const Linear& weight, const Buffer_<int>& indices) const
{
    return impl_->GetOutputSpec(input, weight, indices);
}

std::optional<ExecPlan> LlamaLinear::GetExecPlan(const Tensor&       input,
                                                 const Linear&       weight,
                                                 const Buffer_<int>& indices,
                                                 const Buffer_<int>& offsets)
{
    const bool indexed = static_cast<bool>(indices) && indices.size() > 0;
    const int  k       = input.shape(-1);
    const int  m       = indexed ? indices.size() : input.size() / k;

    OutputSpec output_spec = impl_->GetOutputSpec(input, weight, indices);

    MatrixLayout desc_A{weight.input_format.dtype, kRowMajor, m, k, k};
    MatrixLayout desc_U{};
    MatrixLayout desc_B = weight.k_desc;
    MatrixLayout desc_V = weight.q_desc;
    MatrixLayout desc_D{
        output_spec.dtype, kRowMajor, m, weight.output_dim, static_cast<int>(output_spec.layout.stride(-2))};

    if (offsets) {
        desc_A.num = desc_U.num = desc_D.num = desc_B.num;
        desc_A.offsets = desc_U.offsets = desc_D.offsets = const_cast<int*>(offsets.data());
    }
    if (indexed) {
        desc_A.idxs = desc_U.idxs = const_cast<int*>(indices.data());
    }

    Gemm::Arguments args{};
    args.operation = impl_->GetOperation(weight);
    args.Adesc     = desc_A;
    args.Udesc     = desc_U;
    args.Bdesc     = desc_B;
    args.Vdesc     = desc_V;
    args.Cdesc     = desc_D;
    args.Ddesc     = desc_D;
    args.workspace = impl_->engine_->workspace();

    auto& gemm = impl_->engine_->gemm();
    auto  plan = gemm.GetExecPlan(args);
    if (plan) {
        plan->output_spec_ = std::move(output_spec);
    }
    return plan;
}

std::optional<ExecPlan> LlamaLinear::Tune(const Tensor&       input,
                                          const Tensor&       input_scales,
                                          const Linear&       weight,
                                          const Buffer_<int>& indices,
                                          const Buffer_<int>& offsets,
                                          Ref<Tensor>         output,
                                          Ref<Tensor>         output_scales)
{
    OutputSpec      output_spec = impl_->GetOutputSpec(input, weight, indices);
    Tensor          A;
    Tensor          U;
    Gemm::Arguments args = impl_->GetArguments(
        A, U, output_spec, input, input_scales, weight, indices, offsets, output.get(), output_scales.get());
    args.operation.dispatch = DispatchPolicy::kMeasure;
    auto& gemm              = impl_->engine_->gemm();
    auto  exec_plan         = gemm.Tune(args);
    if (!exec_plan) {
        return std::nullopt;
    }
    exec_plan->output_spec_ = std::move(output_spec);
    if (gemm.Run(*exec_plan, args)) {
        return std::nullopt;
    }
    return exec_plan;
}

[[maybe_unused]] int LlamaLinear::Export(std::ostream& os)
{
    return impl_->engine_->Export(os);
}

[[maybe_unused]] int LlamaLinear::Import(std::istream& is)
{
    return impl_->engine_->Import(is);
}

}  // namespace turbomind
