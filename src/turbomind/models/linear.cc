// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/linear.h"

#include <tuple>
#include <utility>

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/logger.h"
#include "src/turbomind/core/registry.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/kernels/gemm/family.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

Linear::Linear(const core::LinearConfig& cfg):
    input_dim(cfg.input_dim),
    output_dim(cfg.output_dim),
    data_type(cfg.data_type),
    weight_format(cfg.format),
    has_bias_(cfg.has_bias)
{
}

gemm::QuantDesc MakeQuantDesc(const DataFormat& fmt)
{
    if (!fmt.is_quantized()) {
        return {gemm::QuantType::kNone, 0};
    }
    const int group_size = fmt.block_sizes.empty() ? 1 : fmt.block_sizes[0];
    if (fmt.dtype == kFloat8_e4m3 && fmt.block_sizes.size() > 1 && fmt.block_sizes[1] > 1) {
        return {gemm::QuantType::kB, group_size};
    }
    return {gemm::QuantType::kK, group_size};
}

void Linear::copy_metadata_to(Linear& dst) const
{
    dst.input_dim     = input_dim;
    dst.output_dim    = output_dim;
    dst.data_type     = data_type;
    dst.family        = family;
    dst.weight_format = weight_format;
    dst.input_format  = input_format;
    dst.output_format = output_format;
    dst.epilogue      = epilogue;
    dst.has_bias_     = has_bias_;
    dst.prepared_     = prepared_;
    dst.k_desc        = k_desc;
    dst.q_desc        = q_desc;
}

void Linear::set_plan(gemm::WeightPlan plan)
{
    plan_ = std::move(plan);
}

bool Linear::is_graph_compatible() const
{
    return TM_CHECK_NOTNULL(family)->is_graph_compatible();
}

void Linear::prepare()
{
    if (prepared_) {
        return;
    }
    TM_CHECK(plan_);
    plan_->pack(*this, core::Context::stream().handle());
    EnsureFloatDtype(bias, data_type);
    prepared_ = true;
}

// ----- engine binding -----

GEMMEngine& Linear::engine() const
{
    TM_CHECK(engine_ != nullptr) << "Linear '" << full_path() << "' used before bind(Context&)";
    return *engine_;
}

void Linear::bind(Context& ctx)
{
    engine_ = &ctx.linear_engine();
}

// ----- execution (absorbed from the retired executor) -----

std::tuple<Tensor, MatrixLayout, Tensor, MatrixLayout> Linear::GetOperandB() const
{
    return {weight, k_desc, scales, q_desc};
}

std::tuple<Tensor, MatrixLayout, Tensor, MatrixLayout> Linear::GetOperandA(const Tensor&       input,
                                                                           const Tensor&       input_scales,
                                                                           Buffer_<int>        indices,
                                                                           const Buffer_<int>& offsets) const
{
    auto st = core::Context::stream().handle();

    Tensor A;
    Tensor U;

    // Size-0 / null Buffer must not count as indexed MoE (w2 / down path).
    const bool has_indices = static_cast<bool>(indices) && indices.size() > 0;
    const int  m           = has_indices ? indices.size() : input.shape(0);

    TM_CHECK(family);
    family->ConvertInput(A, U, *this, input, input_scales, st);

    MatrixLayout desc_A{A.dtype(), gemm::Order::kRowMajor, m, (int)A.shape(1), (int)A.stride(0)};
    MatrixLayout desc_U{};
    if (U) {
        desc_U = {U.dtype(), gemm::kColMajor, (int)U.shape(1), (int)U.shape(0), (int)U.stride(0)};
    }
    if (offsets) {
        desc_A.num = desc_U.num = k_desc.num;
        desc_A.offsets = desc_U.offsets = const_cast<int*>(offsets.data());
    }
    if (has_indices) {
        desc_A.idxs = desc_U.idxs = const_cast<int*>(indices.data());
    }

    return {A, desc_A, U, desc_U};
}

gemm::Operation Linear::GetOperation() const
{
    gemm::Operation operation{};
    operation.dispatch  = engine().dispatch_policy();
    operation.epilogue  = epilogue;
    operation.quant_a   = MakeQuantDesc(input_format);
    operation.quant_b   = MakeQuantDesc(weight_format);
    operation.batch_dim = 0;
    operation.family    = family->id;
    return operation;
}

gemm::OutputSpec Linear::GetOutputSpec(const Tensor& input, const Buffer_<int>& indices) const
{
    const bool grouped = k_desc.ld == 0;
    const bool indexed = static_cast<bool>(indices) && indices.size() > 0;
    const int  k       = input.shape(-1);
    const int  m       = indexed ? indices.size() : input.size() / k;

    auto output_shape = input.shape();
    if (grouped) {
        output_shape = {m, output_dim};
    }
    else {
        output_shape.back() = output_dim;
    }
    return family->output_spec(core::Layout{std::move(output_shape)}, epilogue);
}

gemm::Gemm::Arguments Linear::GetArguments(Tensor&                 A,
                                           Tensor&                 U,
                                           const gemm::OutputSpec& output_spec,
                                           const Tensor&           input,
                                           const Tensor&           input_scales,
                                           const Buffer_<int>&     indices,
                                           const Buffer_<int>&     offsets,
                                           Tensor&                 output,
                                           Tensor&                 output_scales) const
{
    Tensor       in = input.view({-1, input.shape(-1)});
    MatrixLayout desc_A;
    MatrixLayout desc_U;
    std::tie(A, desc_A, U, desc_U) = GetOperandA(in, input_scales, indices, offsets);
    auto&& [B, desc_B, V, desc_V]  = GetOperandB();

    const Tensor& global_scale = this->global_scale;
    MatrixLayout  global_scale_desc{};
    if (global_scale) {
        global_scale_desc     = {global_scale.dtype(), gemm::kRowMajor, 1, 1, k_desc.ld == 0 ? 0 : 1};
        global_scale_desc.num = k_desc.num;
    }

    Tensor& D = output;
    if (!D) {
        D = Tensor{output_spec.layout, output_spec.dtype, kDEVICE};
    }
    Tensor D_gemm = D.view({-1, D.shape(-1)});

    MatrixLayout desc_D{
        D_gemm.dtype(),
        gemm::kRowMajor,
        static_cast<int>(D_gemm.shape(0)),
        output_dim,
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
                  gemm::kColMajor,
                  static_cast<int>(W.shape(1)),
                  static_cast<int>(W.shape(0)),
                  static_cast<int>(W.stride(0))};
        W_ptr  = W.raw_data();
    }

    gemm::Gemm::Arguments args{};
    args.operation         = GetOperation();
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
    args.workspace         = engine().workspace();
    args.stream            = core::Context::stream().handle();
    return args;
}

void Linear::Forward(const Tensor& input, Ref<Tensor> output)
{
    Tensor input_scales;
    Tensor output_scales;
    Forward(input, input_scales, {}, {}, output, output_scales);
}

void Linear::Forward(const Tensor& input, const Buffer_<int>& indices, const Buffer_<int>& offsets, Ref<Tensor> output)
{
    Tensor input_scales;
    Tensor output_scales;
    Forward(input, input_scales, indices, offsets, output, output_scales);
}

void Linear::Forward(const Tensor& input, const Tensor& input_scales, Ref<Tensor> output, Ref<Tensor> output_scales)
{
    Forward(input, input_scales, {}, {}, output, output_scales);
}

void Linear::Forward(const Tensor&       input,
                     const Tensor&       input_scales,
                     const Buffer_<int>& indices,
                     const Buffer_<int>& offsets,
                     Ref<Tensor>         output,
                     Ref<Tensor>         output_scales)
{
    TM_FUNCTION_SCOPE();
    gemm::OutputSpec      output_spec = GetOutputSpec(input, indices);
    Tensor                A;
    Tensor                U;
    gemm::Gemm::Arguments args =
        GetArguments(A, U, output_spec, input, input_scales, indices, offsets, output.get(), output_scales.get());

    auto& gemm = engine().gemm();
    auto  exec_plan =
        engine().dispatch_policy() & gemm::DispatchPolicy::kMeasure ? gemm.Tune(args) : gemm.GetExecPlan(args);

    TM_CHECK(exec_plan);
    exec_plan->output_spec_ = std::move(output_spec);
    const int ec            = gemm.Run(*exec_plan, args);
    if (ec) {
        TM_LOG_ERROR("{}: {}", __PRETTY_FUNCTION__, ec);
    }
}

void Linear::Forward(const gemm::ExecPlan& plan,
                     const Tensor&         input,
                     const Tensor&         input_scales,
                     const Buffer_<int>&   indices,
                     const Buffer_<int>&   offsets,
                     Ref<Tensor>           output,
                     Ref<Tensor>           output_scales)
{
    TM_FUNCTION_SCOPE();
    Tensor                A;
    Tensor                U;
    gemm::Gemm::Arguments args = GetArguments(
        A, U, plan.output_spec(), input, input_scales, indices, offsets, output.get(), output_scales.get());
    const int ec = engine().gemm().Run(plan, args);
    if (ec) {
        TM_LOG_ERROR("{}: {}", __PRETTY_FUNCTION__, ec);
    }
}

std::optional<gemm::ExecPlan>
Linear::GetExecPlan(const Tensor& input, const Buffer_<int>& indices, const Buffer_<int>& offsets)
{
    const bool indexed = static_cast<bool>(indices) && indices.size() > 0;
    const int  k       = input.shape(-1);
    const int  m       = indexed ? indices.size() : input.size() / k;

    gemm::OutputSpec output_spec = GetOutputSpec(input, indices);

    MatrixLayout desc_A{input_format.dtype, gemm::kRowMajor, m, k, k};
    MatrixLayout desc_U{};
    MatrixLayout desc_B = k_desc;
    MatrixLayout desc_V = q_desc;
    MatrixLayout desc_D{
        output_spec.dtype, gemm::kRowMajor, m, output_dim, static_cast<int>(output_spec.layout.stride(-2))};

    if (offsets) {
        desc_A.num = desc_U.num = desc_D.num = desc_B.num;
        desc_A.offsets = desc_U.offsets = desc_D.offsets = const_cast<int*>(offsets.data());
    }
    if (indexed) {
        desc_A.idxs = desc_U.idxs = const_cast<int*>(indices.data());
    }

    gemm::Gemm::Arguments args{};
    args.operation = GetOperation();
    args.Adesc     = desc_A;
    args.Udesc     = desc_U;
    args.Bdesc     = desc_B;
    args.Vdesc     = desc_V;
    args.Cdesc     = desc_D;
    args.Ddesc     = desc_D;
    args.workspace = engine().workspace();

    auto& gemm = engine().gemm();
    auto  plan = gemm.GetExecPlan(args);
    if (plan) {
        plan->output_spec_ = std::move(output_spec);
    }
    return plan;
}

std::optional<gemm::ExecPlan> Linear::Tune(const Tensor&       input,
                                           const Tensor&       input_scales,
                                           const Buffer_<int>& indices,
                                           const Buffer_<int>& offsets,
                                           Ref<Tensor>         output,
                                           Ref<Tensor>         output_scales)
{
    gemm::OutputSpec      output_spec = GetOutputSpec(input, indices);
    Tensor                A;
    Tensor                U;
    gemm::Gemm::Arguments args =
        GetArguments(A, U, output_spec, input, input_scales, indices, offsets, output.get(), output_scales.get());
    args.operation.dispatch = gemm::DispatchPolicy::kMeasure;
    auto& gemm              = engine().gemm();
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

// ----- per-rank run-mode ops -----

void Linear::set_measure(bool measure)
{
    engine().set_measure(measure);
}

int Linear::Export(std::ostream& os)
{
    return engine().Export(os);
}

int Linear::Import(std::istream& is)
{
    return engine().Import(is);
}

std::vector<int> Linear::GetTuningSeq() const
{
    return engine().GetTuningSeq();
}

TM_MODULE_REGISTER_AS("LinearWeight", Linear, core::LinearConfig);

TM_MODULE_METHODS(Linear, LINEAR_CHILDREN, LINEAR_PARAMS)

}  // namespace turbomind
