// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <istream>
#include <optional>
#include <ostream>
#include <tuple>
#include <vector>

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/data_format.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/weight_plan.h"

namespace turbomind {

struct Context;
class GEMMEngine;

namespace core {

struct LinearConfig: ModuleConfig {
    LinearConfig(): ModuleConfig{"LinearWeight"} {}

#define LINEAR_FIELDS(X)                                                                                               \
    X(int, input_dim)                                                                                                  \
    X(int, output_dim)                                                                                                 \
    X(DataType, data_type)                                                                                             \
    X(DataFormat, format)                                                                                              \
    X(bool, has_bias)

    LINEAR_FIELDS(TM_MEMBER)
    TM_FOR_EACH(LinearConfig, LINEAR_FIELDS)

#undef LINEAR_FIELDS
};

}  // namespace core

using gemm::Epilogue;
using gemm::MatrixLayout;

/// Derive GEMM QuantDesc for an operand described by DataFormat.
/// For unquantized formats, returns {QuantType::kNone, 0}.
gemm::QuantDesc MakeQuantDesc(const DataFormat& fmt);

/// Executable linear module: owns the weights AND the forward. The GEMM
/// engine is captured from the engine-root Context in bind(); forwards
/// assert when called before bind.
class Linear: public core::Module {
public:
    /// Registry name frozen for the Python binding surface: builders create
    /// this module by the string "LinearWeight" via core::LinearConfig.
    const char* type() const override
    {
        return "LinearWeight";
    }

    Linear() = default;
    Linear(const core::LinearConfig& cfg);

    // ----- lifecycle -----

    void prepare() override;               // WeightPlan::pack -> Family::Pack
    void set_plan(gemm::WeightPlan plan);  // stamped by Python after Gemm::GetWeightPlan
    void copy_metadata_to(Linear& dst) const;
    bool is_graph_compatible() const;

    /// Capture the per-rank GEMM engine from the engine root.
    void bind(Context& ctx) override;

    // ----- execution (weight argument dropped: it is *this) -----

    /// Dense: y = x*W^T (+b). Output allocated when null.
    void Forward(const Tensor& input, Ref<Tensor> output);

    /// Dense with dynamic act-scale companions.
    /// ``input_scales``: when ``input`` is already FP8, used as GEMM U (skip QuantizeSymm).
    /// ``output_scales``: when ``output_dtype()`` is FP8, filled with group-128 scales (W).
    void Forward(const Tensor& input, const Tensor& input_scales, Ref<Tensor> output, Ref<Tensor> output_scales);

    /// Grouped (MoE): indices/offsets select experts packed by LinkLinearExperts.
    void Forward(const Tensor& input, const Buffer_<int>& indices, const Buffer_<int>& offsets, Ref<Tensor> output);

    /// Grouped with dynamic act-scale companions.
    void Forward(const Tensor&       input,
                 const Tensor&       input_scales,
                 const Buffer_<int>& indices,
                 const Buffer_<int>& offsets,
                 Ref<Tensor>         output,
                 Ref<Tensor>         output_scales);

    /// Plan-replay: runs a plan returned by GetExecPlan.
    void Forward(const gemm::ExecPlan& plan,
                 const Tensor&         input,
                 const Tensor&         input_scales,
                 const Buffer_<int>&   indices,
                 const Buffer_<int>&   offsets,
                 Ref<Tensor>           output,
                 Ref<Tensor>           output_scales);

    gemm::OutputSpec GetOutputSpec(const Tensor& input, const Buffer_<int>& indices = {}) const;

    std::optional<gemm::ExecPlan>
    GetExecPlan(const Tensor& input, const Buffer_<int>& indices = {}, const Buffer_<int>& offsets = {});

    std::optional<gemm::ExecPlan> Tune(const Tensor&       input,
                                       const Tensor&       input_scales,
                                       const Buffer_<int>& indices,
                                       const Buffer_<int>& offsets,
                                       Ref<Tensor>         output,
                                       Ref<Tensor>         output_scales);

    // ----- per-rank run-mode ops, delegating to the captured engine -----

    void             set_measure(bool measure);
    int              Export(std::ostream& os);
    int              Import(std::istream& is);
    std::vector<int> GetTuningSeq() const;

    // ----- weight-side metadata (frozen shape) ---

    explicit operator bool() const noexcept
    {
        return static_cast<bool>(weight);
    }

    // --- three DataFormats fully describe the GEMM ---
    const gemm::Family* family{};
    DataFormat          weight_format{};  // from cfg.format
    DataFormat          input_format{};   // derived in ctor
    DataFormat          output_format{};  // derived in ctor

    DataType input_dtype() const
    {
        return input_format.dtype;
    }
    DataType output_dtype() const
    {
        return output_format.dtype;
    }

    // --- dimensions + model activation dtype ---
    int      input_dim  = 0;
    int      output_dim = 0;
    DataType data_type{};  // model activation dtype, copied from cfg.data_type

    // --- GEMM knobs ---
    Epilogue     epilogue{};
    MatrixLayout k_desc{};
    MatrixLayout q_desc{};

#define LINEAR_CHILDREN(X)

#define LINEAR_PARAMS(X)                                                                                               \
    X(weight)                                                                                                          \
    X(bias)                                                                                                            \
    X(scales)                                                                                                          \
    X(global_scale)                                                                                                    \
    X(zeros)

    TM_MODULE_DECLARE(Linear, LINEAR_CHILDREN, LINEAR_PARAMS)

private:
    // absorbed from the retired executor
    GEMMEngine& engine() const;

    std::tuple<Tensor, MatrixLayout, Tensor, MatrixLayout> GetOperandB() const;
    std::tuple<Tensor, MatrixLayout, Tensor, MatrixLayout> GetOperandA(const Tensor&       input,
                                                                       const Tensor&       input_scales,
                                                                       Buffer_<int>        indices,
                                                                       const Buffer_<int>& offsets) const;
    gemm::Operation                                        GetOperation() const;
    gemm::Gemm::Arguments                                  GetArguments(Tensor&                 A,
                                                                        Tensor&                 U,
                                                                        const gemm::OutputSpec& output_spec,
                                                                        const Tensor&           input,
                                                                        const Tensor&           input_scales,
                                                                        const Buffer_<int>&     indices,
                                                                        const Buffer_<int>&     offsets,
                                                                        Tensor&                 output,
                                                                        Tensor&                 output_scales) const;

    /// The captured engine; never owned. Null until bind(Context&).
    GEMMEngine* engine_{};

    bool                            has_bias_ = false;
    bool                            prepared_ = false;
    std::optional<gemm::WeightPlan> plan_;
};

/// Create the GEMM output of `weight` with `rows` rows, its row stride padded to
/// a multiple of 16 / sizeof(T). Some consumers, e.g. the fused conv1d kernel,
/// read the output with 16-byte vector loads.
inline Tensor MakePaddedOutput(int rows, const Linear& weight, core::Device device)
{
    const auto dtype = weight.output_dtype();
    const int  ld    = round_up(weight.output_dim, static_cast<int>(16 / byte_size(dtype)));
    return Tensor{core::Layout{{rows, weight.output_dim}, {ld, 1}}, dtype, device};
}

}  // namespace turbomind
