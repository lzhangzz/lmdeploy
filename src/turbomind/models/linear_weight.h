// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/data_format.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/core/module_config.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind {

using gemm::Epilogue;
using gemm::MatrixLayout;
using gemm::QuantDesc;

/// Compute-time dtype policy derived from DataFormat + hardware.
struct LinearPolicy {
    DataType        input_dtype{};
    DataType        output_dtype{};
    gemm::QuantDesc input_quant{};
    gemm::QuantDesc weight_quant{};
};

/// Derive compute dtypes and GEMM quant descriptors from storage format + hardware.
LinearPolicy ResolveLinearPolicy(const DataFormat& format, DataType data_type, int sm);

class LinearWeight: public core::Module {
public:
    const char* type() const override { return "LinearWeight"; }

    LinearWeight() = default;

    LinearWeight(const core::LinearConfig& cfg);

    void configure(int input_dim, int output_dim, DataType data_type, bool has_bias = false);

    /// Allocate weight tensors with explicit format. Public so composite modules
    /// (e.g. FfnWeight) can create fused weights before prepare().
    void allocate(DataType actual_weight_type, int actual_group_size);

    Tensor alloc(const std::string& param_name, const core::WeightSpec& spec) override;

    /// Pre-process: blockwise-to-groupwise scale conversion (before fusion).
    void preprocess();
    void prepare() override;

    /// Set grouped-GEMM mode (for MoE expert weights that need row-major layout).
    void set_grouped(bool grouped) { is_grouped_ = grouped; }

    explicit operator bool() const noexcept { return static_cast<bool>(weight); }

    // Public data fields consumed by execution layers (LlamaLinear, etc.)
    Tensor weight;
    Tensor bias;
    Tensor scales;
    Tensor zeros;

    int  input_dim  = 0;
    int  output_dim = 0;
    int  group_size = 0;

    // --- Input (immutable after setter) ---
    DataType data_type{};       // model-scope default compute dtype, set in configure()
    DataType weight_format{};   // checkpoint weight storage format, set in do_allocate()

    // --- Derived (computed once in do_allocate via ResolveLinearPolicy) ---
    DataFormat    format_{};
    LinearPolicy  policy_{};

    DataType input_dtype() const  { return policy_.input_dtype; }
    DataType output_dtype() const { return policy_.output_dtype; }

    Epilogue    epilogue{};

    MatrixLayout k_desc{};
    MatrixLayout q_desc{};

private:
    void do_allocate(DataType actual_weight_type, int actual_group_size);

    bool has_bias_   = false;
    bool is_grouped_ = false;
};

}  // namespace turbomind