// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/data_format.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::core {

struct LinearConfig: ModuleConfig {
    LinearConfig(): ModuleConfig{"LinearWeight"} {}

    #define LINEAR_FIELDS(X) \
        X(int,      input_dim) \
        X(int,      output_dim) \
        X(DataType, data_type) \
        X(bool,     has_bias)

    LINEAR_FIELDS(TM_MEMBER)
    TM_FOR_EACH(LinearConfig, LINEAR_FIELDS)

    #undef LINEAR_FIELDS
};

}  // namespace turbomind::core

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

    /// Set quantization metadata (weight dtype + group size) before allocation.
    /// For trivial float weights, coerces to model compute dtype to avoid
    /// unsupported dtype combinations in GetConverters.
    void set_weight_spec(DataType weight_dtype, int group_size);

    /// Pre-process: blockwise-to-groupwise scale conversion (before fusion).
    void preprocess();
    void prepare() override;

    /// Set grouped-GEMM mode (for MoE expert weights that need row-major layout).
    void set_grouped(bool grouped) { is_grouped_ = grouped; }

    /// Copy metadata fields to another LinearWeight (for MoE block view).
    void copy_metadata_to(LinearWeight& dst) const;

    explicit operator bool() const noexcept { return static_cast<bool>(weight); }

    int  input_dim  = 0;
    int  output_dim = 0;
    int  group_size = 0;

    // --- Input (immutable after setter) ---
    DataType data_type{};       // model-scope default compute dtype, set in configure()
    DataType weight_format{};   // checkpoint weight storage format, set in do_allocate()

    // --- Derived (computed once in do_allocate via ResolveLinearPolicy) ---
    DataFormat    format{};
    LinearPolicy  policy{};

    DataType input_dtype() const  { return policy.input_dtype; }
    DataType output_dtype() const { return policy.output_dtype; }

    Epilogue    epilogue{};

    MatrixLayout k_desc{};
    MatrixLayout q_desc{};

#define LINEAR_WEIGHT_CHILDREN(X)

#define LINEAR_WEIGHT_PARAMS(X) \
    X(weight) \
    X(bias)   \
    X(scales) \
    X(zeros)

    TM_MODULE_DECLARE(LinearWeight, LINEAR_WEIGHT_CHILDREN, LINEAR_WEIGHT_PARAMS)

private:
    bool has_bias_   = false;
    bool is_grouped_ = false;
};

}  // namespace turbomind
