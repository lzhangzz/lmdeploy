// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/linear_weight.h"

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/registry.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/gemm/cast.h"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

LinearWeight::LinearWeight(const core::LinearConfig& cfg)
{
    configure(cfg.input_dim, cfg.output_dim, cfg.data_type, cfg.has_bias);
}

static bool IsDenseFloatType(DataType t)
{
    return t == kFloat || t == kHalf || t == kBfloat16;
}

LinearPolicy ResolveLinearPolicy(const DataFormat& format, DataType data_type, int sm)
{
    LinearPolicy p;
    p.output_dtype = data_type;
    p.input_dtype  = data_type;

    if (!format.is_quantized()) {
        return p;
    }

    if (format.dtype == kFloat8_e4m3) {
        int gs = format.block_sizes[1];
        p.weight_quant = gemm::QuantDesc{gemm::QuantType::kB, gs};
        if (sm == 90) {
            p.input_dtype  = kFloat8_e4m3;
            p.input_quant  = gemm::QuantDesc{gemm::QuantType::kK, gs};
        }
        return p;
    }

    if (format.dtype == kFloat4_e2m1) {
        int gs = format.block_sizes[1];
        p.weight_quant = gemm::QuantDesc{gemm::QuantType::kK, gs};
        return p;
    }

    if (format.dtype == kUint4 || format.dtype == kUint8) {
        int gs = format.block_sizes[1];
        p.weight_quant = gemm::QuantDesc{gemm::QuantType::kK, gs};
        return p;
    }

    TM_CHECK(0) << "Unsupported weight format for policy: " << to_string(format.dtype);
    return p;
}

// ======================================================================
// configure
// ======================================================================

void LinearWeight::configure(int input_dim, int output_dim, DataType data_type, bool has_bias)
{
    this->data_type   = data_type;
    this->input_dim   = input_dim;
    this->output_dim  = output_dim;
    has_bias_         = has_bias;
}

void LinearWeight::copy_metadata_to(LinearWeight& dst) const
{
    dst.input_dim     = input_dim;
    dst.output_dim    = output_dim;
    dst.group_size    = group_size;
    dst.data_type     = data_type;
    dst.weight_format = weight_format;
    dst.format_       = format_;
    dst.policy_       = policy_;
    dst.epilogue      = epilogue;
    dst.has_bias_     = has_bias_;
    dst.is_grouped_   = is_grouped_;
    dst.k_desc        = k_desc;
    dst.q_desc        = q_desc;
}

// ======================================================================
// do_allocate
// ======================================================================

void LinearWeight::do_allocate(DataType actual_weight_type, int actual_group_size)
{
    weight_format = actual_weight_type;
    group_size    = actual_group_size;
    format_       = MakeLinearWeightFormat(data_type, actual_weight_type, actual_group_size);
    policy_       = ResolveLinearPolicy(format_, data_type, getSMVersion());

    weight = Tensor({input_dim, output_dim}, actual_weight_type, kDEVICE);

    if (has_bias_) {
        bias = Tensor{{output_dim}, data_type, kDEVICE};
    }

    scales = {};
    zeros  = {};

    if (format_.scales.present()) {
        if (actual_weight_type == kFloat8_e4m3) {
            scales = Tensor{{cdiv(input_dim, actual_group_size), cdiv(output_dim, actual_group_size)},
                            format_.scales.dtype, kDEVICE};
        }
        else if (actual_weight_type == kFloat4_e2m1) {
            scales = Tensor{{cdiv(input_dim, actual_group_size), output_dim},
                            format_.scales.dtype, kDEVICE};
        }
        else {
            TM_CHECK(input_dim % actual_group_size == 0) << input_dim << " " << actual_group_size;
            scales = Tensor{{input_dim / actual_group_size, output_dim},
                            format_.scales.dtype, kDEVICE};
        }
    }

    if (format_.zeros.present()) {
        TM_CHECK(input_dim % actual_group_size == 0) << input_dim << " " << actual_group_size;
        zeros = Tensor{{input_dim / actual_group_size, output_dim},
                        format_.zeros.dtype, kDEVICE};
    }

    k_desc = {};
    q_desc = {};

    k_desc.type  = weight.dtype();
    k_desc.order = gemm::kRowMajor;
    k_desc.rows  = input_dim;
    k_desc.cols  = output_dim;
    k_desc.ld    = output_dim;
}

// ======================================================================
// allocate (public, for composite modules that create fused weights)
// ======================================================================

void LinearWeight::allocate(DataType actual_weight_type, int actual_group_size)
{
    do_allocate(actual_weight_type, actual_group_size);
}

// ======================================================================
// alloc
// ======================================================================

Tensor LinearWeight::alloc(const std::string& param_name, const core::WeightSpec& spec)
{
    // Trigger full allocation on first call (when weight is still empty).
    if (!weight) {
        if (param_name == "weight" || param_name == "qweight") {
            // For dense floating-point weights, use the model's compute dtype
            // (data_type) to avoid unsupported dtype combinations in
            // GetConverters.  Quantized types (uint4, fp8, etc.) pass through
            // unchanged.
            DataType alloc_dtype = spec.dtype;
            if (alloc_dtype != data_type && IsDenseFloatType(alloc_dtype) && IsDenseFloatType(data_type)) {
                alloc_dtype = data_type;
            }
            do_allocate(alloc_dtype, spec.group_size);
        }
        else {
            // Cannot allocate scales/zeros before weight — caller error.
            return {};
        }
    }

    if (param_name == "weight" || param_name == "qweight") {
        return weight;
    }
    if (param_name == "bias") {
        return bias;
    }
    if (param_name == "scales") {
        return scales;
    }
    if (param_name == "zeros") {
        return zeros;
    }

    return Module::alloc(param_name, spec);
}

// ======================================================================
// preprocess — now a no-op (blockscale→groupscale handled in Python)
// ======================================================================

void LinearWeight::preprocess()
{
    // No-op: blockscale-to-groupscale conversion is done on the Python side.
}

// ======================================================================
// prepare (weight format conversion)
// ======================================================================

void LinearWeight::prepare()
{
    if (!weight) {
        return;
    }

    auto stream = core::Context::stream().handle();

    if (weight_format == kFloat8_e4m3 && input_dtype() == kFloat8_e4m3) {
        // FP8 native path: transpose weight and scales for native kernels.
        auto process = [&](Tensor& x, MatrixLayout& d, auto dtype) {
            using T = decltype(dtype);
            Tensor trans{{x.shape(1), x.shape(0)}, x.dtype(), kDEVICE};
            invokeTransposeAxis01((T*)trans.raw_data(), (T*)x.raw_data(), x.shape(0), x.shape(1), 1, stream);
            x = std::move(trans);
            d = MatrixLayout{x.dtype(), gemm::kColMajor, (int)x.shape(1), (int)x.shape(0), (int)x.stride(0)};
        };

        TM_CHECK_EQ(weight.dtype(), kFloat8_e4m3);
        process(weight, k_desc, uint8_t{});

        TM_CHECK_EQ(scales.dtype(), kFloat);
        process(scales, q_desc, float{});
    }
    else if (weight_format == kFloat8_e4m3) {
        // FP8 non-native path (non-SM90)
    }
    else {
        // General quantization format conversion path.
        using namespace gemm;

        auto [conv_w, conv_s] =
            GetConverters(data_type, weight_format, input_dtype(), is_grouped_, getSMVersion());

        if (conv_w) {
            const auto order_w = conv_w->order;
            const bool is_A    = get_operand_tag(conv_w->pack) == OPERAND_A;
            const bool is_B    = !is_A;

            const int bits = byte_size(weight_format, 8);

            Tensor_<uint16_t> tmp{{input_dim, output_dim}, kDEVICE};

            if (bits == 4) {
                extend_to_u16(tmp.data(), (const uint4_t*)weight.raw_data(), tmp.size(), stream);
                sync_check_cuda_error();
            }
            else if (bits == 8) {
                extend_to_u16(tmp.data(), (const uint8_t*)weight.raw_data(), tmp.size(), stream);
                sync_check_cuda_error();
            }
            else if (bits == 16) {
                check_cuda_error(
                    cudaMemcpyAsync(tmp.raw_data(), weight.raw_data(), weight.byte_size(), cudaMemcpyDefault, stream));
            }

            if (order_w == kRowMajor) {
                Tensor_<uint16_t> trans{{output_dim, input_dim}, kDEVICE};
                invokeTransposeAxis01(trans.data(), tmp.data(), input_dim, output_dim, 1, stream);
                tmp = trans;
            }

            MatrixLayout w_desc{
                data_type,
                order_w,
                (int)output_dim,
                (int)input_dim,
                order_w == kRowMajor ? (int)input_dim : (int)output_dim,
            };

            if (is_B) {
                std::swap(w_desc.rows, w_desc.cols);
                w_desc.order = ~w_desc.order;
            }

            MatrixLayout kd = w_desc;
            kd.type = weight_format;
            if (bits == 4) {
                kd.type = data_type_v<uint4_t>;
            }
            else if (bits == 8) {
                kd.type = data_type_v<uint8_t>;
            }
            kd.pack = conv_w->pack;

            check_cuda_error(cudaMemsetAsync(weight.raw_data(), 0, weight.byte_size(), stream));
            TM_CHECK(conv_w->Convert(tmp.data(), w_desc, weight.raw_data(), kd, stream) == 0);
            sync_check_cuda_error();

            kd.type = weight_format;
            if (is_A) {
                kd = transpose(kd);
            }
            k_desc = kd;
        }

        if (conv_s) {
            const auto order_s = conv_s->order;
            const auto pack_s  = conv_s->pack;
            const bool is_A    = get_operand_tag(conv_s->pack) == OPERAND_U;

            Tensor   tmp_q;
            DataType scale_type;

            if (zeros) {
                tmp_q = {{scales.size(), 2}, kHalf, kDEVICE};
                fuse_scales_and_zeros(
                    tmp_q.data<half>(), scales.data<half>(), zeros.data<half>(), scales.size(), stream);
                scale_type = kUint32;
                zeros    = {};
                scales   = empty_like(tmp_q);
            }
            else if (weight_format == kFloat8_e4m3) {
                tmp_q = empty_like(scales);
                Copy(scales, tmp_q);
                scale_type = kUint16;
            }
            else {
                tmp_q = empty_like(scales);
                Copy(scales, tmp_q);
                scale_type = kUint8;
            }

            if (data_type == kHalf && weight_format == kFloat4_e2m1) {
                AdjustUe8m0ScaleForHalf(tmp_q.data<uint8_t>(), tmp_q.size(), stream);
                sync_check_cuda_error();
            }

            MatrixLayout s_desc{
                scale_type,
                order_s,
                (int)output_dim,
                (int)input_dim / group_size,
                (int)output_dim,
            };

            if (!is_A) {
                std::swap(s_desc.rows, s_desc.cols);
                s_desc.order = ~s_desc.order;
            }

            MatrixLayout qd = s_desc;
            qd.pack         = pack_s;

            TM_CHECK(conv_s->Convert(tmp_q.raw_data(), s_desc, scales.raw_data(), qd, stream) == 0);
            sync_check_cuda_error();

            if (is_A) {
                qd = transpose(qd);
            }
            q_desc = qd;
        }
    }
}

namespace {
struct LinearWeightRegistrar {
    LinearWeightRegistrar() {
        core::ModuleRegistry::instance().register_type(
            "LinearWeight",
            [](const core::ModuleConfig& base_cfg) -> std::unique_ptr<core::Module> {
                return std::make_unique<LinearWeight>(
                    static_cast<const core::LinearConfig&>(base_cfg));
            });
    }
};
static LinearWeightRegistrar _linear_weight_reg;
}  // anonymous namespace

TM_MODULE_METHODS(LinearWeight, LINEAR_WEIGHT_CHILDREN, LINEAR_WEIGHT_PARAMS)

}  // namespace turbomind
