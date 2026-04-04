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

static bool IsDenseFloatType(DataType t)
{
    return t == kFloat || t == kHalf || t == kBfloat16;
}

LinearDtypes ResolveDtypes(DataType data_type, DataType weight_format, int group_size, int sm)
{
    LinearDtypes r;
    r.output_dtype = data_type;
    r.input_dtype  = data_type;
    r.scale_dtype  = data_type;

    const bool is_qweight = weight_format == kUint4 || weight_format == kUint8;

    if (IsDenseFloatType(weight_format)) {
        // Dense FP16/BF16/FP32 — no quantization descriptors
        return r;
    }

    if (weight_format == kFloat8_e4m3) {
        TM_CHECK_EQ(group_size, 128)
            << "FP8 weight format requires group_size=128, got " << group_size;
        r.weight_quant = QuantDesc{gemm::QuantType::kB, group_size};
        if (sm == 90) {
            r.input_dtype = kFloat8_e4m3;
            r.input_quant = QuantDesc{gemm::QuantType::kK, group_size};
            r.scale_dtype = kFloat;
        }
        return r;
    }

    if (weight_format == kFloat4_e2m1) {
        r.scale_dtype  = kUint8;
        r.weight_quant = QuantDesc{gemm::QuantType::kK, group_size};
        return r;
    }

    if (is_qweight) {
        TM_CHECK(group_size > 0 && group_size <= 256)
            << "Invalid group_size for quantized weight: " << group_size;
        r.weight_quant = QuantDesc{gemm::QuantType::kK, group_size};
        return r;
    }

    TM_CHECK(0) << "Unsupported weight format: " << to_string(weight_format);
    return r;
}

// ======================================================================
// configure
// ======================================================================

void LinearWeight::configure(int input_dim, int output_dim, DataType data_type, bool has_bias)
{
    this->data_type   = data_type;
    this->input_type  = data_type;
    this->weight_type = data_type;
    this->input_dim   = input_dim;
    this->output_dim  = output_dim;
    this->group_size  = 0;
    has_bias_         = has_bias;
}

// ======================================================================
// do_allocate
// ======================================================================

void LinearWeight::do_allocate(DataType actual_weight_type, int actual_group_size)
{
    weight_type  = actual_weight_type;
    group_size   = actual_group_size;
    input_type   = data_type;
    weight_quant = {};
    input_quant  = {};

    const bool is_qweight = actual_weight_type == kUint4 || actual_weight_type == kUint8;

    weight = Tensor({input_dim, output_dim}, actual_weight_type, kDEVICE);
    add_param("weight", weight);

    if (has_bias_) {
        bias = Tensor{{output_dim}, data_type, kDEVICE};
        add_param("bias", bias);
    }

    scales = {};
    zeros  = {};

    if (actual_weight_type == kFloat8_e4m3) {
        TM_CHECK_EQ(actual_group_size, 128);
        scales       = Tensor{{cdiv(input_dim, actual_group_size), cdiv(output_dim, actual_group_size)}, kFloat, kDEVICE};
        weight_quant = QuantDesc{gemm::QuantType::kB, actual_group_size};
        if (getSMVersion() == 90) {
            input_type  = kFloat8_e4m3;
            input_quant = QuantDesc{gemm::QuantType::kK, actual_group_size};
        }
        add_param("scales", scales);
    }
    else if (actual_weight_type == kFloat4_e2m1) {
        scales       = Tensor{{cdiv(input_dim, actual_group_size), output_dim}, kUint8, kDEVICE};
        weight_quant = QuantDesc{gemm::QuantType::kK, actual_group_size};
        add_param("scales", scales);
    }
    else if (is_qweight) {
        TM_CHECK(input_dim % actual_group_size == 0) << input_dim << " " << actual_group_size;
        scales       = Tensor{{input_dim / actual_group_size, output_dim}, data_type, kDEVICE};
        zeros        = Tensor{{input_dim / actual_group_size, output_dim}, data_type, kDEVICE};
        weight_quant = QuantDesc{gemm::QuantType::kK, actual_group_size};
        add_param("scales", scales);
        add_param("zeros", zeros);
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

    if (weight_type == kFloat8_e4m3 && input_type == kFloat8_e4m3) {
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
    else if (weight_type == kFloat8_e4m3) {
        // FP8 non-native path (non-SM90)
    }
    else {
        // General quantization format conversion path.
        using namespace gemm;

        auto [conv_w, conv_s] =
            GetConverters(data_type, weight_type, input_type, is_grouped_, getSMVersion());

        if (conv_w) {
            const auto order_w = conv_w->order;
            const bool is_A    = get_operand_tag(conv_w->pack) == OPERAND_A;
            const bool is_B    = !is_A;

            const int bits = byte_size(weight_type, 8);

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
                    cudaMemcpyAsync(tmp.raw_data(), weight.raw_data(), tmp.byte_size(), cudaMemcpyDefault, stream));
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
            kd.type = weight_type;
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

            kd.type = weight_type;
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
                zeros     = {};
                scales    = empty_like(tmp_q);
            }
            else if (weight_type == kFloat8_e4m3) {
                tmp_q = empty_like(scales);
                Copy(scales, tmp_q);
                scale_type = kUint16;
            }
            else {
                tmp_q = empty_like(scales);
                Copy(scales, tmp_q);
                scale_type = kUint8;
            }

            if (data_type == kHalf && weight_type == kFloat4_e2m1) {
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
            [](const core::ModuleConfig& cfg) -> std::unique_ptr<core::Module> {
                auto m = std::make_unique<LinearWeight>();
                m->configure(
                    std::get<int64_t>(cfg.at("input_dim")),
                    std::get<int64_t>(cfg.at("output_dim")),
                    static_cast<DataType>(std::get<int64_t>(cfg.at("data_type"))),
                    cfg.count("has_bias") && std::get<int64_t>(cfg.at("has_bias")));
                return m;
            });
    }
};
static LinearWeightRegistrar _linear_weight_reg;
}  // anonymous namespace

}  // namespace turbomind
