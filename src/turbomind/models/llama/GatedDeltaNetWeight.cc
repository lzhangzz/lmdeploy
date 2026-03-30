#include "src/turbomind/models/llama/GatedDeltaNetWeight.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

GatedDeltaNetWeight::GatedDeltaNetWeight(int      hidden_dim,
                                         int      num_k_heads,
                                         int      num_v_heads,
                                         int      key_head_dim,
                                         int      value_head_dim,
                                         int      d_conv,
                                         bool     bias,
                                         int      tp_size,
                                         int      tp_rank,
                                         DataType data_type):
    tp_rank_(tp_rank), tp_size_(tp_size)
{
    const int key_dim    = num_k_heads * key_head_dim / tp_size;
    const int value_dim  = num_v_heads * value_head_dim / tp_size;
    const int v_heads_tp = num_v_heads / tp_size;
    const int conv_dim   = key_dim * 2 + value_dim;

    const int out_all = conv_dim + value_dim + 2 * v_heads_tp;
    in_proj_all.emplace(hidden_dim, out_all, data_type, bias);
    out_proj.emplace(value_dim, hidden_dim, data_type, bias);

    register_module("in_proj_all", in_proj_all, tp_rank_);
    register_module("out_proj", out_proj, tp_rank_);

    // conv1d: depthwise weights, shape (conv_dim, d_conv)
    conv1d = Tensor{{conv_dim, d_conv}, data_type, kDEVICE};
    register_parameter("conv1d." + std::to_string(tp_rank_) + ".weight", conv1d);

    // A_log: log-space decay per head, shape (num_v_heads/tp,)
    A_log = Tensor{{v_heads_tp}, data_type, kDEVICE};
    register_parameter("A_log." + std::to_string(tp_rank_) + ".weight", A_log);

    // dt_bias: per head, shape (num_v_heads/tp,)
    dt_bias = Tensor{{v_heads_tp}, data_type, kDEVICE};
    register_parameter("dt_bias." + std::to_string(tp_rank_) + ".weight", dt_bias);

    // norm: RMSNormGated weight, shape (value_head_dim,)
    norm = Tensor{{value_head_dim}, data_type, kDEVICE};
    register_parameter("norm.weight", norm);
}

void GatedDeltaNetWeight::prepare()
{
    auto stream = core::Context::stream().handle();

    in_proj_all.preprocess();
    in_proj_all.prepare();
    out_proj.preprocess();
    out_proj.prepare();

    // Transpose conv1d from checkpoint layout [conv_dim, d_conv] to kernel layout [d_conv, conv_dim]
    {
        const int rows = conv1d.shape(0);  // conv_dim
        const int cols = conv1d.shape(1);  // d_conv

        Tensor conv1d_t{{cols, rows}, conv1d.dtype(), kDEVICE};
        invokeTransposeAxis01((uint16_t*)conv1d_t.raw_data(), (uint16_t*)conv1d.raw_data(), rows, cols, 1, stream);
        sync_check_cuda_error();
        conv1d = std::move(conv1d_t);
    }
}

}  // namespace turbomind
