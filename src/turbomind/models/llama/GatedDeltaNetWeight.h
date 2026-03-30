#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"

namespace turbomind {

struct GatedDeltaNetWeight: public core::Module {

    GatedDeltaNetWeight() = default;

    GatedDeltaNetWeight(int      hidden_dim,
                        int      num_k_heads,
                        int      num_v_heads,
                        int      key_head_dim,
                        int      value_head_dim,
                        int      d_conv,
                        bool     bias,
                        int      tp_size,
                        int      tp_rank,
                        DataType data_type);

    void prepare();

    // Fused projection: hidden -> (conv_dim + value_dim + 2*v_heads_tp).
    // Built by Python from in_proj_qkv, in_proj_z, in_proj_b, in_proj_a
    // with TP-correct interleaving.  Reduces HBM reads from 4× to 1×.
    LlamaDenseWeight in_proj_all;

    LlamaDenseWeight out_proj;  // value_dim -> hidden

    // Non-dense parameters
    Tensor conv1d;   // depthwise conv weights: (d_conv, conv_dim)
    Tensor A_log;    // log-space decay: (num_v_heads,)
    Tensor dt_bias;  // dt bias: (num_v_heads,)
    Tensor norm;     // RMSNormGated weight: (value_head_dim,)

    int tp_rank_;
    int tp_size_;
};

}  // namespace turbomind
