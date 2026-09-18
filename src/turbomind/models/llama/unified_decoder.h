#pragma once

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

class ModelWeight;
class DecoderLayerWeight;

class UnifiedDecoder {
public:
    using WeightType = DecoderLayerWeight;

    UnifiedDecoder(const EngineParam& engine, const Context& ctx, int phases, const ModelWeight& model_weight);

    void Run(BatchOp op, int phase, TensorMap& env);

    void Forward(int phase, TensorMap& env, const std::vector<WeightType*>& weights);

private:
    const size_t layer_num_;
    const size_t hidden_units_;
    const bool   output_norm_zero_centered_;

    const int attn_tp_size_;
    const int attn_dp_size_;
    const int attn_dp_rank_;
    const int mlp_tp_size_;

    const int attn_tp_group_;
    const int mlp_group_;
    const int node_group_;

    // Per-layer post-FFN reduce group, precomputed in the constructor.
    std::vector<int> ffn_group_;

    comm::DeviceCommImpl* const d_comm_;

    const int tune_layer_num_;

    int& is_warm_up_;

    // Python-built layer composites; the op walk dispatches to their
    // module children (kForward keeps the orchestrated loop below).
    std::vector<DecoderLayerWeight*> layers_;

    void AllreduceResidualRMSnorm(Tensor&       hidden_states,
                                  Tensor&       residual,
                                  const Tensor& bias,
                                  const Tensor& weight,
                                  float         eps,
                                  bool          zero_centered,
                                  int           token_num,
                                  int           t0,
                                  int           t1,
                                  const int*    local_token_nums,
                                  int           local_token_nums_count);
};

}  // namespace turbomind
