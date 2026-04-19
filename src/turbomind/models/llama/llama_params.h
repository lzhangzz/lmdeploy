// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cstddef>
#include <vector>

namespace turbomind {

struct EngineParam {
    // batch params
    int max_batch_size;
    int session_len;
    int step_length;

    int   quant_policy   = 0;
    int   tune_layer_num = 1;

    // cache params
    float cache_max_block_count;
    int   cache_chunk_size;
    int   cache_block_seq_len;
    bool  enable_prefix_caching;
    bool  enable_metrics;

    // chunking params
    int max_forward_token_num;
    int max_context_token_num;
    int num_tokens_per_iter;
    int max_prefill_iters;

    // parallel params
    int outer_dp_size;
    int outer_dp_rank;
    int attn_dp_size;
    int attn_dp_rank;
    int attn_tp_size;
    int attn_tp_rank;
    int attn_cp_size;
    int attn_cp_rank;
    int mlp_tp_size;
    int mlp_tp_rank;

    // multi-node
    int nnodes;
    int node_rank;

    std::vector<int> devices;
};

}  // namespace turbomind
