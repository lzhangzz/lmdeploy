// Copyright (c) OpenMMLab. All rights reserved.

#pragma once
#include <cuda_runtime.h>

namespace turbomind {

template<typename T>
struct AttentionParams {
    // token-level buffers, [B, qH + 2kvH, D] or [B, kvH, D]
    T*  out;
    T*  q;
    T*  k;
    T*  v;
    int stride;

    // bias, [qH, D] or [kvH, D]
    T* q_bias;
    T* k_bias;
    T* v_bias;

    const int* cu_seqlens;

    // sequence-level buffers
    const int*   context_length;
    const int*   input_length;
    const bool*  finished;
    const float* rope_theta;

    // kv cache
    // int layer_offset;

    int key_offset;
    int val_offset;

    /// cache layout M,[N,H,x,D]
    /// S: [s0/x, s1/x, s2/x, ..., sn-1/x], si <- block
    /// 1. [L,sum(S),H,x,D]
    void** k_cache_block_ptrs;  // S,[L,H,s,D]
    // void** v_cache_block_ptrs;  // S,[L,H,s,D]
    int* cu_block_cnts;  // [B+1]
    int  kv_cache_block_size;

    T* kv_cache_quant_data;  // [B,H,2,S,2]

    // batch-level params
    int token_num;
    int batch_size;
    int max_input_len;
    int max_seq_len;

    // instance-level params
    int   num_heads;
    int   num_kv_heads;
    int   size_per_head;
    float inv_sqrt_dh;

    // rotary embedding
    int   rotary_embedding_dim;
    float rotary_embedding_base;
    int   max_position_embeddings;

    // log(n) attention
    bool use_logn_attn;

    int    max_split_k;
    int*   split_cnt;
    float* partial_O;
    float* partial_M;
    float* partial_L;
    int*   locks;

    int          arch;
    cudaStream_t stream;

    // debug
    float* qk;
    T*     pr;
};

}  // namespace turbomind
