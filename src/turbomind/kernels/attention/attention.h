// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "attention_params.h"

namespace turbomind {

template<typename T>
void DispatchAttention(const AttentionParams<T>& params);

template<typename T>
void invokeApplyRotaryEmbedding(
    T* k_cache, int seq_len, int head_num, float rotary_embedding_base, int batch_size, cudaStream_t st);

}  // namespace turbomind