/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

namespace turbomind {

template<typename T>
struct MaskedSoftmaxParam {
    // Common parameters.
    T*           attention_score = nullptr;  // (batch_size, head_num, q_length, k_length)
    const float* qk              = nullptr;  // (batch_size, head_num, q_length, k_length)
    const T*     attention_mask  = nullptr;  // (batch_size, q_length, k_length)
    int          batch_size      = 0;
    int          q_length        = 0;
    int          k_length        = 0;
    int          num_heads       = 0;
    const T*     sinks           = nullptr;
};

template<typename T>
void invokeMaskedSoftmax(MaskedSoftmaxParam<T>& param, cudaStream_t stream);

template<typename T>
void invokeTransposeAttentionOutRemovePadding(T*           src,
                                              T*           dst,
                                              const int    valid_word_num,
                                              const int    batch_size,
                                              const int    seq_len,
                                              const int    head_num,
                                              const int    size_per_head,
                                              const int*   mask_offset,
                                              const float* scale,
                                              const int    int8_mode,
                                              cudaStream_t stream);

template<typename T>
void invokeAddRelativeAttentionBias(T*           qk_buf,
                                    const T*     relative_attention_bias,
                                    const int    batch_size,
                                    const int    head_num,
                                    const int    seq_len,
                                    cudaStream_t stream);
}  // namespace turbomind
