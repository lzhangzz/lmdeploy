/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <unordered_map>

#include "src/turbomind/core/core.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

template<typename T>
void invokeTransposeAxis01(T* out, T* in, const int dim0, const int dim1, const int dim2, cudaStream_t stream);

template<typename T>
void invokeTransposeAxis01(
    T* out, T* in, const int* in_skipping_dim1, const int dim0, const int dim1, cudaStream_t stream);

void invokeUpdatePaddingCount(int*         total_padding_count,
                              const int*   input_lengths,
                              const int*   tiled_prompt_lengths,
                              size_t       max_input_length,
                              size_t       max_prompt_length,
                              size_t       batch_size,
                              size_t       beam_width,
                              cudaStream_t stream = 0);

inline void invokeUpdatePaddingCount(int*         total_padding_count,
                                     const int*   input_lengths,
                                     size_t       max_input_length,
                                     size_t       batch_size,
                                     size_t       beam_width,
                                     cudaStream_t stream = 0)
{
    invokeUpdatePaddingCount(
        total_padding_count, input_lengths, (const int*)nullptr, max_input_length, 0, batch_size, beam_width, stream);
}

void invokeMaskPaddingTokens(bool*        masked_tokens,
                             const int*   input_lengths,
                             const int*   tiled_prefix_prompt_lengths,
                             const size_t memory_len,
                             const size_t max_input_length,
                             const size_t initial_step,
                             size_t       batch_size,
                             size_t       beam_width,
                             cudaStream_t stream = 0);

inline void invokeMaskPaddingTokens(bool*        masked_tokens,
                                    const int*   input_lengths,
                                    const size_t memory_len,
                                    const size_t max_input_length,
                                    const size_t initial_step,
                                    size_t       batch_size,
                                    size_t       beam_width,
                                    cudaStream_t stream = 0)
{
    invokeMaskPaddingTokens(masked_tokens,
                            input_lengths,
                            (const int*)nullptr,
                            memory_len,
                            max_input_length,
                            initial_step,
                            batch_size,
                            beam_width,
                            stream);
}

void invokeEmbeddingLookup(Ref<Tensor>         out_,
                           const Buffer_<int>& token_ids,
                           const Tensor&       embedding_table,
                           cudaStream_t        st);

}  // namespace turbomind
