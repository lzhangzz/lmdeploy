/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 * Copyright (c) 2022, SK Telecom Authored by A. Dialog
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

// Modified from
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/turbomind/models/multi_gpu_gpt/ParallelGpt.cc

#include "src/turbomind/models/llama/LlamaV2.h"
#include "src/turbomind/kernels/decoding_kernels.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/models/llama/LlamaBatch.h"
#include "src/turbomind/models/llama/LlamaNcclGuard.h"
#include "src/turbomind/models/llama/LlamaWeight.h"
#include "src/turbomind/models/llama/Request.h"
#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/unified_decoder.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"
#include <functional>
#include <memory>
#include <sstream>

namespace turbomind {

template<typename T>
LlamaV2<T>::LlamaV2(size_t                       head_num,
                    size_t                       kv_head_num,
                    size_t                       size_per_head,
                    size_t                       inter_size,
                    size_t                       num_layer,
                    size_t                       vocab_size,
                    const LlamaAttentionParams&  attn_params,
                    float                        norm_eps,
                    int                          max_batch_size,
                    int                          max_context_token_num,
                    int                          session_len,
                    int                          step_length,
                    int                          start_id,
                    int                          end_id,
                    float                        cache_max_block_count,
                    int                          cache_block_seq_len,
                    int                          cache_chunk_size,
                    int                          quant_policy,
                    bool                         use_context_fmha,
                    std::shared_ptr<SharedState> shared_state,
                    LlamaWeight<T>*              weights,
                    NcclParam                    tensor_para,
                    cudaStream_t                 stream,
                    cublasMMWrapper*             cublas_wrapper,
                    IAllocator*                  allocator,
                    bool                         is_free_buffer_after_forward,
                    cudaDeviceProp*              cuda_device_prop):
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    vocab_size_(vocab_size),
    attn_params_(attn_params),
    vocab_size_padded_(vocab_size),
    rmsnorm_eps_(norm_eps),
    start_id_(start_id),
    end_id_(end_id),
    hidden_units_(head_num * size_per_head),
    local_head_num_(head_num / tensor_para.world_size_),
    local_kv_head_num_(head_num / tensor_para.world_size_),
    weights_(weights),
    tensor_para_(tensor_para),
    stream_(stream),
    cublas_wrapper_(cublas_wrapper),
    allocator_(allocator),
    is_free_buffer_after_forward_(is_free_buffer_after_forward),
    cuda_device_prop_(cuda_device_prop),
    debug_(isDebug()),
    step_length_(step_length),
    shared_state_(shared_state)

{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    TM_LOG_INFO("NCCL group_id = %d", tensor_para_.group_id_);

    vocab_size_padded_ =
        (vocab_size_padded_ + tensor_para_.world_size_ - 1) / tensor_para_.world_size_ * tensor_para_.world_size_;

    size_t elem_bits = 0;
    if (quant_policy & QuantPolicy::kCacheKVInt8) {
        elem_bits = sizeof(int8_t) * 8;
    }
    else {
        elem_bits = sizeof(T) * 8;
    }

    const size_t local_kv_head_num = kv_head_num / tensor_para.world_size_;

    auto sequence_manager = std::make_unique<SequenceManager>(num_layer,
                                                              local_kv_head_num,
                                                              size_per_head_,
                                                              cache_block_seq_len,
                                                              cache_max_block_count,
                                                              cache_chunk_size,
                                                              elem_bits,
                                                              tensor_para_.rank_,
                                                              allocator);

    const size_t max_session_len = sequence_manager->max_block_count() * cache_block_seq_len;
    if (max_session_len < session_len) {
        if (tensor_para.rank_ == 0) {
            TM_LOG_WARNING("No enough blocks for `session_len` (%d), `session_len` truncated to %d.",
                           session_len,
                           max_session_len);
        }
        session_len = max_session_len;
    }

    batch_ = std::make_unique<LlamaBatch<T>>(
        max_batch_size, max_context_token_num, session_len, std::move(sequence_manager), this);

    initialize(attn_params, kv_head_num, use_context_fmha, cache_block_seq_len, quant_policy);

    /// TODO: decouple Llama model and batch inference
    batch_->Start();
}

template<typename T>
LlamaV2<T>::~LlamaV2()
{
    delete decoder_;
    delete dynamic_decode_layer_;
    delete context_decoder_;
}

template<typename T>
void LlamaV2<T>::initialize(const LlamaAttentionParams& attn_params,
                            size_t                      kv_head_num,
                            bool                        use_context_fmha,
                            int                         cache_block_seq_len,
                            int                         quant_policy)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    context_decoder_ = new LlamaContextDecoder<T>(head_num_,
                                                  kv_head_num,
                                                  size_per_head_,
                                                  inter_size_,
                                                  num_layer_,
                                                  attn_params,
                                                  rmsnorm_eps_,
                                                  tensor_para_,
                                                  stream_,
                                                  cublas_wrapper_,
                                                  allocator_,
                                                  is_free_buffer_after_forward_,
                                                  use_context_fmha,
                                                  cache_block_seq_len,
                                                  quant_policy);

    decoder_ = new LlamaDecoder<T>(head_num_,
                                   kv_head_num,
                                   size_per_head_,
                                   inter_size_,
                                   num_layer_,
                                   attn_params,
                                   rmsnorm_eps_,
                                   tensor_para_,
                                   stream_,
                                   cublas_wrapper_,
                                   allocator_,
                                   is_free_buffer_after_forward_,
                                   cache_block_seq_len,
                                   quant_policy);

    dynamic_decode_layer_ = new DynamicDecodeLayer<float>(vocab_size_,
                                                          vocab_size_padded_,
                                                          0,  // end_id, deprecated
                                                          stream_,
                                                          cublas_wrapper_,
                                                          allocator_,
                                                          is_free_buffer_after_forward_,
                                                          cuda_device_prop_);

    unified_decoder_.reset(new UnifiedDecoder<T>(head_num_,
                                                 kv_head_num,
                                                 size_per_head_,
                                                 inter_size_,
                                                 num_layer_,
                                                 attn_params,
                                                 rmsnorm_eps_,
                                                 tensor_para_,
                                                 stream_,
                                                 cublas_wrapper_,
                                                 allocator_,
                                                 is_free_buffer_after_forward_,
                                                 use_context_fmha,
                                                 cache_block_seq_len,
                                                 quant_policy));
}

template<typename T>
void LlamaV2<T>::embeddingLookup(T* embeddings, const int* token_ids_buf, int batch_size, int step)
{
    NvtxScope scope("embeddingLookup");
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    // ! This kernel can't be used in context decoding
    invokeEmbeddingLookupPosEncodingPadCount(embeddings,
                                             weights_->pre_decoder_embedding_table,
                                             static_cast<T*>(nullptr),  // position encoding
                                             token_ids_buf,
                                             static_cast<int*>(nullptr),  // padding count, not used w/o pos-code
                                             batch_size,
                                             hidden_units_,
                                             static_cast<T>(1.),  // scale
                                             step,                // step, used int index into output_ids_buf_
                                             batch_size,          // token_num
                                             0,                   // ite
                                             stream_);
    sync_check_cuda_error();
}

template<typename T>
void LlamaV2<T>::contextDecode(T*           deocder_output,
                               uintptr_t*   k_cache_ptr,
                               uintptr_t*   v_cache_ptr,
                               void**       tmp_k_ptrs,
                               void**       tmp_v_ptrs,
                               T*           context_decoder_input_buf,
                               T*           context_decoder_output_buf,
                               const int*   input_ids,
                               const int*   input_length,
                               const int*   context_length,
                               const int*   cu_block_counts,
                               const float* rope_theta,
                               size_t       token_num,
                               size_t       max_input_len,
                               size_t       max_context_len,
                               size_t       session_len,
                               size_t       batch_size)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (tensor_para_.rank_ == 0) {
        TM_LOG_INFO("context decoding start");
    }

    invokeInputIdsEmbeddingLookupPosEncoding(context_decoder_input_buf,
                                             nullptr,  // processed somewhere else
                                             weights_->pre_decoder_embedding_table,
                                             static_cast<T*>(nullptr),
                                             pPromptTuningParam<T>{},
                                             input_ids,
                                             0,  // only used for position encoding
                                             token_num,
                                             token_num,
                                             1,
                                             hidden_units_,
                                             stream_);
    sync_check_cuda_error();

    const auto dtype = getTensorType<T>();
    const auto bsz   = batch_size;

    const int max_q_len   = max_input_len;
    const int max_kv_len  = max_context_len;
    const int max_seq_len = session_len;

    std::unordered_map<std::string, Tensor> decoder_input_tensors{
        {"decoder_input", {MEMORY_GPU, dtype, {token_num, hidden_units_}, context_decoder_input_buf}},
        {"output_norm_weight", {MEMORY_GPU, dtype, {hidden_units_}, weights_->output_norm_weight}},
        {"input_lengths", {MEMORY_GPU, TYPE_INT32, {bsz}, input_length}},
        {"context_lengths", {MEMORY_GPU, TYPE_INT32, {bsz}, context_length}},
        {"max_q_len", {MEMORY_CPU, TYPE_INT32, {1}, &max_q_len}},
        {"max_kv_len", {MEMORY_CPU, TYPE_INT32, {1}, &max_kv_len}},
        {"max_seq_len", {MEMORY_CPU, TYPE_INT32, {1}, &max_seq_len}},
        {"rope_theta", {MEMORY_GPU, TYPE_FP32, {hidden_units_}, rope_theta}},
        {"cu_block_counts", {MEMORY_GPU, TYPE_INT32, {batch_size}, cu_block_counts}}};

    std::unordered_map<std::string, Tensor> decoder_output_tensors{
        {"decoder_output", {MEMORY_GPU, dtype, {token_num, hidden_units_}, context_decoder_output_buf}},
        {"key_cache", {MEMORY_GPU, TYPE_UINT64, {bsz}, k_cache_ptr}},
        {"value_cache", {MEMORY_GPU, TYPE_UINT64, {bsz}, v_cache_ptr}},
        {"tmp_k", {MEMORY_GPU, TYPE_UINT64, {bsz}, tmp_k_ptrs}},
        {"tmp_v", {MEMORY_GPU, TYPE_UINT64, {bsz}, tmp_v_ptrs}},
        {"last_token_hidden_units", {MEMORY_GPU, dtype, {bsz, hidden_units_}, deocder_output}}};

    context_decoder_->forward(&decoder_output_tensors, &decoder_input_tensors, &weights_->decoder_layer_weights);

    if (tensor_para_.rank_ == 0) {
        TM_LOG_INFO("context decoding end");
    }
}

template<typename T>
void LlamaV2<T>::forwardUnified(T*           out,
                                T*           decoder_output,
                                T*           decoder_input,
                                void**       k_block_ptrs,
                                void**       v_block_ptrs,
                                const int*   input_ids,
                                const int*   cu_block_cnts,
                                const float* rope_theta,
                                const int*   dc_sequence_length,
                                const bool*  dc_finished,
                                const int*   pf_input_length,
                                const int*   pf_context_length,
                                T**          pf_tmp_k_ptrs,
                                T**          pf_tmp_v_ptrs,
                                size_t       token_num,
                                int          dc_batch_size,
                                int          dc_step,
                                int          dc_sum_seq_len,
                                int          dc_max_seq_len,
                                int          pf_batch_size,
                                int          pf_max_input_len,
                                int          pf_max_context_len,
                                int          pf_session_len)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    invokeInputIdsEmbeddingLookupPosEncoding(decoder_input,
                                             nullptr,  // processed somewhere else
                                             weights_->pre_decoder_embedding_table,
                                             static_cast<T*>(nullptr),
                                             pPromptTuningParam<T>{},
                                             input_ids,
                                             0,  // only used for position encoding
                                             token_num,
                                             token_num,
                                             1,
                                             hidden_units_,
                                             stream_);
    sync_check_cuda_error();

    const auto   dtype = getTensorType<T>();
    const size_t bsz   = dc_batch_size + pf_batch_size;

    TensorMap inputs{{"decoder_input", {MEMORY_GPU, dtype, {token_num, hidden_units_}, decoder_input}},
                     {"output_norm_weight", {MEMORY_GPU, dtype, {hidden_units_}, weights_->output_norm_weight}},
                     {"input_lengths", {MEMORY_GPU, TYPE_INT32, {bsz}, pf_input_length}},
                     {"context_lengths", {MEMORY_GPU, TYPE_INT32, {bsz}, pf_context_length}},
                     {"dc_batch_size", {MEMORY_CPU, TYPE_INT32, {1}, &dc_batch_size}},
                     {"dc_sum_seq_len", {MEMORY_CPU, TYPE_INT32, {1}, &dc_sum_seq_len}},
                     {"dc_max_seq_len", {MEMORY_CPU, TYPE_INT32, {1}, &dc_max_seq_len}},
                     {"sequence_lengths", {MEMORY_GPU, TYPE_INT32, {bsz}, dc_sequence_length}},
                     {"finished", {MEMORY_GPU, TYPE_BOOL, {bsz}, dc_finished}},
                     {"step", {MEMORY_CPU, TYPE_INT32, {1}, &dc_step}},
                     {"pf_batch_size", {MEMORY_CPU, TYPE_INT32, {1}, &pf_batch_size}},
                     {"pf_max_q_len", {MEMORY_CPU, TYPE_INT32, {1}, &pf_max_input_len}},
                     {"pf_max_k_len", {MEMORY_CPU, TYPE_INT32, {1}, &pf_max_context_len}},
                     {"session_len", {MEMORY_CPU, TYPE_INT32, {1}, &pf_session_len}},
                     {"rope_theta", {MEMORY_GPU, TYPE_FP32, {hidden_units_}, rope_theta}},
                     {"cu_block_counts", {MEMORY_GPU, TYPE_INT32, {bsz}, cu_block_cnts}}};

    TensorMap outputs{{"decoder_output", {MEMORY_GPU, dtype, {token_num, hidden_units_}, decoder_output}},
                      {"key_cache", {MEMORY_GPU, TYPE_UINT64, {bsz}, k_block_ptrs}},
                      {"value_cache", {MEMORY_GPU, TYPE_UINT64, {bsz}, v_block_ptrs}},
                      {"tmp_k", {MEMORY_GPU, TYPE_UINT64, {bsz}, pf_tmp_k_ptrs}},
                      {"tmp_v", {MEMORY_GPU, TYPE_UINT64, {bsz}, pf_tmp_v_ptrs}},
                      {"last_token_hidden_units", {MEMORY_GPU, dtype, {bsz, hidden_units_}, out}}};

    // context_decoder_->forward(&decoder_output_tensors, &decoder_input_tensors, &weights_->decoder_layer_weights);

    unified_decoder_->forward(&outputs, &inputs, &weights_->decoder_layer_weights);
}

template<typename T>
void LlamaV2<T>::decoderForward(T*           decoder_output,
                                uintptr_t*   k_cache_ptr,
                                uintptr_t*   v_cache_ptr,
                                T*           decoder_input,
                                const int*   sequence_length,
                                const bool*  finished,
                                const int*   cu_block_counts,
                                const float* rope_theta,
                                int          step,
                                int          ite,
                                int          sum_seq_len,
                                int          max_seq_len,
                                size_t       batch_size)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    const auto dtype = getTensorType<T>();

    // max_input_length is not used w/o linear_bias_slopes
    // sequence_lengths_ will be incremented in dynamic decode
    std::unordered_map<std::string, Tensor> decoder_input_tensors{
        {"decoder_input", {MEMORY_GPU, dtype, {batch_size, hidden_units_}, decoder_input}},
        {"sequence_lengths", {MEMORY_GPU, TYPE_INT32, {batch_size}, sequence_length}},
        {"cu_block_counts", {MEMORY_GPU, TYPE_INT32, {batch_size}, cu_block_counts}},
        {"sum_seq_len", {MEMORY_CPU, TYPE_INT32, {1}, &sum_seq_len}},
        {"max_seq_len", {MEMORY_CPU, TYPE_INT32, {1}, &max_seq_len}},
        {"finished", {MEMORY_GPU, TYPE_BOOL, {batch_size}, finished}},
        {"output_norm_weight", {MEMORY_GPU, dtype, {hidden_units_}, weights_->output_norm_weight}},
        {"rope_theta", {MEMORY_GPU, TYPE_FP32, {batch_size}, rope_theta}},
        {"step", {MEMORY_CPU, TYPE_INT32, {1}, &step}},
        {"ite", {MEMORY_CPU, TYPE_INT32, {1}, &ite}},
    };

    // LOG(ERROR) << key_cache_ << " " << value_cache_;
    std::unordered_map<std::string, Tensor> decoder_output_tensors{
        {"decoder_output", {MEMORY_GPU, dtype, {batch_size, hidden_units_}, decoder_output}},
        {"key_cache", {MEMORY_GPU, TYPE_UINT64, {batch_size}, k_cache_ptr}},
        {"value_cache", {MEMORY_GPU, TYPE_UINT64, {batch_size}, v_cache_ptr}},
    };

    decoder_->forward(&decoder_output_tensors, &decoder_input_tensors, &weights_->decoder_layer_weights);
}

template<typename T>
void LlamaV2<T>::postDecodeEmbedding(float* logits, float* local_logits, const T* decoder_output, int batch_size)
{
    NvtxScope scope("postDecodeEmbedding");
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    cudaDataType_t data_type = getCudaDataType<T>();
    float          alpha     = 1.f;
    float          beta      = 0.f;
    if (tensor_para_.world_size_ == 1) {
        cublas_wrapper_->Gemm(CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              vocab_size_,  // n
                              batch_size,
                              hidden_units_,  // k
                              &alpha,
                              weights_->post_decoder_embedding_kernel,
                              data_type,
                              hidden_units_,  // k
                              decoder_output,
                              data_type,
                              hidden_units_,  // k
                              &beta,
                              logits,
                              CUDA_R_32F,
                              vocab_size_,  // n
                              CUDA_R_32F,
                              cublasGemmAlgo_t(-1));
    }
    else {
        FT_CHECK(vocab_size_padded_ % tensor_para_.world_size_ == 0);
        const size_t local_vocab_size = vocab_size_padded_ / tensor_para_.world_size_;
        cublas_wrapper_->Gemm(CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              local_vocab_size,  // n
                              batch_size,
                              hidden_units_,  // k
                              &alpha,
                              weights_->post_decoder_embedding_kernel
                                  + tensor_para_.rank_ * local_vocab_size * hidden_units_,
                              data_type,
                              hidden_units_,  // k
                              decoder_output,
                              data_type,
                              hidden_units_,  // k
                              &beta,
                              local_logits + tensor_para_.rank_ * batch_size * local_vocab_size,
                              CUDA_R_32F,
                              local_vocab_size,  // n
                              CUDA_R_32F,
                              cublasGemmAlgo_t(-1));
        {
            NcclGuard nccl_guard(tensor_para_, stream_);
            ftNcclAllGather(local_logits,                   // send_buf
                            local_logits,                   // recv_buf
                            batch_size * local_vocab_size,  // data_size
                            tensor_para_.rank_,
                            tensor_para_,
                            stream_);
        }
        invokeTransposeAxis01(logits, local_logits, tensor_para_.world_size_, batch_size, local_vocab_size, stream_);
        sync_check_cuda_error();
    }
}

template<typename T>
void LlamaV2<T>::dynamicDecode(int*            token_ids,
                               bool*           finished,
                               int*            sequence_length,
                               bool*           should_stop,
                               curandState_t*  curand_state,
                               TensorMap*      inputs,
                               TensorMap*      outputs,
                               const float*    logits,
                               const uint32_t* seq_limit_len,
                               const int*      context_length,
                               const int*      end_ids,
                               int             step,
                               int             ite,
                               size_t          max_context_len,
                               size_t          token_ids_len,
                               size_t          batch_size)
{
    NvtxScope scope("dynamicDecode");
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    int local_batch_size = (int)batch_size;

    std::unordered_map<std::string, Tensor> dynamic_decode_input_tensors{
        {"logits", {MEMORY_GPU, TYPE_FP32, {batch_size, (size_t)1, vocab_size_padded_}, logits}},
        {"step", {MEMORY_CPU, TYPE_INT32, {1}, &step}},
        {"max_input_length", {MEMORY_CPU, TYPE_INT32, {1}, &max_context_len}},
        {"sequence_limit_length", {MEMORY_GPU, TYPE_UINT32, {batch_size}, seq_limit_len}},
        {"input_lengths", {MEMORY_GPU, TYPE_INT32, {batch_size, 1}, context_length}},
        {"ite", {MEMORY_CPU, TYPE_UINT32, {1}, &ite}},
        {"end_id", {MEMORY_GPU, TYPE_INT32, {batch_size}, end_ids}},
        {"local_batch_size", {MEMORY_CPU, TYPE_INT32, {1}, &local_batch_size}},
    };

    const std::vector<std::string> optional_inputs{"stop_words_list",
                                                   "bad_words_list",
                                                   "runtime_top_k",
                                                   "runtime_top_p",
                                                   "temperature",
                                                   "repetition_penalty",
                                                   "random_seed"};
    for (const auto& key : optional_inputs) {
        if (inputs->isExist(key)) {
            dynamic_decode_input_tensors.insert({key, inputs->at(key)});
        }
    }

    std::unordered_map<std::string, Tensor> dynamic_decode_output_tensors{
        {"output_ids", {MEMORY_GPU, TYPE_INT32, {token_ids_len, batch_size, 1U}, token_ids}},
        {"finished", {MEMORY_GPU, TYPE_BOOL, {batch_size}, finished}},
        {"sequence_length", {MEMORY_GPU, TYPE_INT32, {batch_size}, sequence_length}},
        {"should_stop", {MEMORY_CPU, TYPE_BOOL, {1}, should_stop}},
        {"curand_state", {MEMORY_GPU, TYPE_VOID, {batch_size}, curand_state}}};

    const std::vector<std::string> optional_outputs{"cum_log_probs", "output_log_probs"};
    for (const auto& key : optional_outputs) {
        if (outputs->isExist(key)) {
            dynamic_decode_output_tensors.insert({key, outputs->at(key)});
        }
    }

    dynamic_decode_layer_->forward(&dynamic_decode_output_tensors, &dynamic_decode_input_tensors);
}

static inline Tensor slice(const Tensor& tensor, int index)
{
    auto shape = tensor.shape;
    if (shape.at(0) == 1) {
        return tensor;
    }
    shape[0]          = 1;
    const auto offset = std::accumulate(shape.begin(), shape.end(), (size_t)index, std::multiplies<>{});
    return tensor.slice(shape, offset);
}

// ! implicit conversion from `unordered_map` to `TensorMap` drops 0-sized tensors
static inline TensorMap slice(const std::unordered_map<std::string, Tensor>& src, int index)
{
    TensorMap dst;
    for (const auto& kv : src) {
        dst.insert({kv.first, slice(kv.second, index)});
    }
    return dst;
}

template<typename T>
void LlamaV2<T>::forward(std::unordered_map<std::string, Tensor>*       outputs,
                         const std::unordered_map<std::string, Tensor>* inputs,
                         Control                                        control)
{
    if (debug_) {
        if (tensor_para_.rank_ == 0) {
            for (const auto& kv : *inputs) {
                TM_LOG_INFO("[forward][rank=%d] INPUT: %s", (int)tensor_para_.rank_, format(kv).c_str());
            }
            for (const auto& kv : *outputs) {
                TM_LOG_INFO("[forward][rank=%d] OUTPUT: %s", (int)tensor_para_.rank_, format(kv).c_str());
            }
        }
    }

    const int batch_size = outputs->at("output_ids").shape[0];

    const auto rank = tensor_para_.rank_;

    std::vector<std::shared_ptr<Request>> requests(batch_size);

    // rank-0 allocates all requests for the batch
    if (rank == 0) {
        for (int i = 0; i < batch_size; ++i) {
            requests[i] = std::make_shared<Request>();
            requests[i]->inputs.resize(tensor_para_.world_size_);
            requests[i]->outputs.resize(tensor_para_.world_size_);
        }
        control.comm->setSharedObject(&requests);
    }

    control.comm->barrier();

    if (rank != 0) {
        requests = *(std::vector<std::shared_ptr<Request>>*)control.comm->getSharedObject();
    }

    for (int i = 0; i < batch_size; ++i) {
        auto& r = requests[i];

        r->inputs[rank]  = slice(*inputs, i);
        r->outputs[rank] = slice(*outputs, i);

        if (rank == 0) {
            r->id         = r->inputs[rank].getVal<uint64_t>("CORRID", i);
            r->start_flag = r->inputs[rank].getVal<int>("START", 1);
            r->end_flag   = r->inputs[rank].getVal<int>("END", 1);
            r->stop_flag  = r->inputs[rank].getVal<int>("STOP", 0);
            r->stream_cb  = control.callback;
        }
    }

    control.comm->barrier();

    // rank-0 now takes the ownership of `requests`
    // rank-0 submits the tasks and wait for finish
    std::vector<int> error_codes;
    bool             has_error = 0;
    if (rank == 0) {
        TM_LOG_INFO("[forward] Enqueue requests");

        std::vector<uint64_t> ids;
        for (const auto& r : requests) {
            ids.push_back(r->id);
        }

        auto futures = shared_state_->request_queue.enqueue(std::move(requests));

        FT_CHECK_WITH_INFO(ids.size() == futures.size(), "check failed");

        TM_LOG_INFO("[forward] Wait for requests to complete ...");

        for (int i = 0; i < futures.size(); ++i) {
            auto ec = futures[i].get();
            error_codes.push_back(ec);
            if (ec) {
                has_error = true;
            }
            TM_LOG_INFO("[forward] Request complete for %ld, code %d", (long)ids[i], (int)ec);
        }
    }

    // prevents request tensors being freed before the batch completes
    control.comm->barrier();

    if (rank == 0 && has_error) {
        std::stringstream ss;
        for (int i = 0; i < error_codes.size(); ++i) {
            ss << (i ? "" : " ") << error_codes[i];
        }
        throw std::runtime_error(ss.str());
    }
}

template class LlamaV2<half>;
template class LlamaV2<float>;

}  // namespace turbomind
