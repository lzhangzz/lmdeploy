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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.cc

#include <algorithm>
#include <memory>

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/core.h"
#include "src/turbomind/core/exchange.h"
#include "src/turbomind/core/state.h"
#include "src/turbomind/core/state2.h"
#include "src/turbomind/engine/request.h"
#include "src/turbomind/layers/generation/generation.h"

#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/LlamaV2.h"
#include "src/turbomind/models/llama/LlamaWeight.h"
#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/unified_decoder.h"

#include "src/turbomind/kernels/gpt_kernels.h"

#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

struct InputProcessorData {
    Buffer_<int> input_ids;
    Buffer_<int> input_ids_offsets;

    Tensor       input_embeds;
    Buffer_<int> input_embeds_offsets;
};

class InputProcessor {
public:
    InputProcessor(const EngineParam& engine, int phases):
        max_batch_size_{engine.max_batch_size}, max_forward_token_num_{engine.max_forward_token_num}
    {
        input_ids_buf_         = {max_forward_token_num_, kCPUpinned};
        input_ids_offsets_buf_ = {max_batch_size_ + 1, kCPUpinned};

        data_.reserve(phases);
        for (int i = 0; i < phases; ++i) {
            auto& d             = data_.emplace_back();
            d.input_ids         = empty_like(input_ids_buf_, kDEVICE);
            d.input_ids_offsets = empty_like(input_ids_offsets_buf_, kCPUpinned);
        }
    }

    void Exchange(ExchOp op, int phase, TensorMap& env)
    {
        if (op != ExchOp::kPush) {
            return;
        }

        auto& d = data_.at(phase);

        const Buffer_<RequestCache*> rc   = env.at("requests").buffer();
        const Buffer_<int>           perm = env.at("permutation").buffer();

        const int bs0 = *env.at("bs0").data<int>();
        const int bsz = rc.size();

        d.input_ids_offsets[0] = 0;
        for (int i = 0; i < rc.size(); ++i) {
            d.input_ids_offsets[i + 1] = d.input_ids_offsets[i];
            if (const auto& c = *rc[i]; TM_UNLIKELY(perm[i] >= 0)) {
                const auto src = c.token_ids + c.context_len - c.input_len;
                std::copy_n(src, c.input_len, input_ids_buf_.data() + d.input_ids_offsets[i]);
                d.input_ids_offsets[i + 1] += c.input_len;
            }
        }

        if (auto size = d.input_ids_offsets[bsz]) {
            Copy_(input_ids_buf_, size, d.input_ids);
        }
    }

    void Forward(int phase, TensorMap& args)
    {
        auto& d = data_.at(phase);

        const Buffer_<int> perm = args.at("permutation").buffer();

        const auto bsz = perm.size();

        // last output token + draft tokens
        const Buffer_<int> autoreg_ids         = args.at("autoreg_ids").buffer();
        const Buffer_<int> autoreg_ids_offsets = args.at("autoreg_ids_offsets").buffer();

        Buffer_<int> input_ids{max_forward_token_num_, kDEVICE};
        Select(autoreg_ids,  // auto-regressive token ids from last iteration T0
               autoreg_ids_offsets,
               d.input_ids,  // input token ids from swap-ins T1
               d.input_ids_offsets,
               perm,
               input_ids,
               input_ids_offsets_buf_);

        const int token_num = input_ids_offsets_buf_[bsz];

        Buffer_<int> input_ids_offsets{bsz + 1, kDEVICE};
        Copy_(input_ids_offsets_buf_, bsz + 1, input_ids_offsets);

        args.emplace("input_ids", input_ids.slice(0, token_num));
        args.emplace("input_ids_offsets", input_ids_offsets);
    }

private:
    const int max_batch_size_;
    const int max_forward_token_num_;

    std::vector<InputProcessorData> data_;

    Buffer_<int> input_ids_buf_;
    Buffer_<int> input_ids_offsets_buf_;
};

/// TODO: Padded vocab size should also be divisible by 8
inline int pad_vocab_size(int vocab_size, int tp)
{
    return (vocab_size + tp - 1) / tp * tp;
}

LlamaV2::LlamaV2(DataType                     dtype,
                 const ModelParam&            model,
                 const EngineParam&           engine,
                 const AttentionParam&        attn,
                 const MoeParam&              moe,
                 const LoraParam&             lora,
                 const Context&               ctx,
                 int                          max_batch_size,
                 std::shared_ptr<LlamaWeight> weights,
                 int                          phases):
    dtype_{dtype},
    param_(model),
    attn_param_(attn),
    lora_param_(lora),
    comm_(&ctx.comm),
    tp_size_(engine.attn_tp_size),
    tp_rank_(engine.attn_tp_rank),
    head_num_(model.head_num),
    size_per_head_(model.head_dim),
    hidden_units_(model.hidden_units),
    layer_num_(model.layer_num),
    vocab_size_(model.vocab_size),
    vocab_size_padded_(pad_vocab_size(model.vocab_size, tp_size_)),
    rmsnorm_eps_(model.norm_eps),
    local_head_num_(model.head_num / engine.attn_tp_size),
    local_kv_head_num_(model.kv_head_num / engine.attn_tp_size),
    weights_(std::move(weights)),
    stream_(ctx.stream),
    linear_(*ctx.linear),
    debug_(isDebug())
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (comm_->d_comm && comm_->d_comm->Query(comm::kHasAllGather2D)) {
        use_allgather_2d_ = true;
    }

    input_processor_ = std::make_shared<InputProcessor>(engine, phases);

    unified_decoder_ = std::make_unique<UnifiedDecoder>(model, engine, attn, moe, lora, ctx, phases);

    generation_ = std::make_unique<Generation>(
        kFloat32, max_batch_size, engine.session_len, model.tokenizer_size, vocab_size_padded_, phases);
}

void LlamaV2::Exchange(ExchOp op, int phase, TensorMap& env)
{
    input_processor_->Exchange(op, phase, env);
    unified_decoder_->Exchange(op, phase, env);
    generation_->Exchange(op, phase, env);
}

Tensor LlamaV2::LookupEmbedding(const Buffer_<int>& input_ids, Tensor symm_buf)
{
    const auto& embedding_table = weights_->pre_decoder_embedding.weight;
    TM_CHECK_EQ(embedding_table.shape(1) * tp_size_, hidden_units_);

    const int token_num = input_ids.size();

    Tensor input_embeds{{token_num, (int)hidden_units_}, dtype_, kDEVICE};

    if (tp_size_ == 1) {
        invokeEmbeddingLookup(input_embeds, input_ids, embedding_table, stream_);
        sync_check_cuda_error();
    }
    else if (use_allgather_2d_) {
        const auto local_hidden_units = embedding_table.shape(1);
        Tensor     temp{symm_buf.buffer(), {token_num, tp_size_, local_hidden_units}};

        auto local = temp.slice({0, tp_rank_, 0}, {-1, 1, -1}).squeeze(1);

        invokeEmbeddingLookup(local, input_ids, embedding_table, stream_);
        sync_check_cuda_error();

        comm_->d_comm->AllGather2D(local.raw_data(),
                                   temp.raw_data(),
                                   hidden_units_,
                                   local_hidden_units,
                                   local_hidden_units,
                                   token_num,
                                   local.dtype(),
                                   {true, true},
                                   comm_->d_tp_group,
                                   stream_);
        sync_check_cuda_error();

        Copy(temp.buffer(), input_embeds.buffer());
    }
    else {
        const auto local_hidden_units = embedding_table.shape(1);
        Tensor     temp{symm_buf.buffer(), {tp_size_, token_num, local_hidden_units}};

        auto local = temp.slice(tp_rank_).squeeze(0);

        invokeEmbeddingLookup(local, input_ids, embedding_table, stream_);
        sync_check_cuda_error();

        comm_->d_comm->AllGather(local.raw_data(), temp.raw_data(), local.size(), dtype_, comm_->d_tp_group, stream_);
        sync_check_cuda_error();

        invokeInPlaceTranspose102((uint16_t*)input_embeds.raw_data(),
                                  (uint16_t*)temp.raw_data(),
                                  tp_size_,
                                  token_num,
                                  local_hidden_units,
                                  false,
                                  stream_);
        sync_check_cuda_error();
    }

    return input_embeds;
}

void LlamaV2::updateEmbedding(
    char* decoder_input, const int bsz, const int* h_input_length, const Sequence** sequences, int token_num)
{
    if (isTuning())
        return;

    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    const size_t elem_size = byte_size(dtype_, 1);

    for (int i = 0; i < bsz; i++) {
        const auto& seq        = *sequences[i];
        const auto& embeddings = seq.input_embeddings;
        const auto& ranges     = seq.input_embedding_ranges;
        for (int j = embeddings.size() - 1; j >= 0; j--) {
            int begin = ranges[j].first;
            int end   = ranges[j].second;
            if (seq.cache_len + h_input_length[i] - 1 < begin) {
                continue;
            }
            if (end <= seq.cache_len) {
                break;
            }
            int off_dst = std::max(0, begin - seq.cache_len);
            int off_src = std::max(0, seq.cache_len - begin);
            // calculate intersection of [begin, end) and [seq.cache_len, seq.cache_len + h_input_length[i])
            begin            = std::max(begin, seq.cache_len);
            end              = std::min(end, seq.cache_len + h_input_length[i]);
            size_t byte_size = elem_size * (end - begin) * hidden_units_;
            char*  dst_ptr   = decoder_input + elem_size * off_dst * hidden_units_;
            auto   src_ptr   = embeddings[j].data() + elem_size * off_src * hidden_units_;
            check_cuda_error(cudaMemcpyAsync(dst_ptr, src_ptr, byte_size, cudaMemcpyDefault, stream_));
        }
        decoder_input += elem_size * h_input_length[i] * hidden_units_;
    }
}

void LlamaV2::Forward(Buffer_<int>     input_ids,
                      Tensor           hidden_states_out,
                      Tensor           decoder_out,
                      Buffer           kv_block_ptrs,
                      Buffer           cu_block_nums,
                      Buffer_<int>     h_input_length,
                      Buffer_<int>     h_context_length,
                      Buffer           rope_base,
                      MropeRope*       mrope,
                      Buffer           finished,
                      Buffer           local_token_nums,
                      Buffer           lora_mask,
                      int              decode_num,
                      int              prefil_num,
                      const Sequence** sequences)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    Tensor input_embeds;

    const int token_num = input_ids.size();

    if (token_num) {
        const auto& embedding_table = weights_->pre_decoder_embedding.weight;
        TM_CHECK_EQ(embedding_table.shape(1) * tp_size_, hidden_units_);

        input_embeds = Tensor{{token_num, (int)hidden_units_}, dtype_, kDEVICE};

        if (tp_size_ == 1) {
            invokeEmbeddingLookup(input_embeds, input_ids, embedding_table, stream_);
            sync_check_cuda_error();
        }
        else if (use_allgather_2d_) {
            const auto local_hidden_units = embedding_table.shape(1);
            Tensor     temp{hidden_states_out.buffer(), {token_num, tp_size_, local_hidden_units}};

            auto local = temp.slice({0, tp_rank_, 0}, {-1, 1, -1}).squeeze(1);

            invokeEmbeddingLookup(local, input_ids, embedding_table, stream_);
            sync_check_cuda_error();

            comm_->d_comm->AllGather2D(local.raw_data(),
                                       temp.raw_data(),
                                       hidden_units_,
                                       local_hidden_units,
                                       local_hidden_units,
                                       token_num,
                                       local.dtype(),
                                       {true, true},
                                       comm_->d_tp_group,
                                       stream_);
            sync_check_cuda_error();

            Copy(temp.buffer(), input_embeds.buffer());
        }
        else {
            const auto local_hidden_units = embedding_table.shape(1);
            Tensor     temp{hidden_states_out.buffer(), {tp_size_, token_num, local_hidden_units}};

            auto local = temp.slice(tp_rank_).squeeze(0);

            invokeEmbeddingLookup(local, input_ids, embedding_table, stream_);
            sync_check_cuda_error();

            comm_->d_comm->AllGather(
                local.raw_data(), temp.raw_data(), local.size(), dtype_, comm_->d_tp_group, stream_);
            sync_check_cuda_error();

            invokeInPlaceTranspose102((uint16_t*)input_embeds.raw_data(),
                                      (uint16_t*)temp.raw_data(),
                                      tp_size_,
                                      token_num,
                                      local_hidden_units,
                                      false,
                                      stream_);
            sync_check_cuda_error();
        }
    }

    if (token_num) {
        // Copy input embeddings from corresponding sequences
        updateEmbedding(
            (char*)input_embeds.raw_data(), h_input_length.size(), h_input_length.data(), sequences, token_num);
        sync_check_cuda_error();
    }

    TM_DEBUG_TENSOR(input_embeds, "embeddings", 1);

    TensorMap args{{"decoder_input", input_embeds},
                   {"decoder_output", hidden_states_out.view({-1, (int)hidden_units_}).borrow()},
                   {"last_token_hidden_units", decoder_out},
                   {"output_norm_weight", weights_->output_norm_weight},
                   {"h_q_len", h_input_length},
                   {"h_k_len", h_context_length},
                   {"finished", finished},
                   {"decode_num", Buffer{&decode_num, 1, kCPU}},
                   {"prefil_num", Buffer{&prefil_num, 1, kCPU}},
                   {"rope_base", rope_base},
                   {"cu_block_nums", cu_block_nums},
                   {"kv_block_ptrs", kv_block_ptrs},
                   {"local_token_nums", local_token_nums}};

    if (mrope != nullptr && mrope->position_ids) {
        args.insert({"mrope_position_ids", mrope->position_ids});
        args.insert({"mrope_position_delta", mrope->position_delta});
        args.insert({"mrope_position_length", mrope->length});
    }

    unified_decoder_->Forward(args, weights_->decoder_layer_weights);
}

Tensor LlamaV2::postDecodeEmbedding(const Tensor& features, Buffer local_logits)
{
    NvtxScope scope("postDecodeEmbedding");
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    TM_CHECK(vocab_size_padded_ % tp_size_ == 0) << vocab_size_padded_ << " " << tp_size_;

    const int bsz              = features.shape(0);
    const int local_vocab_size = vocab_size_padded_ / tp_size_;

    if (tp_size_ == 1) {
        Tensor logits{local_logits, {bsz, (int)vocab_size_padded_}};
        linear_.Forward(features, weights_->post_decoder_embedding, logits);
        sync_check_cuda_error();

        TM_DEBUG_TENSOR(logits, "logits", 1);
        return logits;
    }
    else if (use_allgather_2d_) {
        Tensor logits{local_logits, {bsz, tp_size_, local_vocab_size}};
        Tensor local = logits.slice({0, tp_rank_, 0}, {-1, 1, -1});
        linear_.Forward(features, weights_->post_decoder_embedding, local.squeeze(1));
        sync_check_cuda_error();
        comm_->d_comm->AllGather2D(local.raw_data(),
                                   logits.raw_data(),
                                   vocab_size_padded_,
                                   local_vocab_size,
                                   local_vocab_size,
                                   bsz,
                                   logits.dtype(),
                                   {true, true},
                                   comm_->d_tp_group,
                                   stream_);
        sync_check_cuda_error();
        return logits.view({bsz, -1});
    }
    else {
        Tensor logits{local_logits, {tp_size_, bsz, local_vocab_size}};
        Tensor local = logits.slice({tp_rank_, 0, 0}, {1, -1, -1});
        linear_.Forward(features, weights_->post_decoder_embedding, local.squeeze(0));
        sync_check_cuda_error();
        comm_->d_comm->AllGather(
            local.raw_data(), logits.raw_data(), local.size(), local.dtype(), comm_->d_tp_group, stream_);
        sync_check_cuda_error();
        Tensor out{{bsz, (int)vocab_size_padded_}, features.dtype(), features.device()};
        invokeTransposeAxis01(
            (uint16_t*)out.raw_data(), (uint16_t*)logits.raw_data(), tp_size_, bsz, local_vocab_size, stream_);
        sync_check_cuda_error();
        return out;
    }
}

void LlamaV2::dynamicDecode(Buffer token_ids,
                            Buffer finished,
                            Buffer sequence_length,
                            Tensor curand_state,
                            Tensor logits,
                            Buffer seq_limit_len,
                            Buffer init_context_length,
                            Buffer context_length,
                            Buffer prompt_length,
                            Buffer sampled_logprobs,
                            Buffer sampled_indexes,
                            Buffer sampled_nums,
                            int    step,
                            int    max_context_len)
{
    NvtxScope scope("dynamicDecode");
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    TensorMap args{
        {"logits", logits},
        {"step", Buffer{&step, 1, kCPU}},
        {"max_input_length", Buffer{&max_context_len, 1, kCPU}},
        {"sequence_limit_length", seq_limit_len},
        {"init_context_length", init_context_length},
        {"context_length", context_length},
        {"prompt_length", prompt_length},
        {"output_ids", token_ids},             // inout
        {"finished", finished},                // inout
        {"sequence_length", sequence_length},  // inout
        {"curand_state", curand_state},        // inout
    };

    if (sampled_logprobs) {
        args.emplace("sampled_logprobs", sampled_logprobs);
        args.emplace("sampled_indexes", sampled_indexes);
        args.emplace("sampled_nums", sampled_nums);
    }

    // dynamic_decode_->Forward(args);
}

}  // namespace turbomind
