
#include "src/turbomind/models/language_model.h"

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/state.h"
#include "src/turbomind/engine/request.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/layers/generation/generation.h"
#include "src/turbomind/models/llama/LlamaWeight.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/unified_decoder.h"
#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"
#include <memory>

namespace turbomind {

using std::vector;
using std::unique_ptr;
using std::shared_ptr;

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

    void Run(BatchOp op, int phase, TensorMap& env)
    {
        if (op != BatchOp::kSetup) {
            return;
        }

        auto& d = data_.at(phase);

        const Buffer_<RequestCache*> rc   = env.at("requests").buffer();
        const Buffer_<int>           perm = env.at("permutation").buffer();

        const int bs0 = *env.at("bs0").data<int>();
        const int bsz = *env.at("bsz").data<int>();

        TM_LOG_ERROR("bs0 = %d, bsz = %d", bs0, bsz);

        //
        d.input_ids_offsets[0] = 0;
        for (int i = 0; i < rc.size(); ++i) {
            d.input_ids_offsets[i + 1] = d.input_ids_offsets[i];
            if (const auto& c = *rc[i]; TM_UNLIKELY(c.seq_len > c.context_len)) {
                const auto src = c.token_ids + c.context_len - c.input_len;
                std::copy_n(src, c.input_len, input_ids_buf_.data() + d.input_ids_offsets[i]);
                d.input_ids_offsets[i + 1] += c.input_len;
                TM_LOG_ERROR("input len = %d", c.input_len);
            }
        }
        ///  seq_len = history + input

        TM_LOG_ERROR("size = %d", d.input_ids_offsets[bsz]);
        if (auto size = d.input_ids_offsets[bsz]) {
            Copy_(input_ids_buf_, size, d.input_ids);
        }

        Buffer_<int> local_token_num{1, kCPU};
        local_token_num[0] = d.input_ids_offsets[bsz];
        env.emplace("local_token_num", local_token_num);
    }

    void Forward(int phase, TensorMap& args)
    {
        auto& d = data_.at(phase);

        const Buffer_<int> perm = args.at("permutation").buffer();

        const int bs0 = *args.at("bs0").data<int>();
        const int bsz = *args.at("bsz").data<int>();

        // last output token + draft tokens
        const Buffer_<int> autoreg_ids         = args.at("autoreg_ids").buffer();
        const Buffer_<int> autoreg_ids_offsets = args.at("autoreg_ids_offsets").buffer();

        Buffer_<int> input_ids{max_forward_token_num_, kDEVICE};
        Warp(autoreg_ids,  // auto-regressive token ids from last iteration T0
             autoreg_ids_offsets,
             bs0,
             d.input_ids,  // input token ids from swap-ins T1
             d.input_ids_offsets,
             perm,
             input_ids,
             input_ids_offsets_buf_,
             core::CopyT{});

        TM_LOG_ERROR("bs0 = %d, bsz = %d", bs0, bsz);

        const int token_num = input_ids_offsets_buf_[bsz];

        Buffer_<int> input_ids_offsets{bsz + 1, kDEVICE};
        Copy_(input_ids_offsets_buf_, bsz + 1, input_ids_offsets);

        args.emplace("input_ids", input_ids.slice(0, token_num));
        args.emplace("input_ids_offsets", input_ids_offsets);
    }

private:
    struct Data {
        Buffer_<int> input_ids;
        Buffer_<int> input_ids_offsets;

        Tensor       input_embeds;
        Buffer_<int> input_embeds_offsets;
    };

private:
    const int max_batch_size_;
    const int max_forward_token_num_;

    std::vector<Data> data_;

    Buffer_<int> input_ids_buf_;
    Buffer_<int> input_ids_offsets_buf_;
};

struct LanguageModel::Impl {
    const DataType       dtype_;
    const ModelParam     param_;
    const AttentionParam attn_param_;
    const Communicators& comm_;
    const LlamaWeight&   weights_;
    LlamaLinear&         linear_;

    const int  tp_size_;
    const int  tp_rank_;
    const bool use_ag2d_;

    const bool debug_;

    // mutable state
    State finished_;
    State sequence_length_;  // length of known tokens
    // immutable state
    Buffer_<int> autoreg_ids_;
    Buffer_<int> autoreg_ids_offsets_;

    Buffer_<int> sequence_length_buf_;

    struct Data {
        Buffer_<int> sequence_length;
    };

    vector<Data> data_;

    std::shared_ptr<InputProcessor> input_processor_;
    std::unique_ptr<UnifiedDecoder> unified_decoder_;
    std::unique_ptr<Generation>     generation_;  // token generator

    Impl(DataType              dtype,
         const ModelParam&     model,
         const EngineParam&    engine,
         const AttentionParam& attn,
         const MoeParam&       moe,
         const Context&        ctx,
         const LlamaWeight&    weights,
         int                   phases);

    Tensor LookupEmbedding(const Buffer_<int>& input_ids, Buffer symm_buf);
    Tensor PostEmbedding(const Tensor& features, Buffer local_logits);

    void Run(BatchOp op, int phase, TensorMap& env)
    {
        if (op == BatchOp::kSetup) {
            Setup(phase, env);
        }
        else if (op == BatchOp::kPrepare) {
            Prepare(phase, env);
        }
        else if (op == BatchOp::kForward) {
            Forward(phase, env);
        }
        else if (op == BatchOp::kFetch) {
            Fetch(phase, env);
        }

        input_processor_->Run(op, phase, env);
        unified_decoder_->Run(op, phase, env);
        generation_->Run(op, phase, env);
    }

    void Setup(int phase, TensorMap& env);
    void Prepare(int phase, TensorMap& env);
    void Forward(int phase, TensorMap& env);
    void Fetch(int phase, TensorMap& env);
};

LanguageModel::Impl::Impl(DataType              dtype,
                          const ModelParam&     model,
                          const EngineParam&    engine,
                          const AttentionParam& attn,
                          const MoeParam&       moe,
                          const Context&        ctx,
                          const LlamaWeight&    weights,
                          int                   phases):
    dtype_{dtype},
    param_{model},
    attn_param_{attn},
    comm_{ctx.comm},
    weights_{weights},
    linear_{*ctx.linear},
    tp_size_{engine.attn_tp_size},
    tp_rank_{engine.attn_tp_rank},
    use_ag2d_{comm_.d_comm && comm_.d_comm->Query(comm::kHasAllGather2D)},
    debug_{isDebug()}
{

    finished_ = {{engine.max_batch_size}, kBool, kDEVICE};

    autoreg_ids_         = {engine.max_batch_size, kDEVICE};
    autoreg_ids_offsets_ = {engine.max_batch_size + 1, kCPU};
    std::fill_n(autoreg_ids_offsets_.data(), autoreg_ids_offsets_.size(), 0);

    input_processor_ = std::make_shared<InputProcessor>(engine, phases);

    unified_decoder_ = std::make_unique<UnifiedDecoder>(model, engine, attn, moe, LoraParam{}, ctx, phases);

    generation_ = std::make_unique<Generation>(kFloat32,
                                               engine.max_batch_size,
                                               engine.session_len,
                                               model.tokenizer_size,
                                               weights.post_decoder_embedding.output_dim * tp_size_,
                                               phases);
}

Tensor LanguageModel::Impl::LookupEmbedding(const Buffer_<int>& input_ids, Buffer symm_buf)
{
    const auto st = core::Context::stream().handle();

    const int hidden_units = param_.hidden_units;

    const auto& embedding_table = weights_.pre_decoder_embedding.weight;
    TM_CHECK_EQ(embedding_table.shape(1) * tp_size_, hidden_units);

    const int token_num = input_ids.size();

    TM_CHECK_GT(token_num, 0);

    Tensor input_embeds{{token_num, hidden_units}, dtype_, kDEVICE};

    if (tp_size_ == 1) {
        invokeEmbeddingLookup(input_embeds, input_ids, embedding_table, st);
        sync_check_cuda_error();
    }
    else if (use_ag2d_) {
        const auto local_hidden_units = embedding_table.shape(1);
        Tensor     temp{symm_buf, {token_num, tp_size_, local_hidden_units}};

        auto local = temp.slice({0, tp_rank_, 0}, {-1, 1, -1}).squeeze(1);

        invokeEmbeddingLookup(local, input_ids, embedding_table, st);
        sync_check_cuda_error();

        comm_.d_comm->AllGather2D(local.raw_data(),
                                  temp.raw_data(),
                                  hidden_units,
                                  local_hidden_units,
                                  local_hidden_units,
                                  token_num,
                                  local.dtype(),
                                  {true, true},
                                  comm_.d_tp_group,
                                  st);
        sync_check_cuda_error();

        Copy(temp.buffer(), input_embeds.buffer());
    }
    else {
        const auto local_hidden_units = embedding_table.shape(1);
        Tensor     temp{symm_buf, {tp_size_, token_num, local_hidden_units}};

        auto local = temp.slice(tp_rank_).squeeze(0);

        invokeEmbeddingLookup(local, input_ids, embedding_table, st);
        sync_check_cuda_error();

        comm_.d_comm->AllGather(local.raw_data(), temp.raw_data(), local.size(), dtype_, comm_.d_tp_group, st);
        sync_check_cuda_error();

        invokeInPlaceTranspose102((uint16_t*)input_embeds.raw_data(),
                                  (uint16_t*)temp.raw_data(),
                                  tp_size_,
                                  token_num,
                                  local_hidden_units,
                                  false,
                                  st);
        sync_check_cuda_error();
    }

    return input_embeds;
}

Tensor LanguageModel::Impl::PostEmbedding(const Tensor& features, Buffer local_logits)
{
    NvtxScope scope("postDecodeEmbedding");

    const auto st = core::Context::stream().handle();

    const int bsz              = features.shape(0);
    const int local_vocab_size = weights_.post_decoder_embedding.output_dim;
    const int vocab_size       = local_vocab_size * tp_size_;

    if (tp_size_ == 1) {
        Tensor logits{{bsz, vocab_size}, dtype_, kDEVICE};
        linear_.Forward(features, weights_.post_decoder_embedding, logits);
        sync_check_cuda_error();
        TM_DEBUG_TENSOR(logits, "logits", 1);
        return logits;
    }
    else if (use_ag2d_) {
        Tensor logits{local_logits, {bsz, tp_size_, local_vocab_size}};
        Tensor local = logits.slice({0, tp_rank_, 0}, {-1, 1, -1});
        linear_.Forward(features, weights_.post_decoder_embedding, local.squeeze(1));
        sync_check_cuda_error();
        comm_.d_comm->AllGather2D(local.raw_data(),
                                  logits.raw_data(),
                                  vocab_size,
                                  local_vocab_size,
                                  local_vocab_size,
                                  bsz,
                                  logits.dtype(),
                                  {true, true},
                                  comm_.d_tp_group,
                                  st);
        sync_check_cuda_error();
        return logits.view({bsz, -1});
    }
    else {
        Tensor logits{local_logits, {tp_size_, bsz, local_vocab_size}};
        Tensor local = logits.slice({tp_rank_, 0, 0}, {1, -1, -1});
        linear_.Forward(features, weights_.post_decoder_embedding, local.squeeze(0));
        sync_check_cuda_error();
        comm_.d_comm->AllGather(local.raw_data(), logits.raw_data(), local.size(), local.dtype(), comm_.d_tp_group, st);
        sync_check_cuda_error();
        Tensor out{{bsz, vocab_size}, features.dtype(), features.device()};
        invokeTransposeAxis01(
            (uint16_t*)out.raw_data(), (uint16_t*)logits.raw_data(), tp_size_, bsz, local_vocab_size, st);
        sync_check_cuda_error();
        return out;
    }
}

void LanguageModel::Impl::Setup(int phase, TensorMap& env)
{
    auto& d = data_.at(phase);

    const Buffer_<RequestCache*> rc   = env.at("requests").buffer();
    const Buffer_<int>           perm = env.at("permutation").buffer();

    const int bs0 = *env.at("bs0").data<int>();
    const int bsz = *env.at("bsz").data<int>();

    for (int i = 0; i < rc.size(); ++i) {
        if (auto& c = *rc[i]; TM_UNLIKELY(perm[i] >= bs0)) {
            sequence_length_buf_[i] = c.seq_len;
        }
    }

    Copy_(sequence_length_buf_, bsz, d.sequence_length);
}

void LanguageModel::Impl::Prepare(int phase, TensorMap& env)
{
    auto& d = data_.at(phase);

    const Buffer_<int> perm = env.at("permutation").buffer();
    const int          bsz  = *env.at("bsz").data<int>();
    const int          bs0  = *env.at("bs0").data<int>();

    Clear(finished_.back().buffer());

    Warp(finished_.front(), bs0, perm, finished_.back(), core::CopyT{});
    finished_.Swap();

    Warp(sequence_length_.front(), d.sequence_length, bs0, perm, sequence_length_.back(), core::CopyT{});
    sequence_length_.Swap();

    // Buffer_<int> context_length()
    // PrefixSum(sequence_length_.front().slice(0, bsz), )

    env.produce("finished", finished_.front());
    env.produce("sequence_length", sequence_length_.front());

    env.emplace("autoreg_ids", autoreg_ids_);
    env.emplace("autoreg_ids_offsets", autoreg_ids_offsets_);
}

void LanguageModel::Impl::Forward(int phase, TensorMap& env)
{
    input_processor_->Forward(phase, env);  // input_ids

    Buffer symm_buf;
    if (auto buf = env.try_("symm_buf")) {
        symm_buf = buf->buffer();
    }

    {
        auto   input_ids    = env.at("input_ids").buffer();
        Tensor input_embeds = LookupEmbedding(input_ids, symm_buf);
        env.emplace("decoder_input", input_embeds);
    }

    const int global_token_num = *env.at("global_token_num").data<int>();
    const int bsz              = *env.at("bsz").data<int>();

    Tensor decoder_output{{global_token_num, (int)param_.hidden_units}, dtype_, kDEVICE};
    Tensor decode_hidden_states{{bsz, (int)param_.hidden_units}, dtype_, kDEVICE};

    env.emplace("decoder_output", decoder_output);
    env.emplace("output_norm_weight", weights_.output_norm_weight);
    env.emplace("decode_hidden_states", decode_hidden_states);
    unified_decoder_->Forward(phase, env, weights_.decoder_layer_weights);

    // output hidden states
    // output logits

    if (auto decode_hidden_states = env.at("decode_hidden_states")) {
        auto logits     = PostEmbedding(decode_hidden_states, symm_buf);
        auto f32_logits = empty_like(logits, kFloat32);
        invokeCastFloat2D(logits, f32_logits, core::Context::stream().handle());
        env.emplace("logits", f32_logits);
        generation_->Forward(phase, env);
    }
}

void LanguageModel::Impl::Fetch(int phase, TensorMap& env) {}

LanguageModel::~LanguageModel() = default;

LanguageModel::LanguageModel(LanguageModel&&) noexcept = default;

LanguageModel::LanguageModel(DataType              dtype,
                             const ModelParam&     model,
                             const EngineParam&    engine,
                             const AttentionParam& attn,
                             const MoeParam&       moe,
                             const Context&        ctx,
                             const LlamaWeight&    weights,
                             int                   phases)
{
    impl_ = std::make_unique<Impl>(dtype, model, engine, attn, moe, ctx, weights, phases);
}

void LanguageModel::Run(BatchOp op, int phase, TensorMap& env)
{
    return TM_CHECK_NOTNULL(impl_)->Run(op, phase, env);
}

const ModelParam& LanguageModel::model_param() const noexcept
{
    return TM_CHECK_NOTNULL(impl_)->param_;
}

const AttentionParam& LanguageModel::attn_param() const noexcept
{
    return TM_CHECK_NOTNULL(impl_)->attn_param_;
}

}  // namespace turbomind