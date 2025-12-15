
#include "src/turbomind/models/language_model.h"

#include <memory>

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/copy.h"
#include "src/turbomind/core/exchange.h"
#include "src/turbomind/core/interval.h"
#include "src/turbomind/core/state.h"
#include "src/turbomind/engine/batch_data.h"
#include "src/turbomind/engine/request.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/layers/generation/generation.h"
#include "src/turbomind/models/llama/LlamaWeight.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/unified_decoder.h"
#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"

#include "dbg.h"

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
        decode_token_pos_buf_  = {max_batch_size_, kCPUpinned};

        data_.reserve(phases);
        for (int i = 0; i < phases; ++i) {
            auto& d              = data_.emplace_back();
            d.input_ids          = empty_like(input_ids_buf_, kDEVICE);
            d.input_ids_offsets  = empty_like(input_ids_offsets_buf_, kDEVICE);
            d.selected_token_pos = empty_like(decode_token_pos_buf_, kDEVICE);

            d.autoreg_ids_pos = {max_batch_size_, kCPU};  // !
        }
    }

    void Setup(int phase, TensorMap& env)
    {
        auto& d = data_.at(phase);

        const Buffer_<RequestCache*> rc   = env.at("requests").buffer();
        const Buffer_<int>           perm = env.at("permutation").buffer();

        const int bs0 = *env.at("bs0").data<int>();
        const int bsz = *env.at("bsz").data<int>();

        auto& copy = *env.at("copy").data<BatchCopy*>()[0];
        // core::CopyT copy{};

        input_ids_offsets_buf_[0] = 0;
        for (int i = 0; i < rc.size(); ++i) {
            input_ids_offsets_buf_[i + 1] = input_ids_offsets_buf_[i];
            if (const auto& c = *rc[i]; TM_UNLIKELY(!c.is_decoding)) {
                const auto src = c.token_ids + c.history_len + c.alpha;
                std::copy_n(src, c.input_len, input_ids_buf_.data() + input_ids_offsets_buf_[i]);
                // dbg(std::vector<int>(src, src + c.input_len));
                d.autoreg_ids_pos[i] = -1;
                input_ids_offsets_buf_[i + 1] += c.input_len;
            }
            else {
                d.autoreg_ids_pos[i] = input_ids_offsets_buf_[i];
                input_ids_offsets_buf_[i + 1] += 1;
            }
            decode_token_pos_buf_[i] = input_ids_offsets_buf_[i + 1] - 1;
        }

        // dbg(core::to_vector<int>(input_ids_offsets_buf_.slice(0, bsz + 1)));
        // dbg(core::to_vector<int>(decode_token_pos_buf_.slice(0, bsz)));

        copy(input_ids_buf_, input_ids_offsets_buf_[bsz], d.input_ids);
        copy(decode_token_pos_buf_, bsz, d.selected_token_pos);
        copy(input_ids_offsets_buf_, bsz + 1, d.input_ids_offsets);

        // dbg(decode_token_pos_buf_[0]);

        d.input_token_num = input_ids_offsets_buf_[bsz];
        // dbg(d.input_token_num);

        env.produce("local_token_num", Buffer{&d.input_token_num, 1, kCPU});
    }

    void Prepare(int phase, TensorMap& env)
    {
        auto& d = data_.at(phase);

        const Buffer_<int> perm = env.at("permutation").buffer();

        const int bs0  = *env.at("bs0").data<int>();
        const int bsz  = *env.at("bsz").data<int>();
        auto&     copy = *env.at("copy").data<BatchCopy*>()[0];

        // last output token + draft tokens
        const Buffer_<int> autoreg_ids = env.at("autoreg_ids").buffer();

        // core::CopyT copy{};

        if (auto g = copy.group()) {
            for (int i = 0; i < bsz; ++i) {
                if (auto pos = d.autoreg_ids_pos[i]; pos >= 0) {
                    TM_CHECK_LT(perm[i], bs0);
                    copy(autoreg_ids.data() + perm[i], 1, &d.input_ids[pos]);
                }
            }
        }

        env.produce("input_ids", d.input_ids.slice(0, d.input_token_num));
        env.produce("q_offsets", d.input_ids_offsets.slice(0, bsz + 1));
        env.produce("selected_token_pos", d.selected_token_pos.slice(0, bsz));
    }

    void Run(BatchOp op, int phase, TensorMap& env)
    {
        switch (op) {
            case BatchOp::kSetup:
                return Setup(phase, env);
            case BatchOp::kPrepare:
                return Prepare(phase, env);
            default:
                return;
        }
    }

private:
    struct Data {
        Buffer_<int> input_ids;
        Buffer_<int> input_ids_offsets;
        int          input_token_num;

        Buffer_<int> selected_token_pos;

        Buffer_<int> autoreg_ids_pos;

        Tensor       input_embeds;
        Buffer_<int> input_embeds_offsets;
    };

private:
    const int max_batch_size_;
    const int max_forward_token_num_;

    std::vector<Data> data_;

    Buffer_<int> input_ids_buf_;
    Buffer_<int> input_ids_offsets_buf_;

    Buffer_<int> decode_token_pos_buf_;
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

    Buffer_<bool> false_;

    // mutable state
    State finished_;
    State sequence_length_;  // length of known tokens
    // immutable state
    Buffer_<int> autoreg_ids_;
    // Buffer_<int> autoreg_ids_offsets_;

    // Symmetric buffer for holding global hidden states or logits
    Buffer_<uint8_t> symm_buf_;

    // Max chunk size for compute / output full logits
    int max_logits_len_ = 0;

    Buffer_<int>  sequence_length_buf_;
    Buffer_<bool> finished_buf_;

    struct Data {
        Buffer_<int>  sequence_length;
        Buffer_<bool> finished;

        Buffer_<bool> is_decoding;
        Buffer_<bool> is_generate;

        int generative;

        Interval full_hstate;  // requested range for full hidden states
        Interval full_logits;  // requested range for full logits

        // prevent requests being freed before hstate/logits are written
        vector<shared_ptr<Request>> requests;

        vector<std::tuple<int, int, Interval, Interval>> output_hstate;
        vector<std::tuple<int, int, Interval, Interval>> output_logits;
    };

    vector<Data> data_;

    std::shared_ptr<InputProcessor> input_processor_;
    std::unique_ptr<UnifiedDecoder> unified_decoder_;
    std::unique_ptr<Generation>     generation_;  // token generator

    void Run(BatchOp op, int phase, TensorMap& env)
    {
        switch (op) {
            case BatchOp::kSetup:
                return Setup(phase, env);
            case BatchOp::kPrepare:
                return Prepare(phase, env);
            case BatchOp::kForward:
                return Forward(phase, env);
            case BatchOp::kUnprep:
                return Unprep(phase, env);
            case BatchOp::kFetch:
                return Fetch(phase, env);
            default:
                input_processor_->Run(op, phase, env);
                unified_decoder_->Run(op, phase, env);
                generation_->Run(op, phase, env);
        }
    }

    Impl(DataType              dtype,
         const ModelParam&     model,
         const EngineParam&    engine,
         const AttentionParam& attn,
         const MoeParam&       moe,
         const Context&        ctx,
         const LlamaWeight&    weights,
         int                   phases);

    Tensor LookupEmbedding(const Buffer_<int>& input_ids, Buffer symm_buf);
    Tensor PostEmbedding(const Tensor& features, Buffer symm_buf);

    void Setup(int phase, TensorMap& env);
    void Prepare(int phase, TensorMap& env);
    void Forward(int phase, TensorMap& env);
    void Unprep(int phase, TensorMap& env);
    void Fetch(int phase, TensorMap& env);

    template<class Ranges>
    void OutputHiddenStates(const Ranges& ranges, const Tensor& h, int type, const vector<shared_ptr<Request>>& rs)
    {
        for (const auto& [i, t, src, dst] : ranges) {
            if (t == type) {
                auto& out = rs[i]->outputs.at("last_hidden_state");
                if (tp_rank_ == 0) {
                    dbg(&src, &dst);
                    Copy(h.slice(src.begin(), (int)src.size()), out.slice(dst.begin(), (int)dst.size()));
                }
            }
        }
    }

    void ComputeAndOutputLogits(const Data& data, const Tensor& h, const vector<shared_ptr<Request>>& rs)
    {

        const int step_size = max_logits_len_;

        TM_CHECK_GT(step_size, 0);

        // Coroutine frame
        int  p      = 0;
        auto ranges = data.output_logits;

        bool success = false;
        // Erode the range iteratively until empty
        for (auto r = data.full_logits; r; r = -step_size | r) {
            dbg(&r);
            if (auto chunk = r & Interval{r.begin(), step_size}) {
                dbg(&chunk);
                // Compute & output full logits by chunks
                auto logits = PostEmbedding(h.slice(chunk.begin(), (int)chunk.size()), symm_buf_);
                success     = OutputLogitsImpl(ranges, p, logits, chunk.begin(), 2, rs);
                if (success) {
                    // all requests satisfied, exit early
                    break;
                }
            }
        }

        TM_CHECK(success);
    }

    template<class Ranges>
    void OutputLogits(Ranges& ranges_, const Tensor& l, int type, const vector<shared_ptr<Request>>& rs)
    {
        // Coroutine frame
        int  p      = 0;
        auto ranges = ranges_;

        TM_CHECK(OutputLogitsImpl(ranges, p, l, /* base */ 0, type, rs));
    }

    template<class Ranges>
    bool
    OutputLogitsImpl(Ranges& ranges, int& p, const Tensor& l, int base, int type, const vector<shared_ptr<Request>>& rs)
    {
        const auto stream = core::Context::stream().handle();
        for (; p < ranges.size(); ++p) {
            if (auto& [i, t, src, dst] = ranges[p]; t == type) {
                Tensor&        out   = rs[i]->outputs.at("logits");
                const DataType dtype = out.dtype();
                TM_CHECK_LE(base, src.begin());  // logical error
                if (Interval msrc = src & Interval{base, Interval::Size{(int)l.shape(0)}}) {
                    const int tokens = (int)msrc.size();
                    Interval  mdst{dst.begin(), msrc.size()};
                    // TODO: support strides in `DLTensor`, so that batched 1D copy can be used
                    if (tp_rank_ == 0) {
                        TM_CHECK_EQ(cudaMemcpy2DAsync(out.slice(mdst.begin(), tokens).raw_data(),
                                                      byte_size(dtype, out.stride(0)),
                                                      l.slice(msrc.begin(), tokens).raw_data(),
                                                      byte_size(dtype, l.stride(0)),
                                                      byte_size(dtype, param_.vocab_size),
                                                      tokens,
                                                      cudaMemcpyDefault,
                                                      stream),
                                    0);
                    }
                    // move to next request if they are empty after the erosion
                    src = -(int)msrc.size() | src;
                    dst = -(int)mdst.size() | dst;
                }
                if (src) {
                    // request not compeleted, suspend and wait for next chunk
                    return false;
                }
            }
        }
        return true;
    }
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

    false_ = {engine.max_batch_size, kDEVICE};
    Clear(false_);

    finished_buf_ = {engine.max_batch_size, kCPUpinned};
    finished_     = {{engine.max_batch_size}, kBool, kDEVICE};

    autoreg_ids_ = {engine.max_batch_size, kDEVICE};
    // autoreg_ids_offsets_ = {engine.max_batch_size + 1, kCPU};
    // std::fill_n(autoreg_ids_offsets_.data(), autoreg_ids_offsets_.size(), 0);

    sequence_length_buf_ = {engine.max_batch_size, kCPUpinned};
    sequence_length_     = {{engine.max_batch_size}, kInt, kDEVICE};
    for (int i = 0; i < phases; ++i) {
        auto& d           = data_.emplace_back();
        d.sequence_length = empty_like(sequence_length_buf_, kDEVICE);
        d.finished        = empty_like(finished_buf_, kDEVICE);
        d.is_decoding     = {engine.max_batch_size, kCPU};
        d.is_generate     = {engine.max_batch_size, kCPU};
    }

    input_processor_ = std::make_shared<InputProcessor>(engine, phases);

    unified_decoder_ = std::make_unique<UnifiedDecoder>(model, engine, attn, moe, LoraParam{}, ctx, phases);

    generation_ = std::make_unique<Generation>(kFloat32,
                                               engine.max_batch_size,
                                               engine.session_len,
                                               model.tokenizer_size,
                                               weights.post_decoder_embedding.output_dim * tp_size_,
                                               phases);

    const int     vocab_size     = weights_.post_decoder_embedding.output_dim * tp_size_;
    const ssize_t max_fwd_tokens = engine.max_forward_token_num;

    if (ctx.comm.d_comm) {
        auto symm_alloc = GetSymmAllocator(ctx.comm.d_comm);
        // Native comm fuses allreduce & rmsnorm in token granularity
        TM_CHECK(engine.max_forward_token_num % tp_size_ == 0);

        ssize_t bytes{};
        bytes = std::max(bytes, byte_size(dtype_, max_fwd_tokens * engine.attn_dp_size * model.hidden_units));
        bytes = std::max(bytes, byte_size(dtype_, engine.max_batch_size * model.vocab_size_padded));

        symm_buf_ = {bytes, symm_alloc};
        // Compute max logits length based on symm buffer size
        max_logits_len_ = symm_buf_.view(dtype_).size() / vocab_size;
    }
    else {
        max_logits_len_ = std::max<int>(max_fwd_tokens * model.hidden_units / vocab_size, engine.max_batch_size);
    }
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

        Tensor temp{symm_buf.view(dtype_), {token_num, tp_size_, local_hidden_units}};
        Tensor local{temp.slice({0, tp_rank_, 0}, {-1, 1, -1}).squeeze(1)};

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

        Tensor temp{symm_buf.view(dtype_), {tp_size_, token_num, local_hidden_units}};
        Tensor local{temp.slice(tp_rank_).squeeze(0)};

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

Tensor LanguageModel::Impl::PostEmbedding(const Tensor& features, Buffer symm_buf)
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
        Tensor logits{symm_buf.view(dtype_), {bsz, tp_size_, local_vocab_size}};
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
        Tensor logits{symm_buf.view(dtype_), {tp_size_, bsz, local_vocab_size}};
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

struct IntervalMatching {
    Interval& target;
    const int offset_d;
    Interval  src;
    Interval  dst;

    bool operator()(const Interval& x, int offset_s, Interval& merged)
    {
        if (auto y = target & x; y && y.begin() == target.begin()) {
            dst    = {y.begin() - offset_d, y.size()};
            src    = {offset_s + (y.begin() - x.begin()), y.size()};
            merged = merged | src;
            target = -(int)y.size() | target;
            return true;
        }
        return false;
    }
};

void LanguageModel::Impl::Setup(int phase, TensorMap& env)
{
    input_processor_->Run(BatchOp::kSetup, phase, env);

    auto& d = data_.at(phase);

    const Buffer_<RequestCache*> rc   = env.at("requests").buffer();
    const Buffer_<int>           perm = env.at("permutation").buffer();

    const int bs0  = *env.at("bs0").data<int>();
    const int bsz  = *env.at("bsz").data<int>();
    auto&     copy = *env.at("copy").data<BatchCopy*>()[0];

    d.generative = 0;

    for (int i = 0; i < rc.size(); ++i) {
        auto& c          = *rc[i];
        d.is_decoding[i] = c.is_decoding;
        d.is_generate[i] = c.is_generate;
        d.generative += c.is_generate;
        if (TM_UNLIKELY(!c.is_decoding)) {
            sequence_length_buf_[i] = c.history_len + c.alpha + c.input_len;
        }
    }

    // core::CopyT copy{};
    copy(sequence_length_buf_, bsz, d.sequence_length);

    vector<Interval> all_tokens;
    vector<Interval> sel_tokens;

    for (int i = 0; i < rc.size(); ++i) {
        using Size = Interval::Size;
        auto& c    = *rc[i];
        all_tokens.emplace_back(c.history_len + c.alpha, Size{c.input_len});
        sel_tokens.emplace_back(c.history_len + c.alpha + c.input_len - 1, Size{1});
        if (!d.is_generate[i]) {
            sel_tokens.back() = {};
        }
    }

    const int token_num = *env.at("local_token_num").data<int>();

    d.full_logits = {INT_MAX, 0};
    d.full_hstate = {INT_MAX, 0};

    Interval select_states{INT_MAX, 0};
    Interval select_logits{INT_MAX, 0};

    d.output_logits = {};
    d.output_hstate = {};

    int offset = 0;

    vector<shared_ptr<Request>> rs;
    rs.reserve(rc.size());
    for (int i = 0; i < rc.size(); ++i) {
        rs.push_back(rc[i]->request);
    }
    d.requests.swap(rs);

    for (int i = 0; i < rc.size(); ++i) {
        auto& c = *rc[i];
        auto& g = c.request->gen_cfg;
        if (c.output_hidden_states) {
            IntervalMatching m{c.output_hidden_states, c.hidden_states_offset};
            int              type = 0;
            if (m(sel_tokens[i], i, select_states)) {
                type = 1;
            }
            else if (m(all_tokens[i], offset, d.full_hstate)) {
                type = 2;
            }
            if (type) {
                d.output_hstate.emplace_back(i, type, m.src, m.dst);
            }
        }
        if (c.output_logits) {
            IntervalMatching m{c.output_logits, c.logits_offset};
            int              type = 0;
            if (m(sel_tokens[i], i, select_logits)) {
                type = 1;
            }
            else if (m(all_tokens[i], offset, d.full_logits)) {
                type = 2;
            }
            if (type) {
                d.output_logits.emplace_back(i, type, m.src, m.dst);
            }
        }
        offset += c.input_len;
    }

    // logits depends on hidden states
    d.full_hstate = d.full_hstate | d.full_logits;

    unified_decoder_->Run(BatchOp::kSetup, phase, env);
    generation_->Run(BatchOp::kSetup, phase, env);
}

void LanguageModel::Impl::Prepare(int phase, TensorMap& env)
{
    env.emplace("autoreg_ids", autoreg_ids_);

    input_processor_->Run(BatchOp::kPrepare, phase, env);

    auto& d = data_.at(phase);

    const Buffer_<int> perm = env.at("permutation").buffer();
    const int          bsz  = *env.at("bsz").data<int>();
    const int          bs0  = *env.at("bs0").data<int>();
    auto&              copy = *env.at("copy").data<BatchCopy*>()[0];

    // core::CopyT copy{};

    if (auto group = copy.group()) {
        for (int i = 0; i < bsz; ++i) {
            if (const int j = perm[i]; j < bs0) {
                copy(finished_.front().data<bool>() + j, 1, finished_.back().data<bool>() + i);
            }
            else {
                copy(false_.data() + i, 1, finished_.back().data<bool>() + i);
            }
        }
        finished_.Swap();
    }

    if (auto group = copy.group()) {
        // sequence_length = history_len + input_len
        for (int i = 0; i < bsz; ++i) {
            if (const int j = perm[i]; j < bs0 && d.is_decoding[i]) {
                copy(sequence_length_.front().data<int>() + j, 1, sequence_length_.back().data<int>() + i);
            }
            else {
                copy(d.sequence_length.data() + i, 1, sequence_length_.back().data<int>() + i);
            }
        }
        sequence_length_.Swap();
    }

    Buffer_<int> k_offsets{bsz + 1, kDEVICE};
    // PrefixSum(sequence_length_.front().data<int>(), bsz, k_offsets.data(), core::Context::stream().handle());

    // Buffer_<int> k_offsets_tmp{k_offsets.size(), kCPU};
    // Buffer_<int> sequence_length_tmp{sequence_length_.front().size(), kCPU};

    // Copy(k_offsets, k_offsets_tmp);
    // Copy(sequence_length_.front().buffer(), sequence_length_tmp);

    // core::Context::stream().Sync();

    // dbg(core::to_vector<int>(sequence_length_tmp.slice(0, bsz)));
    // dbg(core::to_vector<int>(k_offsets_tmp.slice(0, bsz + 1)));

    env.produce("finished", finished_.front());
    env.produce("sequence_length", sequence_length_.front());
    env.produce("k_offsets", k_offsets);

    unified_decoder_->Run(BatchOp::kPrepare, phase, env);
    generation_->Run(BatchOp::kPrepare, phase, env);
}

void LanguageModel::Impl::Forward(int phase, TensorMap& env)
{
    const int bsz = *env.at("bsz").data<int>();

    auto& d = data_.at(phase);

    {
        Buffer_<int> k_offsets = env.at("k_offsets").buffer();
        PrefixSum(sequence_length_.front().data<int>(), bsz, k_offsets.data(), core::Context::stream().handle());
    }

    input_processor_->Run(BatchOp::kForward, phase, env);  // input_ids

    {
        auto   input_ids    = env.at("input_ids").buffer();
        Tensor input_embeds = LookupEmbedding(input_ids, symm_buf_);
        TM_DEBUG_TENSOR(input_embeds, "embeddings", 1);
        env.produce("input_embeds", std::move(input_embeds));
    }

    if (d.full_hstate) {
        env.produce("output_hidden_states", {});
    }

    if (symm_buf_) {
        env.produce("symm_buf", symm_buf_);
    }

    env.produce("output_norm_weight", weights_.output_norm_weight);

    unified_decoder_->Forward(phase, env, weights_.decoder_layer_weights);

    // env.at("batch").data<BatchData*>()[0]->Notify();

    auto select_hstate = env.try_consume("selected_hidden_states");
    auto full_hstate   = env.try_consume("hidden_states");

    auto& rs = d.requests;

    OutputHiddenStates(d.output_hstate, select_hstate, 1, d.requests);
    if (d.full_hstate) {
        OutputHiddenStates(d.output_hstate, full_hstate, 2, d.requests);
    }

    if (d.full_logits) {
        ComputeAndOutputLogits(d, full_hstate, d.requests);
    }

    Tensor logits;
    if (select_hstate) {
        logits = PostEmbedding(select_hstate, symm_buf_);
        OutputLogits(d.output_logits, logits, 1, d.requests);
    }

    if (d.generative) {
        TM_CHECK(logits);
        env.emplace("logits", logits);
        generation_->Run(BatchOp::kForward, phase, env);
        Copy(env.at("output_ids").buffer(), autoreg_ids_);
    }
}

void LanguageModel::Impl::Unprep(int phase, TensorMap& env)
{
    auto& d    = data_.at(phase);
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    copy(sequence_length_.front().buffer(), d.sequence_length.size(), d.sequence_length);

    copy(finished_.front().buffer(), d.finished.size(), d.finished);

    generation_->Run(BatchOp::kUnprep, phase, env);
}

void LanguageModel::Impl::Fetch(int phase, TensorMap& env)
{
    auto& d    = data_.at(phase);
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    copy(d.sequence_length, d.sequence_length.size(), sequence_length_buf_);
    env.produce("sequence_length", sequence_length_buf_);

    copy(d.finished, d.finished.size(), finished_buf_);
    env.produce("finished", finished_buf_);

    env.produce("is_generate", d.is_generate);
    env.produce("is_decoding", d.is_decoding);

    generation_->Run(BatchOp::kFetch, phase, env);
}

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