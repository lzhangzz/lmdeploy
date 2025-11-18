
#include <memory>

#include "src/turbomind/layers/generation/generation.h"

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/exchange.h"
#include "src/turbomind/core/state.h"
#include "src/turbomind/engine/request.h"

#include "src/turbomind/layers/generation/logits_processor.h"
#include "src/turbomind/layers/generation/sampling.h"
#include "src/turbomind/layers/generation/stop_criteria.h"

#include "src/turbomind/kernels/gpt_kernels.h"            // invokeTranspose2D
#include "src/turbomind/kernels/sampling_topk_kernels.h"  // InitializeRandomStates
#include "src/turbomind/models/llama/llama_kernels.h"     // invokePadLastTokenIds

#include "dbg.h"

namespace turbomind {

// token in:       C
// cache in:  HHHHH
// cache out: HHHHHC
// token out:       G
// -------------------------
// token in:        C
// cache in:  HHHHHH
// cache out: HHHHHHC
// token out:        G

// token in:       C
// cache in:  HHHHH
// cache out: HHHHHC
// token out:       G
// draft out:        DDDD
// --------------------------
// token in:        CDDDD
// cache in:  HHHHHH
// cache out: HHHHHHCDDDD
// token out:        VVG
// draft out:           DDDD
// --------------------------
// token in:           CDDDD
// cache in:  HHHHHHHHH
// cache out: HHHHHHHHHCDDDD
// token out:           VG
// draft out:             DDDD

struct GenerationData {
    Buffer_<uint8_t>  random_state;
    Buffer_<uint64_t> random_seed;
    Buffer_<bool>     random_init;

    Buffer_<int> max_seq_len;

    Buffer_<int> token_ids;
    Buffer_<int> token_ids_offsets;

    Buffer_<int>  output_ids;
    Buffer_<bool> finished;

    bool random_init_needed;
    int  max_context_len;
};

struct Generation::Impl {

    // child modules
    std::unique_ptr<LogitsProcessor> logits_processor;
    std::unique_ptr<Sampling>        sampling;
    std::shared_ptr<StopCriteria>    stop_criteria;

    // execution states
    State random_state_;
    State token_ids_;  // (bsz, session_len)
    State token_ids_size_;
    // immutable states
    Buffer_<int> output_ids_;

    std::vector<std::unique_ptr<GenerationData>> data_;

    // staging buffers
    Buffer_<uint8_t>  random_state_buf_;
    Buffer_<uint64_t> random_seed_buf_;
    Buffer_<bool>     random_init_buf_;
    Buffer_<int>      token_ids_buf_;
    Buffer_<bool>     finished_buf_;

    const int max_batch_size_;
    const int session_len_;

    Impl(DataType dtype, int max_batch_size, int session_len, int vocab_size, int vocab_size_padded, int phases):
        max_batch_size_{max_batch_size}, session_len_{session_len}
    {
        TM_CHECK_EQ(dtype, kFloat32);
        BaseGenerationParam base{max_batch_size, vocab_size, vocab_size_padded};
        logits_processor = std::make_unique<LogitsProcessor>(base, phases);
        sampling         = std::make_unique<Sampling>(base, phases);
        stop_criteria    = std::make_unique<StopCriteria>(base, phases);

        static_assert(sizeof(curandState_t) % alignof(curandState_t) == 0);
        random_state_   = {{max_batch_size_, (int)sizeof(curandState_t)}, kUint8, kDEVICE};
        token_ids_      = {{max_batch_size_, session_len_}, kInt, kDEVICE};
        token_ids_size_ = {{max_batch_size_}, kInt, kCPUpinned};  // !
        output_ids_     = {max_batch_size_, kDEVICE};
        Clear(token_ids_size_.front());

        random_state_buf_ = {max_batch_size_ * (int)sizeof(curandState_t), kCPUpinned};
        random_seed_buf_  = {max_batch_size_, kCPUpinned};
        random_init_buf_  = {max_batch_size_, kCPUpinned};
        /// TODO: min(max_batch_size * session_len, total_kv_cache_len)
        token_ids_buf_ = {max_batch_size_ * (ssize_t)session_len_, kCPUpinned};
        finished_buf_  = {max_batch_size_, kCPUpinned};

        for (int i = 0; i < phases; ++i) {
            auto d = std::make_unique<GenerationData>();

            d->random_state = empty_like(random_state_buf_, kDEVICE);
            d->random_seed  = empty_like(random_seed_buf_, kDEVICE);
            d->random_init  = empty_like(random_init_buf_, kDEVICE);
            d->token_ids    = empty_like(token_ids_buf_, kDEVICE);
            d->output_ids   = empty_like(output_ids_, kDEVICE);
            d->finished     = empty_like(finished_buf_, kDEVICE);

            d->token_ids_offsets = {max_batch_size_ + 1, kCPUpinned};

            data_.push_back(std::move(d));
        }
    }

    void Setup(int phase, TensorMap& env)
    {
        auto& d = *data_.at(phase);

        const Buffer_<RequestCache*> rc   = env.at("requests").buffer();
        const Buffer_<int>           perm = env.at("permutation").buffer();

        const int  bs0 = *env.at("bs0").buffer().data<int>();
        const auto bsz = perm.size();

        // random states
        d.random_init_needed = false;
        for (int i = 0; i < perm.size(); ++i) {
            const auto& c = *rc[i];
            if (TM_LIKELY(perm[i] < bs0)) {  // existing
                random_init_buf_[i] = false;
            }
            else if (c.random_state) {  // already initialized
                std::copy_n(
                    c.random_state, sizeof(curandState_t), random_state_buf_.data() + i * sizeof(curandState_t));
            }
            else {  // uninitialized
                d.random_init_needed = true;
                random_init_buf_[i]  = true;
                random_seed_buf_[i]  = rc[i]->gen_cfg.random_seed;
            }
        }
        Copy_(random_state_buf_, bsz, d.random_state);
        if (d.random_init_needed) {
            Copy_(random_init_buf_, bsz, d.random_init);
            Copy_(random_seed_buf_, bsz, d.random_seed);
        }

        // swap-in token_ids
        d.token_ids_offsets[0] = 0;
        for (int i = 0; i < rc.size(); ++i) {
            d.token_ids_offsets[i + 1] = d.token_ids_offsets[i];
            if (const auto& c = *rc[i]; TM_UNLIKELY(perm[i] >= bs0)) {
                std::copy_n(c.token_ids, c.seq_len, token_ids_buf_.data() + d.token_ids_offsets[i]);
                d.token_ids_offsets[i + 1] += c.seq_len;
            }
        }
        if (auto size = d.token_ids_offsets[bsz]) {
            Copy_(token_ids_buf_, size, d.token_ids);
        }

        // TM_LOG_ERROR("bsz = %d, token_ids_offsets = %d", bsz, d.token_ids_offsets[bsz]);
        // TM_LOG_INFO("FUCK %d %d %p", d.token_ids_offsets[0], d.token_ids_offsets[1], &d.token_ids_offsets[0]);

        logits_processor->Setup(phase, env);
        sampling->Setup(phase, env);
        stop_criteria->Setup(phase, env);

        // TM_LOG_INFO("FUCK %d %d", d.token_ids_offsets[0], d.token_ids_offsets[1]);
    }

    void Unprep(int phase, TensorMap& env)
    {
        const int bsz = *env.at("bsz").buffer().data<int>();
        auto&     d   = *data_.at(phase);

        // state -> data
        Copy(random_state_.front().buffer(), bsz * sizeof(curandState_t), d.random_state);

        Buffer_<bool> finished = env.at("finished").buffer();
        Copy(finished, bsz, d.finished);

        Copy(output_ids_, bsz, d.output_ids);
    }

    void Fetch(int phase, TensorMap& env)
    {
        auto& d = *data_.at(phase);

        env.produce("random_state", d.random_state);

        env.produce("finished", d.finished);

        env.produce("output_ids", d.output_ids);
    }

    void Forward(int phase, TensorMap& env)
    {

        TM_CHECK_EQ(phase, 0);
        auto& d = *data_.at(phase);
        TM_LOG_INFO("FUCK %d %d %p", d.token_ids_offsets[0], d.token_ids_offsets[1], &d.token_ids_offsets[0]);

        const Buffer_<int> perm = env.at("permutation").buffer();

        const int bs0 = *env.at("bs0").buffer().data<int>();
        const int bsz = perm.size();

        Warp(random_state_.front(), d.random_state, bs0, perm, random_state_.back(), core::CopyT{});
        random_state_.Swap();

        const auto stream = core::Context::stream().handle();

        if (d.random_init_needed) {
            InitializeRandomStates((curandState_t*)random_state_.front().raw_data(),
                                   d.random_seed.data(),
                                   d.random_init.data(),
                                   bsz,
                                   stream);
            sync_check_cuda_error();
        }

        TM_LOG_INFO("FUCK %d %d", d.token_ids_offsets[0], d.token_ids_offsets[1]);
        Append(token_ids_.front(),
               token_ids_size_.front().buffer(),
               output_ids_,
               d.token_ids,  // from swap-in seqs
               d.token_ids_offsets,
               perm,
               bs0,
               token_ids_.back(),
               token_ids_size_.back().buffer(),
               core::CopyT{});

        token_ids_.Swap();
        token_ids_size_.Swap();

        std::vector x{token_ids_size_.front().data<int>(), token_ids_size_.front().data<int>() + bsz};
        dbg("token_ids_size: ", x);

        env.emplace("output_ids", output_ids_);              // out
        env.emplace("curand_state", random_state_.front());  // inout

        logits_processor->Forward(phase, env);
        sampling->Forward(phase, env);
        stop_criteria->Forward(phase, env);
    }
};

Generation::~Generation() = default;

Generation::Generation(
    DataType dtype, int max_batch_size, int session_len, int vocab_size, int vocab_size_padded, int phases):
    impl_{std::make_unique<Impl>(dtype, max_batch_size, session_len, vocab_size, vocab_size_padded, phases)}
{
}

void Generation::Run(ExchOp op, int phase, TensorMap& env)
{
    if (op == ExchOp::kSetup) {
        return impl_->Setup(phase, env);
    }
    else if (op == ExchOp::kUnprep) {
        return impl_->Unprep(phase, env);
    }
    else if (op == BatchOp::kFetch) {
        return impl_->Fetch(phase, env);
    }
}

void Generation::Forward(int phase, TensorMap& env)
{
    /**
     * @brief
     * input_tensors:
     *   \param  logits [batch_size, beam_width, vocab_size_padded]
     *   \param  input_lengths [batch_size, beam_width], optional
     *   \param  sequence_limit_length [batch_size]
     *   \param  local_batch_size [1] on cpu
     *
     * output_tensors:
     *   \param  output_ids [max_seq_len, batch_size, 1]
     *   \param  curand_state [local_batch_size]
     *   \param  finished [batch_size * beam_width], optional
     *   \param  sequence_length [batch_size * beam_width], optional
     *   \param  sampled_indexes [batch_size, 1, kMaxLogProb], optional
     *   \param  sampled_logprobs [batch_size, 1, kMaxLogProb], optional
     *   \param  sampled_nums [batch_size, 1], optional
     */

    return impl_->Forward(phase, env);
}

}  // namespace turbomind
