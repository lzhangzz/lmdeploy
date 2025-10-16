
#include "nvtx3/nvToolsExt.h"

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/engine/engine.h"
#include "src/turbomind/engine/model_executor.h"
#include "src/turbomind/engine/request.h"
#include <algorithm>

namespace turbomind {

using std::shared_ptr;
using std::vector;

struct RequestData {
    vector<shared_ptr<Request>> infer;  // incoming inference request
    vector<shared_ptr<Request>> kill;   // incoming kill request

    vector<int> cancel;  // canceled indices in current batch
    bool        abort;
};

void Engine::Accept(const Requests& rs, std::vector<Signal>& signals)
{
    auto get_next_idx = [&, idx = 0]() mutable {
        while (idx < max_batch_size_ && info_[idx] && ++idx) {}
        return idx;
    };

    for (const auto& r : rs) {

        if (r->ec) {
            signals.push_back([r] { UpdateState(*r, r->ec, 0); });
            continue;
        }

        const int input_length = r->inputs.at("input_ids").shape(0);

        if (input_length > session_len_) {
            signals.push_back([r] { UpdateState(*r, Request::kTooLong, 0); });
            continue;
        }

        auto ptr = r->session.start_flag ? seq_mgr_->Create(r->id) : seq_mgr_->Get(r->id);
        if (!ptr) {
            signals.push_back([r] { UpdateState(*r, Request::kInvalid, 0); });
            continue;
        }

        const int step = [&] {
            int s = r->session.step;
            if (s < 0) {
                s = ptr->tokens.size();
            }
            else if (s > ptr->tokens.size()) {
                if (tp_rank_ == 0) {
                    TM_LOG_WARNING("[ProcessInferRequests] Skipping invalid step (%d) setting for ID %lu", s, ptr->id);
                }
                s = ptr->tokens.size();
            }
            return s;
        }();

        if (step + input_length > session_len_) {
            signals.push_back([r] { UpdateState(*r, Request::kTooLong, 0); });
            continue;
        }

        const int idx = get_next_idx();

        TM_CHECK(info_[idx] == nullptr);

        auto& seq = *ptr;

        auto info = std::make_shared<RequestInfo>();

        if (step < seq.tokens.size()) {
            seq.tokens.resize(step);
            seq.cache_len = std::min(seq.cache_len, step);
        }

        {
            const int* input_ids = r->inputs.at("input_ids").data<int>();

            info->token_ids = r->output_ids.data();
            int* token_ids  = info->token_ids;

            token_ids = std::copy_n(seq.tokens.data(), seq.tokens.size(), token_ids);
            token_ids = std::copy_n(input_ids, input_length, token_ids);

            (void)token_ids;
        }

        info->request        = r;
        info->input_length   = input_length;
        info->context_length = seq.tokens.size() + input_length;
        info->prompt_length  = info->context_length;

        const int max_new_tokens = info->request->gen_cfg.max_new_tokens;

        int max_seq_len = info->context_length + max_new_tokens;
        // `length_criterion` sets finish flag when step >= seq_limit_len, however when step == seq_limit_len
        // the actual sequence length is seq_limit_len + 1, hence seq_limit_len must truncated to session_len - 1
        if (max_seq_len >= session_len_) {
            max_seq_len = session_len_ - 1;
            if (tp_rank_ == 0) {
                const int trunc_output_len = max_seq_len - info->context_length;
                TM_LOG_WARNING(
                    "[ProcessInferRequests] [%ld] total sequence length (%d + %d) exceeds `session_len` (%d), `max_new_tokens` is truncated to %d",
                    (long)seq.id,
                    info->context_length,
                    max_new_tokens,
                    (int)session_len_,
                    trunc_output_len);
            }
        }

        info->max_seq_len = max_seq_len;

        info_[idx] = info;
    }

    batch_size_ = std::max(batch_size_, get_next_idx());
}

void Engine::Schedule(SchedBatch& batch)
{
    vector<const Sequence*>  sequences;
    vector<Sequence::Status> status;
    vector<uint64_t>         priorities;
    vector<int>              context_lengths;

    for (int i = 0; i < batch.size; ++i) {
        const auto& info = info_[i];
        sequences.push_back(info->sequence);
        status.push_back(info->sequence->status);
        priorities.push_back(info->request->unique_id);
        context_lengths.push_back(info->context_length);
    }

    auto adjust = [this](const Sequences&, const std::vector<int>&) -> int { return max_forward_token_num_; };

    auto outcome = seq_mgr_->Materialize(sequences, context_lengths, priorities, 1, adjust);

    std::vector<int> idxs(sequences.size());
    std::iota(idxs.begin(), idxs.end(), 0);

    auto inactive = std::stable_partition(idxs.begin(), idxs.end(), [&](int idx) {
        return sequences[idx]->status == Sequence::kActive;  // current status
    });

    if (!sequences.empty()) {
        TM_CHECK(inactive != idxs.begin()) << "No enough blocks";
    }

    // move the partial seq to the back
    auto partial = std::stable_partition(idxs.begin(), inactive, [&](int i) {
        return sequences[i]->cache_len + sequences[i]->input_length == context_lengths[i];
    });
    TM_CHECK_LE(inactive - partial, 1);

    auto swap_in = std::stable_partition(idxs.begin(), partial, [&](int i) {
        return status[i] == Sequence::kActive;  // past status
    });

    // sort existing according to context length
    std::stable_sort(idxs.begin(), swap_in, [&](int i, int j) {  //
        return context_lengths[i] < context_lengths[j];
    });

    // sort swap-ins according to input length
    std::stable_sort(swap_in, partial, [&](int i, int j) {  //
        return sequences[i]->input_length < sequences[j]->input_length;
    });

    // [x] token_ids
    // [x] input_ids
    // [x] h_prompt_length
    // [x] h_context_length
    // [x] h_is_finished
    // [x] h_rope_theta
    // [x] h_block_ptrs
    // [x] h_block_ptrs_offsets
    // [ ] curand_state
    // [ ] seq_len_limit

    vector<shared_ptr<RequestInfo>> info;

    for (int i = 0; i < idxs.size(); ++i) {
        const int j = idxs[i];

        back_->h_context_length[i] = state_->h_context_length[j];
        back_->h_is_finished[i]    = state_->h_is_finished[j];

        info[i] = info_[j];

        h_prompt_length_[i] = info[i]->prompt_length;
        h_rope_theta_[i]    = info[i]->sequence->rope_theta;
    }

    const int size = idxs.size();

    Copy_(back_->h_is_finished, size, batch.is_finished);

    if (async_) {
        std::copy_n(idxs.begin(), size, h_perm_.data());
        Copy_(h_perm_, size, batch.permutation);
    }

    auto input_ids = h_input_ids_.data();
    for (int i = 0; i < size; ++i) {
        const auto ids = info[i]->token_ids + back_->h_context_length[i] - info[i]->sequence->input_length;
        input_ids      = std::copy_n(ids, info[i]->sequence->input_length, input_ids);
    }
    Copy_(h_input_ids_, input_ids - h_input_ids_.data(), batch.input_ids);

    auto token_ids = h_token_ids_.data();
    for (int i = 0; i < size; ++i) {
        token_ids = std::copy_n(info[i]->token_ids, back_->h_context_length[i], token_ids);
    }
    Copy_(h_token_ids_, token_ids - h_token_ids_.data(), batch.token_ids);

    auto h_block_ptrs        = h_block_ptrs_.data();
    h_block_ptrs_offsets_[0] = 0;
    for (int i = 0; i < size; ++i) {
        const auto& s                = *info[i]->sequence;
        h_block_ptrs_offsets_[i + 1] = h_block_ptrs_offsets_[i] + s.blocks.size();
        h_block_ptrs = std::transform(s.blocks.cbegin(), s.blocks.cend(), h_block_ptrs, [&](int block_id) {
            return reinterpret_cast<uintptr_t>(seq_mgr_->GetBlockPtr(block_id));
        });
    }

    Copy_(h_block_ptrs_, h_block_ptrs_offsets_[size], batch.block_ptrs);
    Copy_(h_block_ptrs_offsets_, size + 1, batch.block_ptrs_offsets);

    info_.swap(info);
    state_.swap(back_);
}

void Engine::SetupBatch(SchedBatch& batch) {}

void Engine::SetupSampling(SchedBatch& batch) {}

void Engine::Synchronize(const FeedbackBatch& b, std::vector<Signal>& signals)
{
    Copy_(b.is_finished, b.size, back_->h_is_finished);
    Copy_(b.context_length, b.size, back_->h_context_length);
    Copy_(b.output_ids, b.size, h_output_ids_);

    // perm :: curr -> prev
    for (int i = 0; i < batch_size_; ++i) {
        if (const int j = h_perm_[i]; j < b.size && state_->h_is_finished[i] == 0) {

            state_->h_is_finished[i]    = back_->h_is_finished[j];
            state_->h_context_length[i] = back_->h_context_length[j];

            info_[i]->token_ids[state_->h_context_length[i] - 1] = h_output_ids_[j];
        }
    }

    for (int i = 0; i < batch_size_; ++i) {
        auto& r = info_[i]->request;
    }

    core::Context::stream().Sync();
}

void Engine::InternalThreadEntry()
{
    while (true) {
        shared_ptr<RequestData> rs;

        if (tp_rank_ == 0) {
            rs = std::make_shared<RequestData>();
            gateway_.pop(rs->infer, rs->kill, max_batch_size_ - batch_size_, batch_size_ == 0, rs->abort, dp_rank_);
            // DisableInvalidRequests(rs->infer, rs->kill);
            // FindCanceledIndices(rs->cancel);
        }

        if (rs->abort) {
            TM_LOG_INFO("[Engine] stop requested.");
            break;
        }

        vector<Signal> signals;
        // ProcessKillRequests(rs->kill, signals);
        Accept(rs->infer, signals);
        // ProcessCancelRequests(rs->cancel, signals);
        if (tp_rank_ == 0) {
            gateway_.notify(std::move(signals));
        }
        signals.clear();

        shared_ptr<SchedBatch> sched = std::make_shared<SchedBatch>();

        Schedule(*sched);

        SetupBatch(*sched);

        sched->batch_ready_event.Record(core::Context::stream());

        // Reset host signal before sending to the executor
        std::promise<void> sampling_promise;
        sched->sampling_ready_signal = sampling_promise.get_future();

        outbound_.push(sched);

        // Setup sampling (CPU | HtoD)
        SetupSampling(*sched);

        sched->sampling_ready_event.Record(core::Context::stream());

        // Signal the executor that the event is ready to be waited on.
        sampling_promise.set_value();

        shared_ptr<FeedbackBatch> feedback;
        if (!inbound_.pop(feedback)) {
            break;
        }

        core::Context::stream().Wait(TM_CHECK_NOTNULL(feedback)->ready_event);

        Synchronize(*feedback, signals);

        if (tp_rank_ == 0) {
            gateway_.notify(std::move(signals));
        }
    }
}

}  // namespace turbomind