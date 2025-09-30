
#include "nvtx3/nvToolsExt.h"

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

void Engine::Accept(Batch& batch, const Requests& rs, std::vector<Signal>& signals)
{

    auto get_next_idx = [&, idx = 0]() mutable {
        while (idx < max_batch_size_ && batch.info[idx] && ++idx) {}
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

        TM_CHECK(batch.info[idx] == nullptr);

        auto& seq = *ptr;

        auto info = std::make_shared<RequestInfo>();

        if (step < seq.tokens.size()) {
            seq.tokens.resize(step);
            seq.cache_len = std::min(seq.cache_len, step);
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

        batch.info[idx] = info;
    }

    batch.size = std::max(batch.size, get_next_idx());
}

void Engine::Schedule(Batch& batch)
{
    vector<const Sequence*>  sequences;
    vector<Sequence::Status> status;
    vector<uint64_t>         priorities;
    vector<int>              context_lengths;

    for (int i = 0; i < batch.size; ++i) {
        const auto& info = batch.info[i];
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

    for (int i = 0; i < idxs.size(); ++i) {
        const int j = idxs[i];
    }
}

void Engine::Update(const Batch& batch)
{
    Buffer_<int> h_is_finished;
    Buffer_<int> h_output_ids;
    Buffer_<int> h_seq_len;

    Copy(batch.is_finished, h_is_finished);
    Copy(batch.output_ids, h_output_ids);
    Copy(batch.)

        core::Context::stream()
            .Sync();
}

void Engine::InternalThreadEntry()
{
    shared_ptr<Batch> batch = std::make_shared<Batch>();

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
        Accept(*batch, rs->infer, signals);
        // ProcessCancelRequests(rs->cancel, signals);
        if (tp_rank_ == 0) {
            gateway_.notify(std::move(signals));
        }
        signals.clear();

        batch->event.Record(core::Context::stream());
        outbound_.push(batch);

        if (!inbound_.pop(batch)) {
            break;
        }

        core::Context::stream().Wait(batch->event);

        Synchronize(*batch, signals);
        if (tp_rank_ == 0) {
            gateway_.notify(std::move(signals));
        }

        if (async_) {
            Update(*batch);
        }
        else {
            state_ = batch;
        }
    }
}

}  // namespace turbomind