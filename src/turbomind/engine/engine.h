
#pragma once

#include <memory>

#include "src/turbomind/engine/gateway.h"

#include "src/turbomind/models/language_model.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

// sched batch
// ----
// new    request  +
// cancel request  -
// finish request  -
// ----
// 0 1 2 3
// _ 1 2 _ 4 5
// 1 2 _ _

// exec batch
// ----
// 0 1 2 3
// x x 2 3

// fixed-sized data (a --> exec)
// ---
// 1. scatter-copy-back (batch-field is a superset of state-field)
// scatter(state_field, idxs, batch_field)
// copy   (batch_field,       state_field)

// variable-sized data (sched --> exec)
// scatter_offseted(state_field, state_offsets, idxs, batch_field, batch_offsets, batch_end_offsets)
//    copy_offseted(batch_field, batch_offsets, batch_end_offsets, state_field, state_offsets)

// fixed-sized data (sched <-- exec)
// ---
// scatter(state_field, idxs, batch_field)
// copy   (batch_field,       state_field)

// baseline
// state = state.append(gather(incoming))

// Sync
// ---
// batch = Receive()
// batch = Synchronize(batch)
// state = batch
// state = Accept(state, requests)
// batch = Schedule(state)
// Send(batch)

// Async
// ---
// state = {}
// batch = Receive()
// batch = Synchronize(batch)
// state = Update(state, batch)     # update request states (e.g. seq_len, is_finished)
// state = Accept(state, requests)
// batch = Schedule(state)
// state = Clone(batch)             # no mutable sharing
// Send(batch)

/// Decoupled asynchronous model execution

class Engine {
public:
    ~Engine();

    Engine();
    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;

    explicit operator bool() const noexcept
    {
        return static_cast<bool>(impl_);
    }

    Engine(DataType      dtype,
           EngineParam   param,
           LanguageModel model,
           Context&      ctx,
           Gateway&      gateway,
           int           device_id,
           int           dp_rank,
           int           phases);

    void WarmUp();

    void Start();

    ScheduleMetrics GetScheduleMetrics();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#if 0
class Engine {
public:
    // Engine_()

private:
    using Requests = std::vector<std::shared_ptr<Request>>;
    using Signal   = std::function<void()>;

    void InternalThreadEntry();

    void DisableInvalidRequests(Requests& infer_rs, Requests& kill_rs);

    void ProcessKillRequests(const Requests& rs, std::vector<Signal>& signals);

    void FindCanceledIndices(std::vector<int>& indices);

    void ProcessCancelRequests(std::vector<int>& indices, std::vector<Signal>& signals);

    void Accept(const Requests& rs, std::vector<Signal>& signals);  // abcd|EF

    // Allocation of memory / compute resources
    Buffer_<int> Schedule();

    // intiailize RC from `Sequence`
    void Setup(BatchData& d, const Buffer_<int>& perm);

    // Sync vars from batch output to RC
    void Synchronize(const BatchData& d, std::vector<Signal>& signals);

    // 1. (Add)    Se ->  Rc (unlikely)
    // 2. (Setup)  Rc ->  D
    // 3. (Update) D  ->  Rc
    // 4. (Done)   RC ->  Se (finished, canceled)

private:
    Gateway& gateway_;

    const int tp_rank_;
    const int dp_rank_;

    const int device_id_;

    const int max_batch_size_;
    const int max_forward_token_num_;
    const int session_len_;

    const int async_;

    std::shared_ptr<SequenceManager> seq_mgr_;

    Queue<std::shared_ptr<BatchData>>& inbound_;
    Queue<std::shared_ptr<BatchData>>& outbound_;

    std::thread internal_thread_;

    int batch_size_{};
    int active_size_{};

    std::vector<std::shared_ptr<RequestCache>> rc_;  // |rc_| <= max_batch_size

    Buffer_<int> perm_;

    LlamaV2& model_;

    Buffer_<uint64_t> block_ptrs_;
    Buffer_<int>      block_ptrs_offsets_;
};
#endif

}  // namespace turbomind