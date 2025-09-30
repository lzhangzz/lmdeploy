
#pragma once

#include <memory>
#include <thread>

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/gateway.h"
#include "src/turbomind/engine/queue.h"

#include "src/turbomind/models/llama/SequenceManager.h"

namespace turbomind {

class SequenceManager;

struct RequestInfo {
    std::shared_ptr<Request> request;
    const Sequence*          sequence;

    int  input_length;
    int  prompt_length;
    int  context_length;
    int  max_seq_len;
    int* token_ids;  // alias of "output_ids" buffer in request
};

struct Batch {
    Event event;

    int size;

    int active_size;
    int partial_size;

    int prefill_size;
    int decode_size;

    bool init_sampling;

    Buffer_<int> permutation;

    //
    Buffer_<int> input_ids;
    Buffer_<int> input_ids_offsets;

    Tensor       input_embeds;
    Buffer_<int> input_embeds_offsets;

    Buffer_<int> token_ids;
    Buffer_<int> token_ids_offsets;

    Buffer_<uint64_t> block_ptrs;
    Buffer_<int>      block_ptrs_offsets;

    Buffer_<int> h_q_lens;
    Buffer_<int> h_k_lens;

    Buffer_<bool> is_finished;  // is_finished -> input_length = 0

    Buffer_<int> local_token_nums;

    Buffer_<int> output_ids;
    Buffer_<int> output_ids_offsets;

    std::vector<std::shared_ptr<RequestInfo>> info;

    // std::vector<std::shared_ptr<Request>> requests;
    // std::vector<const Sequence*>          sequences;
};

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

class BatchedGatherScatter {
public:
    using Offsets = Buffer_<int>;
    using Indexes = Buffer_<int>;

    void Gather(const Tensor& src, const Indexes& idxs, Tensor dst);
    void Scatter(const Tensor& src, const Indexes& idxs, Tensor dst);

    void Gather(const Tensor& src, const Offsets& src_offsets, const Indexes& idxs, Tensor dst, Offsets dst_offsets);
    void Scatter(const Tensor& src, const Offsets& src_offsets, const Indexes& idxs, Tensor dst, Offsets dst_offsets);

    void Run();

private:
};

void Scatter(const Tensor& src, const Buffer_<int>& idxs, Tensor dst);

void Scatter(const Tensor&       src,
             const Buffer_<int>& src_offsets,
             const Buffer_<int>& idxs,
             Tensor              dst,
             const Buffer_<int>& dst_offsets,
             Buffer_<int>        dst_end_offsets);

void Copy(const Tensor&       src,
          const Buffer_<int>& src_offsets,
          const Buffer_<int>& src_end_offsets,
          Tensor              dst,
          Buffer_<int>        dst_offsets);

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

class Engine {
public:
    // Engine_()

private:
    using Requests = std::vector<std::shared_ptr<Request>>;
    using Signal   = std::function<void()>;

    void InternalThreadEntry();

    void DisableInvalidRequests(Requests& infer_rs, Requests& kill_rs);

    void ProcessKillRequests(const Requests& rs, std::vector<Signal>& signals);

    void Accept(Batch& batch, const Requests& rs, std::vector<Signal>& signals);

    void FindCanceledIndices(std::vector<int>& indices);

    void ProcessCancelRequests(std::vector<int>& indices, std::vector<Signal>& signals);

    void Schedule(Batch& b);

    void Update(const Batch& b);

    void Synchronize(Batch& b, std::vector<Signal>& signals);

    void CopyBatch(const Batch& a, const std::vector<int>& idxs, Batch& b);

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

    std::shared_ptr<Batch> state_;

    int batch_size_{};

    Queue<std::shared_ptr<Batch>>& inbound_;
    Queue<std::shared_ptr<Batch>>& outbound_;

    Buffer_<int> h_input_ids_;
    Buffer_<int> h_input_ids_offset_;

    std::thread internal_thread_;
};

}  // namespace turbomind