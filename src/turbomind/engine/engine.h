
#pragma once

#include <memory>
#include <thread>

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/gateway.h"
#include "src/turbomind/engine/queue.h"

namespace turbomind {

struct Batch {

    Event event;

    int active_size;
    int size;
};

class Engine_ {
public:
    // Engine_()

private:
    using Requests = std::vector<std::shared_ptr<Request>>;
    using Signal   = std::function<void()>;

    void InternalThreadEntry();

    void DisableInvalidRequests(Requests& infer_reqs, Requests& kill_reqs);

    void ProcessKillRequests(const Requests& reqs, std::vector<Signal>& signals);

    void ProcessInferRequests(const Requests& reqs, std::vector<Signal>& signals);

    void FindCanceledIndices(std::vector<int>& indices);

    void ProcessCancelRequests(std::vector<int>& indices, std::vector<Signal>& signals);

    void Finish(std::vector<Signal>& signals);

private:
    Gateway& gateway_;

    const int tp_rank_;
    const int dp_rank_;

    const int device_id_;

    const int max_batch_size_;

    std::array<Batch, 3> batches_;

    cudaStream_t stream_;

    Batch* front_;
    Batch* back_;
    Batch* incoming_;

    Queue<std::shared_ptr<Batch>>& inbound_;
    Queue<std::shared_ptr<Batch>>& outbound_;

    std::thread internal_thread_;
};

}  // namespace turbomind