
#include "src/turbomind/engine/engine.h"
#include "src/turbomind/engine/model_executor.h"
#include "src/turbomind/engine/request.h"
#include <cuda_runtime_api.h>

namespace turbomind {

using std::shared_ptr;
using std::vector;

namespace {

struct RequestData {
    vector<shared_ptr<Request>> infer;  // incoming inference request
    vector<shared_ptr<Request>> kill;   // incoming kill request

    vector<int> cancel;  // canceled indices in current batch
    bool        abort;
};

}  // namespace

void Engine_::InternalThreadEntry()
{
    shared_ptr<Batch> batch;
    while (true) {
        shared_ptr<RequestData> rs;

        if (tp_rank_ == 0) {
            rs = std::make_shared<RequestData>();
            {
                const int free_size = max_batch_size_ - front_->size;
                gateway_.pop(rs->infer, rs->kill, free_size, free_size == max_batch_size_, rs->abort, dp_rank_);
            }
            DisableInvalidRequests(rs->infer, rs->kill);
            FindCanceledIndices(rs->cancel);
        }

        if (rs->abort) {
            TM_LOG_INFO("[Engine] stop requested.");
            break;
        }

        vector<Signal> signals;
        ProcessKillRequests(rs->kill, signals);
        ProcessInferRequests(rs->infer, signals);
        ProcessCancelRequests(rs->cancel, signals);
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
        Finish(signals);

        if (tp_rank_ == 0) {
            gateway_.notify(std::move(signals));
        }
    }
}

}  // namespace turbomind