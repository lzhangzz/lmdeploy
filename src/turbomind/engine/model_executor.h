#include <memory>

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/queue.h"

namespace turbomind {

struct Batch;
class LlamaV2;

class ModelExecutor {
private:
    void InternalThreadEntry();

    void Forward();

private:
    cudaStream_t stream_;

    Queue<std::shared_ptr<Batch>>& inbound_;
    Queue<std::shared_ptr<Batch>>& outbound_;

    std::shared_ptr<LlamaV2> model_;
};

}  // namespace turbomind
