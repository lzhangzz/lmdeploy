#include <memory>

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/queue.h"

namespace turbomind {

class LlamaV2;

struct SchedBatch;
struct FeedbackBatch {
    int size;

    Buffer_<int> context_length;

    Buffer_<int> is_finished;

    Buffer_<int> output_ids;
    Buffer_<int> output_ids_offsets;

    Event ready_event;
};

class ModelExecutor {
public:
    ModelExecutor(Queue<std::shared_ptr<SchedBatch>>& inbound, Queue<std::shared_ptr<FeedbackBatch>>& outbound):
        session_len_{}, inbound_{inbound}, outbound_{outbound}
    {
    }

private:
    void InternalThreadEntry();

    void Forward(SchedBatch& batch);

private:
    cudaStream_t stream_;

    const int session_len_;

    Queue<std::shared_ptr<SchedBatch>>&    inbound_;
    Queue<std::shared_ptr<FeedbackBatch>>& outbound_;

    Buffer_<int> input_ids_;
    Buffer_<int> input_ids_offsets_;

    Tensor symm_hidden_states_;
    Tensor symm_local_logits_;

    Tensor decoder_output_;
    Tensor sampling_logits_;

    Tensor curand_state_;

    Buffer_<int> token_ids_sb_;
    Tensor_<int> token_ids_;

    Buffer_<bool> is_finished_;

    std::shared_ptr<LlamaV2> model_;
};

}  // namespace turbomind
