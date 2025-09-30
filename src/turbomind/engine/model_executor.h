#include <memory>

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/queue.h"

namespace turbomind {

struct Batch;
class LlamaV2;

class ModelExecutor {
private:
    void InternalThreadEntry();

    void Forward(Batch& batch);

private:
    cudaStream_t stream_;

    const int session_len_;

    Queue<std::shared_ptr<Batch>>& inbound_;
    Queue<std::shared_ptr<Batch>>& outbound_;

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
