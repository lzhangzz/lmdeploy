
#include "src/turbomind/engine/model_executor.h"
#include "src/turbomind/engine/engine.h"

namespace turbomind {

using std::shared_ptr;

// Match   ::  prompt_ids                         -> token_ids, input_ids
// ---
// Forward ::   token_ids,  input_ids             -> output_ids
// Draft   ::   token_ids,  input_ids, output_ids -> draft_ids
// ---
// Update  ::   token_ids, output_ids             -> token_ids'
// Next    ::  output_ids,  draft_ids             -> input_ids'

void ModelExecutor::InternalThreadEntry()
{
    shared_ptr<Batch> batch;
    while (inbound_.pop(batch)) {
        core::Context::stream().Wait(batch->event);

        // Update MUTABLE fields

        batch->event.Record(core::Context::stream());
        outbound_.push(batch);
    }
}

void ModelExecutor::Forward()
{
    // Forward
    // ---
    // input_ids         MUTABLE  (local_q)
    // hidden_states              (global_q, hidden_dim)
    // decoder_output             (bsz,      hidden_dim)
    // block_ptrs                 (local_k)
    // block_offsets              (bsz)
    // h_q_len           MUTABLE  (bsz)
    // h_k_len           MUTABLE  (bsz)
    // rope_theta                 (bsz)
    // mrope                      (bsz)
    // finished          MUTABLE  (bsz)
    // local_token_nums           (dp_size)
    // sequences                  (bsz)

    // DyanmicDecode
    // ---
    // token_ids_buf     MUTABLE  (seq_len, bsz)
    // finished_buf      MUTABLE  (bsz)
    // sequence_lengths  MUTABLE  (bsz)
    // seq_limit_len     MUTABLE  (bsz)
    // init_context_len  MUTABLE  (bsz)
    // h_k_len           MUTABLE  (bsz)
    // h_p_len                    (bsz)
    // output_logprobs            (bsz)
    // sampled_indices            (bsz)
    // sampled_nums               (bsz)
}

}  // namespace turbomind