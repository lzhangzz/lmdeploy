
#include "src/turbomind/engine/model_executor.h"
#include "src/turbomind/engine/engine.h"

#include "src/turbomind/models/llama/LlamaV2.h"

namespace turbomind {

using std::shared_ptr;

// Match   ::  prompt_ids                         -> token_ids, input_ids
// ---
// Forward ::   token_ids,  input_ids             -> output_ids
// Draft   ::   token_ids,  input_ids, output_ids -> draft_ids
// ---
// Update  ::   token_ids, output_ids             -> token_ids'
// Next    ::  output_ids,  draft_ids             -> input_ids'


// Match(token_ids):
//   cache_ids, input_ids = Split(token_ids)
//
// Forward(cache_ids, input_ids):
//   output_ids = Model(cache_ids, input_ids)
// 
// Update(token_ids, output_ids):
//   cache_ids' = cache_ids ++ input_ids
//   input_ids'  = output_ids
//
// TODO: add Draft

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

void ModelExecutor::Forward(Batch& batch)
{
    // Forward
    // ---
    // input_ids         MUTABLE  sched --> exec  (local_q)
    // hidden_states     OUT                      (global_q, hidden_dim)
    // decoder_output    OUT                      (bsz,      hidden_dim)
    // block_ptrs                 sched --> exec  (local_k)
    // block_offsets              sched --> exec  (bsz)
    // h_q_len           MUTABLE  sched --> exec  (bsz)
    // h_k_len           MUTABLE  sched --> exec  (bsz)
    // rope_theta                 sched --> exec  (bsz)
    // mrope                      sched --> exec  (bsz, session_len, 3)
    // finished          MUTABLE  sched <-- exec  (bsz)
    // local_token_nums           sched --> exec  (dp_size)
    // sequences                  sched --> exec  (bsz)

    const auto& perm = batch.permutation;

    Buffer_<int> input_ids_end_offsets;
    Scatter(input_ids_, input_ids_offsets_, perm, batch.input_ids, batch.input_ids_offsets, input_ids_end_offsets);
    Copy(batch.input_ids, batch.input_ids_offsets, input_ids_end_offsets, input_ids_, input_ids_offsets_);

    Scatter(is_finished_, perm, batch.is_finished);

    model_->Forward(input_ids_,
                    symm_hidden_states_,
                    decoder_output_,
                    batch.block_ptrs,
                    batch.block_ptrs_offsets,
                    {},
                    batch.h_q_lens,
                    batch.h_k_lens,
                    {},  // skip
                    batch.is_finished,
                    batch.local_token_nums,
                    {},  // skip
                    batch.decode_size,
                    batch.prefill_size,
                    {}  // skip
    );

    if (const auto bsz = batch.active_size - batch.partial_size) {

        Tensor logits = model_->postDecodeEmbedding(decoder_output_, symm_local_logits_.buffer());

        if (batch.init_sampling) {
            // TODO
        }

        auto sampling_logits = sampling_logits_.slice(0, bsz);
        invokeCastFloat2D(logits, sampling_logits, stream_);
        sync_check_cuda_error();

        // DyanmicDecode
        // ---
        // token_ids_buf     MUTABLE                  (2 * session_len, bsz)
        // seq_limit_len     MUTABLE  sched --> exec  (bsz)
        // init_context_len  MUTABLE  sched --> exec  (bsz)
        // h_k_len           MUTABLE  sched --> exec  (bsz)
        // h_p_len                    sched --> exec  (bsz)
        // finished_buf      MUTABLE  sched <-- exec  (bsz)
        // sequence_lengths  MUTABLE  sched <-- exec  (bsz)
        // sampled_logprobs           sched <-- exec  (bsz, max_logprobs)
        // sampled_indices            sched <-- exec  (bsz, max_logprobs)
        // sampled_nums               sched <-- exec  (bsz)
        model_->dynamicDecode(token_ids_sb_,  //
                              is_finished_,
                              {},
                              {},
                              sampling_logits,
                              {},
                              {},
                              {},
                              {},
                              {},
                              {},
                              {},
                              0,
                              0);

        // [s,b] -> [b,s] and skip padding in [context_len, max_context_len)
        invokeGatherOutput(token_ids_.data(),  //
                           token_ids_sb_.data(),
                           {},
                           {},
                           {},
                           session_len_,
                           bsz,
                           stream_);
        sync_check_cuda_error();
    }

    // Other
    // output_ids         MUTABLE (bsz, session_len) * 3
}

}  // namespace turbomind