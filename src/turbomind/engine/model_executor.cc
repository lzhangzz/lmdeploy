
#include "src/turbomind/engine/model_executor.h"

#include <memory>

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/exchange.h"
#include "src/turbomind/engine/engine.h"
#include "src/turbomind/models/language_model.h"

#include "src/turbomind/utils/anomaly_handler.h"

namespace turbomind {

using std::shared_ptr;
using std::unique_ptr;

struct ModelExecutor::Impl {

    LanguageModel& model_;

    Queue<unique_ptr<BatchData>>& inbound_;
    Queue<unique_ptr<BatchData>>& outbound_;

    std::thread internal_thread_;

    void InternalThreadEntry()
    {

        Stream    stream  = Stream::create();
        Allocator h_alloc = Allocator(kCPU);
        Allocator d_alloc = Allocator(kDEVICE);

        AnomalyHandler::instance().Init(0, 1000, 0, 1000, stream.handle());

        core::ContextGuard ctx{stream, h_alloc, d_alloc};

        unique_ptr<BatchData> d;

        while (inbound_.pop(d)) {
            TM_CHECK_NOTNULL(d);
            core::Context::stream().Wait(d->ready);
            Run(*d);
            d->done.Record(core::Context::stream());
            outbound_.push(std::move(d));
        }
    }

    void Run(BatchData& d)
    {
        TensorMap env{{"bs0", Buffer{&d.bs0, 1, kCPU}},  //
                      {"bsz", Buffer{&d.bsz, 1, kCPU}},
                      {"permutation", Buffer{d.perm.data(), d.bsz, kCPU}},
                      {"local_token_nums", Buffer{d.local_token_num.data(), (int)d.local_token_num.size(), kCPU}},
                      {"global_token_num", Buffer{&d.global_token_num, 1, kCPU}}};
        model_.Run(BatchOp::kPrepare, d.phase, env);
        model_.Run(BatchOp::kForward, d.phase, env);
        model_.Run(BatchOp::kUnprep, d.phase, env);
    }

    Impl(LanguageModel& model, Queue<unique_ptr<BatchData>>& inbound, Queue<unique_ptr<BatchData>>& outbound):
        model_{model}, inbound_{inbound}, outbound_{outbound}
    {
    }

    ~Impl()
    {
        if (internal_thread_.joinable()) {
            internal_thread_.join();
        }
    }

    void Start()
    {
        internal_thread_ = std::thread(&Impl::InternalThreadEntry, this);
    }
};

ModelExecutor::~ModelExecutor() = default;

ModelExecutor::ModelExecutor()                                    = default;
ModelExecutor::ModelExecutor(ModelExecutor&&) noexcept            = default;
ModelExecutor& ModelExecutor::operator=(ModelExecutor&&) noexcept = default;

ModelExecutor::ModelExecutor(LanguageModel&                model,
                             Queue<unique_ptr<BatchData>>& inbound,
                             Queue<unique_ptr<BatchData>>& outbound):
    impl_{std::make_unique<Impl>(model, inbound, outbound)}
{
}

void ModelExecutor::Start()
{
    return impl_->Start();
}

#if 0
void ModelExecutor::InternalThreadEntry()
{
    shared_ptr<SchedBatch> batch;
    while (inbound_.pop(batch)) {
        core::Context::stream().Wait(batch->forward_ready_event);
        // Update MUTABLE fields
        batch->forward_ready_event.Record(core::Context::stream());
        outbound_.push(batch);
    }
}

void ModelExecutor::Forward(BatchData& batch)
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

    batch.sampling_ready_signal.get();

    core::Context::stream().Wait(batch.sampling_ready_event);

    if (const auto bsz = batch.active_size - batch.partial_size) {

        Tensor logits = model_->postDecodeEmbedding(decoder_output_, symm_local_logits_.buffer());



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
#endif

}  // namespace turbomind