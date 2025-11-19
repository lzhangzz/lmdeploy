
#include <algorithm>
#include <memory>
#include <thread>

#include "nvtx3/nvToolsExt.h"

#include "src/turbomind/comm/host_comm.h"
#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/engine/engine.h"
#include "src/turbomind/engine/model_executor.h"
#include "src/turbomind/engine/request.h"

#include "src/turbomind/models/language_model.h"
#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/utils/metrics.h"

#include "dbg.h"

namespace turbomind {

using std::shared_ptr;
using std::unique_ptr;
using std::vector;

struct RequestData {
    vector<shared_ptr<Request>> infer;  // incoming inference request
    vector<shared_ptr<Request>> kill;   // incoming kill request

    vector<int> cancel;  // canceled indices in current batch
    bool        abort;
};

struct Engine::Impl {

    using Requests = vector<shared_ptr<Request>>;
    using Signal   = std::function<void()>;

    Impl(DataType      dtype,
         EngineParam   param,
         LanguageModel model,
         Context&      ctx,
         Gateway&      gateway,
         int           device_id,
         int           dp_rank);

    void CreateSequenceManager();

    void InternalThreadEntry();

    void DisableInvalidRequests(Requests& infer_rs, Requests& kill_rs);

    void ProcessKillRequests(const Requests& rs, vector<Signal>& signals);

    void FindCanceledIndices(vector<int>& indices);

    void ProcessCancelRequests(vector<int>& indices, vector<Signal>& signals);

    void Accept(const Requests& rs, vector<Signal>& signals);

    // Allocation of memory / compute resources
    void Schedule();

    // intiailize RC from `Sequence`
    void Setup(BatchData& d);

    // Sync vars from batch output to RC
    void Update(const BatchData& d, std::vector<Signal>& signals);

    void Run(BatchOp op, int phase, Ref<TensorMap> env)
    {
        model_.Run(op, phase, env);
    }

    void Start()
    {
        internal_thread_ = std::thread(&Impl::InternalThreadEntry, this);
        executor_.Start();
    }

    ~Impl()
    {
        inbound_.close();
        outbound_.close();
        if (internal_thread_.joinable()) {
            internal_thread_.join();
        }
    }

    const DataType    dtype_;
    const EngineParam param_;

    Gateway& gateway_;

    comm::HostComm& tp_group_;
    comm::HostComm& dp_group_;

    const int tp_rank_;
    const int dp_rank_;

    const int device_id_;

    const int async_;

    unique_ptr<SequenceManager> seq_mgr_;

    Queue<unique_ptr<BatchData>> inbound_;
    Queue<unique_ptr<BatchData>> outbound_;

    LanguageModel model_;
    ModelExecutor executor_;

    std::thread internal_thread_;

    int session_len_trunc_;

    ScheduleMetrics metrics_;
    std::mutex      metrics_mutex_;

    struct State {
        vector<unique_ptr<RequestCache>> rc;
        vector<int>                      perm;

        int bs0     = 0;
        int active  = 0;
        int swapout = 0;

        int size() const noexcept
        {
            return rc.size();
        }
    };

    vector<State> states_;

    // staging buffers
    Buffer_<void*> block_ptrs_buf_;
    Buffer_<int>   block_ptrs_offsets_buf_;
};

Engine::Impl::Impl(
    DataType dtype, EngineParam param, LanguageModel model, Context& ctx, Gateway& gateway, int device_id, int dp_rank):
    dtype_{dtype},
    param_{param},
    gateway_{gateway},
    tp_group_{ctx.comm.h_tp_group},
    dp_group_{ctx.comm.h_dp_group},
    tp_rank_{param.attn_tp_rank},
    dp_rank_{dp_rank},
    device_id_{device_id},
    async_{0},
    model_{std::move(model)}
{
    states_.emplace_back();

    executor_ = ModelExecutor{model_, outbound_, inbound_};

    CreateSequenceManager();  // initializes `session_len_trunc_`

    const ssize_t max_batch_block_num =
        param.max_batch_size * cdiv(session_len_trunc_, model_.attn_param().cache_block_seq_len);
    block_ptrs_buf_         = {max_batch_block_num, kCPUpinned};
    block_ptrs_offsets_buf_ = {param.max_batch_size + 1, kCPUpinned};
}

void Engine::Impl::CreateSequenceManager()
{
    const auto cache_block_seq_len = model_.attn_param().cache_block_seq_len;

    const int dbits = byte_size(dtype_, 8);

    const auto& model_param = model_.model_param();

    const auto quant_policy = model_param.quant_policy;
    const int  elem_bits    = quant_policy ? quant_policy : dbits;

    SequenceManager::BlockConfig block_config{
        (int)model_param.head_dim,
        (int)model_param.kv_head_num,
        cache_block_seq_len,
        elem_bits == dbits ? 0 : dbits,
        elem_bits,
    };

    const auto get_free_size = [&] {  //
        size_t free{}, total{};
        check_cuda_error(cudaMemGetInfo(&free, &total));
        return AllReduce(tp_group_, free, comm::RedOp::kMin);
    };

    seq_mgr_ = std::make_unique<SequenceManager>(model_param.layer_num,
                                                 block_config,
                                                 param_.cache_max_block_count,
                                                 param_.cache_chunk_size,
                                                 param_.enable_prefix_caching,
                                                 tp_rank_,
                                                 core::Context::alloc(kDEVICE),
                                                 get_free_size);

    const auto max_cached_tokens = seq_mgr_->max_block_count() * (size_t)cache_block_seq_len;
    session_len_trunc_           = std::min(max_cached_tokens, (size_t)param_.session_len);
    TM_LOG_INFO("max cached tokens: %lld", max_cached_tokens);
    if (session_len_trunc_ != param_.session_len) {
        TM_LOG_WARNING("`session_len` truncated to %d due to limited KV cache memory", session_len_trunc_);
    }
}

void Engine::Impl::Accept(const Requests& rs, vector<Signal>& signals)
{
    auto& s = states_.at(0);

    const int offset = s.rc.size();
    int       index  = offset;

    vector<RequestCache*> incoming;

    // const int session_len = param_.session_len;

    for (const auto& r : rs) {

        if (r->ec) {
            signals.push_back([r] { UpdateState(*r, r->ec, 0); });
            continue;
        }

        const int input_len = r->inputs.at("input_ids").shape(0);

        if (input_len > session_len_trunc_) {
            signals.push_back([r] { UpdateState(*r, Request::kTooLong, 0); });
            continue;
        }

        auto ptr = r->session.start_flag ? seq_mgr_->Create(r->id) : seq_mgr_->Get(r->id);
        if (!ptr) {
            signals.push_back([r] { UpdateState(*r, Request::kInvalid, 0); });
            continue;
        }

        const int step = [&] {
            int s = r->session.step;
            if (s < 0) {
                s = ptr->tokens.size();
            }
            else if (s > ptr->tokens.size()) {
                if (tp_rank_ == 0) {
                    TM_LOG_WARNING("[ProcessInferRequests] Skipping invalid step (%d) setting for ID %lu", s, ptr->id);
                }
                s = ptr->tokens.size();
            }
            return s;
        }();

        if (step + input_len > session_len_trunc_) {
            signals.push_back([r] { UpdateState(*r, Request::kTooLong, 0); });
            continue;
        }

        auto& seq = *ptr;

        // auto c = std::make_unique<RequestCache>(r, seq);
        auto c = new RequestCache{r, seq};

        if (step < seq.tokens.size()) {
            seq.tokens.resize(step);
            seq.cache_len = std::min(seq.cache_len, step);
        }

        const int* input_ids = r->inputs.at("input_ids").data<int>();

        int* token_ids = c->token_ids = r->output_ids.data();

        token_ids = std::copy_n(seq.tokens.data(), seq.tokens.size(), token_ids);
        token_ids = std::copy_n(input_ids, input_len, token_ids);

        c->prompt_len = c->seq_len = token_ids - c->token_ids;  // all known tokens

        int max_seq_len = c->prompt_len + c->gen_cfg.max_new_tokens;
        if (max_seq_len > session_len_trunc_) {
            max_seq_len = session_len_trunc_;
            if (tp_rank_ == 0) {
                const int trunc_output_len = max_seq_len - c->prompt_len;
                // clang-format off
                TM_LOG_WARNING("[ProcessInferRequests] [%ld] total sequence length (%d + %d) exceeds `session_len` (%d), `max_new_tokens` is truncated to %d",
                    (long)seq.id, c->prompt_len, c->gen_cfg.max_new_tokens, session_len_trunc_, trunc_output_len);
                // clang-format on
            }
        }
        c->max_seq_len = max_seq_len;

        incoming.push_back(c);
        s.rc.emplace_back(std::move(c));
    }

    TensorMap env{{"requests", Buffer{incoming.data(), (int)incoming.size(), kCPU}}};
    model_.Run(ExchOp::kAdd, -1, env);

    /// TODO: remove invalid requests (which failed in `Add`)
}

void Engine::Impl::Schedule()
{
    auto& s = states_.at(0);

    vector<const Sequence*>  sequences;
    vector<Sequence::Status> status;
    vector<uint64_t>         priorities;
    vector<int>              context_length;
    vector<RequestCache*>    cache;
    vector<int>              inv;

    for (int i = 0; i < s.size(); ++i) {
        // skip invalid positions
        if (const auto& c = s.rc[i]) {
            cache.push_back(c.get());
            sequences.push_back(&c->sequence);
            status.push_back(c->sequence.status);
            priorities.push_back(c->request->unique_id);
            context_length.push_back(c->seq_len /* plus draft tokens */);
            inv.push_back(i);
            c->input_len = c->history_len = 0;
        }
    }

    auto adjust = [this](const Sequences&, const std::vector<int>&) -> int { return param_.max_forward_token_num; };

    auto outcome = seq_mgr_->Materialize(sequences, context_length, priorities, 1, adjust);

    vector<int> idxs(sequences.size());
    std::iota(idxs.begin(), idxs.end(), 0);

    auto inactive = std::stable_partition(idxs.begin(), idxs.end(), [&](int i) {
        return sequences[i]->status == Sequence::kActive;  // IS active
    });

    TM_CHECK(sequences.empty() || inactive != idxs.begin()) << "No enough blocks";

    // |<----------- active ----------->|<------- inactive ----->|

    // ! past-the-end of swap-outs
    auto swap_out = std::stable_partition(inactive, idxs.end(), [&](int i) {
        return status[i] == Sequence::kActive;  // WAS active
    });

    //                                  |<- swap-out ->|
    // |<----------- active ----------->|<------- inactive ----->|

    // move partially prefilled to the back
    auto partial = std::stable_partition(idxs.begin(), inactive, [&](int i) {
        return sequences[i]->cache_len + sequences[i]->input_length == context_length[i];
    });

    TM_CHECK_LE(inactive - partial, 1);

    auto swap_in = std::stable_partition(idxs.begin(), partial, [&](int i) {
        return status[i] == Sequence::kActive;  // past status
    });

    // |<-- existing -->|<-- swap-in -->|<- swap-out ->|
    // |<----------- active ----------->|<------- inactive ----->|

    for (auto i : subrange{idxs.begin(), swap_in}) {
        TM_CHECK_NE(cache[i]->stage, RequestCache::kInactive);
        cache[i]->stage = RequestCache::kDecoding;
    }
    for (auto i : subrange{swap_in, partial}) {
        TM_CHECK_EQ(cache[i]->stage, RequestCache::kInactive);
        cache[i]->stage = RequestCache::kPrefill;
    }
    for (auto i : subrange{inactive, swap_out}) {
        TM_CHECK_NE(cache[i]->stage, RequestCache::kInactive);
        cache[i]->stage = RequestCache::kInactive;
    }

    vector<unique_ptr<RequestCache>> rc(idxs.size());
    vector<int>                      perm(idxs.size());
    for (int i = 0; i < idxs.size(); ++i) {
        perm[i] = inv[idxs[i]];              // inverse map to original indices
        rc[i]   = std::move(s.rc[perm[i]]);  // warp the request cache
    }
    s.rc.swap(rc);
    s.perm.swap(perm);

    for (auto& c : s.rc) {
        c->input_len   = c->sequence.input_length;
        c->history_len = c->sequence.cache_len;
        dbg(c->history_len, c->input_len, c->stage);
    }

    s.bs0     = std::exchange(s.active, inactive - idxs.begin());
    s.swapout = swap_out - inactive;
}

void Engine::Impl::Setup(BatchData& d)
{
    auto& st = states_.at(0);

    Buffer_<RequestCache*> rc{st.active, kCPU};
    for (int i = 0; i < st.active; ++i) {
        rc[i] = st.rc[i].get();
    }

    block_ptrs_offsets_buf_[0] = 0;
    auto block_ptrs            = block_ptrs_buf_.data();
    for (int i = 0; i < st.active; ++i) {
        const auto& s                  = st.rc[i]->sequence;
        block_ptrs_offsets_buf_[i + 1] = block_ptrs_offsets_buf_[i] + s.blocks.size();
        block_ptrs = std::transform(s.blocks.cbegin(), s.blocks.cend(), block_ptrs, [&](int block_id) {
            return seq_mgr_->GetBlockPtr(block_id);
        });
    }

    d.bs0  = st.bs0;
    d.bsz  = st.active;
    d.perm = st.perm;

    TensorMap env{{"block_ptrs", block_ptrs_buf_},
                  {"block_ptrs_offsets", block_ptrs_offsets_buf_},
                  {"requests", rc},
                  {"bs0", Buffer{&st.bs0, 1, kCPU}},
                  {"bsz", Buffer{&st.active, 1, kCPU}},
                  {"permutation", Buffer{st.perm.data(), st.active, kCPU}}};
    Run(BatchOp::kSetup, d.phase, env);

    /// FIXME: all-gather
    d.local_token_num  = {*env.at("local_token_num").data<int>()};
    d.global_token_num = d.local_token_num[0];
}

void Engine::Impl::Update(const BatchData& b, std::vector<Signal>& signals)
{
    auto& s = states_.at(0);

    Buffer_<bool> finished;
    Buffer_<int>  output_ids;
    Buffer_<int>  sequence_length;
    {
        TensorMap env;
        Run(ExchOp::kFetch, b.phase, env);
        finished        = env.at("finished").buffer();
        output_ids      = env.at("output_ids").buffer();
        sequence_length = env.at("sequence_length").buffer();
    }

    core::Context::stream().Sync();

    Run(BatchOp::kUpdate, -1, TensorMap{});

    dbg(finished.size());
    dbg(s.rc.size());
    dbg(b.bs0, b.bsz);
    dbg(core::to_vector<bool>(finished.slice(0, b.bsz)));

    vector<int> perm(b.bsz);

    std::vector<RequestCache*> cs;
    for (int i = 0; i < b.bsz; ++i) {
        if (auto& c = *s.rc[i]; finished[i] && !c.request->session.end_flag) {
            cs.push_back(s.rc[i].get());
        }
    }

    for (int i = 0; i < b.bsz; ++i) {
        auto& c = *s.rc[i];
        dbg(c.seq_len, sequence_length[i], output_ids[i]);
        c.token_ids[c.seq_len] = output_ids[i];
        c.sequence.cache_len   = sequence_length[i] - 1;
        c.seq_len              = sequence_length[i];
        signals.push_back([this, r = c.request, l = c.seq_len] {  //
            UpdateState(*r, Request::kOk, l);
        });
    }

    if (!cs.empty()) {  // Rc -> Seq
        Run(ExchOp::kDel, -1, TensorMap{{"requests", Buffer{cs.data(), (int)cs.size(), kCPU}}});
    }

    for (int i = 0; i < b.bsz; ++i) {
        if (finished[i]) {
            auto& c = *s.rc[i];
            if (c.request->session.end_flag) {
                seq_mgr_->CacheGeneration(c.sequence);
                TM_CHECK(seq_mgr_->Erase(c.request->id));
            }
            else {
                seq_mgr_->UpdateAndSetUnlock(c.sequence);
            }
            signals.push_back([this, len = c.seq_len, r = std::move(c.request)] {  //
                UpdateState(*r, Request::kFinish, len);
            });
            s.rc[i] = {};
        }
    }
}

void Engine::Impl::InternalThreadEntry()
{
    core::ContextGuard ctx{Stream::create(), Allocator(kCPU), Allocator(kDEVICE)};

    unique_ptr<BatchData> d = std::make_unique<BatchData>();

    while (true) {
        shared_ptr<RequestData> rs;

        auto& st = states_.at(0);

        if (tp_rank_ == 0) {
            rs = std::make_shared<RequestData>();
            gateway_.pop(rs->infer,  //
                         rs->kill,
                         param_.max_batch_size - st.size(),
                         st.size() == 0,
                         rs->abort,
                         dp_rank_);
            // DisableInvalidRequests(rs->infer, rs->kill);
            // FindCanceledIndices(rs->cancel);
        }

        if (rs->abort) {
            TM_LOG_INFO("[Engine] stop requested.");
            break;
        }

        vector<Signal> signals;
        // ProcessKillRequests(rs->kill, signals);  // Erase
        Accept(rs->infer, signals);
        // ProcessCancelRequests(rs->cancel, signals);  // Forced swap out / Sync data
        if (tp_rank_ == 0) {
            gateway_.notify(std::move(signals));
        }
        signals.clear();

        Schedule();  // Forced swap out / Sync data

        Setup(*d);

        while (d->bsz == 0) {};

        d->ready.Record(core::Context::stream());

        outbound_.push(std::move(d));

        if (!inbound_.pop(d)) {
            break;
        }

        TM_CHECK_NOTNULL(d);

        core::Context::stream().Wait(d->done);

        Update(*d, signals);

        // Unlink finished

        if (tp_rank_ == 0) {
            gateway_.notify(std::move(signals));
        }
    }
}

Engine::~Engine() = default;

Engine::Engine()                             = default;
Engine::Engine(Engine&&) noexcept            = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

Engine::Engine(
    DataType dtype, EngineParam param, LanguageModel model, Context& ctx, Gateway& gateway, int device_id, int dp_rank):
    impl_{std::make_unique<Impl>(dtype, param, std::move(model), ctx, gateway, device_id, dp_rank)}
{
}

void Engine::WarmUp() {}

void Engine::Start()
{
    return impl_->Start();
}

ScheduleMetrics Engine::GetScheduleMetrics()
{
    if (!impl_->param_.enable_metrics) {
        return {};
    }
    std::lock_guard lock{impl_->metrics_mutex_};
    auto            metrics = impl_->metrics_;
    return metrics;
}

}  // namespace turbomind