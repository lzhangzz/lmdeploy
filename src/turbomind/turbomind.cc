// Copyright (c) OpenMMLab. All rights reserved.

#include <future>
#include <random>

#include "src/turbomind/turbomind.h"

#include "src/turbomind/comm/host_comm.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/core.h"

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/engine/engine.h"
#include "src/turbomind/engine/gateway.h"
#include "src/turbomind/engine/model_executor.h"
#include "src/turbomind/engine/model_request.h"

#include "src/turbomind/models/language_model.h"
#include "src/turbomind/models/model_weight.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"

#include "src/turbomind/kernels/gemm/tuner/params.h"

#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/metrics.h"

#include <yaml-cpp/yaml.h>

// #include "dbg.h"

namespace turbomind {

using std::vector;
using std::string;
using std::shared_ptr;
using std::unique_ptr;

static DataType data_type_from_string(std::string str)
{
    if (str == "fp16" || str == "float16") {
        return kFloat16;
    }
    else if (str == "bf16" || str == "bfloat16") {
        return kBfloat16;
    }
    else if (str == "fp32") {
        return kFloat32;
    }
    else if (str == "int8") {
        return kUint8;
    }
    else if (str == "int4") {
        return kUint4;
    }
    else if (str == "fp8") {
        return kFloat8_e4m3;
    }
    else if (str == "e2m1") {
        return kFloat4_e2m1;
    }
    TM_CHECK(0) << "unsupported weight type: " << str;
    return {};
}

struct TurboMind::Impl {
    DataType       data_type_;
    EngineParam    engine_param_;
    size_t         comm_size_;

    vector<EngineParam> engine_params_;

    string communicator_type_;  // communicator backend

    unique_ptr<comm::HostGroupId> group_id_;

    shared_ptr<Gateway> gateway_;

    FFICtxFactory ffi_ctx_factory_;

    vector<int> global_rank_;

    // Weights & engine instances for the ranks
    vector<shared_ptr<ModelWeight>> weights_;
    vector<shared_ptr<Context>>     contexts_;
    vector<Engine>                  engines_;

    string model_dir_;

    vector<int> queue_id_;
    int         n_queues_{0};

    int need_warm_up_{1};
    int phases_{1};

    ~Impl();

    Impl(string model_dir, string config, FFICtxFactory ffi_ctx_factory);

    unique_ptr<ModelRequest> CreateRequest()
    {
        return std::make_unique<ModelRequest>(gateway_.get(),  //
                                              data_type_,
                                              engine_param_.session_len,
                                              weights_[0]->vocab_size_,
                                              weights_[0]->hidden_units_);
    }

    void CreateWeights(int index)
    {
        CudaDeviceGuard dev_guard(engine_param_.devices[index]);

        CreateContext(index);

        weights_[index] = std::make_shared<ModelWeight>(engine_params_.at(index));
    }

    void ProcessWeights(int index)
    {
        CudaDeviceGuard dev_guard(engine_param_.devices[index]);
        FT_CHECK(weights_[index] != nullptr);

        auto ctx_guard = weights_[index]->context();
        weights_[index]->prepare();
        sync_check_cuda_error();
    }

    void CreateEngine(int index);

    void CreateContext(int index);

    void WarmUp(int index);

    void Sleep(int index, int level)
    {
        // Sleep/wakeup is broken — disabled
    }

    void WakeUp(int index, const std::vector<std::string>& tags)
    {
        // Sleep/wakeup is broken — disabled
    }

    void HandleMissingParams()
    {
        if (!engine_param_.max_context_token_num) {
            engine_param_.max_context_token_num = engine_param_.session_len;
            TM_LOG_WARNING("[TM] `max_context_token_num` is not set, default to %d.",
                           (int)engine_param_.max_context_token_num);
        }

        if (engine_param_.max_context_token_num <= engine_param_.max_batch_size) {
            engine_param_.max_context_token_num *= engine_param_.session_len;
            TM_LOG_WARNING("[TM] `max_context_token_num` = %d.", (int)engine_param_.max_context_token_num);
        }
    }
};

TurboMind::Impl::~Impl()
{
    TM_LOG_INFO(__PRETTY_FUNCTION__);
    if (gateway_) {
        gateway_->shutdown();
    }
    for (int i = 0; i < (int)engines_.size(); ++i) {
        /// TODO: make device part of core::Context
        CudaDeviceGuard device(engine_param_.devices[i]);
        {
            core::ContextGuard context{contexts_[i]->core_stream};
            engines_[i]  = {};
            contexts_[i] = {};
        }
        weights_[i] = {};
    }
}

TurboMind::Impl::Impl(string model_dir, string config, FFICtxFactory ffi_ctx_factory):
    data_type_{}, engine_param_{}, ffi_ctx_factory_{ffi_ctx_factory}
{
    TM_CHECK(!config.empty());

    YAML::Node node;
    try {
        node = YAML::Load(config);
    }
    catch (const YAML::Exception& e) {
        TM_CHECK(0) << "Error loading YAML config: " << e.what() << "\nconfig:\n" << config;
    }

    /// TODO: move config parsing to suitable place
    const auto attention = node["attention_config"];
    const auto engine    = node["engine_config"];

    data_type_ = data_type_from_string(engine["dtype"].as<std::string>());
    TM_CHECK(data_type_ == kBfloat16 || data_type_ == kHalf);

    engine_param_.cache_block_seq_len = attention["cache_block_seq_len"].as<int>(0);
    engine_param_.quant_policy        = engine["quant_policy"].as<int>(0);
    engine_param_.tune_layer_num      = engine["tune_layer_num"].as<int>(1);

    engine_param_.max_batch_size = engine["max_batch_size"].as<int>(0);
    auto max_forward_token_num   = engine["max_prefill_token_num"].as<int>(0);
    max_forward_token_num += engine_param_.max_batch_size;

    engine_param_.max_context_token_num = engine["max_context_token_num"].as<int>(0);
    engine_param_.session_len           = engine["session_len"].as<int>(0);

    engine_param_.cache_max_block_count = engine["cache_max_entry_count"].as<float>(0);
    engine_param_.cache_chunk_size      = engine["cache_chunk_size"].as<int>(0);
    engine_param_.enable_prefix_caching = engine["enable_prefix_caching"].as<bool>(false);
    engine_param_.enable_metrics        = engine["enable_metrics"].as<bool>(false);

    engine_param_.num_tokens_per_iter = engine["num_tokens_per_iter"].as<int>(0);
    engine_param_.max_prefill_iters   = engine["max_prefill_iters"].as<int>(1);

    phases_ = engine["async_"].as<int>() ? 2 : 1;

    engine_param_.outer_dp_size = engine["outer_dp_size"].as<int>();

    engine_param_.attn_dp_size = engine["attn_dp_size"].as<int>();
    engine_param_.attn_tp_size = engine["attn_tp_size"].as<int>();
    engine_param_.attn_cp_size = engine["attn_cp_size"].as<int>();

    engine_param_.mlp_tp_size = engine["mlp_tp_size"].as<int>();

    engine_param_.devices = engine["devices"].as<std::vector<int>>();

    // multi-node information
    engine_param_.nnodes    = engine["nnodes"].as<int>();
    engine_param_.node_rank = engine["node_rank"].as<int>();

    {
        auto sp                             = engine_param_.attn_tp_size * engine_param_.attn_cp_size;
        engine_param_.max_forward_token_num = ((size_t)max_forward_token_num + sp - 1) / sp * sp;
    }

    comm_size_ = engine_param_.attn_dp_size * engine_param_.attn_tp_size * engine_param_.attn_cp_size;
    FT_CHECK(engine_param_.mlp_tp_size == comm_size_);

    communicator_type_ = engine["communicator"].as<std::string>();

    HandleMissingParams();

    weights_.resize(engine_param_.devices.size());
    engines_.resize(engine_param_.devices.size());
    contexts_.resize(engine_param_.devices.size());

    // NOTE: This runs on Python main thread
    group_id_ = comm::CreateHostGroupId((engine_param_.nnodes == 1) ? "" : "hybrid");
    group_id_->Initialize();

    const int devices = engine_param_.devices.size();

    for (int i = 0; i < devices; ++i) {
        global_rank_.push_back(engine_param_.node_rank * devices + i);
    }

    queue_id_.resize(devices);
    engine_params_.resize(devices, engine_param_);
}

void TurboMind::Impl::CreateContext(int index)
{
    auto& p = engine_params_[index];

    CudaDeviceGuard dev_guard(p.devices[index]);

    TM_CHECK(contexts_[index] == nullptr);

    auto& ctx = contexts_[index] = std::make_shared<Context>(p.devices[index]);

    // Layout: (outer, dp, tp, cp)

    const int global_rank = global_rank_[index];

    const int outer_rank = global_rank / comm_size_;
    const int inner_rank = global_rank % comm_size_;

    p.outer_dp_rank = outer_rank;

    const int tp_cp_size = p.attn_tp_size * p.attn_cp_size;

    const int tp_color = inner_rank / tp_cp_size;
    const int dp_color = inner_rank % tp_cp_size;
    const int cp_color = inner_rank / p.attn_cp_size;

    auto& c = ctx->comm;

    c.h_global = group_id_->CreateCommunicator(comm_size_, global_rank, p.node_rank);

    c.h_comm = c.h_global->Split(outer_rank, 0);

    c.h_tp_group = c.h_comm->Split(tp_color, 0);
    c.h_dp_group = c.h_comm->Split(dp_color, 0);

    if (comm_size_ > 1) {
        c.d_comm = CreateDeviceCommunicator(communicator_type_, comm_size_, inner_rank, c.h_comm);

        c.d_tp_group = 0;
        c.d_cp_group = 0;

        if (p.attn_dp_size > 1) {  // has attn_dp
            c.d_tp_group   = c.d_comm->Split(tp_color, 0, 0);
            p.attn_dp_rank = c.h_dp_group->rank();
        }

        if (p.attn_cp_size > 1) {  // has attn_cp
            c.d_cp_group   = c.d_comm->Split(cp_color, 0, 0);
            p.attn_cp_rank = c.d_comm->rank(c.d_cp_group);
        }

        p.attn_tp_rank = c.d_comm->rank(c.d_tp_group) / p.attn_cp_size;
        p.mlp_tp_rank  = c.d_comm->rank(0);
    }

    if (c.h_tp_group->rank() == 0) {
        queue_id_[index] = 1;
    }

    c.h_global->Sync();

    if (index == 0) {
        n_queues_ = 0;
        for (size_t i = 0; i < queue_id_.size(); ++i) {
            queue_id_[i] = queue_id_[i] ? n_queues_++ : -1;
        }
        gateway_ = std::make_shared<Gateway>(n_queues_, ffi_ctx_factory_);
    }

    c.h_global->Sync();
}

void TurboMind::Impl::CreateEngine(int index)
{
    CudaDeviceGuard dev_guard(engine_param_.devices[index]);

    auto& ctx = *TM_CHECK_NOTNULL(contexts_[index]);

    core::ContextGuard guard{ctx.core_stream, ctx.allocator, Allocator{kCPUpinned}};

    const auto& param = engine_params_.at(index);

    ctx.comm.h_comm->Sync();

    // create model
    LanguageModel model{param,
                        ctx,
                        *weights_[index],
                        phases_};

    // create engine
    engines_[index] = Engine{param,
                             std::move(model),
                             *weights_[index],
                             ctx,
                             *gateway_,
                             engine_param_.devices[index],
                             queue_id_[index],
                             phases_};

    core::Context::stream().Sync();

    ctx.comm.h_comm->Sync();

    engines_[index].Start();

    if (need_warm_up_) {
        WarmUp(index);
    }
}

template<class Iter>
static std::string Join(Iter first, Iter last, const std::string& delim)
{
    if (first == last) {
        return {};
    }
    std::ostringstream oss;
    oss << *first++;
    while (first != last) {
        oss << delim << *first++;
    }
    return oss.str();
}

void TurboMind::Impl::WarmUp(int index)
{
    auto& ctx = *TM_CHECK_NOTNULL(contexts_[index]);

    auto& global = ctx.comm.h_global;
    auto& linear = *ctx.linear;

    if (auto str = std::getenv("TM_GEMM_IMPORT")) {
        std::ifstream ifs(str);
        const int     n_imported = linear.Import(ifs);
        if (index == 0) {
            TM_LOG_INFO("[GEMM] %d records imported", n_imported);
        }
        return;
    }

    global->Sync();

    *ctx.is_warm_up = 1;
    linear.set_measure(true);

    if (index == 0) {
        gateway_->set_threshold(engine_param_.attn_dp_size);
    }

    global->Sync();

    if (ctx.comm.h_tp_group->rank() == 0) {

        std::vector<int> bss = linear.GetTuningSeq();
        if (bss.empty()) {
            bss = gemm::GenerateTuningSequence(gemm::GetDefaultTuningGenerators());
        }

        const int max_fwd_token_num = engine_param_.max_forward_token_num;

        // remove bs that is too large
        bss.erase(std::remove_if(bss.begin(), bss.end(), [&](auto x) { return x > max_fwd_token_num; }), bss.end());

        if (bss.empty() || bss.back() < max_fwd_token_num) {
            bss.push_back(max_fwd_token_num);
        }

        auto str = Join(bss.begin(), bss.end(), ", ");
        TM_LOG_INFO("[Engine] Warm-up lengths: %s", str.c_str());

        if (!bss.empty()) {
            const auto                         max_bs = *std::max_element(bss.begin(), bss.end());
            Buffer_<int>                       input_ids(max_bs, kCPU);
            std::mt19937                       g{};
            std::uniform_int_distribution<int> d{0, (int)weights_[index]->vocab_size_ - 1};
            for (auto& x : input_ids) {
                x = d(g);
            }

            auto tick = std::chrono::steady_clock::now();

            for (auto token_num : bss) {

                TM_LOG_INFO("[WarmUp] %d", token_num);

                auto r = CreateRequest();

                TensorMap inputs{{"input_ids", input_ids.slice(0, token_num)}};

                ModelRequest::InputParam param{};
                param.session.start_flag     = true;
                param.session.end_flag       = true;
                param.gen_cfg.max_new_tokens = 1;
                param.tensors                = std::make_shared<TensorMap>(inputs);

                struct Channel {
                    int                flag = 1;
                    std::promise<void> promise;
                };
                auto c = std::make_shared<Channel>();

                ModelRequest::OutputParam out = r->Forward(std::move(param), [c] {
                    /// NOTE: It's risky to set `out.state` here, `out` may not be initialized at this point
                    if (std::exchange(c->flag, 0)) {
                        c->promise.set_value();
                    }
                });

                c->promise.get_future().get();

                int status = -1;
                if (auto state = out.state->exchange(nullptr)) {
                    status = state->status;
                }

                if (status != Request::kFinish) {
                    TM_LOG_ERROR("[Engine] Warm-up for %d tokens failed with status %d", (int)token_num, (int)status);
                }
            }

            auto tock = std::chrono::steady_clock::now();

            TM_LOG_INFO("[WarmUp] Warm-up finished in %.2f seconds.",
                        std::chrono::duration<float, std::ratio<1, 1>>(tock - tick).count());
        }
    }

    global->Sync();

    linear.set_measure(false);
    *ctx.is_warm_up = 0;

    if (index == 0) {
        if (auto path = std::getenv("TM_GEMM_EXPORT")) {
            std::ofstream ofs(path);
            const auto    n_records = linear.Export(ofs);
            TM_LOG_INFO("[GEMM] %d records exported.", n_records);
        }

        gateway_->set_threshold(1);
        need_warm_up_ = 0;
    }

    global->Sync();
}

TurboMind::~TurboMind() = default;

TurboMind::TurboMind(string model_dir, string config, FFICtxFactory ffi_ctx_factory):
    impl_{std::make_unique<Impl>(model_dir, config, ffi_ctx_factory)}
{
}

void TurboMind::CreateWeights(int index)
{
    return impl_->CreateWeights(index);
}

core::Module* TurboMind::root(int index)
{
    return impl_->weights_[index].get();
}

std::pair<core::Stream, core::Allocator> TurboMind::weight_context(int index)
{
    auto& mw = impl_->weights_.at(index);
    TM_CHECK(mw != nullptr);
    return {mw->stream(), mw->allocator()};
}

void TurboMind::ProcessWeights(int index)
{
    return impl_->ProcessWeights(index);
}

void TurboMind::CreateEngine(int index)
{
    return impl_->CreateEngine(index);
}

void TurboMind::Sleep(int index, int level)
{
    return impl_->Sleep(index, level);
}

void TurboMind::WakeUp(int index, const vector<string>& tags)
{
    return impl_->WakeUp(index, tags);
}

shared_ptr<ScheduleMetrics> TurboMind::GetScheduleMetrics(int index)
{
    return impl_->engines_[index].GetScheduleMetrics();
}

unique_ptr<ModelRequest> TurboMind::CreateRequest()
{
    return impl_->CreateRequest();
}

bool TurboMind::is_dummy_node() const noexcept
{
    return impl_->n_queues_ == 0;
}

int TurboMind::GetAttnTpRank(int index)
{
    return impl_->engine_params_.at(index).attn_tp_rank;
}

int TurboMind::GetMlpTpRank(int index)
{
    return impl_->engine_params_.at(index).mlp_tp_rank;
}

}  // namespace turbomind
