// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/delta_net.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/logger.h"
#include "src/turbomind/core/registry.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/engine/block.h"
#include "src/turbomind/engine/cache_registry.h"
#include "src/turbomind/engine/request.h"
#include "src/turbomind/models/decoder_layer_weight.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/gated_delta_net_kernels.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {
namespace {

using linear_attn::delta_rule::ContextParallelLevel;

ContextParallelLevel GetCPLevel()
{
    const char* value = std::getenv("TM_GDR_CP_LEVEL");
    if (value == nullptr || std::strcmp(value, "2") == 0) {
        return ContextParallelLevel::kAll;
    }
    if (std::strcmp(value, "1") == 0) {
        return ContextParallelLevel::kExact;
    }
    if (std::strcmp(value, "0") == 0) {
        return ContextParallelLevel::kOff;
    }
    TM_CHECK(false) << "TM_GDR_CP_LEVEL must be 0 (off), 1 (exact), or 2 (all), got " << value;
    return ContextParallelLevel::kAll;
}

}  // namespace

static std::pair<int, int> get_lc_state_size(const DeltaNet& module, int tp)
{
    int num_k_heads    = module.num_k_heads / tp;
    int num_v_heads    = module.num_v_heads / tp;
    int key_head_dim   = module.key_head_dim;
    int value_head_dim = module.value_head_dim;
    int d_conv         = module.d_conv;
    int key_dim        = num_k_heads * key_head_dim;
    int value_dim      = num_v_heads * value_head_dim;
    int conv_dim       = key_dim * 2 + value_dim;
    return {num_v_heads * key_head_dim * value_head_dim, conv_dim * d_conv};
}

DeltaNet::DeltaNet(const core::DeltaNetConfig& cfg):
    hidden_dim(cfg.hidden_dim),
    num_k_heads(cfg.num_k_heads),
    num_v_heads(cfg.num_v_heads),
    key_head_dim(cfg.key_head_dim),
    value_head_dim(cfg.value_head_dim),
    d_conv(cfg.d_conv),
    data_type(cfg.data_type),
    tp_size(cfg.tp_size),
    tp_rank(cfg.tp_rank)
{
}

void DeltaNet::prepare()
{
    Module::prepare();

    EnsureFloatDtype(A_log, data_type);
    EnsureFloatDtype(dt_bias, data_type);
    EnsureFloatDtype(conv1d, data_type);
}

void DeltaNet::register_cache(CacheRegistry& registry)
{
    Module::register_cache(registry);

    // This layer's conv-state bytes: the checkpoint category's accumulation
    // channel (part 0). The old executor accumulated all layers' element
    // counts and registered once; per-layer registration in walk order
    // reproduces the identical layout. Keep the element offset the conv
    // kernel expects.
    const auto [linear_state_size, conv_state_size] = get_lc_state_size(*this, tp_size);
    const int    elem                               = byte_size(data_type, 1);
    const size_t bytes = registry.checkpoint().Register(byte_size(data_type, conv_state_size), 1);
    TM_CHECK_EQ(bytes % elem, 0);
    conv_state_offset_ = static_cast<int>(bytes / elem);
}

void DeltaNet::bind(BindEnv& env)
{
    Module::bind(env);
    if (auto it = env.shared.find("deltanet"); it != env.shared.end()) {
        state_ = std::static_pointer_cast<DeltaNetState>(it->second);
    }
    else {
        // First DeltaNet module of the walk: create the per-rank state.
        // The state spans all layers — collect this rank's DeltaNet
        // modules from the decoder-layer list this module lives in; the
        // collection order (layer order) is what RegisterLayer checks the
        // bind walk against.
        auto* layers = TM_CHECK_NOTNULL(TM_CHECK_NOTNULL(parent())->parent());
        TM_CHECK(std::string_view{layers->type()} == "ModuleList")
            << "DeltaNet parent chain is not a decoder-layer list: " << layers->full_path();
        std::vector<DeltaNet*> modules;
        for (int i = 0; i < static_cast<core::ModuleList*>(layers)->size(); ++i) {
            auto* layer = static_cast<DecoderLayerWeight*>(layers->child(std::to_string(i)));
            if (auto* gdn = layer->linear_attn.get()) {
                modules.push_back(gdn);
            }
        }
        TM_CHECK(!modules.empty());
        state_ = std::make_shared<DeltaNetState>(env.engine, env.registry, env.phases, std::move(modules));
        env.shared.emplace("deltanet", state_);
    }
    layer_index_         = state_->RegisterLayer(*this);
    linear_state_offset_ = state_->layer_state_offset(layer_index_);
}

// ---------------------------------------------------------------------------
// DeltaNetState
// ---------------------------------------------------------------------------

DeltaNetState::DeltaNetState(const EngineParam&     engine,
                             CacheRegistry&         registry,
                             int                    phases,
                             std::vector<DeltaNet*> modules):
    tp_size{engine.attn_tp_size * engine.attn_cp_size},
    recurrent_state_dtype{engine.state_dtype},
    gdr_cp_level{GetCPLevel()}
{
    TM_CHECK(!modules.empty());
    const auto& first = *TM_CHECK_NOTNULL(modules.front());
    layer_num_        = static_cast<int>(modules.size());

    arch_     = getSMVersion() * 10;
    sm_count_ = getSMCount();

    TM_CHECK_EQ(first.num_k_heads % tp_size, 0);
    TM_CHECK_EQ(first.num_v_heads % tp_size, 0);
    TM_CHECK_EQ(first.key_head_dim, 128);
    TM_CHECK_EQ(first.value_head_dim, 128);
    for (const auto* module_ptr : modules) {
        const auto& module = *TM_CHECK_NOTNULL(module_ptr);
        TM_CHECK_EQ(module.num_k_heads, first.num_k_heads);
        TM_CHECK_EQ(module.num_v_heads, first.num_v_heads);
        TM_CHECK_EQ(module.key_head_dim, first.key_head_dim);
        TM_CHECK_EQ(module.value_head_dim, first.value_head_dim);
        TM_CHECK_EQ(module.d_conv, first.d_conv);
        TM_CHECK_EQ(module.data_type, first.data_type);
        TM_CHECK_EQ(module.num_k_heads % tp_size, 0);
        TM_CHECK_EQ(module.num_v_heads % tp_size, 0);
    }

    input_dtype_ = first.data_type;
    num_k_heads_ = first.num_k_heads / tp_size;
    num_v_heads_ = first.num_v_heads / tp_size;
    head_dim_    = first.key_head_dim;
    gate_stride_ = num_v_heads_;
    modules_     = std::move(modules);
    TM_CHECK_EQ(num_v_heads_ % num_k_heads_, 0);
    TM_CHECK(recurrent_state_dtype == kFloat32 || recurrent_state_dtype == input_dtype_)
        << "GDN recurrent state dtype must be float32 or match the input dtype, got state_dtype="
        << recurrent_state_dtype << " input_dtype=" << input_dtype_;

    const auto [linear_state_size, conv_state_size] = get_lc_state_size(first, tp_size);
    const int cell_elements                         = first.key_head_dim * first.value_head_dim;
    TM_CHECK_EQ(linear_state_size, num_v_heads_ * cell_elements);

    int layers_per_block = 1;
    int heads_per_block  = num_v_heads_;
    if (const char* value = std::getenv("TM_GDN_BLOCK_CONFIG")) {
        TM_CHECK_EQ(std::sscanf(value, "%d,%d", &layers_per_block, &heads_per_block), 2)
            << "expected TM_GDN_BLOCK_CONFIG=l,h (e.g. 4,16)";
    }
    TM_CHECK_GT(layers_per_block, 0);
    TM_CHECK_GT(heads_per_block, 0);

    auto ceil_div     = [](int value, int divisor) { return (value + divisor - 1) / divisor; };
    layers_per_block_ = layers_per_block;
    heads_per_block_  = heads_per_block;
    num_head_groups_  = ceil_div(num_v_heads_, heads_per_block_);
    num_layer_groups_ = ceil_div(layer_num_, layers_per_block_);
    num_blocks_       = num_layer_groups_ * num_head_groups_;
    block_bytes_      = byte_size(recurrent_state_dtype, size_t(layers_per_block_) * heads_per_block_ * cell_elements);

    auto require_mode = [&](linear_attn::delta_rule::GdrMode mode) {
        using namespace linear_attn::delta_rule;
        PlanningContext planning{};
        planning.arch              = arch_;
        planning.sm_count          = sm_count_;
        planning.input_dtype       = input_dtype_;
        planning.state_dtype       = recurrent_state_dtype;
        planning.physical_batch    = 1;
        planning.token_slots       = mode == GdrMode::kRecurrent ? 1 : 16;
        planning.hq                = num_k_heads_;
        planning.hv                = num_v_heads_;
        planning.head_dim          = head_dim_;
        planning.gate_stride       = gate_stride_;
        planning.gate_batch_stride = int64_t(planning.token_slots) * gate_stride_;
        planning.beta_stride       = planning.gate_stride;
        planning.beta_batch_stride = planning.gate_batch_stride;
        planning.num_head_groups   = num_head_groups_;
        planning.heads_per_block   = heads_per_block_;
        if (mode == GdrMode::kChunked) {
            planning.q_offsets = {0, 16};
        }
        Operation operation{};
        operation.mode     = mode;
        operation.cp_level = gdr_cp_level;
        Plan plan;
        TM_CHECK(delta_rule_.Plan(operation, planning, &plan));
    };
    require_mode(linear_attn::delta_rule::GdrMode::kRecurrent);
    require_mode(linear_attn::delta_rule::GdrMode::kChunked);

    // The recurrent block grid spans ALL layers (L_b layers share one
    // block-row): one per-rank composite registration. Runs during bind —
    // before category finalization.
    rec_base_ = registry.checkpoint().Register({{block_bytes_, 1, static_cast<size_t>(num_blocks_)}});

    // The per-module conv registrations from the register_cache pass have
    // accumulated into part 0; their total is the region the position-0
    // reset clears.
    conv_total_bytes_ = registry.checkpoint().accumulation_bytes();

    const size_t prefix_bytes = registry.prefix().accumulation_bytes();
    TM_LOG_INFO("[GDN] input_dtype={} state_dtype={} gdr_cp_level={} block config L_b={} H_b={} -> "
                "num_layer_groups={} num_head_groups={} num_blocks={} block_bytes={} "
                "prefix_object_bytes={} ({})",
                input_dtype_,
                recurrent_state_dtype,
                static_cast<int>(gdr_cp_level),
                layers_per_block_,
                heads_per_block_,
                num_layer_groups_,
                num_head_groups_,
                num_blocks_,
                block_bytes_,
                prefix_bytes,
                (prefix_bytes != 0 && block_bytes_ == prefix_bytes) ? "slab-shared" : "separate-slab-class");

    conv_state_ptrs_buf_      = {engine.max_batch_size, kCPUpinned};
    recurrent_state_ptrs_buf_ = {core::ssize_t(num_layer_groups_) * engine.max_batch_size * num_head_groups_,
                                 kCPUpinned};

    batches_.resize(phases);
    for (auto& batch : batches_) {
        batch.conv_state_ptrs      = empty_like(conv_state_ptrs_buf_, kDEVICE);
        batch.recurrent_state_ptrs = empty_like(recurrent_state_ptrs_buf_, kDEVICE);
    }

    work_counter_ = {1, kDEVICE};

    TM_CUDA_CHECK(cudaStreamCreateWithPriority(&aux_stream_, cudaStreamNonBlocking, -1));
    TM_CUDA_CHECK(cudaEventCreateWithFlags(&ev_before_, cudaEventDisableTiming));
    TM_CUDA_CHECK(cudaEventCreateWithFlags(&ev_after_, cudaEventDisableTiming));
}

DeltaNetState::~DeltaNetState()
{
    TM_CUDA_CHECK(cudaStreamDestroy(aux_stream_));
    TM_CUDA_CHECK(cudaEventDestroy(ev_before_));
    TM_CUDA_CHECK(cudaEventDestroy(ev_after_));

    aux_stream_ = {};
    ev_before_ = ev_after_ = {};
}

int DeltaNetState::RegisterLayer(const DeltaNet& module)
{
    TM_CHECK_EQ(module.tp_size, tp_size) << "DeltaNet module tp_size must equal attn_tp_size * attn_cp_size";
    // Idempotent across re-binds: a re-binding module keeps its existing
    // index instead of taking the next one.
    for (int index = 0; index < next_layer_index_; ++index) {
        if (modules_[index] == &module) {
            return index;
        }
    }
    const int index = next_layer_index_++;
    TM_CHECK_LT(index, layer_num_) << "more DeltaNet modules bound than created with the state";
    TM_CHECK_EQ(modules_[index], &module) << "bind order must match layer order";
    return index;
}

int64_t DeltaNetState::layer_state_offset(int layer_index) const
{
    const int cell_elements = head_dim_ * head_dim_;
    return (layer_index % layers_per_block_) * heads_per_block_ * cell_elements;
}

// ---------------------------------------------------------------------------
// BatchOp handling
// ---------------------------------------------------------------------------

void DeltaNet::Run(BatchOp op, int phase, TensorMap& env)
{
    if (op != BatchOp::kSetup) {
        // Engine progress since the last fill: the batch-level slot is stale.
        state_->AdvanceWave();
    }

    if (op == BatchOp::kSetup) {
        Setup(phase, env);
    }
    else if (op == BatchOp::kPrepare) {
        // Batch-level work, done once per pass by layer 0 (the retired
        // executor ran it once per layer class): borrows, position-0 state
        // clears, TMA descriptor preparation.
        if (layer_index_ != 0) {
            return;
        }
        auto& data     = state_->batch(phase);
        data.q_offsets = env.at("q_offsets").buffer().borrow();
        data.k_offsets = env.at("k_offsets").buffer().borrow();
        data.finished  = env.at("finished").buffer().borrow();
        for (const auto& [ptr, bytes] : data.reset_ptrs) {
            Clear(Buffer_<uint8_t>{ptr, static_cast<core::ssize_t>(bytes), kDEVICE});
        }
        data.reset_ptrs.clear();

        if (data.recurrent_plan) {
            core::Tensor state_ptrs{
                data.recurrent_state_ptrs,
                core::Layout{{state_->num_layer_groups_, data.decode_count, state_->num_head_groups_},
                             {data.batch_size * state_->num_head_groups_, state_->num_head_groups_, 1}},
                core::Tensor::PreserveBufferCapacity{}};
            core::Tensor state_descs;
            if (data.recurrent_state_tma_descs) {
                state_descs = core::Tensor{
                    data.recurrent_state_tma_descs,
                    core::Layout{{state_->num_layer_groups_, data.decode_count, state_->num_head_groups_, 128}}};
            }
            state_->delta_rule_.PrepareState(state_ptrs,
                                             state_descs,
                                             state_->num_layer_groups_,
                                             state_->layers_per_block_,
                                             *data.recurrent_plan,
                                             core::Context::stream().handle());
        }
    }
}

void DeltaNet::Setup(int phase, TensorMap& env)
{
    // Versioned fill guard: the first module of this kSetup wave fills the
    // batch-level slot; the rest see the stamp and skip.
    auto& data = state_->batch(phase);
    if (data.wave == state_->wave()) {
        return;
    }

    Buffer_<Sequence*> requests = env.at("requests").buffer();

    data.batch_size = requests.size();
    data.input_lens.resize(data.batch_size);
    data.reset_ptrs.clear();

    std::vector<int32_t> host_offsets(data.batch_size + 1, 0);
    for (int sequence = 0; sequence < data.batch_size; ++sequence) {
        data.input_lens[sequence]  = requests[sequence]->input_len;
        host_offsets[sequence + 1] = host_offsets[sequence] + data.input_lens[sequence];
    }
    const int token_slots = *env.at("token_num").data<int>();
    data.decode_count     = 0;
    while (data.decode_count < data.batch_size && data.input_lens[data.decode_count] == 1) {
        ++data.decode_count;
    }
    data.prefill_count = data.batch_size - data.decode_count;

    data.recurrent_plan.reset();
    data.chunked_plan.reset();
    data.chunked_workspace         = {};
    data.recurrent_state_tma_descs = {};

    auto make_context = [&] {
        linear_attn::delta_rule::PlanningContext planning{};
        planning.arch            = state_->arch_;
        planning.sm_count        = state_->sm_count_;
        planning.input_dtype     = state_->input_dtype_;
        planning.state_dtype     = state_->recurrent_state_dtype;
        planning.hq              = state_->num_k_heads_;
        planning.hv              = state_->num_v_heads_;
        planning.head_dim        = state_->head_dim_;
        planning.gate_stride     = state_->gate_stride_;
        planning.beta_stride     = state_->gate_stride_;
        planning.num_head_groups = state_->num_head_groups_;
        planning.heads_per_block = state_->heads_per_block_;
        return planning;
    };

    if (data.decode_count != 0) {
        auto planning              = make_context();
        planning.physical_batch    = data.decode_count;
        planning.token_slots       = 1;
        planning.gate_batch_stride = state_->gate_stride_;
        planning.beta_batch_stride = state_->gate_stride_;
        linear_attn::delta_rule::Operation operation{};
        operation.mode = linear_attn::delta_rule::GdrMode::kRecurrent;
        linear_attn::delta_rule::Plan plan;
        TM_CHECK(state_->delta_rule_.Plan(operation, planning, &plan));
        data.recurrent_plan.emplace(std::move(plan));
    }

    if (data.prefill_count != 0) {
        auto planning              = make_context();
        planning.physical_batch    = 1;
        planning.token_slots       = token_slots;
        planning.gate_batch_stride = int64_t(token_slots) * state_->gate_stride_;
        planning.beta_batch_stride = planning.gate_batch_stride;
        planning.q_offsets.assign(host_offsets.begin() + data.decode_count, host_offsets.end());
        linear_attn::delta_rule::Operation operation{};
        operation.mode     = linear_attn::delta_rule::GdrMode::kChunked;
        operation.cp_level = state_->gdr_cp_level;
        linear_attn::delta_rule::Plan plan;
        TM_CHECK(state_->delta_rule_.Plan(operation, planning, &plan));
        data.chunked_plan.emplace(std::move(plan));
    }

    if (data.chunked_plan && data.chunked_plan->workspace_bytes != 0) {
        data.chunked_workspace = core::Tensor{
            core::Layout{{static_cast<core::ssize_t>(data.chunked_plan->workspace_bytes)}}, kUint8, kDEVICE};
    }
    if (data.recurrent_plan && data.recurrent_plan->state_tma_desc_bytes_per_layer_group != 0) {
        const core::ssize_t descriptor_bytes =
            core::ssize_t(state_->num_layer_groups_) * data.recurrent_plan->state_tma_desc_bytes_per_layer_group;
        data.recurrent_state_tma_descs = {descriptor_bytes, kDEVICE};
    }

    Buffer_<void*>& conv_state_ptrs_buf      = state_->conv_state_ptrs_buf_;
    Buffer_<void*>& recurrent_state_ptrs_buf = state_->recurrent_state_ptrs_buf_;

    for (int sequence = 0; sequence < data.batch_size; ++sequence) {
        auto& request = *requests[sequence];

        const CacheBlock& block = *TM_CHECK_NOTNULL(request.frontier.get());
        TM_CHECK_NOTNULL(block.allocation.a);

        conv_state_ptrs_buf[sequence] = block.base(0);
        for (int layer_group = 0; layer_group < state_->num_layer_groups_; ++layer_group) {
            for (int head_group = 0; head_group < state_->num_head_groups_; ++head_group) {
                const int part = state_->rec_base_ + layer_group * state_->num_head_groups_ + head_group;
                recurrent_state_ptrs_buf[(layer_group * data.batch_size + sequence) * state_->num_head_groups_
                                         + head_group] = block.base(part);
            }
        }

        if (request.history_len + request.inflight_input_len == 0) {
            data.reset_ptrs.push_back({reinterpret_cast<uint8_t*>(block.base(0)), state_->conv_total_bytes_});
            for (int recurrent_block = 0; recurrent_block < state_->num_blocks_; ++recurrent_block) {
                data.reset_ptrs.push_back({reinterpret_cast<uint8_t*>(block.base(state_->rec_base_ + recurrent_block)),
                                           state_->block_bytes_});
            }
        }
    }

    Copy(conv_state_ptrs_buf, data.batch_size, data.conv_state_ptrs);
    Copy(recurrent_state_ptrs_buf,
         core::ssize_t(state_->num_layer_groups_) * data.batch_size * state_->num_head_groups_,
         data.recurrent_state_ptrs);

    data.wave = state_->wave();
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

void DeltaNet::Forward(int phase, const Tensor& input, Ref<Tensor> output)
{
    TM_FUNCTION_SCOPE();

    const int token_num = input.shape(0);
    if (token_num == 0) {
        return;
    }

    const auto dtype  = input.dtype();
    const auto device = input.device();
    const auto stream = core::Context::stream().handle();
    auto&      data   = state_->batch(phase);

    TM_CHECK(dtype == kHalf || dtype == kBfloat16);

    const int num_k_heads = state_->num_k_heads_;
    const int num_v_heads = state_->num_v_heads_;
    const int head_dim    = state_->head_dim_;

    const int key_dim   = num_k_heads * head_dim;
    const int value_dim = num_v_heads * head_dim;
    const int conv_dim  = key_dim * 2 + value_dim;

    Tensor all_proj = MakePaddedOutput(token_num, *in_proj_all, device);
    TM_SCOPE_CALL(in_proj_all->Forward(input, all_proj));

    const int value_heads       = num_v_heads;
    const int value_gate_offset = conv_dim + value_dim;
    const int decay_gate_offset = value_gate_offset + value_heads;

    const core::ssize_t gate_capacity = core::ssize_t(token_num) * state_->gate_stride_;
    const core::Layout  gate_layout{{1, token_num, num_v_heads}, {gate_capacity, state_->gate_stride_, 1}};
    Tensor beta{core::Buffer{gate_capacity, kFloat32, device}, gate_layout, Tensor::PreserveBufferCapacity{}};
    Tensor g{core::Buffer{gate_capacity, kFloat32, device}, gate_layout, Tensor::PreserveBufferCapacity{}};

    Tensor beta_projection  = all_proj.slice({0, value_gate_offset}, {-1, value_heads});
    Tensor decay_projection = all_proj.slice({0, decay_gate_offset}, {-1, value_heads});
    ComputeBetaG(beta, g, beta_projection, decay_projection, A_log, dt_bias, stream);

    Tensor attn_out{{token_num, value_dim}, dtype, device};
    Tensor conv_out{{token_num, conv_dim}, dtype, device};

    invokeFusedConv1dSiLU(conv_out,
                          all_proj,
                          conv1d,
                          Tensor{},
                          data.conv_state_ptrs,
                          data.q_offsets,
                          data.k_offsets,
                          data.finished,
                          data.batch_size,
                          conv_state_offset_,
                          state_->sm_count_,
                          state_->work_counter_.data(),
                          stream);

    auto make_view = [](const Tensor& storage, core::ssize_t offset, core::Layout layout) {
        return Tensor{storage.buffer().slice(offset, storage.buffer().size() - offset),
                      std::move(layout),
                      Tensor::PreserveBufferCapacity{}};
    };

    const core::Layout qk_layout{{1, token_num, num_k_heads, 128}, {int64_t(token_num) * conv_dim, conv_dim, 128, 1}};
    const core::Layout v_layout{{1, token_num, num_v_heads, 128}, {int64_t(token_num) * conv_dim, conv_dim, 128, 1}};
    const core::Layout out_layout{{1, token_num, num_v_heads, 128},
                                  {int64_t(token_num) * value_dim, value_dim, 128, 1}};
    Tensor             q = make_view(conv_out, 0, qk_layout);
    Tensor             k = make_view(conv_out, key_dim, qk_layout);
    Tensor             v = make_view(conv_out, 2 * key_dim, v_layout);
    Tensor             out{attn_out.buffer(), out_layout};
    invokeL2NormalizeQK(q, k, 1e-6f, stream);

    const int     layer              = layer_index_;
    const int     layer_group        = layer / state_->layers_per_block_;
    const int64_t state_layer_offset = linear_state_offset_;

    auto pointer_view = [&](int first_sequence, int sequence_count) {
        const core::ssize_t offset =
            (core::ssize_t(layer_group) * data.batch_size + first_sequence) * state_->num_head_groups_;
        const core::ssize_t count = core::ssize_t(sequence_count) * state_->num_head_groups_;
        return Tensor{data.recurrent_state_ptrs.slice(offset, count),
                      core::Layout{{sequence_count, state_->num_head_groups_}}};
    };

    const bool mixed = data.recurrent_plan.has_value() && data.chunked_plan.has_value();
    if (mixed) {
        TM_CUDA_CHECK(cudaEventRecord(state_->ev_before_, stream));
        TM_CUDA_CHECK(cudaStreamWaitEvent(state_->aux_stream_, state_->ev_before_));
    }
    const cudaStream_t chunk_stream = mixed ? state_->aux_stream_ : stream;

    if (data.recurrent_plan) {
        const core::Layout recurrent_qk_layout{{data.decode_count, 1, num_k_heads, 128}, {conv_dim, conv_dim, 128, 1}};
        const core::Layout recurrent_v_layout{{data.decode_count, 1, num_v_heads, 128}, {conv_dim, conv_dim, 128, 1}};
        const core::Layout recurrent_out_layout{{data.decode_count, 1, num_v_heads, 128},
                                                {value_dim, value_dim, 128, 1}};
        const core::Layout recurrent_gate_layout{{data.decode_count, 1, num_v_heads},
                                                 {state_->gate_stride_, state_->gate_stride_, 1}};
        Tensor             recurrent_q{q.buffer(), recurrent_qk_layout, Tensor::PreserveBufferCapacity{}};
        Tensor             recurrent_k{k.buffer(), recurrent_qk_layout, Tensor::PreserveBufferCapacity{}};
        Tensor             recurrent_v{v.buffer(), recurrent_v_layout, Tensor::PreserveBufferCapacity{}};
        Tensor             recurrent_out{out.buffer(), recurrent_out_layout, Tensor::PreserveBufferCapacity{}};
        Tensor             recurrent_g{g.buffer(), recurrent_gate_layout, Tensor::PreserveBufferCapacity{}};
        Tensor             recurrent_beta{beta.buffer(), recurrent_gate_layout, Tensor::PreserveBufferCapacity{}};
        Tensor             recurrent_state_ptrs = pointer_view(0, data.decode_count);
        Tensor recurrent_finished{data.finished.slice(0, data.decode_count), core::Layout{{data.decode_count}}};
        Tensor recurrent_state_descs;
        if (data.recurrent_state_tma_descs) {
            const core::ssize_t descriptor_count  = core::ssize_t(data.decode_count) * state_->num_head_groups_ * 128;
            const core::ssize_t descriptor_offset = core::ssize_t(layer_group) * descriptor_count;
            recurrent_state_descs = Tensor{data.recurrent_state_tma_descs.slice(descriptor_offset, descriptor_count),
                                           core::Layout{{data.decode_count, state_->num_head_groups_, 128}}};
        }

        linear_attn::delta_rule::Arguments arguments{};
        arguments.q                  = recurrent_q;
        arguments.k                  = recurrent_k;
        arguments.v                  = recurrent_v;
        arguments.g                  = recurrent_g;
        arguments.beta               = recurrent_beta;
        arguments.state_ptrs         = recurrent_state_ptrs;
        arguments.state_tma_descs    = recurrent_state_descs;
        arguments.finished           = recurrent_finished;
        arguments.out                = &recurrent_out;
        arguments.state_layer_offset = state_layer_offset;
        state_->delta_rule_.Run(arguments, *data.recurrent_plan, stream);
    }

    if (data.chunked_plan) {
        Tensor chunk_state_ptrs = pointer_view(data.decode_count, data.prefill_count);
        Tensor chunk_finished{data.finished.slice(data.decode_count, data.prefill_count),
                              core::Layout{{data.prefill_count}}};
        Tensor chunk_q_offsets{data.q_offsets.slice(data.decode_count, data.prefill_count + 1),
                               core::Layout{{data.prefill_count + 1}}};

        linear_attn::delta_rule::Arguments arguments{};
        arguments.q                  = q;
        arguments.k                  = k;
        arguments.v                  = v;
        arguments.g                  = g;
        arguments.beta               = beta;
        arguments.state_ptrs         = chunk_state_ptrs;
        arguments.q_offsets          = chunk_q_offsets;
        arguments.finished           = chunk_finished;
        arguments.out                = &out;
        arguments.workspace          = data.chunked_workspace ? &data.chunked_workspace : nullptr;
        arguments.state_layer_offset = state_layer_offset;
        state_->delta_rule_.Run(arguments, *data.chunked_plan, chunk_stream);
    }

    if (mixed) {
        TM_CUDA_CHECK(cudaEventRecord(state_->ev_after_, state_->aux_stream_));
        TM_CUDA_CHECK(cudaStreamWaitEvent(stream, state_->ev_after_));
    }

    Tensor gate        = all_proj.slice({0, conv_dim}, {-1, value_dim});
    Tensor hidden_view = attn_out.view({token_num * num_v_heads, head_dim});
    invokeRMSNormGated(hidden_view, gate, norm->weight, norm->norm_eps_, stream);

    TM_SCOPE_CALL(out_proj->Forward(attn_out, output));
}

TM_MODULE_REGISTER_AS("DeltaNetWeight", DeltaNet, core::DeltaNetConfig);

TM_MODULE_METHODS(DeltaNet, DELTA_NET_CHILDREN, DELTA_NET_PARAMS)

}  // namespace turbomind
