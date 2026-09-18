// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/moe.h"

#include <memory>

#include <cuda_runtime.h>

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/registry.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/kernels/copy/copy.h"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/norm/rms_norm.h"

#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_utils.h"

#include "src/turbomind/models/moe_default.h"
#include "src/turbomind/models/moe_deepep.h"

#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

Moe::Moe(const core::MoeConfig& cfg)
{
    experts_per_token = cfg.experts_per_token;
    norm_topk_prob    = cfg.norm_topk_prob;
    routed_scale      = static_cast<float>(cfg.routed_scale);
    topk_group        = cfg.topk_group;
    topk_method       = cfg.topk_method;
    n_group           = cfg.n_group;
    scoring_func      = cfg.scoring_func;
    router_n_groups   = cfg.router_n_groups;
    data_type_        = cfg.data_type;
    act_type_         = static_cast<ActivationType>(cfg.act_type);
    fuse_silu_act_    = cfg.fuse_silu;
    expert_num        = cfg.expert_num;
    ep_size           = cfg.ep_size;
    ep_rank           = cfg.ep_rank;
}

// Shallow alias of an already-prepared linear: shares its tensors and
// copies its metadata. The alias is never prepared itself (Moe::prepare
// aliases only after its own Module::prepare() pass).
static std::unique_ptr<Linear> AliasLinear(const Linear& src)
{
    auto dst    = std::make_unique<Linear>();
    dst->weight = src.weight;
    dst->bias   = src.bias;
    dst->scales = src.scales;
    dst->zeros  = src.zeros;
    src.copy_metadata_to(*dst);
    return dst;
}

static std::unique_ptr<Ffn> AliasFfn(const Ffn& src)
{
    auto dst           = std::make_unique<Ffn>();
    dst->hidden_dim    = src.hidden_dim;
    dst->inter_size    = src.inter_size;
    dst->act_type      = src.act_type;
    dst->is_fused_silu = src.is_fused_silu;
    dst->tp_size       = src.tp_size;
    dst->tp_rank       = src.tp_rank;
    if (src.w1w3) {
        dst->add_child("w1w3", AliasLinear(*src.w1w3));
    }
    else {
        dst->add_child("w1", AliasLinear(*TM_CHECK_NOTNULL(src.w1.get())));
        dst->add_child("w3", AliasLinear(*TM_CHECK_NOTNULL(src.w3.get())));
    }
    dst->add_child("w2", AliasLinear(*TM_CHECK_NOTNULL(src.w2.get())));
    return dst;
}

void Moe::AliasRouted(const Moe& meta_pack)
{
    TM_CHECK(meta_pack.gate && meta_pack.experts);
    TM_CHECK(!gate && !experts);
    TM_CHECK_NOTNULL(meta_pack.expert(0));

    add_child("gate", AliasLinear(*meta_pack.gate));

    // Experts are EP-sharded; iterate the local slice and keep global
    // expert indices as child names (see Moe::expert()).
    TM_CHECK_EQ(ep_size, meta_pack.ep_size);
    TM_CHECK_EQ(ep_rank, meta_pack.ep_rank);
    const int local_num = meta_pack.num_local_experts();
    const int offset    = meta_pack.local_expert_offset();
    create_child("experts", core::ModuleListConfig{});
    for (int e = 0; e < local_num; ++e) {
        experts->add_child(std::to_string(offset + e), AliasFfn(*TM_CHECK_NOTNULL(meta_pack.expert(e))));
    }
    // never touch shared_gate
}

void LinkLinearExperts(const std::vector<Linear*>& experts, Linear& destination)
{
    const auto& e0 = *experts.front();

    e0.copy_metadata_to(destination);

    const int n            = experts.size();
    destination.k_desc.num = destination.q_desc.num = n;

    if (e0.bias) {
        destination.bias = Tensor{{n, e0.output_dim}, e0.bias.dtype(), kDEVICE};
    }

    std::vector<std::pair<void*, int>> weights;
    std::vector<std::pair<void*, int>> scales;
    std::vector<std::pair<void*, int>> global_scales;
    weights.reserve(n);
    scales.reserve(n);
    global_scales.reserve(n);

    for (int i = 0; i < n; ++i) {
        auto& expert = *experts[i];
        weights.emplace_back(expert.weight.raw_data(), expert.k_desc.ld);
        if (expert.scales) {
            scales.emplace_back(expert.scales.raw_data(), expert.q_desc.ld);
        }
        if (expert.global_scale) {
            global_scales.emplace_back(expert.global_scale.raw_data(), 1);
        }
        if (expert.bias) {
            Copy(expert.bias, destination.bias.slice(i, 1).squeeze(0));
        }
    }

    auto stream           = core::Context::stream();
    auto make_strided_ptr = [stream](const auto& ptrs) {
        return std::shared_ptr<void>{gemm::MakeStridedPtrs(ptrs, stream.handle()),
                                     [stream](void* p) { TM_CUDA_CHECK(cudaFreeAsync(p, stream.handle())); }};
    };
    destination.weight = Tensor{make_strided_ptr(weights), {n}, destination.weight_format.dtype, kDEVICE};
    if (e0.scales) {
        destination.scales = Tensor{make_strided_ptr(scales), {n}, e0.scales.dtype(), kDEVICE};
    }
    if (e0.global_scale) {
        destination.global_scale = Tensor{make_strided_ptr(global_scales), {n}, e0.global_scale.dtype(), kDEVICE};
    }
    destination.k_desc.ld = destination.q_desc.ld = 0;
    destination.k_desc.offsets = destination.q_desc.offsets = nullptr;
}

Ffn* Moe::expert(int i) const
{
    if (!experts) {
        return nullptr;
    }
    return static_cast<Ffn*>(experts->child(std::to_string(local_expert_offset() + i)));
}

void Moe::prepare()
{
    Module::prepare();

    if (meta_pack_) {
        // Meta-MoE layer weight: alias the routed gate/experts from the
        // shared pack. ModelWeight::prepare() prepares meta_experts before
        // the layers, so the pack is already prepared at this point.
        AliasRouted(*meta_pack_);
    }

    if (experts) {
        link_block();
    }
}

void Moe::link_block()
{
    const int local_expert_num = num_local_experts();

    // Create batched block view for fused MoE path
    auto e0 = TM_CHECK_NOTNULL(expert(0));  // exemplar expert

    core::FfnConfig block_cfg;
    block_cfg.hidden_dim = e0->hidden_dim;
    block_cfg.inter_size = e0->inter_size;
    // tp_size=1: expert weights are already TP-sharded — the block
    // just batches them and must not divide inter_size a second time.
    block_cfg.tp_size   = 1;
    block_cfg.tp_rank   = 0;
    block_cfg.data_type = data_type_;
    block_cfg.act_type  = static_cast<int>(act_type_);
    block_cfg.fuse_silu = fuse_silu_act_;
    block_              = std::make_unique<Ffn>(block_cfg);

    // Link each linear in the block to the corresponding expert linears
    auto get_expert_w1w3 = [this](int i) -> Linear* {
        auto* exp = expert(i);
        return exp ? exp->w1w3.get() : nullptr;
    };
    auto get_expert_w1 = [this](int i) -> Linear* {
        auto* exp = expert(i);
        return exp ? exp->w1.get() : nullptr;
    };
    auto get_expert_w3 = [this](int i) -> Linear* {
        auto* exp = expert(i);
        return exp ? exp->w3.get() : nullptr;
    };
    auto get_expert_w2 = [this](int i) -> Linear* {
        auto* exp = expert(i);
        return exp ? exp->w2.get() : nullptr;
    };

    auto link = [local_expert_num](const auto& get_expert, Linear& destination) {
        std::vector<Linear*> linears;
        linears.reserve(local_expert_num);
        for (int i = 0; i < local_expert_num; ++i) {
            linears.push_back(get_expert(i));
        }
        LinkLinearExperts(linears, destination);
    };

    if (get_expert_w1w3(0)) {
        // Fused w1w3 path: experts have a single fused gate+up projection
        block_->add_child("w1w3", std::make_unique<Linear>());
        link(get_expert_w1w3, *block_->w1w3);
    }
    else {
        // Separate w1/w3 path: link individually
        block_->add_child("w1", std::make_unique<Linear>());
        block_->add_child("w3", std::make_unique<Linear>());
        if (get_expert_w1(0)) {
            link(get_expert_w1, *block_->w1);
        }
        if (get_expert_w3(0)) {
            link(get_expert_w3, *block_->w3);
        }
    }

    block_->add_child("w2", std::make_unique<Linear>());
    if (get_expert_w2(0)) {
        link(get_expert_w2, *block_->w2);
    }

    // Propagate the actual fused-silu state from the first expert to
    // the block.  Each expert's prepare() has already run above, so
    // is_fused_silu() now reflects whether the GEMM epilogue applies
    // SiLU.
    block_->is_fused_silu = e0->is_fused_silu;
}

Tensor_<float> Moe::Gate(const Tensor& input, Linear& gate)
{
    TM_FUNCTION_SCOPE();

    TM_CHECK_EQ(input.shape(1), gate.input_dim);
    Tensor_<float> logits{{input.shape(0), gate.output_dim}, kDEVICE};
    TM_SCOPE_CALL(gate.Forward(input, logits));
    TM_SCOPE_CALL(ApplyBias(logits, gate.bias, core::Context::stream().handle()));
    return logits;
}

// ---------------------------------------------------------------------------
// Registry factory — Implementation selection at construction (ADR 0001)
// ---------------------------------------------------------------------------

namespace {

std::unique_ptr<core::Module> CreateMoe(const core::ModuleConfig& config)
{
    const auto& cfg = static_cast<const core::MoeConfig&>(config);

    const EngineParam* engine = EngineParamGuard::current();
    TM_CHECK(engine != nullptr)
        << "MoE module construction requires ambient engine params: no engine-backed weight-context guard is active";

    if (engine->moe_a2a_backend == "default") {
        return std::make_unique<MoeDefaultImpl>(cfg);
    }
#ifdef USE_NCCL
    if (engine->moe_a2a_backend == "deepep") {
        return std::make_unique<MoeDeepEPImpl>(cfg);
    }
#endif
    TM_LOG_FATAL("Unsupported MoE transport config: moe_a2a_backend = {}", engine->moe_a2a_backend);
}

const bool _tm_moe_factory_registered = [] {
    core::ModuleRegistry::instance().register_type("MoeWeight", CreateMoe);
    return true;
}();

}  // namespace

TM_MODULE_METHODS(Moe, MOE_CHILDREN, MOE_PARAMS);

}  // namespace turbomind
