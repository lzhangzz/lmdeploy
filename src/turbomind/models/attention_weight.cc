// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/attention_weight.h"

#include "src/turbomind/core/registry.h"
#include "src/turbomind/kernels/core/math.h"

namespace turbomind {

AttentionWeight::AttentionWeight(int          hidden_dim,
                                 int          head_dim,
                                 int          head_num,
                                 int          kv_head_num,
                                 MLAParam     mla,
                                 bool         bias,
                                 bool         qk_norm,
                                 int          tp_size,
                                 int          tp_rank,
                                 DataType     data_type,
                                 int          window_size,
                                 bool         sink,
                                 bool         attn_output_gate)
    : hidden_dim_(hidden_dim)
    , head_dim_(head_dim)
    , head_num_(head_num)
    , kv_head_num_(kv_head_num)
    , mla_(mla)
    , bias_(bias)
    , qk_norm_(qk_norm)
    , tp_size_(tp_size)
    , tp_rank_(tp_rank)
    , data_type_(data_type)
    , window_size_(window_size)
    , sink_(sink)
    , attn_output_gate_(attn_output_gate)
{
}

void AttentionWeight::prepare()
{
    for (auto& [name, child] : children_) {
        child->prepare();
    }
}

// Convenience tensor accessors
Tensor* AttentionWeight::q_norm() const
{
    return q_norm_mod_ ? &q_norm_mod_->weight() : nullptr;
}
Tensor* AttentionWeight::k_norm() const
{
    return k_norm_mod_ ? &k_norm_mod_->weight() : nullptr;
}
Tensor* AttentionWeight::q_a_layernorm() const
{
    return q_a_layernorm_mod_ ? &q_a_layernorm_mod_->weight() : nullptr;
}
Tensor* AttentionWeight::kv_a_layernorm() const
{
    return kv_a_layernorm_mod_ ? &kv_a_layernorm_mod_->weight() : nullptr;
}
Tensor* AttentionWeight::sinks() const
{
    return sinks_mod_ ? &sinks_mod_->weight() : nullptr;
}

namespace {
static int64_t cfg_get(const core::ModuleConfig& cfg, const std::string& key, int64_t def = 0)
{
    auto it = cfg.find(key);
    return it != cfg.end() ? std::get<int64_t>(it->second) : def;
}

static bool cfg_bool(const core::ModuleConfig& cfg, const std::string& key)
{
    auto it = cfg.find(key);
    return it != cfg.end() && std::get<int64_t>(it->second);
}

struct AttentionWeightRegistrar {
    AttentionWeightRegistrar() {
        core::ModuleRegistry::instance().register_type(
            "AttentionWeight",
            [](const core::ModuleConfig& cfg) -> std::unique_ptr<core::ModuleBase> {
                MLAParam mla;
                mla.kv_lora_rank = cfg_get(cfg, "kv_lora_rank");
                mla.q_lora_rank  = cfg_get(cfg, "q_lora_rank");
                mla.qk_rope_dim  = cfg_get(cfg, "qk_rope_dim");
                mla.v_head_dim   = cfg_get(cfg, "v_head_dim");
                return std::make_unique<AttentionWeight>(
                    cfg_get(cfg, "hidden_dim"),
                    cfg_get(cfg, "head_dim"),
                    cfg_get(cfg, "head_num"),
                    cfg_get(cfg, "kv_head_num"),
                    mla,
                    cfg_bool(cfg, "has_bias"),
                    cfg_bool(cfg, "qk_norm"),
                    cfg_get(cfg, "tp_size"),
                    cfg_get(cfg, "tp_rank"),
                    static_cast<DataType>(cfg_get(cfg, "data_type")),
                    cfg_get(cfg, "window_size"),
                    cfg_bool(cfg, "attn_sink"),
                    cfg_bool(cfg, "attn_output_gate"));
            });
    }
};
static AttentionWeightRegistrar _attention_weight_reg;
}  // anonymous namespace

}  // namespace turbomind
