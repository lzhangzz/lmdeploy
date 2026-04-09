// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "src/turbomind/core/data_type.h"

namespace turbomind::core {

// ======================================================================
// Self-registering config field infrastructure
// ======================================================================

enum class FieldType : uint8_t { Int, Bool, Double, String, DataType };

struct FieldDescriptor {
    const char* name;
    FieldType   type_tag;
};

template<typename T> struct FieldTypeTag;
template<> struct FieldTypeTag<int>         { static constexpr FieldType v = FieldType::Int; };
template<> struct FieldTypeTag<bool>        { static constexpr FieldType v = FieldType::Bool; };
template<> struct FieldTypeTag<double>      { static constexpr FieldType v = FieldType::Double; };
template<> struct FieldTypeTag<std::string> { static constexpr FieldType v = FieldType::String; };
template<> struct FieldTypeTag<DataType>    { static constexpr FieldType v = FieldType::DataType; };

// Forward declaration
struct ModuleConfig;

template<typename T>
class ConfigField: public FieldDescriptor {
    T value_;

public:
    template<typename... Args>
    ConfigField(ModuleConfig& parent, const char* name, Args&&... args);

    // Implicit conversions — existing code works unchanged
    operator T&()             { return value_; }
    operator const T&() const { return value_; }
    ConfigField& operator=(const T& v) { value_ = v; return *this; }
};

#define TM_CONFIG_FIELD(Type, name, ...) \
    ConfigField<Type> name{*this, #name, ##__VA_ARGS__}

// ======================================================================
// ModuleConfig — base with field registry
// ======================================================================

struct ModuleConfig {
    std::string module_type;

    ModuleConfig() = default;

    // Copy: module_type is copied, fields_ starts empty.
    // ConfigField members self-register via NSDMI, then copy_values_from() copies values.
    ModuleConfig(const ModuleConfig& other)
        : module_type(other.module_type) {}

    ModuleConfig(ModuleConfig&&) = delete;
    ModuleConfig& operator=(ModuleConfig&&) = delete;

    // Copy assignment: fields are already registered, just copy values.
    ModuleConfig& operator=(const ModuleConfig& other) {
        if (this != &other) {
            module_type = other.module_type;
            copy_values_from(other);
        }
        return *this;
    }

    void register_field(FieldDescriptor* f) {
        fields_.push_back(f);
    }

    // Copy values from source by field index (fields registered in declaration order).
    void copy_values_from(const ModuleConfig& src) {
        for (size_t i = 0; i < fields_.size(); ++i) {
            auto* dst   = fields_[i];
            auto* src_f = src.fields_[i];
            switch (dst->type_tag) {
            case FieldType::Int:      static_cast<ConfigField<int>&>(*dst)         = static_cast<const ConfigField<int>&>(*src_f);         break;
            case FieldType::Bool:     static_cast<ConfigField<bool>&>(*dst)        = static_cast<const ConfigField<bool>&>(*src_f);        break;
            case FieldType::Double:   static_cast<ConfigField<double>&>(*dst)      = static_cast<const ConfigField<double>&>(*src_f);      break;
            case FieldType::String:   static_cast<ConfigField<std::string>&>(*dst) = static_cast<const ConfigField<std::string>&>(*src_f); break;
            case FieldType::DataType: static_cast<ConfigField<DataType>&>(*dst)    = static_cast<const ConfigField<DataType>&>(*src_f);    break;
            }
        }
    }

    // Find a field by name.
    FieldDescriptor* field(const char* name) const {
        for (auto* f : fields_)
            if (std::strcmp(f->name, name) == 0) return f;
        return nullptr;
    }

    const std::vector<FieldDescriptor*>& fields() const { return fields_; }

protected:
    ModuleConfig(std::string type): module_type(std::move(type)) {}

private:
    std::vector<FieldDescriptor*> fields_;
};

// Out-of-line constructor definition — ModuleConfig is now complete
template<typename T>
template<typename... Args>
ConfigField<T>::ConfigField(ModuleConfig& parent, const char* name, Args&&... args)
    : FieldDescriptor{name, FieldTypeTag<T>::v}
    , value_(std::forward<Args>(args)...)
{
    parent.register_field(this);
}

// ======================================================================
// Config structs — fields use TM_CONFIG_FIELD for self-registration
// ======================================================================

struct LinearConfig: ModuleConfig {
    LinearConfig(): ModuleConfig{"LinearWeight"} {}
    LinearConfig(const LinearConfig& other): ModuleConfig(other) { copy_values_from(other); }

    TM_CONFIG_FIELD(int,      input_dim);
    TM_CONFIG_FIELD(int,      output_dim);
    TM_CONFIG_FIELD(DataType, data_type);
    TM_CONFIG_FIELD(bool,     has_bias);
};

struct AttentionConfig: ModuleConfig {
    AttentionConfig(): ModuleConfig{"AttentionWeight"} {}
    AttentionConfig(const AttentionConfig& other): ModuleConfig(other) { copy_values_from(other); }

    TM_CONFIG_FIELD(int,      hidden_dim);
    TM_CONFIG_FIELD(int,      head_dim);
    TM_CONFIG_FIELD(int,      head_num);
    TM_CONFIG_FIELD(int,      kv_head_num);
    TM_CONFIG_FIELD(int,      kv_lora_rank);
    TM_CONFIG_FIELD(int,      q_lora_rank);
    TM_CONFIG_FIELD(int,      qk_rope_dim);
    TM_CONFIG_FIELD(int,      v_head_dim);
    TM_CONFIG_FIELD(bool,     has_bias);
    TM_CONFIG_FIELD(bool,     qk_norm);
    TM_CONFIG_FIELD(int,      tp_size);
    TM_CONFIG_FIELD(int,      tp_rank);
    TM_CONFIG_FIELD(DataType, data_type);
    TM_CONFIG_FIELD(int,      window_size, -1);
    TM_CONFIG_FIELD(bool,     attn_sink);
    TM_CONFIG_FIELD(bool,     attn_output_gate);
};

struct FfnConfig: ModuleConfig {
    FfnConfig(): ModuleConfig{"FfnWeight"} {}
    FfnConfig(const FfnConfig& other): ModuleConfig(other) { copy_values_from(other); }

    TM_CONFIG_FIELD(int,      hidden_dim);
    TM_CONFIG_FIELD(int,      inter_size);
    TM_CONFIG_FIELD(bool,     has_bias);
    TM_CONFIG_FIELD(int,      tp_size);
    TM_CONFIG_FIELD(int,      tp_rank);
    TM_CONFIG_FIELD(DataType, data_type);
    TM_CONFIG_FIELD(int,      act_type);
    TM_CONFIG_FIELD(bool,     fuse_silu);
    TM_CONFIG_FIELD(bool,     fused_moe);
};

struct MoeConfig: ModuleConfig {
    MoeConfig(): ModuleConfig{"MoeWeight"} {}
    MoeConfig(const MoeConfig& other): ModuleConfig(other) { copy_values_from(other); }

    TM_CONFIG_FIELD(int,         layer_id);
    TM_CONFIG_FIELD(int,         method);
    TM_CONFIG_FIELD(int,         experts_per_token);
    TM_CONFIG_FIELD(int,         inter_size);
    TM_CONFIG_FIELD(bool,        norm_topk_prob);
    TM_CONFIG_FIELD(bool,        shared_gate);
    TM_CONFIG_FIELD(double,      routed_scale);
    TM_CONFIG_FIELD(bool,        router_bias);
    TM_CONFIG_FIELD(int,         topk_group);
    TM_CONFIG_FIELD(std::string, topk_method);
    TM_CONFIG_FIELD(int,         n_group);
    TM_CONFIG_FIELD(std::string, scoring_func);
    TM_CONFIG_FIELD(int,         router_n_groups);
    TM_CONFIG_FIELD(int,         expert_num);
    TM_CONFIG_FIELD(int,         hidden_dim);
    TM_CONFIG_FIELD(bool,        mlp_bias);
    TM_CONFIG_FIELD(DataType,    data_type);
    TM_CONFIG_FIELD(int,         tp_size);
    TM_CONFIG_FIELD(int,         tp_rank);
    TM_CONFIG_FIELD(int,         act_type);
    TM_CONFIG_FIELD(bool,        fuse_silu);
};

struct DeltaNetConfig: ModuleConfig {
    DeltaNetConfig(): ModuleConfig{"DeltaNetWeight"} {}
    DeltaNetConfig(const DeltaNetConfig& other): ModuleConfig(other) { copy_values_from(other); }

    TM_CONFIG_FIELD(int,      hidden_dim);
    TM_CONFIG_FIELD(int,      num_k_heads);
    TM_CONFIG_FIELD(int,      num_v_heads);
    TM_CONFIG_FIELD(int,      key_head_dim);
    TM_CONFIG_FIELD(int,      value_head_dim);
    TM_CONFIG_FIELD(int,      d_conv, 4);
    TM_CONFIG_FIELD(bool,     has_bias);
    TM_CONFIG_FIELD(int,      tp_size);
    TM_CONFIG_FIELD(int,      tp_rank);
    TM_CONFIG_FIELD(DataType, data_type);
};

struct ModuleListConfig: ModuleConfig {
    ModuleListConfig(): ModuleConfig{"ModuleList"} {}
    ModuleListConfig(const ModuleListConfig& other): ModuleConfig(other) { copy_values_from(other); }
};

struct NormConfig: ModuleConfig {
    NormConfig(): ModuleConfig{"NormWeight"} {}
    NormConfig(const NormConfig& other): ModuleConfig(other) { copy_values_from(other); }

    TM_CONFIG_FIELD(int,      dim);
    TM_CONFIG_FIELD(DataType, data_type);
};

struct DecoderLayerConfig: ModuleConfig {
    DecoderLayerConfig(): ModuleConfig{"DecoderLayerWeight"} {}
    DecoderLayerConfig(const DecoderLayerConfig& other): ModuleConfig(other) { copy_values_from(other); }
};

}  // namespace turbomind::core
