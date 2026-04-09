# Config-Unified Registry Design

Date: 2026-04-09
Scope: C++ module_config.h, registry.h/cc, bind.cpp, weight class constructors, Python callers
Goal: Unify module creation through the registry using typed config inheritance, eliminating duplicate creation paths and redundant constructors.

## Problem Statement

After the April 8 refactoring, module creation has two independent paths:

1. **Config-based path** (bind.cpp:700-774): hard-coded try-cast chain of `py::cast` for each config type, constructs weight classes directly. Bypasses the registry entirely.
2. **Dict-based path** (bind.cpp:776+): calls `Module::create_child(name, type_name, dict)` through `ModuleRegistry::create()` using the old `std::map<string, ConfigValue>` config.

Additionally, each weight class has three constructors:
- Default constructor
- Config constructor (delegates to detailed constructor)
- Detailed multi-param constructor (only exists as delegation target)

The registry + factory functions + detailed constructors form an unreachable path for the config-based flow. The config constructors delegate to detailed constructors that add no value — they just unpack the same fields the config already carries.

## Design

### 1. Config base class

All C++ config structs inherit from `ModuleConfig` which carries `module_type`:

```cpp
struct ModuleConfig {
    std::string module_type;
};

struct AttentionConfig: ModuleConfig {
    AttentionConfig(): ModuleConfig{"AttentionWeight"} {}
    int      hidden_dim{};
    int      head_dim{};
    int      head_num{};
    int      kv_head_num{};
    int      kv_lora_rank{};
    int      q_lora_rank{};
    int      qk_rope_dim{};
    int      v_head_dim{};
    bool     has_bias{};
    bool     qk_norm{};
    int      tp_size{};
    int      tp_rank{};
    DataType data_type{};
    int      window_size{-1};
    bool     attn_sink{};
    bool     attn_output_gate{};
};

struct FfnConfig: ModuleConfig {
    FfnConfig(): ModuleConfig{"FfnWeight"} {}
    int      hidden_dim{};
    int      inter_size{};
    bool     has_bias{};
    int      tp_size{};
    int      tp_rank{};
    DataType data_type{};
    int      act_type{};
    bool     fuse_silu{};
    bool     fused_moe{};
};

struct MoeConfig: ModuleConfig {
    MoeConfig(): ModuleConfig{"MoeWeight"} {}
    int            layer_id{};
    int            method{};
    int            experts_per_token{};
    int            inter_size{};
    bool           norm_topk_prob{};
    bool           shared_gate{};
    double         routed_scale{};
    bool           router_bias{};
    int            topk_group{};
    std::string    topk_method{};
    int            n_group{};
    std::string    scoring_func{};
    int            router_n_groups{};
    int            expert_num{};
    int            hidden_dim{};
    bool           mlp_bias{};
    DataType       data_type{};
    int            tp_size{};
    int            tp_rank{};
    int            act_type{};
    bool           fuse_silu{};
};

struct LinearConfig: ModuleConfig {
    LinearConfig(): ModuleConfig{"LinearWeight"} {}
    int      input_dim{};
    int      output_dim{};
    DataType data_type{};
    bool     has_bias{};
};

struct DeltaNetConfig: ModuleConfig {
    DeltaNetConfig(): ModuleConfig{"DeltaNetWeight"} {}
    int      hidden_dim{};
    int      num_k_heads{};
    int      num_v_heads{};
    int      key_head_dim{};
    int      value_head_dim{};
    int      d_conv{4};
    bool     has_bias{};
    int      tp_size{};
    int      tp_rank{};
    DataType data_type{};
};

struct NormConfig: ModuleConfig {
    NormConfig(): ModuleConfig{"NormWeight"} {}
    int      dim{};
    DataType data_type{};
};

struct ModuleListConfig: ModuleConfig {
    ModuleListConfig(): ModuleConfig{"ModuleList"} {}
};

struct DecoderLayerConfig: ModuleConfig {
    DecoderLayerConfig(): ModuleConfig{"DecoderLayerWeight"} {}
};
```

The existing `using ModuleConfig = std::map<std::string, ConfigValue>` and `using ConfigValue = std::variant<int64_t, std::string, double>` typedefs are renamed to `DictConfig` / `DictConfigValue` during migration and removed after dict callers are converted.

### 2. bind.cpp — single-line delegation

```cpp
// Config-based create_child: accepts any ModuleConfig subclass
.def("create_child",
    [](Module& m, const std::string& name, ModuleConfig& config) -> Module* {
        return m.create_child(name, config.module_type, config);
    },
    py::return_value_policy::reference,
    "name"_a, "config"_a)
```

Takes `ModuleConfig&` directly. pybind11 handles derived-to-base conversion at the binding level — no `py::object`, no try-catch, no manual casting. `Module::create_child` (which already exists) does the registry lookup and `add_child`.

The dict-based `create_child(name, type_name, dict)` overload is removed after dict callers are migrated.

### 3. Registry factory accepts base config ref

```cpp
// registry.h
class ModuleRegistry {
public:
    using Factory = std::function<std::unique_ptr<Module>(const ModuleConfig&)>;

    static ModuleRegistry& instance();

    void register_type(const std::string& name, Factory factory);

    /// Create by type name + typed config.
    std::unique_ptr<Module> create(const std::string& type,
                                    const ModuleConfig& config) const;
};
```

The `ModuleConfig` parameter type changes from the old `DictConfig` (`std::map<string, ConfigValue>`) to the new base class. Each factory `static_cast`s to the concrete type:

```cpp
// ffn_weight.cc
struct FfnWeightRegistrar {
    FfnWeightRegistrar() {
        core::ModuleRegistry::instance().register_type(
            "FfnWeight",
            [](const core::ModuleConfig& base_cfg) -> std::unique_ptr<core::Module> {
                return std::make_unique<FfnWeight>(
                    static_cast<const core::FfnConfig&>(base_cfg));
            });
    }
};
static FfnWeightRegistrar _reg;
```

The old `cfg_get`/`cfg_bool` helpers and dict-based factory bodies are removed.

### 4. Weight class constructors simplified

Each weight class keeps only two constructors:

- **Default constructor** — needed for MoE block view (`MoeWeight` creates `FfnWeight()` then links).
- **Config constructor** — absorbs the full initialization logic directly.

The detailed multi-param constructor is removed. Example for `FfnWeight`:

```cpp
// Before (3 constructors):
FfnWeight() = default;
FfnWeight(const core::FfnConfig& cfg);
FfnWeight(int hidden_dim, int inter_size, bool bias, int tp_size, int tp_rank,
          DataType data_type, ActivationType act_type, bool fuse_silu_act);

// After (2 constructors):
FfnWeight() = default;
FfnWeight(const core::FfnConfig& cfg);
```

Config constructor body contains the full initialization (no delegation to detailed ctor):

```cpp
FfnWeight::FfnWeight(const core::FfnConfig& cfg)
    : hidden_dim_{cfg.hidden_dim}
    , inter_size_{cfg.inter_size}
    , bias_{cfg.has_bias}
    , tp_size_{cfg.tp_size}
    , tp_rank_{cfg.tp_rank}
    , data_type_{cfg.data_type}
    , act_type_{static_cast<ActivationType>(cfg.act_type)}
    , is_fused_silu_{cfg.fuse_silu}
    , is_fused_moe_{cfg.fused_moe}
{
}
```

### 5. MoE block view uses FfnConfig

`moe_weight.cc` creates an `FfnWeight` for the fused MoE block view using individual params. This is converted to use `FfnConfig`:

```cpp
// Before:
block_ = std::make_unique<FfnWeight>(hidden_dim_, moe_param_.inter_size, mlp_bias_,
                                      tp_size_, tp_rank_, data_type_, act_type_,
                                      fuse_silu_act_);

// After:
core::FfnConfig block_cfg;
block_cfg.hidden_dim = hidden_dim_;
block_cfg.inter_size = moe_param_.inter_size;
block_cfg.has_bias   = mlp_bias_;
block_cfg.tp_size    = tp_size_;
block_cfg.tp_rank    = tp_rank_;
block_cfg.data_type  = data_type_;
block_cfg.act_type   = act_type_;
block_cfg.fuse_silu  = fuse_silu_act_;
block_ = std::make_unique<FfnWeight>(block_cfg);
```

### 6. Remaining dict callers migrated

Call sites still using the dict-based path:

1. **`load_context.py:260`** (`commit_linear`) — creates LinearWeight from dict. Convert to `LinearConfig(...).to_cpp()`.
2. **`load_context.py:481`** (`LoadContext.load_linear`) — creates LinearWeight from dict. Same conversion.
3. **`text_model_loader.py:412`** — creates NormWeight from dict. Convert to `NormConfig(...).to_cpp()`.
4. **`load_context.py:507`** (`LoadContext.load_tensor`) — creates arbitrary module from dict. Convert to accept typed config.

After migration, the dict-based `create_child` overload and `DictConfig`/`DictConfigValue` typedefs are removed.

### 7. pybind11 config inheritance bindings

The inheritance must be registered so pybind11 can cast derived configs to `ModuleConfig&`:

```cpp
py::class_<ModuleConfig>(m, "ModuleConfig")
    .def_readwrite("module_type", &ModuleConfig::module_type);

py::class_<AttentionConfig, ModuleConfig>(m, "AttentionConfig")
    .def(py::init<>())
    .def_readwrite("hidden_dim", &AttentionConfig::hidden_dim)
    // ... all fields
    ;

py::class_<FfnConfig, ModuleConfig>(m, "FfnConfig")
    .def(py::init<>())
    // ...
    ;
// Same for MoeConfig, LinearConfig, DeltaNetConfig, NormConfig,
// ModuleListConfig, DecoderLayerConfig
```

## Files Changed

### C++
- `src/turbomind/core/module_config.h` — Add `ModuleConfig` base class, all configs inherit
- `src/turbomind/core/registry.h` — `Factory` accepts `const ModuleConfig&`; rename `ModuleConfig`/`ConfigValue` typedefs to `DictConfig`/`DictConfigValue`
- `src/turbomind/core/registry.cc` — Update `create` signature
- `src/turbomind/core/module.h/cc` — `create_child` accepts `const ModuleConfig&` (new base class)
- `src/turbomind/python/bind.cpp` — Single-line `create_child`, register config inheritance, remove try-catch chain and dict overload
- `src/turbomind/models/attention_weight.h/cc` — Remove detailed constructor, simplify config constructor
- `src/turbomind/models/ffn_weight.h/cc` — Remove detailed constructor, simplify config constructor, remove `cfg_get`/`cfg_bool`
- `src/turbomind/models/moe_weight.cc` — Convert block creation to `FfnConfig`, update registration
- `src/turbomind/models/linear_weight.h/cc` — Remove dict-based registration
- `src/turbomind/models/delta_net_weight.h/cc` — Remove detailed constructor
- `src/turbomind/models/norm_weight.h/cc` — Remove dict-based registration
- `src/turbomind/models/decoder_layer_weight.cc` — Remove dict-based registration

### Python
- `lmdeploy/turbomind/deploy/load_context.py` — Convert dict callers to typed config
- `lmdeploy/turbomind/deploy/text_model_loader.py` — Convert dict callers to typed config

## Migration Order

1. Add `ModuleConfig` base class, make all config structs inherit (backward compatible)
2. Rename old `ModuleConfig`/`ConfigValue` typedefs to `DictConfig`/`DictConfigValue`
3. Update `ModuleRegistry::Factory` to accept `const ModuleConfig&` (new base)
4. Update all factory registrations — `static_cast` from base to concrete config
5. Update `Module::create_child` signature to accept new `ModuleConfig&`
6. Simplify bind.cpp — single-line `create_child` with `ModuleConfig&`, register inheritance
7. Inline detailed constructors into config constructors, remove detailed constructors
8. Convert MoE block view to use `FfnConfig`
9. Convert remaining Python dict callers to typed config
10. Remove dict-based `create_child` overload, `DictConfig`/`DictConfigValue` typedefs, `cfg_get`/`cfg_bool` helpers
