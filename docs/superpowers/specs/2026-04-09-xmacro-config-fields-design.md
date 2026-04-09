# X-Macro Config Fields

## Problem

The current `ConfigField<T>` self-registration approach (spec: `2026-04-09-self-registering-config-fields-design.md`) works but introduces complexity:

- `ConfigField<T>` wrapper type causes `static_cast<Enum>` failures (user-defined conversions ignored by `static_cast` to enum)
- Per-instance `vector<FieldDescriptor*>` overhead in every config object
- Custom copy constructors with NSDMI re-registration + `copy_values_from()`
- `def_property` with index capture instead of direct `def_readwrite`

These stem from one root cause: field values are wrapped in a template type rather than being plain C++ members.

## Goal

Replace `ConfigField<T>` with an X-macro approach where config fields are plain C++ members, generated from a single field list per config. A static `for_each` member function iterates all fields via member pointers, enabling:

- Automatic pybind11 binding with `def_readwrite` (generic `bind_config<T>()` template)
- Generic introspection (to_dict, from_dict, repr) via `for_each`
- Default copy/move construction
- Zero per-instance overhead
- No implicit conversion issues

## Design

### ModuleConfig base

```cpp
struct ModuleConfig {
    std::string_view module_type;
};
```

No constructors, no field registry, no virtual methods. Compiler generates default copy/move/destruction. `module_type` is `string_view` since it always refers to a string literal.

### Expansion macros

Three macros defined at file scope in `module_config.h`:

```cpp
// Generates plain struct member with optional default value
#define TM_MEMBER(Type, name, ...) Type name{__VA_ARGS__};

// Generates member pointer visitor call (ignores optional default)
#define TM_PTR(Type, name, ...) visitor(#name, &Config::name);

// Generates static for_each member function
#define TM_FOR_EACH(ClassName, field_list) \
    template<typename Visitor> \
    static void for_each(Visitor&& visitor) { \
        using Config = ClassName; \
        field_list(TM_PTR) \
    }
```

`TM_MEMBER` uses `__VA_ARGS__` for the optional initializer. `TM_PTR` captures the `...` but ignores it. `TM_FOR_EACH` generates the static `for_each` method — the `field_list` argument is the X-macro name, which the preprocessor rescans and expands.

### Config struct pattern

Each config struct follows this pattern:

```cpp
struct AttentionConfig: ModuleConfig {
    AttentionConfig(): ModuleConfig{"AttentionWeight"} {}

    #define ATTENTION_FIELDS(X) \
        X(int,      hidden_dim) \
        X(int,      head_dim) \
        X(int,      head_num) \
        X(int,      kv_head_num) \
        X(int,      kv_lora_rank) \
        X(int,      q_lora_rank) \
        X(int,      qk_rope_dim) \
        X(int,      v_head_dim) \
        X(bool,     has_bias) \
        X(bool,     qk_norm) \
        X(int,      tp_size) \
        X(int,      tp_rank) \
        X(DataType, data_type) \
        X(int,      window_size, -1) \
        X(bool,     attn_sink) \
        X(bool,     attn_output_gate)

    ATTENTION_FIELDS(TM_MEMBER)
    TM_FOR_EACH(AttentionConfig, ATTENTION_FIELDS)

    #undef ATTENTION_FIELDS
};
```

The field list X-macro is `#define`'d inside the struct body and `#undef`'d at the end — limited scope, no namespace pollution. It's used twice: once for `TM_MEMBER` (struct members) and once inside `TM_FOR_EACH` (member pointer iteration).

**Expansion of `TM_FOR_EACH(AttentionConfig, ATTENTION_FIELDS)`:**

```cpp
template<typename Visitor>
static void for_each(Visitor&& visitor) {
    using Config = AttentionConfig;
    ATTENTION_FIELDS(TM_PTR)
}
```

Which further expands to:

```cpp
template<typename Visitor>
static void for_each(Visitor&& visitor) {
    using Config = AttentionConfig;
    visitor("hidden_dim", &Config::hidden_dim);
    visitor("head_dim", &Config::head_dim);
    // ... all 16 fields
    visitor("attn_output_gate", &Config::attn_output_gate);
}
```

### Empty configs

Configs with no fields (ModuleListConfig, DecoderLayerConfig) don't need `for_each`. In bind.cpp, `bind_config` calls `Config::for_each(...)` which won't compile for these. Options:

1. Give them an empty `for_each`: `static void for_each(...) {}`
2. Specialize `bind_config` to detect absence of `for_each` (SFINAE)
3. Just give them empty field list macros

Option 1 is simplest:

```cpp
struct ModuleListConfig: ModuleConfig {
    ModuleListConfig(): ModuleConfig{"ModuleList"} {}
    template<typename Visitor>
    static void for_each(Visitor&&) {}
};
```

### All 8 config structs

**LinearConfig:**

```cpp
struct LinearConfig: ModuleConfig {
    LinearConfig(): ModuleConfig{"LinearWeight"} {}

    #define LINEAR_FIELDS(X) \
        X(int,      input_dim) \
        X(int,      output_dim) \
        X(DataType, data_type) \
        X(bool,     has_bias)

    LINEAR_FIELDS(TM_MEMBER)
    TM_FOR_EACH(LinearConfig, LINEAR_FIELDS)

    #undef LINEAR_FIELDS
};
```

**AttentionConfig:** (shown above)

**FfnConfig:**

```cpp
struct FfnConfig: ModuleConfig {
    FfnConfig(): ModuleConfig{"FfnWeight"} {}

    #define FFN_FIELDS(X) \
        X(int,      hidden_dim) \
        X(int,      inter_size) \
        X(bool,     has_bias) \
        X(int,      tp_size) \
        X(int,      tp_rank) \
        X(DataType, data_type) \
        X(int,      act_type) \
        X(bool,     fuse_silu) \
        X(bool,     fused_moe)

    FFN_FIELDS(TM_MEMBER)
    TM_FOR_EACH(FfnConfig, FFN_FIELDS)

    #undef FFN_FIELDS
};
```

**MoeConfig:**

```cpp
struct MoeConfig: ModuleConfig {
    MoeConfig(): ModuleConfig{"MoeWeight"} {}

    #define MOE_FIELDS(X) \
        X(int,         layer_id) \
        X(int,         method) \
        X(int,         experts_per_token) \
        X(int,         inter_size) \
        X(bool,        norm_topk_prob) \
        X(bool,        shared_gate) \
        X(double,      routed_scale) \
        X(bool,        router_bias) \
        X(int,         topk_group) \
        X(std::string, topk_method) \
        X(int,         n_group) \
        X(std::string, scoring_func) \
        X(int,         router_n_groups) \
        X(int,         expert_num) \
        X(int,         hidden_dim) \
        X(bool,        mlp_bias) \
        X(DataType,    data_type) \
        X(int,         tp_size) \
        X(int,         tp_rank) \
        X(int,         act_type) \
        X(bool,        fuse_silu)

    MOE_FIELDS(TM_MEMBER)
    TM_FOR_EACH(MoeConfig, MOE_FIELDS)

    #undef MOE_FIELDS
};
```

**DeltaNetConfig:**

```cpp
struct DeltaNetConfig: ModuleConfig {
    DeltaNetConfig(): ModuleConfig{"DeltaNetWeight"} {}

    #define DELTANET_FIELDS(X) \
        X(int,      hidden_dim) \
        X(int,      num_k_heads) \
        X(int,      num_v_heads) \
        X(int,      key_head_dim) \
        X(int,      value_head_dim) \
        X(int,      d_conv, 4) \
        X(bool,     has_bias) \
        X(int,      tp_size) \
        X(int,      tp_rank) \
        X(DataType, data_type)

    DELTANET_FIELDS(TM_MEMBER)
    TM_FOR_EACH(DeltaNetConfig, DELTANET_FIELDS)

    #undef DELTANET_FIELDS
};
```

**NormConfig:**

```cpp
struct NormConfig: ModuleConfig {
    NormConfig(): ModuleConfig{"NormWeight"} {}

    #define NORM_FIELDS(X) \
        X(int,      dim) \
        X(DataType, data_type)

    NORM_FIELDS(TM_MEMBER)
    TM_FOR_EACH(NormConfig, NORM_FIELDS)

    #undef NORM_FIELDS
};
```

**ModuleListConfig, DecoderLayerConfig:** (no fields)

```cpp
struct ModuleListConfig: ModuleConfig {
    ModuleListConfig(): ModuleConfig{"ModuleList"} {}
    template<typename Visitor>
    static void for_each(Visitor&&) {}
};

struct DecoderLayerConfig: ModuleConfig {
    DecoderLayerConfig(): ModuleConfig{"DecoderLayerWeight"} {}
    template<typename Visitor>
    static void for_each(Visitor&&) {}
};
```

### bind.cpp — generic bind_config\<T\>

```cpp
template<typename Config>
void bind_config(py::module_& m, const char* name) {
    py::class_<Config, ModuleConfig> cls(m, name);
    cls.def(py::init<>());
    Config::for_each([&](const char* fname, auto member_ptr) {
        cls.def_readwrite(fname, member_ptr);
    });
    cls.def("clone", [](const Config& c) { return Config(c); });
}
```

`Config::for_each` passes each field's name (C string) and member pointer (`T Config::*`) to the generic lambda. `def_readwrite` works directly because the member pointer types are resolved per-field by the generic lambda.

bind.cpp reduces to:

```cpp
py::class_<turbomind::core::ModuleConfig>(m, "ModuleConfig")
    .def_readwrite("module_type", &turbomind::core::ModuleConfig::module_type);

bind_config<turbomind::core::LinearConfig>(m, "LinearConfig");
bind_config<turbomind::core::AttentionConfig>(m, "AttentionConfig");
bind_config<turbomind::core::FfnConfig>(m, "FfnConfig");
bind_config<turbomind::core::MoeConfig>(m, "MoeConfig");
bind_config<turbomind::core::DeltaNetConfig>(m, "DeltaNetConfig");
bind_config<turbomind::core::ModuleListConfig>(m, "ModuleListConfig");
bind_config<turbomind::core::NormConfig>(m, "NormConfig");
bind_config<turbomind::core::DecoderLayerConfig>(m, "DecoderLayerConfig");
```

### Introspection utilities

Built on `for_each`, defined in bind.cpp or a utility header:

```cpp
// to_dict — all configs through one template
template<typename Config>
py::dict config_to_dict(const Config& c) {
    py::dict d;
    Config::for_each([&](const char* name, auto ptr) {
        d[name] = c.*ptr;
    });
    return d;
}

// from_dict
template<typename Config>
void config_from_dict(Config& c, const py::dict& d) {
    Config::for_each([&](const char* name, auto ptr) {
        if (d.contains(name)) {
            using T = std::remove_reference_t<decltype(c.*ptr)>;
            c.*ptr = d[name].cast<T>();
        }
    });
}

// __repr__
template<typename Config>
std::string config_repr(const Config& c) {
    std::ostringstream oss;
    oss << c.module_type << "(";
    bool first = true;
    Config::for_each([&](const char* name, auto ptr) {
        if (!first) oss << ", ";
        first = false;
        oss << name << "=" << c.*ptr;
    });
    oss << ")";
    return oss.str();
}
```

These can be wired into `bind_config` as `.def("to_dict", ...)`, `.def("__repr__", ...)`.

### What changes for consumers

Existing C++ code like `config.hidden_dim = 42` or `int x = config.hidden_dim` works unchanged — fields are plain `int`, `bool`, etc.

The `static_cast<ActivationType>(cfg.act_type)` issue from `ConfigField<T>` disappears — `cfg.act_type` is a plain `int`, `static_cast` to enum just works.

### Files changed

- `src/turbomind/core/module_config.h` — remove ConfigField\<T\>/FieldDescriptor/FieldType infrastructure, replace with TM_MEMBER/TM_PTR/TM_FOR_EACH macros and X-macro field lists. All 8 config structs rewritten.
- `src/turbomind/python/bind.cpp` — remove bind_field helper and old bind_config that uses FieldType dispatch. Replace with generic `bind_config<T>()` using `for_each` + `def_readwrite`.
- `src/turbomind/models/ffn_weight.cc` — revert `static_cast` workaround, back to direct `static_cast<ActivationType>(cfg.act_type)`.
- `src/turbomind/models/moe_weight.cc` — revert `static_cast` workaround similarly.

### Comparison with ConfigField\<T\> approach

| Aspect | ConfigField\<T\> | X-macro + for_each |
|---|---|---|
| Field type | `ConfigField<int>` wrapper | Plain `int` |
| Copy construction | Custom: NSDMI re-register + copy_values_from | `= default` |
| Per-instance overhead | `vector<FieldDescriptor*>` | None |
| Binding | `def_property` with index capture | `def_readwrite` directly |
| Type dispatch | Runtime `FieldType` enum + switch | Compile-time generic lambda |
| Introspection | `fields()` vector iteration | `for_each` visitor |
| Enum static_cast | Workaround needed | Just works |
| Single source of truth | Per-field `TM_CONFIG_FIELD` line | Per-config X-macro field list |
| Macro hygiene | One macro (`TM_CONFIG_FIELD`) | Three macros + scoped field lists |
