# Self-Registering Config Fields

## Problem

Config structs in `turbomind::core` (LinearConfig, AttentionConfig, etc.) define fields as plain C++ members. Every field must be manually mirrored in `bind.cpp` with a `.def_readwrite()` call. Adding or modifying a field requires touching two files — easy to forget, tedious to maintain. 8 config structs, 50+ fields, plus identical `clone()` lambdas.

Configs also lack introspection: no iteration, no serialization, no printing.

## Goal

Apply the same self-registration pattern used by `Submodule<T>` and `Parameter` to config fields. Each field registers itself with its parent config on construction, enabling:

- Automatic pybind11 binding (zero `.def_readwrite()` boilerplate)
- C++ field iteration, serialization, printing
- Single source of truth per field

## Design

### FieldDescriptor — plain data struct

```cpp
enum class FieldType : uint8_t { Int, Bool, Double, String, DataType };

struct FieldDescriptor {
    const char* name;
    FieldType   type_tag;
};
```

### FieldTypeTag — compile-time type mapping

```cpp
template<typename T> struct FieldTypeTag;
template<> struct FieldTypeTag<int>         { static constexpr FieldType v = FieldType::Int; };
template<> struct FieldTypeTag<bool>        { static constexpr FieldType v = FieldType::Bool; };
template<> struct FieldTypeTag<double>      { static constexpr FieldType v = FieldType::Double; };
template<> struct FieldTypeTag<std::string> { static constexpr FieldType v = FieldType::String; };
template<> struct FieldTypeTag<DataType>    { static constexpr FieldType v = FieldType::DataType; };
```

### ConfigField<T> — typed field, inherits FieldDescriptor

```cpp
template<typename T>
class ConfigField: public FieldDescriptor {
    T value_;

public:
    template<typename... Args>
    ConfigField(ModuleConfig& parent, const char* name, Args&&... args)
        : FieldDescriptor{name, FieldTypeTag<T>::v}
        , value_(std::forward<Args>(args)...)
    {
        parent.register_field(this);
    }

    // Implicit conversions — existing code works unchanged
    operator T&()             { return value_; }
    operator const T&() const { return value_; }
    ConfigField& operator=(const T& v) { value_ = v; return *this; }
};
```

ConfigField IS-A FieldDescriptor — the field is its own descriptor. Base class initializes first (`name`, `type_tag`), then `value_` is constructed, then the body registers `this` with the parent. Value access goes through `static_cast<ConfigField<T>*>(desc)` using the `type_tag` to determine `T`. No `value_ptr`, no virtual methods, no parent back-pointer.

### ModuleConfig — field registry

```cpp
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
        module_type = other.module_type;
        copy_values_from(other);
        return *this;
    }

    void register_field(FieldDescriptor* f) {
        fields_.push_back(f);
    }

    // Copy values from source by field index (fields are registered in declaration order).
    void copy_values_from(const ModuleConfig& src) {
        for (size_t i = 0; i < fields_.size(); ++i) {
            auto* dst = fields_[i];
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

    // Find a field by name (used by pybind11 binding).
    FieldDescriptor* field(const char* name) const {
        for (auto* f : fields_)
            if (strcmp(f->name, name) == 0) return f;
        return nullptr;
    }

    const std::vector<FieldDescriptor*>& fields() const { return fields_; }

private:
    std::vector<FieldDescriptor*> fields_;  // pointers to ConfigField<T> base (FieldDescriptor)
};
```

### TM_CONFIG_FIELD macro

```cpp
#define TM_CONFIG_FIELD(Type, name, ...) \
    ConfigField<Type> name{*this, #name, ##__VA_ARGS__}
```

Expansions:
- `TM_CONFIG_FIELD(int, hidden_dim)` → `ConfigField<int> hidden_dim{*this, "hidden_dim"}`
- `TM_CONFIG_FIELD(int, d_conv, 4)` → `ConfigField<int> d_conv{*this, "d_conv", 4}`
- `TM_CONFIG_FIELD(bool, has_bias)` → `ConfigField<bool> has_bias{*this, "has_bias"}`

### Example: AttentionConfig before and after

**Before:**
```cpp
struct AttentionConfig: ModuleConfig {
    AttentionConfig(): ModuleConfig{"AttentionWeight"} {}
    int      hidden_dim{};
    int      head_dim{};
    int      d_conv{-1};
    bool     has_bias{};
    DataType data_type{};
};
```

**After:**
```cpp
struct AttentionConfig: ModuleConfig {
    AttentionConfig(): ModuleConfig{"AttentionWeight"} {}
    AttentionConfig(const AttentionConfig& other): ModuleConfig(other) {
        copy_values_from(other);
    }

    TM_CONFIG_FIELD(int,      hidden_dim);
    TM_CONFIG_FIELD(int,      head_dim);
    TM_CONFIG_FIELD(int,      d_conv, -1);
    TM_CONFIG_FIELD(bool,     has_bias);
    TM_CONFIG_FIELD(DataType, data_type);
};
```

Copy construction flow:
1. `ModuleConfig(other)` — copies `module_type`, `fields_` is empty
2. ConfigField members use NSDMI → self-register into `fields_`
3. Constructor body: `copy_values_from(other)` — copies values by index

Existing usage like `config.hidden_dim = 42` and `int x = config.hidden_dim` compiles unchanged due to implicit conversion operators.

### bind.cpp — generic bind_config<T>()

```cpp
template<typename Config, typename T>
void bind_field(py::class_<Config, ModuleConfig>& cls, const std::string& name, size_t index) {
    cls.def_property(name.c_str(),
        [index](const Config& c) -> T {
            return static_cast<const ConfigField<T>&>(*c.fields()[index]);
        },
        [index](Config& c, const T& v) {
            static_cast<ConfigField<T>&>(*c.fields()[index]) = v;
        });
}

template<typename Config>
void bind_config(py::module_& m, const char* name) {
    py::class_<Config, ModuleConfig> cls(m, name);
    cls.def(py::init<>());

    Config tmp;  // construct to discover registered fields (names + types)
    for (size_t i = 0; i < tmp.fields().size(); ++i) {
        auto* desc = tmp.fields()[i];
        std::string fname(desc->name);
        switch (desc->type_tag) {
        case FieldType::Int:      bind_field<Config, int>(cls, fname, i);         break;
        case FieldType::Bool:     bind_field<Config, bool>(cls, fname, i);        break;
        case FieldType::Double:   bind_field<Config, double>(cls, fname, i);      break;
        case FieldType::String:   bind_field<Config, std::string>(cls, fname, i); break;
        case FieldType::DataType: bind_field<Config, DataType>(cls, fname, i);    break;
        }
    }

    cls.def("clone", [](const Config& c) { return Config(c); });
}
```

The temporary config discovers fields (names, types, indices). The getter/setter lambdas capture the field index and access the target instance's `fields_` directly by index — O(1) per access, no name lookup needed.

bind.cpp reduces to:
```cpp
bind_config<LinearConfig>(m, "LinearConfig");
bind_config<AttentionConfig>(m, "AttentionConfig");
bind_config<FfnConfig>(m, "FfnConfig");
bind_config<MoeConfig>(m, "MoeConfig");
bind_config<DeltaNetConfig>(m, "DeltaNetConfig");
bind_config<NormConfig>(m, "NormConfig");
bind_config<ModuleListConfig>(m, "ModuleListConfig");
bind_config<DecoderLayerConfig>(m, "DecoderLayerConfig");
```

### What this unlocks

- **Adding a field** — one line in the struct, zero changes to bind.cpp
- **Adding a new config** — define struct with `TM_CONFIG_FIELD`, one `bind_config` line
- **C++ introspection** — `for (auto& f : config.fields())` iteration
- **Serialization** — generic `to_dict()` / `from_dict()` without per-struct code
- **Debugging** — automatic `__repr__` in Python, `toString()` in C++
- **Clone** — copy constructor with `copy_values_from` works generically

### Constraints

- No RTTI (`typeid` prohibited) — type dispatch uses `FieldType` enum
- No virtual methods — `FieldDescriptor` is a plain struct
- No pointer arithmetic — value access via `static_cast<ConfigField<T>&>` downcast from FieldDescriptor
- Move deleted — only copy (construction + assignment with `copy_values_from`) is supported
- No external dependencies
- Implicit conversion preserves backward compatibility with existing C++ usage
- GCC/Clang `##__VA_ARGS__` for optional macro default values
- C++ field registration order matches declaration order (C++ guarantee)
- `def_property` used instead of `def_readwrite` because field discovery is runtime (def_readwrite requires compile-time member pointers)

### Files changed

- `src/turbomind/core/module_config.h` — FieldDescriptor, FieldType enum, FieldTypeTag, ConfigField<T>, TM_CONFIG_FIELD macro, updated ModuleConfig, all config structs
- `src/turbomind/python/bind.cpp` — replace manual bindings with `bind_config<T>()` calls
