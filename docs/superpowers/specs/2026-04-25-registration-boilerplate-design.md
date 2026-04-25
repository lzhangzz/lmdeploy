# Registration Boilerplate Reduction

## Problem

9 `.cc` files each contain an identical ~12-line registration block that registers a `Module` subclass with `ModuleRegistry`. The block consists of an anonymous namespace, a struct whose constructor calls `register_type`, a lambda that casts `ModuleConfig` to the concrete config type, and a static instance. The only variation is the module class name, config type, and registered string.

## Solution

### 1. Template overload on `ModuleRegistry`

Add a template method to `ModuleRegistry` in `registry.h`:

```cpp
template<typename T, typename CfgT = ModuleConfig>
void register_type(const std::string& name) {
    register_type(name, [](const ModuleConfig& cfg) -> std::unique_ptr<Module> {
        return std::make_unique<T>(static_cast<const CfgT&>(cfg));
    });
}
```

The default `CfgT = ModuleConfig` handles modules that don't need a typed config (`DecoderLayerWeight`, `ModuleList`).

### 2. `TM_MODULE_REGISTER` macro

Defined in `registry.h`:

```cpp
#define TM_MODULE_REGISTER(ModuleClass, ConfigType)                              \
    namespace {                                                                   \
    static const bool _tm_module_registered_ =                                   \
        ::turbomind::core::ModuleRegistry::instance()                             \
            .register_type<ModuleClass, ConfigType>(#ModuleClass);                \
    }
```

- `#ModuleClass` stringifies to the registered name (e.g., `"ModelWeight"`).
- Anonymous namespace prevents ODR issues across translation units.
- Fixed variable name `_tm_module_registered_` is sufficient since each `.cc` file has exactly one registration.
- The macro works uniformly for all 9 sites. In `module.cc`, the enclosing `namespace turbomind::core` means `ModuleList` resolves without qualification, and `#ModuleList` correctly produces `"ModuleList"`.

### 3. Migration

Each file replaces its registration block with a single line:

| File | Replacement |
|---|---|
| `src/turbomind/models/model_weight.cc` | `TM_MODULE_REGISTER(ModelWeight, core::ModelWeightConfig);` |
| `src/turbomind/models/attention_weight.cc` | `TM_MODULE_REGISTER(AttentionWeight, core::AttentionConfig);` |
| `src/turbomind/models/decoder_layer_weight.cc` | `TM_MODULE_REGISTER(DecoderLayerWeight, core::ModuleConfig);` |
| `src/turbomind/models/ffn_weight.cc` | `TM_MODULE_REGISTER(FfnWeight, core::FfnConfig);` |
| `src/turbomind/models/linear_weight.cc` | `TM_MODULE_REGISTER(LinearWeight, core::LinearConfig);` |
| `src/turbomind/models/moe_weight.cc` | `TM_MODULE_REGISTER(MoeWeight, core::MoeConfig);` |
| `src/turbomind/models/norm_weight.cc` | `TM_MODULE_REGISTER(NormWeight, core::NormConfig);` |
| `src/turbomind/models/delta_net_weight.cc` | `TM_MODULE_REGISTER(DeltaNetWeight, core::DeltaNetConfig);` |
| `src/turbomind/core/module.cc` | `TM_MODULE_REGISTER(ModuleList, ModuleListConfig);` |

## Scope

- Modify `src/turbomind/core/registry.h` — add template overload + macro
- Modify 9 `.cc` files — replace boilerplate blocks with one-liners
- No changes to module class definitions, constructors, or the registry implementation (`registry.cc`)

## Verification

Build with `ninja` in the `build` folder and run model tests with `scripts/test_turbomind_model.py` to confirm registration still works correctly.
