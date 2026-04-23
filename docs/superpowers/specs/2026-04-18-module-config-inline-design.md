# Inline Config Structs into `_weight.h` Files

## Goal

Co-locate each config struct with its corresponding weight class. Delete `core/module_config.h`. No behavioral changes.

## Current State

`src/turbomind/core/module_config.h` (178 lines) is a monolithic header containing:

- X-macro config infrastructure: `TM_MEMBER`, `TM_PTR`, `TM_FOR_EACH`
- `ModuleConfig` base struct
- 8 derived config structs: `LinearConfig`, `AttentionConfig`, `FfnConfig`, `MoeConfig`, `DeltaNetConfig`, `ModuleListConfig`, `NormConfig`, `DecoderLayerConfig`

Every `*_weight.h` file includes the entire header even though it only needs one config struct. `core/module.h` and `core/registry.h` also include it.

## Changes

### 1. Move X-macro infrastructure + `ModuleConfig` + `ModuleListConfig` into `core/module.h`

Insert above the `Module` class definition (after includes, inside namespace):

```cpp
// X-macro config field infrastructure
#define TM_MEMBER(Type, name, ...) Type name{__VA_ARGS__};
#define TM_PTR(Type, name, ...)    visitor(#name, &Config::name);
#define TM_FOR_EACH(ClassName, field_list) \
    template<typename Visitor> \
    static void for_each(Visitor&& visitor) { \
        using Config = ClassName; \
        field_list(TM_PTR) \
    }

struct ModuleConfig {
    std::string_view module_type;
};

struct ModuleListConfig: ModuleConfig {
    ModuleListConfig(): ModuleConfig{"ModuleList"} {}
    template<typename Visitor>
    static void for_each(Visitor&&) {}
};
```

Remove the `#include "src/turbomind/core/module_config.h"` from `module.h`.

### 2. Forward-declare `ModuleConfig` in `core/registry.h`

Replace `#include "src/turbomind/core/module_config.h"` with:

```cpp
namespace turbomind::core {
struct ModuleConfig;
}  // namespace turbomind::core
```

The factory signature `std::function<std::unique_ptr<Module>(const ModuleConfig&)>` only needs a declaration, not a definition. The full definition is available in `.cc` files that include `module.h`.

### 3. Inline each config struct into its `_weight.h`

Each config struct moves from `module_config.h` into its corresponding weight header, placed **before** the weight class definition, still in `turbomind::core` namespace. The `_weight.h` files already include `core/module.h`, which now provides the infrastructure.

| Config struct | Target file |
|---|---|
| `LinearConfig` | `models/linear_weight.h` |
| `AttentionConfig` | `models/attention_weight.h` |
| `FfnConfig` | `models/ffn_weight.h` |
| `MoeConfig` | `models/moe_weight.h` |
| `DeltaNetConfig` | `models/delta_net_weight.h` |
| `NormConfig` | `models/norm_weight.h` |
| `DecoderLayerConfig` | `models/decoder_layer_weight.h` |

### 4. Update `python/bind.cpp` includes

Replace `#include "src/turbomind/core/module_config.h"` with includes of each weight header that provides a config struct:

```cpp
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/attention_weight.h"
#include "src/turbomind/models/ffn_weight.h"
#include "src/turbomind/models/moe_weight.h"
#include "src/turbomind/models/delta_net_weight.h"
#include "src/turbomind/models/norm_weight.h"
#include "src/turbomind/models/decoder_layer_weight.h"
```

(`ModuleListConfig` is already in `module.h` which `bind.cpp` includes.)

### 5. Delete `core/module_config.h`

## Scope

Pure file reorganization. No struct body changes. No behavioral changes.

Single commit touching:
- `core/module.h` (add infrastructure, add ModuleListConfig, remove include)
- `core/registry.h` (forward-declare ModuleConfig, remove include)
- 7 weight headers (add config struct, remove include of module_config.h)
- `python/bind.cpp` (update includes)
- 1 file deleted (`core/module_config.h`)
