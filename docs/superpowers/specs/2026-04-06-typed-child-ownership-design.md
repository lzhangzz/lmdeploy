# Typed Child Ownership for TurboMind Weight Modules

## Problem

Every typed accessor in weight modules (e.g. `w_qkv()`, `ffn()`, `attention()`) calls `child("name")`, which performs a linear scan over a string-keyed vector on every invocation. During inference, this means dozens of string comparisons per layer per forward pass. More importantly, the string key provides no compile-time safety — typos are runtime errors.

## Goal

Replace string-based child lookup with compile-time typed member pointers. Each concrete module declares its children statically. Accessor methods return cached pointers directly. Zero string matching on the inference hot path. Zero boilerplate per child beyond a static tuple declaration.

## Design

### Class hierarchy

Split `Module` into a non-template base and a CRTP template:

```
ModuleBase              — non-template, owns traversal + lifecycle
 └─ Module<Derived>     — CRTP, generic add_child via kChildren tuple
     ├─ FfnWeight
     ├─ AttentionWeight
     ├─ NormWeight
     ├─ LinearWeight
     ├─ DecoderLayerWeight
     ├─ DeltaNetWeight
     ├─ MoeWeight
     └─ ModelWeight
```

`ModuleList` stays non-CRTP (no typed children, variable-length).

### Ownership model

`ModuleBase` keeps a single `children_` vector of `vector<pair<string, unique_ptr<Module>>>` — same as today. The CRTP override in `Module<Derived>::add_child` populates typed member pointers from `kChildren`, then delegates to `ModuleBase::add_child` for ownership.

### Tuple declaration

Each concrete module declares a static constexpr tuple:

```cpp
class FfnWeight: public Module<FfnWeight> {
    LinearWeight* w1_   = nullptr;
    LinearWeight* w3_   = nullptr;
    LinearWeight* w2_   = nullptr;
    LinearWeight* w1w3_ = nullptr;

    static constexpr auto kChildren = std::make_tuple(
        std::pair{"w1",   &FfnWeight::w1_},
        std::pair{"w3",   &FfnWeight::w3_},
        std::pair{"w2",   &FfnWeight::w2_},
        std::pair{"w1w3", &FfnWeight::w1w3_}
    );
    friend class Module<FfnWeight>;

public:
    LinearWeight* w1()   { return w1_; }
    LinearWeight* w3()   { return w3_; }
    LinearWeight* w2()   { return w2_; }
    LinearWeight* w1w3() { return w1w3_; }
};
```

Heterogeneous types work naturally — each `pair` can have a different pointer-to-member type:

```cpp
static constexpr auto kChildren = std::make_tuple(
    std::pair{"w_qkv",  &AttentionWeight::w_qkv_},    // LinearWeight*
    std::pair{"q_norm", &AttentionWeight::q_norm_mod_} // NormWeight*
);
```

### CRTP add_child

```cpp
template<typename Derived>
class Module: public ModuleBase {
public:
    Module* add_child(std::string name, std::unique_ptr<Module> child) override {
        Module* raw = child.get();

        // Populate typed member pointers from kChildren
        bool matched = false;
        std::apply([&](auto&... entry) {
            matched = (try_match(static_cast<Derived*>(this), raw, name, entry.first, entry.second) || ...);
        }, Derived::kChildren);
        TM_CHECK(matched) << "child '" << name << "' not recognized in " << type();

        raw->parent_ = this;
        raw->name_   = name;
        children_.emplace_back(name, raw);
        owned_.emplace_back(std::move(name), std::move(child));
        return raw;
    }

private:
    template<typename T, typename MemberPtr>
    static bool try_match(T* self, Module* raw,
                          const std::string& name, const char* target, MemberPtr member) {
        if (name == target) {
            self->*member = static_cast<std::remove_pointer_t<MemberPtr>>(raw);
            return true;
        }
        return false;
    }
};
```

### ModuleBase changes

- `children_` changes from `vector<pair<string, unique_ptr<Module>>>` to `vector<pair<string, Module*>>`
- New `owned_` : `vector<pair<string, unique_ptr<Module>>>`
- `child(name)` searches `children_` (linear scan on raw pointers — same algorithm, slightly faster)
- `prepare()`, `release()`, `to_device()`, `verify()`, `collect_params()` iterate `children_` — unchanged
- `add_child` in `ModuleBase` becomes the fallback for non-CRTP classes (ModuleList, ModuleBase itself)

### ModuleList

Stays non-CRTP. Overrides `add_child` directly, calls `ModuleBase::add_child`, populates `indexed_`. No `kChildren` — its children are dynamic.

### What stays the same

- Python `create_child("name", "Type", config)` — unchanged
- Registry system — unchanged
- `alloc()`, `prepare()`, `release()`, `to_device()`, `verify()` — unchanged (iterate `children_`)
- `aliases_`, `params_` — unchanged
- Inference code — zero changes

### What changes

| Item | Before | After |
|------|--------|-------|
| `Module` class | Single class | `ModuleBase` + `Module<Derived>` template |
| Inheritance | `class Foo : public Module` | `class Foo : public Module<Foo>` |
| Child storage | `children_` owns via `unique_ptr` | `owned_` owns, `children_` non-owning |
| Typed accessors | `child("name") + static_cast` | Return member pointer directly |
| `ModuleList` | Inherits `Module` | Inherits `ModuleBase` |
| Per-class `add_child` | Needed for caching | Eliminated (template handles it) |

## Impact

- **Inference path**: No string comparisons on child access. Accessor methods become single pointer returns.
- **Child creation path**: One-time string comparison per child during `add_child` (same as today, just moves the match to creation time).
- **Generic traversal**: Unchanged — iterate `children_` vector.
- **Python side**: Unchanged — `create_child` flow identical.
- **Compile-time safety**: Tuple declares exact child set. Mismatched names in Python fail at creation time with clear error.
