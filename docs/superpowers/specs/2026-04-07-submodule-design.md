# Submodule Design — Replace CRTP Module\<Derived\> with Auto-Registering Submodule\<T\>

## Problem

Every composite weight class (AttentionWeight, FfnWeight, MoeWeight, etc.) repeats each child **three times**:

1. Typed member pointer (`LinearWeight* w_qkv_ = nullptr`)
2. kChildren tuple entry (`std::pair{"w_qkv", &AttentionWeight::w_qkv_}`)
3. Typed accessor (`LinearWeight* w_qkv() const { return w_qkv_; }`)

Plus boilerplate: `friend class core::Module<AttentionWeight>`, CRTP base `Module<AttentionWeight>`.

## Solution

Introduce `Submodule<T>` — a name + typed pointer that auto-registers with its parent during construction. Eliminates the CRTP `Module<Derived>` layer entirely.

### Submodule\<T\>

```cpp
template<class T>
struct Submodule {
    const char* name;
    Module*     pointer = nullptr;

    Submodule(Module& parent, const char* n): name(n)
    {
        parent.add_slot(n, &pointer);
    }

    explicit operator bool() const { return pointer != nullptr; }

    operator T*() const
    {
        return static_cast<T*>(TM_CHECK_NOTNULL(pointer));
    }

    T* operator->() const
    {
        return static_cast<T*>(TM_CHECK_NOTNULL(pointer));
    }
};
```

- `explicit operator bool()` — check if the submodule has been wired
- `operator T*()` / `operator->()` — `TM_CHECK_NOTNULL` fires if accessed before wiring
- No virtual, no `std::function`, no function pointer — just `Module**`

### Module (was ModuleBase)

```cpp
class Module {
    std::vector<std::pair<const char*, Module**>> slots_;
    std::vector<std::pair<std::string, std::unique_ptr<Module>>> children_;

public:
    void add_slot(const char* name, Module** pp)
    {
        slots_.emplace_back(name, pp);
    }

    void add_child(std::string name, std::unique_ptr<Module> child)
    {
        for (auto& [slot_name, pp] : child->slots_) {
            for (auto& [cname, cptr] : children_) {
                if (cname == slot_name) {
                    *pp = cptr;
                    break;
                }
            }
        }
        children_.emplace_back(std::move(name), std::move(child));
    }
};
```

- `add_slot` — called by Submodule constructor, stores `Module**`
- `add_child` — iterates child's slots, matches by name against existing children, assigns pointer

### Derived class — before vs after

**Before:**
```cpp
class AttentionWeight: public core::Module<AttentionWeight> {
    LinearWeight* w_qkv_  = nullptr;
    LinearWeight* wo_     = nullptr;
    NormWeight*   q_norm_ = nullptr;

    static constexpr auto kChildren = std::make_tuple(
        std::pair{"w_qkv",  &AttentionWeight::w_qkv_},
        std::pair{"wo",     &AttentionWeight::wo_},
        std::pair{"q_norm", &AttentionWeight::q_norm_}
    );
    friend class core::Module<AttentionWeight>;

    LinearWeight* w_qkv() const { return w_qkv_; }
    LinearWeight* wo() const { return wo_; }
    NormWeight*   q_norm() const { return q_norm_; }
};
```

**After:**
```cpp
class AttentionWeight: public core::Module {
    Submodule<LinearWeight> w_qkv  {*this, "w_qkv"};
    Submodule<LinearWeight> wo     {*this, "wo"};
    Submodule<NormWeight>   q_norm {*this, "q_norm"};
};
```

### Usage

```cpp
// Access — implicit conversion, TM_CHECK_NOTNULL guards null
attn->w_qkv->alloc();

// Check validity
if (attn.w_qkv) { /* wired */ }
```

### Wiring flow (Python → C++)

```
Python: parent.create_child("attention", "AttentionWeight", cfg)
  → factory creates AttentionWeight
    → Submodule constructors register slots in AttentionWeight::slots_
  → parent.add_child("attention", std::move(child))
    → iterates child->slots_, matches names against parent's children_
    → assigns pointer (child.w_qkv.pointer = parent's "w_qkv" child)
```

## What gets deleted

- `Module<Derived>` CRTP template (module.h)
- `kChildren` tuple in every composite weight class
- `friend class core::Module<...>` in every composite weight class
- Typed accessor methods in every composite weight class
- `try_match`, `std::apply` machinery in Module\<Derived\>::add_child
- `ModuleBase` renamed back to `Module`

## What gets added

- `Submodule<T>` template (new header or in module.h)
- `add_slot` method on Module
- `slots_` vector on Module
- Submodule member declarations in derived classes (replaces member pointer + accessor)
- Submodule constructors in derived class in-class initializers (replaces kChildren)

## Edge cases

- **mutable members** (e.g., FfnWeight): `mutable Submodule<LinearWeight> w1 {*this, "w1"};`
- **ModuleList children**: `Submodule<ModuleList> experts {*this, "experts"};` — same pattern
- **Optional children**: only declared if used; `operator bool()` check handles absence
- **Child ordering**: slots registered in member declaration order (C++ guarantees this matches class definition order)
