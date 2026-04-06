// Copyright (c) OpenMMLab. All rights reserved.
#ifndef TURBOMIND_CORE_MODULE_H
#define TURBOMIND_CORE_MODULE_H

#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/registry.h"
#include "src/turbomind/core/tensor.h"

namespace turbomind::core {

namespace detail {

/// Extract the member type from a pointer-to-member type.
/// E.g., `member_type<FfnWeight* Derived::*>` → `FfnWeight`.
template<typename T>
struct member_type;

template<typename Class, typename Member>
struct member_type<Member Class::*> {
    using type = Member;
};

template<typename T>
using member_type_t = typename member_type<T>::type;

}  // namespace detail

/// Quantization metadata passed to ``ModuleBase::alloc``.
struct WeightSpec {
    DataType dtype{};        // storage dtype of the weight (e.g., kUint4, kFloat8_e4m3, kFloat16)
    int      group_size = 0; // quantization group size (0 = not quantized)
};

// Forward declarations.
template<typename Derived>
class Module;

/// Type-erased hierarchical module with virtual lifecycle.
///
/// The module tree is built explicitly via ``create_child()`` from the Python
/// loading pipeline. Children are looked up by name; no lazy creation.
///   - ``alloc(param_name, spec)`` allocates tensors on demand and returns
///     a handle for data copying.
///   - ``prepare()`` runs post-load processing (format conversion, fusion).
///   - ``verify()`` walks the tree and collects uninitialized params/modules.
class ModuleBase {
public:
    virtual ~ModuleBase();

    ModuleBase();

    ModuleBase(const ModuleBase&)            = delete;
    ModuleBase& operator=(const ModuleBase&) = delete;
    ModuleBase(ModuleBase&&)                 = delete;
    ModuleBase& operator=(ModuleBase&&)      = delete;

    // ----- Hierarchy (type-erased, owning) -----

    /// Owns child; registers it under the given local name.
    /// Returns raw pointer to the added child.
    virtual ModuleBase* add_child(std::string name, std::unique_ptr<ModuleBase> child);

    /// Non-owning alias (for fused refs, views).
    void add_alias(std::string name, ModuleBase& target);

    // ----- Parameters -----

    /// Register a named parameter tensor.
    void add_param(std::string name, Tensor& tensor);

    // ----- Type info -----

    /// Returns a static string identifying the module type (e.g., "LinearWeight", "NormWeight").
    virtual const char* type() const;

    // ----- Lifecycle (virtual, default = recurse / no-op) -----

    /// Allocate tensors for a named parameter and return for data copy.
    /// Returns empty Tensor if param_name is not recognized.
    /// ``spec`` carries quantization metadata — only used by LinearWeight.
    virtual Tensor alloc(const std::string& param_name, const WeightSpec& spec);

    /// Post-load processing: weight format conversion, fusion.
    /// Default recurses into children.
    virtual void prepare();

    /// Free all owned tensors (for Sleep level 2).
    /// Default recurses into children and resets registered parameters.
    virtual void release();

    /// Move tensors between CPU and GPU (for Sleep level 1 / WakeUp).
    /// Default recurses into children and moves registered parameters.
    virtual void to_device(DeviceType dev);

    // ----- Registry-driven child creation -----

    /// Create a child module using the type registry and attach it.
    /// Returns pointer to the created child, or nullptr on failure.
    ModuleBase* create_child(const std::string& name,
                         const std::string& type_name,
                         const ModuleConfig& config = {});

    /// Typed child accessor. Aborts if child not found.
    template<typename T>
    T* get(const std::string& name) const {
        auto* c = child(name);
        TM_CHECK(c != nullptr) << "child '" << name << "' not found in " << type();
        return static_cast<T*>(c);
    }

    /// Expose children for iteration (execution side).
    const auto& children() const { return children_; }

    // ----- Lookup -----

    /// Find a direct child by name (no creation).
    ModuleBase* child(const std::string& name) const;

    /// Find a child by single segment name.
    ModuleBase* get(const std::string& segment);

    /// Find a parameter by name within this module.
    Tensor* param(const std::string& name) const;

    /// Collect all parameters in the subtree (fully-qualified names).
    std::unordered_map<std::string, Tensor*> params() const;

    // ----- Verification -----

    /// Walk subtree, collect paths of uninitialized params/modules into ``missing``.
    /// Composite modules override to also check required children exist.
    /// Returns true if everything is OK.
    virtual bool verify(std::vector<std::string>& missing);

    // ----- Utilities -----

    /// Build the fully-qualified path by walking up the parent chain.
    std::string full_path() const;

    /// Access the parent module (nullptr for root).
    ModuleBase* parent() const noexcept
    {
        return parent_;
    }

    /// Access the local name of this module within its parent.
    const std::string& name() const noexcept
    {
        return name_;
    }

protected:
    ModuleBase*   parent_ = nullptr;
    std::string   name_;

    std::vector<std::pair<std::string, std::unique_ptr<ModuleBase>>> children_;
    std::vector<std::pair<std::string, ModuleBase*>>                 aliases_;
    std::vector<std::pair<std::string, Tensor*>>                     params_;

private:
    void collect_params(const std::string& prefix, std::unordered_map<std::string, Tensor*>& out) const;
};

// ======================================================================
// Module<Derived> — CRTP base for concrete modules with typed children
// ======================================================================

/// CRTP template that overrides ``add_child`` to populate typed member
/// pointers from ``Derived::kChildren`` (a static constexpr tuple).
///
/// Each entry in ``kChildren`` is a ``std::pair<const char*, ChildType* Derived::*>``
/// mapping a child name to a pointer-to-member.  When ``add_child`` is called,
/// the template iterates the tuple via ``std::apply`` and sets the matching
/// member pointer, then delegates to ``ModuleBase::add_child`` for ownership.
///
/// Concrete modules inherit ``Module<ConcreteModule>`` and declare:
///   - ``static constexpr auto kChildren = std::make_tuple(...);``
///   - ``static constexpr const char* kTypeName = "ConcreteModule";``
///   - Typed member pointers for each child (e.g., ``FfnWeight* ffn_;``)
template<typename Derived>
class Module: public ModuleBase {
public:
    ModuleBase* add_child(std::string name, std::unique_ptr<ModuleBase> child) override
    {
        ModuleBase* raw = child.get();
        bool matched = false;
        auto* self = static_cast<Derived*>(this);
        std::apply([&](const auto&... entry) {
            (try_match(entry, name, raw, self, matched), ...);
        }, Derived::kChildren);

        TM_CHECK(matched)
            << "child name '" << name << "' is not recognized by "
            << Derived::kTypeName;

        return ModuleBase::add_child(std::move(name), std::move(child));
    }

    const char* type() const override
    {
        return Derived::kTypeName;
    }

private:
    /// Attempt to match a single kChildren tuple entry against the child name.
    /// On match, static_cast the raw pointer and assign to the typed member.
    template<typename MemberPtr>
    static void try_match(
        const std::pair<const char*, MemberPtr>& entry,
        const std::string& name,
        ModuleBase* raw,
        Derived* self,
        bool& matched)
    {
        if (matched) {
            return;
        }
        if (name == entry.first) {
            // MemberPtr is e.g. `FfnWeight* Derived::*`.
            // detail::member_type_t<MemberPtr> is `FfnWeight*`.
            // So ChildType = FfnWeight.
            using ChildPtr  = detail::member_type_t<MemberPtr>;
            using ChildType = std::remove_pointer_t<ChildPtr>;
            self->*(entry.second) = static_cast<ChildType*>(raw);
            matched = true;
        }
    }
};

// ======================================================================
// ModuleList — indexed container for layer/expert sequences
// ======================================================================

/// A systematic container for indexed module sequences (layers, experts).
/// Children are added explicitly via ``add_child`` or ``create_child``.
class ModuleList: public ModuleBase {
public:
    const char* type() const override
    {
        return "ModuleList";
    }

    ModuleList() = default;

    /// Override to also track the child in the indexed_ vector.
    ModuleBase* add_child(std::string name, std::unique_ptr<ModuleBase> child) override;

    /// Number of children created so far.
    int size() const;

private:
    std::vector<ModuleBase*> indexed_;
};

}  // namespace turbomind::core

#endif  // TURBOMIND_CORE_MODULE_H
