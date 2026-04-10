// Copyright (c) OpenMMLab. All rights reserved.
#ifndef TURBOMIND_CORE_MODULE_H
#define TURBOMIND_CORE_MODULE_H

#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/module_config.h"
#include "src/turbomind/core/registry.h"
#include "src/turbomind/core/tensor.h"

namespace turbomind::core {

// ======================================================================
// X-macro expansion macros for Module-derived classes
//
// Usage in a derived class header:
//
//   #define MY_CHILDREN(X) \
//       X(LinearWeight, w1) \
//       X(NormWeight,   norm)
//
//   #define MY_PARAMS(X) \
//       X(weight) \
//       X(bias)
//
//   class MyWeight: public Module {
//   public:
//       MY_CHILDREN(TM_CHILD_MEMBER)
//       MY_PARAMS(TM_PARAM_MEMBER)
//
//       // Optional: override virtuals using the CASE macros
//       Module* add_child(std::string name, std::unique_ptr<Module> child) override;
//       Module* child(const std::string& name) const override;
//       Tensor* param(const std::string& name) const override;
//       void for_each_child(std::function<void(const char*, Module*)> visitor) const override;
//       void for_each_param(std::function<void(const char*, Tensor&)> visitor) const override;
//   };
//
//   // In the .cc file:
//   Module* MyWeight::add_child(std::string name, std::unique_ptr<Module> child) {
//       MY_CHILDREN(TM_ADD_CHILD_CASE)
//       return nullptr;
//   }
//   // ... etc.
// ======================================================================

/// Declares a unique_ptr<Type> member named `name`.
#define TM_CHILD_MEMBER(Type, name) std::unique_ptr<Type> name;

/// Declares a mutable Tensor member named `name`.
#define TM_PARAM_MEMBER(name) mutable Tensor name{};

/// Fragment for add_child() override body: matches name and stores child.
#define TM_ADD_CHILD_CASE(Type, name)                     \
    if (name_str == #name) {                              \
        TM_CHECK_EQ(child->type(), Type().type());        \
        name##_ = static_cast<Type*>(child.release());    \
        name##_->parent_ = this;                          \
        name##_->name_   = std::move(name);               \
        return name##_;                                   \
    }

/// Fragment for child() override body: matches name and returns pointer.
#define TM_CHILD_CASE(Type, name)    \
    if (name_str == #name) {         \
        return name.get();           \
    }

/// Fragment for param() override body: matches name and returns pointer.
#define TM_PARAM_CASE(name)          \
    if (name_str == #name) {         \
        return &name;                \
    }

/// Fragment for for_each_child() override body: visits child.
#define TM_VISIT_CHILD(Type, name)    \
    visitor(#name, name.get());

/// Fragment for for_each_param() override body: visits param.
#define TM_VISIT_PARAM(name)          \
    visitor(#name, name);

// ======================================================================
// WeightSpec — quantization metadata
// ======================================================================

/// Quantization metadata passed to ``Module::alloc``.
struct WeightSpec {
    DataType dtype{};        // storage dtype of the weight (e.g., kUint4, kFloat8_e4m3, kFloat16)
    int      group_size = 0; // quantization group size (0 = not quantized)
};

// ======================================================================
// Module — type-erased hierarchical module with virtual lifecycle
// ======================================================================

/// Type-erased hierarchical module with virtual lifecycle.
///
/// The module tree is built explicitly via ``create_child()`` from the Python
/// loading pipeline. Children are looked up by name; no lazy creation.
///   - ``alloc(param_name, spec)`` allocates tensors on demand and returns
///     a handle for data copying.
///   - ``prepare()`` runs post-load processing (format conversion, fusion).
///   - ``verify()`` walks the tree and collects uninitialized params/modules.
///
/// Derived classes use X-macro hooks (TM_CHILD_MEMBER, TM_PARAM_MEMBER, etc.)
/// to declare children and parameters as direct members, overriding the
/// virtual lookup methods to match by name.
class Module {
    friend class ModuleList;
public:
    virtual ~Module();

    Module();

    Module(const Module&)            = delete;
    Module& operator=(const Module&) = delete;
    Module(Module&&)                 = delete;
    Module& operator=(Module&&)      = delete;

    // ----- Type info -----

    /// Returns a static string identifying the module type (e.g., "LinearWeight", "NormWeight").
    virtual const char* type() const;

    // ----- Hierarchy (virtual, overridden by derived classes) -----

    /// Owns child; registers it under the given local name.
    /// Returns raw pointer to the added child, or nullptr if name not recognized.
    /// Default: returns nullptr.
    virtual Module* add_child(std::string name, std::unique_ptr<Module> child);

    /// Find a direct child by name. Default: returns nullptr.
    virtual Module* child(const std::string& name) const;

    /// Iterate over all children. Default: no-op.
    virtual void for_each_child(std::function<void(const char*, Module*)> visitor) const;

    // ----- Parameters (virtual, overridden by derived classes) -----

    /// Find a parameter by name within this module. Default: returns nullptr.
    virtual Tensor* param(const std::string& name) const;

    /// Iterate over all parameters. Default: no-op.
    virtual void for_each_param(std::function<void(const char*, Tensor&)> visitor) const;

    // ----- Lifecycle (virtual, default = recurse / no-op) -----

    /// Allocate tensors for a named parameter and return for data copy.
    /// Returns empty Tensor if param_name is not recognized.
    /// ``spec`` carries quantization metadata — only used by LinearWeight.
    virtual Tensor alloc(const std::string& param_name, const WeightSpec& spec);

    /// Create and register a named parameter tensor with the given shape/dtype.
    /// Returns the allocated Tensor for the caller to fill via copy_from().
    Tensor create_param(const std::string& name,
                        const std::vector<size_t>& shape,
                        DataType dtype,
                        int group_size = 0);

    /// Post-load processing: weight format conversion, fusion.
    /// Default recurses into children via for_each_child.
    virtual void prepare();

    // ----- Registry-driven child creation -----

    /// Create a child module using the type registry and attach it.
    /// Uses config.module_type to look up the factory.
    /// Returns pointer to the created child, or nullptr on failure.
    Module* create_child(const std::string& name,
                         const ModuleConfig& config = {});

    /// Typed child accessor. Aborts if child not found.
    template<typename T>
    T* get(const std::string& name) const {
        auto* c = child(name);
        TM_CHECK(c != nullptr) << "child '" << name << "' not found in " << type();
        return static_cast<T*>(c);
    }

    /// Find a child by single segment name. Aborts on null.
    Module* get(const std::string& segment);

    // ----- Verification -----

    /// Walk subtree, collect paths of uninitialized params/modules into ``missing``.
    /// Composite modules override to also check required children exist.
    /// Returns true if everything is OK.
    virtual bool verify(std::vector<std::string>& missing);

    // ----- Utilities -----

    /// Build the fully-qualified path by walking up the parent chain.
    std::string full_path() const;

    /// Access the parent module (nullptr for root).
    Module* parent() const noexcept
    {
        return parent_;
    }

    /// Access the local name of this module within its parent.
    const std::string& name() const noexcept
    {
        return name_;
    }

protected:
    Module*    parent_ = nullptr;
    std::string   name_;
};

// ======================================================================
// ModuleList — indexed container for layer/expert sequences
// ======================================================================

/// A systematic container for indexed module sequences (layers, experts).
/// Children are added explicitly via ``add_child`` or ``create_child``.
class ModuleList: public Module {
public:
    const char* type() const override
    {
        return "ModuleList";
    }

    ModuleList() = default;

    explicit ModuleList(const core::ModuleListConfig&) {}  // empty config, no-op

    /// Override to also track the child in the indexed_ vector.
    Module* add_child(std::string name, std::unique_ptr<Module> child) override;

    /// Find child by name.
    Module* child(const std::string& name) const override;

    /// Iterate over children.
    void for_each_child(std::function<void(const char*, Module*)> visitor) const override;

    /// Number of children created so far.
    int size() const;

private:
    std::vector<std::pair<std::string, std::unique_ptr<Module>>> items_;
    std::vector<Module*> indexed_;
};

}  // namespace turbomind::core

#endif  // TURBOMIND_CORE_MODULE_H
