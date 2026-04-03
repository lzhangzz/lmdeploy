// Copyright (c) OpenMMLab. All rights reserved.
#ifndef TURBOMIND_CORE_MODULE_H
#define TURBOMIND_CORE_MODULE_H

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/registry.h"
#include "src/turbomind/core/tensor.h"

namespace turbomind::core {

/// Quantization metadata passed to ``Module::alloc``.
struct WeightSpec {
    DataType dtype{};        // storage dtype of the weight (e.g., kUint4, kFloat8_e4m3, kFloat16)
    int      group_size = 0; // quantization group size (0 = not quantized)
};

/// Type-erased hierarchical module with virtual lifecycle and lazy child creation.
///
/// The module tree is built incrementally as weights arrive:
///   - ``get(segment)`` returns an existing child or lazily creates one via
///     the virtual ``ensure_child()`` hook.
///   - ``alloc(param_name, spec)`` allocates tensors on demand and returns
///     a handle for data copying.
///   - ``prepare()`` runs post-load processing (format conversion, fusion).
///   - ``verify()`` walks the tree and collects uninitialized params/modules.
class Module {
public:
    virtual ~Module();

    Module();

    Module(const Module&)            = delete;
    Module& operator=(const Module&) = delete;
    Module(Module&&)                 = delete;
    Module& operator=(Module&&)      = delete;

    // ----- Hierarchy (type-erased, owning) -----

    /// Owns child; registers it under the given local name.
    /// Returns raw pointer to the added child.
    Module* add_child(std::string name, std::unique_ptr<Module> child);

    /// Non-owning alias (for fused refs, views).
    void add_alias(std::string name, Module& target);

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
    Module* create_child(const std::string& name,
                         const std::string& type_name,
                         const ModuleConfig& config = {});

    /// Typed child accessor. Returns nullptr if child not found or wrong type.
    template<typename T>
    T* get(const std::string& name) const {
        return static_cast<T*>(child(name));
    }

    /// Expose children for iteration (execution side).
    const auto& children() const { return children_; }

    // ----- Lazy child creation -----

    /// Override in composite modules to create children on demand.
    /// Called by get() when a child doesn't exist yet.
    /// Returns pointer to the newly-created child, or nullptr if segment is invalid.
    virtual Module* ensure_child(const std::string& segment);

    // ----- Lookup -----

    /// Find a direct child by name (no creation).
    Module* child(const std::string& name) const;

    /// Find or lazily create a child by single segment.
    Module* get(const std::string& segment);

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
    Module*       parent_ = nullptr;
    std::string   name_;

    std::vector<std::pair<std::string, std::unique_ptr<Module>>> children_;
    std::vector<std::pair<std::string, Module*>>                 aliases_;
    std::vector<std::pair<std::string, Tensor*>>                 params_;

private:
    void collect_params(const std::string& prefix, std::unordered_map<std::string, Tensor*>& out) const;
};

// ======================================================================
// ModuleList — indexed container for layer/expert sequences
// ======================================================================

/// A systematic container for indexed module sequences (layers, experts).
/// Children are created lazily from a factory function.
class ModuleList: public Module {
public:
    using Factory = std::function<std::unique_ptr<Module>(int index)>;

    /// Factory is called lazily when an index is first accessed.
    explicit ModuleList(Factory factory);

    const char* type() const override
    {
        return "ModuleList";
    }

    /// Parses ``segment`` as an integer index, creates child via factory if not exists.
    Module* ensure_child(const std::string& segment) override;

    /// Number of children created so far.
    int size() const;

private:
    Factory              factory_;
    std::vector<Module*> indexed_;  // lazily populated
};

}  // namespace turbomind::core

#endif  // TURBOMIND_CORE_MODULE_H
