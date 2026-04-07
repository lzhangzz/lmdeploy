// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <variant>

namespace turbomind::core {

// Forward declaration — full definition in module.h.
class Module;

/// Configuration value type for module factory system.
/// Supports integer, string, and floating-point values.
using ConfigValue = std::variant<int64_t, std::string, double>;

/// Configuration map passed to module factory functions.
using ModuleConfig = std::map<std::string, ConfigValue>;

/// Module type registry. Maps type name strings to factory functions.
class ModuleRegistry {
public:
    using Factory = std::function<std::unique_ptr<Module>(const ModuleConfig&)>;

    static ModuleRegistry& instance();

    /// Register a factory under the given type name.
    /// Duplicate names overwrite silently.
    void register_type(const std::string& name, Factory factory);

    /// Create a module instance by type name.
    /// Returns nullptr if type name is not registered.
    std::unique_ptr<Module> create(const std::string& type,
                                    const ModuleConfig& config) const;

    /// Check if a type name is registered.
    bool has_type(const std::string& name) const;

private:
    ModuleRegistry() = default;
    std::map<std::string, Factory> factories_;
};

}  // namespace turbomind::core
