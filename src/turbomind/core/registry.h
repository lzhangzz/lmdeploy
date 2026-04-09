// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>

#include "src/turbomind/core/module_config.h"

namespace turbomind::core {

// Forward declaration — full definition in module.h.
class Module;

/// Module type registry. Maps type name strings to factory functions.
class ModuleRegistry {
public:
    using Factory = std::function<std::unique_ptr<Module>(const ModuleConfig&)>;

    static ModuleRegistry& instance();

    /// Register a factory under the given type name.
    /// Duplicate names overwrite silently.
    void register_type(const std::string& name, Factory factory);

    /// Create a module instance by type name and typed config.
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
