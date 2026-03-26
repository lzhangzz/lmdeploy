// Copyright (c) OpenMMLab. All rights reserved.
#ifndef TURBOMIND_CORE_MODULE_H
#define TURBOMIND_CORE_MODULE_H

#include "src/turbomind/core/tensor.h"

namespace turbomind::core {

class Module {
public:
    virtual ~Module();

    Module();

    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;

    Module(Module&&) noexcept = delete;
    Module& operator=(Module&&) noexcept = delete;

    void register_module(std::string name, Module& module, std::optional<int> index = {});
    void register_parameter(std::string name, Tensor& param);

    void remove_module(Module& module);
    void remove_parameter(Tensor& param);

    std::unordered_map<std::string, Tensor*> get_parameters() const;

    /// Walk the module tree and return the module at ``path``.
    /// Path segments are separated by '.' and matched greedily against
    /// registered module names (which may themselves contain '.' for indexed
    /// modules, e.g. "layers.0").  Returns ``nullptr`` when no module matches.
    Module* find_module(const std::string& path);

private:
    void get_parameters_impl(std::string prefix, std::unordered_map<std::string, Tensor*>& m) const;

protected:
    Module* parent_;

    std::vector<std::pair<std::string, Module*>> modules_;
    std::vector<std::pair<std::string, Tensor*>> params_;
};

}  // namespace turbomind::core

#endif  // TURBOMIND_CORE_MODULE_H
