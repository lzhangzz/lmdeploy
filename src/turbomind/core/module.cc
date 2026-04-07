// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/core/module.h"

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/registry.h"

#include <sstream>

namespace turbomind::core {

// ======================================================================
// Module
// ======================================================================

Module::Module() = default;

Module::~Module() = default;

// ----- Hierarchy -----

Module* Module::add_child(std::string name, std::unique_ptr<Module> child)
{
    TM_CHECK(child != nullptr);
    TM_CHECK(child->parent_ == nullptr) << "module already has a parent";

    // Wire child's slots against existing children
    for (auto& [slot_name, pp] : child->slots_) {
        for (auto& [cname, cptr] : children_) {
            if (cname == slot_name) {
                *pp = cptr.get();
                break;
            }
        }
    }

    child->parent_ = this;
    child->name_   = name;

    Module* raw = child.get();
    children_.emplace_back(std::move(name), std::move(child));

    // Wire parent's slots to the new child
    for (auto& [slot_name, pp] : slots_) {
        if (*pp == nullptr && children_.back().first == slot_name) {
            *pp = raw;
        }
    }

    // Wire existing siblings' slots to the new child
    for (auto& [cname, cptr] : children_) {
        if (cptr.get() == raw) {
            continue;  // skip the child we just added (already handled above)
        }
        for (auto& [slot_name, pp] : cptr->slots_) {
            if (*pp == nullptr && children_.back().first == slot_name) {
                *pp = raw;
            }
        }
    }

    return raw;
}

void Module::add_alias(std::string name, Module& target)
{
    aliases_.emplace_back(std::move(name), &target);
}

// ----- Parameters -----

void Module::add_param(std::string name, Tensor& tensor)
{
    params_.emplace_back(std::move(name), &tensor);
}

// ----- Type info -----

const char* Module::type() const
{
    return "Module";
}

// ----- Lifecycle -----

Tensor Module::alloc(const std::string& param_name, const WeightSpec& spec)
{
    // Default: return pre-existing param tensor if registered.
    if (auto* t = param(param_name)) {
        return *t;
    }
    return {};
}

Tensor Module::create_param(const std::string& name,
                            const std::vector<size_t>& shape,
                            DataType dtype,
                            int group_size)
{
    auto layout = Layout{std::vector<ssize_t>(shape.begin(), shape.end())};
    auto tensor = Tensor{std::move(layout), dtype, kDEVICE};
    add_param(name, tensor);
    return tensor;
}

void Module::prepare()
{
    for (auto& [name, child] : children_) {
        child->prepare();
    }
}

// ----- Lifecycle: release / to_device -----

void Module::release()
{
    for (auto& [name, child] : children_) {
        child->release();
    }
    for (auto& [name, tensor] : params_) {
        if (tensor && *tensor) {
            *tensor = Tensor{};
        }
    }
}

void Module::to_device(DeviceType dev)
{
    for (auto& [name, child] : children_) {
        child->to_device(dev);
    }
    for (auto& [name, tensor] : params_) {
        if (tensor && *tensor && tensor->device().type != dev) {
            Tensor dst{tensor->layout(), tensor->dtype(), Device{dev, tensor->device().id}};
            Copy(*tensor, dst);
            *tensor = std::move(dst);
        }
    }
}

// ----- Registry-driven child creation -----

Module* Module::create_child(const std::string& name,
                              const std::string& type_name,
                              const ModuleConfig& config)
{
    auto mod = ModuleRegistry::instance().create(type_name, config);
    if (!mod) {
        return nullptr;
    }
    return add_child(name, std::move(mod));
}

// ----- Lookup -----

Module* Module::child(const std::string& name) const
{
    for (auto& [n, c] : children_) {
        if (n == name) {
            return c.get();
        }
    }
    for (auto& [n, c] : aliases_) {
        if (n == name) {
            return c;
        }
    }
    return nullptr;
}

Module* Module::get(const std::string& segment)
{
    auto* c = child(segment);
    TM_CHECK(c != nullptr) << "child '" << segment << "' not found in " << type();
    return c;
}

Tensor* Module::param(const std::string& name) const
{
    for (auto& [n, p] : params_) {
        if (n == name) {
            return p;
        }
    }
    return nullptr;
}

std::unordered_map<std::string, Tensor*> Module::params() const
{
    std::unordered_map<std::string, Tensor*> out;
    collect_params("", out);
    return out;
}

// ----- Verification -----

bool Module::verify(std::vector<std::string>& missing)
{
    for (auto& [name, child] : children_) {
        child->verify(missing);
    }
    for (auto& [name, tensor] : params_) {
        if (!tensor || !*tensor) {
            missing.push_back(full_path() + "." + name);
        }
    }
    return missing.empty();
}

// ----- Utilities -----

std::string Module::full_path() const
{
    if (!parent_) {
        return name_;
    }
    std::string pp = parent_->full_path();
    if (pp.empty()) {
        return name_;
    }
    return pp + "." + name_;
}

// ---- Private ----

void Module::collect_params(const std::string& prefix, std::unordered_map<std::string, Tensor*>& out) const
{
    std::string p = prefix.empty() ? "" : prefix + ".";
    for (auto& [n, t] : params_) {
        out.emplace(p + n, t);
    }
    for (auto& [n, c] : children_) {
        c->collect_params(prefix.empty() ? n : prefix + "." + n, out);
    }
}

// ======================================================================
// ModuleList
// ======================================================================

Module* ModuleList::add_child(std::string name, std::unique_ptr<Module> child)
{
    // Parse index before moving name.
    int index = -1;
    {
        std::istringstream iss(name);
        iss >> index;
        if (!iss.eof()) {
            index = -1;
        }
    }
    auto* raw = Module::add_child(std::move(name), std::move(child));
    if (index >= 0) {
        if (index >= static_cast<int>(indexed_.size())) {
            indexed_.resize(index + 1, nullptr);
        }
        indexed_[index] = raw;
    }
    return raw;
}

int ModuleList::size() const
{
    int n = 0;
    for (auto* p : indexed_) {
        if (p) {
            ++n;
        }
    }
    return n;
}

namespace {
struct ModuleListRegistrar {
    ModuleListRegistrar() {
        core::ModuleRegistry::instance().register_type(
            "ModuleList",
            [](const core::ModuleConfig&) -> std::unique_ptr<core::Module> {
                return std::make_unique<core::ModuleList>();
            });
    }
};
static ModuleListRegistrar _module_list_reg;
} // anonymous namespace

}  // namespace turbomind::core
