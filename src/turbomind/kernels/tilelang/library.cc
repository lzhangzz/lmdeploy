#include "src/turbomind/kernels/tilelang/library.h"

#include <exception>
#include <stdexcept>
#include <system_error>

namespace turbomind::tilelang {

namespace ffi = tvm::ffi;

void Library::Add(const std::string& symbol, const std::string& path)
{
    if (symbol.empty() || path.empty())
        throw std::invalid_argument("TileLang kernel symbol and path must not be empty");
    std::error_code ec;
    auto            file = std::filesystem::weakly_canonical(path, ec);
    if (ec)
        file = std::filesystem::absolute(path);
    if (!std::filesystem::is_regular_file(file))
        throw std::invalid_argument("TileLang module for " + symbol + " not found: " + file.string());
    std::lock_guard lock(mutex_);
    const auto [it, inserted] = entries_.try_emplace(symbol, Entry{file, nullptr});
    if (!inserted && it->second.path != file)
        throw std::invalid_argument("TileLang kernel " + symbol + " is already registered from "
                                    + it->second.path.string() + ", not " + file.string());
}

bool Library::Contains(const std::string& symbol)
{
    std::lock_guard lock(mutex_);
    return entries_.count(symbol) != 0;
}

const ffi::Function* Library::Find(const std::string& symbol)
{
    std::lock_guard lock(mutex_);
    const auto      it = entries_.find(symbol);
    if (it == entries_.end())
        return nullptr;
    auto& entry = it->second;
    if (!entry.loaded) {
        const auto path   = entry.path.string();
        auto       module = [&] {
            try {
                return ffi::Module::LoadFromFile(path);
            }
            catch (const std::exception& error) {
                throw std::runtime_error("Cannot load TileLang module " + path + ": " + error.what()
                                         + " (is the TileLang runtime loaded? see "
                                           "lmdeploy.turbomind.kernels.tilelang.runtime.load_runtime)");
            }
        }();
        auto function = module->GetFunction(symbol);
        if (!function)
            throw std::invalid_argument("TileLang module " + path + " does not export " + symbol);
        entry.loaded = std::make_unique<Loaded>(Loaded{std::move(module), std::move(function.value())});
    }
    return &entry.loaded->function;
}

}  // namespace turbomind::tilelang
