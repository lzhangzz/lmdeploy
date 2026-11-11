#pragma once

#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>

namespace turbomind::tilelang {

// Registry of exported TileLang kernels for one kernel family. Each family owns its own instance, so
// families never see each other's kernels; each kernel is a module file exporting a function named after
// its symbol. The family's Python module registers exactly the modules a model needs through the family's
// own binding (e.g. `_tm.add_flashqla_kernel`).
//
// Families should own a heap instance they never delete (see flashqla/host.cc): loaded CUDA modules must
// not be torn down during static destruction, after the CUDA context may already be gone.
//
//   - Add(symbol, path) records the module; it is loaded on the first Find, i.e. on the engine thread
//     that first plans with it, with that thread's device current.
//   - Find(symbol) returns nullptr for an unregistered symbol (the caller falls back), and throws if the
//     registered module fails to load or does not export the symbol. Loads are memoized.
//
// The TileLang runtime (libtvm_runtime.so, libtilelang.so) must already be loaded with RTLD_GLOBAL; the
// Python front end does that (lmdeploy.turbomind.kernels.tilelang.runtime).
class Library {
public:
    Library() = default;

    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;

    // The file must exist. Re-registering the same file is a no-op; a different file for a registered
    // symbol throws.
    void Add(const std::string& symbol, const std::string& path);

    // Registered, whether or not loaded yet; never loads.
    bool Contains(const std::string& symbol);

    // The returned pointer stays valid for the life of the process.
    const tvm::ffi::Function* Find(const std::string& symbol);

private:
    struct Loaded {
        tvm::ffi::Module   module;
        tvm::ffi::Function function;
    };
    struct Entry {
        std::filesystem::path   path;
        std::unique_ptr<Loaded> loaded;
    };

    std::mutex                             mutex_;
    std::unordered_map<std::string, Entry> entries_;
};

}  // namespace turbomind::tilelang
