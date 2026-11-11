#pragma once

#include <stdexcept>

#include <cuda_runtime.h>
#include <tvm/ffi/extra/c_env_api.h>

namespace turbomind::tilelang {

// Exported TVM FFI kernels launch on the environment stream of the current thread. Sets it to
// `stream` for this scope and restores the previous one, so nested or concurrent callers on other
// streams (e.g. the engine's auxiliary prefill stream) are unaffected.
class StreamScope {
public:
    StreamScope(int device, cudaStream_t stream): device_(device)
    {
        if (TVMFFIEnvSetStream(kDLCUDA, device_, stream, &previous_))
            throw std::runtime_error("Cannot set the TVM FFI CUDA stream");
    }

    ~StreamScope()
    {
        TVMFFIEnvSetStream(kDLCUDA, device_, previous_, nullptr);
    }

    StreamScope(const StreamScope&) = delete;
    StreamScope& operator=(const StreamScope&) = delete;

private:
    int                device_;
    TVMFFIStreamHandle previous_{};
};

}  // namespace turbomind::tilelang
