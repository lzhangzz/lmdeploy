#include "rmsnorm_symbols.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

void check_cuda(cudaError_t error)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

void check_driver(CUresult error)
{
    if (error != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(error, &message);
        throw std::runtime_error(message ? message : "CUDA driver error");
    }
}

class Library {
public:
    explicit Library(const char* path)
    {
        handle_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!handle_) {
            throw std::runtime_error(dlerror());
        }
    }
    ~Library() { dlclose(handle_); }
    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;

    template<class T>
    T symbol(const std::string& name) const
    {
        dlerror();
        void* address = dlsym(handle_, name.c_str());
        if (const char* error = dlerror()) {
            throw std::runtime_error(error);
        }
        return reinterpret_cast<T>(address);
    }

private:
    void* handle_ = nullptr;
};

class Kernel {
public:
    using Launch = CUresult (*)(CUstream, CUdeviceptr, CUdeviceptr, CUdeviceptr, int32_t, int32_t,
                                int64_t, int64_t, double);
    using Load = void (*)();
    using Unload = void (*)();

    Kernel(const Library& library, const std::string& name)
    {
        launch_ = library.symbol<Launch>(name);
        unload_ = library.symbol<Unload>("unload_" + name);
        library.symbol<Load>("load_" + name)();
    }
    ~Kernel() { unload_(); }
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;

    void run(void* x, void* weight, void* y, int rows, int cols, int64_t x_stride, int64_t y_stride,
             double eps, cudaStream_t stream) const
    {
        check_driver(launch_(reinterpret_cast<CUstream>(stream), reinterpret_cast<CUdeviceptr>(x),
                             reinterpret_cast<CUdeviceptr>(weight), reinterpret_cast<CUdeviceptr>(y),
                             rows, cols, x_stride, y_stride, eps));
    }

private:
    Launch launch_ = nullptr;
    Unload unload_ = nullptr;
};

class Stream {
public:
    Stream() { check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(stream_); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t bytes) { check_cuda(cudaMalloc(&data_, bytes)); }
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void* get() const { return data_; }

private:
    void* data_ = nullptr;
};

template<class T>
T from_float(float value);

template<>
float from_float<float>(float value) { return value; }

template<>
__half from_float<__half>(float value) { return __float2half_rn(value); }

template<>
__nv_bfloat16 from_float<__nv_bfloat16>(float value) { return __float2bfloat16_rn(value); }

float as_float(float value) { return value; }
float as_float(__half value) { return __half2float(value); }
float as_float(__nv_bfloat16 value) { return __bfloat162float(value); }

void capture_and_replay(const Kernel& kernel, void* x, void* weight, void* y, int rows, int cols,
                        int64_t x_stride, int64_t y_stride, double eps, cudaStream_t stream)
{
    cudaGraph_t graph = nullptr;
    check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    kernel.run(x, weight, y, rows, cols, x_stride, y_stride, eps, stream);
    check_cuda(cudaStreamEndCapture(stream, &graph));
    size_t count = 0;
    check_cuda(cudaGraphGetNodes(graph, nullptr, &count));
    if (count != 1) {
        cudaGraphDestroy(graph);
        throw std::runtime_error("Expected one kernel captured on the caller's stream, got " + std::to_string(count));
    }
    cudaGraphNode_t node = nullptr;
    check_cuda(cudaGraphGetNodes(graph, &node, &count));
    cudaGraphNodeType type;
    check_cuda(cudaGraphNodeGetType(node, &type));
    if (type != cudaGraphNodeTypeKernel) {
        cudaGraphDestroy(graph);
        throw std::runtime_error("Captured node is not a CUDA kernel");
    }
    cudaGraphExec_t executable = nullptr;
    check_cuda(cudaGraphInstantiate(&executable, graph, 0));
    check_cuda(cudaGraphLaunch(executable, stream));
    check_cuda(cudaStreamSynchronize(stream));
    check_cuda(cudaGraphExecDestroy(executable));
    check_cuda(cudaGraphDestroy(graph));
}

template<class T>
int verify(const Library& library, const std::string& name, const std::string& symbol, float atol, float rtol)
{
    Kernel kernel(library, symbol);
    std::mt19937 random(42);
    std::uniform_real_distribution<float> values(-2.0f, 2.0f);
    Stream owner;
    cudaStream_t stream = owner.get();
    double max_error = 0;
    int cases = 0;
    int graph_replays = 0;

    for (int cols : {1, 7, 31, 32, 33, 128, 1000, 4096, 8192}) {
        for (int rows : {1, 3, 17}) {
            for (int pattern = 0; pattern < 3; ++pattern) {
                const int64_t x_stride = pattern == 1 ? ((cols + 7) / 8 + 1) * 8 : cols;
                const int64_t y_stride = pattern == 1 ? x_stride + 8 : cols;
                const size_t x_count = rows * x_stride;
                const size_t y_count = rows * y_stride;
                const T sentinel = from_float<T>(-123.0f);
                std::vector<T> host_x(x_count, sentinel);
                std::vector<T> host_w(cols);
                std::vector<T> host_y(y_count, sentinel);
                for (int col = 0; col < cols; ++col) {
                    host_w[col] = from_float<T>(values(random));
                }
                for (int row = 0; row < rows; ++row) {
                    for (int col = 0; col < cols; ++col) {
                        float value = pattern == 0 ? 0.0f : values(random);
                        if (pattern == 2) {
                            value *= 1e-4f;
                        }
                        host_x[row * x_stride + col] = from_float<T>(value);
                    }
                }
                DeviceBuffer device_x(x_count * sizeof(T));
                DeviceBuffer device_w(cols * sizeof(T));
                DeviceBuffer device_y(y_count * sizeof(T));
                check_cuda(cudaMemcpyAsync(device_x.get(), host_x.data(), x_count * sizeof(T),
                                           cudaMemcpyHostToDevice, stream));
                check_cuda(cudaMemcpyAsync(device_w.get(), host_w.data(), cols * sizeof(T),
                                           cudaMemcpyHostToDevice, stream));
                check_cuda(cudaMemcpyAsync(device_y.get(), host_y.data(), y_count * sizeof(T),
                                           cudaMemcpyHostToDevice, stream));

                for (double eps : {static_cast<double>(1e-6f), static_cast<double>(1e-5f), static_cast<double>(1e-3f)}) {
                    kernel.run(device_x.get(), device_w.get(), device_y.get(), rows, cols, x_stride, y_stride, eps, stream);
                    if (rows == 1 && pattern == 1 && eps == static_cast<double>(1e-6f)) {
                        check_cuda(cudaStreamSynchronize(stream));
                        std::fill(host_y.begin(), host_y.end(), sentinel);
                        check_cuda(cudaMemcpyAsync(device_y.get(), host_y.data(),
                                                   y_count * sizeof(T), cudaMemcpyHostToDevice, stream));
                        check_cuda(cudaStreamSynchronize(stream));
                        capture_and_replay(kernel, device_x.get(), device_w.get(), device_y.get(), rows, cols,
                                           x_stride, y_stride, eps, stream);
                        ++graph_replays;
                    }
                    check_cuda(cudaMemcpyAsync(host_y.data(), device_y.get(), y_count * sizeof(T),
                                               cudaMemcpyDeviceToHost, stream));
                    check_cuda(cudaStreamSynchronize(stream));
                    for (int row = 0; row < rows; ++row) {
                        double sum_sq = 0;
                        for (int col = 0; col < cols; ++col) {
                            double value = as_float(host_x[row * x_stride + col]);
                            sum_sq += value * value;
                        }
                        double inv_rms = 1.0 / std::sqrt(sum_sq / cols + eps);
                        for (int col = 0; col < cols; ++col) {
                            double expected = as_float(from_float<T>(static_cast<float>(
                                as_float(host_x[row * x_stride + col]) * inv_rms * as_float(host_w[col]))));
                            double actual = as_float(host_y[row * y_stride + col]);
                            double error = std::abs(actual - expected);
                            max_error = std::max(max_error, error);
                            if (!std::isfinite(actual) || error > atol + rtol * std::abs(expected)) {
                                throw std::runtime_error(name + " mismatch at " + std::to_string(rows) + "x"
                                                         + std::to_string(cols) + ", row " + std::to_string(row)
                                                         + ", col " + std::to_string(col) + ": got "
                                                         + std::to_string(actual) + ", expected "
                                                         + std::to_string(expected));
                            }
                        }
                        for (int64_t col = cols; col < y_stride; ++col) {
                            if (as_float(host_y[row * y_stride + col]) != as_float(sentinel)) {
                                throw std::runtime_error(name + " overwrote output padding");
                            }
                        }
                    }
                    ++cases;
                }
                std::vector<T> after_x(x_count);
                std::vector<T> after_w(cols);
                check_cuda(cudaMemcpy(after_x.data(), device_x.get(), x_count * sizeof(T), cudaMemcpyDeviceToHost));
                check_cuda(cudaMemcpy(after_w.data(), device_w.get(), cols * sizeof(T), cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < x_count; ++i) {
                    if (as_float(after_x[i]) != as_float(host_x[i])) {
                        throw std::runtime_error(name + " modified input");
                    }
                }
                for (int col = 0; col < cols; ++col) {
                    if (as_float(after_w[col]) != as_float(host_w[col])) {
                        throw std::runtime_error(name + " modified weights");
                    }
                }
            }
        }
    }
    std::cout << "PASS " << name << ": " << cases << " cases, max_abs_error=" << max_error
              << ", graph_replays=" << graph_replays << '\n';
    return cases;
}

int main(int argc, char** argv)
{
    try {
        if (argc != 2) {
            throw std::runtime_error("Usage: test_triton_rmsnorm KERNEL_LIBRARY");
        }
        check_cuda(cudaSetDevice(0));
        check_cuda(cudaFree(nullptr));
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, 0));
        std::cout << "Device: " << properties.name << ", SM " << properties.major << properties.minor << '\n';
        Library library(argv[1]);
        int cases = 0;
        cases += verify<float>(library, "FP32", RMSNORM_F32_SYMBOL, 2e-5f, 2e-5f);
        cases += verify<__half>(library, "FP16", RMSNORM_F16_SYMBOL, 2e-3f, 2e-3f);
        cases += verify<__nv_bfloat16>(library, "BF16", RMSNORM_BF16_SYMBOL, 2e-2f, 2e-2f);
        std::cout << "ALL " << cases << " RMSNorm cases passed via Triton .so dynamic loading\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
