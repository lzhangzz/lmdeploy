#include "CuteDSLRuntime.h"
#include "rmsnorm_bf16.h"
#include "rmsnorm_f16.h"
#include "rmsnorm_f32.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

void check_cuda(cudaError_t error)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

void check_dsl(CuteDSLRT_Error_t error)
{
    if (error != CuteDSLRT_Error_Success) {
        throw std::runtime_error(std::string(CuteDSLRT_GetErrorName(error)) + ": "
                                 + CuteDSLRT_GetErrorString(error));
    }
}

class Module {
public:
    explicit Module(const std::string& path)
    {
        if (path.size() >= 3 && path.substr(path.size() - 3) == ".so") {
            const char* libraries[] = {path.c_str()};
            check_dsl(CuteDSLRT_Module_Create_From_Bytes(&module_, nullptr, 0, libraries, 1));
        }
        else {
            std::ifstream file(path, std::ios::binary);
            if (!file) {
                throw std::runtime_error("Cannot open " + path);
            }
            std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)), {});
            check_dsl(CuteDSLRT_Module_Create_From_Bytes(&module_, bytes.data(), bytes.size(), nullptr, 0));
        }
    }

    ~Module()
    {
        if (module_) {
            CuteDSLRT_Module_Destroy(module_);
        }
    }

    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;

    CuteDSLRT_Function_t* function(const std::string& name)
    {
        CuteDSLRT_Function_t* result = nullptr;
        check_dsl(CuteDSLRT_Module_Get_Function(&result, module_, name.c_str()));
        return result;
    }

    void close()
    {
        check_dsl(CuteDSLRT_Module_Destroy(module_));
        module_ = nullptr;
    }

private:
    CuteDSLRT_Module_t* module_ = nullptr;
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
float from_float<float>(float value)
{
    return value;
}

template<>
__half from_float<__half>(float value)
{
    return __float2half_rn(value);
}

template<>
__nv_bfloat16 from_float<__nv_bfloat16>(float value)
{
    return __float2bfloat16_rn(value);
}

float as_float(float value) { return value; }
float as_float(__half value) { return __half2float(value); }
float as_float(__nv_bfloat16 value) { return __bfloat162float(value); }

template<class T, class X, class W, class Y>
int verify(CuteDSLRT_Function_t* function, const std::string& name, float atol, float rtol)
{
    const std::vector<std::pair<int, int>> shapes = {
        {1, 1}, {3, 7}, {2, 31}, {4, 32}, {7, 33}, {16, 128}, {5, 1000}, {8, 4096}, {2, 8192}};
    std::mt19937 random(42);
    std::uniform_real_distribution<float> values(-2.0f, 2.0f);
    Stream owner;
    cudaStream_t stream = owner.get();
    double max_error = 0;
    int cases = 0;

    for (const auto& shape : shapes) {
        const int rows = shape.first;
        const int cols = shape.second;
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

            X x{};
            x.data = device_x.get();
            x.dynamic_shapes[0] = rows;
            x.dynamic_shapes[1] = cols;
            x.dynamic_strides[0] = x_stride;
            W weight{};
            weight.data = device_w.get();
            weight.dynamic_shapes[0] = cols;
            Y y{};
            y.data = device_y.get();
            y.dynamic_shapes[0] = rows;
            y.dynamic_shapes[1] = cols;
            y.dynamic_strides[0] = y_stride;

            for (float eps : {1e-6f, 1e-5f, 1e-3f}) {
                int32_t launch_status = -1;
                void* args[] = {&x, &weight, &y, &eps, &stream, &launch_status};
                CuteDSLRT_Error_t runtime_status = CuteDSLRT_Function_Run(function, args, 6);
                check_dsl(runtime_status);
                check_cuda(static_cast<cudaError_t>(launch_status));
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
                            throw std::runtime_error(name + " mismatch at shape " + std::to_string(rows) + "x"
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
    std::cout << "PASS " << name << ": " << cases << " cases, max_abs_error=" << max_error << '\n';
    return cases;
}

int main(int argc, char** argv)
{
    try {
        if (argc != 2 && argc != 3) {
            throw std::runtime_error("Usage: test_cute_dsl_rmsnorm ARTIFACT_DIR [SHARED_LIBRARY]");
        }
        check_cuda(cudaSetDevice(0));
        check_cuda(cudaFree(nullptr));
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, 0));
        std::cout << "Device: " << properties.name << ", SM " << properties.major << properties.minor << '\n';
        const std::string directory = argv[1];
        int cases = 0;
        {
            Module module(argc == 3 ? argv[2] : directory + "/rmsnorm_f32.o");
            cases += verify<float, rmsnorm_f32_Tensor_x_t, rmsnorm_f32_Tensor_weight_t, rmsnorm_f32_Tensor_y_t>(
                module.function("rmsnorm_f32"), "FP32", 2e-5f, 2e-5f);
            module.close();
        }
        {
            Module module(argc == 3 ? argv[2] : directory + "/rmsnorm_f16.o");
            cases += verify<__half, rmsnorm_f16_Tensor_x_t, rmsnorm_f16_Tensor_weight_t, rmsnorm_f16_Tensor_y_t>(
                module.function("rmsnorm_f16"), "FP16", 2e-3f, 2e-3f);
            module.close();
        }
        {
            Module module(argc == 3 ? argv[2] : directory + "/rmsnorm_bf16.o");
            cases += verify<__nv_bfloat16, rmsnorm_bf16_Tensor_x_t, rmsnorm_bf16_Tensor_weight_t,
                            rmsnorm_bf16_Tensor_y_t>(module.function("rmsnorm_bf16"), "BF16", 2e-2f, 2e-2f);
            module.close();
        }
        std::cout << "ALL " << cases << " RMSNorm cases passed via " << (argc == 3 ? ".so" : ".o")
                  << " dynamic loading\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
