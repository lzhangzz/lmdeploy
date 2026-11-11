#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <dlpack/dlpack.h>

#include "src/turbomind/core/tensor.h"

// Helpers for passing TurboMind tensors to exported TileLang (TVM FFI) kernels.
// Exported functions accept `DLTensor*` arguments directly; no wrapper is needed.
namespace turbomind::tilelang {

inline DLDataType ToDLDataType(DataType type)
{
    switch (type) {
        case kBfloat16:
            return {kDLBfloat, 16, 1};
        case kFloat16:
            return {kDLFloat, 16, 1};
        case kFloat32:
            return {kDLFloat, 32, 1};
        case kInt32:
            return {kDLInt, 32, 1};
        case kInt64:
        case kPointer:
            return {kDLInt, 64, 1};
        case kUint8:
            return {kDLUInt, 8, 1};
        case kBool:
            return {kDLBool, 8, 1};
        default:
            throw std::invalid_argument(std::string{"Unsupported tensor dtype for a TileLang kernel: "}
                                        + to_string(type));
    }
}

// Borrows the tensor's shape/stride storage; the result is valid while `source` is alive and unmodified.
inline DLTensor ToDLTensor(const core::Tensor& source)
{
    static_assert(std::is_same_v<core::ssize_t, int64_t>);
    if (source.device().type != kDEVICE)
        throw std::invalid_argument("TileLang kernels require CUDA tensors");
    return {const_cast<void*>(source.raw_data()),
            {kDLCUDA, source.device().id},
            static_cast<int32_t>(source.ndim()),
            ToDLDataType(source.dtype()),
            const_cast<int64_t*>(source.shape().data()),
            const_cast<int64_t*>(source.stride().data()),
            0};
}

// A leading dimension of extent 1 never affects addressing, so callers may hand it any stride
// (e.g. 0 for a broadcast view). TileLang's TMA descriptor setup expects the dense value. If the
// stride differs, point `tensor.strides` at `storage` holding a corrected copy; borrowed source
// metadata is never modified.
template<size_t N>
void CollapseBatchStride(DLTensor& tensor, int64_t (&storage)[N])
{
    if (tensor.ndim < 2 || static_cast<size_t>(tensor.ndim) > N || tensor.shape[0] != 1)
        throw std::invalid_argument("CollapseBatchStride requires a leading dimension of extent 1");
    const int64_t dense = tensor.shape[1] * tensor.strides[1];
    if (tensor.strides[0] == dense)
        return;
    std::copy_n(tensor.strides, tensor.ndim, storage);
    storage[0]     = dense;
    tensor.strides = storage;
}

}  // namespace turbomind::tilelang
