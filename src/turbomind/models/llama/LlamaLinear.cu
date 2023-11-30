// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

template<typename T>
void LlamaLinear<T>::forward(
    T* output_data, const T* input_data, int batch_size, const LlamaDenseWeight<T>& weight, Type type)
{
    switch (weight.type) {
        case WeightType::kFP16:
        case WeightType::kFP32:
            forwardFp(output_data, input_data, batch_size, weight, type);
            break;
        case WeightType::kINT4:
            forwardInt4(output_data, input_data, batch_size, weight, type);
            break;
        default:
            FT_CHECK(0);
    }
}

template<typename T>
void gemm_sm80(const T* A, const T* b, T* c, int m, int n, int k, cudaStream_t stream);

template<typename T>
void LlamaLinear<T>::forwardFp(
    T* output_data, const T* input_data, int batch_size, const LlamaDenseWeight<T>& weight, Type type)
{
    FT_CHECK(type == kGemm);
    if (std::is_same_v<T, half> && sm_ >= 80 && batch_size <= 8) {
        gemm_sm80((const T*)weight.kernel,
                  input_data,
                  output_data,
                  weight.output_dims,
                  batch_size,
                  weight.input_dims,
                  stream_);
    }
    else {
        cublas_wrapper_->Gemm(CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              weight.output_dims,
                              batch_size,
                              weight.input_dims,
                              (const T*)weight.kernel,
                              weight.output_dims,
                              input_data,
                              weight.input_dims,
                              output_data,
                              weight.output_dims);
        sync_check_cuda_error();
    }
}

template<typename T>
void LlamaLinear<T>::forwardInt4(
    T* output_data, const T* input_data, int batch_size, const LlamaDenseWeight<T>& weight, Type type)
{
    if constexpr (std::is_same_v<T, half>) {
        gemm_s4_f16_.Run(output_data,
                         (const uint*)weight.kernel,
                         input_data,
                         (const half2*)weight.scales_and_zeros,
                         weight.output_dims,
                         batch_size,
                         weight.input_dims,
                         weight.group_size,
                         type == kFusedSiluFfn ? GemmS4F16::kFusedSiluFfn : GemmS4F16::kGemm,
                         -1,
                         stream_);
        sync_check_cuda_error();
    }
    else {
        FT_CHECK_WITH_INFO(0, "Not implemented");
    }
}

template class LlamaLinear<float>;
template class LlamaLinear<half>;

}  // namespace turbomind