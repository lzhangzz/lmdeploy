// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm_s_f16/gemm_s4_f16.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/utils/cublasMMWrapper.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"
#include <type_traits>

namespace turbomind {

template<typename T>
class LlamaLinear {
public:
    enum Type {
        kGemm,
        kFusedSiluFfn
    };

    LlamaLinear(cublasMMWrapper* cublas_wrapper, cudaStream_t stream): cublas_wrapper_(cublas_wrapper), stream_(stream)
    {
        sm_ = getSMVersion();
    }

    void
    forward(T* output_data, const T* input_data, int batch_size, const LlamaDenseWeight<T>& weight, Type type = kGemm);

private:
    void forwardFp(T* output_data, const T* input_data, int batch_size, const LlamaDenseWeight<T>& weight, Type type);

    void forwardInt4(T* output_data, const T* input_data, int batch_size, const LlamaDenseWeight<T>& weight, Type type);

private:
    cublasMMWrapper* cublas_wrapper_;
    cudaStream_t     stream_{};
    GemmS4F16        gemm_s4_f16_;
    int              sm_;
};

}  // namespace turbomind
