// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/models/llama/LlamaDenseWeight.h"

namespace turbomind {

enum class GemmMode {
    kGemm,
    kFusedSiluFfn
};

template<typename T>
class Linear {
public:
    Linear();

    ~Linear();

    struct Arguments {
        GemmMode                   mode;        // gemm mode
        const LlamaDenseWeight<T>* A;           // weight
        const T*                   b;           // input
        T*                         c;           // output
        float*                     partial_c;   // split-k partials (hidden_dims, batch_size, split_k)
        int                        batch_size;  // batch size
        int*                       split_k;     // split-k factor, 0 to disable
        int                        algo;
    };

    void Forward(const Arguments& args);

    void Measure(const Arguments& args);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind