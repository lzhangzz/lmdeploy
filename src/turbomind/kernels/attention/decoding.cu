// Copyright (c) OpenMMLab. All rights reserved.

#include "decoding.h"
#include "decoding_config.h"

namespace turbomind {

template<class Kernel>
void invokeDecoding(const typename Kernel::ParamType& params);

template<class T>
void dispatchDecoding(const AttentionParams<T>& params)
{
    using namespace attention;
    if (params.size_per_head == 128) {
        if (0) {}
        else if (params.arch >= 80) {
            // using Config = DecodingConfig<T, T, std::integral_constant<int, 128>, 128>;
            using Config = DecodingConfig<T, T, int, 128>;
            invokeDecoding<typename Config::Kernel>(params);
        }
    }
}

template void dispatchDecoding(const AttentionParams<half>& params);

}  // namespace turbomind