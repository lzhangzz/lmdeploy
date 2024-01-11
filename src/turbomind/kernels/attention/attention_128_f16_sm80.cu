// Copyright (c) OpenMMLab. All rights reserved.

#include "attention_config.h"
#include "attention_template.h"
#include <type_traits>

namespace turbomind {

// using Kernel =
//     typename attention::AttentionConfig<arch::Sm80, half, half, std::integral_constant<int, 128>, 128>::Kernel;

using Kernel = typename attention::AttentionConfig<arch::Sm80, half, half, int, 128>::Kernel;

template void invokeAttention<Kernel>(const typename Kernel::ParamType& params);

}  // namespace turbomind