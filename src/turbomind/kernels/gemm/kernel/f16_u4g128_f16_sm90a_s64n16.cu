// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch/config_sm90_s64n16.h"
#include "src/turbomind/kernels/gemm/cp_async.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/transform.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

void Registry::f16_u4g128_f16_sm90a_s64n16()
{
    using namespace sm90_s64n16;
    using D = cache_policy::Default;

    using C = Sm90_s64n16<Sm90,
                          Operand_A_Pack<uint4_t, kColMajor, 1>,  // A
                          Transform_HMMA_16816<0, 1>,          // tarnsform A
                          Operand_UV_Pack<uint32_t, false>,    // U
                          Operand_B<half, kRowMajor, 16>,      // B
                          Transform_Default,                   // transform B
                          VoidOperand,                         // V
                          kColMajor,                           // order_C
                          half,                                // Tc
                          CtaMapN>;

    //           64  16  16
    Add<C::Type<128, 128, 64, 1, 1, 1, D, D, 3, true, 128, 1>>();

    // Print_(typename C::Type<128, 16, 64, 1, 1, 1, D, D, 3, true, 128, 1>::MMA_Map{});
    // std::abort();
}

}  // namespace turbomind::gemm