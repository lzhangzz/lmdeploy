// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/meta.h"

namespace turbomind {

__inline__ __device__ void gmma_m64k16_rs(constant<8>, Array<float, 4>& d, const Array<half, 8>& a, uint64_t desc_b)
{
    const uint32_t *A = reinterpret_cast<const uint32_t*>(&a);
    // clang-format off
    asm volatile(
      "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16" 
        "{%0,  %1,  %2,  %3 },"
        "{%4,  %5,  %6,  %7 },"
        " %8,   p,  %10, %11, %12;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      :  "r"(A[0]),  "r"(A[1]),  "r"(A[2]),  "r"(A[3]),
        "l"(desc_b), 
        "n"(1), "n"(1), "n"(1), "n"(0));
    // clang-format on
}

__inline__ __device__ void gmma_m64k16_rs(constant<16>, Array<float, 8>& d, const Array<half, 8>& a, uint64_t desc_b)
{
    const uint32_t *A = reinterpret_cast<const uint32_t*>(&a);
    // clang-format off
    asm volatile(
      "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %13, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16" 
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        "{%8,  %9,  %10, %11},"
        " %12,  p,  %14, %15, %16;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), 
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      :  "r"(A[0]),  "r"(A[1]),  "r"(A[2]),  "r"(A[3]),
        "l"(desc_b),
        "n"(1), "n"(1), "n"(1), "n"(0));
    // clang-format on
}

__inline__ __device__ void gmma_m64k16_rs(constant<32>, Array<float, 16>& d, const Array<half, 8>& a, uint64_t desc_b)
{
    const uint32_t *A = reinterpret_cast<const uint32_t*>(&a);
    // clang-format off
    asm volatile(
      "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %21, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16" 
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15},"
        "{%16, %17, %18, %19},"
        " %20,  p,  %22, %23, %24;\n"
      "}\n"
      : "+f"(d[ 0]), "+f"(d[ 1]), "+f"(d[ 2]), "+f"(d[ 3]), 
        "+f"(d[ 4]), "+f"(d[ 5]), "+f"(d[ 6]), "+f"(d[ 7]),
        "+f"(d[ 8]), "+f"(d[ 9]), "+f"(d[10]), "+f"(d[11]), 
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      :  "r"(A[0]),   "r"(A[1]),   "r"(A[2]),   "r"(A[3]),
        "l"(desc_b),
        "n"(1), "n"(1), "n"(1), "n"(0));
    // clang-format on
}

__inline__ __device__ void gmma_m64k16_rs(constant<64>, Array<float, 32>& d, const Array<half, 8>& a, uint64_t desc_b)
{
    const uint32_t *A = reinterpret_cast<const uint32_t*>(&a);
    // clang-format off
    asm volatile(
      "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16" 
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31},"
        "{%32, %33, %34, %35},"
        " %36,  p,  %38, %39, %40;\n"
      "}\n"
      : "+f"(d[ 0]), "+f"(d[ 1]), "+f"(d[ 2]), "+f"(d[ 3]), 
        "+f"(d[ 4]), "+f"(d[ 5]), "+f"(d[ 6]), "+f"(d[ 7]),
        "+f"(d[ 8]), "+f"(d[ 9]), "+f"(d[10]), "+f"(d[11]), 
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), 
        "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), 
        "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      :  "r"(A[0]),   "r"(A[1]),   "r"(A[2]),   "r"(A[3]),
        "l"(desc_b),
        "n"(1), "n"(1), "n"(1), "n"(0));
    // clang-format on
}

__inline__ __device__ void gmma_m64k16_rs(constant<128>, Array<float, 64>& d, const Array<half, 8>& a, uint64_t desc_b)
{
    const uint32_t *A = reinterpret_cast<const uint32_t*>(&a);
    // clang-format off
    asm volatile(
      "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %69, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16" 
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47, "
        " %48, %49, %50, %51, %52, %53, %54, %55, "
        " %56, %57, %58, %59, %60, %61, %62, %63},"
        "{%64, %65, %66, %67},"
        " %68,  p,  %70, %71, %72;\n"
      "}\n"
      : "+f"(d[ 0]), "+f"(d[ 1]), "+f"(d[ 2]), "+f"(d[ 3]), 
        "+f"(d[ 4]), "+f"(d[ 5]), "+f"(d[ 6]), "+f"(d[ 7]),
        "+f"(d[ 8]), "+f"(d[ 9]), "+f"(d[10]), "+f"(d[11]), 
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), 
        "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), 
        "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), 
        "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), 
        "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), 
        "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), 
        "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      :  "r"(A[0]),   "r"(A[1]),   "r"(A[2]),   "r"(A[3]),
        "l"(desc_b),
        "n"(1), "n"(1), "n"(1), "n"(0));
    // clang-format on
}

}  // namespace turbomind