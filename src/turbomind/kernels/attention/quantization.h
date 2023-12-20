#pragma once

#include "src/turbomind/kernels/attention/array_ops.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace turbomind {

template<class T>
__device__ T Infinity()
{
    if constexpr (std::is_same_v<T, half>) {
        return __ushort_as_half((unsigned short)0x7C00U);
    }

    if constexpr (std::is_same_v<T, nv_bfloat16>) {
        return __ushort_as_bfloat16((unsigned short)0x7F80U);
    }

    if constexpr (std::is_same_v<T, float>) {
        return __int_as_float(0x7f800000U);
    }

    return T{};
}

template<class T>
__device__ constexpr T Max(T a, T b)
{
    if constexpr (sizeof(T) == 2) {
        return __hmax(a, b);
    }

    if constexpr (std::is_same_v<T, float>) {
        return fmaxf(a, b);
    }

    if constexpr (std::is_same_v<T, int>) {
        return max(a, b);
    }

    return T{};
}

template<class T>
__device__ constexpr T Min(T a, T b)
{
    if constexpr (sizeof(T) == 2) {
        return __hmin(a, b);
    }

    if constexpr (std::is_same_v<T, float>) {
        return fminf(a, b);
    }

    if constexpr (std::is_same_v<T, int>) {
        return min(a, b);
    }

    return T{};
}

template<bool norm = true>
inline __device__ Array<half, 4> cvt_f16x4_u8(const Array<uint8_t, 4>& src)
{
    static constexpr uint32_t f16_magic = 0x64000000;
    // 01234567 01234567
    // SEEEEEMM MMMMMMMM
    //      1MM XXXXXXXX
    // (1 + x/2^10) * 2^(e-15) -> e-15=10 -> e=25=16+8+1 -> 01100100b -> 0x64
    Array<uint32_t, 2> dst;
    dst[0] = __byte_perm((uint32_t&)src, f16_magic, 0x7170);
    dst[1] = __byte_perm((uint32_t&)src, f16_magic, 0x7372);
    if constexpr (norm) {
        for (int i = 0; i < 4; ++i) {
            ((Array<half, 4>&)dst)[i] -= half(1024);
        }
    }
    return (Array<half, 4>&)dst;
}

template<class T>
inline __device__ T round(float x)
{
    uint32_t y;
    if constexpr (std::is_same_v<T, uint8_t>) {
        asm("cvt.rni.sat.u8.f32 %0, %1;\n" : "=r"(y) : "f"(x));
    }
    if constexpr (std::is_same_v<T, uint16_t>) {
        asm("cvt.rni.sat.u16.f32 %0, %1;\n" : "=r"(y) : "f"(x));
    }
    if constexpr (std::is_same_v<T, uint32_t>) {
        asm("cvt.rni.sat.u32.f32 %0, %1;\n" : "=r"(y) : "f"(x));
    }
    return y;
}

template<class T>
inline __device__ T round(half x)
{
    uint32_t y;
    if constexpr (std::is_same_v<T, uint8_t>) {
        asm("cvt.rni.sat.u8.f16 %0, %1;\n" : "=r"(y) : "h"((uint16_t&)x));
    }
    if constexpr (std::is_same_v<T, uint16_t>) {
        asm("cvt.rni.sat.u16.f16 %0, %1;\n" : "=r"(y) : "h"((uint16_t&)x));
    }
    if constexpr (std::is_same_v<T, uint32_t>) {
        asm("cvt.rni.sat.u32.f16 %0, %1;\n" : "=r"(y) : "h"((uint16_t&)x));
    }
    return y;
}

template<class T, class B>
inline __device__ T quant(float x, B n_bits)
{
    auto y = round<T>(x);
    if constexpr (n_bits < sizeof(T) * 8) {
        return min(y, (1 << n_bits) - 1);
    }
    else {
        return y;
    }
}

template<int WarpThreadC, class T, int C>
__device__ inline void warp_minmax(Array<T, 2>& stats, const Array<T, C>& x)
{
    PRAGMA_UNROLL
    for (int i = 0; i < C; ++i) {
        stats[0] = Min(stats[0], x[i]);
        stats[1] = Max(stats[1], x[i]);
    }
    if constexpr (sizeof(T) == 2) {
        PRAGMA_UNROLL
        for (int mask = WarpThreadC / 2; mask > 0; mask /= 2) {
            Array<T, 2> tmp;
            (uint32_t&)tmp = __shfl_xor_sync(uint32_t(-1), (uint32_t&)stats, mask);
            stats[0]       = Min(stats[0], tmp[0]);
            stats[1]       = Max(stats[1], tmp[1]);
        }
    }
    else {
        PRAGMA_UNROLL
        for (int mask = WarpThreadC / 2; mask > 0; mask /= 2) {
            stats[0] = Min(stats[0], __shfl_xor_sync(uint32_t(-1), stats[0], mask));
            stats[1] = Max(stats[1], __shfl_xor_sync(uint32_t(-1), stats[1], mask));
        }
    }
}

template<int WarpThreadC, class P, class T, class B, int N, int C, int S>
__device__ void warp_stats(Array<P, 2> (&param)[S], const Array<T, N> (&x)[S][C], B n_bits)
{
    PRAGMA_UNROLL
    for (int s = 0; s < S; ++s) {
        Array<T, 2> stats{Infinity<T>(), -Infinity<T>()};
        PRAGMA_UNROLL
        for (int c = 0; c < C; ++c) {
            warp_minmax<WarpThreadC>(stats, x[s][c]);
        }
        const float inv_q_max = fdividef(1.f, float((1 << n_bits) - 1));
        const float scale     = ((float)stats[1] - (float)stats[0]) * inv_q_max;
        param[s][0]           = (P)scale;
        param[s][1]           = (P)stats[0];
    }
}

template<class Q, class T, class P, class B, int N, int C, int S>
__device__ void
quantize(Array<Q, N> (&dst)[S][C], const Array<T, N> (&src)[S][C], const Array<P, 2> (&params)[S], B n_bits)
{
    PRAGMA_UNROLL
    for (int s = 0; s < S; ++s) {
        P inv_scale = (P)fdividef(1.f, (float)params[s][0]);
        P zero      = params[s][1];
        PRAGMA_UNROLL
        for (int c = 0; c < C; ++c) {
            PRAGMA_UNROLL
            for (int i = 0; i < N; ++i) {
                const auto v = ((P)src[s][c][i] - zero) * inv_scale;
                dst[s][c][i] = quant<Q>(v, n_bits);
            }
        }
    }
}

template<class T, class Q, class P, class B, int N, int C, int S>
__device__ void
dequantize(Array<T, N> (&dst)[S][C], const Array<Q, N> (&src)[S][C], const Array<P, 2> (&params)[S], B n_bits)
{
    static_assert(N % 4 == 0);
    PRAGMA_UNROLL
    for (int s = 0; s < S; ++s) {
        auto scale = params[s][0];
        auto zero  = params[s][1];
        PRAGMA_UNROLL
        for (int c = 0; c < C; ++c) {
            if constexpr (1) {
                PRAGMA_UNROLL
                for (int i = 0; i < N; i += 4) {
                    using namespace ops;
                    (Array<T, 4>&)dst[s][c][i] =
                        cast<T>(cast<P>(cvt_f16x4_u8((Array<Q, 4>&)src[s][c][i])) * scale + zero);
                }
            }
            else {
                using signed_t = std::make_signed_t<Q>;
                for (int i = 0; i < N; ++i) {
                    dst[s][c][i] = T(P((signed_t)src[s][c][i] - (signed_t)quant<Q>(-zero / scale, n_bits)) * scale);
                }
            }
        }
    }
}

template<int D, int S>
__device__ void
dequantize_for_frag_K(Array<half, D> (&dst)[S], const Array<uint8_t, D> (&src)[S], const Array<half, 2> (&param)[S])
{
    static_assert(D % 4 == 0);
    PRAGMA_UNROLL
    for (int s = 0; s < S; ++s) {
        PRAGMA_UNROLL
        for (int d = 0; d < D; d += 4) {
            (Array<half, 4>&)dst[s][d] = cvt_f16x4_u8((Array<uint8_t, 4>&)src[s][d]);
        }
    }

    PRAGMA_UNROLL
    for (int s = 0; s < S; ++s) {
        using namespace ops;
        dst[s] = dst[s] * param[s][0] + param[s][1];
    }
}

template<int S, int D>
__device__ void
dequantize_for_frag_V(Array<half, S> (&dst)[D], const Array<uint8_t, S> (&src)[D], const Array<half, 2> (&param)[S])
{
    static_assert(S % 4 == 0);
    PRAGMA_UNROLL
    for (int d = 0; d < D; ++d) {
        PRAGMA_UNROLL
        for (int s = 0; s < S; s += 4) {
            (Array<half, 4>&)dst[d][s] = cvt_f16x4_u8((Array<uint8_t, 4>&)src[d][s]);
        }
    }

    PRAGMA_UNROLL
    for (int s = 0; s < S; ++s) {
        PRAGMA_UNROLL
        for (int d = 0; d < D; ++d) {
            using namespace ops;
            dst[d][s] = dst[d][s] * param[s][0] + param[s][1];
        }
    }
}

}  // namespace turbomind