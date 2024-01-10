#pragma once

#include "iterator.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"

namespace turbomind {

#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 4)
#define L2_CACHEHINT(size) ".L2::" #size "B"
#else
#define L2_CACHEHINT(size)
#endif

template<class T, class Map, class BlockSeqLen, class SmemLayout>
struct Sm80GmemIterator: BaseGmemIterator<T, Map, BlockSeqLen, SmemLayout> {

    using Base = BaseGmemIterator<T, Map, BlockSeqLen, SmemLayout>;

    using typename Base::AccessType;

    using Base::kElementSize;

    // using Base::block_;
    using Base::local_offset_;
    using Base::init_offset_;
    using Base::dst_offset_;
    using Base::smem_int_ptr_;

    using Base::Base;

    template<bool is_residue>
    __device__ void Prefetch(const T* block, int local_id, std::bool_constant<is_residue>, int max_s, int offset)
    {
        auto      src      = block + local_offset_ + local_id * Map::kDimS * Map::kDimC + init_offset_;
        const int offset_s = Map::get_offset(threadIdx.x / WARP_SIZE, threadIdx.x % WARP_SIZE).y;
        PRAGMA_UNROLL
        for (int s = 0; s < Map::kIterS; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < Map::kIterC; ++c) {
                const int idx =
                    SmemLayout::swizzle(dst_offset_ + s * Map::kDeltaS * SmemLayout::kStride + c * Map::kDeltaC);
                if constexpr (is_residue) {
                    CpAsync(offset + idx,
                            &src[s * Map::kDeltaS * Map::kDimC + c * Map::kDeltaC],
                            offset_s + s * Map::kDeltaS < max_s);
                }
                else {
                    CpAsync(offset + idx, &src[s * Map::kDeltaS * Map::kDimC + c * Map::kDeltaC]);
                }
            }
        }
    }

    __device__ void CpAsync(int offset, const T* __restrict__ src, bool mask)
    {
        constexpr int cp_size = sizeof(AccessType);

        offset = smem_int_ptr_ + kElementSize * offset;
#if TURBOMIND_ARCH_SM80
        // clang-format off
        asm volatile("{\n"
                     "  .reg .pred p;\n"
                     "  setp.ne.b32 p, %0, 0;\n"
                     "  @p cp.async.cg.shared.global " L2_CACHEHINT(128) " [%1], [%2], %3;\n"
                     "}\n" ::"r"((int)mask),
                     "r"(offset),
                     "l"(src),
                     "n"(cp_size));
        // clang-format on
#else
        assert(TURBOMIND_ARCH_SM80);
#endif
    }

    __device__ void CpAsync(int offset, const T* __restrict__ src)
    {
        constexpr int cp_size = sizeof(AccessType);

        offset = smem_int_ptr_ + kElementSize * offset;
#if TURBOMIND_ARCH_SM80
        asm volatile(
            "cp.async.cg.shared.global " L2_CACHEHINT(128) " [%0], [%1], %2;\n" ::"r"(offset), "l"(src), "n"(cp_size));
#else
        assert(TURBOMIND_ARCH_SM80);
#endif
    }
};

template<class T, class Layout, int M, int WARPS>
struct Sm80SmemIterQ: BaseSmemIterator<T, Layout> {

    using Base = BaseSmemIterator<T, Layout>;
    using Base::Base;
    using Base::smem_;

    __device__ void Load(Array<T, 8> (&frag_Q)[M], int k)
    {
        PRAGMA_UNROLL
        for (int m = 0; m < M; ++m) {
            Lds(frag_Q[m], &smem_[((k * M + m) * WARPS * WARP_SIZE + threadIdx.x) * 8]);
        }
    }
};

template<class T, class Layout, int N>
struct Sm80SmemIterK: BaseSmemIterator<T, Layout> {

    using Base = BaseSmemIterator<T, Layout>;

    using Base::Base;
    using Base::swizzle_uint_ptr;

    static_assert(N % 2 == 0);

    __device__ void Load(Array<T, 4> (&frag_K)[N], int k)
    {
        const int lane_id       = threadIdx.x % WARP_SIZE;
        const int group_id      = lane_id / 16;
        const int group_lane_id = lane_id % 16;
        PRAGMA_UNROLL
        for (int n = 0; n < N; n += 2) {  // Load (s16,d16) tiles
            const int s = n * +8 + group_lane_id % 8 + group_id * 8;
            const int c = k * 16 + group_lane_id / 8 * 8;
            ldsm_x4((Array<uint32_t, 4>&)frag_K[n], swizzle_uint_ptr(s, c));
        }
    }
};

template<class T, class Layout, int N>
struct Sm80SmemIterV: BaseSmemIterator<T, Layout> {

    using Base = BaseSmemIterator<T, Layout>;

    using Base::swizzle_uint_ptr;

    static_assert(N % 2 == 0);

    Array<int, N / 2> offsets_;

    static constexpr bool kFlag = false;

    __device__ Sm80SmemIterV(const T* smem): Base{smem}
    {
        if constexpr (kFlag) {
            const int lane_id = threadIdx.x % WARP_SIZE;
            PRAGMA_UNROLL
            for (int n = 0; n < N; n += 2) {  //  (d16,s16) tiles
                const int si    = 0 * 16 + lane_id % 16;
                const int di    = n * +8 + lane_id / 16 * 8;
                offsets_[n / 2] = swizzle_uint_ptr(si, di);
            }
        }
    }

    __device__ void Load(Array<T, 4> (&frag_V)[N], int k)
    {
        if constexpr (kFlag) {
            PRAGMA_UNROLL
            for (int n = 0; n < N; n += 2) {  //  (d16,s16) tiles
                const int offset = offsets_[n / 2] + sizeof(T) * k * 16 * Layout::kStride;
                ldsm_x4_trans((Array<uint32_t, 4>&)frag_V[n], offset);
            }
        }
        else {
            const int lane_id = threadIdx.x % WARP_SIZE;
            PRAGMA_UNROLL
            for (int n = 0; n < N; n += 2) {  // Load (d16,s16) tiles
                const int si = k * 16 + lane_id % 16;
                const int di = n * +8 + lane_id / 16 * 8;
                ldsm_x4_trans((Array<uint32_t, 4>&)frag_V[n], swizzle_uint_ptr(si, di));
            }
        }
    }
};

template<class Layout, int N>
struct Sm80SmemIterK<uint8_t, Layout, N>: BaseSmemIterator<uint8_t, Layout> {

    using Base = BaseSmemIterator<uint8_t, Layout>;

    using Base::Base;
    using Base::swizzle_uint_ptr;

    static_assert(N % 4 == 0);

    __device__ void Load(Array<uint8_t, 4> (&frag_K)[N], int k)
    {
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int n = 0; n < N; n += 4) {  // K16,N32 (1x4) per LDSM.x4
            auto&     r = (Array<uint32_t, 4>&)frag_K[n];
            const int s = n * 8 + lane_id;
            const int c = k * 16 + 0;
            ldmatrix_m8n8_x4_b16(r[0], r[1], r[2], r[3], swizzle_uint_ptr(s, c));
        }
    }
};

template<class Layout, int N>
struct Sm80SmemIterV<uint8_t, Layout, N>: BaseSmemIterator<uint8_t, Layout> {

    using Base = BaseSmemIterator<uint8_t, Layout>;

    using Base::Base;
    using Base::swizzle_uint_ptr;

    static_assert(N % 4 == 0);

    __device__ void Load(Array<uint8_t, 4> (&frag_V)[N], int k)
    {
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int n = 0; n < N; n += 4) {  // K16,N32 (2x2) per LDSM.x4
            auto&     r = (Array<uint32_t, 4>&)frag_V[n];
            const int s = lane_id % 16 + k * 16;      // v
            const int c = lane_id / 16 * 16 + n * 8;  // d
            ldsm_x4_trans(r[0], r[1], r[2], r[3], swizzle_uint_ptr(s, c));
        }
    }
};

}  // namespace turbomind