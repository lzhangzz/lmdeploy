#pragma once

#include "iterator.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"

namespace turbomind {

#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 4)
#define L2_CACHEHINT(size) ".L2::" #size "B"
#else
#define L2_CACHEHINT(size)
#endif

template<class T, class Map, class SmemLayout>
struct Sm80GmemIterator: BaseGmemIterator<T, Map, SmemLayout> {

    using Base = BaseGmemIterator<T, Map, SmemLayout>;

    using typename Base::AccessType;

    using Base::kElementSize;

    using Base::local_offset_;
    using Base::init_offset_;
    using Base::dst_offset_;
    using Base::smem_int_ptr_;

    // static constexpr int kStepS = kElementSize * Map::kDeltaS * SmemLayout::kStride;
    // static constexpr int kStepC = kElementSize * Map::kDeltaC;

    static constexpr int kStepS = Map::kDeltaS * SmemLayout::kStride;
    static constexpr int kStepC = Map::kDeltaC;

    __device__ Sm80GmemIterator(int local_offset, int warp_id, int lane_id): Base{local_offset, warp_id, lane_id}
    {
        // dst_offset_ *= kElementSize;
    }

    template<bool is_residue, class BlockIter>
    __device__ void Prefetch(const BlockIter& block_iter, int s_begin, int s_count, int max_s, int offset)
    {
        auto      src = block_iter.block + local_offset_ + block_iter.local_id * Map::kDimS * Map::kDimC + init_offset_;
        const int offset_s   = Map::get_offset(threadIdx.x / WARP_SIZE, threadIdx.x % WARP_SIZE).y;
        const int dst_offset = dst_offset_;
        PRAGMA_UNROLL
        for (int s = s_begin; s < s_begin + s_count; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < Map::kIterC; ++c) {
                const int idx = SmemLayout::swizzle_x(kElementSize * (dst_offset + s * kStepS + c * kStepC));
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

    template<bool is_residue, class BlockIter>
    __device__ void Prefetch(const BlockIter& block_iter, int max_s, int offset)
    {
        Prefetch<is_residue>(block_iter, 0, Map::kIterS, max_s, offset);
    }

    __device__ void CpAsync(int offset, const T* __restrict__ src, bool mask)
    {
        constexpr int cp_size = sizeof(AccessType);

        offset = 0 + offset;
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

        offset = 0 + offset;
#if TURBOMIND_ARCH_SM80
        asm volatile(
            "cp.async.cg.shared.global " L2_CACHEHINT(128) " [%0], [%1], %2;\n" ::"r"(offset), "l"(src), "n"(cp_size));
#else
        assert(TURBOMIND_ARCH_SM80);
#endif
    }
};

}  // namespace turbomind