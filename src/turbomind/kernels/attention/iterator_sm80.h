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

}  // namespace turbomind