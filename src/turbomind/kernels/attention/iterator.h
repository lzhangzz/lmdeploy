// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "../gemm_s_f16/common.h"
#include "array_ops.h"
#include <type_traits>

namespace turbomind {

#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 4)
#define L2_CACHEHINT(size) ".L2::" #size "B"
#else
#define L2_CACHEHINT(size)
#endif

constexpr int SMEM_PAD = 8;

template<class T, class Map, class BlockSeqLen, class Swizzle, int Stages>
struct GmemIterator {
    using ElementType = T;
    using AccessType  = Array<T, Map::kAccessC>;

    static constexpr int kElementSize = sizeof(ElementType);
    static constexpr int kAccessSize  = sizeof(AccessType);
    static constexpr int kSizePerTile = Map::kDimS * (Map::kDimC + SMEM_PAD);
    static constexpr int kIterCount   = Map::kIterS * Map::kIterC;

    Swizzle swizzle_;

    const T** block_ptrs_;
    const T*  block_;

    T* smem_;

    // const int block_shifter_;
    // const int block_mask_;
    BlockSeqLen block_seqlen_;

    // const Array<int, 2> local_offsets_;
    const int local_offset_;

    int init_offset_;
    int src_offset_;
    int dst_offset_;

    int tile_idx_{-1};

    // __device__ GmemIterator(
    //     const T** block_ptrs, BlockSeqLen block_seqlen, Array<int, 2> local_offsets, T* smem, int warp_id, int lane_id):
    //     block_ptrs_(block_ptrs), smem_(smem), block_seqlen_{block_seqlen}, local_offsets_(local_offsets)
    // {
    //     int2 offsets = Map::get_offset(warp_id, lane_id);
    //     // printf("tid=%3d, offset x=%3d, y=%3d\n", (int)threadIdx.x, offsets.x, offsets.y);
    //     init_offset_ = offsets.x + offsets.y * Map::kDimC;
    //     // dst_offset_  = init_offset_;
    //     dst_offset_ = offsets.x + offsets.y * (Map::kDimC + SMEM_PAD);
    // }

    __device__
    GmemIterator(const T** block_ptrs, BlockSeqLen block_seqlen, int local_offset, T* smem, int warp_id, int
    lane_id):
        block_ptrs_(block_ptrs), smem_(smem), block_seqlen_{block_seqlen}, local_offset_{local_offset}
    {
        int2 offsets = Map::get_offset(warp_id, lane_id);
        init_offset_ = offsets.x + offsets.y * Map::kDimC;
        dst_offset_  = offsets.x + offsets.y * (Map::kDimC + SMEM_PAD);
    }

    __device__ void AdjustBlockTileIdx(int tile_idx)  // Interprept step as (block_idx, local_tile_idx)
    {
        // const int tiles_per_block = block_seqlen_ / Map::kDimS;
        // // Has different block idx
        // if ((tile_idx_ ^ tile_idx) & ~(tiles_per_block - 1)) {
        //     // Get new block idx
        //     const int block_idx = tile_idx / tiles_per_block;
        //     const int local_idx = tile_idx % tiles_per_block;

        //     // Load base addresses for new block
        //     block_ = block_ptrs_[block_idx];

        //     // Move to specified tile
        //     src_offset_ = init_offset_ + local_idx * Map::kDimS * Map::kDimC;
        // }
        // else {
        //     // Move within the same block
        //     src_offset_ += (tile_idx - tile_idx_) * Map::kDimS * Map::kDimC;
        // }
        // tile_idx_ = tile_idx;

        const int block_idx = tile_idx / (block_seqlen_ / Map::kDimS);
        const int local_idx = tile_idx % (block_seqlen_ / Map::kDimS);
        block_              = block_ptrs_[block_idx];
        src_offset_         = init_offset_ + local_idx * Map::kDimS * Map::kDimC;
    }

    // Pass `I` by `std::integral_constant` to avoid explict template keyword at the call site
    template<bool is_residue, int I>
    __device__ void PrefetchStage(std::integral_constant<int, I>, std::bool_constant<is_residue>, int max_s)
    {
        // auto      src      = block_ + local_offsets_[I] + src_offset_;
        // auto      dst      = smem_ + I * kSizePerTile;
        auto src = block_ + local_offset_ + src_offset_;
        auto dst = smem_;
        const int offset_s = Map::get_offset(threadIdx.x / WARP_SIZE, threadIdx.x % WARP_SIZE).y;
        PRAGMA_UNROLL
        for (int s = 0; s < Map::kIterS; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < Map::kIterC; ++c) {
                if constexpr (is_residue) {
                    const bool pred = offset_s + s * Map::kDeltaS < max_s;
                    Copy(dst + swizzle_(dst_offset_ + s * Map::kDeltaS * (Map::kDimC + SMEM_PAD) + c * Map::kDeltaC),
                         src + s * Map::kDeltaS * Map::kDimC + c * Map::kDeltaC,
                         pred);
                }
                else {
                    Copy(dst + swizzle_(dst_offset_ + s * Map::kDeltaS * (Map::kDimC + SMEM_PAD) + c * Map::kDeltaC),
                         src + s * Map::kDeltaS * Map::kDimC + c * Map::kDeltaC,
                         true);
                }
            }
        }
    }

    template<int I>
    __device__ void ClearSmem(std::integral_constant<int, I>)
    {
        auto dst = smem_ + I * kSizePerTile;
        PRAGMA_UNROLL
        for (int s = 0; s < Map::kIterS; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < Map::kIterC; ++c) {
                clear((Array<T, Map::kAccessC>&)
                          dst[dst_offset_ + s * Map::kDeltaS * (Map::kDimC + SMEM_PAD) + c * Map::kDeltaC]);
            }
        }
    }

    static __device__ void CpAsync(T* __restrict__ dst, const T* __restrict__ src, bool mask)
    {
        const int     smem_int_ptr = cast_smem_ptr_to_uint(dst);
        constexpr int cp_size      = sizeof(AccessType);
#if TURBOMIND_ARCH_SM80
        // clang-format off
        asm volatile("{\n"
                     "  .reg .pred p;\n"
                     "  setp.ne.b32 p, %0, 0;\n"
                     "  @p cp.async.cg.shared.global " L2_CACHEHINT(128) " [%1], [%2], %3;\n"
                     "}\n" ::"r"((int)mask),
                     "r"(smem_int_ptr),
                     "l"(src),
                     "n"(cp_size));
        // clang-format on
        // " L2_CACHEHINT(128) "
#else
        assert(TURBOMIND_ARCH_SM80);
#endif
    }

    static __device__ void CpLdg(T* __restrict__ dst, const T* __restrict__ src, bool mask)
    {
        if (mask) {
            Ldg(*(AccessType*)dst, src);
        }
    }

    __device__ void Copy(T* __restrict__ dst, const T* __restrict__ src, bool mask)
    {
        if constexpr (TURBOMIND_ARCH_SM80) {
            CpAsync(dst, src, mask);
        }
        else {
            CpLdg(dst, src, mask);
        }
    }
};

template<class T, int DIMS, class Swizzle>
struct SmemIterator {
    static_assert(sizeof(T) == 2);
    static constexpr int kElemSize = sizeof(T);
    uint32_t             smem_int_ptr_;
    Swizzle              swizzle_;

    __device__ SmemIterator(const T* smem): smem_int_ptr_{cast_smem_ptr_to_uint(smem)} {}

    template<int ITER_N>
    __device__ void LoadK(Array<T, 4> (&frag_K)[ITER_N], int k)
    {
        static_assert(ITER_N % 2 == 0);
        const int lane_id       = threadIdx.x % WARP_SIZE;
        const int group_id      = lane_id / 16;
        const int group_lane_id = lane_id % 16;
        PRAGMA_UNROLL
        for (int n = 0; n < ITER_N / 2; ++n) {  // Load (s16,d16) tiles
            auto&     r   = (Array<uint32_t, 4>&)frag_K[n * 2];
            const int s   = n * 16 + group_lane_id % 8 + group_id * 8;
            const int c   = k * 16 + group_lane_id / 8 * 8;
            const int idx = swizzle_(s * (DIMS + SMEM_PAD) + c);
            ldmatrix_m8n8_x4_b16(r[0], r[1], r[2], r[3], smem_int_ptr_ + kElemSize * idx);
        }
    }

    template<int ITER_M>
    __device__ void LoadQ(Array<T, 8> (&frag_Q)[ITER_M], int k)
    {
        constexpr int WARP_Q  = 16;
        const int     lane_id = threadIdx.x % WARP_SIZE;
        const int     warp_id = threadIdx.x / WARP_SIZE;

        PRAGMA_UNROLL
        for (int m = 0; m < ITER_M; ++m) {
            auto&     Q   = (Array<uint32_t, 4>&)frag_Q[m];
            const int mm  = m * 16 + lane_id % 16 + warp_id * WARP_Q;
            const int kk  = k * 16 + lane_id / 16 * 8;
            const int idx = swizzle_(mm * (DIMS + SMEM_PAD) + kk);
            ldmatrix_m8n8_x4_b16(Q[0], Q[1], Q[2], Q[3], smem_int_ptr_ + kElemSize * idx);
        }
    }

    template<int ITER_N>  // 16
    __device__ void LoadV(Array<T, 4> (&frag_V)[ITER_N], int k)
    {
        // __syncthreads();
        static_assert(ITER_N % 2 == 0);
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int n = 0; n < ITER_N / 2; ++n) {  // Load (d16,s16) tiles
            auto&     r   = (Array<uint32_t, 4>&)frag_V[n * 2];
            const int kk  = k * 16 + lane_id % 16;      // s
            const int nn  = n * 16 + lane_id / 16 * 8;  // d
            const int idx = swizzle_(kk * (DIMS + SMEM_PAD) + nn);
            ldsm_x4_trans(r[0], r[1], r[2], r[3], smem_int_ptr_ + kElemSize * idx);
        }
    }
};

}  // namespace turbomind
