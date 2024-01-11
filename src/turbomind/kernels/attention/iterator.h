// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "../gemm_s_f16/common.h"
#include "array_ops.h"

namespace turbomind {

template<class T, class Map, class BlockSeqLen, class SmemLayout>
struct BaseGmemIterator {
    using ElementType = T;
    using AccessType  = Array<T, Map::kAccessC>;

    static constexpr int kElementSize = sizeof(ElementType);
    static constexpr int kAccessSize  = sizeof(AccessType);
    static constexpr int kIterCount   = Map::kIterS * Map::kIterC;

    using Fragment = Array<T, Map::kAccessC>[Map::kIterS][Map::kIterC];

    const T** block_ptrs_;
    const T*  block_;

    T*       smem_;
    uint32_t smem_int_ptr_;

    BlockSeqLen block_seqlen_;

    const int local_offset_;

    int init_offset_;
    int dst_offset_;

    __device__ BaseGmemIterator(
        const T** block_ptrs, BlockSeqLen block_seqlen, int local_offset, T* smem, int warp_id, int lane_id):
        block_ptrs_(block_ptrs),
        smem_(smem),
        smem_int_ptr_(cast_smem_ptr_to_uint(smem)),
        block_seqlen_{block_seqlen},
        local_offset_{local_offset}
    {
        int2 offsets = Map::get_offset(warp_id, lane_id);
        init_offset_ = offsets.x + offsets.y * Map::kDimC;
        dst_offset_  = offsets.x + offsets.y * SmemLayout::kStride;
    }

    __device__ void AdjustBlockTileIdx(int tile_idx)  // Interprept step as (block_idx, local_tile_idx)
    {
        // const int block_idx = tile_idx / (block_seqlen_ / Map::kDimS);
        // const int local_idx = tile_idx % (block_seqlen_ / Map::kDimS);
        // block_ = block_ptrs_[block_idx] + local_offset_ + local_idx * Map::kDimS * Map::kDimC + init_offset_;

        // block_ = block_ptrs_[tile_idx] + local_offset_ + init_offset_;
        block_ = block_ptrs_[tile_idx];
    }

    __device__ void ClearSmem(int offset)
    {
        auto dst = smem_;
        PRAGMA_UNROLL
        for (int s = 0; s < Map::kIterS; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < Map::kIterC; ++c) {
                constexpr Array<T, Map::kAccessC> kZeros{};
                Store(&dst[dst_offset_ + offset + s * Map::kDeltaS * SmemLayout::kStride + c * Map::kDeltaC], kZeros);
            }
        }
    }
};

template<int Stride, class _Swizzle>
struct SmemLayout {
    static constexpr int kStride = Stride;

    using Swizzle = _Swizzle;

    __forceinline__ __device__ static int offset(int s, int c)
    {
        return s * kStride + c;
    }
    __forceinline__ __device__ static int swizzle(int s, int c)
    {
        return Swizzle{}(s * kStride + c);
    }
    __forceinline__ __device__ static int swizzle(int offset)
    {
        return Swizzle{}(offset);
    }
};

template<class T, class Layout>
struct BaseSmemIterator {
    static constexpr int kElemSize = sizeof(T);
    const T*             smem_;
    uint32_t             smem_int_ptr_;

    __device__ explicit BaseSmemIterator(const T* smem): smem_{smem}, smem_int_ptr_{cast_smem_ptr_to_uint(smem)} {}

    __forceinline__ __device__ const T* ptr(int s, int c)
    {
        return &smem_[Layout::offset(s, c)];
    }

    __forceinline__ __device__ const T* ptr(int offset)
    {
        return &smem_[offset];
    }

    __forceinline__ __device__ uint32_t uint_ptr(int s, int c)
    {
        return smem_int_ptr_ + kElemSize * Layout::offset(s, c);
    }

    __forceinline__ __device__ uint32_t uint_ptr(int offset)
    {
        return smem_int_ptr_ + kElemSize * offset;
    }

    __forceinline__ __device__ const T* swizzle_ptr(int s, int c)
    {
        return &smem_[Layout::swizzle(s, c)];
    }

    __forceinline__ __device__ const T* swizzle_ptr(int offset)
    {
        return &smem_[Layout::swizzle(offset)];
    }

    __forceinline__ __device__ uint32_t swizzle_uint_ptr(int s, int c)
    {
        return smem_int_ptr_ + kElemSize * Layout::swizzle(s, c);
    }

    __forceinline__ __device__ uint32_t swizzle_uint_ptr(int offset)
    {
        return smem_int_ptr_ + kElemSize * Layout::swizzle(offset);
    }
};

template<class T>
struct NullSmemIter {
    __device__ explicit NullSmemIter(const T*){};
};

struct Identity {
    template<class T>
    __device__ T operator()(T offset)
    {
        return offset;
    }

    template<int D>
    __device__ int AdvanceS(int offset, int s0, int s1)
    {
        return offset;
    }
};

template<class BlockSeqLen, int CTA_S>
struct Block {};

template<int BLK_S, int CTA_S>
struct Block<std::integral_constant<int, BLK_S>, CTA_S> {

    __device__ Block(int) {}

    __device__ void GetTile(int tile_id, int& block_id, int& local_id)
    {
        block_id = tile_id / (BLK_S / CTA_S);
        local_id = tile_id % (BLK_S / CTA_S);
    }

    __device__ void NextTile(int& block_id, int& local_id)
    {
        --local_id;
        if (local_id < 0) {
            local_id += BLK_S / CTA_S;
            block_id -= 1;
        }
    }
};

template<int CTA_S>
struct Block<int, CTA_S> {

    // int block_seqlen_;
    const int tiles_per_block_;

    __device__ Block(int block_seqlen): tiles_per_block_{block_seqlen / CTA_S} {}

    __device__ void GetTile(int tile_id, int& block_id, int& local_id)
    {

        block_id = tile_id >> (31 - __clz(tiles_per_block_));
        local_id = tile_id & (tiles_per_block_ - 1);
    }

    __device__ void NextTile(int& block_id, int& local_id)
    {
        --local_id;
        if (local_id < 0) {
            local_id += tiles_per_block_;
            block_id -= 1;
        }
    }
};

}  // namespace turbomind
