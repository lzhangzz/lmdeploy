// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "../gemm_s_f16/common.h"
#include "array_ops.h"
#include <type_traits>

namespace turbomind {

template<class T, class Map, class SmemLayout>
struct BaseGmemIterator {
    using ElementType = T;
    using AccessType  = Array<T, Map::kAccessC>;

    static constexpr int kElementSize = sizeof(ElementType);
    static constexpr int kAccessSize  = sizeof(AccessType);
    static constexpr int kIterCount   = Map::kIterS * Map::kIterC;

    using Fragment = Array<T, Map::kAccessC>[Map::kIterS][Map::kIterC];

    T*       smem_;
    uint32_t smem_int_ptr_;

    const int local_offset_;

    int init_offset_;
    int dst_offset_;

    __device__ BaseGmemIterator(int local_offset, int warp_id, int lane_id): local_offset_{local_offset}
    {
        int2 offsets = Map::get_offset(warp_id, lane_id);
        init_offset_ = offsets.x + offsets.y * Map::kDimC;
        dst_offset_  = offsets.x + offsets.y * SmemLayout::kStride;
    }

    __device__ void SetSmem(T* smem)
    {
        smem_         = smem;
        smem_int_ptr_ = cast_smem_ptr_to_uint(smem);
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

template<class T, int CTA_S, class BlockSeqLen>
struct Block {

    const int tiles_per_block_;
    const T** block_ptrs_;

    int block_id_;

    const T* block;
    int      local_id;

    __device__ Block(const T** block_ptrs, BlockSeqLen block_seqlen):
        block_ptrs_{block_ptrs}, tiles_per_block_{block_seqlen / CTA_S}
    {
    }

    __device__ void SetTile(int tile_id)
    {
        if constexpr (std::is_integral_v<BlockSeqLen>) {
            block_id_ = tile_id >> (31 - __clz(tiles_per_block_));  // this is some how faster than `__ffs`
            local_id  = tile_id & (tiles_per_block_ - 1);
        }
        else {
            block_id_ = tile_id / tiles_per_block_;
            local_id  = tile_id % tiles_per_block_;
        }
        block = block_ptrs_[block_id_];
    }

    __device__ void Advance()
    {
        --local_id;
        if (local_id < 0) {
            local_id += tiles_per_block_;
            block_id_ -= 1;
        }
        if (block_id_ >= 0) {
            block = block_ptrs_[block_id_];
        }
    }
};

}  // namespace turbomind
