// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "../gemm_s_f16/common.h"
#include "array_ops.h"

namespace turbomind {

#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 4)
#define L2_CACHEHINT(size) ".L2::" #size "B"
#else
#define L2_CACHEHINT(size)
#endif

template<class T, class TMap, int Stages>
struct GmemIterator {
    using ElementType = T;
    using AccessType  = Array<T, TMap::kAccessC>;

    static constexpr int kElementSize = sizeof(ElementType);
    static constexpr int kAccessSize  = sizeof(AccessType);

    static constexpr int kSizePerTile  = TMap::kDimS * TMap::kDimC;
    static constexpr int kSmemByteSize = kElementSize * Stages * kSizePerTile;

    struct __align__(sizeof(AccessType)) SharedStorage
    {
        T smem_[Stages][kSizePerTile];
    };

    static constexpr int kIterCount = TMap::kIterS * TMap::kIterC;

    static constexpr int kStepC = TMap::kDeltaC;
    static constexpr int kStepS = TMap::kDeltaS * TMap::kDimC - TMap::kIterC * kStepC;
    static constexpr int kStepK = TMap::kDimS * TMap::kDimC - TMap::kIterS * TMap::kDeltaS * TMap::kDimC;

    const T** __restrict__ block_ptr_;
    const T* __restrict__ src_;
    T* __restrict__ smem_;

    const int block_len_;
    const int block_local_offset_;
    const int block_mask_;
    const int max_iter_;

    int init_offset_;
    int src_offset_;
    int dst_offset_;

    int  iter_   = 0;
    int  iter_c_ = 0;
    bool mask_   = true;

    __device__ GmemIterator(
        const T** block_ptr, int block_len, int block_local_offset, T* smem, int max_iter, int warp_id, int lane_id):
        block_ptr_(block_ptr),
        smem_(smem),
        block_len_(block_len),
        block_local_offset_(block_local_offset),
        block_mask_(block_len / TMap::kDimS - 1),
        max_iter_(max_iter)
    {
        int2 offsets = TMap::get_offset(warp_id, lane_id);
        src_         = *block_ptr_++;
        init_offset_ = offsets.x + offsets.y * TMap::kDimC;
        src_offset_  = init_offset_ + block_local_offset_;
        dst_offset_  = init_offset_;
    }

    __device__ GmemIterator& operator++()
    {
        src_offset_ += kStepC;
        dst_offset_ += kStepC;
        ++iter_c_;
        if (iter_c_ < TMap::kIterC) {
            return *this;
        }

        iter_c_ = 0;
        src_offset_ += kStepS;
        dst_offset_ += kStepS;
        return *this;
    }

    __device__ void AdvanceStage()
    {
        ++iter_;
        if (iter_ == max_iter_) {
            mask_ = false;
        }

        if (!mask_) {
            return;
        }

        if ((iter_ & block_mask_) == 0) {
            src_        = *block_ptr_++;
            src_offset_ = init_offset_ + block_local_offset_;
        }
        else {
            src_offset_ += kStepK;
        }

        dst_offset_ += kStepK;
        if (dst_offset_ >= Stages * kSizePerTile) {
            dst_offset_ -= Stages * kSizePerTile;
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
                     "  @p cp.async.ca.shared.global" L2_CACHEHINT(128) " [%1], [%2], %3;\n"
                     "}\n" ::"r"((int)mask),
                     "r"(smem_int_ptr),
                     "l"(src),
                     "n"(cp_size));
        // clang-format on
#else
        assert(TURBOMIND_ARCH_SM80);
#endif
    }

    static __device__ void Copy(T* __restrict__ dst, const T* __restrict__ src, bool mask)
    {
        if (mask) {
            Ldg(*(AccessType*)dst, src);
        }
    }

    __device__ void PrefetchStage()
    {
        PRAGMA_UNROLL
        for (int i = 0; i < kIterCount; ++i) {
            Prefetch(mask_);
            ++(*this);
        }
        AdvanceStage();
    }

    __device__ void PrefetchBatch(int batch_idx, int batch_size)
    {
        PRAGMA_UNROLL
        for (int i = 0; i < batch_size; ++i) {
            if (batch_idx * batch_size + i < kIterCount) {
                Prefetch(mask_);
                ++(*this);
            }
        }
    }

    __device__ void Prefetch(bool mask)
    {
        if constexpr (TURBOMIND_ARCH_SM80) {
            CpAsync(&smem_[dst_offset_], &src_[src_offset_], mask);
        }
        else {
            Copy(&smem_[dst_offset_], &src_[src_offset_], mask);
        }
    }
};

template<class T, class TMap, int Stages>
struct SmemIterator {
    using ElementType = T;
    using AccessType  = Array<T, TMap::kAccessC>;

    static constexpr int kElementSize = sizeof(ElementType);
    static constexpr int kAccessSize  = sizeof(AccessType);

    static constexpr int kSizePerTile  = TMap::kDimS * TMap::kDimC;
    static constexpr int kSmemByteSize = kElementSize * Stages * kSizePerTile;

    static constexpr int kStepS = TMap::kDeltaS * TMap::kDimC;
    static constexpr int kStepK = TMap::kDimS * TMap::kDimC - TMap::kIterS * kStepS;

    const T* smem_;
    int      smem_offset_;

    __device__ SmemIterator(const T* smem, int warp_id, int lane_id): smem_(smem)
    {
        int2 offsets = TMap::get_offset(warp_id, lane_id);
        smem_offset_ = offsets.x + offsets.y * TMap::kDimC;
    }

    __device__ void Load(AccessType (&frag)[TMap::kIterC])
    {
        PRAGMA_UNROLL
        for (int vi = 0; vi < TMap::kIterC; ++vi) {
            Lds(frag[vi], smem_ + smem_offset_ + vi * TMap::kDeltaC);
        }
        smem_offset_ += TMap::kDeltaS * TMap::kDimC;
    }

    __device__ void AdvanceStage()
    {
        smem_offset_ += kStepK;
        if (smem_offset_ >= Stages * kSizePerTile) {
            smem_offset_ -= Stages * kSizePerTile;
        }
    }
};

}  // namespace turbomind
