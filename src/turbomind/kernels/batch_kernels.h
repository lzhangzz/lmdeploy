// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/core/core.h"

#include <cstdint>

#include <cuda_runtime.h>
namespace turbomind {

// ABCDe            ABCDe     e
// ABCDEFGHIJk      ABCDEFGHIJk
// ABCDEFGHi    ->  ABCDEFGHi i
// ABCDEFGh         ABCDEFGh  h
// ABCd             ABCd      d
void invokePadLastTokenIds(
    int* token_ids, const int* context_length, int max_context_len, int batch_size, cudaStream_t stream);

void invokeCastFloat2D(const core::Tensor& src, core::Tensor& dst, cudaStream_t stream);

void CollectHiddenStates(const Tensor& src, const Buffer_<int>& idxs, Ref<Tensor> dst, cudaStream_t st);

void BatchPrefixSum(const int** srcs, const int* ns, int** dsts, int count, cudaStream_t st);

inline void PrefixSum(const int* src, int n, int* dst, cudaStream_t st)
{
    return BatchPrefixSum(&src, &n, &dst, 1, st);
}

void invokeAppendOneTokenAndAdvanceSequence(
    int* const* token_ids_ptrs, const int* selected_tokens, int* sequence_length, int batch_size, cudaStream_t stream);

// Upper bound for attention DP size supported by `invokeBuildTokenMask` (per-rank token bases
// are passed to the kernel by value).
inline constexpr int kMaxAttnDPSize = 64;

// Build the global per-token validity mask: every forward token is valid except the token
// ranges (`q_offsets`) of finished sequences. With attn_dp_size > 1, `finished`/`q_offsets`
// address the per-rank metadata blocks gathered across attention DP ranks, `rank_stride`
// bytes apart, and `token_base[r]` is rank r's base offset within the global mask; otherwise
// the local arrays are passed directly with `rank_stride == 0` and `token_base[0] == 0`.
void invokeBuildTokenMask(bool*        token_mask,
                          const bool*  finished,
                          const int*   q_offsets,
                          size_t       rank_stride,
                          const int*   token_base,  // [attn_dp_size], host
                          int          attn_dp_size,
                          int          batch_size,  // per-rank slots to scan
                          int          global_token_num,
                          cudaStream_t stream);

}  // namespace turbomind
