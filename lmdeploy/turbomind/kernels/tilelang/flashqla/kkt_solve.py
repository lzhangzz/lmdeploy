# MIT License
#
# Copyright (c) 2026 Qwen, Alibaba
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Adapted from FlashQLA `flash_qla/ops/gated_delta_rule/chunk/blackwell_sm120/kkt_solve.py`
# (https://github.com/QwenLM/FlashQLA, revision a14dbf3474514e1674526431e960325f72081e9f)
# for TurboMind's ordinary prefill.

import tilelang
import tilelang.language as T


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def kkt_prefill(H, Hg):
    data_batch_size = T.dynamic('data_batch_size')
    k_s0 = T.dynamic('k_s0', 'int64')
    k_s1 = T.dynamic('k_s1', 'int64')
    real_batch_size = T.dynamic('real_batch_size')
    num_tokens = T.dynamic('num_tokens')
    block_S = 32
    k_shape = (data_batch_size, num_tokens, Hg, 128)
    a_shape = (data_batch_size, num_tokens, H, 32)
    b_shape = (data_batch_size, num_tokens, H)

    @T.macro
    def kernel_body_32(bb, bc, bh, bhg, batch_idx, chunk_idx, seq_start_idx, seq_end_idx, k, b, a):
        left = seq_start_idx + chunk_idx * block_S
        right = left + block_S
        k_shared = T.alloc_shared((block_S, 128), dtype='bfloat16')
        b_shared = T.alloc_shared(block_S, dtype='float32', scope='shared')
        a32_fragment = T.alloc_fragment((block_S, block_S), dtype='float32')
        a16i_row = T.alloc_fragment((2, 16), dtype='float32')
        a16i_sum = T.alloc_fragment((2, 16), dtype='float32')
        a16i_shared = T.alloc_shared((2, 17, 16), dtype='float32')
        a16o_shared = T.alloc_shared((1, 17, 16), dtype='float32')
        a16o_fragment = T.alloc_fragment((1, 16, 16), dtype='float32')
        a32_shared = T.alloc_shared((block_S, block_S), dtype='float32')
        T.annotate_layout(
            {
                a16i_shared: tilelang.layout.make_linear_layout(a16i_shared),
                a16o_shared: tilelang.layout.make_linear_layout(a16o_shared),
            }
        )
        if right <= seq_end_idx:
            T.async_copy(k[bb, left:right, bhg, 0:128], k_shared)
        else:
            for j_s, j_k in T.Parallel(block_S, 128):
                if left + j_s < seq_end_idx:
                    k_shared[j_s, j_k] = k[bb, left + j_s, bhg, j_k]
                else:
                    k_shared[j_s, j_k] = 0
        if right <= seq_end_idx:
            for j_s in T.Parallel(block_S):
                b_shared[j_s] = b[bb, left + j_s, bh]
        else:
            for j_s in T.Parallel(block_S):
                if left + j_s < seq_end_idx:
                    b_shared[j_s] = b[bb, left + j_s, bh]
                else:
                    b_shared[j_s] = 0
        if right <= seq_end_idx:
            T.ptx_wait_group(0)
        T.gemm(k_shared, k_shared, a32_fragment, transpose_B=True, clear_accum=True)
        for j_s, j_t in T.Parallel(block_S, block_S):
            a32_fragment[j_s, j_t] *= b_shared[j_s]
        for j_s, j_t in T.Parallel(block_S, block_S):
            if j_s < j_t:
                a32_fragment[j_s, j_t] = 0
            elif j_s == j_t:
                a32_fragment[j_s, j_t] = 1
        for j_s, j_t in T.Parallel(block_S, block_S):
            if j_s // 16 == j_t // 16 + 1:
                a16o_shared[j_s // 32, j_s % 16, j_t % 16] = -a32_fragment[j_s, j_t]
            elif j_s // 16 == j_t // 16:
                a16i_shared[j_s // 16, j_s % 16, j_t % 16] = a32_fragment[j_s, j_t]
        for k_s in T.unroll(1, 16):
            for j_s, k_t in T.Parallel(2, 16):
                if k_t < k_s:
                    a16i_row[j_s, k_t] = a16i_shared[j_s, k_s, k_t]
            T.clear(a16i_sum)
            for k_r in T.unroll(k_s):
                for j_s, k_t in T.Parallel(2, 16):
                    a16i_sum[j_s, k_t] -= a16i_shared[j_s, k_r, k_t] * a16i_row[j_s, k_r]
            for j_s, k_t in T.Parallel(2, 16):
                if k_t < k_s:
                    a16i_shared[j_s, k_s, k_t] = a16i_sum[j_s, k_t]
            T.sync_threads()
        T.clear(a16o_fragment)
        for k_r in T.unroll(16):
            for j_s, k_s, k_t in T.Parallel(1, 16, 16):
                a16o_fragment[j_s, k_s, k_t] += a16i_shared[j_s * 2 + 1, k_s, k_r] * a16o_shared[j_s, k_r, k_t]
        for j_s, k_s, k_t in T.Parallel(1, 16, 16):
            a16o_shared[j_s, k_t, k_s] = a16o_fragment[j_s, k_s, k_t]
        T.clear(a16o_fragment)
        for k_r in T.unroll(16):
            for j_s, k_s, k_t in T.Parallel(1, 16, 16):
                a16o_fragment[j_s, k_s, k_t] += a16o_shared[j_s, k_r, k_s] * a16i_shared[j_s * 2, k_r, k_t]
        T.copy(a16o_fragment, a16o_shared[:, 0:16, 0:16])
        for k_s, k_t in T.Parallel(16, 16):
            a32_shared[k_s, k_t] = a16i_shared[0, k_s, k_t]
        for k_s, k_t in T.Parallel(16, 16):
            a32_shared[k_s, 16 + k_t] = 0
        for k_s, k_t in T.Parallel(16, 16):
            a32_shared[16 + k_s, k_t] = a16o_fragment[0, k_s, k_t]
        for k_s, k_t in T.Parallel(16, 16):
            a32_shared[16 + k_s, 16 + k_t] = a16i_shared[1, k_s, k_t]
        if right <= seq_end_idx:
            T.copy(a32_shared, a[bb, left:right, bh, 0:block_S])
        else:
            for j_s, j_t in T.Parallel(block_S, block_S):
                if left + j_s < seq_end_idx:
                    a[bb, left + j_s, bh, j_t] = a32_shared[j_s, j_t]

    @T.prim_func
    def tilelang_kkt_solve_kernel(
        k: T.StridedTensor(k_shape, (k_s0, k_s1, 128, 1), 'bfloat16'),
        b: T.Tensor(b_shape, dtype='float32'),
        cu_seqlens: T.Tensor([real_batch_size + 1], dtype='int32'),
        a: T.Tensor(a_shape, dtype='bfloat16'),
        num_chunks: T.int32,
    ):
        T.assume(k_s0 % 8 == 0)
        T.assume(k_s1 % 8 == 0)
        with T.Kernel(num_chunks * H, threads=128) as (bch,):
            bc, bh = (bch // H, bch % H)
            bhg = bh // (H // Hg)
            batch_idx = T.alloc_var('int32')
            chunk_idx = T.alloc_var('int32')
            seq_start_idx = T.alloc_var('int32')
            seq_end_idx = T.alloc_var('int32')
            bb = 0
            batch_idx = -1
            chunk_idx = bc
            for candidate in T.serial(real_batch_size):
                if batch_idx == -1:
                    count = T.ceildiv(cu_seqlens[candidate + 1] - cu_seqlens[candidate], block_S)
                    if chunk_idx < count:
                        batch_idx = candidate
                    else:
                        chunk_idx -= count
            seq_start_idx = cu_seqlens[batch_idx]
            seq_end_idx = cu_seqlens[batch_idx + 1]
            kernel_body_32(bb, bc, bh, bhg, batch_idx, chunk_idx, seq_start_idx, seq_end_idx, k, b, a)

    return tilelang_kkt_solve_kernel
