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

# Adapted from FlashQLA `flash_qla/ops/gated_delta_rule/chunk/blackwell_sm120/fused_fwd.py`
# (https://github.com/QwenLM/FlashQLA, revision a14dbf3474514e1674526431e960325f72081e9f)
# for TurboMind's ordinary prefill.

import tilelang
import tilelang.language as T


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def fused_prefill(H, Hg, state_dtype):
    batch_size = T.dynamic('batch_size')
    num_tokens = T.dynamic('num_tokens')
    head_groups = T.dynamic('head_groups')
    state_bytes = 4 if state_dtype == 'float32' else 2
    q_s0 = T.dynamic('q_s0', 'int64')
    q_s1 = T.dynamic('q_s1', 'int64')
    k_s0 = T.dynamic('k_s0', 'int64')
    k_s1 = T.dynamic('k_s1', 'int64')
    v_s0 = T.dynamic('v_s0', 'int64')
    v_s1 = T.dynamic('v_s1', 'int64')
    o_s0 = T.dynamic('o_s0', 'int64')
    o_s1 = T.dynamic('o_s1', 'int64')
    block_S = 32
    q_shape = (1, num_tokens, Hg, 128)
    k_shape = (1, num_tokens, Hg, 128)
    v_shape = (1, num_tokens, H, 128)
    o_shape = (1, num_tokens, H, 128)
    a_shape = (1, num_tokens, H, 32)
    g_shape = (1, num_tokens, H)
    b_shape = (1, num_tokens, H)

    @T.prim_func
    def tilelang_fused_chunk_gdr_fwd_kernel(
        q: T.StridedTensor(q_shape, (q_s0, q_s1, 128, 1), 'bfloat16'),
        k: T.StridedTensor(k_shape, (k_s0, k_s1, 128, 1), 'bfloat16'),
        v: T.StridedTensor(v_shape, (v_s0, v_s1, 128, 1), 'bfloat16'),
        a: T.Tensor(a_shape, 'bfloat16'),
        g: T.Tensor(g_shape, 'float32'),
        b: T.Tensor(b_shape, 'float32'),
        state_ptrs: T.Tensor((batch_size, head_groups), T.ptr),
        cu_seqlens: T.Tensor((batch_size + 1,), 'int32'),
        finished: T.Tensor((batch_size,), 'bool'),
        o: T.StridedTensor(o_shape, (o_s0, o_s1, 128, 1), 'bfloat16'),
        state_layer_offset: T.int64,
        heads_per_block: T.int32,
    ):
        with T.Kernel(T.ceildiv(128, 64) * batch_size * H, threads=512) as (bbhv,):
            bbh, bv = (bbhv // T.ceildiv(128, 64), bbhv % T.ceildiv(128, 64))
            bb, bh = (bbh // H, bbh % H)
            bhg = bh // (H // Hg)
            DV_start = bv * 64
            DV_end = (bv + 1) * 64
            batch_idx = T.alloc_var('int32')
            seq_start_idx = T.alloc_var('int32')
            seq_end_idx = T.alloc_var('int32')
            seq_split_idx = T.alloc_var('int32')
            batch_idx = 0
            seq_start_idx = cu_seqlens[bb]
            seq_end_idx = cu_seqlens[bb + 1]
            num_iters = T.alloc_var('int32')
            num_unmasked_iters = T.alloc_var('int32')
            num_iters = T.ceildiv(seq_end_idx - seq_start_idx, block_S)
            num_unmasked_iters = (seq_end_idx - seq_start_idx) // block_S
            q_shared = T.alloc_shared((2, block_S, 128), dtype='bfloat16')
            k_shared = T.alloc_shared((2, block_S, 128), dtype='bfloat16')
            v_shared = T.alloc_shared((2, block_S, 64), dtype='bfloat16')
            a_shared = T.alloc_shared((2, block_S, block_S), dtype='bfloat16')
            g_shared = T.alloc_shared((2, block_S), dtype='float32', scope='shared')
            b_shared = T.alloc_shared((2, block_S), dtype='float32', scope='shared')
            o_shared = T.alloc_shared((block_S, 64), dtype='bfloat16')
            h_shared = T.alloc_shared((128, 64), dtype='bfloat16')
            vd_shared = T.alloc_shared((block_S, 64), dtype='bfloat16')
            vn_shared = T.alloc_shared((block_S, 64), dtype='bfloat16')
            p_shared = T.alloc_shared((block_S, block_S), dtype='bfloat16')
            g_exp_shared = T.alloc_shared(block_S, dtype='float32', scope='shared')
            g_rev_exp_shared = T.alloc_shared(block_S, dtype='float32', scope='shared')
            h_fragment = T.alloc_fragment((128, 64), dtype='float32')
            o_fragment = T.alloc_fragment((block_S, 64), dtype='float32')
            v_fragment = T.alloc_fragment((block_S, 64), dtype='float32')
            u_fragment = T.alloc_fragment((block_S, 64), dtype='float32')
            p_fragment = T.alloc_fragment((block_S, block_S), dtype='float32')
            a_fragment = T.alloc_fragment((block_S, block_S), dtype='float32')
            g_fragment = T.alloc_fragment((block_S, block_S), dtype='float32')
            g_last_local = T.alloc_local(1, dtype='float32')
            data_is_ready = T.alloc_barrier(arrive_count=[96] * 2)
            data_is_free = T.alloc_barrier(arrive_count=[384] * 2)
            bar_o = T.alloc_barrier(arrive_count=128)
            bar_0 = T.alloc_barrier(arrive_count=416)
            bar_1 = T.alloc_barrier(arrive_count=256)
            _bar_2 = T.alloc_barrier(arrive_count=128)
            bar_3 = T.alloc_barrier(arrive_count=128)
            bar_4 = T.alloc_barrier(arrive_count=128)
            bar_5 = T.alloc_barrier(arrive_count=416)
            T.use_swizzle(10)
            tx = T.get_thread_binding()
            PRODUCER_NREG = 32
            CONSUMER_V_NREG = 128
            CONSUMER_S_NREG = 128
            CONSUMER_O_NREG = 128
            if tx < 128:
                T.set_max_nreg(CONSUMER_S_NREG, 1)
                head_group = bh // heads_per_block
                local_head = bh % heads_per_block
                element_offset = state_layer_offset + T.cast(local_head, 'int64') * 128 * 128
                address = state_ptrs[bb, head_group] + element_offset * state_bytes
                state = T.make_tensor_from_addr(address, (128, 128), dtype=state_dtype, strides=(128, 1))
                T.copy(state[:, DV_start:DV_end], h_fragment)
                for i_s in T.serial(num_iters):
                    T.barrier_wait(data_is_ready[i_s % 2], (i_s // 2 + 0) % 2)
                    T.barrier_arrive(bar_0)
                    T.barrier_wait(bar_0, i_s % 2)
                    T.copy(h_fragment, h_shared)
                    T.barrier_arrive(bar_1)
                    T.barrier_wait(bar_1, i_s % 2)
                    g_last_local[0] = g_exp_shared[block_S - 1]
                    for j_k, j_v in T.Parallel(128, 64):
                        h_fragment[j_k, j_v] *= g_last_local[0]
                    T.barrier_arrive(bar_5)
                    T.barrier_wait(bar_5, i_s % 2)
                    T.gemm(k_shared[i_s % 2, :, :], vn_shared, h_fragment, transpose_A=True, clear_accum=False)
                    T.barrier_arrive(data_is_free[i_s % 2])
                if not finished[bb]:
                    T.copy(h_fragment, state[:, DV_start:DV_end])
            elif tx < 256:
                T.set_max_nreg(CONSUMER_V_NREG, 1)
                for i_s in T.serial(num_iters):
                    T.barrier_wait(data_is_ready[i_s % 2], (i_s // 2 + 0) % 2)
                    T.barrier_arrive(bar_0)
                    T.barrier_wait(bar_0, i_s % 2)
                    for j_s in T.Parallel(block_S):
                        g_exp_shared[j_s] = T.exp2(g_shared[i_s % 2, j_s] * 1.442695)
                    for j_s in T.Parallel(block_S):
                        g_rev_exp_shared[j_s] = T.if_then_else(
                            seq_start_idx + i_s * block_S + j_s < seq_end_idx,
                            T.exp2((g_shared[i_s % 2, block_S - 1] - g_shared[i_s % 2, j_s]) * 1.442695),
                            0.0,
                        )
                    T.barrier_arrive(bar_1)
                    T.barrier_wait(bar_1, i_s % 2)
                    T.gemm(k_shared[i_s % 2, :, :], h_shared, u_fragment, clear_accum=True)
                    for j_s, j_v in T.Parallel(block_S, 64):
                        u_fragment[j_s, j_v] *= -g_exp_shared[j_s]
                    for j_s, j_v in T.Parallel(block_S, 64):
                        u_fragment[j_s, j_v] += v_shared[i_s % 2, j_s, j_v]
                    for j_s, j_v in T.Parallel(block_S, 64):
                        v_shared[i_s % 2, j_s, j_v] = u_fragment[j_s, j_v]
                    T.barrier_wait(bar_3, i_s % 2)
                    T.gemm(a_shared[i_s % 2, :, :], v_shared[i_s % 2, :, :], v_fragment, clear_accum=True)
                    T.copy(v_fragment, vd_shared)
                    T.barrier_arrive(bar_4)
                    for j_s, j_v in T.Parallel(block_S, 64):
                        v_fragment[j_s, j_v] *= g_rev_exp_shared[j_s]
                    T.copy(v_fragment, vn_shared)
                    T.barrier_arrive(bar_5)
                    T.barrier_wait(bar_5, i_s % 2)
                    T.barrier_arrive(data_is_free[i_s % 2])
            elif tx < 384:
                T.set_max_nreg(CONSUMER_O_NREG, 1)
                for i_s in T.serial(num_iters):
                    T.barrier_wait(data_is_ready[i_s % 2], (i_s // 2 + 0) % 2)
                    T.barrier_arrive(bar_0)
                    T.barrier_wait(bar_0, i_s % 2)
                    T.gemm(
                        q_shared[i_s % 2, :, :], k_shared[i_s % 2, :, :], p_fragment, transpose_B=True, clear_accum=True
                    )
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        g_fragment[j_s, j_t] = g_shared[i_s % 2, j_s] - g_shared[i_s % 2, j_t]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        if j_s >= j_t:
                            g_fragment[j_s, j_t] = T.exp2(g_fragment[j_s, j_t] * 1.442695)
                        else:
                            g_fragment[j_s, j_t] = 0
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] = a_shared[i_s % 2, j_s, j_t]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] *= g_fragment[j_s, j_t]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] *= b_shared[i_s % 2, j_t]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_shared[i_s % 2, j_s, j_t] = a_fragment[j_s, j_t]
                    T.barrier_wait(bar_1, i_s % 2)
                    T.gemm(q_shared[i_s % 2, :, :], h_shared, o_fragment, clear_accum=True)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        p_fragment[j_s, j_t] *= 0.08838834764831845 * g_fragment[j_s, j_t]
                    T.copy(p_fragment, p_shared)
                    T.barrier_arrive(bar_3)
                    for j_s, j_v in T.Parallel(block_S, 64):
                        o_fragment[j_s, j_v] *= 0.08838834764831845 * g_exp_shared[j_s]
                    T.barrier_wait(bar_4, i_s % 2)
                    T.gemm(p_shared, vd_shared, o_fragment, clear_accum=False)
                    T.barrier_arrive(bar_5)
                    T.barrier_wait(bar_5, i_s % 2)
                    T.copy(o_fragment, o_shared)
                    T.barrier_arrive(data_is_free[i_s % 2])
                T.barrier_arrive(bar_o)
            else:
                T.set_max_nreg(PRODUCER_NREG, 0)
                if tx < 384 + 32:
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(data_is_free[i_s % 2], (i_s // 2 + 1) % 2)
                        left = seq_start_idx + i_s * block_S
                        right = left + block_S
                        if right <= seq_end_idx:
                            T.tma_copy(
                                q[batch_idx, left:right, bhg, 0:128],
                                q_shared[i_s % 2, :, :],
                                barrier=data_is_ready[i_s % 2],
                            )
                        else:
                            for j_s, j_k in T.Parallel(block_S, 128):
                                if left + j_s < seq_end_idx:
                                    q_shared[i_s % 2, j_s, j_k] = q[batch_idx, left + j_s, bhg, j_k]
                                else:
                                    q_shared[i_s % 2, j_s, j_k] = 0
                        if right <= seq_end_idx:
                            T.tma_copy(
                                k[batch_idx, left:right, bhg, 0:128],
                                k_shared[i_s % 2, :, :],
                                barrier=data_is_ready[i_s % 2],
                            )
                        else:
                            for j_s, j_k in T.Parallel(block_S, 128):
                                if left + j_s < seq_end_idx:
                                    k_shared[i_s % 2, j_s, j_k] = k[batch_idx, left + j_s, bhg, j_k]
                                else:
                                    k_shared[i_s % 2, j_s, j_k] = 0
                        T.barrier_arrive(data_is_ready[i_s % 2])
                elif tx < 384 + 64:
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(data_is_free[i_s % 2], (i_s // 2 + 1) % 2)
                        left = seq_start_idx + i_s * block_S
                        right = left + block_S
                        if right <= seq_end_idx:
                            T.tma_copy(
                                v[batch_idx, left:right, bh, DV_start:DV_end],
                                v_shared[i_s % 2, :, :],
                                barrier=data_is_ready[i_s % 2],
                            )
                        else:
                            for j_s, j_v in T.Parallel(block_S, 64):
                                if left + j_s < seq_end_idx:
                                    v_shared[i_s % 2, j_s, j_v] = v[batch_idx, left + j_s, bh, DV_start + j_v]
                                else:
                                    v_shared[i_s % 2, j_s, j_v] = 0
                        if right <= seq_end_idx:
                            for j_s in T.Parallel(block_S):
                                b_shared[i_s % 2, j_s] = b[batch_idx, left + j_s, bh]
                        else:
                            for j_s in T.Parallel(block_S):
                                if left + j_s < seq_end_idx:
                                    b_shared[i_s % 2, j_s] = b[batch_idx, left + j_s, bh]
                                else:
                                    b_shared[i_s % 2, j_s] = 0
                        T.barrier_arrive(data_is_ready[i_s % 2])
                elif tx < 384 + 96:
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(data_is_free[i_s % 2], (i_s // 2 + 1) % 2)
                        left = seq_start_idx + i_s * block_S
                        right = left + block_S
                        if right <= seq_end_idx:
                            T.tma_copy(
                                a[batch_idx, left:right, bh, 0:block_S],
                                a_shared[i_s % 2, :, :],
                                barrier=data_is_ready[i_s % 2],
                            )
                        else:
                            for j_s, j_t in T.Parallel(block_S, block_S):
                                if left + j_s < seq_end_idx:
                                    a_shared[i_s % 2, j_s, j_t] = a[batch_idx, left + j_s, bh, j_t]
                                else:
                                    a_shared[i_s % 2, j_s, j_t] = 0
                        if right <= seq_end_idx:
                            for j_s in T.Parallel(block_S):
                                g_shared[i_s % 2, j_s] = g[batch_idx, left + j_s, bh]
                        else:
                            for j_s in T.Parallel(block_S):
                                if left + j_s < seq_end_idx:
                                    g_shared[i_s % 2, j_s] = g[batch_idx, left + j_s, bh]
                                else:
                                    g_shared[i_s % 2, j_s] = g[batch_idx, seq_end_idx - 1, bh]
                        T.barrier_arrive(data_is_ready[i_s % 2])
                else:
                    for i_s in T.serial(num_unmasked_iters):
                        right = seq_start_idx + i_s * block_S
                        left = right - block_S
                        T.barrier_arrive(bar_0)
                        T.barrier_wait(bar_0, i_s % 2)
                        if i_s > 0:
                            T.copy(o_shared, o[batch_idx, left:right, bh, DV_start:DV_end])
                        T.barrier_arrive(bar_5)
                        T.barrier_wait(bar_1, i_s % 2)
                    if num_unmasked_iters < num_iters:
                        seq_split_idx = seq_start_idx + num_unmasked_iters * block_S
                        right = seq_split_idx
                        left = right - block_S
                        T.barrier_arrive(bar_0)
                        T.barrier_wait(bar_0, num_unmasked_iters % 2)
                        if num_unmasked_iters > 0:
                            T.copy(o_shared, o[batch_idx, left:right, bh, DV_start:DV_end])
                        T.barrier_arrive(bar_5)
                        T.barrier_wait(bar_1, num_unmasked_iters % 2)
                    seq_split_idx = seq_start_idx + (num_iters - 1) * block_S
                    T.barrier_wait(bar_o, 0)
                    if num_iters > 0:
                        for j_s, j_v in T.Parallel(block_S, 64):
                            if seq_split_idx + j_s < seq_end_idx:
                                o[batch_idx, seq_split_idx + j_s, bh, DV_start + j_v] = o_shared[j_s, j_v]
                    if bb == batch_size - 1:
                        for i_pad in T.serial(num_tokens - seq_end_idx):
                            for j_v in T.Parallel(64):
                                o[batch_idx, seq_end_idx + i_pad, bh, DV_start + j_v] = 0

    return tilelang_fused_chunk_gdr_fwd_kernel
