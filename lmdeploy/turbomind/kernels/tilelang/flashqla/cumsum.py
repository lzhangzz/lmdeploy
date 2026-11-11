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

# Adapted from FlashQLA `flash_qla/ops/utils/cumsum.py`
# (https://github.com/QwenLM/FlashQLA, revision a14dbf3474514e1674526431e960325f72081e9f)
# for TurboMind's ordinary prefill.

import tilelang
import tilelang.language as T


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def cumsum_prefill(H):
    real_batch_size = T.dynamic('real_batch_size')
    num_tokens = T.dynamic('num_tokens')
    g_shape = (1, num_tokens, H)

    @T.prim_func
    def tilelang_chunk_local_cumsum_kernel(
        g_raw: T.Tensor(g_shape, 'float32'),
        cu_seqlens: T.Tensor((real_batch_size + 1,), 'int32'),
        g_cumsum: T.Tensor(g_shape, 'float32'),
        num_chunks: T.int32,
    ):
        with T.Kernel(num_chunks, threads=128) as (bc,):
            sequence = T.alloc_var('int32')
            chunk = T.alloc_var('int32')
            sequence = -1
            chunk = bc
            for candidate in T.serial(real_batch_size):
                if sequence == -1:
                    count = T.ceildiv(cu_seqlens[candidate + 1] - cu_seqlens[candidate], 32)
                    if chunk < count:
                        sequence = candidate
                    else:
                        chunk -= count
            left = cu_seqlens[sequence] + chunk * 32
            seq_end = cu_seqlens[sequence + 1]
            g_fragment = T.alloc_fragment((H, 32), 'float32')
            gT_fragment = T.alloc_fragment((32, H), 'float32')
            gT_shared = T.alloc_shared((32, H + 1), 'float32')
            if left + 32 <= seq_end:
                T.copy(g_raw[0, left : left + 32, 0:H], gT_fragment)
            else:
                for j, i in T.Parallel(32, H):
                    if left + j < seq_end:
                        gT_fragment[j, i] = g_raw[0, left + j, i]
                    else:
                        gT_fragment[j, i] = 0
            T.copy(gT_fragment, gT_shared[:, :H])
            for i, j in T.Parallel(H, 32):
                g_fragment[i, j] = gT_shared[j, i]
            T.cumsum(g_fragment, dim=1)
            for i, j in T.Parallel(H, 32):
                gT_shared[j, i] = g_fragment[i, j]
            T.copy(gT_shared[:, :H], gT_fragment)
            if left + 32 <= seq_end:
                T.copy(gT_fragment, g_cumsum[0, left : left + 32, 0:H])
            else:
                for j, i in T.Parallel(32, H):
                    if left + j < seq_end:
                        g_cumsum[0, left + j, i] = gT_fragment[j, i]

    return tilelang_chunk_local_cumsum_kernel
