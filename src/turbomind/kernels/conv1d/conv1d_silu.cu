
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/conv1d/conv1d_silu.h"
#include "src/turbomind/kernels/core/common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <type_traits>

#include "src/turbomind/utils/cuda_utils.h"

#include "src/turbomind/kernels/core/array_ops.h"

namespace turbomind {

// =============================================================================
// Fused Conv1d + SiLU — persistent batched kernel
//
// Weight layout: [d_conv, conv_dim], State layout: [d_conv, conv_dim] per batch.
//
// Persistent 1D grid. Each block has a fixed channel tile
// (blockIdx.x % num_ch_tiles) and atomically claims single-token work items
// via a global counter. Token-major work ordering with grid size a multiple of
// num_ch_tiles guarantees monotonically increasing tokens and a fixed channel
// tile per block.
// =============================================================================
template<int D_CONV, int CHANNELS_PER_THREAD, int BLOCK_DIM, int NUM_TOKENS, typename T>
__global__ void __launch_bounds__(BLOCK_DIM) fused_conv1d_batched_kernel_v2(T*           out,
                                                                            const T*     in,
                                                                            const T*     weight,
                                                                            const T*     bias,
                                                                            void* const* conv_state_ptrs,
                                                                            const int*   q_offsets,
                                                                            const int*   k_offsets,
                                                                            const bool*  finished,
                                                                            int*         work_counter,
                                                                            int          batch_size,
                                                                            int          conv_dim,
                                                                            int          in_stride,
                                                                            int          num_token_tiles,
                                                                            int          state_layer_offset,
                                                                            int          total_work,
                                                                            int          num_ch_tiles)
{
    static_assert(BLOCK_DIM * CHANNELS_PER_THREAD > 0);

    int prev_ch_tile = -1;
    int c_base       = 0;

    Array<T, CHANNELS_PER_THREAD> w_tap[D_CONV];
    Array<T, CHANNELS_PER_THREAD> bias_vals;

    __shared__ int  s_work_id;
    __shared__ int4 s_batch_info;
    int             b_start = 0;

    while (true) {
        if (threadIdx.x == 0)
            s_work_id = atomicAdd(work_counter, 1);
        __syncthreads();

        if (s_work_id >= total_work)
            break;

        const int t_tile  = s_work_id % num_token_tiles;
        const int ch_tile = s_work_id / num_token_tiles;

        if (ch_tile != prev_ch_tile) {
            prev_ch_tile = ch_tile;
            b_start      = 0;
        }

        c_base = (ch_tile * BLOCK_DIM + threadIdx.x) * CHANNELS_PER_THREAD;

        const bool ch_active = (c_base < conv_dim);

        if (ch_active) {
            PRAGMA_UNROLL
            for (int d = 0; d < D_CONV; ++d) {
                Load(w_tap[d], weight + d * conv_dim + c_base);
            }
            if (bias)
                Load(bias_vals, bias + c_base);
        }

        if constexpr (NUM_TOKENS == 1) {
            for (int b = b_start + threadIdx.x; b < batch_size; b += BLOCK_DIM) {
                int lo = __ldg(&q_offsets[b]);
                if (lo > t_tile)
                    break;
                int hi = __ldg(&q_offsets[b + 1]);
                if (t_tile < hi) {
                    int seq      = hi - lo;
                    int hist     = (__ldg(&k_offsets[b + 1]) - __ldg(&k_offsets[b])) - seq;
                    s_batch_info = make_int4(b, lo, seq, hist);
                }
            }
        }
        else {
            for (int b = b_start + threadIdx.x; b < batch_size; b += BLOCK_DIM) {
                int tile_off = __ldg(&q_offsets[b]) / NUM_TOKENS + b;
                if (tile_off > t_tile)
                    break;
                int tile_off_next = __ldg(&q_offsets[b + 1]) / NUM_TOKENS + b + 1;
                if (t_tile < tile_off_next) {
                    int lo       = __ldg(&q_offsets[b]);
                    int seq      = __ldg(&q_offsets[b + 1]) - lo;
                    int hist     = (__ldg(&k_offsets[b + 1]) - __ldg(&k_offsets[b])) - seq;
                    s_batch_info = make_int4(b, lo, seq, hist);
                }
            }
        }
        __syncthreads();

        b_start = s_batch_info.x;

        const int4 bi          = s_batch_info;
        const int  b           = bi.x;
        const int  seq_off     = bi.y;
        const int  seq_len     = bi.z;
        const int  history_len = bi.w;

        int t_local_start;
        int n_tokens;
        if constexpr (NUM_TOKENS == 1) {
            t_local_start = t_tile - seq_off;
            n_tokens      = 1;
        }
        else {
            const int tile_off_b = seq_off / NUM_TOKENS + b;
            t_local_start        = (t_tile - tile_off_b) * NUM_TOKENS;
            if (t_local_start >= seq_len)
                continue;
            n_tokens = min(NUM_TOKENS, seq_len - t_local_start);
        }

        const bool skip_state_store = finished != nullptr && finished[b];
        const int  ring_start       = (history_len + t_local_start + 1) % D_CONV;
        T*         state_base       = (T*)conv_state_ptrs[b] + state_layer_offset;

        if (ch_active) {
            constexpr int                 VALS_SIZE = NUM_TOKENS + D_CONV - 1;
            Array<T, CHANNELS_PER_THREAD> vals[VALS_SIZE];
            const int                     n_vals = n_tokens + D_CONV - 1;

            PRAGMA_UNROLL
            for (int i = 0; i < VALS_SIZE; ++i) {
                if (i < n_vals) {
                    int pos = t_local_start - (D_CONV - 1) + i;
                    if (pos >= 0) {
                        Load(vals[i], in + (seq_off + pos) * in_stride + c_base);
                    }
                    else {
                        int ring_d = (ring_start + i) % D_CONV;
                        Load(vals[i], state_base + ring_d * conv_dim + c_base);
                    }
                }
            }

            PRAGMA_UNROLL
            for (int tok = 0; tok < NUM_TOKENS; ++tok) {
                if (tok < n_tokens) {
                    float acc[CHANNELS_PER_THREAD] = {};
                    PRAGMA_UNROLL
                    for (int d = 0; d < D_CONV; ++d) {
                        PRAGMA_UNROLL
                        for (int ch = 0; ch < CHANNELS_PER_THREAD; ++ch) {
                            acc[ch] += static_cast<float>(vals[tok + d][ch]) * static_cast<float>(w_tap[d][ch]);
                        }
                    }

                    Array<T, CHANNELS_PER_THREAD> out_vals;
                    PRAGMA_UNROLL
                    for (int ch = 0; ch < CHANNELS_PER_THREAD; ++ch) {
                        if (bias)
                            acc[ch] += static_cast<float>(bias_vals[ch]);
                        out_vals[ch] = static_cast<T>(acc[ch] / (1.0f + expf(-acc[ch])));
                    }

                    Store(out + (seq_off + t_local_start + tok) * conv_dim + c_base, out_vals);
                }
            }

            if (!skip_state_store && t_local_start + n_tokens >= seq_len) {
                PRAGMA_UNROLL
                for (int i = 0; i < VALS_SIZE; ++i) {
                    int pos = t_local_start - (D_CONV - 1) + i;
                    if (pos >= 0 && pos >= seq_len - D_CONV && pos < seq_len) {
                        int ring_d = (ring_start + i) % D_CONV;
                        Store(state_base + ring_d * conv_dim + c_base, vals[i]);
                    }
                }
            }
        }
    }
}

void invokeFusedConv1dSiLU(Ref<Tensor>           out_,
                           const Tensor&         in,
                           const Tensor&         weight,
                           const Tensor&         bias,
                           const Buffer_<void*>& conv_state_ptrs,
                           const Buffer_<int>&   q_offsets,
                           const Buffer_<int>&   k_offsets,
                           const Buffer_<bool>&  finished,
                           int                   batch_size,
                           int                   state_layer_offset,
                           int                   sm_count,
                           int*                  work_counter,
                           cudaStream_t          stream)
{
    auto& out = out_.get();

    const int total_tokens = in.shape(0);
    const int d_conv       = weight.shape(0);
    const int conv_dim     = weight.shape(1);
    const int in_stride    = in.stride(0);

    constexpr int threads = 128;

    auto invoke = [&](auto t) {
        using T = decltype(t);
        if (d_conv == 4) {
            constexpr int kDConv     = 4;
            constexpr int kChPerT    = 8;
            const int     ch_per_blk = threads * kChPerT;
            TM_CHECK(conv_dim % kChPerT == 0);
            const int num_ch_tiles = cdiv(conv_dim, ch_per_blk);

            auto launch = [&](auto num_tok_tag) {
                constexpr int kNumTok         = decltype(num_tok_tag)::value;
                const int     num_token_tiles = (kNumTok == 1) ? total_tokens : total_tokens / kNumTok + batch_size;
                const int     total_work      = num_token_tiles * num_ch_tiles;

                auto kernel        = fused_conv1d_batched_kernel_v2<kDConv, kChPerT, threads, kNumTok, T>;
                int  blocks_per_sm = 1;
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, threads, 0);
                int grid = min(total_work, blocks_per_sm * sm_count);

                cudaMemsetAsync(work_counter, 0, sizeof(int), stream);
                kernel<<<grid, threads, 0, stream>>>(out.data<T>(),
                                                     in.data<T>(),
                                                     weight.data<T>(),
                                                     bias ? bias.data<T>() : (T*)nullptr,
                                                     conv_state_ptrs.data(),
                                                     q_offsets.data(),
                                                     k_offsets.data(),
                                                     finished ? finished.data() : nullptr,
                                                     work_counter,
                                                     batch_size,
                                                     conv_dim,
                                                     in_stride,
                                                     num_token_tiles,
                                                     state_layer_offset,
                                                     total_work,
                                                     num_ch_tiles);
            };

            int avg_seq = total_tokens / batch_size;
            if (avg_seq >= 4)
                launch(std::integral_constant<int, 5>{});
            else
                launch(std::integral_constant<int, 1>{});
        }
        else {
            TM_LOG_FATAL("Only d_conv == 4 is supported by fused_conv1d_batched_kernel_v2");
        }
    };
    TM_DISPATCH_PRIMARY_DTYPES(out.dtype(), invoke);
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace turbomind
