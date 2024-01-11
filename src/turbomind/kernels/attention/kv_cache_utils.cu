
#include "array_ops.h"
#include "kv_cache_utils.h"
#include "quantization.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"
#include "thread_map.h"

namespace turbomind {

template<class T, int N>
__device__ void CpAsync(T* dst, const Array<T, N>* __restrict__ src, bool mask)
{
    const int     smem_int_ptr = cast_smem_ptr_to_uint(dst);
    constexpr int cp_size      = sizeof(Array<T, N>);
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

template<class T, class Tkv, int CTA_Q, int kHeadDim, int kWarpCount, class ParamType>
__global__ void __launch_bounds__(128, 8) ProcessKV(ParamType params)
{
    constexpr int kVecSize = sizeof(uint4) / sizeof(T);

    using Vec = Array<T, kVecSize>;
    using Map = RakedThreadMap<kHeadDim, CTA_Q, kVecSize, kWarpCount>;

    constexpr int ITER_C = Map::kIterC;
    constexpr int ITER_S = Map::kIterS;

    const int token_idx = blockIdx.x * CTA_Q;  // local offset into `input_length`
    const int head_idx  = blockIdx.y;
    const int batch_idx = blockIdx.z;

    const int qi_beg = params.cu_seqlens[batch_idx] + token_idx;  // global offset into `cu_seqlens`
    const int qi_end = params.cu_seqlens[batch_idx + 1];

    const int input_len   = params.input_length[batch_idx];
    const int history_len = params.context_length[batch_idx] - input_len;

    if (token_idx >= input_len) {  // empty tile
        return;
    }

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int2 offset = Map::get_offset(warp_id, lane_id);

    Vec __align__(16) vec_K[ITER_S][ITER_C];
    Vec __align__(16) vec_V[ITER_S][ITER_C];

    Vec bias_V[ITER_C];
    Vec bias_K[ITER_C];

    if (params.k_bias) {
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int di = offset.x + c * Map::kDeltaC;
            Ldg(bias_K[c], &params.k_bias[head_idx * kHeadDim + di]);
        }
    }
    if (params.v_bias) {
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int di = offset.x + c * Map::kDeltaC;
            Ldg(bias_V[c], &params.v_bias[head_idx * kHeadDim + di]);
        }
    }

    PRAGMA_UNROLL
    for (int s = 0; s < ITER_S; ++s) {
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int qi = offset.y + s * Map::kDeltaS + qi_beg;
            const int di = offset.x + c * Map::kDeltaC;
            if (qi < qi_end) {
                Ldg(vec_K[s][c], &params.k[qi * params.stride + head_idx * kHeadDim + di]);
                Ldg(vec_V[s][c], &params.v[qi * params.stride + head_idx * kHeadDim + di]);
            }
        }
    }

    if (params.k_bias) {
        using namespace ops;
        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                vec_K[s][c] = vec_K[s][c] + bias_K[c];
            }
        }
    }
    if (params.v_bias) {
        using namespace ops;
        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                vec_V[s][c] = vec_V[s][c] + bias_V[c];
            }
        }
    }

    Tkv** k_cache_block_ptrs = (Tkv**)params.k_cache_block_ptrs + params.cu_block_cnts[batch_idx];

    Array<Tkv, kVecSize> out_K[ITER_S][ITER_C];
    Array<Tkv, kVecSize> out_V[ITER_S][ITER_C];

    // quant param
    using PType = T;
    Array<PType, 2> param_K[ITER_S];
    Array<PType, 2> param_V[ITER_S];

    if constexpr (std::is_same_v<T, Tkv>) {
        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                out_K[s][c] = vec_K[s][c];
                out_V[s][c] = vec_V[s][c];
            }
        }
    }
    else if constexpr (1) {
        constexpr std::integral_constant<int, sizeof(Tkv) * 8> n_bits{};
        warp_stats<Map::kWarpThreadC>(param_K, vec_K, n_bits);
        warp_stats<Map::kWarpThreadC>(param_V, vec_V, n_bits);
        quantize(out_K, vec_K, param_K, n_bits);
        quantize(out_V, vec_V, param_V, n_bits);
        fuse_magic(param_K);
        fuse_magic(param_V);
    }
    else {
        using QType = uint8_t;
        constexpr std::integral_constant<int, sizeof(QType) * 8> n_bits{};
        // quant data
        Array<QType, kVecSize> quant_K[ITER_S][ITER_C];
        Array<QType, kVecSize> quant_V[ITER_S][ITER_C];
        warp_stats<Map::kWarpThreadC>(param_K, vec_K, n_bits);
        warp_stats<Map::kWarpThreadC>(param_V, vec_V, n_bits);
        quantize(quant_K, vec_K, param_K, n_bits);
        quantize(quant_V, vec_V, param_V, n_bits);
        dequantize(out_K, quant_K, param_K, n_bits);
        dequantize(out_V, quant_V, param_V, n_bits);
    }

    // if constexpr (std::is_same_v<Tkv, uint8_t>) {
    //     PRAGMA_UNROLL
    //     for (int s = 0; s < ITER_S; ++s) {
    //         PRAGMA_UNROLL
    //         for (int c = 0; c < ITER_C; ++c) {
    //             permute_K(out_K[s][c]);
    //         }
    //     }
    //     permute_V<Map>(out_V);
    // }

    PRAGMA_UNROLL
    for (int s = 0; s < ITER_S; ++s) {
        const int si = offset.y + s * Map::kDeltaS;
        const int qi = si + token_idx;  // local offset into `input_length`

        if (qi < input_len) {
            const int ti = history_len + qi;  // timestep

            const int block_seqlen = params.kv_cache_block_size;
            // block index and local offsets
            const int cache_block_index  = ti / block_seqlen;
            const int cache_block_offset = ti % block_seqlen;
            // [H, s, D]
            Tkv* k_cache = k_cache_block_ptrs[cache_block_index] + params.key_offset
                           + head_idx * block_seqlen * kHeadDim + cache_block_offset * kHeadDim;
            Tkv* v_cache = k_cache_block_ptrs[cache_block_index] + params.val_offset
                           + head_idx * block_seqlen * kHeadDim + cache_block_offset * kHeadDim;
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                int di = offset.x + c * Map::kDeltaC;
                // di ^= ((si & 0x7) << 3);
                Stcs(&k_cache[di], out_K[s][c]);
                Stcs(&v_cache[di], out_V[s][c]);
            }

            if (std::is_same_v<Tkv, uint8_t>) {
                int max_context_len = params.max_input_len + params.max_seq_len;
                // [B, H, 2, S, 2]
                auto k_cache_quant_data = params.kv_cache_quant_data
                                          + batch_idx * params.num_kv_heads * 2 * max_context_len * 2
                                          + head_idx * 2 * max_context_len * 2  //
                                          + (history_len + qi) * 2;
                auto v_cache_quant_data = k_cache_quant_data + max_context_len * 2;

                if (offset.x == 0) {  // thread group leader stores
                    Stcs(k_cache_quant_data, param_K[s]);
                    Stcs(v_cache_quant_data, param_V[s]);
                }
            }
        }
    }
}

template<class T>
void invokeProcessKV(const AttentionParams<T>& params)
{
    constexpr int WARPS = 4;
    constexpr int DIMS  = 128;
    constexpr int CTA_Q = 64;
    using Tkv           = T;

    int  block = WARPS * WARP_SIZE;
    dim3 grid((params.max_input_len + CTA_Q - 1) / CTA_Q, params.num_kv_heads, params.batch_size);

    ProcessKV<T, Tkv, CTA_Q, DIMS, WARPS><<<grid, block, 0, params.stream>>>(params);
}

template void invokeProcessKV(const AttentionParams<half>& params);

}  // namespace turbomind