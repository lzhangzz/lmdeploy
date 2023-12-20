// Copyright (c) OpenMMLab. All rights reserved.

#include "attention.h"
#include "kv_cache.h"
#include "src/turbomind/kernels/attention/reference.h"
#include "src/turbomind/kernels/unfused_attention_kernels.h"
#include "test_utils.h"
#include <cmath>
#include <ios>
#include <iostream>
#include <limits>
#include <thrust/universal_vector.h>

#include <algorithm>
#include <iomanip>
#include <numeric>
#include <random>

using namespace turbomind;

// [S/S, H, S, D] <-> [S/b, H, b, D]
void TestBlocks(const thrust::universal_vector<half>& k_cache,  // [B, H, S, D]
                const thrust::universal_vector<half>& v_cache,  // [B, H, S, D]
                thrust::universal_vector<half>&       blocks,   // block data
                thrust::universal_vector<half*>&      k_ptrs,   // block ptrs
                thrust::universal_vector<half*>&      v_ptrs,
                thrust::universal_vector<int>&        cu_block_cnts,  // cumulative block counts
                const int                             head_num,
                const int                             head_dim,
                const int                             block_seq_len,
                const int                             batch_size)
{
    const int seq_len  = k_cache.size() / (head_dim * head_num * batch_size);
    const int n_blocks = (seq_len + block_seq_len - 1) / block_seq_len;

    const int kHSD = head_num * seq_len * head_dim;

    std::cout << "batch_size = " << batch_size << ", seq_len = " << seq_len << ", block_size = " << block_seq_len
              << ", block_num = " << n_blocks << "\n";

    thrust::universal_vector<half> kv_cache(k_cache.size() * 2);  // [B, 2, H, S, D]

    {  // interleave K/V
        auto k_src = k_cache.begin();
        auto v_src = v_cache.begin();
        auto dst   = kv_cache.begin();
        for (int i = 0; i < batch_size; ++i) {
            dst = thrust::copy_n(k_src, kHSD, dst);
            dst = thrust::copy_n(v_src, kHSD, dst);
            k_src += kHSD;
            v_src += kHSD;
        }
    }

    const int kHsD = head_num * block_seq_len * head_dim;

    // [B, S/s, 2, H, s, D]
    blocks.resize(batch_size * n_blocks * 2 * kHsD);
    thrust::fill(blocks.begin(), blocks.end(), NAN);
    k_ptrs.resize(batch_size * n_blocks + 1);  // +1 padding
    v_ptrs.resize(batch_size * n_blocks + 1);

    std::vector<size_t> idxs(batch_size * n_blocks);
    std::iota(idxs.begin(), idxs.end(), 0);

    std::random_device rd;
    std::mt19937       g(rd());
    std::shuffle(idxs.begin(), idxs.end(), g);

    for (size_t i = 0; i < idxs.size(); ++i) {
        k_ptrs[i] = blocks.data().get() + idxs[i] * 2 * kHsD;
        v_ptrs[i] = k_ptrs[i] + kHsD;
    }

    thrust::universal_vector<int> seq_lens(batch_size);
    thrust::fill(seq_lens.begin(), seq_lens.end(), seq_len);

    std::vector<int> n_blocks_vec(batch_size + 1, n_blocks);
    cu_block_cnts.resize(batch_size + 1);
    std::exclusive_scan(n_blocks_vec.begin(), n_blocks_vec.end(), cu_block_cnts.begin(), 0);

    // [B, 2H, S, D] -> [B, S/s] x [2H, s, D]
    for (int i = 0; i < 1; ++i) {
        ConvertLinearToBlocks((const half*)kv_cache.data().get(),
                              k_ptrs.data().get(),
                              cu_block_cnts.data().get(),
                              seq_lens.data().get(),
                              0,
                              seq_len,
                              block_seq_len,
                              2 * head_num,
                              head_dim,
                              batch_size,
                              0);
    }

    thrust::universal_vector<half> kv_cache_2(kv_cache.size());
    // round trip test
    for (int i = 0; i < 1; ++i) {
        ConvertBlocksToLinear((const half**)k_ptrs.data().get(),
                              kv_cache_2.data().get(),
                              cu_block_cnts.data().get(),
                              seq_lens.data().get(),
                              0,
                              block_seq_len,
                              seq_len,
                              2 * head_num,
                              head_dim,
                              batch_size,
                              0);
    }
    cudaDeviceSynchronize();

    if (0) {
        std::cout << ">>> Compare\n";
        Compare(
            kv_cache.data().get(), kv_cache_2.data().get(), head_dim, head_dim, batch_size * 2 * head_num * seq_len);
        std::cout << "<<< Compare\n";
    }
}

int main(int argc, char* argv[])
{
    AttentionParams<half> params{};

    constexpr int kHeadNum = 16;
    // constexpr int kHeadNum     = 1;
    constexpr int kHeadDim   = 128;
    constexpr int KvHeadNum  = kHeadNum;
    constexpr int kBatchSize = 2;
    // constexpr int kBatchSize   = 1;
    constexpr int kInputLen    = 8192;
    constexpr int kSequenceLen = 0;
    // constexpr int kInputLen    = 4096 - 20;
    // constexpr int kSequenceLen = 32 + 16 + 8 + 4;  // force partial tile
    // constexpr int kSequenceLen = 983;
    // constexpr int kInputLen    = 2387;
    // constexpr int kSequenceLen = 72;
    // constexpr int kInputLen    = 98;
    constexpr int kContextLen = kSequenceLen + kInputLen;
    constexpr int kBlockSz    = 64;
    constexpr int kTestIter   = 10;
    constexpr int kMaxSplitK  = 1;

    RNG rng{};

    thrust::universal_vector<half> k_cache(kBatchSize * KvHeadNum * kContextLen * kHeadDim);
    thrust::universal_vector<half> v_cache(kBatchSize * KvHeadNum * kContextLen * kHeadDim);

    thrust::universal_vector<half> qkv(kBatchSize * kInputLen * (kHeadNum + KvHeadNum * 2) * kHeadDim);
    thrust::universal_vector<half> output(kBatchSize * kInputLen * kHeadNum * kHeadDim);

    thrust::universal_vector<bool> finished(kBatchSize);
    thrust::universal_vector<int>  sequence_length(kBatchSize);
    thrust::universal_vector<int>  input_length(kBatchSize);
    thrust::universal_vector<int>  context_length(kBatchSize);
    thrust::universal_vector<int>  cu_seqlens(kBatchSize + 1);

    thrust::universal_vector<float> partial_M(kBatchSize * kHeadNum * kMaxSplitK);
    thrust::universal_vector<float> partial_L(kBatchSize * kHeadNum * kMaxSplitK);
    thrust::universal_vector<float> partial_O(kBatchSize * kHeadNum * kMaxSplitK * kHeadDim);
    thrust::universal_vector<int>   semaphores(kBatchSize * kHeadNum * kMaxSplitK);

    std::fill(semaphores.begin(), semaphores.end(), 0);

    rng.GenerateNormal(qkv.data().get(), qkv.size(), 1.f, 0.f);

    rng.GenerateNormal(k_cache.data().get(), kBatchSize * KvHeadNum * kContextLen * kHeadDim);
    rng.GenerateNormal(v_cache.data().get(), kBatchSize * KvHeadNum * kContextLen * kHeadDim);

    // Set input range to zero
    // (BH, SD)
    cudaMemset2DAsync(k_cache.data().get() + kSequenceLen * kHeadDim,
                      sizeof(half) * kContextLen * kHeadDim,
                      0,
                      sizeof(half) * kInputLen * kHeadDim,
                      kBatchSize * KvHeadNum);
    cudaMemset2DAsync(v_cache.data().get() + kSequenceLen * kHeadDim,
                      sizeof(half) * kContextLen * kHeadDim,
                      0,
                      sizeof(half) * kInputLen * kHeadDim,
                      kBatchSize * KvHeadNum);

    thrust::universal_vector<half>  blocks;
    thrust::universal_vector<half*> k_ptrs;
    thrust::universal_vector<half*> v_ptrs;
    thrust::universal_vector<int>   cu_block_cnts;

    TestBlocks(k_cache, v_cache, blocks, k_ptrs, v_ptrs, cu_block_cnts, KvHeadNum, kHeadDim, kBlockSz, kBatchSize);

    thrust::universal_vector<half>  k_cache_ref = k_cache;
    thrust::universal_vector<half>  v_cache_ref = v_cache;
    thrust::universal_vector<half>  output_ref  = output;
    thrust::universal_vector<void*> k_cache_ref_ptrs(kBatchSize);
    thrust::universal_vector<void*> v_cache_ref_ptrs(kBatchSize);

    cudaDeviceSynchronize();

    for (int i = 0; i <= kBatchSize; ++i) {
        cu_seqlens[i] = i * kInputLen;
    }

    for (int i = 0; i < kBatchSize; ++i) {
        input_length[i]     = kInputLen;
        sequence_length[i]  = kSequenceLen;
        context_length[i]   = kContextLen;
        k_cache_ref_ptrs[i] = k_cache_ref.data().get() + i * k_cache_ref.size() / kBatchSize;
        v_cache_ref_ptrs[i] = v_cache_ref.data().get() + i * v_cache_ref.size() / kBatchSize;
    }

    // getchar();

    params.out    = output_ref.data().get();
    params.q      = qkv.data().get();
    params.k      = params.q + kHeadNum * kHeadDim;
    params.v      = params.k + KvHeadNum * kHeadDim;
    params.stride = (kHeadNum + 2 * KvHeadNum) * kHeadDim;

    params.batch_size    = kBatchSize;
    params.max_input_len = kInputLen;
    params.max_seq_len   = kSequenceLen;
    params.cu_block_cnts = cu_block_cnts.data().get();

    params.k_cache_block_ptrs = (void**)k_ptrs.data().get();
    // params.v_cache_block_ptrs  = (void**)v_ptrs.data().get();
    params.kv_cache_block_size = kBlockSz;

    params.finished       = finished.data().get();
    params.input_length   = input_length.data().get();
    params.context_length = context_length.data().get();
    params.cu_seqlens     = cu_seqlens.data().get();
    // params.layer_offset   = 0;
    // [L, 2, H, s, D]
    params.key_offset = 0;
    params.val_offset = params.key_offset + KvHeadNum * kBlockSz * kHeadDim;

    params.num_heads     = kHeadNum;
    params.num_kv_heads  = KvHeadNum;
    params.size_per_head = kHeadDim;
    params.inv_sqrt_dh   = M_LOG2E / std::sqrt((float)params.size_per_head);

    params.rotary_embedding_dim  = kHeadDim;
    params.rotary_embedding_base = 10000.f;

    params.partial_L = partial_L.data().get();
    params.partial_M = partial_M.data().get();
    params.partial_O = partial_O.data().get();
    params.locks     = semaphores.data().get();

    params.max_split_k = kMaxSplitK;
    params.arch        = 80;

    // thrust::universal_vector<float> qk(kBatchSize * kInputLen * kSequenceLen);
    // thrust::universal_vector<half>  pr(kBatchSize * kInputLen * kSequenceLen);
    // params.qk = qk.data().get();
    // params.pr = pr.data().get();

    Reference<half> reference(Reference<half>::kFLASH_ATTENTION, {});
    reference.Reshape(kInputLen, kContextLen, kHeadNum, kHeadDim, KvHeadNum, kBatchSize);

    for (int i = 0; i < 10; ++i) {
        // mmha_ft_reference(params,
        //                   (half**)k_cache_ref_ptrs.data().get(),
        //                   (half**)v_cache_ref_ptrs.data().get(),
        //                   sequence_length.data().get(),
        //                   kContextLen,
        //                   cudaStream_t{});

        reference.Execute(params.out, k_cache_ref.data().get(), v_cache_ref.data().get(), qkv.data().get());
    }

    cudaDeviceSynchronize();

    if (0) {
        for (int b = 0; b < kBatchSize; ++b) {
            for (int i = 0; i < kInputLen; ++i) {
                auto qk = reference.qk() + b * kInputLen * kContextLen + i * kInputLen;
                for (int j = 0; j < kInputLen; ++j) {
                    std::cout << qk[j] * params.inv_sqrt_dh << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }

        std::cout << "\n";

        // for (int i = 0; i < kInputLen; ++i) {
        //     auto pr = unfused_attention.pr() + i * kInputLen;
        //     for (int j = 0; j < kInputLen; ++j) {
        //         std::cout << (float)pr[j] << " ";
        //     }
        //     std::cout << "\n";
        // }
    }

    if (auto err = cudaGetLastError(); err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << "\n";
        return -1;
    }
    std::cout << "---------------------------------------------------\n";

    params.out = output.data().get();

    std::vector<thrust::universal_vector<half>> outputs;

    for (int i = 0; i < std::max(kTestIter, 1); ++i) {
        invokeProcessKV<half>(params);
        dispatchAttention<half>(params);
        if (auto err = cudaGetLastError(); err != cudaSuccess) {
            std::cout << cudaGetErrorString(err) << "\n";
            return -1;
        }
        if (1) {
            outputs.push_back(output);
        }
    }

    if (1) {
        ConvertBlocksToLinear((const half**)k_ptrs.data().get(),
                              k_cache.data().get(),
                              cu_block_cnts.data().get(),
                              context_length.data().get(),
                              0,
                              kBlockSz,
                              kContextLen,
                              KvHeadNum,
                              kHeadDim,
                              kBatchSize,
                              0);
        ConvertBlocksToLinear((const half**)v_ptrs.data().get(),
                              v_cache.data().get(),
                              cu_block_cnts.data().get(),
                              context_length.data().get(),
                              0,
                              kBlockSz,
                              kContextLen,
                              KvHeadNum,
                              kHeadDim,
                              kBatchSize,
                              0);
    }

    cudaDeviceSynchronize();

    if (outputs.size() > 1) {
        std::cout << "Evaluating consistency..." << std::endl;
        for (size_t i = 1; i < outputs.size(); ++i) {
            Compare(outputs[i].data().get(), outputs[0].data().get(), kHeadDim, kHeadDim, kHeadNum);
        }
    }

    std::cout << "---------------------------------------------------\n";

    // [B, S, H, D]
    // Compare(output.data().get(),  //
    //         output_ref.data().get(),
    //         kHeadDim,
    //         kHeadDim,
    //         kBatchSize * kInputLen * kHeadNum,
    //         0);
    Compare(output.data().get(),  //
            output_ref.data().get(),
            kHeadNum * kHeadDim,
            kHeadNum * kHeadDim,
            kBatchSize * kInputLen,
            0);

    // [BH, SD]
    Compare(k_cache.data().get() + kSequenceLen * kHeadDim,
            k_cache_ref.data().get() + kSequenceLen * kHeadDim,
            kContextLen * kHeadDim,
            kInputLen * kHeadDim,
            kBatchSize * KvHeadNum);
    Compare(v_cache.data().get() + kSequenceLen * kHeadDim,
            v_cache_ref.data().get() + kSequenceLen * kHeadDim,
            kContextLen * kHeadDim,
            kInputLen * kHeadDim,
            kBatchSize * KvHeadNum);

    return 0;
}

// if (0) {
//     for (int i = 0; i < kBatchSize; ++i) {
//         std::cout << "Compare Keys " << i << ": ";
//         Compare((half*)params.k_cache_block_ptrs[cu_block_cnts[i]],                 // [H, s, D]
//                 k_cache_ref.data().get() + i * KvHeadNum * kContextLen * kHeadDim,  // [H, S, D]
//                 kHeadDim,
//                 kHeadDim,
//                 KvHeadNum * kContextLen,
//                 1);
//     }
// }