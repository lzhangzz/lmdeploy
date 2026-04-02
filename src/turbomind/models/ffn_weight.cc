// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>

#include "src/turbomind/models/ffn_weight.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/kernels/gemm/cast.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

FfnWeight::FfnWeight(int hidden_dim, int inter_size, bool bias, int tp_size, int tp_rank,
                     DataType data_type, ActivationType act_type, bool fuse_silu_act)
    : hidden_dim_(hidden_dim)
    , inter_size_(inter_size)
    , bias_(bias)
    , tp_size_(tp_size)
    , tp_rank_(tp_rank)
    , data_type_(data_type)
    , act_type_(act_type)
    , is_fused_silu_(fuse_silu_act && act_type == ActivationType::kSilu)
{
    TM_CHECK(inter_size_ % tp_size_ == 0) << inter_size_ << " " << tp_size_;
    inter_size_ /= tp_size_;
}

core::Module* FfnWeight::ensure_child(const std::string& segment)
{
    if (segment == "w1") {
        auto child = std::make_unique<LinearWeight>();
        child->configure(hidden_dim_, inter_size_, data_type_, bias_);
        return add_child("w1", std::move(child));
    }
    if (segment == "w3") {
        auto child = std::make_unique<LinearWeight>();
        child->configure(hidden_dim_, inter_size_, data_type_, bias_);
        return add_child("w3", std::move(child));
    }
    if (segment == "w2") {
        auto child = std::make_unique<LinearWeight>();
        child->configure(inter_size_, hidden_dim_, data_type_, bias_);
        return add_child("w2", std::move(child));
    }
    if (segment == "w1w3") {
        // Fused gate+up projection
        auto child = std::make_unique<LinearWeight>();
        child->configure(hidden_dim_, inter_size_ * 2, data_type_, bias_);
        return add_child("w1w3", std::move(child));
    }
    return nullptr;
}

// ======================================================================
// Tensor-level fusion helpers
// ======================================================================

static void InterleaveTensors(const Tensor& a, const Tensor& b, Tensor& c, cudaStream_t st)
{
    TM_CHECK(a.layout() == b.layout());
    int M, K;
    if (a.ndim() == 2) {
        std::tie(K, M) = a.shapes(0, 1);
    }
    else {
        M = a.shape(0);
        K = 1;
    }
    auto a_ = a.raw_data();
    auto b_ = b.raw_data();
    auto c_ = c.raw_data();

    const int bits = byte_size(a.dtype(), 8);
    if (bits == 4) {
        Buffer_<uint8_t> ta{a.size(), kDEVICE};
        Buffer_<uint8_t> tb{b.size(), kDEVICE};
        Buffer_<uint8_t> tc{c.size(), kDEVICE};
        extend_to_u8(ta.data(), (uint4_t*)a_, a.size(), st);
        extend_to_u8(tb.data(), (uint4_t*)b_, b.size(), st);
        interleave_output_dims(tc.data(), ta.data(), tb.data(), M, K, st);
        compact_to_u4((uint4_t*)c_, tc.data(), c.size(), st);
    }
    else if (bits == 8) {
        interleave_output_dims((uint8_t*)c_, (uint8_t*)a_, (uint8_t*)b_, M, K, st);
    }
    else if (bits == 16) {
        interleave_output_dims((uint16_t*)c_, (uint16_t*)a_, (uint16_t*)b_, M, K, st);
    }
    else if (bits == 32) {
        interleave_output_dims((uint32_t*)c_, (uint32_t*)a_, (uint32_t*)b_, M, K, st);
    }
    else {
        TM_CHECK(0);
    }
}

static void ChunkTensors(const Tensor& a, const Tensor& b, Tensor& c, cudaStream_t st)
{
    TM_CHECK(a.layout() == b.layout());
    int M, K, spitch, dpitch;
    if (a.ndim() == 2) {
        std::tie(K, M) = a.shapes(0, 1);
        spitch         = byte_size(a.dtype(), a.stride(0));
        dpitch         = byte_size(c.dtype(), c.stride(0));
    }
    else {
        M      = a.shape(0);
        K      = 1;
        spitch = byte_size(a.dtype(), M);
        dpitch = byte_size(c.dtype(), c.shape(0));
    }
    int height = K;
    int width  = byte_size(a.dtype(), M);
    check_cuda_error(cudaMemcpy2DAsync((char*)c.raw_data(),  //
                                       dpitch,
                                       (const char*)a.raw_data(),
                                       spitch,
                                       width,
                                       height,
                                       cudaMemcpyDefault,
                                       st));
    check_cuda_error(cudaMemcpy2DAsync((char*)c.raw_data() + width,  //
                                       dpitch,
                                       (const char*)b.raw_data(),
                                       spitch,
                                       width,
                                       height,
                                       cudaMemcpyDefault,
                                       st));
}

/// Interleave w1 and w3 into fused w1w3 (for fused SiLU epilogue).
static void FuseInterleave(LinearWeight& fused, LinearWeight& w1, LinearWeight& w3, cudaStream_t st)
{
    TM_CHECK_EQ(fused.input_dim, w1.input_dim);
    TM_CHECK_EQ(fused.output_dim, w1.output_dim * 2);
    TM_CHECK_EQ(fused.group_size, w1.group_size);

    InterleaveTensors(w1.weight, w3.weight, fused.weight, st);
    sync_check_cuda_error();

    if (w1.scales) {
        InterleaveTensors(w1.scales, w3.scales, fused.scales, st);
        sync_check_cuda_error();
    }
    if (w1.zeros) {
        InterleaveTensors(w1.zeros, w3.zeros, fused.zeros, st);
        sync_check_cuda_error();
    }
    if (w1.bias) {
        InterleaveTensors(w1.bias, w3.bias, fused.bias, st);
        sync_check_cuda_error();
    }
}

/// Concatenate w1 and w3 into fused w1w3 (chunk layout: [w1 | w3]).
static void FuseChunk(LinearWeight& fused, LinearWeight& w1, LinearWeight& w3, cudaStream_t st)
{
    TM_CHECK_EQ(fused.input_dim, w1.input_dim);
    TM_CHECK_EQ(fused.output_dim, w1.output_dim * 2);
    TM_CHECK_EQ(fused.group_size, w1.group_size);

    ChunkTensors(w1.weight, w3.weight, fused.weight, st);
    sync_check_cuda_error();

    if (w1.scales) {
        ChunkTensors(w1.scales, w3.scales, fused.scales, st);
        sync_check_cuda_error();
    }
    if (w1.zeros) {
        ChunkTensors(w1.zeros, w3.zeros, fused.zeros, st);
        sync_check_cuda_error();
    }
    if (w1.bias) {
        ChunkTensors(w1.bias, w3.bias, fused.bias, st);
        sync_check_cuda_error();
    }
}

// ======================================================================
// FfnWeight::prepare — fuse w1+w3 into w1w3, then convert all weights
// ======================================================================

void FfnWeight::prepare()
{
    auto* w1_ptr = w1();
    auto* w3_ptr = w3();
    auto* fused_ptr = w1w3();

    // Fuse w1+w3 into w1w3 if both exist and w1w3 doesn't yet.
    // This must happen BEFORE format conversion (prepare) so that the fused
    // weight is converted as a single unit, matching the old code's behavior.
    if (w1_ptr && w3_ptr && !fused_ptr && *w1_ptr && *w3_ptr) {
        // Pre-process individual weights (blockscale → groupwise conversion)
        w1_ptr->preprocess();
        w3_ptr->preprocess();

        // Determine if fused SiLU epilogue can be used.
        // Disabled for quantized formats with element size >= 16 bits or FP8 input.
        bool fused_silu = is_fused_silu_;
        if (byte_size(w1_ptr->weight_type, 8) >= 16) {
            fused_silu = false;
        }
        if (w1_ptr->input_type == kFloat8_e4m3) {
            fused_silu = false;
        }

        // Create the fused w1w3 weight
        auto fused = std::make_unique<LinearWeight>();
        fused->configure(w1_ptr->input_dim, w1_ptr->output_dim * 2, w1_ptr->data_type, (bool)w1_ptr->bias);
        fused->allocate(w1_ptr->weight_type, w1_ptr->group_size);

        auto stream = core::Context::stream().handle();

        if (fused_silu) {
            FuseInterleave(*fused, *w1_ptr, *w3_ptr, stream);
            fused->epilogue = gemm::Epilogue::kGatedSilu;
        }
        else {
            FuseChunk(*fused, *w1_ptr, *w3_ptr, stream);
        }

        // Remove w1 and w3 from children (data now lives in w1w3)
        auto remove = [this](const std::string& name) {
            auto it = std::find_if(children_.begin(), children_.end(),
                                   [&](const auto& p) { return p.first == name; });
            if (it != children_.end()) {
                children_.erase(it);
            }
        };
        remove("w1");
        remove("w3");

        // Add the fused weight as a child
        add_child("w1w3", std::move(fused));

        // Update is_fused_silu_ to reflect the actual epilogue state.
        // The forward pass uses this flag to decide whether to apply SiLU
        // explicitly.  It must be false when the GEMM epilogue is kNone
        // (dense bf16/fp16, FP8 input) so that Activation() is called.
        is_fused_silu_ = fused_silu;
    }

    // Prepare (format conversion) for all remaining children
    for (auto& [name, child] : children_) {
        // Propagate grouped-GEMM flag for MoE expert weights
        if (is_fused_moe_) {
            if (auto* lw = dynamic_cast<LinearWeight*>(child.get())) {
                lw->set_grouped(true);
            }
        }
        child->prepare();
    }
}

}  // namespace turbomind
