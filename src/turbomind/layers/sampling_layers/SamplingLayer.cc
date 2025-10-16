/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/turbomind/layers/sampling_layers/SamplingLayer.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/kernels/sampling_kernels.h"
#include "src/turbomind/kernels/sampling_topk_kernels.h"
#include "src/turbomind/kernels/sampling_topp_kernels.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind {

struct SamplerStates {
    int   max_topk;
    int   min_topk;
    float min_topp;
    float max_minp;

    Buffer_<int>   top_k_buf;
    Buffer_<float> top_p_buf;
    Buffer_<float> min_p_buf;

    Buffer_<int> kept_buf;  // kept sample
};

template<typename T>
SamplingLayer<T>::SamplingLayer(const BaseParam& param, const std::vector<std::shared_ptr<SamplingStates>>& states):
    BaseDynamicDecodeLayer{param}
{
    top_k_ = {max_batch_size_, kCPUpinned};
    top_p_ = {max_batch_size_, kCPUpinned};
    min_p_ = {max_batch_size_, kCPUpinned};
    kept_  = {max_batch_size_, kCPUpinned};

    // constant array
    std::fill_n(kept_.data(), max_batch_size_, vocab_size_);

    for (auto& state : states) {
        auto s = std::make_shared<SamplerStates>();

        s->top_k_buf = {max_batch_size_, kDEVICE};
        s->top_p_buf = {max_batch_size_, kDEVICE};
        s->min_p_buf = {max_batch_size_, kDEVICE};
        s->kept_buf  = {max_batch_size_, kDEVICE};

        state->sampler = std::move(s);
    }
}

template<typename T>
void SamplingLayer<T>::Forward(const std::shared_ptr<SamplingStates>& states, TensorMap& args) const
{
    // step1:
    //  - use topk / topp_minp kernel to sort and filter the scores
    //  - softmax the left score
    // step2:
    //  - sampling from left and sorted scores

    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    auto& s = states->sampler;

    Tensor_<T> logits = args.at("logits");

    const auto bsz = logits.shape(0);

    const int step = *args.at("step").data<int>();

    core::Copy(kept_.data(), bsz, s->kept_buf.data());

    Buffer_<int> indices(bsz * vocab_size_padded_, kDEVICE);

    // use topk sort if some request use topk filter
    if (s->max_topk > 0) {
        // TODO: top_k >= 64 is much slower than torch.topk()
        TopKSortFilterParams params{};
        params.logits            = logits.data();
        params.sorted_logits     = logits.data();
        params.sorted_indices    = indices.data();
        params.kept              = s->kept_buf.data();
        params.top_ks            = s->top_k_buf.data();
        params.max_top_k         = s->max_topk;
        params.batch_size        = bsz;
        params.vocab_size        = vocab_size_;
        params.vocab_size_padded = vocab_size_padded_;
        invokeTopKSortFilter<T>(params, stream_);
    }

    // use topp sort if some request skip topk filter
    if (s->min_topk == 0) {
        invokeSoftmax<T>(logits.data(), vocab_size_padded_, vocab_size_, bsz, s->kept_buf.data(), stream_);

        TopPSortParams params{};
        params.logits            = logits.data();
        params.sorted_logits     = logits.data();
        params.sorted_indices    = indices.data();
        params.kept              = s->kept_buf.data();
        params.top_ks            = s->top_k_buf.data();
        params.top_ps            = s->top_p_buf.data();
        params.batch_size        = bsz;
        params.vocab_size        = vocab_size_;
        params.vocab_size_padded = vocab_size_padded_;
        invokeTopPSort<T>(params, stream_);
    }

    // apply topp minp filter
    if (s->max_minp != 0.f || s->min_topp != 1.f) {
        TopPMinPFilterParams params{};
        params.sorted_logits     = logits.data();
        params.sorted_indices    = indices.data();
        params.kept              = s->kept_buf.data();
        params.top_ps            = s->top_p_buf.data();
        params.min_ps            = s->min_p_buf.data();
        params.batch_size        = bsz;
        params.vocab_size        = vocab_size_;
        params.vocab_size_padded = vocab_size_padded_;
        invokeTopPMinPFilter<T>(params, stream_);
    }

    // sample
    {
        SamplingParams params{};
        params.logits          = logits.data();
        params.stride          = vocab_size_padded_;
        params.indices         = indices.data();
        params.kept            = s->kept_buf.data();
        params.curandstate     = (curandState_t*)args.at("curand_state").raw_data();
        params.batch_size      = bsz;
        params.output_ids      = args.at("output_ids").data<int>() + step * bsz;
        params.sequence_length = args.at("sequence_length").data<int>();

        if (auto sampled_logprobs = args.try_("sampled_logprobs")) {
            params.sampled_logprobs = sampled_logprobs->data<T>();
            params.sampled_indexes  = args.at("sampled_indexes").data<uint32_t>();
            params.sampled_nums     = args.at("sampled_nums").data<uint32_t>();
        }

        invokeSampling<T>(params, stream_);
        sync_check_cuda_error();
    }

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
void SamplingLayer<T>::Setup(const std::shared_ptr<SamplingStates>& states, const TensorMap& args)
{
    Buffer_<const Request*> rs = args.at("requests").buffer();

    const auto bsz = rs.size();

    for (int i = 0; i < bsz; ++i) {
        top_k_[i] = rs[i]->gen_cfg.top_k;
        top_p_[i] = rs[i]->gen_cfg.top_p;
        min_p_[i] = rs[i]->gen_cfg.min_p;
    }

    auto& s = states->sampler;

    s->max_topk = *std::max_element(top_k_.begin(), top_k_.begin() + bsz);
    s->min_topk = *std::min_element(top_k_.begin(), top_k_.begin() + bsz);
    s->min_topp = *std::min_element(top_p_.begin(), top_p_.begin() + bsz);
    s->max_minp = *std::max_element(min_p_.begin(), min_p_.begin() + bsz);

    core::Copy(top_k_.data(), bsz, s->top_k_buf.data());
    core::Copy(top_p_.data(), bsz, s->top_p_buf.data());
    core::Copy(min_p_.data(), bsz, s->min_p_buf.data());
}

template class SamplingLayer<float>;

}  // namespace turbomind
