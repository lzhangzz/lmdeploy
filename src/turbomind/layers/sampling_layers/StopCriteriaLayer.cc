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

#include "src/turbomind/layers/sampling_layers/StopCriteriaLayer.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/stop_criteria_kernels.h"
#include "src/turbomind/layers/sampling_layers/sampling_states.h"
#include "src/turbomind/layers/sampling_layers/utils.h"
#include "src/turbomind/macro.h"

namespace turbomind {

struct StopCriteriaStates {
    Buffer_<int> stop_words_buf;
    Tensor_<int> stop_words_ten;
};

template<typename T>
StopCriteriaLayer<T>::StopCriteriaLayer(const BaseParam&                                    param,
                                        const std::vector<std::shared_ptr<SamplingStates>>& states):
    BaseDynamicDecodeLayer{param}
{
    stop_words_ = {max_batch_size_ * 2 * kMaxStopBadWordsLen, kCPUpinned};

    for (auto& state : states) {
        auto s = std::make_shared<StopCriteriaStates>();

        s->stop_words_buf = {max_batch_size_ * 2 * kMaxStopBadWordsLen, kDEVICE};

        state->stop_criteria = std::move(s);
    }
}

template<typename T>
void StopCriteriaLayer<T>::Setup(const std::shared_ptr<SamplingStates>& states, const TensorMap& args)
{
    Buffer_<const Request*> rs = args.at("requests").buffer();

    auto& s = states->stop_criteria;

    s->stop_words_ten = {};
    init_stop_bad_words(&GenerationConfig::stop_ids,  //
                        "stop_words",
                        rs,
                        stop_words_.data(),
                        s->stop_words_buf.data(),
                        s->stop_words_ten);
}

template<typename T>
void StopCriteriaLayer<T>::Forward(const std::shared_ptr<SamplingStates>& states, TensorMap& args) const
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    const int batch_size = args.at("logits").shape(0);
    const int step       = *args.at("step").data<int>();

    auto& s = states->stop_criteria;

    if (auto& stop_words = s->stop_words_ten) {
        TM_CHECK_EQ(stop_words.ndim(), 3);  // [batch, 2, len]
        size_t stop_words_len = stop_words.shape(2);
        invokeStopWordsCriterion(args.at("output_ids").data<int>(),
                                 nullptr,
                                 stop_words.data(),
                                 args.at("finished").data<bool>(),
                                 0,
                                 stop_words_len,
                                 batch_size,
                                 1,
                                 step,
                                 stream_);
        sync_check_cuda_error();
    }

    if (auto seq_lim_len = args.try_("sequence_limit_length")) {
        invokeLengthCriterion(args.at("finished").data<bool>(),  //
                              seq_lim_len->data<int>(),
                              batch_size,
                              1,
                              step,
                              stream_);
        sync_check_cuda_error();
    }

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template class StopCriteriaLayer<float>;

}  // namespace turbomind
