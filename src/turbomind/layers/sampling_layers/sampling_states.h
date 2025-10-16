#pragma once

#include <memory>

#include "src/turbomind/core/core.h"

namespace turbomind {

struct LogitsProcessorStates;
struct SamplerStates;
struct StopCriteriaStates;

struct SamplingStates {
    std::shared_ptr<LogitsProcessorStates> logits_processor;
    std::shared_ptr<SamplerStates>         sampler;
    std::shared_ptr<StopCriteriaStates>    stop_criteria;
};

}  // namespace turbomind