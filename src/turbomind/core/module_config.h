// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

// Transitional header — config structs pending migration to their _weight.h files.
// Infrastructure (TM_MEMBER, TM_PTR, TM_FOR_EACH, ModuleConfig, ModuleListConfig) lives in module.h.

#include "src/turbomind/core/module.h"

namespace turbomind::core {

struct DecoderLayerConfig: ModuleConfig {
    DecoderLayerConfig(): ModuleConfig{"DecoderLayerWeight"} {}
    template<typename Visitor>
    static void for_each(Visitor&&) {}
};

}  // namespace turbomind::core
