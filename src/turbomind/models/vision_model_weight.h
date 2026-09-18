// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/module.h"

namespace turbomind {

/// Polymorphic root for the vision sub-graph.
///
/// Sits next to ``ModelWeight`` under ``ModelRoot`` (as the
/// ``vision_model`` child). Concrete subclasses (one per VLM family)
/// declare the per-family child layout via ``TM_MODULE_DECLARE`` and
/// own their own scalars + config.
class VisionModelWeight: public core::Module {
public:
    const char* type() const override = 0;

    VisionModelWeight()           = default;
    ~VisionModelWeight() override = default;

    /// The vision sub-graph registers no engine cache bytes: stop the
    /// registration walk so vision-side modules (e.g. Attention children)
    /// never register into the text engine's categories.
    void register_cache(::turbomind::CacheRegistry&) override {}
};

}  // namespace turbomind
