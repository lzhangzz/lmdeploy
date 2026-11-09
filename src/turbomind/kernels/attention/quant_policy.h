// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind {

enum QuantPolicy
{
    kNone = 0x00,
    // reserve 0x01 and 0x02 for backward compatibility
    kReserve1 = 0x01,
    kReserve2 = 0x02,
    // quantize cache kv
    kCacheKVInt8 = 0x08,
    kCacheKVInt4 = 0x04,
};

}  // namespace turbomind
