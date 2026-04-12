// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/core/data_format.h"
#include "src/turbomind/core/check.h"

namespace turbomind {

bool DataFormat::is_quantized() const noexcept
{
    if (scales.present() || zeros.present()) {
        return true;
    }
    for (int bs : block_sizes) {
        if (bs > 1) {
            return true;
        }
    }
    return false;
}

DataFormat MakeLinearWeightFormat(DataType data_type, DataType weight_format, int group_size)
{
    DataFormat fmt;
    fmt.dtype = weight_format;

    if (IsTrivialFloatType(weight_format)) {
        fmt.block_sizes = {1, 1};
        return fmt;
    }

    if (weight_format == kFloat8_e4m3) {
        TM_CHECK_EQ(group_size, 128)
            << "FP8 weight format requires group_size=128, got " << group_size;
        fmt.block_sizes  = {128, 128};
        fmt.scales.dtype = kFloat;
        return fmt;
    }

    if (weight_format == kFloat4_e2m1) {
        TM_CHECK(group_size > 0)
            << "FP4 weight format requires group_size > 0, got " << group_size;
        fmt.block_sizes  = {1, group_size};
        fmt.scales.dtype = kUint8;
        return fmt;
    }

    const bool is_qweight = weight_format == kUint4 || weight_format == kUint8;
    if (is_qweight) {
        TM_CHECK(group_size > 0 && group_size <= 256)
            << "Invalid group_size for quantized weight: " << group_size;
        fmt.block_sizes  = {1, group_size};
        fmt.scales.dtype = data_type;
        fmt.zeros.dtype  = data_type;
        return fmt;
    }

    TM_CHECK(0) << "Unsupported weight format: " << to_string(weight_format);
    return fmt;
}

}  // namespace turbomind
