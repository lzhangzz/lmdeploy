// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "attention_params.h"

namespace turbomind {

template<typename T>
void dispatchAttention(const AttentionParams<T>& params);

template<class T>
void invokeProcessKV(const AttentionParams<T>& params);

}
