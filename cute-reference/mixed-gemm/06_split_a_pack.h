/***************************************************************************************************
 * Shared pack kernel and host function for split A loading (iteration 06).
 *
 * Pack kernel is unchanged from iteration 05. This header is a thin wrapper.
 * The consumer mainloop (06_bf16_gemm_sm90_split_a_wgmma.cu) adds threadblock swizzling.
 **************************************************************************************************/
#pragma once

#include "05_split_a_pack.h"
