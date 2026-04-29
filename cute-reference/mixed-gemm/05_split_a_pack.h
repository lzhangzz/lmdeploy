/***************************************************************************************************
 * Shared pack kernel and host function for split A loading (iteration 05).
 *
 * Pack kernel is unchanged from iteration 04. This header is a thin wrapper.
 * The consumer mainloop (05_bf16_gemm_sm90_split_a_wgmma.cu) adds k_block-level
 * interleaving, delayed stage release, and consumer_try_wait prefetch.
 **************************************************************************************************/
#pragma once

#include "04_split_a_pack.h"
