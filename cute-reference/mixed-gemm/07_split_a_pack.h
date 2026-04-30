/***************************************************************************************************
 * Shared pack kernel and host function for split A loading (iteration 07).
 *
 * Pack kernel is unchanged from iteration 06. This header is a thin wrapper.
 * The consumer mainloop (07_bf16_gemm_sm90_split_a_wgmma.cu) defers TMA store
 * wait for epilogue-compute overlap.
 **************************************************************************************************/
#pragma once

#include "06_split_a_pack.h"
