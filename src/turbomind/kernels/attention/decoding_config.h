// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "arch.h"
#include "cta_map.h"
#include "decoding_simt.h"
#include "mainloop_sm80_multistage.h"
#include "src/turbomind/kernels/attention/attention_universal.h"

namespace turbomind::attention {

template<class T, class Tkv, class BlockSeqLen, int HeadDim>
struct DecodingConfig {
    static constexpr int CTA_Q  = 1;
    static constexpr int CTA_S  = 64;
    static constexpr int WARP_Q = 1;
    static constexpr int WARP_S = 16;
    //
    using Attention = Impl<Sm70_Simt, T, Tkv, CTA_Q, CTA_S, WARP_Q, WARP_S, HeadDim>;
    using Mainloop  = Mainloop<Sm80_CpAsyncMultistage<3>, Attention>;
    using Kernel    = AttentionUniversal<Mainloop, BlockSeqLen, DecodingCtaMap>;
};

}  // namespace turbomind::attention