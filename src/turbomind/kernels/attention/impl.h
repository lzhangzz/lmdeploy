// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind {

class TensorOp {};
class Simt {};

namespace attention {

struct Sm80_16816 {};
struct Sm75_1688 {};
struct Sm70_884 {};
struct Sm70_Simt {};

template<class Tag, class T, class Tkv, int CTA_Q, int CTA_S, int WARP_Q, int WARP_S, int HeadDim>
struct Impl {};

}  // namespace attention

// template<class Tag, class T, class Tkv, int CTA_Q, int CTA_S, int HeadDim>
// struct Attention {};

// template<class T, int CTA_Q, int CTA_S, int HeadDim, class BlockSeqLen, int Stages>
// struct AttentionImpl<Sm80<Stages>, T, T, CTA_Q, CTA_S, HeadDim, BlockSeqLen> {
//     // cp.async, 2-stage
//     // m16n8k16
// };

// template<class T, class Tkv, int CTA_Q, int CTA_S, int HeadDim, class BlockSeqLen, int Stages>
// struct AttentionImpl<Sm80Decoding<TensorOp, Stages>, T, Tkv, CTA_Q, CTA_S, HeadDim, BlockSeqLen> {
//     // cp.async, multi-stage, unrolled
//     // m16n8k16-trans
// };

// template<class T, class Tkv, int CTA_Q, int CTA_S, int HeadDim, class BlockSeqLen, int Stages>
// struct AttentionImpl<Sm80Decoding<Simt, Stages>, T, Tkv, CTA_Q, CTA_S, HeadDim, BlockSeqLen> {
//     // cp.async, multi-stage, unrolled
//     // fma
// };

// template<class T, int CTA_Q, int CTA_S, int HeadDim, class BlockSeqLen>
// struct AttentionImpl<Sm75, T, T, CTA_Q, CTA_S, HeadDim, BlockSeqLen> {
//     // ldg, 2 stage
//     // m16n8k8
// };

// template<class T, class Tkv, int CTA_Q, int CTA_S, int HeadDim, class BlockSeqLen>
// struct AttentionImpl<Sm75Decoding<TensorOp>, T, Tkv, CTA_Q, CTA_S, HeadDim, BlockSeqLen> {
//     // ldg, 2 stage, unrolled
//     // m16n8k8-trans
// };

// template<class T, class Tkv, int CTA_Q, int CTA_S, int HeadDim, class BlockSeqLen>
// struct AttentionImpl<Sm75Decoding<Simt>, T, Tkv, CTA_Q, CTA_S, HeadDim, BlockSeqLen> {
//     // ldg, 2 stage, unrolled
//     // fma
// };

// template<class T, int CTA_Q, int CTA_S, int HeadDim, class BlockSeqLen>
// struct AttentionImpl<Sm70, T, T, CTA_Q, CTA_S, HeadDim, BlockSeqLen> {
//     // ldg, 2 stage
//     // m8n8k4
// };

// template<class T, class Tkv, int CTA_Q, int CTA_S, int HeadDim, class BlockSeqLen>
// struct AttentionImpl<Sm70Decoding<Simt>, T, Tkv, CTA_Q, CTA_S, HeadDim, BlockSeqLen> {
//     // ldg, 2 stage, unrolled
//     // fma
// };

}  // namespace turbomind