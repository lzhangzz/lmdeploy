#pragma once

namespace turbomind {

struct sm80_t {};
struct sm75_t {};
struct sm70_t {};
struct simt_t {};

struct Identity {
    template<class X>
    __device__ X operator()(X x)
    {
        return x;
    }
};

template<class FeatureLevel, class T, class Tkv, int CTA_Q, int CTA_S, int HeadDim>
struct AttentionPolicy {};

}  // namespace turbomind