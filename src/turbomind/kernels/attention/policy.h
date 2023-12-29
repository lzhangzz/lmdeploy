#pragma once

namespace turbomind {

struct sm80_t {};
struct sm75_t {};
struct sm70_t {};
struct simt_t {};

struct Identity {
    template<class T>
    __device__ T operator()(T offset)
    {
        return offset;
    }

    template <int D>
    __device__ int AdvanceS(int offset, int s0, int s1) {
        return offset;
    }
};

template<class FeatureLevel, class T, class Tkv, int CTA_Q, int CTA_S, int HeadDim>
struct AttentionPolicy {};

}  // namespace turbomind