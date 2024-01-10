// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind::attention {

struct Sm80_CpAsync {};
struct Sm70_Ldg {};
struct Sm70_LdgUnrolled {};

template<int Stages>
struct Sm80_CpAsyncMultistage {};

template<class Tag, class Attention>
struct Mainloop {};

}  // namespace turbomind::attention