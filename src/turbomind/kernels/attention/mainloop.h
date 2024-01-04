// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind::attention {

struct Sm80_CpAsync {};
struct Sm80_CpAsyncUnrolled {};
struct Sm70_Ldg {};
struct Sm70_LdgUnrolled {};

template<class Tag, class Attention>
struct Mainloop {};

}  // namespace turbomind::attention