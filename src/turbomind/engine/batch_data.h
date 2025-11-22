
#pragma once

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/core.h"

namespace turbomind {

struct RequestCache;

struct BatchData {

    explicit BatchData(int phase): phase{phase}
    {
        ready = Event::create();
        done  = Event::create();
    }

    const int phase;

    int bs0 = 0;
    int bsz = 0;

    std::vector<int> perm;

    std::vector<int> local_token_num;
    int              global_token_num = 0;

    Event ready;
    Event done;
};

}  // namespace turbomind