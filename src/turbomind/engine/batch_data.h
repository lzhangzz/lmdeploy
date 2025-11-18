
#pragma once

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/core.h"

namespace turbomind {

struct RequestCache;

struct BatchData {
    int phase;

    int bs0;
    int bsz;

    std::vector<int> perm;

    std::vector<int> local_token_num;
    int              global_token_num;

    Event ready;
    Event done;

    BatchData()
    {
        ready = Event::create();
        done  = Event::create();
    }
};

}  // namespace turbomind