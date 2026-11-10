// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <pthread.h>

namespace turbomind::comm {

class Barrier {
public:
    explicit Barrier(int count): barrier_{}
    {
        pthread_barrier_init(&barrier_, {}, count);
    }

    ~Barrier()
    {
        pthread_barrier_destroy(&barrier_);
    }

    void arrive_and_wait()
    {
        pthread_barrier_wait(&barrier_);
    }

private:
    pthread_barrier_t barrier_;
};

}  // namespace turbomind::comm
