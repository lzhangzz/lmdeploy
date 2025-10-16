#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/state.h"

namespace turbomind {

void test_const_state()
{
    const int max_bsz = 1024;

    struct Batch {
        ConstState_<int> prompt_length;
        int              size = 0;
        explicit Batch(int max_bsz): prompt_length{{max_bsz}} {}
    };

    auto b = std::make_shared<Batch>(max_bsz);

    Buffer_<int> perm{max_bsz, kCPU};

    while (true) {
        // modification on host
        b->prompt_length[b->size++] = 100;
        b->prompt_length[b->size++] = 10;

        // reorder
        b->prompt_length.Reorder(perm.slice(0, b->size));

        // release to executor
        b->prompt_length.Release();
    }
}

void test_mutable_state()
{
    const int max_bsz = 1024;

    struct Batch {
        MutableState_<int> context_length;

        Buffer_<int> perm;
        Buffer_<int> mask;

        int size = 0;
        explicit Batch(int max_bsz): context_length{{max_bsz}} {}
    };

    Buffer_<int> perm{max_bsz, kCPUpinned};
    Buffer_<int> mask{max_bsz, kCPUpinned};

    auto b = std::make_shared<Batch>(max_bsz);

    // Scheduling thread / stream
    while (true) {
        ///////////////////////////////////////////////////
        /// Step T+X

        // Add
        b->context_length[b->size++] = 100;
        b->context_length[b->size++] = 10;

        // Modify
        b->context_length[0] += 4;

        Copy_(perm, b->size, b->perm);
        Copy_(mask, b->size, b->mask);

        // reorder
        b->context_length.Reorder_0(perm.slice(0, b->size));

        // release (H2D)
        b->context_length.Release_0(b->size);

        // receive batch from executor
        // b = receive();

        ///////////////////////////////////////////////////
        /// Step T

        b->context_length.Acquire_0(b->size);

        core::Context::stream().Sync();

        b->context_length.Patch_0(perm, mask);
    }

    // Execution thread / stream
    while (true) {
        // b = receive()
        b->context_length.Acquire_1(b->size);
        b->context_length.Patch_1(b->perm, b->mask);

        //

        b->context_length.Release_1(b->size);
    }
}

void test_mutable_state_2()
{
    struct Batch {
        Tensor_<int> context_length;

        Tensor_<int> input_ids;
        Buffer_<int> input_ids_offsets;

        Buffer_<int> perm;
        Buffer_<int> mask;
    };

    if (1) {  // sched
        int bsz = 0;

        ConstVarlenState0_<float> input_embeds;

        MutableState0_<int> context_length;

        std::shared_ptr<Batch> b = std::make_shared<Batch>();

        while (true) {
            // Add
            context_length->data()[bsz++] = 10;

            // Schedule
            Buffer_<int> perm;
            Buffer_<int> mask;

            // Reorder
            context_length.Reorder(perm);

            // Release
            context_length.Release(b->context_length, bsz);

            // Send(B)
            // b = Receive()

            // Acquire
            context_length.Acquire(b->context_length, bsz);

            // Patch
            context_length.Patch(perm, mask);
        }
    }

    if (1) {  // exec
        MutableState1_<int> context_length;

        std::shared_ptr<Batch> b;

        while (true) {
            // b = Receive()

            context_length.Patch(b->context_length, b->perm, b->mask);

            // forward

            context_length.Release(b->context_length);
        }
    }
}

}  // namespace turbomind