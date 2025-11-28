
#pragma once

#include <memory>

#include "src/turbomind/engine/gateway.h"

#include "src/turbomind/models/language_model.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

// sched batch
// ----
// new    request  +
// cancel request  -
// finish request  -
// ----
// 0 1 2 3
// _ 1 2 _ 4 5
// 1 2 _ _

// exec batch
// ----
// 0 1 2 3
// x x 2 3

// fixed-sized data (a --> exec)
// ---
// 1. scatter-copy-back (batch-field is a superset of state-field)
// scatter(state_field, idxs, batch_field)
// copy   (batch_field,       state_field)

// variable-sized data (sched --> exec)
// scatter_offseted(state_field, state_offsets, idxs, batch_field, batch_offsets, batch_end_offsets)
//    copy_offseted(batch_field, batch_offsets, batch_end_offsets, state_field, state_offsets)

// fixed-sized data (sched <-- exec)
// ---
// scatter(state_field, idxs, batch_field)
// copy   (batch_field,       state_field)

// baseline
// state = state.append(gather(incoming))

// Sync
// ---
// batch = Receive()
// batch = Synchronize(batch)
// state = batch
// state = Accept(state, requests)
// batch = Schedule(state)
// Send(batch)

// Async
// ---
// state = {}
// batch = Receive()
// batch = Synchronize(batch)
// state = Update(state, batch)     # update request states (e.g. seq_len, is_finished)
// state = Accept(state, requests)
// batch = Schedule(state)
// state = Clone(batch)             # no mutable sharing
// Send(batch)

/// Decoupled asynchronous model execution

class Engine {
public:
    ~Engine();

    Engine();
    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;

    explicit operator bool() const noexcept
    {
        return static_cast<bool>(impl_);
    }

    Engine(DataType      dtype,
           EngineParam   param,
           LanguageModel model,
           Context&      ctx,
           Gateway&      gateway,
           int           device_id,
           int           dp_rank,
           int           phases);

    void WarmUp();

    void Start();

    ScheduleMetrics GetScheduleMetrics();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind