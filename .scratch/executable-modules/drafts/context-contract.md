# ExecContext: per-rank execution services contract

Draft spec section from [Context shared-state contract](../issues/02-context-shared-state-contract.md). Decisions are final for the spec; wording may be tightened at assembly.

## Role

`ExecContext` is the per-rank provider of shared execution resources. One instance per device/rank, owned by the engine (the site that owns today's `Context` in `turbomind.cc`). Modules request services from it; they never create their own. `core::Context` (the thread-local stream/allocator scope, `core/context.h`) is unchanged and keeps its name.

## Definition

```cpp
// src/turbomind/models/exec_context.h (new home; replaces models/llama/context.h)
class ExecContext {
public:
    // Immutable after build; safe from any thread.
    const cudaDeviceProp& props() const;
    const Communicators&  comm() const;   // TP/DP/EP/CP/node handles + device comm
    SymmAllocatorRef      symm() const;   // comm-registered memory (GetSymmAllocator today)

    // Per-kind persistent workspace pools, shared across all layer instances of a kind.
    WorkspacePools& pools();

    // Model-executor-thread only (contracts.executor-only).
    gemm::Gemm&      gemm();              // engine + dispatch cache + tuning
    gemm::Workspace& gemm_workspace();    // the ~33 MiB shared GEMM workspace

    // Factories; safe from any thread. Centralizes priority and count choices.
    StreamHandle make_stream(int priority = 0);
    EventHandle  make_event();
};
```

## Reception: injected member

Python constructs modules by name with no `ExecContext` available (binding surface frozen). The engine performs exactly one bind pass over the module tree after weights are staged and before the first `Run`:

```cpp
// core::Module grows:
virtual void bind(ExecContext& xc)
{
    xc_ = &xc;                 // default: store the sanctioned alias
    for (auto& child : children()) child.bind(xc);   // composites recurse
}
```

Each executable module reads services only through its `ExecContext* xc_` member. A forward that runs before `bind` is a contract violation and asserts. The standalone Python Linear API (`linear.py`) constructs its own `ExecContext` instead of receiving the engine's.

## What a module instance may never hold

- Raw `cudaStream_t`, `cudaEvent_t`, or communicator handles. Request via `make_stream` / `make_event` / `comm()`. Today's per-executor aux streams and events (attention `aux_stream_`/`qkv_event_`, GDN, MoE A2A `scales_copy_stream_`) migrate to factory-made handles.
- Its own `gemm::Gemm`, `DispatchCache`, or GEMM workspace. The per-rank instance in `ExecContext` is the only one on the engine path.
- Per-kind persistent workspaces. Request from `pools()` at Setup; capacity is shared across all layer instances of a kind (this is what preserves today's memory profile when modules multiply per-layer: attention `partial_O_`/`partial_ML_`, MoE routing buffers, the decoder `symm_buf_`).
  - **Amendment (Merged FFN and MoE module design)**: MoE transport/routing state — `TokenDispatcher`, routing buffers, scales-copy stream — is owned per module instance instead, by explicit decision. See `drafts/ffn-moe-design.md` for the recorded consequences (per-layer memory and JIT multiplication; migration gates must check).

Module-owned, allowed and expected: weights and their packed forms; persistent device buffers sized at Setup from pools or module registrations (pinned staging included); cache-category byte offsets per `ownership.module-cache`.

Forward-time scratch (`qkv`, `inter`, `temp`, outputs allocated by `GetArguments`) stays on the thread-local device allocator, unchanged.

## Thread placement

| Accessor | Rule |
|---|---|
| `props()`, `comm()`, `symm()`, pool capacities | immutable after build; any thread |
| `pools()` buffer contents | model executor thread (`kPrepare`/`kForward`/`kUnprep`) |
| `gemm()`, `gemm_workspace()` | model executor thread only; dispatch-cache mutation happens only there, and measure mode (`kMeasure`) only while the run is warm-up |
| `make_stream`, `make_event` | any thread |

Engine-thread ops (`kAdd`/`kSetup`/`kFetch`/`kUpdate`/`kDel` per `contracts.batchop*`) must not touch pool buffer contents or `gemm()`. Address resolution stays metadata-only per `contracts.object-address`. Cache object backing memory is touched only where `contracts.executor-only` already permits.

## Warm-up and tuning

`is_warm_up` is engine-owned state; the scheduler keeps its own copy for its suppressions (`contracts.checkpoint-publish`). Modules read warm-up as a per-run property of the executor environment: `env.warm_up()`. The shared heap `unique_ptr<int>` is removed. Warm-up GEMM tuning keeps flowing through the normal executor path (`gemm()` measure branch), preserving single-thread dispatch-cache safety without locks. `Export`/`Import` of the dispatch cache stays on the warm-up driver (`TM_GEMM_EXPORT`/`TM_GEMM_IMPORT`), unchanged.

## Migration notes

- Today's `models/llama/context.h` `Context` becomes `ExecContext`: `linear` retires into the merged Linear module, `is_warm_up` moves to env, `core_stream`/`allocator` remain as build-time services used during weight processing.
- Aux stream/event owners migrate to the factory in their module-design tickets (FFN/MoE, Attention, GDN); no behavior change is intended beyond centralized creation.
