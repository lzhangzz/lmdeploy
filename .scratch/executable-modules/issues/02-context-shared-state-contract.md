# Context shared-state contract

Type: grilling
Status: resolved

## Question

Define the Context shared-state contract for executable modules: the exact inventory of per-rank shared execution resources (GEMM engine plus dispatch/exec-plan cache, tuning state including `Gemm::Tune` and `Export`/`Import` plan serialization, workspace pools, `OutputSpec` allocations, streams/events), what each executable Module may request from Context, and the ownership and thread-placement rules consistent with `src/turbomind/engine/README.md` (cite `contracts.executor-only` and friends). Also decide what a module instance may never hold. Output: the contract section destined for the spec, with concrete C++ snippets of the Context accessor surface.

## Answer

Contract drafted at [`drafts/context-contract.md`](../drafts/context-contract.md). Decisions:

- **Name**: `ExecContext`, a new name for the per-rank execution services object; `core::Context` (thread-local scope) is unchanged and keeps its name. Glossary updated.
- **Inventory**: GEMM engine + per-rank dispatch cache + tuning + the ~33 MiB GEMM workspace; comm handles; device props; per-kind persistent workspace pools; stream/event factory; symm allocator. Output temporaries stay on the thread-local allocator.
- **Reception**: injected member. Python builds modules by name (binding surface frozen); the engine runs one `bind(ExecContext&)` pass over the tree after weights are staged, before the first `Run`. Forward before bind asserts. Standalone `linear.py` constructs its own `ExecContext`.
- **Never hold**: raw streams/events/comm handles (factory/accessors only), own GEMM engines/caches/workspaces, per-kind persistent workspaces (request from `pools()` at Setup; shared across layer instances, preserving today's per-kind memory profile when modules multiply per-layer). Weights, packed forms, Setup-sized buffers, and cache-category offsets stay module-owned.
- **Thread placement**: immutable-after-build accessors any thread; pool contents and `gemm()` on the model executor thread only (`contracts.executor-only`); dispatch-cache mutation only there, measure mode only during warm-up; engine-thread ops stay metadata-only per `contracts.object-address`.
- **Warm-up**: engine-owned flag exposed per-run as `env.warm_up()`; the shared heap int is removed; tuning still flows through the executor path, keeping lock-free cache safety. `Export`/`Import` unchanged on the warm-up driver.

Warm-up flow specifics per module kind (attention noise-fill, MoE random routing, decoder layer-skipping) are designed inside the respective module tickets, not here.
